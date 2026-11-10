"""Old-data CUDA check for CPU batch gathering, without changing any model."""
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from experiments.round65_common import ROOT, CONFIG, read, write, sha
from experiments.native_source_diversity_data import OUT, OLD_DATA, check
from experiments.native_source_diversity_sampling import gather


def main():
    check()
    destination = OUT / 'cuda_preflight.json'
    assert not destination.exists()
    cards = [[v.strip() for v in line.split(',')] for line in subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu',
         '--format=csv,noheader,nounits'], text=True).strip().splitlines()]
    available = [r for r in cards if int(r[3]) >= 16384]
    if not available:
        print('NO_GPU_WITH_16_GIB_FREE', flush=True)
        return 75
    idle = [r for r in available if int(r[2]) < 512 and int(r[4]) < 10]
    selected = max(idle or available, key=lambda r: int(r[3]))
    os.environ['CUDA_VISIBLE_DEVICES'] = selected[0]
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import torch
    import torch.nn.functional as functional
    from experiments import round65_run as runner
    from models.round65_actual_gs import NativeGSClassifier
    from models.round73_native_relative5 import CompactNativeClassifier
    from models.native_shape_attribution import NativeFixedIsotropic

    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    cap = 14 * 2**30
    torch.cuda.set_per_process_memory_fraction(cap / torch.cuda.get_device_properties(0).total_memory)
    runner.configure()
    state = dict(pid=os.getpid(), started_unix=time.time(), state='running', gpu=selected,
        shared_at_launch=bool(not idle), effective_allocator_cap_bytes=cap, records=[],
        source_sha256=sha(__file__), preparation_lock_sha256=sha(OUT / 'preparation_lock.json'),
        new_data_or_scores_used=False, full_training_started=False)
    status_path = OUT / 'preflight_runtime.json'
    write(status_path, state)
    manifest = read(OLD_DATA / 'manifest.json')
    n = next(r['count'] for r in manifest['partitions'] if r['split'] == 'fit')
    indices = np.linspace(0, n - 1, 32, dtype=np.int64)
    bank = {key: np.array(np.load(OLD_DATA / 'fit' / (key + '.npy'), mmap_mode='r')[indices])
            for key in ('centers', 'covariance', 'appearance', 'labels')}
    rng = np.random.default_rng(92011)
    object_index = rng.choice(len(indices), 16, replace=False)
    subset = np.stack([rng.choice(2048, 512, replace=False) for _ in object_index])
    quaternion = rng.normal(size=(16, 4))
    jitter = rng.normal(0, CONFIG['jitter_std'], size=(16, 512, 3)).clip(
        -CONFIG['jitter_clip'], CONFIG['jitter_clip']).astype(np.float32)
    mapping = np.column_stack([np.zeros(16, dtype=np.int64), object_index])
    cpu = gather(bank, bank, mapping, subset)
    models = [('relative5', lambda: CompactNativeClassifier('relative5')),
              ('distance_mlp', lambda: NativeGSClassifier('distance_mlp')),
              ('fixed_isotropic5', lambda: NativeFixedIsotropic())]
    common = []
    try:
        for role, factory in models:
            torch.manual_seed(11)
            model = factory().cuda().train()
            common.append(runner.shared_digest(model))
            torch.cuda.reset_peak_memory_stats()
            original = {key: torch.tensor(bank[key], device='cuda')
                        for key in ('centers', 'covariance', 'appearance', 'labels')}
            ii = torch.as_tensor(object_index, device='cuda')
            jj = torch.as_tensor(subset, device='cuda')
            reference_inputs = tuple(original[key][ii[:, None], jj]
                                     for key in ('centers', 'covariance', 'appearance'))
            transferred = tuple(torch.tensor(cpu[key], device='cuda')
                                for key in ('centers', 'covariance', 'appearance'))
            assert all(torch.equal(a, b) for a, b in zip(reference_inputs, transferred))
            assert torch.equal(original['labels'][ii], torch.tensor(cpu['labels'], device='cuda'))
            results = []
            for inputs in (reference_inputs, transferred):
                c, s = runner.augment(inputs[0], inputs[1], quaternion, jitter)
                torch.manual_seed(110000000)
                model.zero_grad(set_to_none=True)
                output = model(c, s, inputs[2])
                loss = functional.cross_entropy(output, original['labels'][ii],
                                                label_smoothing=CONFIG['label_smoothing'])
                loss.backward()
                assert torch.isfinite(loss)
                grads = {name: p.grad.detach().cpu().clone() for name, p in model.named_parameters() if p.grad is not None}
                results.append((output.detach().cpu().clone(), grads, float(loss)))
                del c, s, output, loss
            logit_error = float((results[0][0] - results[1][0]).abs().max())
            gradient_error = max(float((results[0][1][name] - grad).abs().max())
                                 for name, grad in results[1][1].items())
            assert torch.allclose(results[0][0], results[1][0], rtol=1e-5, atol=1e-6)
            assert all(torch.allclose(results[0][1][name], grad, rtol=1e-4, atol=2e-6)
                       for name, grad in results[1][1].items())
            optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'],
                                          weight_decay=CONFIG['weight_decay'])
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['clip_norm'], error_if_nonfinite=True)
            optimizer.step()
            assert common[-1] != runner.shared_digest(model)
            torch.cuda.synchronize()
            record = dict(role=role, cpu_and_gpu_gather_equal=True, maximum_logit_difference=logit_error,
                maximum_gradient_difference=gradient_error, loss=results[1][2], gradient_norm=float(norm),
                peak_allocated=torch.cuda.max_memory_allocated(), parameters=sum(p.numel() for p in model.parameters()),
                position_parameters=sum(p.numel() for p in model.position.parameters()), initial_common_sha256=common[-1])
            state['records'].append(record)
            write(status_path, state)
            print('OLD_DATA_PREFLIGHT', role, logit_error, gradient_error, record['peak_allocated'], flush=True)
            del model, optimizer, original, reference_inputs, transferred, results
            torch.cuda.empty_cache()
        assert len(set(common)) == 1
        state.update(state='complete', finished_unix=time.time())
        write(status_path, state)
        write(destination, dict(state, passed=True, common_initialization_matched=True,
            sampler_source_sha256=sha(ROOT / 'experiments/native_source_diversity_sampling.py'),
            test_source_sha256=sha(ROOT / 'tests/test_native_source_diversity_sampling.py')))
        print('OLD_DATA_PREFLIGHT_COMPLETE', flush=True)
        return 0
    except Exception as error:
        state.update(state='failed', finished_unix=time.time(), error_type=type(error).__name__, reason=str(error))
        write(status_path, state)
        raise


if __name__ == '__main__':
    raise SystemExit(main())
