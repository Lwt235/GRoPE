"""Read-only, same-device whole-model cost measurements on old validation data."""
import argparse
import hashlib
import math
import os
from pathlib import Path
import statistics
import subprocess
import time

import numpy as np
import torch
import torch.nn.functional as F

from experiments.round65_common import ROOT, read, write, sha
from experiments import round65_run as native_engine
from experiments import round68_fixed_region_run as region_engine
from experiments.round73_native_relative5 import check as native_check, local_dependencies
from experiments.region_shape_attribution import check as region_check
from models.round65_actual_gs import NativeGSClassifier
from models.round73_native_relative5 import CompactNativeClassifier
from models.round68_fixed_region import FixedRegionMatcher
from models.round69_compact_relation import CompactMatcher

OUT = ROOT / 'results/pe_runtime_diagnostic/attempt1'
ROLES = ('distance_mlp', 'direct_mlp', 'gaussian_relation', 'relative5')
ROOTS = {
    'native_old': ROOT / 'results/round65_actual_gs/attempt1',
    'native_new': ROOT / 'results/round73_native_relative5/attempt1',
    'region_old': ROOT / 'results/round68_fixed_region/attempt1',
    'region_new': ROOT / 'results/round69_compact_relation/attempt1',
}


def freeze():
    native_check()
    region_check()
    assert not (OUT / 'plan.json').exists()
    sources = {str(p): sha(p) for p in local_dependencies([Path(__file__)])}
    checkpoints = {}
    for task in ('region', 'native'):
        for role in ROLES:
            root = ROOTS[task + ('_new' if role == 'relative5' else '_old')]
            selected = read(root / 'selected.json')
            choice = next(r for r in selected['records'] if (r['role'], r['seed']) == (role, 11))
            folder = root / 'runs' / role / '11'
            assert sha(folder / 'best.pt') == choice['checkpoint_sha256']
            checkpoints[f'{task}/{role}'] = dict(path=str(folder / 'best.pt'),
                sha256=choice['checkpoint_sha256'], validation=str(folder / 'validation.npz'))
            for path in (root / 'lock.json', root / 'selected.json', folder / 'validation.npz'):
                sources[str(path)] = sha(path)
    for key in ('native_old', 'region_old'):
        path = ROOTS[key] / 'data/manifest.json'
        sources[str(path)] = sha(path)
    write(OUT / 'plan.json', dict(created_unix=time.time(), sources=sources, checkpoints=checkpoints,
        hypothesis='The five-input representation has practical whole-model cost comparable to matched MLP controls at current task sizes.',
        competing_explanation='Pairwise geometry or hidden features impose material overhead despite the compact descriptor.',
        comparison=dict(tasks=['region', 'native'], roles=ROLES, checkpoint_seed=11,
            input='First 16 old validation examples; 64 regional tokens per view or 512 native primitives.',
            modes=['inference', 'forward_backward'], warmup_calls=5, blocks=5, calls_per_block=5,
            order='Cyclic role order by block; all roles on the same GPU.',
            precision='Original FP32 network / FP64 geometry, TF32 disabled, deterministic algorithms.',
            loss='Original task loss; backward only, no optimizer updates or geometry-input gradients.'),
        decision='If measured overhead is material relative to within-run variation, prioritize equivalent contraction profiling before scaling; otherwise preserve the present implementation. No task-effectiveness decision follows.',
        scope='Whole model forward or forward+loss+backward. Excludes loading, preprocessing, optimizer state/step and persistent full training-cache VRAM. One old batch and one GPU are not a general performance benchmark.',
        resource='One whole-idle GPU, at least 34 GiB free, CUDA probe and occupancy recheck. Preserve the three native fits.',
        fresh_data_access=False, model_or_training_changed=False, goal_complete=False))
    (OUT / 'plan.sha256').write_text(sha(OUT / 'plan.json') + '\n')
    print('COST_PLAN_FROZEN', sha(OUT / 'plan.json'), flush=True)


def checked():
    plan = read(OUT / 'plan.json')
    assert sha(OUT / 'plan.json') == (OUT / 'plan.sha256').read_text().strip()
    for path, digest in plan['sources'].items():
        assert sha(path) == digest, path
    for row in plan['checkpoints'].values():
        assert sha(row['path']) == row['sha256']
    return plan


def gpu_snapshot():
    return subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True, timeout=10)


def input_batch(task):
    root = ROOTS[task + '_old']
    manifest = read(root / 'data/manifest.json')
    if task == 'region':
        row = manifest['caches']['val']
        assert sha(row['path']) == row['sha256']
        with np.load(row['path'], allow_pickle=False) as z:
            cache = {k: z[k][:16] for k in z.files}
        values = region_engine.inputs(cache, slice(None), device='cuda')
        target = torch.as_tensor(cache['overlap'], device='cuda')
        return values, target, dict(ids=cache['ids'].tolist(), cache_sha256=row['sha256'])
    row = next(r for r in manifest['partitions'] if r['split'] == 'val')
    bundle = {}
    for name in ('centers', 'covariance', 'appearance', 'labels'):
        path = root / 'data/val' / (name + '.npy')
        assert sha(path) == row['hashes'][name + '.npy']
        bundle[name] = np.load(path, mmap_mode='r', allow_pickle=False)
    values = native_engine.evaluation_batch(bundle, 0, 16, 'base', 'cuda')
    target = torch.tensor(np.array(bundle['labels'][:16]), device='cuda')
    return values, target, dict(ids=row['ids'][:16], manifest_sha256=sha(root / 'data/manifest.json'))


def state_digest(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def run():
    plan = checked()
    assert not (OUT / 'result.json').exists() and not (OUT / 'execution.json').exists()
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    before = gpu_snapshot()
    row = next(line.split(',') for line in before.splitlines() if line.split(',')[1].strip() == visible)
    assert int(row[2]) <= 512 and int(row[3]) >= 34 * 1024 and int(row[4]) <= 10
    native_engine.configure()
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    probe = torch.ones(4, device='cuda')
    assert float(probe.sum()) == 4
    torch.cuda.synchronize()
    del probe
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
        '--format=csv,noheader,nounits'], text=True, timeout=10)
    assert all(int(line.split(',')[1]) == os.getpid() for line in apps.splitlines()
               if line.split(',')[0].strip() == visible)
    identity = dict(pid=os.getpid(), gpu_uuid=visible, gpu_name=torch.cuda.get_device_name(),
        torch_version=torch.__version__, started_unix=time.time(), gpu_before=before,
        plan_sha256=sha(OUT / 'plan.json'), optimizer_updates=0)
    write(OUT / 'execution.json', identity)
    results, replays, inputs = [], [], {}
    try:
        for task in ('region', 'native'):
            print('COST_INPUT', task, flush=True)
            data, target, inputs[task] = input_batch(task)
            models, initial = {}, {}
            for role in ROLES:
                if task == 'native':
                    model = CompactNativeClassifier(role) if role == 'relative5' else NativeGSClassifier(role)
                else:
                    model = CompactMatcher(role) if role == 'relative5' else FixedRegionMatcher(role)
                checkpoint = plan['checkpoints'][f'{task}/{role}']
                model.load_state_dict(torch.load(checkpoint['path'], map_location='cpu', weights_only=True)['state_dict'])
                model = model.cuda().eval()
                initial[role] = state_digest(model)
                # Replay the original evaluation batch size before cost measurements.
                count = 4 if task == 'native' else 16
                with torch.no_grad():
                    output = model(*(x[:count] for x in data)).cpu()
                with np.load(checkpoint['validation'], allow_pickle=False) as z:
                    expected = torch.from_numpy(z['logits'][:count])
                    error = float((output - expected).abs().max())
                    changes = int((output.argmax(-1) != expected.argmax(-1)).sum())
                assert error < 2e-4 and changes == 0, (task, role, error, changes)
                replays.append(dict(task=task, role=role, maximum_logit_error=error, prediction_changes=changes))
                models[role] = model
                print('COST_REPLAY', task, role, error, flush=True)
            for mode in ('inference', 'forward_backward'):
                def step(role):
                    model = models[role]
                    model.zero_grad(set_to_none=True)
                    if mode == 'inference':
                        with torch.no_grad():
                            model(*data)
                    else:
                        output = model(*data)
                        loss = F.cross_entropy(output, target, label_smoothing=.1) if task == 'native' else region_engine.loss_function(output, target)
                        loss.backward()

                for model in models.values():
                    model.train(mode == 'forward_backward')
                    model.zero_grad(set_to_none=True)
                for role in ROLES:
                    for _ in range(5):
                        step(role)
                    if mode == 'forward_backward':
                        assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in models[role].parameters())
                for block in range(5):
                    order = ROLES[block % len(ROLES):] + ROLES[:block % len(ROLES)]
                    for role in order:
                        for model in models.values():
                            model.zero_grad(set_to_none=True)
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                        baseline = torch.cuda.memory_allocated()
                        torch.cuda.reset_peak_memory_stats()
                        torch.manual_seed(71001 + block)
                        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        wall = time.perf_counter()
                        start.record()
                        for _ in range(5):
                            step(role)
                        end.record()
                        torch.cuda.synchronize()
                        results.append(dict(task=task, role=role, mode=mode, block=block,
                            cuda_ms_per_call=start.elapsed_time(end) / 5,
                            wall_ms_per_call=(time.perf_counter() - wall) * 1000 / 5,
                            baseline_allocated_bytes=baseline, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                            incremental_peak_bytes=torch.cuda.max_memory_allocated() - baseline,
                            parameters=sum(p.numel() for p in models[role].parameters()),
                            pe_parameters=sum(p.numel() for p in models[role].position.parameters())))
                        write(OUT / 'progress.json', dict(last=results[-1], completed_blocks=len(results), observed_unix=time.time()))
                        print('COST_BLOCK', task, mode, role, block, round(results[-1]['cuda_ms_per_call'], 3), flush=True)
            for role, model in models.items():
                assert state_digest(model) == initial[role], (task, role, 'model changed')
            del model, models, data, target, output, expected
            torch.cuda.empty_cache()
        summary = []
        for task in ('region', 'native'):
            for mode in ('inference', 'forward_backward'):
                for role in ROLES:
                    group = [r for r in results if (r['task'], r['mode'], r['role']) == (task, mode, role)]
                    times = [r['cuda_ms_per_call'] for r in group]
                    summary.append(dict(task=task, mode=mode, role=role, median_cuda_ms=statistics.median(times),
                        min_cuda_ms=min(times), max_cuda_ms=max(times),
                        peak_gib=max(r['peak_allocated_bytes'] for r in group) / 2**30,
                        incremental_peak_gib=max(r['incremental_peak_bytes'] for r in group) / 2**30,
                        parameters=group[0]['parameters'], pe_parameters=group[0]['pe_parameters']))
        write(OUT / 'result.json', dict(execution=identity, completed_unix=time.time(), summary=summary,
            blocks=results, replays=replays, input_identity=inputs, all_model_states_unchanged=True,
            gpu_after=gpu_snapshot(), scope=plan['scope'], scientific_effectiveness_claim=False,
            review_pending=True, goal_complete=False))
        print('COST_COMPLETE', len(results), 'blocks', flush=True)
    except BaseException as error:
        write(OUT / 'failure.json', dict(error_type=type(error).__name__, message=str(error), completed_blocks=len(results)))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('freeze', 'run'))
    args = parser.parse_args()
    globals()[args.action]()
