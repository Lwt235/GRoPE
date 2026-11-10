"""Bounded, single-thread correctness checks and exploratory CPU timings."""
import copy
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

import numpy as np  # Host compatibility: NumPy before Torch.
import torch
import torch.nn.functional as F

from experiments.efficiency_sidecar.adapters import adapt, quadratic_descriptors, radial_descriptors
from experiments.round65_run import synthetic_inputs
from experiments.round65_common import ROOT, sha, read, write
from models.round65_actual_gs import NativeGSClassifier
from models.round73_native_relative5 import CompactNativeClassifier
from models.native_shape_attribution import NativeFixedIsotropic
from models.round69_compact_relation import descriptors
from models.region_shape_attribution import fixed_inputs

OUT = ROOT / 'results/efficiency_sidecar/cpu1'
ROLES = ('relative5', 'distance_mlp', 'fixed_isotropic5')
VARIANTS = ('original', 'factored', 'geometry_factored', 'geometry_factored_recompute')


def model_for(role):
    if role == 'relative5':
        return CompactNativeClassifier(role)
    if role == 'fixed_isotropic5':
        return NativeFixedIsotropic(role)
    return NativeGSClassifier(role)


def variant(model, name):
    model = copy.deepcopy(model)
    if name == 'original':
        return model
    return adapt(model, quadratic='geometry' in name, radial_shortcut='geometry' in name,
                 recompute='recompute' in name)


def state_hash(model):
    h = hashlib.sha256()
    for name, v in model.state_dict().items():
        h.update(name.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def tensor_hash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def old_inputs(batch=16, tokens=1024):
    root = ROOT / 'results/round65_actual_gs/attempt1/data'
    manifest = read(root / 'manifest.json')
    row = next(r for r in manifest['partitions'] if r['split'] == 'val')
    arrays = {name: np.load(root / 'val' / (name + '.npy'), mmap_mode='r')
              for name in ('centers', 'covariance', 'appearance', 'labels')}
    data = tuple(torch.from_numpy(np.array(arrays[k][:batch, :tokens]))
                 for k in ('centers', 'covariance', 'appearance'))
    labels = torch.from_numpy(np.array(arrays['labels'][:batch])).long()
    return data, labels, dict(manifest_sha256=sha(root / 'manifest.json'), ids=row['ids'][:batch],
        subset_sha256={k: tensor_hash(v) for k, v in zip(('centers', 'covariance', 'appearance'), data)},
        source_array_expected_sha256=row['hashes'], labels_sha256=tensor_hash(labels),
        full_array_hash_recomputed=False, note='Read-only old validation prefix, not fresh task confirmation.')


def gradient_record(base, name, data, target):
    model = variant(base, name).train()
    state_before = state_hash(model)
    torch.manual_seed(921110)
    output = model(*data)
    loss = F.cross_entropy(output, target, label_smoothing=.1)
    loss.backward()
    gradients = {k: p.grad.detach().clone() for k, p in model.named_parameters() if p.grad is not None}
    assert all(torch.isfinite(g).all() for g in gradients.values())
    assert state_before == state_hash(model)
    # A private one-step optimizer probe; no checkpoint or scientific fit is changed.
    opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    opt.step()
    return output.detach(), float(loss.detach()), gradients, model.state_dict()


def parameter_checks():
    records = []
    data = synthetic_inputs(2, 17, 'cpu')
    target = torch.tensor([1, 13])
    for dtype in (torch.float64, torch.float32):
        for role in ROLES:
            torch.manual_seed(921109)
            base = model_for(role).to(dtype=dtype)
            actual = tuple(v.to(dtype) if i != 1 else v.double() for i, v in enumerate(data))
            reference = gradient_record(base, 'original', actual, target)
            for name in VARIANTS[1:]:
                trial = gradient_record(base, name, actual, target)
                assert reference[2].keys() == trial[2].keys()
                errors = [(reference[2][k].double() - trial[2][k].double()).square().sum()
                          for k in reference[2]]
                norm = sum(g.double().square().sum() for g in reference[2].values())
                grad_error = float((sum(errors) / norm).sqrt())
                logit_error = float((reference[0] - trial[0]).abs().max())
                update_error = max(float((reference[3][k] - trial[3][k]).abs().max())
                                   for k in reference[3])
                assert logit_error < 2e-4 and grad_error < 1e-4
                records.append(dict(dtype=str(dtype), role=role, variant=name, logit_max_abs=logit_error,
                    loss_abs=abs(reference[1] - trial[1]), gradient_relative_l2=grad_error,
                    private_adamw_state_max_abs=update_error))
    return records


def saved_storage():
    records = []
    data = synthetic_inputs(1, 128, 'cpu')
    for role in ROLES:
        torch.manual_seed(921109)
        base = model_for(role).train()
        for name in VARIANTS:
            model = variant(base, name)
            storage = {}
            def pack(tensor):
                s = tensor.untyped_storage()
                storage[s.data_ptr()] = s.nbytes()
                return tensor
            with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
                output = model(*data)
            records.append(dict(role=role, variant=name, unique_saved_storage_bytes=sum(storage.values()),
                scope='Unique forward tensors saved for backward; not allocator peak; B1 N128 FP32.'))
            del output, model
    return records


def sensitivity():
    rows = []
    for condition in (1e2, 1e6, 1e10):
        for offset in (0., 1e9):
            torch.manual_seed(921111)
            c = torch.randn(1, 24, 3, dtype=torch.float64)
            c[:, 1] = c[:, 0] + 1e-7
            c[:, 2] = c[:, 0]
            c += offset
            q, _ = torch.linalg.qr(torch.randn(1, 24, 3, 3, dtype=torch.float64))
            eig = torch.tensor([1., condition**-.5, condition**-1], dtype=torch.float64)
            s = (q * eig) @ q.transpose(-1, -2)
            a, b = descriptors('relative5', c, s), quadratic_descriptors(c, s)
            rows.append(dict(condition=condition, offset=offset, max_abs=float((a-b).abs().max()),
                near_pair_max_abs=float((a[:, 0, 1] - b[:, 0, 1]).abs().max()),
                finite=bool(torch.isfinite(b).all()),
                duplicate_metric_exact_zero=bool((b[:, 0, 2, 1:3] == 0).all())))
    return rows


def timing(functions, warmup=2, blocks=5, calls=3):
    for fn in functions.values():
        for _ in range(warmup):
            fn()
    results = {k: [] for k in functions}
    keys = list(functions)
    for block in range(blocks):
        for key in keys[block % len(keys):] + keys[:block % len(keys)]:
            start = time.perf_counter()
            for _ in range(calls):
                functions[key]()
            results[key].append((time.perf_counter() - start) * 1000 / calls)
    return {k: dict(block_ms=v, median_ms=statistics.median(v), min_ms=min(v), max_ms=max(v))
            for k, v in results.items()}


def run():
    if OUT.exists():
        raise FileExistsError('Preserve the previous sidecar result.')
    OUT.mkdir(parents=True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    record = dict(started_unix=time.time(), pid=os.getpid(), threads=1, nice=os.getpriority(os.PRIO_PROCESS, 0),
        torch_version=torch.__version__, numpy_version=np.__version__, device='cpu',
        gpu_used=False, main_training_changed=False, fresh_confirmation=False,
        sources={str(p.relative_to(ROOT)): sha(p) for p in Path(__file__).parent.glob('*') if p.is_file()})
    try:
        record['parameter_checks'] = parameter_checks()
        record['saved_storage'] = saved_storage()
        record['sensitivity'] = sensitivity()
        write(OUT / 'progress.json', record)
        print('CPU_SYNTHETIC_COMPLETE', flush=True)
        data, labels, record['input_identity'] = old_inputs()
        c, s, _ = data
        record['descriptor_checks'] = []
        record['geometry_timings'] = []
        for tokens in (64, 128, 256, 512, 1024):
            ci, si = c[:, :tokens], s[:, :tokens]
            a = descriptors('relative5', ci, si)
            b = quadratic_descriptors(ci, si)
            radial_error = float((fixed_inputs(a) - radial_descriptors(ci)).abs().max())
            record['descriptor_checks'].append(dict(tokens=tokens, batch=len(ci),
                max_abs=float((a-b).abs().max()), radial_max_abs=radial_error,
                per_channel_max_abs=(a-b).abs().flatten(0, -2).max(0).values.tolist()))
            assert radial_error == 0
            funcs = {'original': lambda: descriptors('relative5', ci, si),
                     'quadratic': lambda: quadratic_descriptors(ci, si),
                     'sphere_original': lambda: fixed_inputs(descriptors('relative5', ci, si)),
                     'sphere_direct': lambda: radial_descriptors(ci)}
            with torch.no_grad():
                record['geometry_timings'].append(dict(tokens=tokens, batch=len(ci), timings=timing(funcs)))
            write(OUT / 'progress.json', record)
            print('CPU_GEOMETRY', tokens, record['descriptor_checks'][-1]['max_abs'], flush=True)
        lock = read(ROOT / 'results/modelnet_cross_source/attempt1/lock.json')
        checkpoints = {m['role']: m for m in lock['models'] if m['seed'] == 11 and m['role'] in ROLES}
        record['checkpoint_checks'], record['model_timings'] = [], []
        for role in ROLES:
            ck = checkpoints[role]
            assert sha(ck['path']) == ck['sha256']
            base = model_for(role).eval()
            base.load_state_dict(torch.load(ck['path'], map_location='cpu', weights_only=True)['state_dict'])
            before = state_hash(base)
            batch = tuple(t[:2, :512] for t in data)
            models = {name: variant(base, name).eval() for name in VARIANTS}
            with torch.no_grad():
                outputs = {name: model(*batch) for name, model in models.items()}
            for name, output in outputs.items():
                error = float((outputs['original'] - output).abs().max())
                changes = int((outputs['original'].argmax(-1) != output.argmax(-1)).sum())
                assert error < 2e-4 and changes == 0
                assert state_hash(models[name]) == before
                record['checkpoint_checks'].append(dict(role=role, variant=name, examples=2, tokens=512,
                    max_logit_abs=error, changed_predictions=changes, checkpoint=ck))
            # Tiny bounded whole-model timings; no CPU result is promoted to GPU throughput.
            for tokens in (128, 512):
                batch = tuple(t[:1, :tokens] for t in data)
                for mode in ('inference', 'forward_backward'):
                    def step(name):
                        model = models[name]
                        model.zero_grad(set_to_none=True)
                        model.train(mode == 'forward_backward')
                        if mode == 'inference':
                            with torch.no_grad():
                                model(*batch)
                        else:
                            F.cross_entropy(model(*batch), labels[:1], label_smoothing=.1).backward()
                    funcs = {name: lambda name=name: step(name) for name in VARIANTS}
                    record['model_timings'].append(dict(role=role, batch=1, tokens=tokens, mode=mode,
                        timings=timing(funcs, warmup=1, blocks=3, calls=1)))
                write(OUT / 'progress.json', record)
                print('CPU_MODEL', role, tokens, flush=True)
            assert all(state_hash(m) == before for m in models.values())
            del models, base
        record.update(state='complete', elapsed_seconds=time.time()-record['started_unix'],
            gpu_benchmark_pending=True, scientific_training_changed=False)
        write(OUT / 'result.json', record)
    except BaseException as error:
        write(OUT / 'failure.json', dict(error_type=type(error).__name__, message=str(error), partial=record))
        raise


if __name__ == '__main__':
    run()
