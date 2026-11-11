"""Matched four-arm native query adaptation, fixed final checkpoints only."""
import argparse
import hashlib
import json
import time

import numpy as np
import torch

from experiments.gaussian_memory_native_pe_data import (
    ROOT, OUT, CACHE, KINDS, SEEDS, load, read, write_json, save_tensor, sha256,
    load_plan, verify, examples, sample_path, sample_key, input_path)
from experiments.gaussian_memory_native_run import model_load, scores, metric_depth
from models.gaussian_memory_native_pe import NativePEReader, native_tail, relations, loss_for


def device_sample(cpu):
    names = ('prefix', 'early', 'memory', 'means', 'covariances', 'target', 'valid', 'hidden')
    def move(value):
        if isinstance(value, torch.Tensor): return value.cuda(non_blocking=True)
        if isinstance(value, list): return [move(v) for v in value]
        if isinstance(value, dict): return {k: move(v) for k, v in value.items()}
        return value
    return {k: move(v) if k in names else v for k, v in cpu.items() if k not in ('geometry_depths', 'geometry_poses', 'predicted_intrinsics', 'predicted_extrinsics')}


def model_digest(adapter):
    h = hashlib.sha256()
    for k, v in adapter.state_dict().items(): h.update(k.encode()); h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def schedule(seed, n, steps):
    rng = np.random.default_rng(seed); order = []
    while len(order) < steps * 2: order.extend(rng.permutation(n).tolist())
    return np.asarray(order[:steps * 2], dtype=np.int64).reshape(steps, 2)


def read_samples(plan, kind, keys=None):
    keys = keys or examples(plan, 'train'); result = []
    for key in keys:
        path = sample_path(*key)
        rec = read(OUT / 'cache' / (sample_key(*key).replace('/', '_') + '.json'))
        assert rec['plan_sha256'] == sha256(OUT / 'plan.json') and sha256(path) == rec['sha256']
        s = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        z = relations(s, kind).cpu()
        result.append((s, z))
    return result


def fit(model, data, plan, kind, seed, steps, folder, pilot=False):
    folder.mkdir(parents=True, exist_ok=True); torch.manual_seed(seed)
    adapter = NativePEReader().cuda(); optimizer = torch.optim.AdamW(adapter.parameters(), lr=plan['lr'], weight_decay=plan['weight_decay'])
    initial_digest = model_digest(adapter); sequence = schedule(seed, len(data), steps)
    identity = {'kind': kind, 'seed': seed, 'plan_sha256': sha256(OUT / 'plan.json'),
                'schedule_sha256': hashlib.sha256(sequence.tobytes()).hexdigest(), 'initial_state_sha256': initial_digest,
                'training_lock_sha256': sha256(OUT / 'training_lock.json') if (OUT / 'training_lock.json').exists() else None,
                'pilot': pilot, 'parameters': sum(p.numel() for p in adapter.parameters())}
    latest = folder / 'latest.pt'; first = 0
    if latest.exists():
        previous = load(latest)
        for k, v in identity.items(): assert previous[k] == v
        adapter.load_state_dict(previous['model']); optimizer.load_state_dict(previous['optimizer']); first = previous['step']
    else: save_tensor(folder / 'initial.pt', dict(identity, model=adapter.state_dict()))
    logfile = folder / 'training.jsonl'
    if logfile.exists():
        lines = logfile.read_text().splitlines(); good = [l for l in lines if json.loads(l)['step'] <= first]
        if len(good) < len(lines):
            (folder / f'interrupted_after_{first}.jsonl').write_text('\n'.join(lines[len(good):]) + '\n')
            logfile.write_text('\n'.join(good) + ('\n' if good else ''))
    started = time.monotonic(); torch.cuda.reset_peak_memory_stats(); losses = []
    with logfile.open('a', buffering=1) as log:
        for step in range(first, steps):
            optimizer.zero_grad(set_to_none=True); total = 0.; keys = []
            for index in sequence[step]:
                cpu, z = data[int(index)]; sample = device_sample(cpu); z_gpu = z.cuda(non_blocking=True)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    depth = native_tail(model, sample, adapter, kind, z_gpu)
                    loss = loss_for(depth, sample)
                if not torch.isfinite(loss): raise FloatingPointError(f'Nonfinite loss {kind}/{seed}/{step}')
                (loss / 2).backward(); total += float(loss.detach()) / 2
                keys.append([cpu['id'], cpu['query'], cpu['condition']]); del sample, z_gpu, depth, loss
            norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), plan['gradient_clip'])
            if not torch.isfinite(norm): raise FloatingPointError(f'Nonfinite gradient {kind}/{seed}/{step}')
            grads = {n: None if p.grad is None else float(p.grad.norm()) for n, p in adapter.named_parameters()} if step in (0, 1, 19, steps - 1) else None
            optimizer.step(); losses.append(total)
            row = {'step': step + 1, 'loss': total, 'gradient_norm': float(norm), 'samples': keys,
                   'elapsed_s': time.monotonic() - started}
            if grads is not None: row['gradient_by_parameter'] = grads
            log.write(json.dumps(row, allow_nan=False) + '\n')
            if (step + 1) % 50 == 0 or step + 1 == steps:
                torch.cuda.synchronize()
                progress = dict(identity, step=step + 1, steps=steps, last20_loss=float(np.mean(losses[-20:])),
                                elapsed_s=time.monotonic() - started, peak_GiB=torch.cuda.max_memory_allocated() / 2**30)
                write_json(folder / 'progress.json', progress); print(json.dumps(progress), flush=True)
            if (step + 1) % 200 == 0 or step + 1 == steps:
                save_tensor(latest, dict(identity, step=step + 1, model=adapter.state_dict(), optimizer=optimizer.state_dict()))
    final = folder / f'step_{steps}.pt'; save_tensor(final, dict(identity, step=steps, model=adapter.state_dict()))
    rows = [json.loads(l) for l in logfile.read_text().splitlines()]; assert len(rows) == steps
    status = dict(identity, complete=True, steps=steps, checkpoint_sha256=sha256(final),
                  first20=float(np.mean([r['loss'] for r in rows[:20]])), last20=float(np.mean([r['loss'] for r in rows[-20:]])),
                  seconds_this_invocation=time.monotonic() - started, peak_GiB=torch.cuda.max_memory_allocated() / 2**30)
    write_json(folder / 'status.json', status); del adapter, optimizer; torch.cuda.empty_cache(); return status


@torch.no_grad()
def replay_checks(model, cpu):
    checks = {}; outputs = {}; torch.manual_seed(SEEDS[0]); reader = NativePEReader().cuda()
    sample = device_sample(cpu)
    for kind in KINDS:
        z = relations(cpu, kind)
        with torch.autocast('cuda', dtype=torch.bfloat16): d = native_tail(model, sample, reader, kind, z, False)
        outputs[kind] = d.cpu()
        if kind == 'native':
            delta = d.cpu() - cpu['baseline_depth']
            rms = float(delta.square().mean().sqrt() / cpu['baseline_depth'].square().mean().sqrt())
            assert rms < .01, rms
            checks['native_relative_rms'] = rms; checks['native_max_abs'] = float(delta.abs().max())
    checks['center_sphere_initial_max_abs'] = float((outputs['center'] - outputs['sphere']).abs().max())
    checks['center_gaussian_initial_max_abs'] = float((outputs['center'] - outputs['gaussian']).abs().max())
    assert checks['center_sphere_initial_max_abs'] == checks['center_gaussian_initial_max_abs'] == 0
    assert not any(p.requires_grad for p in model.parameters())
    del sample, reader; torch.cuda.empty_cache(); return checks


def preflight(model, plan):
    preflight_folder = OUT / 'preflight_final'
    keys = [(rid, 11, 'persistent') for rid in plan['training_ids'][:2]]
    checks = []; resources = {}
    for key in keys:
        cpu = load(sample_path(*key)); checks.append({'key': key, **replay_checks(model, cpu)})
    for kind in KINDS:
        data = read_samples(plan, kind, keys)
        resources[kind] = fit(model, data, plan, kind, SEEDS[0], 4, preflight_folder / kind, True)
        rows = [json.loads(l) for l in (preflight_folder / kind / 'training.jsonl').read_text().splitlines()]
        gradients = rows[1]['gradient_by_parameter']
        assert gradients['qkv.0.a.weight'] > 0 and gradients['qkv.0.b.weight'] > 0
        assert gradients['relation.2.weight'] > 0
        del data; torch.cuda.empty_cache()
    write_json(preflight_folder / 'complete.json', {'complete': True, 'checks': checks, 'resources': resources,
        'plan_sha256': sha256(OUT / 'plan.json'), 'pilot_models_excluded': True, 'development_scores_used': False})


@torch.inference_mode()
def evaluate(model, plan, kind, seed):
    folder = OUT / f'{kind}_{seed}'; status = read(folder / 'status.json'); checkpoint = folder / f'step_{plan["steps"]}.pt'
    assert status['complete'] and status['checkpoint_sha256'] == sha256(checkpoint)
    adapter = NativePEReader().cuda().eval(); adapter.load_state_dict(load(checkpoint)['model'])
    rows = []; pred = []; keys = []; started = time.monotonic()
    for key in examples(plan, 'development'):
        cpu = load(sample_path(*key)); sample = device_sample(cpu); z = relations(cpu, kind)
        with torch.autocast('cuda', dtype=torch.bfloat16): depth = native_tail(model, sample, adapter, kind, z, False)
        depth = depth.float().cpu()
        rows.append({'id': cpu['id'], 'place': cpu['place'], 'condition': cpu['condition'], 'kind': kind, 'seed': seed, **scores(depth, cpu)})
        pred.append(metric_depth(depth).float().cpu()); keys.append(list(key))
        del cpu, sample, z, depth
    path = CACHE / 'evaluation' / f'{kind}_{seed}.pt'; path.parent.mkdir(parents=True, exist_ok=True)
    save_tensor(path, {'keys': keys, 'depth_metric': torch.stack(pred)})
    write_json(OUT / 'evaluation' / f'{kind}_{seed}.json', {'complete': True, 'rows': rows,
        'checkpoint_sha256': sha256(checkpoint), 'prediction_path': str(path), 'prediction_sha256': sha256(path),
        'plan_sha256': sha256(OUT / 'plan.json'), 'seconds': time.monotonic() - started,
        'peak_GiB': torch.cuda.max_memory_allocated() / 2**30})
    del adapter; torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['preflight', 'train', 'evaluate'])
    p.add_argument('--kind', choices=KINDS); p.add_argument('--seeds', nargs='+', type=int, choices=SEEDS, default=list(SEEDS))
    args = p.parse_args(); torch.set_num_threads(4); plan = load_plan(); verify(plan); model = model_load(plan)
    if args.action == 'preflight': preflight(model, plan); return
    assert args.kind
    if args.action == 'train':
        pre = read(OUT / 'preflight_final/complete.json'); assert pre['complete'] and pre['plan_sha256'] == sha256(OUT / 'plan.json')
        assert (OUT / 'training_lock.json').exists()
        data = read_samples(plan, args.kind)
        for seed in args.seeds:
            folder = OUT / f'{args.kind}_{seed}'
            if not (folder / 'status.json').exists(): fit(model, data, plan, args.kind, seed, plan['steps'], folder)
        del data
        for seed in args.seeds: evaluate(model, plan, args.kind, seed)
        return
    for seed in args.seeds: evaluate(model, plan, args.kind, seed)


if __name__ == '__main__': main()
