"""Matched fits for one geometry-conditioned internal memory read."""
import argparse
import hashlib
import json
import time

import numpy as np
import torch

from experiments.gaussian_memory_setup import ROOT, sha256
from experiments.gaussian_memory_view_data import load_backbone
from experiments.gaussian_memory_incremental_train import read_samples, schedule, loss_for
from experiments.gaussian_memory_bound_data import (
    OUT, KINDS, SEEDS, load_plan, load_priors, inputs_for, write_json, save_tensor, device_tree)
from models.gaussian_memory_vggt_adapter import native_tail
from models.gaussian_memory_bound_read import BoundMemoryAdapter, bound_native_tail


def source_identity():
    names = ['models/gaussian_memory_bound_read.py', 'experiments/gaussian_memory_bound_data.py',
             'experiments/gaussian_memory_bound_train.py', 'models/gaussian_memory_vggt_adapter.py',
             'experiments/gaussian_memory_incremental_data.py', 'experiments/gaussian_memory_incremental_train.py']
    return {name: sha256(ROOT / name) for name in names}


def fit(backbone, samples, initial, priors, kind, seed, steps, folder):
    folder.mkdir(parents=True, exist_ok=True); torch.manual_seed(seed)
    adapter = BoundMemoryAdapter().cuda()
    assert sum(p.numel() for p in adapter.parameters()) == 137472
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=3e-4, weight_decay=.01)
    sequence = schedule(seed, len(samples), steps)
    sequence_hash = hashlib.sha256(sequence.tobytes()).hexdigest(); identities = source_identity()
    memory = {rid: device_tree(x) for rid, x in initial.items()}; first_step = 0
    latest = folder / 'latest.pt'
    if latest.exists():
        saved = torch.load(latest, map_location='cuda', weights_only=False)
        assert saved['source_sha256'] == identities and saved['schedule_sha256'] == sequence_hash
        adapter.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer']); first_step = saved['step']
    else:
        save_tensor(folder / 'initial.pt', {'model': adapter.state_dict(), 'seed': seed, 'kind': kind,
                                          'source_sha256': identities, 'schedule_sha256': sequence_hash})
    logfile = folder / 'training.jsonl'
    if logfile.exists():
        lines = logfile.read_text().splitlines(); excess = [line for line in lines if json.loads(line)['step'] > first_step]
        if excess:
            (folder / f'interrupted_after_{first_step}.jsonl').write_text('\n'.join(excess) + '\n')
            logfile.write_text('\n'.join(line for line in lines if json.loads(line)['step'] <= first_step) + ('\n' if first_step else ''))
    start = time.monotonic(); losses = []; torch.cuda.reset_peak_memory_stats()
    with logfile.open('a', buffering=1) as log:
        for step in range(first_step, steps):
            optimizer.zero_grad(set_to_none=True); total = 0.; keys = []
            for index in sequence[step]:
                sample = device_tree(samples[int(index)])
                content, prior = inputs_for(sample, memory[sample['id']], kind, priors)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    depth, _ = bound_native_tail(backbone, sample, adapter, content, prior, True)
                    loss = loss_for(depth, sample)
                if not torch.isfinite(loss): raise FloatingPointError(f'Nonfinite loss: {kind}/{seed}/{step + 1}')
                (loss / 2).backward(); total += loss.item() / 2
                keys.append([sample['id'], sample['query'], sample['condition']])
                del sample, content, prior, depth, loss
            norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), 5.)
            if not torch.isfinite(norm): raise FloatingPointError(f'Nonfinite gradient: {kind}/{seed}/{step + 1}')
            gradients = {n: None if p.grad is None else float(p.grad.norm()) for n, p in adapter.named_parameters()} if step in (0, 1, 19, steps - 1) else None
            optimizer.step(); losses.append(total)
            row = {'step': step + 1, 'loss': total, 'gradient_norm': norm.item(), 'samples': keys, 'elapsed_s': time.monotonic() - start}
            if gradients is not None: row['gradient_by_parameter'] = gradients
            log.write(json.dumps(row, allow_nan=False) + '\n')
            if (step + 1) % 100 == 0 or step + 1 == steps:
                torch.cuda.synchronize()
                save_tensor(latest, {'model': adapter.state_dict(), 'optimizer': optimizer.state_dict(), 'step': step + 1,
                                    'seed': seed, 'kind': kind, 'source_sha256': identities, 'schedule_sha256': sequence_hash})
                progress = {'kind': kind, 'seed': seed, 'step': step + 1, 'steps': steps,
                            'last20_loss': float(np.mean(losses[-20:])), 'elapsed_s': time.monotonic() - start,
                            'peak_GiB': torch.cuda.max_memory_allocated() / 1024**3}
                write_json(folder / 'progress.json', progress); print(json.dumps(progress), flush=True)
    save_tensor(folder / f'step_{steps}.pt', {'model': adapter.state_dict(), 'step': steps, 'kind': kind, 'seed': seed,
                'source_sha256': identities, 'schedule_sha256': sequence_hash})
    rows = [json.loads(x) for x in logfile.read_text().splitlines()]
    status = {'complete': True, 'steps': steps, 'kind': kind, 'seed': seed, 'parameters': 137472,
              'seconds_this_invocation': time.monotonic() - start,
              'peak_GiB': torch.cuda.max_memory_allocated() / 1024**3,
              'first20': float(np.mean([r['loss'] for r in rows[:20]])),
              'last20': float(np.mean([r['loss'] for r in rows[-20:]])),
              'schedule_sha256': sequence_hash, 'checkpoint_sha256': sha256(folder / f'step_{steps}.pt')}
    write_json(folder / 'status.json', status)
    del optimizer, adapter, memory; torch.cuda.empty_cache()
    return status


def preflight():
    plan = load_plan(); ids = plan['training_ids'][:2]
    samples, initial = read_samples(ids); priors = load_priors(ids); backbone = load_backbone()
    sample = device_tree(samples[0]); state = device_tree(initial[sample['id']]); torch.manual_seed(SEEDS[0])
    adapter = BoundMemoryAdapter().cuda(); content, prior = inputs_for(sample, state, 'bound', priors)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        original, _ = native_tail(backbone, sample)
        zero, _ = bound_native_tail(backbone, sample, adapter, content, prior)
    check = {'native_replay_max_abs': float((original - sample['query_depth_native']).abs().max()),
             'zero_adapter_max_abs': float((zero - original).abs().max()), 'parameters': 137472,
             'source_sha256': source_identity()}
    write_json(OUT / 'preflight/interface_checks.json', check)
    assert check['native_replay_max_abs'] < 1e-5 * max(1., float(original.mean())) and check['zero_adapter_max_abs'] < 1e-6 * max(1., float(original.mean()))
    del adapter, sample, state, original, zero, content, prior; torch.cuda.empty_cache()
    resource = fit(backbone, samples, initial, priors, 'bound', SEEDS[0], 20, OUT / 'preflight/resource20')
    learning = fit(backbone, samples, initial, priors, 'bound', SEEDS[0], 200, OUT / 'preflight/learning200')
    write_json(OUT / 'preflight/complete.json', {'complete': True, 'interface': check, 'resource20': resource,
               'learning200': learning, 'pilot_models_excluded': True})


def train(kind=None, seed=None):
    plan = load_plan(); assert json.loads((OUT / 'preflight/complete.json').read_text())['complete']
    samples, initial = read_samples(plan['training_ids']); priors = load_priors(plan['training_ids'], kind == 'center')
    backbone = load_backbone()
    for s in ([seed] if seed is not None else SEEDS):
        for k in ([kind] if kind is not None else KINDS):
            folder = OUT / f'{k}_{s}'
            if (folder / 'status.json').exists():
                old = json.loads((folder / 'status.json').read_text())
                if old['complete']:
                    assert old['checkpoint_sha256'] == sha256(folder / 'step_2000.pt'); continue
            fit(backbone, samples, initial, priors, k, s, 2000, folder)
    write_json(OUT / f'training_{kind or "all"}_{seed or "all"}_complete.json', {'complete': True, 'kind': kind, 'seed': seed})


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=('preflight', 'train'))
    p.add_argument('--kind', choices=KINDS + ('center',)); p.add_argument('--seed', type=int); args = p.parse_args()
    torch.set_num_threads(4)
    if args.stage == 'preflight': preflight()
    else: train(args.kind, args.seed)
