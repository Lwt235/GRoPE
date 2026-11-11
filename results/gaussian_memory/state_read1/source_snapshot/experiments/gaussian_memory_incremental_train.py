"""Matched internal VGGT adapter training; all native network weights stay frozen."""
import argparse
import hashlib
import json
import time

import numpy as np
import torch

from experiments.gaussian_memory_view_data import load_backbone
from experiments.gaussian_memory_setup import ROOT, sha256
from experiments.gaussian_memory_incremental_data import (
    OUT, CACHE, SEEDS, KINDS, QUERIES, load_plan, write_json, save_tensor, device_tree)
from models.gaussian_memory_vggt_adapter import InternalMemoryAdapter, native_tail, local_position, padded_geometry


def inputs_for(sample, initial, kind, statistics, replacement=None):
    memory = initial['memory'] if replacement is None else replacement
    content = memory['features'] if kind in ('content', 'combined', 'center') else sample['local_content']
    if kind in ('geometry', 'combined', 'center'):
        rms = sample['prefix'].new_tensor(statistics['geometry_rms'])
        if replacement is None and kind != 'center':
            pe = (sample['pe_raw'] / rms).clamp(-20, 20)
        else:
            pe = padded_geometry(memory, sample['c2w'], sample['K'], rms,
                                 statistics['center_variance'] if kind == 'center' else None)
    else:
        pe = local_position(sample['prefix'].device)
    return pe, content


def read_samples(ids):
    samples, initial = [], {}
    for rid in ids:
        identity = json.loads((CACHE / rid / 'identity.json').read_text())
        for name, digest in identity['sha256'].items():
            assert sha256(CACHE / rid / name) == digest, (rid, name)
        initial[rid] = torch.load(CACHE / rid / 'initial.pt', map_location='cpu', weights_only=False)
        for q in QUERIES:
            for c in range(3):
                s = torch.load(CACHE / rid / f'q{q}_c{c}.pt', map_location='cpu', weights_only=False)
                if s['metric_scale'] is not None:
                    samples.append(s)
    assert samples
    return samples, initial


def schedule(seed, count, steps):
    rng = np.random.default_rng(seed)
    indices = []
    while len(indices) < steps * 2:
        indices.extend(rng.permutation(count).tolist())
    return np.asarray(indices[:steps * 2], dtype=np.int64).reshape(steps, 2)


def loss_for(prediction, sample):
    log_error = (prediction.float().clamp_min(1e-6) * sample['metric_scale']).log() - sample['target'].log()
    return log_error[sample['valid']].square().mean()


def source_identity():
    names = ['models/gaussian_memory_vggt_adapter.py', 'experiments/gaussian_memory_incremental_data.py',
             'experiments/gaussian_memory_incremental_train.py']
    return {name: sha256(ROOT / name) for name in names}


def fit(backbone, samples, initial, statistics, kind, seed, steps, folder, checkpoint_blocks=True):
    folder.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    adapter = InternalMemoryAdapter().cuda()
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=3e-4, weight_decay=.01)
    sequence = schedule(seed, len(samples), steps)
    sequence_hash = hashlib.sha256(sequence.tobytes()).hexdigest()
    identities = source_identity()
    memory_gpu = {rid: device_tree(value) for rid, value in initial.items()}
    latest = folder / 'latest.pt'
    first_step = 0
    if latest.exists():
        state = torch.load(latest, map_location='cuda', weights_only=False)
        assert state['source_sha256'] == identities
        assert state['schedule_sha256'] == sequence_hash
        adapter.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        first_step = state['step']
    else:
        save_tensor(folder / 'initial.pt', {'model': adapter.state_dict(), 'seed': seed,
                    'kind': kind, 'source_sha256': identities, 'schedule_sha256': sequence_hash})
    log_path = folder / 'training.jsonl'
    if log_path.exists():
        lines = log_path.read_text().splitlines()
        excess = [x for x in lines if json.loads(x)['step'] > first_step]
        if excess:
            (folder / f'interrupted_after_{first_step}.jsonl').write_text('\n'.join(excess) + '\n')
            log_path.write_text('\n'.join(x for x in lines if json.loads(x)['step'] <= first_step) + ('\n' if first_step else ''))
    loss_values = []
    start = time.monotonic(); torch.cuda.reset_peak_memory_stats()
    with log_path.open('a', buffering=1) as log:
        for step in range(first_step, steps):
            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.; keys = []
            for sample_index in sequence[step]:
                sample = device_tree(samples[int(sample_index)])
                pe, content = inputs_for(sample, memory_gpu[sample['id']], kind, statistics)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    depth, _ = native_tail(backbone, sample, adapter, pe, content, checkpoint_blocks)
                    loss = loss_for(depth, sample)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Nonfinite loss at {kind}/{seed}/{step + 1}')
                (loss / 2).backward()
                total_loss += loss.detach().item() / 2
                keys.append([sample['id'], sample['query'], sample['condition']])
                del sample, pe, content, depth, loss
            norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), 5.)
            if not torch.isfinite(norm):
                raise FloatingPointError(f'Nonfinite gradient at {kind}/{seed}/{step + 1}')
            gradients = None
            if step in (0, 1, 19, steps - 1):
                gradients = {name: float(p.grad.norm()) if p.grad is not None else None for name, p in adapter.named_parameters()}
            optimizer.step(); loss_values.append(total_loss)
            row = {'step': step + 1, 'loss': total_loss, 'gradient_norm': norm.item(),
                   'samples': keys, 'elapsed_s': time.monotonic() - start}
            if gradients is not None: row['gradient_by_parameter'] = gradients
            log.write(json.dumps(row, allow_nan=False) + '\n')
            if (step + 1) % 100 == 0 or step + 1 == steps:
                torch.cuda.synchronize()
                state = {'model': adapter.state_dict(), 'optimizer': optimizer.state_dict(), 'step': step + 1,
                         'seed': seed, 'kind': kind, 'schedule_sha256': sequence_hash,
                         'source_sha256': identities, 'statistics': statistics}
                save_tensor(latest, state)
                progress = {'kind': kind, 'seed': seed, 'step': step + 1, 'steps': steps,
                            'last20_loss': float(np.mean(loss_values[-20:])),
                            'elapsed_s': time.monotonic() - start,
                            'peak_GiB': torch.cuda.max_memory_allocated() / 1024**3}
                write_json(folder / 'progress.json', progress)
                print(json.dumps(progress), flush=True)
    torch.cuda.synchronize()
    save_tensor(folder / f'step_{steps}.pt', {'model': adapter.state_dict(), 'step': steps,
                'kind': kind, 'seed': seed, 'source_sha256': identities,
                'schedule_sha256': sequence_hash, 'statistics': statistics})
    all_rows = [json.loads(line) for line in log_path.read_text().splitlines()]
    status = {'complete': True, 'steps': steps, 'kind': kind, 'seed': seed,
              'seconds_this_invocation': time.monotonic() - start,
              'peak_GiB': torch.cuda.max_memory_allocated() / 1024**3,
              'first20': float(np.mean([r['loss'] for r in all_rows[:20]])),
              'last20': float(np.mean([r['loss'] for r in all_rows[-20:]])),
              'parameters': sum(p.numel() for p in adapter.parameters()),
              'schedule_sha256': sequence_hash,
              'checkpoint_sha256': sha256(folder / f'step_{steps}.pt')}
    write_json(folder / 'status.json', status)
    del optimizer, adapter, memory_gpu
    torch.cuda.empty_cache()
    return status


def preflight():
    plan = load_plan(); ids = plan['training_ids'][:2]
    samples, initial = read_samples(ids)
    points = torch.cat([s['pe_raw'][:, 5:].reshape(-1, 10) for s in samples if s['condition'] == 0])
    statistics = {'geometry_rms': points.square().mean(0).sqrt().clamp_min(1e-4).tolist(),
                  'center_variance': 1., 'pilot_only': True, 'training_ids': ids}
    folder = OUT / 'preflight'; folder.mkdir(parents=True, exist_ok=True)
    backbone = load_backbone(); sample = device_tree(samples[0]); state = device_tree(initial[sample['id']])
    torch.manual_seed(SEEDS[0]); adapter = InternalMemoryAdapter().cuda()
    pe, content = inputs_for(sample, state, 'combined', statistics)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        original, _ = native_tail(backbone, sample)
        zero, _ = native_tail(backbone, sample, adapter, pe, content)
    comparison = {'replayed_native_max_abs': (original - sample['query_depth_native']).abs().max().item(),
                  'zero_adapter_max_abs': (zero - original).abs().max().item(),
                  'native_depth_mean': original.mean().item(),
                  'original_loss': loss_for(original, sample).item(),
                  'parameter_count': sum(p.numel() for p in adapter.parameters()),
                  'source_sha256': source_identity()}
    write_json(folder / 'interface_checks.json', comparison)
    assert comparison['replayed_native_max_abs'] < 1e-5 * max(1., comparison['native_depth_mean']), comparison
    assert comparison['zero_adapter_max_abs'] < 1e-6 * max(1., comparison['native_depth_mean']), comparison
    del adapter, sample, state, original, zero, pe, content
    torch.cuda.empty_cache()
    timing = fit(backbone, samples, initial, statistics, 'combined', SEEDS[0], 20, folder / 'resource20')
    write_json(folder / 'resource20_summary.json', timing)
    estimate = timing['seconds_this_invocation'] / 20 * 2000 * 12
    print(json.dumps({'pilot_resource': timing, 'rough_12_fit_seconds': estimate}), flush=True)
    learning = fit(backbone, samples, initial, statistics, 'combined', SEEDS[0], 200, folder / 'learning200')
    write_json(folder / 'complete.json', {'complete': True, 'interface': comparison,
               'resource20': timing, 'learning200': learning, 'pilot_models_excluded': True,
               'rough_12_fit_seconds': estimate})


def train(kind=None, seed=None):
    plan = load_plan()
    assert json.loads((OUT / 'preflight/complete.json').read_text())['complete']
    statistics = json.loads((OUT / 'training_statistics.json').read_text())
    samples, initial = read_samples(plan['training_ids'])
    backbone = load_backbone()
    for s in ([seed] if seed is not None else SEEDS):
        for k in ([kind] if kind is not None else KINDS):
            folder = OUT / f'{k}_{s}'
            if (folder / 'status.json').exists():
                old = json.loads((folder / 'status.json').read_text())
                if old['complete']:
                    assert old['checkpoint_sha256'] == sha256(folder / 'step_2000.pt')
                    continue
            fit(backbone, samples, initial, statistics, k, s, 2000, folder)
    write_json(OUT / f'training_{kind or "all"}_{seed or "all"}_complete.json',
               {'complete': True, 'kind': kind, 'seed': seed})


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=('preflight', 'train'))
    p.add_argument('--kind', choices=KINDS + ('center',)); p.add_argument('--seed', type=int)
    args = p.parse_args()
    torch.set_num_threads(4)
    if args.stage == 'preflight': preflight()
    else: train(args.kind, args.seed)
