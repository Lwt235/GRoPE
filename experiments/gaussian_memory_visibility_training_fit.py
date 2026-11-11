"""Nine fixed-budget task-matched fits; no evaluation-based checkpoint choice."""
import argparse
import hashlib
import json
import time

import numpy as np
import torch

from experiments.gaussian_memory_visibility_training_data import (
    ROOT, OUT, KINDS, SEEDS, load_plan, verify_sources, read_samples, training_inputs,
    inputs_for, sha256, write_json, save_tensor, device_tree)
from experiments.gaussian_memory_incremental_train import schedule, loss_for
from experiments.gaussian_memory_view_data import load_backbone
from models.gaussian_memory_bound_read import BoundMemoryAdapter, bound_native_tail
from models.gaussian_memory_vggt_adapter import native_tail


def fit(backbone, samples, inputs, plan, kind, seed, steps, folder, pilot=False):
    folder.mkdir(parents=True, exist_ok=True); torch.manual_seed(seed)
    adapter = BoundMemoryAdapter().cuda()
    assert sum(p.numel() for p in adapter.parameters()) == plan['parameters']
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=plan['lr'], weight_decay=plan['weight_decay'])
    sequence = schedule(seed, len(samples), steps)
    identity = {'plan_sha256': sha256(OUT / 'plan.json'), 'source_sha256': plan['source_sha256'],
                'schedule_sha256': hashlib.sha256(sequence.tobytes()).hexdigest(), 'kind': kind, 'seed': seed}
    first_step = 0; latest = folder / 'latest.pt'
    if latest.exists():
        saved = torch.load(latest, map_location='cuda', weights_only=False)
        assert all(saved[k] == v for k, v in identity.items())
        adapter.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer']); first_step = saved['step']
    else:
        save_tensor(folder / 'initial.pt', dict(identity, model=adapter.state_dict()))
    logfile = folder / 'training.jsonl'
    if logfile.exists():
        lines = logfile.read_text().splitlines(); excess = [l for l in lines if json.loads(l)['step'] > first_step]
        if excess:
            (folder / f'interrupted_after_{first_step}.jsonl').write_text('\n'.join(excess) + '\n')
            keep = [l for l in lines if json.loads(l)['step'] <= first_step]
            logfile.write_text('\n'.join(keep) + ('\n' if keep else ''))
    start = time.monotonic(); losses = []; torch.cuda.reset_peak_memory_stats()
    with logfile.open('a', buffering=1) as log:
        for step in range(first_step, steps):
            optimizer.zero_grad(set_to_none=True); total = 0.; keys = []
            for index in sequence[step]:
                sample = device_tree(samples[int(index)]); content, prior = inputs_for(sample, inputs)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    depth, _ = bound_native_tail(backbone, sample, adapter, content, prior, True)
                    loss = loss_for(depth, sample)
                if not torch.isfinite(loss): raise FloatingPointError(f'Nonfinite loss {kind}/{seed}/{step + 1}')
                (loss / 2).backward(); total += loss.item() / 2
                keys.append([sample['id'], sample['query'], sample['condition'], sample['policy']])
                del sample, content, prior, depth, loss
            norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), plan['gradient_clip'])
            if not torch.isfinite(norm): raise FloatingPointError(f'Nonfinite gradient {kind}/{seed}/{step + 1}')
            grads = {n: None if p.grad is None else float(p.grad.norm()) for n, p in adapter.named_parameters()} if step in (0, 1, 19, steps - 1) else None
            optimizer.step(); losses.append(total)
            row = {'step': step + 1, 'loss': total, 'gradient_norm': norm.item(), 'samples': keys,
                   'elapsed_s': time.monotonic() - start}
            if grads is not None: row['gradient_by_parameter'] = grads
            log.write(json.dumps(row, allow_nan=False) + '\n')
            if (step + 1) % 100 == 0 or step + 1 == steps:
                torch.cuda.synchronize()
                save_tensor(latest, dict(identity, model=adapter.state_dict(), optimizer=optimizer.state_dict(), step=step + 1))
                progress = {'kind': kind, 'seed': seed, 'step': step + 1, 'steps': steps,
                    'last20_loss': float(np.mean(losses[-20:])), 'elapsed_s': time.monotonic() - start,
                    'peak_GiB': torch.cuda.max_memory_allocated() / 2**30}
                write_json(folder / 'progress.json', progress); print(json.dumps(progress), flush=True)
    assert all(torch.isfinite(t).all() for t in adapter.state_dict().values())
    save_tensor(folder / f'step_{steps}.pt', dict(identity, model=adapter.state_dict(), step=steps))
    rows = [json.loads(l) for l in logfile.read_text().splitlines()]; assert len(rows) == steps
    status = dict(identity, complete=True, steps=steps, parameters=plan['parameters'], pilot=pilot,
        seconds_this_invocation=time.monotonic() - start, peak_GiB=torch.cuda.max_memory_allocated() / 2**30,
        first20=float(np.mean([r['loss'] for r in rows[:20]])), last20=float(np.mean([r['loss'] for r in rows[-20:]])),
        checkpoint_sha256=sha256(folder / f'step_{steps}.pt'))
    write_json(folder / 'status.json', status)
    del optimizer, adapter; torch.cuda.empty_cache()
    return status


def preflight(plan):
    samples = read_samples(plan); backbone = load_backbone(); checks = {}; resources = {}
    for kind in KINDS:
        inputs = training_inputs(plan, samples, kind)
        sample = device_tree(samples[4]); content, prior = inputs_for(sample, inputs)
        torch.manual_seed(SEEDS[0]); adapter = BoundMemoryAdapter().cuda()
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            native, _ = native_tail(backbone, sample)
            zero, _ = bound_native_tail(backbone, sample, adapter, content, prior)
        checks[kind] = {'zero_max_abs': float((native - zero).abs().max()),
                        'native_replay_max_abs': float((native - sample['query_depth_native']).abs().max())}
        assert checks[kind]['zero_max_abs'] < 1e-6 * max(1., float(native.mean()))
        assert checks[kind]['native_replay_max_abs'] < 1e-5 * max(1., float(native.mean()))
        del sample, content, prior, adapter, native, zero; torch.cuda.empty_cache()
        folder = OUT / 'preflight' / f'{kind}_resource2'
        resources[kind] = fit(backbone, samples, inputs, plan, kind, SEEDS[0], 2, folder, True)
        rows = [json.loads(l) for l in (folder / 'training.jsonl').read_text().splitlines()]
        assert rows[1]['gradient_by_parameter']['q.weight'] > 0 and rows[1]['gradient_by_parameter']['out.weight'] > 0
        del inputs; torch.cuda.empty_cache()
    write_json(OUT / 'preflight/complete.json', {'complete': True, 'interfaces': checks, 'resources': resources,
        'plan_sha256': sha256(OUT / 'plan.json'), 'pilot_models_excluded': True})


def main():
    p = argparse.ArgumentParser(); p.add_argument('--preflight', action='store_true')
    p.add_argument('--kind', choices=KINDS); p.add_argument('--seed', type=int, choices=SEEDS)
    args = p.parse_args(); torch.set_num_threads(4); plan = load_plan(); verify_sources(plan)
    if args.preflight: preflight(plan); return
    assert args.kind and args.seed
    pre = json.loads((OUT / 'preflight/complete.json').read_text())
    assert pre['complete'] and pre['plan_sha256'] == sha256(OUT / 'plan.json')
    folder = OUT / f'{args.kind}_{args.seed}'
    if (folder / 'status.json').exists():
        s = json.loads((folder / 'status.json').read_text())
        if s['complete']:
            assert s['plan_sha256'] == sha256(OUT / 'plan.json') and s['checkpoint_sha256'] == sha256(folder / 'step_2000.pt')
            return
    samples = read_samples(plan); inputs = training_inputs(plan, samples, args.kind); backbone = load_backbone()
    fit(backbone, samples, inputs, plan, args.kind, args.seed, plan['steps'], folder)


if __name__ == '__main__': main()
