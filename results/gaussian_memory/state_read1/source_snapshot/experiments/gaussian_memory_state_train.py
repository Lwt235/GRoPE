"""Matched dynamic-state fits; the architecture and frozen parent experiments are unchanged."""
import argparse
import hashlib
import json
import time

import numpy as np
import torch

from experiments.gaussian_memory_setup import sha256
from experiments.gaussian_memory_view_data import load_backbone
from experiments.gaussian_memory_incremental_train import read_samples, schedule, loss_for
from experiments.gaussian_memory_state_data import (
    OUT, MATCH_OUT, SEEDS, KINDS, prepare, load_plan, source_identity, training_inputs,
    inputs_for, write_json, save_tensor, device_tree)
from models.gaussian_memory_bound_read import BoundMemoryAdapter, bound_native_tail
from models.gaussian_memory_vggt_adapter import native_tail


def fit(backbone, samples, inputs, kind, seed, steps, folder, match_parent=False):
    folder.mkdir(parents=True, exist_ok=True); torch.manual_seed(seed)
    adapter = BoundMemoryAdapter().cuda()
    assert sum(p.numel() for p in adapter.parameters()) == 137472
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=3e-4, weight_decay=.01)
    sequence = schedule(seed, len(samples), steps)
    sequence_hash = hashlib.sha256(sequence.tobytes()).hexdigest(); identities = source_identity()
    paired_rows = None
    if match_parent:
        parent = MATCH_OUT / f'content_{seed}'
        parent_initial = torch.load(parent / 'initial.pt', map_location='cpu', weights_only=False)
        assert sequence_hash == parent_initial['schedule_sha256']
        assert all(torch.equal(t.cpu(), parent_initial['model'][n]) for n, t in adapter.state_dict().items())
        paired_rows = [json.loads(line) for line in (parent / 'training.jsonl').read_text().splitlines()]
        assert len(paired_rows) == steps
    first_step = 0
    latest = folder / 'latest.pt'
    if latest.exists():
        saved = torch.load(latest, map_location='cuda', weights_only=False)
        assert saved['source_sha256'] == identities and saved['schedule_sha256'] == sequence_hash
        assert saved['kind'] == kind and saved['seed'] == seed
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
                content, prior = inputs_for(sample, inputs)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    depth, _ = bound_native_tail(backbone, sample, adapter, content, prior, True)
                    loss = loss_for(depth, sample)
                if not torch.isfinite(loss): raise FloatingPointError(f'Nonfinite loss: {kind}/{seed}/{step + 1}')
                (loss / 2).backward(); total += loss.item() / 2
                keys.append([sample['id'], sample['query'], sample['condition']])
                del sample, content, prior, depth, loss
            if paired_rows is not None:
                assert keys == paired_rows[step]['samples'], f'Parent sampling mismatch at step {step + 1}'
            norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), 5.)
            if not torch.isfinite(norm): raise FloatingPointError(f'Nonfinite gradient: {kind}/{seed}/{step + 1}')
            gradients = {n: None if p.grad is None else float(p.grad.norm()) for n, p in adapter.named_parameters()} if step in (0, 1, 19, steps - 1) else None
            optimizer.step(); losses.append(total)
            row = {'step': step + 1, 'loss': total, 'gradient_norm': norm.item(), 'samples': keys,
                   'state_indices': [(q - 3) // 2 for _, q, _ in keys],
                   'elapsed_s': time.monotonic() - start}
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
    assert all(torch.isfinite(t).all() for t in adapter.state_dict().values())
    save_tensor(folder / f'step_{steps}.pt', {'model': adapter.state_dict(), 'step': steps, 'kind': kind, 'seed': seed,
                'source_sha256': identities, 'schedule_sha256': sequence_hash})
    rows = [json.loads(line) for line in logfile.read_text().splitlines()]
    status = {'complete': True, 'steps': steps, 'kind': kind, 'seed': seed, 'parameters': 137472,
              'seconds_this_invocation': time.monotonic() - start,
              'peak_GiB': torch.cuda.max_memory_allocated() / 1024**3,
              'first20': float(np.mean([r['loss'] for r in rows[:20]])),
              'last20': float(np.mean([r['loss'] for r in rows[-20:]])),
              'schedule_sha256': sequence_hash, 'checkpoint_sha256': sha256(folder / f'step_{steps}.pt'),
              'parent_content_initialization_equal': match_parent,
              'all_sampling_steps_equal_to_parent_content': match_parent,
              'pilot': not match_parent, 'training_policy': 'corrective_clean'}
    write_json(folder / 'status.json', status)
    del optimizer, adapter; torch.cuda.empty_cache()
    return status


def preflight():
    plan = prepare(); ids = plan['training_ids']
    samples, initial = read_samples(ids); del initial
    backbone = load_backbone(); resources = {}; interface = None
    for kind in ('content', 'soft'):
        inputs, identities = training_inputs(ids, samples, kind)
        write_json(OUT / 'preflight' / f'{kind}_input_identity.json', identities)
        sample = device_tree(samples[0]); torch.manual_seed(SEEDS[0])
        adapter = BoundMemoryAdapter().cuda(); content, prior = inputs_for(sample, inputs)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            native, _ = native_tail(backbone, sample)
            zero, _ = bound_native_tail(backbone, sample, adapter, content, prior)
        check = {'zero_adapter_max_abs': float((zero - native).abs().max()),
                 'native_replay_max_abs': float((native - sample['query_depth_native']).abs().max()),
                 'source_sha256': source_identity(), 'parameters': 137472}
        assert check['zero_adapter_max_abs'] < 1e-6 * max(1., float(native.mean()))
        assert check['native_replay_max_abs'] < 1e-5 * max(1., float(native.mean()))
        write_json(OUT / 'preflight' / f'{kind}_interface.json', check)
        interface = check
        del adapter, sample, native, zero, content, prior; torch.cuda.empty_cache()
        resources[kind] = fit(backbone, samples, inputs, kind, SEEDS[0], 2,
                              OUT / 'preflight' / f'{kind}_resource2')
        rows = [json.loads(line) for line in (OUT / 'preflight' / f'{kind}_resource2' / 'training.jsonl').read_text().splitlines()]
        assert rows[1]['gradient_by_parameter']['q.weight'] > 0
        assert rows[1]['gradient_by_parameter']['out.weight'] > 0
        del inputs; torch.cuda.empty_cache()
    write_json(OUT / 'preflight/complete.json', {'complete': True, 'interface': interface,
               'resource2': resources, 'training_inputs_causal': True, 'pilot_models_excluded': True,
               'scope': 'Matched inputs, zero residual, two finite gradient steps and resource check only.'})


def train(kind='soft', seed=None):
    plan = prepare(); pilot = json.loads((OUT / 'preflight/complete.json').read_text())
    assert pilot['complete'] and pilot['interface']['source_sha256'] == source_identity()
    assert kind in KINDS
    if kind == 'soft_center':
        assert json.loads((OUT / 'conditional_decision.json').read_text())['center_control_triggered']
    seeds = [seed] if seed is not None else SEEDS
    assert all(s in SEEDS for s in seeds)
    pending = []
    for s in seeds:
        path = OUT / f'{kind}_{s}/status.json'
        if path.exists() and json.loads(path.read_text())['complete']:
            saved = torch.load(path.parent / 'step_2000.pt', map_location='cpu', weights_only=False)
            assert saved['source_sha256'] == source_identity()
            assert sha256(path.parent / 'step_2000.pt') == json.loads(path.read_text())['checkpoint_sha256']
        else: pending.append(s)
    if not pending: return
    # Separate per-fit marker avoids concurrent writers to a shared progress file.
    write_json(OUT / f'training_start_{kind}_{seed or "all"}.json',
               {'started': True, 'scientific_source_sha256': source_identity()})
    samples, initial = read_samples(plan['training_ids']); del initial
    inputs, identities = training_inputs(plan['training_ids'], samples, kind)
    write_json(OUT / f'input_identity_{kind}_{seed or "all"}.json', identities)
    backbone = load_backbone()
    for s in pending:
        fit(backbone, samples, inputs, kind, s, 2000, OUT / f'{kind}_{s}', match_parent=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=('prepare', 'preflight', 'train'))
    p.add_argument('--kind', choices=KINDS, default='soft'); p.add_argument('--seed', type=int)
    args = p.parse_args(); torch.set_num_threads(4)
    if args.stage == 'prepare': prepare()
    elif args.stage == 'preflight': preflight()
    else: train(args.kind, args.seed)
