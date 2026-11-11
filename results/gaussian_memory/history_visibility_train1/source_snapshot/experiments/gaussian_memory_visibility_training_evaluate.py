"""Fixed evaluation of nine final checkpoints against unchanged parent inputs."""
import argparse
import json
import time

import numpy as np
import torch

from experiments.gaussian_memory_visibility_training_data import (
    ROOT, OUT, CACHE, PARENT, EVAL_CACHE, KINDS, SEEDS, CONDITIONS, POLICIES,
    load_plan, verify_sources, sha256, write_json, save_tensor, device_tree)
from experiments.gaussian_memory_view_data import load_backbone
from experiments.gaussian_memory_state_data import memory_prior
from experiments.gaussian_memory_bound_evaluate import predict
from experiments.gaussian_memory_incremental_evaluate import score
from models.gaussian_memory_bound_read import BoundMemoryAdapter

FAMILY = {'content': 'new_content', 'soft': 'new_soft', 'soft_center': 'new_center'}


def load_adapters(plan):
    adapters = {}; checks = {}
    for kind in KINDS:
        for seed in SEEDS:
            p = OUT / f'{kind}_{seed}/step_2000.pt'
            status = json.loads((p.parent / 'status.json').read_text())
            assert status['complete'] and not status['pilot'] and status['steps'] == 2000
            assert sha256(p) == status['checkpoint_sha256']
            state = torch.load(p, map_location='cpu', weights_only=False)
            assert state['plan_sha256'] == sha256(OUT / 'plan.json') and state['source_sha256'] == plan['source_sha256']
            model = BoundMemoryAdapter().cuda().eval(); model.load_state_dict(state['model'])
            adapters[kind, seed] = model; checks[f'{kind}_{seed}'] = sha256(p)
    return adapters, checks


@torch.no_grad()
def evaluate(meta, backbone, adapters, checkpoints, plan):
    rid = meta['id']; marker = OUT / 'evaluation' / f'{rid}.json'
    source_sha = sha256(ROOT / 'experiments/gaussian_memory_visibility_training_evaluate.py')
    if marker.exists():
        rec = json.loads(marker.read_text())
        assert rec['complete'] and rec['checkpoints'] == checkpoints and rec['source_sha256'] == source_sha
        assert sha256(rec['prediction_path']) == rec['prediction_sha256']
        return rec
    start = time.monotonic(); src = PARENT / 'inputs' / f'{rid}.json'
    assert sha256(src) == plan['reused_record_sha256'][str(src)]
    source = json.loads(src.read_text())
    for name, h in source['files'].items(): assert sha256(EVAL_CACHE / rid / name) == h
    samples = {c: torch.load(EVAL_CACHE / rid / f'q11_c{c}.pt', map_location='cpu', weights_only=False) for c in range(3)}
    states = device_tree(torch.load(EVAL_CACHE / rid / 'states.pt', map_location='cpu', weights_only=False))
    priors = {(key, kind): memory_prior(m, samples[0], kind, plan['center_variance'])
              for key, m in states.items() for kind in KINDS}
    rows = []; predictions = []
    for c, sample in samples.items():
        state_condition = 'persistent' if c == 2 else 'full'
        for policy in POLICIES:
            key = state_condition + '/' + policy
            for kind in KINDS:
                for seed in SEEDS:
                    pred = predict(backbone, adapters[kind, seed], sample, states[key]['features'], priors[key, kind])
                    r = score(pred, sample)
                    r.update(family=FAMILY[kind], kind=kind, seed=seed, policy=policy, place=meta['place'],
                             visibility_condition=CONDITIONS[c])
                    rows.append(r); predictions.append(pred.float().cpu())
    assert len(rows) == 54
    folder = CACHE / 'evaluation'; folder.mkdir(parents=True, exist_ok=True)
    target = folder / f'{rid}.pt'
    save_tensor(target, {'prediction_native': torch.stack(predictions),
        'keys': [[r['condition'], r['family'], r['seed'], r['policy']] for r in rows]})
    rec = {'complete': True, 'id': rid, 'place': meta['place'], 'rows': rows,
        'prediction_path': str(target), 'prediction_sha256': sha256(target),
        'plan_sha256': sha256(OUT / 'plan.json'), 'parent_input_record_sha256': sha256(src),
        'source_sha256': source_sha, 'checkpoints': checkpoints, 'seconds': time.monotonic() - start}
    write_json(marker, rec)
    print(json.dumps({'evaluated': rid, 'outputs': len(rows), 'seconds': rec['seconds']}), flush=True)
    return rec


def main():
    p = argparse.ArgumentParser(); p.add_argument('--shard', type=int, default=0); p.add_argument('--shards', type=int, default=1)
    args = p.parse_args(); torch.set_num_threads(4); plan = load_plan(); verify_sources(plan)
    backbone = load_backbone(); adapters, checks = load_adapters(plan); start = time.monotonic(); ids = []
    for meta in plan['catalog'][args.shard::args.shards]:
        ids.append(evaluate(meta, backbone, adapters, checks, plan)['id'])
    write_json(OUT / f'eval_worker_{args.shard}.json', {'complete': True, 'ids': ids,
        'seconds': time.monotonic() - start, 'peak_GiB': torch.cuda.max_memory_allocated() / 2**30})


if __name__ == '__main__': main()
