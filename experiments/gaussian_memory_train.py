"""Shared writer optimization across scenes; queries are supervision only."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.gaussian_memory_data import CACHE, REPORT, TRAIN_QUERY
from experiments.gaussian_memory_setup import ROOT, sha256
from models.gaussian_memory import GaussianWriter, seeds_to_device, read_memory, detached_memory


def train(variant, steps=2000):
    torch.set_num_threads(4)
    torch.manual_seed(20261110)
    np.random.seed(20261110)
    device = 'cuda'
    manifest = json.loads((REPORT / 'data_manifest.json').read_text())
    identity = json.loads((REPORT / 'cache_identity.json').read_text())
    rows = [r for r in manifest['episodes'] if r['split'] == 'fit']
    data = []
    for row in rows:
        path = CACHE / (row['id'] + '.pt')
        assert sha256(path) == identity['cache_sha256'][str(path)]
        raw = torch.load(path, map_location='cpu', weights_only=False)
        data.append({'seeds': seeds_to_device(raw, device),
                     'teacher': raw['teacher'][TRAIN_QUERY].to(device),
                     'c2w': raw['c2w'][TRAIN_QUERY].to(device), 'K': raw['K'][TRAIN_QUERY].to(device)})
    out = REPORT / variant
    out.mkdir(exist_ok=True)
    if (out / 'training.jsonl').exists():
        raise RuntimeError('Existing attempt must be retained; choose an explicit new attempt before rerunning')
    model = GaussianWriter(learn_geometry=variant == 'learned_geometry').to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
    configuration = {'variant': variant, 'steps': steps, 'seed': 20261110, 'lr': 3e-4, 'weight_decay': .01,
        'queries_per_step': 2, 'training_episodes': [r['id'] for r in rows],
        'trainable_parameters': sum(p.numel() for p in model.parameters()),
        'data_identity_sha256': sha256(REPORT / 'cache_identity.json'),
        'source_hashes': {p: sha256(ROOT / p) for p in ['models/gaussian_memory.py',
                          'experiments/gaussian_memory_train.py', 'experiments/gaussian_memory_data.py']}}
    (out / 'config.json').write_text(json.dumps(configuration, indent=2) + '\n')
    torch.save({'model': model.state_dict(), 'config': configuration}, out / 'initial.pt')
    rng = np.random.default_rng(20261110)
    started = time.monotonic()
    recent = []
    log = open(out / 'training.jsonl', 'w', buffering=1)
    for step in range(1, steps + 1):
        episode = int(rng.integers(len(data)))
        d = data[episode]
        queries = rng.choice(len(TRAIN_QUERY), 2, replace=False).tolist()
        model.train()
        memory = model(d['seeds'])
        predicted, alpha = read_memory(memory, d['c2w'][queries], d['K'][queries])
        loss = (predicted - d['teacher'][queries]).square().mean()
        if not torch.isfinite(loss):
            raise RuntimeError(f'nonfinite loss at {step}')
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gradnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        if not torch.isfinite(gradnorm):
            raise RuntimeError(f'nonfinite gradient at {step}')
        opt.step()
        entry = {'step': step, 'episode': rows[episode]['id'], 'loss': loss.item(),
                 'grad_norm': gradnorm.item(), 'coverage_alpha': alpha.mean().item(),
                 'seconds': time.monotonic() - started}
        log.write(json.dumps(entry) + '\n')
        recent.append(entry['loss'])
        if step == 1 or step % 100 == 0:
            print(json.dumps({'variant': variant, **entry, 'recent_mean_loss': float(np.mean(recent[-100:])),
                              'memory_GiB': torch.cuda.max_memory_allocated() / 1024**3}), flush=True)
        if step % 500 == 0 or step == steps:
            torch.save({'model': model.state_dict(), 'config': configuration, 'step': step,
                        'elapsed_seconds': time.monotonic() - started}, out / f'step_{step:04d}.pt')
    log.close()
    model.eval()
    # Separate serialized memory for every scene, including scenes never optimized.
    saved = out / 'memories'
    saved.mkdir(exist_ok=True)
    with torch.no_grad():
        for row in manifest['episodes']:
            raw = torch.load(CACHE / (row['id'] + '.pt'), weights_only=False, map_location='cpu')
            mem = detached_memory(model(seeds_to_device(raw, device)))
            torch.save({'memory': mem, 'coordinate_unit_meters': raw['diagnostics']['unit_meters'],
                        'origin': 'first reference camera', 'reference_frames':
                        [row['frames'][i]['index'] for i in row['reference']],
                        'scene': row['scene'], 'episode': row['id'],
                        'checkpoint_sha256': sha256(out / f'step_{steps:04d}.pt')}, saved / (row['id'] + '.pt'))
    status = {'complete': True, 'steps': steps, 'elapsed_seconds': time.monotonic() - started,
              'memory_GiB': torch.cuda.max_memory_allocated() / 1024**3,
              'saved_memories': len(manifest['episodes']), 'evaluation_pending': True}
    (out / 'status.json').write_text(json.dumps(status, indent=2) + '\n')
    print(json.dumps(status), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--variant', choices=['learned_geometry', 'fixed_geometry'], required=True)
    p.add_argument('--steps', type=int, default=2000)
    a = p.parse_args()
    train(a.variant, a.steps)
