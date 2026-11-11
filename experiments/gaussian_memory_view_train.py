"""Matched fixed-budget learning of a residual layer on frozen VGGT tokens."""
import argparse
import json
import time

import numpy as np
import torch

from experiments.gaussian_memory_attention_train import load_inputs, write_json
from experiments.gaussian_memory_setup import sha256
from experiments.gaussian_memory_view_data import OUT, TOKEN_CACHE, SEEDS, VARIANTS, checked
from models.gaussian_memory_projection import projected_inputs
from models.gaussian_memory_view_token import ViewTokenAdapter, normalized_error


def load_training():
    identity = json.loads((OUT / 'cache_identity.json').read_text())
    data, coverage = [], []
    for item in load_inputs('cuda', fit_only=True):
        path = TOKEN_CACHE / (item['row']['id'] + '.pt')
        assert sha256(path) == identity['sha256'][str(path)]
        rec = torch.load(path, weights_only=False)
        keep = torch.tensor([r == 'train' for r in rec['roles']])
        query = rec['query'][keep]
        g = projected_inputs(item['memory'], item['raw']['c2w'].cuda(), item['raw']['K'].cuda())
        support = (g['log_weight'][:, None] - .5 * g['distance2'] / .7146816253662109).exp().amax(-1)
        coverage.append(support[query].flatten())
        data.append({'id': item['row']['id'], 'x': rec['input_tokens'][keep].cuda(),
                     'y': rec['teacher_tokens'][query].cuda(), 'mask': rec['mask'][keep].cuda(),
                     'geometry': {k: v[query] for k, v in g.items()}})
    return data, float(torch.cat(coverage).median())


def train():
    plan = checked(); torch.set_num_threads(4)
    data, support_threshold = load_training()
    write_json(OUT / 'support_threshold.json', {'threshold': support_threshold,
               'source': 'median projected opacity-weighted center support, training inputs only',
               'not_true_visibility': True})
    for seed in SEEDS:
        for variant in VARIANTS:
            dest = OUT / f'{variant}_{seed}'
            assert not dest.exists(), 'preserve prior attempts'
            dest.mkdir()
            torch.manual_seed(seed)
            model = ViewTokenAdapter(variant, plan['center_variance']).cuda()
            opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
            config = {'variant': variant, 'seed': seed, 'steps': 2000, 'batch': 2, 'lr': 3e-4,
                      'weight_decay': .01, 'hidden_visible_weight': [.8, .2], 'clip': 5.,
                      'center_variance': plan['center_variance'], 'width': model.width,
                      'parameters': sum(p.numel() for p in model.parameters()),
                      'plan_sha256': sha256(OUT / 'plan.json'), 'cache_sha256': sha256(OUT / 'cache_identity.json')}
            write_json(dest / 'config.json', config)
            torch.save({'config': config, 'model': model.state_dict()}, dest / 'initial.pt')
            rng, started, recent = np.random.default_rng(seed), time.monotonic(), []
            torch.cuda.reset_peak_memory_stats()
            with (dest / 'training.jsonl').open('w', buffering=1) as log:
                for step in range(1, 2001):
                    episode = int(rng.integers(len(data))); d = data[episode]
                    ids = rng.choice(len(d['x']), 2, replace=False).tolist()
                    geometry = {k: v[ids] for k, v in d['geometry'].items()}
                    pred = model(d['x'][ids], geometry)
                    error = normalized_error(pred, d['y'][ids]); mask = d['mask'][ids]
                    hidden, visible = error[mask].mean(), error[~mask].mean()
                    loss = .8 * hidden + .2 * visible
                    opt.zero_grad(set_to_none=True); loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                    assert torch.isfinite(loss) and torch.isfinite(norm), (variant, seed, step)
                    opt.step()
                    row = {'step': step, 'episode': d['id'], 'sample_ids': ids, 'loss': loss.item(),
                           'hidden': hidden.item(), 'visible': visible.item(), 'gradient_norm': norm.item(),
                           'seconds': time.monotonic() - started}
                    log.write(json.dumps(row) + '\n'); recent.append(loss.item())
                    if step == 1 or step % 250 == 0:
                        print(json.dumps({'run': dest.name, **row, 'last100': float(np.mean(recent[-100:]))}), flush=True)
                    if step % 500 == 0:
                        torch.save({'model': model.state_dict(), 'config': config, 'step': step}, dest / f'step_{step:04d}.pt')
            torch.cuda.synchronize()
            write_json(dest / 'status.json', {'complete': True, 'steps': 2000,
                       'seconds': time.monotonic() - started,
                       'peak_GiB': torch.cuda.max_memory_allocated() / 1024**3})
            print(json.dumps({'finished': dest.name, 'seconds': time.monotonic() - started}), flush=True)
            del model, opt
    write_json(OUT / 'training_status.json', {'complete': True, 'runs': 9})


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('command', choices=['train'])
    p.parse_args(); train()
