"""Fixed-memory, paired reader training; evaluation never selects checkpoints."""
import argparse
import json
import time

import numpy as np
import torch

from experiments.gaussian_memory_data import CACHE, REPORT, TRAIN_QUERY
from experiments.gaussian_memory_setup import ROOT, sha256
from models.gaussian_memory_attention import GaussianAttentionReader, geometry_inputs

OUT = ROOT / 'results/gaussian_memory/attention1'
VARIANTS = ('mlp', 'center', 'gaussian')
SEEDS = (20261110, 20261111, 20261112)
SOURCES = ('models/gaussian_memory_attention.py', 'experiments/gaussian_memory_attention_train.py',
           'experiments/gaussian_memory_attention_evaluate.py', 'experiments/gaussian_memory_attention_read.py')


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def load_inputs(device='cpu', fit_only=False):
    manifest = json.loads((REPORT / 'data_manifest.json').read_text())
    identity = json.loads((REPORT / 'cache_identity.json').read_text())
    result = []
    for row in manifest['episodes']:
        if fit_only and row['split'] != 'fit':
            continue
        path = CACHE / (row['id'] + '.pt')
        assert sha256(path) == identity['cache_sha256'][str(path)]
        raw = torch.load(path, map_location='cpu', weights_only=False)
        saved_path = REPORT / 'learned_geometry/memories' / (row['id'] + '.pt')
        saved = torch.load(saved_path, map_location=device, weights_only=False)
        assert saved['checkpoint_sha256'] == sha256(REPORT / 'learned_geometry/step_2000.pt')
        result.append({'row': row, 'raw': raw, 'memory': saved['memory'], 'path': saved_path})
    return result


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'identity.json').exists():
        raise RuntimeError('Preparation already exists; preserve the recorded experiment')
    torch.set_num_threads(4)
    values, state_hashes = [], {}
    for item in load_inputs():
        state_hashes[str(item['path'])] = sha256(item['path'])
        if item['row']['split'] == 'fit':
            raw = item['raw']
            g = geometry_inputs(item['memory'], raw['c2w'][TRAIN_QUERY], raw['K'][TRAIN_QUERY])
            values.append(g['projected_variance'][g['valid']])
    variance = torch.cat(values).median().item()
    identity = {'scope': 'previously used development scenes; frozen writer and memory',
                'center_variance': variance, 'scale_source': 'training query geometry only, no labels',
                'writer_checkpoint_sha256': sha256(REPORT / 'learned_geometry/step_2000.pt'),
                'cache_identity_sha256': sha256(REPORT / 'cache_identity.json'),
                'manifest_sha256': sha256(REPORT / 'data_manifest.json'),
                'memory_sha256': state_hashes,
                'source_sha256': {p: sha256(ROOT / p) for p in SOURCES},
                'plan_sha256': sha256(ROOT / 'docs/gaussian_memory_attention_plan.md'),
                'variants': VARIANTS, 'seeds': SEEDS, 'steps': 2000}
    write_json(OUT / 'identity.json', identity)
    print(json.dumps({'prepared': True, 'center_variance': variance, 'memories': len(state_hashes)}), flush=True)


def assert_identity():
    identity = json.loads((OUT / 'identity.json').read_text())
    assert sha256(REPORT / 'cache_identity.json') == identity['cache_identity_sha256']
    assert sha256(REPORT / 'data_manifest.json') == identity['manifest_sha256']
    for path, value in identity['memory_sha256'].items():
        assert sha256(path) == value, path
    for path, value in identity['source_sha256'].items():
        assert sha256(ROOT / path) == value, path
    return identity


def train_all():
    torch.set_num_threads(4)
    identity = assert_identity()
    assert torch.cuda.is_available()
    data = []
    for item in load_inputs('cuda', fit_only=True):
        raw = item['raw']
        g = geometry_inputs(item['memory'], raw['c2w'][TRAIN_QUERY].cuda(), raw['K'][TRAIN_QUERY].cuda())
        assert all(torch.isfinite(v).all() for k, v in g.items() if k != 'log_weight')
        data.append({'id': item['row']['id'], 'inputs': g,
                     'target': raw['teacher'][TRAIN_QUERY].cuda().flatten(1, 2)})
    for seed in SEEDS:
        for variant in VARIANTS:
            dest = OUT / f'{variant}_{seed}'
            if (dest / 'status.json').exists():
                assert json.loads((dest / 'status.json').read_text())['complete']
                print(json.dumps({'already_complete': dest.name}), flush=True)
                continue
            if (dest / 'training.jsonl').exists():
                raise RuntimeError(f'Incomplete training retained at {dest}; inspect before a new attempt')
            dest.mkdir(exist_ok=True)
            torch.manual_seed(seed)
            model = GaussianAttentionReader(variant, identity['center_variance']).cuda()
            opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4, weight_decay=.01)
            config = {'variant': variant, 'seed': seed, 'steps': 2000, 'width': 64, 'heads': 4,
                      'center_variance': identity['center_variance'], 'lr': 3e-4, 'weight_decay': .01,
                      'queries_per_step': 2, 'training_episodes': [d['id'] for d in data],
                      'parameters': sum(p.numel() for p in model.parameters()),
                      'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
                      'identity_sha256': sha256(OUT / 'identity.json'), 'device': torch.cuda.get_device_name(0)}
            write_json(dest / 'config.json', config)
            torch.save({'model': model.state_dict(), 'config': config}, dest / 'initial.pt')
            rng = np.random.default_rng(seed)
            started = time.monotonic()
            recent = []
            with open(dest / 'training.jsonl', 'w', buffering=1) as log:
                for step in range(1, 2001):
                    episode = int(rng.integers(len(data)))
                    d = data[episode]
                    queries = rng.choice(6, 2, replace=False).tolist()
                    inputs = {k: v[queries] for k, v in d['inputs'].items()}
                    prediction = model(inputs)
                    loss = (prediction - d['target'][queries]).square().mean()
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                    if not torch.isfinite(loss) or not torch.isfinite(norm):
                        raise RuntimeError(f'Nonfinite optimization at {dest.name} step {step}')
                    opt.step()
                    row = {'step': step, 'episode': d['id'], 'queries': queries, 'loss': loss.item(),
                           'gradient_norm': norm.item(), 'seconds': time.monotonic() - started}
                    log.write(json.dumps(row) + '\n')
                    recent.append(row['loss'])
                    if step == 1 or step % 250 == 0:
                        print(json.dumps({'run': dest.name, **row, 'last100_loss': float(np.mean(recent[-100:]))}), flush=True)
                    if step % 500 == 0:
                        torch.save({'model': model.state_dict(), 'config': config, 'step': step}, dest / f'step_{step:04d}.pt')
            torch.cuda.synchronize()
            status = {'complete': True, 'steps': 2000, 'elapsed_seconds': time.monotonic() - started,
                      'peak_allocated_GiB': torch.cuda.max_memory_allocated() / 1024**3,
                      'position_scales': torch.nn.functional.softplus(model.position_scale).detach().cpu().tolist(),
                      'evaluation_pending': True}
            write_json(dest / 'status.json', status)
            print(json.dumps({'finished': dest.name, **status}), flush=True)
            del opt, model
    write_json(OUT / 'training_status.json', {'complete': True, 'runs': 9, 'evaluation_pending': True})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'train'])
    args = parser.parse_args()
    prepare() if args.command == 'prepare' else train_all()
