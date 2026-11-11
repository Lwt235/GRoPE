"""Three paired fits after a concrete projection-interface correction."""
import argparse
import inspect
import json
import time

import numpy as np
import torch

from experiments.gaussian_memory_attention_train import (OUT, SEEDS, VARIANTS, load_inputs,
                                                        assert_identity, write_json)
from experiments.gaussian_memory_attention_evaluate import aggregate
from experiments.gaussian_memory_data import TRAIN_QUERY
from experiments.gaussian_memory_setup import ROOT, sha256
from models.gaussian_memory_attention import GaussianAttentionReader, geometry_inputs
from models.gaussian_memory_projection import projected_inputs

DEST = OUT / 'projection_matched'


def prepare():
    from gsplat.cuda._torch_impl import _persp_proj
    identity = assert_identity()
    DEST.mkdir(exist_ok=True)
    assert not (DEST / 'plan.json').exists()
    sources = [__file__, ROOT / 'models/gaussian_memory_projection.py', inspect.getsourcefile(_persp_proj),
               ROOT / 'docs/gaussian_memory_projection_alignment.md']
    write_json(DEST / 'plan.json', {'source_sha256': {str(p): sha256(p) for p in sources},
               'original_identity_sha256': sha256(OUT / 'identity.json'),
               'center_variance': identity['center_variance'], 'seeds': SEEDS, 'steps': 2000,
               'fixed_controls': {str(OUT / f'{v}_{s}/step_2000.pt'): sha256(OUT / f'{v}_{s}/step_2000.pt')
                                  for v in ('mlp', 'center') for s in SEEDS},
               'only_change': 'Explicit Gaussian position term uses the writer renderer gsplat projection at train and read time'})


def checked():
    assert_identity()
    plan = json.loads((DEST / 'plan.json').read_text())
    for k in ('source_sha256', 'fixed_controls'):
        for path, expected in plan[k].items():
            assert sha256(path) == expected, path
    return plan


def train():
    torch.set_num_threads(4)
    plan = checked()
    data = []
    for item in load_inputs('cuda', fit_only=True):
        raw = item['raw']
        inputs = projected_inputs(item['memory'], raw['c2w'][TRAIN_QUERY].cuda(), raw['K'][TRAIN_QUERY].cuda())
        data.append({'id': item['row']['id'], 'inputs': inputs, 'target': raw['teacher'][TRAIN_QUERY].cuda().flatten(1, 2)})
    for seed in SEEDS:
        dest = DEST / f'gaussian_{seed}'
        assert not dest.exists()
        dest.mkdir()
        initial = torch.load(OUT / f'gaussian_{seed}/initial.pt', weights_only=False, map_location='cuda')
        model = GaussianAttentionReader('gaussian', plan['center_variance']).cuda()
        model.load_state_dict(initial['model'])
        config = dict(initial['config'], projection='gsplat',
                      adaptation_plan_sha256=sha256(DEST / 'plan.json'))
        write_json(dest / 'config.json', config)
        torch.save({'model': model.state_dict(), 'config': config}, dest / 'initial.pt')
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
        rng = np.random.default_rng(seed)
        started, recent = time.monotonic(), []
        with open(dest / 'training.jsonl', 'w', buffering=1) as stream:
            for step in range(1, 2001):
                d = data[int(rng.integers(len(data)))]
                queries = rng.choice(6, 2, replace=False).tolist()
                prediction = model({k: v[queries] for k, v in d['inputs'].items()})
                loss = (prediction - d['target'][queries]).square().mean()
                opt.zero_grad(set_to_none=True); loss.backward()
                grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                assert torch.isfinite(loss) and torch.isfinite(grad)
                opt.step()
                row = {'step': step, 'episode': d['id'], 'queries': queries, 'loss': loss.item(),
                       'gradient_norm': grad.item(), 'seconds': time.monotonic() - started}
                stream.write(json.dumps(row) + '\n'); recent.append(row['loss'])
                if step == 1 or step % 500 == 0:
                    print(json.dumps({'run': dest.name, **row, 'last100_loss': float(np.mean(recent[-100:]))}), flush=True)
                if step % 500 == 0:
                    torch.save({'model': model.state_dict(), 'config': config, 'step': step}, dest / f'step_{step:04d}.pt')
        write_json(dest / 'status.json', {'complete': True, 'steps': 2000, 'elapsed_seconds': time.monotonic() - started,
                   'peak_allocated_GiB': torch.cuda.max_memory_allocated() / 1024**3,
                   'position_scales': torch.nn.functional.softplus(model.position_scale).detach().cpu().tolist()})
    checked()


def evaluate():
    torch.set_num_threads(4)
    checked()
    assert not (DEST / 'scores.json').exists()
    models = {}
    for seed in SEEDS:
        ckpt = torch.load(DEST / f'gaussian_{seed}/step_2000.pt', weights_only=False, map_location='cuda')
        model = GaussianAttentionReader('gaussian', ckpt['config']['center_variance']).cuda().eval()
        model.load_state_dict(ckpt['model']); models[seed] = model
    scores, figures, unchanged = [], {}, 0.
    for item in load_inputs('cuda'):
        row, raw, memory = item['row'], item['raw'], item['memory']
        qids = sorted(row['training_query'] + row['heldout_query'])
        c2w, K = raw['c2w'][qids].cuda(), raw['K'][qids].cuda()
        with torch.no_grad():
            data = projected_inputs(memory, c2w, K)
            original = geometry_inputs(memory, c2w, K)
            for k in ('query', 'key', 'value', 'distance2', 'log_weight', 'valid'):
                assert torch.equal(data[k], original[k]), k
            target = raw['teacher'][qids].cuda().flatten(1, 2)
            for seed, model in models.items():
                for condition in ('normal', 'isotropic'):
                    prediction, weights = model(data, condition, return_attention=True)
                    mse = (prediction - target).square().mean((1, 2))
                    cos = torch.nn.functional.cosine_similarity(prediction, target, dim=-1).mean(1)
                    ratio = prediction.norm(dim=-1).mean(1) / target.norm(dim=-1).mean(1).clamp_min(1e-8)
                    if seed == SEEDS[0]:
                        figures.setdefault(row['id'], {})[condition] = prediction.cpu().reshape(-1, 15, 20, 32)
                    for i, q in enumerate(qids):
                        role = ('train_query' if q in row['training_query'] else
                                'heldout_view' if row['split'] == 'fit' else 'heldout_scene')
                        scores.append({'episode': row['id'], 'scene': row['scene'], 'query_ordinal': q,
                            'frame': row['frames'][q]['index'], 'role': role, 'variant': 'gaussian', 'seed': seed,
                            'condition': condition, 'mse': mse[i].item(), 'cosine': cos[i].item(),
                            'norm_ratio': ratio[i].item(), 'background_mass': weights[i, ..., -1].mean().item()})
    fields = ['role', 'variant', 'condition', 'seed']
    result = {'scores': scores, 'summary': aggregate(scores, fields), 'per_scene': aggregate(scores, ['scene', *fields]),
              'non_gaussian_model_inputs_exactly_unchanged': True, 'plan_sha256': sha256(DEST / 'plan.json')}
    write_json(DEST / 'scores.json', result)
    torch.save(figures, DEST / 'visual_data.pt')
    checked()
    print(json.dumps({'matched_evaluation_complete': True, 'summary': result['summary']}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['prepare', 'train', 'evaluate'])
    a = p.parse_args()
    {'prepare': prepare, 'train': train, 'evaluate': evaluate}[a.command]()
