"""Evaluate every prescribed query; no checkpoint or scene selection."""
import argparse
import collections
import json

import numpy as np
import torch
from torch.nn import functional as F

from experiments.gaussian_memory_data import REPORT, HOLD_QUERY
from experiments.gaussian_memory_attention_train import (OUT, VARIANTS, SEEDS, load_inputs,
                                                        assert_identity, write_json)
from experiments.gaussian_memory_setup import sha256
from models.gaussian_memory_attention import GaussianAttentionReader, geometry_inputs


def aggregate(scores, fields):
    groups = collections.defaultdict(list)
    for row in scores:
        groups[tuple(row[k] for k in fields)].append(row)
    return [dict(zip(fields, key), queries=len(rows),
                 **{metric: float(np.mean([r[metric] for r in rows]))
                    for metric in ('mse', 'cosine', 'background_mass', 'norm_ratio')})
            for key, rows in groups.items()]


def evaluate():
    torch.set_num_threads(4)
    assert_identity()
    dest = OUT / 'evaluation'
    dest.mkdir(exist_ok=True)
    if (dest / 'scores.json').exists():
        raise RuntimeError('Evaluation already exists; keep its original results')
    models = {}
    for seed in SEEDS:
        for variant in VARIANTS:
            path = OUT / f'{variant}_{seed}/step_2000.pt'
            ckpt = torch.load(path, weights_only=False, map_location='cuda')
            assert ckpt['step'] == 2000
            model = GaussianAttentionReader(variant, ckpt['config']['center_variance']).cuda().eval()
            model.load_state_dict(ckpt['model'])
            models[(variant, seed)] = model
    source_figures = torch.load(REPORT / 'evaluation/visual_data.pt', weights_only=False, map_location='cpu')
    scores, figures = [], {}
    with torch.no_grad():
        for item in load_inputs('cuda'):
            row, raw, memory = item['row'], item['raw'], item['memory']
            rid = row['id']
            qids = sorted(row['training_query'] + row['heldout_query'])
            c2w, K = raw['c2w'][qids].cuda(), raw['K'][qids].cuda()
            target = raw['teacher'][qids].cuda().flatten(1, 2)
            inputs = geometry_inputs(memory, c2w, K)
            assert all(torch.isfinite(v).all() for k, v in inputs.items() if k != 'log_weight')
            figures[rid] = {'qids': qids, 'scene': row['scene'], 'rgb': raw['rgb'][qids],
                            'target': target.cpu().reshape(-1, 15, 20, 32),
                            'renderer': source_figures[rid]['learned_geometry/normal']}
            rng = torch.Generator(device='cuda').manual_seed(6401)
            permutation = torch.randperm(256, generator=rng, device='cuda')
            shuffled = {k: v.clone() for k, v in memory.items()}
            shuffled['features'] = shuffled['features'][permutation]
            shuffled_inputs = geometry_inputs(shuffled, c2w, K)
            for (variant, seed), model in models.items():
                conditions = ['normal', 'isotropic', 'no_position', 'shuffle_features'] if variant == 'gaussian' else ['normal']
                for condition in conditions:
                    data = shuffled_inputs if condition == 'shuffle_features' else inputs
                    prediction, attention = model(data, condition, return_attention=True)
                    mse = (prediction - target).square().mean((1, 2))
                    cosine = F.cosine_similarity(prediction, target, dim=-1).mean(1)
                    norm = prediction.norm(dim=-1).mean(1) / target.norm(dim=-1).mean(1).clamp_min(1e-8)
                    background = attention[..., -1].mean((1, 2))
                    if seed == SEEDS[0]:
                        figures[rid][variant + '/' + condition] = prediction.cpu().reshape(-1, 15, 20, 32)
                    for j, q in enumerate(qids):
                        role = ('train_query' if q in row['training_query'] else
                                'heldout_view' if row['split'] == 'fit' else 'heldout_scene')
                        scores.append({'episode': rid, 'scene': row['scene'], 'query_ordinal': q,
                                       'frame': row['frames'][q]['index'], 'role': role,
                                       'variant': variant, 'seed': seed, 'condition': condition,
                                       'mse': mse[j].item(), 'cosine': cosine[j].item(),
                                       'background_mass': background[j].item(), 'norm_ratio': norm[j].item()})
            print(json.dumps({'evaluated': rid, 'queries': len(qids)}), flush=True)
    fields = ['role', 'variant', 'condition', 'seed']
    result = {'identity_sha256': sha256(OUT / 'identity.json'), 'scores': scores,
              'summary': aggregate(scores, fields),
              'per_scene': aggregate(scores, ['scene', *fields]),
              'checkpoint_sha256': {f'{v}_{s}': sha256(OUT / f'{v}_{s}/step_2000.pt') for v, s in models}}
    write_json(dest / 'scores.json', result)
    torch.save(figures, dest / 'visual_data.pt')
    assert_identity()
    print(json.dumps({'evaluation_complete': True, 'score_rows': len(scores),
                      'normal_summary': [r for r in result['summary'] if r['condition'] == 'normal']}), flush=True)


def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dest = OUT / 'evaluation'
    data = torch.load(dest / 'visual_data.pt', weights_only=False, map_location='cpu')
    exported = []
    keys = [('rgb', 'Query RGB (display only)'), ('target', 'Teacher feature'), ('renderer', 'Existing renderer'),
            ('mlp/normal', 'Attention + MLP'), ('center/normal', '+ Center distance'),
            ('gaussian/normal', '+ Gaussian footprint'), ('gaussian/isotropic', 'Circular bias intervention')]
    for rid, item in data.items():
        fig, axes = plt.subplots(3, len(keys), figsize=(15, 5.7))
        for i, query in enumerate(HOLD_QUERY):
            index = item['qids'].index(query)
            for j, (key, label) in enumerate(keys):
                a = item[key][index].numpy()
                if key != 'rgb':
                    a = np.clip(a[..., :3] / 4 + .5, 0, 1)
                axes[i, j].imshow(a, interpolation='nearest')
                axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
                if i == 0:
                    axes[i, j].set_title(label, fontsize=8)
                if j == 0:
                    axes[i, j].set_ylabel(f'Query ordinal {query}')
        fig.suptitle(item['scene'] + ' | same saved memory, different query cameras\n'
                     'Seed 20261110 fixed before training; first 3 PCA channels, common display scale', fontsize=10)
        fig.tight_layout()
        path = dest / f'{rid}.png'
        fig.savefig(path, dpi=140); plt.close(fig)
        exported.append(str(path))
    fig, ax = plt.subplots(figsize=(8, 4))
    for variant in VARIANTS:
        curves = []
        for seed in SEEDS:
            lines = (OUT / f'{variant}_{seed}/training.jsonl').read_text().splitlines()
            values = np.array([json.loads(x)['loss'] for x in lines])
            curves.append(np.convolve(values, np.ones(100) / 100, mode='valid'))
        curves = np.stack(curves)
        x = np.arange(100, 2001)
        line, = ax.plot(x, curves.mean(0), label=variant)
        ax.fill_between(x, curves.min(0), curves.max(0), color=line.get_color(), alpha=.15)
    ax.set(xlabel='Optimizer step', ylabel='Training MSE, 100-step average',
           title='Three paired seeds: mean and observed seed range')
    ax.legend(); fig.tight_layout()
    path = dest / 'training.png'
    fig.savefig(path, dpi=150); plt.close(fig)
    exported.append(str(path))
    write_json(dest / 'figures.json', exported)
    print(json.dumps({'figures': len(exported)}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['evaluate', 'plot'])
    args = p.parse_args()
    evaluate() if args.command == 'evaluate' else plot()
