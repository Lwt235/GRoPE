"""Evaluate saved memory directly, including views and scenes not optimized."""
import argparse
import collections
import json

import numpy as np
import torch

from experiments.gaussian_memory_data import CACHE, REPORT, HOLD_QUERY
from experiments.gaussian_memory_setup import sha256
from models.gaussian_memory import GaussianWriter, seeds_to_device, detached_memory, read_memory


def altered(memory, condition, other):
    m = {k: v.clone() for k, v in memory.items()}
    rng = torch.Generator(device=m['means'].device).manual_seed(6401)
    if condition == 'removed':
        m['opacities'].zero_()
    elif condition == 'wrong_scene':
        m = other
    elif condition == 'shuffle_features':
        order = torch.randperm(len(m['means']), generator=rng, device=m['means'].device)
        m['features'] = m['features'][order]
    elif condition == 'jitter_centers':
        m['means'] += torch.randn(m['means'].shape, device=m['means'].device, generator=rng) * .15
    elif condition == 'uniform_scene_feature':
        m['features'] = m['features'].mean(0, keepdim=True).expand_as(m['features']).contiguous()
    return m


def evaluate():
    torch.set_num_threads(4)
    device = 'cuda'
    rows = json.loads((REPORT / 'data_manifest.json').read_text())['episodes']
    identities = json.loads((REPORT / 'cache_identity.json').read_text())
    raw = {}
    for row in rows:
        path = CACHE / (row['id'] + '.pt')
        assert sha256(path) == identities['cache_sha256'][str(path)]
        raw[row['id']] = torch.load(path, weights_only=False, map_location='cpu')
    variants = ['initial', 'learned_geometry', 'fixed_geometry']
    memories = {}
    for variant in variants:
        if variant == 'initial':
            model = GaussianWriter().to(device)
            ckpt = torch.load(REPORT / 'learned_geometry/initial.pt', weights_only=False, map_location=device)
            model.load_state_dict(ckpt['model'])
            model.eval()
            with torch.no_grad():
                memories[variant] = {r['id']: detached_memory(model(seeds_to_device(raw[r['id']], device)), device)
                                     for r in rows}
        else:
            memories[variant] = {}
            for row in rows:
                # Evaluations consume serialized states, not a fresh write from query data.
                saved = torch.load(REPORT / variant / 'memories' / (row['id'] + '.pt'),
                                   weights_only=False, map_location=device)
                memories[variant][row['id']] = saved['memory']
    output = REPORT / 'evaluation'
    output.mkdir(exist_ok=True)
    conditions = ['normal', 'removed', 'wrong_scene', 'shuffle_features', 'jitter_centers',
                  'uniform_scene_feature', 'first_reference_camera']
    scores, geometry = [], []
    figures = {}
    with torch.no_grad():
        for row in rows:
            rid = row['id']
            d = raw[rid]
            qids = sorted(row['training_query'] + row['heldout_query'])
            cams, K = d['c2w'][qids].to(device), d['K'][qids].to(device)
            np.savez(output / (rid + '_cameras.npz'), c2w=d['c2w'][qids].numpy(), K=d['K'][qids].numpy())
            target = d['teacher'][qids].to(device)
            figures[rid] = {'qids': qids, 'rgb': d['rgb'][qids], 'target': target.cpu()}
            other_row = next(r for r in rows if r['scene'] != row['scene'])
            for variant in variants:
                memory = memories[variant][rid]
                for condition in (conditions if variant == 'learned_geometry' else ['normal']):
                    m = altered(memory, condition, memories[variant][other_row['id']])
                    c = d['c2w'][0:1].to(device).expand_as(cams) if condition == 'first_reference_camera' else cams
                    prediction, alpha = read_memory(m, c, K)
                    mse = (prediction - target).square().mean(dim=(1, 2, 3))
                    cosine = torch.nn.functional.cosine_similarity(prediction, target, dim=-1).mean(dim=(1, 2))
                    if condition == 'normal' or (variant == 'learned_geometry' and condition in ['shuffle_features', 'wrong_scene']):
                        figures[rid][variant + '/' + condition] = prediction.cpu()
                    for i, q in enumerate(qids):
                        role = ('train_query' if q in row['training_query'] else
                                'heldout_view' if row['split'] == 'fit' else 'heldout_scene')
                        scores.append({'episode': rid, 'scene': row['scene'], 'frame': row['frames'][q]['index'],
                            'query_ordinal': q, 'role': role, 'variant': variant, 'condition': condition,
                            'mse': mse[i].item(), 'cosine': cosine[i].item(),
                            'alpha': alpha[i].mean().item(), 'zero_baseline_mse': target[i].square().mean().item()})
                if variant == 'learned_geometry':
                    base = memories['initial'][rid]
                    delta = (memory['means'] - base['means']).norm(dim=-1)
                    geometry.append({'episode': rid, 'scene': row['scene'],
                        'center_shift_median': delta.median().item(), 'center_shift_p90': delta.quantile(.9).item(),
                        'mean_covariance_trace_ratio': (memory['covars'].diagonal(dim1=-2, dim2=-1).sum(-1) /
                            base['covars'].diagonal(dim1=-2, dim2=-1).sum(-1)).mean().item(),
                        'min_covariance_eigenvalue': torch.linalg.eigvalsh(memory['covars']).min().item(),
                        'mean_opacity': memory['opacities'].mean().item(),
                        'memory_floats': sum(t.numel() for t in memory.values())})
            print(json.dumps({'evaluated_episode': rid, 'queries': len(qids)}), flush=True)
    grouped = collections.defaultdict(list)
    scene_grouped = collections.defaultdict(list)
    for x in scores:
        grouped[(x['role'], x['variant'], x['condition'])].append(x)
        scene_grouped[(x['scene'], x['role'], x['variant'], x['condition'])].append(x)
    summary = []
    for key, values in grouped.items():
        summary.append(dict(zip(['role', 'variant', 'condition'], key), queries=len(values),
            mse=float(np.mean([x['mse'] for x in values])), cosine=float(np.mean([x['cosine'] for x in values])),
            alpha=float(np.mean([x['alpha'] for x in values]))))
    per_scene = []
    for key, values in scene_grouped.items():
        per_scene.append(dict(zip(['scene', 'role', 'variant', 'condition'], key), queries=len(values),
                              mse=float(np.mean([x['mse'] for x in values]))))
    report = {'scope': 'development conditional RGB + known cameras; frozen VGGT; no sensor-depth input',
              'scores': scores, 'summary': summary, 'per_scene': per_scene, 'geometry': geometry,
              'data_identity_sha256': sha256(REPORT / 'cache_identity.json')}
    (output / 'scores.json').write_text(json.dumps(report, indent=2) + '\n')
    torch.save(figures, output / 'visual_data.pt')
    print(json.dumps({'summary': summary, 'evaluation_complete': True}), flush=True)


def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures = torch.load(REPORT / 'evaluation/visual_data.pt', weights_only=False, map_location='cpu')
    scores = json.loads((REPORT / 'evaluation/scores.json').read_text())
    rows = json.loads((REPORT / 'data_manifest.json').read_text())['episodes']
    exported = []
    keys = [('rgb', 'Query RGB (display only)'), ('target', 'Teacher feature target'),
            ('initial/normal', 'Untrained lift'), ('fixed_geometry/normal', 'Fixed-geometry writer'),
            ('learned_geometry/normal', 'Learned-geometry writer'),
            ('learned_geometry/shuffle_features', 'Features shuffled in space')]
    for row in rows:
        d = figures[row['id']]
        indices = [d['qids'].index(q) for q in HOLD_QUERY]
        fig, axes = plt.subplots(3, 6, figsize=(13, 5.8))
        for i, qi in enumerate(indices):
            for j, (key, title) in enumerate(keys):
                a = d[key][qi].numpy()
                if key != 'rgb': a = np.clip(a[..., :3] / 4 + .5, 0, 1)
                axes[i, j].imshow(a, interpolation='nearest')
                axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
                if i == 0: axes[i, j].set_title(title, fontsize=9)
                if j == 0: axes[i, j].set_ylabel(f"Frame {row['frames'][d['qids'][qi]]['index']}")
        fig.suptitle(row['scene'] + ' | same saved memory, 3 withheld views\n'
                     'First 3 PCA feature channels; fixed display scale; given cameras', fontsize=10)
        fig.tight_layout()
        path = REPORT / 'evaluation' / (row['id'] + '.png')
        fig.savefig(path, dpi=140); plt.close(fig)
        exported.append(str(path))
    (REPORT / 'evaluation/figures.json').write_text(json.dumps(exported, indent=2) + '\n')
    # Plot actual complete optimization histories, not selected loss snippets.
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for variant in ['learned_geometry', 'fixed_geometry']:
        a = [json.loads(x) for x in (REPORT / variant / 'training.jsonl').read_text().splitlines()]
        loss = np.array([x['loss'] for x in a])
        smooth = np.convolve(loss, np.ones(100) / 100, 'valid')
        ax.plot(np.arange(100, len(a) + 1), smooth, label=variant.replace('_', ' '))
    ax.set(xlabel='Optimizer step', ylabel='Training MSE (100-step average)')
    ax.legend(); fig.tight_layout()
    fig.savefig(REPORT / 'evaluation/training.png', dpi=150); plt.close(fig)
    # Show the actual saved geometric state, independent of query-image plots.
    from matplotlib.patches import Ellipse
    for row in rows:
        if row['start'] != 0:
            continue
        d = torch.load(CACHE / (row['id'] + '.pt'), weights_only=False, map_location='cpu')
        saved = torch.load(REPORT / 'learned_geometry/memories' / (row['id'] + '.pt'),
                           weights_only=False, map_location='cpu')['memory']
        initial = {'means': d['seed_mean'], 'covars': d['seed_chol'] @ d['seed_chol'].transpose(-1, -2),
                   'features': d['seed_teacher']}
        unit = d['diagnostics']['unit_meters']
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True, sharey=True)
        for ax, state, name in zip(axes, [initial, saved], ['Initial merged VGGT geometry', 'Saved learned Gaussian memory']):
            xy = state['means'][:, [0, 2]].numpy() * unit
            cov = state['covars'][:, [0, 2]][:, :, [0, 2]].numpy() * unit**2
            colors = np.clip(state['features'][:, :3].numpy() / 4 + .5, 0, 1)
            for point, covariance, color in zip(xy, cov, colors):
                values, vectors = np.linalg.eigh(covariance)
                angle = np.degrees(np.arctan2(vectors[1, 1], vectors[0, 1]))
                ax.add_patch(Ellipse(point, 2 * np.sqrt(values[1]), 2 * np.sqrt(values[0]),
                                     angle=angle, facecolor=color, edgecolor='none', alpha=.25))
            ax.scatter(xy[:, 0], xy[:, 1], c=colors, s=5)
            cameras = d['c2w'][:, [0, 2], 3].numpy() * unit
            ax.plot(cameras[:, 0], cameras[:, 1], 'k--', linewidth=.8)
            ax.scatter(cameras[row['reference'], 0], cameras[row['reference'], 1], marker='^', c='red', s=30, label='Reference cameras')
            ax.scatter(cameras[HOLD_QUERY, 0], cameras[HOLD_QUERY, 1], marker='x', c='blue', s=25, label='Withheld-view cameras')
            ax.set(title=name, xlabel='x (meters)', ylabel='z (meters)')
            ax.set_aspect('equal', adjustable='box')
        axes[0].legend(fontsize=8)
        fig.suptitle(row['scene'] + '\nXZ projection; ellipses are 1-sigma; colors are feature channels', fontsize=10)
        fig.tight_layout()
        path = REPORT / 'evaluation' / (row['id'] + '_memory.png')
        fig.savefig(path, dpi=140); plt.close(fig)
        exported.append(str(path))
    (REPORT / 'evaluation/figures.json').write_text(json.dumps(exported, indent=2) + '\n')
    print(json.dumps({'plots': len(exported) + 1}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['evaluate', 'plot'])
    a = p.parse_args()
    evaluate() if a.command == 'evaluate' else plot()
