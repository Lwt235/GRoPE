"""Paired scene statistics and loss-control visualizations, with all groups."""
import json

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from experiments.gaussian_memory_attention_train import write_json
from experiments.gaussian_memory_fidelity_data import OUT, REGIMES, VARIANTS, SEEDS, model_path

METRICS = ('hidden_mse', 'visible_mse', 'all_patch_mse', 'clean_input_mse')
LABELS = ('Hidden', 'Visible', 'All patches', 'Clean input')
GROUPS = [r + '_' + v for r in REGIMES for v in VARIANTS]


def group(row):
    return 'backbone' if row['variant'] == 'backbone' else row['regime'] + '_' + row['variant']


def paired(rows, left, right, metric):
    scenes = sorted({r['scene'] for r in rows})
    values = []
    for scene in scenes:
        means = {g: float(np.mean([r[metric] for r in rows if r['scene'] == scene and group(r) == g]))
                 for g in (left, right)}
        values.append({'scene': scene, 'left': means[left], 'right': means[right],
                       'gain': means[left] - means[right]})
    delta = np.array([v['gain'] for v in values])
    rng = np.random.default_rng(44102)
    boot = delta[rng.integers(len(delta), size=(10000, len(delta)))].mean(-1)
    seed_gain = []
    if left != 'backbone':
        for seed in SEEDS:
            seed_gain.append(float(np.mean([r[metric] for r in rows if group(r) == left and r['seed'] == seed]) -
                                   np.mean([r[metric] for r in rows if group(r) == right and r['seed'] == seed])))
    return {'mean_gain': float(delta.mean()),
            'relative_reduction_percent': float(100 * delta.mean() / np.mean([v['left'] for v in values])),
            'scene_bootstrap_95': np.quantile(boot, [.025, .975]).tolist(),
            'scenes_better': int((delta > 0).sum()), 'scenes_total': len(scenes),
            'paired_seed_gain': seed_gain, 'scenes': values}


def figures(directory):
    paths = []
    for source in sorted(directory.glob('*_maps.pt')):
        d = torch.load(source, weights_only=False, map_location='cpu')
        q, mask = int(d['query'][0]), d['mask'][0].reshape(15, 20).numpy()
        rgb = d['rgb'].numpy(); observed = rgb[q].copy()
        observed[mask.repeat(14, 0).repeat(14, 1)] = np.array([.485, .456, .406]) * 255
        fig, axes = plt.subplots(3, 3, figsize=(11, 9), constrained_layout=True)
        for ax, pic, name in zip(axes[0], [rgb[0], rgb[q], observed],
                                 ['First of four references', 'Full query: target only in masked branch', 'Observed query']):
            ax.imshow(pic); ax.set_title(name, fontsize=10)
        errors = [d['errors'][g][0].reshape(15, 20).numpy() for g in GROUPS]
        maximum = float(np.quantile(np.stack(errors), .98))
        for r, regime in enumerate(REGIMES):
            for v, variant in enumerate(VARIANTS):
                error = d['errors'][regime + '_' + variant][0].reshape(15, 20).numpy()
                ax = axes[v + 1, r]
                im = ax.imshow(error, vmin=0, vmax=maximum, cmap='magma')
                ax.contour(mask.astype(float), [.5], colors='cyan', linewidths=.7)
                ax.set_title(f'{regime} {variant}: hidden {error[mask].mean():.4f}\nvisible {error[~mask].mean():.4f}; all {error.mean():.4f}', fontsize=10)
        fig.colorbar(im, ax=list(axes[1:].flat), shrink=.7, label='Normalized token MSE (common scale)')
        for ax in axes.flat: ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(d['episode']['scene'] + ' | ' + d['episode']['id'] + ' | first query/mask; seed 20261110', fontsize=11)
        path = source.with_name(d['episode']['id'] + '.png')
        fig.savefig(path, dpi=120); plt.close(fig); paths.append(str(path))
    return paths


def main():
    analysis, paths = {}, []
    comparisons = [(r + '_local', r + '_center') for r in REGIMES]
    comparisons += [(a + '_' + v, b + '_' + v) for v in VARIANTS for a,b in [('A','B'), ('B','C'), ('A','C')]]
    for name in ('development_evaluation', 'transfer_evaluation'):
        directory = OUT / name
        if not (directory / 'scores.json').exists(): continue
        result = json.loads((directory / 'scores.json').read_text())
        for split in sorted({r['split'] for r in result['rows']}):
            normal = [r for r in result['rows'] if r['split'] == split and r['condition'] == 'normal']
            clean = [r for r in result['preservation'] if r['split'] == split]
            stats = {g: {m: float(np.mean([r[m] for r in normal if group(r) == g])) for m in METRICS[:3]}
                     for g in GROUPS + ['backbone']}
            for g in GROUPS: stats[g]['clean_input_mse'] = float(np.mean([r['clean_input_mse'] for r in clean if group(r) == g]))
            stats['backbone']['clean_input_mse'] = 0.
            paired_stats = {}
            for m in METRICS:
                data = clean if m == 'clean_input_mse' else normal
                pairs = comparisons + ([] if m == 'clean_input_mse' else [('backbone', r + '_center') for r in REGIMES])
                paired_stats[m] = {left + '_minus_' + right: paired(data, left, right, m) for left,right in pairs}
            interventions = {}
            for regime in REGIMES:
                interventions[regime] = {}
                for condition in ('normal', 'removed', 'shuffled', 'swapped', 'wrong_camera'):
                    rows = [r for r in result['rows'] if r['split'] == split and r['regime'] == regime and
                            r['variant'] == 'center' and r['condition'] == condition]
                    interventions[regime][condition] = {m: float(np.mean([r[m] for r in rows])) for m in METRICS[:3]}
            analysis[split] = {'summary': stats, 'comparisons': paired_stats, 'interventions': interventions}
        paths += figures(directory)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
    colors = ('#7d8391', '#007d79', '#d27616')
    for regime, color in zip(REGIMES, colors):
        for variant, style in zip(VARIANTS, ('--', '-')):
            curves = []
            for seed in SEEDS:
                log = [json.loads(x) for x in (model_path(regime, variant, seed).parent / 'training.jsonl').read_text().splitlines()]
                curves.append(np.array([(70*r['hidden'] + 230*r['visible']) / 300 for r in log]).reshape(20,100).mean(-1))
            axes[0].plot(np.arange(100,2001,100), np.mean(curves,0), style, color=color, label=regime + ' ' + variant)
    axes[0].set(xlabel='Step', ylabel='Training masked-input all-patch MSE'); axes[0].legend(fontsize=8)
    split = 'transfer' if 'transfer' in analysis else 'development_scene'
    s = analysis[split]['summary']
    for i,(regime,color) in enumerate(zip(REGIMES,colors)):
        change = [100*(s[regime+'_center'][m]/s[regime+'_local'][m]-1) for m in METRICS[:3]]
        axes[1].bar(np.arange(3)+(i-1)*.25, change, .25, color=color, label=regime)
        axes[2].bar(np.arange(2)+(i-1)*.25, [s[regime+'_'+v]['clean_input_mse'] for v in VARIANTS], .25, color=color, label=regime)
    axes[1].axhline(0,color='black',lw=.7); axes[1].set_xticks(range(3), LABELS[:3])
    axes[1].set_ylabel('Center error change vs matched Local (%)\nNegative = improvement'); axes[1].legend()
    axes[2].set_xticks(range(2), ['Local','Center']); axes[2].set_ylabel('Clean-input token change (lower = better)'); axes[2].legend()
    fig.suptitle(split + ' | all three fixed seeds')
    fig.savefig(OUT / 'comparison.png', dpi=150); plt.close(fig); paths.append(str(OUT / 'comparison.png'))
    write_json(OUT / 'analysis.json', analysis)
    write_json(OUT / 'figure_manifest.json', {'figures': paths})
    print(json.dumps({s: d['summary'] for s,d in analysis.items()}), flush=True)


if __name__ == '__main__': main()
