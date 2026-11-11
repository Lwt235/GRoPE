"""Fixed cases for the content-only positional comparison; no result selection."""
import json

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from experiments.gaussian_memory_attention_train import write_json
from experiments.gaussian_memory_content_center import OUT, SEEDS, checked, model_path, merged_maps

ORDER = ('content_gaussian', 'content_center', 'full_gaussian', 'global', 'large_local')
LABELS = dict(zip(ORDER, ('Content Gaussian', 'Content Center', 'Full Gaussian', 'Global summary', 'Strong Local')))


def main():
    plan = checked(True); torch.set_num_threads(4)
    stats = json.loads((OUT / 'analysis.json').read_text()); figures = []; seed = SEEDS[0]
    for rid in plan['figure_cases']:
        d = merged_maps(rid)
        q = int(d['query'][0]); mask = d['mask'][0].reshape(15, 20).numpy()
        valid = d['valid'][0].reshape(15, 20).numpy(); truth = d['target'][0].reshape(15, 20).numpy()
        rgb = d['rgb'].numpy(); observed = rgb[q].copy()
        observed[mask.repeat(14, 0).repeat(14, 1)] = [.485*255, .456*255, .406*255]
        fig, axes = plt.subplots(3, 5, figsize=(17, 10), constrained_layout=True)
        for ax, pic, title in zip(axes[0, :3], [rgb[0], rgb[12], observed], ['Reference 0', 'Reference 12', 'Observed query']):
            ax.imshow(pic); ax.set_title(title)
        im = axes[0, 3].imshow(np.where(valid, np.exp(truth), np.nan), vmin=.1, vmax=6, cmap='viridis')
        axes[0, 3].set_title('Sensor target (m)'); axes[0, 4].axis('off')
        axes[0, 4].text(.02, .7, 'First query/mask\nSeed 20261110\nFixed cases\nAll data previously used\nScoring is unclipped', fontsize=10)
        fig.colorbar(im, ax=list(axes[:2].flat), shrink=.6, label='Depth; display saturates at 6 m')
        errors = {}
        for ax, kind in zip(axes[1], ORDER):
            pred = d['predictions'][f'{kind}_{seed}_normal'][0].reshape(15, 20).numpy()
            error = (pred - truth) ** 2; errors[kind] = np.where(valid, error, np.nan)
            ax.imshow(np.exp(pred), vmin=.1, vmax=6, cmap='viridis')
            ax.contour(mask.astype(float), [.5], colors='white', linewidths=.5)
            ax.set_title(f'{LABELS[kind]}: {error[valid].mean():.4f}', fontsize=9)
        vmax = max(float(np.nanquantile(np.stack(list(errors.values())), .95)), .001)
        for ax, kind in zip(axes[2], ORDER):
            im = ax.imshow(errors[kind], vmin=0, vmax=vmax, cmap='magma'); ax.set_title(LABELS[kind], fontsize=9)
        fig.colorbar(im, ax=list(axes[2]), shrink=.7, label='Squared log-depth error')
        for ax in axes.flat:
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(d['episode']['scene'] + ' | ' + d['split'])
        path = OUT / 'evaluation' / (rid + '.png'); fig.savefig(path, dpi=110); plt.close(fig); figures.append(str(path))
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
    for kind in ORDER[:3]:
        curves = []
        for seed in SEEDS:
            rows = [json.loads(s) for s in (model_path(kind, seed).parent / 'training.jsonl').read_text().splitlines()]
            curves.append(np.array([r['loss'] for r in rows]).reshape(20, 100).mean(-1))
        axes[0, 0].plot(np.arange(100, 2001, 100), np.mean(curves, 0), label=LABELS[kind])
    axes[0, 0].set(title='Matched initialization and 2000-step samples', xlabel='Step', ylabel='Training log-MSE')
    axes[0, 0].legend(fontsize=9)
    groups = ('primary51', 'old12', 'used_summary20', 'used_writer19')
    for metric, offset, color in [('log_mse', -.16, '#008a75'), ('centered_mse', .16, '#b46c35')]:
        for i, group in enumerate(groups):
            data = stats[group]; e = data['comparisons'][metric]['content_center_minus_content_gaussian']
            value = e['relative_reduction_percent']
            lo, hi = np.array(e['paired_scene_seed_bootstrap_95']) / data['summary']['content_center'][metric] * 100
            axes[0, 1].barh(i + offset, value, height=.29, color=color,
                           label=('Overall' if metric == 'log_mse' else 'Centered spatial') if i == 0 else None)
            axes[0, 1].errorbar(value, i + offset, xerr=[[max(0, value-lo)], [max(0, hi-value)]], color='black', capsize=3)
    axes[0, 1].set_yticks(range(4), ['All 51 places', 'Old 12', 'Used 20', 'Last-round 19'])
    axes[0, 1].axvline(0, color='black', lw=.7); axes[0, 1].legend(fontsize=9)
    axes[0, 1].set(title='Content Gaussian vs matched Content Center', xlabel='Error reduction (%); positive favors Gaussian')
    data = stats['primary51']; effect = data['comparisons']['log_mse']['content_center_minus_content_gaussian']
    colors = {'old12': '#8797a7', 'used_summary20': '#008a75', 'used_writer19': '#b46c35'}
    for group, color in colors.items():
        scenes = stats[group]['scenes']
        positions = [data['scenes'].index(s) for s in scenes]
        axes[1, 0].scatter(positions, [effect['scene_gain'][s] for s in scenes], color=color, label=group, s=18)
    axes[1, 0].axhline(0, color='black', lw=.7); axes[1, 0].legend(fontsize=9)
    axes[1, 0].set(title='All 51 non-training places; no exclusions', xlabel='Fixed alphabetical place index',
                   ylabel='Content Center minus Content Gaussian log-MSE')
    bias = [data['summary'][k]['bias_squared'] for k in ORDER]
    spatial = [data['summary'][k]['centered_mse'] for k in ORDER]
    axes[1, 1].bar(range(5), bias, color='#9eafbf', label='Query mean bias squared')
    axes[1, 1].bar(range(5), spatial, bottom=bias, color='#b46c35', label='Centered spatial error')
    axes[1, 1].set_xticks(range(5), [LABELS[k] for k in ORDER], rotation=20, ha='right', fontsize=9)
    axes[1, 1].set(title='51-place equal-weight error decomposition', ylabel='Log-depth MSE')
    axes[1, 1].legend(fontsize=9)
    fig.suptitle('Fixed geometry + shared content writing: covariance position term vs center distance')
    path = OUT / 'comparison.png'; fig.savefig(path, dpi=120); plt.close(fig); figures.append(str(path))
    write_json(OUT / 'figure_manifest.json', {'figures': figures, 'selection': 'prospectively fixed six cases and overview'})


if __name__ == '__main__':
    main()
