"""Prospectively selected cases and paired error tradeoffs for fresh confirmation."""
import json

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from experiments.gaussian_memory_attention_train import write_json
from experiments.gaussian_memory_write_confirmation import OUT
from experiments.gaussian_memory_joint_write_figures import LABELS

ORDER = ('full_gaussian', 'content_gaussian', 'gaussian', 'full_center', 'global', 'large_local')


def main():
    torch.set_num_threads(4); stats = json.loads((OUT / 'analysis.json').read_text()); seed = 20261110
    catalog = json.loads((OUT / 'cache_identity.json').read_text())['catalog']; figures = []
    indices = sorted({min(i, len(catalog) - 1) for i in (0, 4, 8, 12, 16, 19)})
    for index in indices:
        rid = catalog[index]['id']
        d = torch.load(OUT / 'evaluation' / (rid + '_maps.pt'), map_location='cpu', weights_only=False)
        q = int(d['query'][0]); mask = d['mask'][0].reshape(15, 20).numpy(); valid = d['valid'][0].reshape(15, 20).numpy()
        target = d['target'][0].reshape(15, 20).numpy(); rgb = d['rgb'].numpy(); observed = rgb[q].copy()
        observed[mask.repeat(14, 0).repeat(14, 1)] = [.485*255, .456*255, .406*255]
        fig, axes = plt.subplots(3, 6, figsize=(20, 10), constrained_layout=True)
        for ax, pic, title in zip(axes[0, :3], [rgb[0], rgb[12], observed], ['Reference 0', 'Reference 12', 'Observed query']):
            ax.imshow(pic); ax.set_title(title)
        im = axes[0, 3].imshow(np.where(valid, np.exp(target), np.nan), vmin=.1, vmax=6, cmap='viridis')
        axes[0, 3].set_title('Sensor target (m)'); axes[0, 4].axis('off'); axes[0, 5].axis('off')
        axes[0, 4].text(.02, .72, 'First query/mask\nSeed 20261110\nFixed case order\nFresh directory identity\nNo model updates', fontsize=10)
        fig.colorbar(im, ax=list(axes[:2].flat), shrink=.6, label='Depth; display saturates at 6 m')
        errors = {}
        for ax, kind in zip(axes[1], ORDER):
            p = d['predictions'][f'{kind}_{seed}_normal'][0].reshape(15, 20).numpy(); e = (p - target) ** 2
            errors[kind] = np.where(valid, e, np.nan)
            ax.imshow(np.exp(p), vmin=.1, vmax=6, cmap='viridis'); ax.contour(mask.astype(float), [.5], colors='white', linewidths=.5)
            ax.set_title(f'{LABELS[kind]}\n{e[valid].mean():.4f}', fontsize=9)
        vmax = max(float(np.nanquantile(np.stack(list(errors.values())), .95)), .001)
        for ax, kind in zip(axes[2], ORDER):
            im = ax.imshow(errors[kind], vmin=0, vmax=vmax, cmap='magma'); ax.set_title(LABELS[kind], fontsize=8)
        fig.colorbar(im, ax=list(axes[2]), shrink=.7, label='Squared log-depth error; scores unclipped')
        for ax in axes.flat:
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(d['episode']['scene'] + ' | frozen checkpoints, new place directory')
        path = OUT / 'evaluation' / (rid + '.png'); fig.savefig(path, dpi=100); plt.close(fig); figures.append(str(path))
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), constrained_layout=True)
    controls = ('large_local', 'global', 'full_center', 'gaussian', 'content_gaussian')
    for ax, metric, title in zip(axes[0], ['log_mse', 'centered_mse'], ['Overall error reduction', 'Centered spatial error reduction']):
        for i, control in enumerate(controls):
            e = stats['comparisons'][metric][control + '_minus_full_gaussian']
            den = stats['summary'][control][metric]; value = e['relative_reduction_percent']
            lo, hi = np.array(e['paired_scene_seed_bootstrap_95']) / den * 100
            ax.barh(i, value, color='#008a75' if value >= 0 else '#b46c35')
            ax.errorbar(value, i, xerr=[[max(0, value-lo)], [max(0, hi-value)]], color='black', capsize=3)
        ax.set_yticks(range(len(controls)), ['Full Gaussian vs ' + LABELS[c] for c in controls], fontsize=8)
        ax.axvline(0, color='black', lw=.7); ax.set(title=title, xlabel='Reduction (%); positive favors Full Gaussian')
    scenes = stats['scenes']; gains = stats['comparisons']['log_mse']['large_local_minus_full_gaussian']['scene_gain']
    values = [gains[s] for s in scenes]
    axes[1, 0].barh(np.arange(len(scenes)), values, color=['#008a75' if v > 0 else '#b46c35' for v in values])
    axes[1, 0].set_yticks(np.arange(len(scenes)), [s.split('/')[0] for s in scenes], fontsize=8)
    axes[1, 0].axvline(0, color='black', lw=.7)
    axes[1, 0].set(title=f'All {len(scenes)} available places retained', xlabel='Strong Local minus Full Gaussian log-MSE')
    bias = [stats['summary'][k]['bias_squared'] for k in ORDER]
    space = [stats['summary'][k]['centered_mse'] for k in ORDER]
    axes[1, 1].bar(range(len(ORDER)), bias, color='#9eafbf', label='Query mean bias squared')
    axes[1, 1].bar(range(len(ORDER)), space, bottom=bias, color='#b46c35', label='Centered spatial error')
    axes[1, 1].set_xticks(range(len(ORDER)), [LABELS[k] for k in ORDER], rotation=25, ha='right', fontsize=8)
    axes[1, 1].set(title='Error decomposition', ylabel='Log-depth MSE'); axes[1, 1].legend(fontsize=8)
    fig.suptitle('Unchanged shared Gaussian memory models; fresh-place confirmation; no retraining')
    path = OUT / 'comparison.png'; fig.savefig(path, dpi=120); plt.close(fig); figures.append(str(path))
    write_json(OUT / 'figure_manifest.json', {'figures': figures, 'selection_indices': indices,
                                           'selection': 'fixed available-catalog positions; last available if fewer than 20'})


if __name__ == '__main__':
    main()
