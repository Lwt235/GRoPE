"""Fixed-case visualizations for the development-only shared-writer experiment."""
import json

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from experiments.gaussian_memory_attention_train import write_json
from experiments.gaussian_memory_joint_write import OUT, NEW, SEEDS, model_path

ORDER = ('full_gaussian', 'content_gaussian', 'gaussian', 'full_center', 'large_local')
LABELS = {'full_gaussian': 'Learned geometry + content', 'content_gaussian': 'Learned content only',
          'gaussian': 'Frozen Gaussian', 'full_center': 'Learned Center', 'large_local': 'Strong Local',
          'global': 'Global summary', 'center': 'Frozen Center', 'local': 'Small Local'}


def main():
    torch.set_num_threads(4); stats = json.loads((OUT / 'analysis.json').read_text()); figures = []; seed = SEEDS[0]
    for rid in ('fit_00', 'transfer_00', 'fidelity_07', 'confirm_00', 'confirm_05', 'confirm_10', 'confirm_15'):
        d = torch.load(OUT / 'evaluation' / (rid + '_maps.pt'), map_location='cpu', weights_only=False)
        q = int(d['query'][0]); mask = d['mask'][0].reshape(15, 20).numpy(); valid = d['valid'][0].reshape(15, 20).numpy()
        truth = d['target'][0].reshape(15, 20).numpy(); rgb = d['rgb'].numpy(); observed = rgb[q].copy()
        observed[mask.repeat(14, 0).repeat(14, 1)] = [.485*255, .456*255, .406*255]
        fig, axes = plt.subplots(3, 5, figsize=(17, 10), constrained_layout=True)
        for ax, pic, title in zip(axes[0, :3], [rgb[0], rgb[12], observed], ['Reference 0', 'Reference 12', 'Observed query']):
            ax.imshow(pic); ax.set_title(title)
        im = axes[0, 3].imshow(np.where(valid, np.exp(truth), np.nan), vmin=.1, vmax=6, cmap='viridis')
        axes[0, 3].set_title('Sensor target (m)'); axes[0, 4].axis('off')
        axes[0, 4].text(.05, .7, 'First query/mask\nSeed 20261110\nFixed case selection\nAll data previously used\nScoring is unclipped', fontsize=11)
        fig.colorbar(im, ax=list(axes[:2].flat), shrink=.6, label='Depth; display saturates at 6 m')
        errors = {}
        for ax, kind in zip(axes[1], ORDER):
            p = d['predictions'][f'{kind}_{seed}_normal'][0].reshape(15, 20).numpy(); e = (p - truth) ** 2
            errors[kind] = np.where(valid, e, np.nan)
            ax.imshow(np.exp(p), vmin=.1, vmax=6, cmap='viridis')
            ax.contour(mask.astype(float), [.5], colors='white', linewidths=.5)
            ax.set_title(f'{LABELS[kind]}: {e[valid].mean():.4f}', fontsize=9)
        vmax = max(float(np.nanquantile(np.stack(list(errors.values())), .95)), .001)
        for ax, kind in zip(axes[2], ORDER):
            im = ax.imshow(errors[kind], vmin=0, vmax=vmax, cmap='magma'); ax.set_title(LABELS[kind], fontsize=9)
        fig.colorbar(im, ax=list(axes[2]), shrink=.7, label='Squared log-depth error')
        for ax in axes.flat:
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(d['episode']['scene'] + ' | ' + d['split'])
        path = OUT / 'evaluation' / (rid + '.png'); fig.savefig(path, dpi=110); plt.close(fig); figures.append(str(path))
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)
    for kind in (*NEW, 'gaussian'):
        curves = []
        for seed in SEEDS:
            rows = [json.loads(s) for s in (model_path(kind, seed).parent / 'training.jsonl').read_text().splitlines()]
            curves.append(np.array([r['loss'] for r in rows]).reshape(20, 100).mean(-1))
        axes[0, 0].plot(np.arange(100, 2001, 100), np.mean(curves, 0), label=LABELS[kind])
    axes[0, 0].set(title='Same 2000-step sample sequence', xlabel='Step', ylabel='Training log-depth MSE')
    axes[0, 0].legend(fontsize=8)
    pairs = [('gaussian', 'full_gaussian'), ('content_gaussian', 'full_gaussian'),
             ('full_center', 'full_gaussian'), ('large_local', 'full_gaussian')]
    for split, offset, color in [('old12', -.17, '#9eafbf'), ('used_summary20', .17, '#008a75')]:
        for i, (left, right) in enumerate(pairs):
            data = stats[split]; e = data['comparisons']['log_mse'][left + '_minus_' + right]
            lo, hi = np.array(e['paired_scene_seed_bootstrap_95']) / data['summary'][left]['log_mse'] * 100
            value = e['relative_reduction_percent']
            axes[0, 1].barh(i + offset, value, height=.31, color=color)
            axes[0, 1].errorbar(value, i + offset, xerr=[[max(0, value-lo)], [max(0, hi-value)]], color='black', capsize=3)
    axes[0, 1].set_yticks(range(4), ['Full Gaussian vs ' + LABELS[l] for l, r in pairs], fontsize=8)
    axes[0, 1].axvline(0, color='black', lw=.7)
    axes[0, 1].set(title='Gray: old 12; green: previously used 20', xlabel='Error reduction (%)')
    data = stats['primary32']; scenes = data['scenes']
    effect = data['comparisons']['log_mse']['gaussian_minus_full_gaussian']
    axes[1, 0].barh(np.arange(len(scenes)), [effect['scene_gain'][s] for s in scenes], color='#008a75')
    axes[1, 0].set_yticks(np.arange(len(scenes)), [s.split('/')[0] for s in scenes], fontsize=6)
    axes[1, 0].axvline(0, color='black', lw=.7)
    axes[1, 0].set(title='All 32 non-training places; no fresh confirmation', xlabel='Frozen minus learned Gaussian log-MSE')
    bias = [data['summary'][k]['bias_squared'] for k in ORDER]
    spatial = [data['summary'][k]['centered_mse'] for k in ORDER]
    axes[1, 1].bar(range(5), bias, color='#9eafbf', label='Query mean bias squared')
    axes[1, 1].bar(range(5), spatial, bottom=bias, color='#b46c35', label='Centered spatial error')
    axes[1, 1].set_xticks(range(5), [LABELS[k] for k in ORDER], rotation=25, ha='right', fontsize=8)
    axes[1, 1].set(title='Scene-balanced primary error decomposition', ylabel='Log-depth MSE')
    axes[1, 1].legend(fontsize=8)
    fig.suptitle('Compact shared write — stored Gaussian state — fixed attention read')
    path = OUT / 'comparison.png'; fig.savefig(path, dpi=130); plt.close(fig); figures.append(str(path))
    write_json(OUT / 'figure_manifest.json', {'figures': figures, 'selection': 'prospectively fixed'})


if __name__ == '__main__':
    main()
