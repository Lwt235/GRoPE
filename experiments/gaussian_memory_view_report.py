"""Scene-paired statistics and inspectable maps; no evaluation-time tuning."""
import json

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from experiments.gaussian_memory_attention_train import write_json
from experiments.gaussian_memory_view_data import OUT, SEEDS, VARIANTS


def comparisons(rows, split, metric='hidden_mse'):
    selected = [r for r in rows if r['split'] == split and r['condition'] == 'normal']
    scenes = sorted({r['scene'] for r in selected})
    result = {}
    for left, right in [('local', 'center'), ('local', 'gaussian'), ('center', 'gaussian'), ('backbone', 'center')]:
        by_scene = []
        for scene in scenes:
            values = {v: float(np.mean([r[metric] for r in selected if r['scene'] == scene and r['variant'] == v]))
                      for v in (left, right)}
            by_scene.append({'scene': scene, 'left': values[left], 'right': values[right],
                             'gain': values[left] - values[right]})
        difference = np.array([r['gain'] for r in by_scene])
        rng = np.random.default_rng(44102)
        bootstrap = difference[rng.integers(len(difference), size=(10000, len(difference)))].mean(-1)
        seed_gain = []
        if left != 'backbone':
            for seed in SEEDS:
                seed_gain.append(float(np.mean([r[metric] for r in selected if r['variant'] == left and r['seed'] == seed]) -
                                       np.mean([r[metric] for r in selected if r['variant'] == right and r['seed'] == seed])))
        result[left + '_minus_' + right] = {'mean_gain': float(difference.mean()),
            'relative_reduction_percent': float(100 * difference.mean() / np.mean([r['left'] for r in by_scene])),
            'scene_bootstrap_95': np.quantile(bootstrap, [.025, .975]).tolist(),
            'scenes_better': int((difference > 0).sum()), 'scenes_total': len(scenes),
            'paired_seed_gain': seed_gain, 'scenes': by_scene}
    return result


def figures(directory):
    paths = []
    for source in sorted(directory.glob('*_maps.pt')):
        d = torch.load(source, weights_only=False)
        q, mask = int(d['query'][0]), d['mask'][0].reshape(15, 20).numpy()
        rgb = d['rgb'].numpy(); observed = rgb[q].copy()
        pixels = mask.repeat(14, 0).repeat(14, 1)
        observed[pixels] = np.array([.485, .456, .406]) * 255
        fig, axes = plt.subplots(3, 4, figsize=(12, 8), constrained_layout=True)
        for col, ref in enumerate([0, 4, 8, 12]):
            axes[0, col].imshow(rgb[ref]); axes[0, col].set_title(f'Reference {ref}')
        axes[1, 0].imshow(rgb[q]); axes[1, 0].set_title('Full query: teacher only')
        axes[1, 1].imshow(observed); axes[1, 1].set_title('Observed RGB: network input')
        axes[1, 2].imshow(mask, vmin=0, vmax=1, cmap='gray'); axes[1, 2].set_title('Hidden patches: 23.3%')
        axes[1, 3].imshow(d['support'][0].reshape(15, 20), vmin=0, vmax=1, cmap='viridis')
        axes[1, 3].set_title('Projected support (proxy)')
        errors = [d['errors'][v][0].reshape(15, 20).numpy() for v in ('backbone', 'local', 'center', 'gaussian')]
        maximum = float(np.quantile(np.stack(errors), .98))
        for ax, error, name in zip(axes[2], errors, ['Frozen VGGT', 'View-only MLP', 'Center memory', 'Shape memory']):
            im = ax.imshow(error, vmin=0, vmax=maximum, cmap='magma')
            ax.contour(mask.astype(float), [.5], colors='cyan', linewidths=.6)
            ax.set_title(f'{name}\nHidden MSE {error[mask].mean():.4f}', fontsize=10)
        fig.colorbar(im, ax=list(axes[2]), shrink=.65, label='Normalized token MSE')
        for ax in axes.flat: ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(d['episode']['scene'] + ' | ' + d['episode']['id'] + ' | seed 20261110')
        path = directory / (d['episode']['id'] + '.png')
        fig.savefig(path, dpi=130); plt.close(fig); paths.append(str(path))
    return paths


def main():
    analysis, paths = {}, []
    for name in ('development_evaluation', 'transfer_evaluation'):
        directory = OUT / name
        if not (directory / 'scores.json').exists(): continue
        result = json.loads((directory / 'scores.json').read_text())
        # Descriptive accounting of the already recorded hidden/visible scores.
        # No model change or alternate checkpoint selection uses this summary.
        for r in result['rows']:
            r['all_patch_mse'] = (70 * r['hidden_mse'] + 230 * r['visible_mse']) / 300
        for split in sorted({r['split'] for r in result['rows']}):
            analysis[split] = {'comparisons': comparisons(result['rows'], split), 'summary': result['summary'][split],
                'clean_input_mse': {v: float(np.mean([r['clean_input_mse'] for r in result['preservation']
                                                     if r['split'] == split and r['variant'] == v])) for v in VARIANTS},
                'visible_comparisons': comparisons(result['rows'], split, 'visible_mse'),
                'all_patch_comparisons': comparisons(result['rows'], split, 'all_patch_mse')}
            for values in analysis[split]['summary'].values():
                values['all_patch_mse'] = (70 * values['hidden_mse'] + 230 * values['visible_mse']) / 300
        paths += figures(directory)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for variant, color in zip(VARIANTS, ('#697080', '#007d79', '#d27616')):
        curves = []
        for seed in SEEDS:
            rows = [json.loads(l) for l in (OUT / f'{variant}_{seed}/training.jsonl').read_text().splitlines()]
            curves.append(np.array([r['loss'] for r in rows]).reshape(20, 100).mean(-1))
        values = np.array(curves)
        axes[0].plot(np.arange(100, 2001, 100), values.mean(0), label=variant, color=color)
        axes[0].fill_between(np.arange(100, 2001, 100), values.min(0), values.max(0), alpha=.15, color=color)
    axes[0].set(xlabel='Training step', ylabel='Hidden/visible weighted token loss'); axes[0].legend()
    labels = list(analysis)
    for i, variant in enumerate(VARIANTS):
        values = [analysis[s]['summary'][variant + ':normal']['hidden_mse'] for s in labels]
        axes[1].bar(np.arange(len(labels)) + (i - 1) * .24, values, width=.24, label=variant)
    axes[1].set_xticks(np.arange(len(labels)), [s.replace('_', '\n') for s in labels])
    axes[1].set_ylabel('Hidden token MSE (lower is better)'); axes[1].legend()
    if 'transfer' in analysis:
        s = analysis['transfer']['summary']; metrics = ['hidden_mse', 'visible_mse', 'all_patch_mse']
        for i, variant in enumerate(['center', 'gaussian']):
            change = [100 * (s[variant + ':normal'][m] / s['local:normal'][m] - 1) for m in metrics]
            axes[2].bar(np.arange(3) + (i - .5) * .32, change, .32, label=variant)
        axes[2].axhline(0, color='black', linewidth=.7)
        axes[2].set_xticks(range(3), ['Hidden', 'Visible', 'All patches'])
        axes[2].set_ylabel('Transfer error change vs local (%)\nNegative = improvement')
        axes[2].legend()
    fig.savefig(OUT / 'comparison.png', dpi=150); plt.close(fig)
    paths.append(str(OUT / 'comparison.png'))
    write_json(OUT / 'analysis.json', analysis); write_json(OUT / 'figure_manifest.json', {'figures': paths})
    print(json.dumps(analysis), flush=True)


if __name__ == '__main__': main()
