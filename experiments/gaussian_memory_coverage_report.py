"""Scene-balanced paired changes and development visualizations, no model selection."""
import json

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from experiments.gaussian_memory_attention_train import write_json
from experiments.gaussian_memory_coverage_data import OUT, SEEDS, VARIANTS, model_path

METRICS = ('hidden_mse', 'visible_mse', 'all_patch_mse', 'clean_input_mse')
PAIRS = (('local', 'center'), ('wide', 'direct'))
REGIMES = ('narrow', 'broad')


def matrix(rows, regime, variant, metric, scenes):
    out = np.empty((len(scenes), len(SEEDS)))
    for i, scene in enumerate(scenes):
        for j, seed in enumerate(SEEDS):
            values = [r[metric] for r in rows if r['regime'] == regime and r['variant'] == variant
                      and r['scene'] == scene and r['seed'] == seed]
            assert values, (regime, variant, metric, scene, seed)
            out[i, j] = np.mean(values)
    return out


def estimate(values, scenes):
    rng = np.random.default_rng(44102)
    si = rng.integers(len(scenes), size=(10000, len(scenes)))
    ki = rng.integers(len(SEEDS), size=(10000, len(SEEDS)))
    boot = values[si[:, :, None], ki[:, None, :]].mean((1, 2))
    return {'mean_gain': float(values.mean()), 'paired_scene_seed_bootstrap_95': np.quantile(boot, [.025, .975]).tolist(),
            'paired_seed_gain': values.mean(0).tolist(), 'scenes_better': int((values.mean(1) > 0).sum()),
            'scenes_total': len(scenes), 'scene_gain': dict(zip(scenes, values.mean(1).tolist())),
            'scene_seed_gain': values.tolist(), 'single_scene_only': len(scenes) == 1}


def analysis():
    scores = json.loads((OUT / 'evaluation/scores.json').read_text())
    result = {}
    for split in sorted({r['split'] for r in scores['rows']}) + ['primary_both_clips']:
        def selected(row):
            return row['split'] in ('used_clip40', 'used_clip80') if split == 'primary_both_clips' else row['split'] == split
        all_rows = [r for r in scores['rows'] if selected(r)]
        rows = [r for r in all_rows if r['condition'] == 'normal']
        clean = [r for r in scores['preservation'] if selected(r)]
        scenes = sorted({r['scene'] for r in rows})
        summary = {regime: {v: {} for v in VARIANTS} for regime in REGIMES}
        comparisons, changes, model_changes = {}, {}, {}
        for metric in METRICS:
            source = clean if metric == 'clean_input_mse' else rows
            values = {(regime, variant): matrix(source, regime, variant, metric, scenes)
                      for regime in REGIMES for variant in VARIANTS}
            for (regime, variant), value in values.items():
                summary[regime][variant][metric] = float(value.mean())
            comparisons[metric] = {}; changes[metric] = {}; model_changes[metric] = {}
            for left, right in PAIRS:
                name = left + '_minus_' + right
                comparisons[metric][name] = {}
                for regime in REGIMES:
                    gain = values[(regime, left)] - values[(regime, right)]
                    estimate_result = estimate(gain, scenes)
                    estimate_result['relative_reduction_percent'] = float(100 * gain.mean() / values[(regime, left)].mean())
                    comparisons[metric][name][regime] = estimate_result
                delta = (values[('broad', left)] - values[('broad', right)] -
                         values[('narrow', left)] + values[('narrow', right)])
                changes[metric][name] = estimate(delta, scenes)
            for v in VARIANTS:
                delta = values[('narrow', v)] - values[('broad', v)]
                model_changes[metric][v] = estimate(delta, scenes)
                model_changes[metric][v]['relative_reduction_percent'] = float(100 * delta.mean() / values[('narrow', v)].mean())
        interventions = {}
        for regime in REGIMES:
            interventions[regime] = {}
            for variant in ('center', 'direct'):
                interventions[regime][variant] = {}
                for condition in ('normal', 'removed', 'swapped'):
                    part = [r for r in all_rows if r['condition'] == condition]
                    interventions[regime][variant][condition] = {
                        m: float(matrix(part, regime, variant, m, scenes).mean()) for m in METRICS[:3]}
        result[split] = {'summary': summary, 'comparisons': comparisons, 'gain_change_broad_minus_narrow': changes,
                         'model_improvement_broad_over_narrow': model_changes, 'interventions': interventions,
                         'scenes': scenes, 'episodes': len({r['episode'] for r in rows}),
                         'confirmation_claim': False, 'all_means_scene_balanced': True}
    write_json(OUT / 'analysis.json', result)
    return result


def figures(stats):
    figures = []
    for path in sorted((OUT / 'evaluation').glob('*_maps.pt')):
        d = torch.load(path, map_location='cpu', weights_only=False)
        q = int(d['query'][0]); mask = d['mask'][0].reshape(15, 20).numpy(); rgb = d['rgb'].numpy()
        observed = rgb[q].copy()
        observed[mask.repeat(14, 0).repeat(14, 1)] = np.array([.485, .456, .406]) * 255
        fig, axes = plt.subplots(3, 4, figsize=(14, 10), constrained_layout=True)
        for ax, picture, title in zip(axes[0], [rgb[0], rgb[12], rgb[q], observed],
                                      ['Reference 0', 'Reference 12', 'Full query: target only', 'Observed query']):
            ax.imshow(picture); ax.set_title(title, fontsize=9)
        error_maps = [d['errors'][r + '_' + v][0].reshape(15, 20).numpy() for r in REGIMES for v in VARIANTS]
        vmax = float(np.quantile(np.stack(error_maps), .98))
        for i, regime in enumerate(REGIMES):
            for ax, variant in zip(axes[i + 1], VARIANTS):
                e = d['errors'][regime + '_' + variant][0].reshape(15, 20).numpy()
                im = ax.imshow(e, vmin=0, vmax=vmax, cmap='magma')
                ax.contour(mask.astype(float), [.5], colors='cyan', linewidths=.6)
                ax.set_title(f'{regime} {variant}: hidden {e[mask].mean():.4f}\nvisible {e[~mask].mean():.4f}; all {e.mean():.4f}', fontsize=9)
        fig.colorbar(im, ax=list(axes[1:].flat), shrink=.7, label='Normalized token MSE; common scale')
        for ax in axes.flat:
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(d['episode']['scene'] + ' | ' + d['split'] + '\nFirst query/mask, seed 20261110', fontsize=11)
        output = path.with_name(d['episode']['id'] + '.png')
        fig.savefig(output, dpi=110); plt.close(fig); figures.append(str(output))
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    colors = {'local': '#85868e', 'center': '#007d79', 'wide': '#ab72a3', 'direct': '#d27616'}
    for variant in VARIANTS:
        for regime in REGIMES:
            curves = []
            for seed in SEEDS:
                rows = [json.loads(x) for x in (model_path(regime, variant, seed).parent / 'training.jsonl').read_text().splitlines()]
                curves.append(np.array([r['loss'] for r in rows]).reshape(20, 100).mean(-1))
            axes[0].plot(np.arange(100, 2001, 100), np.mean(curves, 0), color=colors[variant],
                         ls='-' if regime == 'broad' else '--', label=regime + ' ' + variant)
    axes[0].set(xlabel='Step (same 2000-step budget)', ylabel='Training token MSE', title='Training sets differ; curves are diagnostic')
    axes[0].legend(fontsize=7, ncol=2)
    splits = ['used_clip40', 'used_clip80', 'primary_both_clips']
    for ax, (left, right) in zip(axes[1:], PAIRS):
        for i, regime in enumerate(REGIMES):
            means, lows, highs = [], [], []
            for split in splits:
                s = stats[split]; r = s['comparisons']['all_patch_mse'][left + '_minus_' + right][regime]
                denominator = s['summary'][regime][left]['all_patch_mse']
                means.append(r['relative_reduction_percent'])
                low, high = np.asarray(r['paired_scene_seed_bootstrap_95']) / denominator * 100
                lows.append(means[-1] - low); highs.append(high - means[-1])
            x = np.arange(3) + (i - .5) * .32
            ax.bar(x, means, .3, color='#9ca7b3' if i == 0 else colors[right], label=regime)
            ax.errorbar(x, means, yerr=[lows, highs], fmt='none', color='black', capsize=3, lw=.8)
        ax.axhline(0, color='black', lw=.8)
        ax.set_xticks(range(3), ['Clip40', 'Clip80', 'Both (12 scenes)'])
        ax.set(title=right + ' vs ' + left, ylabel='All-patch error reduction (%)\nPositive = memory benefit')
        ax.legend()
    output = OUT / 'comparison.png'; fig.savefig(output, dpi=140); plt.close(fig); figures.append(str(output))
    write_json(OUT / 'figure_manifest.json', {'figures': figures})


def main():
    torch.set_num_threads(4)
    stats = analysis(); figures(stats)
    for split in ('used_clip40', 'used_clip80', 'primary_both_clips'):
        print(json.dumps({'split': split, 'summary': stats[split]['summary'],
                          'main_comparisons': stats[split]['comparisons']['all_patch_mse'],
                          'gain_change': stats[split]['gain_change_broad_minus_narrow']['all_patch_mse']}), flush=True)


if __name__ == '__main__':
    main()
