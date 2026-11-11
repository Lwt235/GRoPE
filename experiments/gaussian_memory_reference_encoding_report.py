"""Scene-balanced paired encoding comparison and fixed diagnostic figures."""
import json

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from experiments.gaussian_memory_attention_train import write_json
from experiments.gaussian_memory_coverage_report import estimate
from experiments.gaussian_memory_reference_encoding_data import OUT, SEEDS, VARIANTS, model_path

METRICS = ('hidden_mse', 'visible_mse', 'all_patch_mse', 'clean_input_mse')
PAIRS = (('wide', 's1'), ('s4', 's1'), ('wide', 's4'))


def matrix(rows, variant, metric, scenes, condition='normal'):
    result = np.empty((len(scenes), len(SEEDS)))
    for i, scene in enumerate(scenes):
        for j, seed in enumerate(SEEDS):
            values = [r[metric] for r in rows if r['variant'] == variant and r['scene'] == scene
                      and (r['seed'] == seed or variant == 'backbone')
                      and r.get('condition', 'normal') == condition]
            assert values, (variant, metric, scene, seed, condition)
            result[i, j] = np.mean(values)
    return result


def analysis():
    scores = json.loads((OUT / 'evaluation/scores.json').read_text())
    result = {}
    for split in sorted({r['split'] for r in scores['rows']}) + ['primary_both_clips']:
        def selected(row):
            return row['split'] in ('used_clip40', 'used_clip80') if split == 'primary_both_clips' else row['split'] == split
        rows = [r for r in scores['rows'] if selected(r)]
        clean = [r for r in scores['preservation'] if selected(r)]
        scenes = sorted({r['scene'] for r in rows})
        summary = {v: {} for v in ('backbone',) + VARIANTS}
        comparisons, interventions, perturbed_utility = {}, {}, {}
        for metric in METRICS:
            source = clean if metric == 'clean_input_mse' else rows
            variants = VARIANTS if metric == 'clean_input_mse' else ('backbone',) + VARIANTS
            values = {v: matrix(source, v, metric, scenes) for v in variants}
            for variant, value in values.items():
                summary[variant][metric] = float(value.mean())
            comparisons[metric] = {}
            for left, right in PAIRS:
                gain = values[left] - values[right]
                detail = estimate(gain, scenes)
                detail['relative_reduction_percent'] = float(100 * gain.mean() / values[left].mean())
                comparisons[metric][left + '_minus_' + right] = detail
            if metric == 'clean_input_mse':
                continue
            interventions[metric], perturbed_utility[metric] = {}, {}
            for variant in ('s4', 's1'):
                interventions[metric][variant], perturbed_utility[metric][variant] = {}, {}
                for condition in ('removed', 'shuffled'):
                    perturbed = matrix(rows, variant, metric, scenes, condition)
                    damage = perturbed - values[variant]
                    detail = estimate(damage, scenes)
                    detail['positive_means_damage'] = True
                    detail['normal_mse'] = float(values[variant].mean())
                    detail['perturbed_mse'] = float(perturbed.mean())
                    detail['relative_error_increase_percent'] = float(100 * damage.mean() / values[variant].mean())
                    interventions[metric][variant][condition] = detail
                    gain = values['wide'] - perturbed
                    detail = estimate(gain, scenes)
                    detail['relative_reduction_percent'] = float(100 * gain.mean() / values['wide'].mean())
                    perturbed_utility[metric][variant][condition] = detail
        backbone = [r for r in rows if r['variant'] == 'backbone']
        result[split] = {'summary': summary, 'comparisons': comparisons, 'interventions': interventions,
                         'wide_minus_perturbed': perturbed_utility, 'scenes': scenes,
                         'episodes': len(backbone), 'masked_samples': sum(r['query_count'] for r in backbone),
                         'confirmation_claim': False, 'all_means_scene_balanced': True,
                         'uncertainty': 'Paired 10000 bootstrap resamples across scenes and seeds; clips grouped within scene.'}
    write_json(OUT / 'analysis.json', result)
    return result


def figures(stats):
    outputs = []
    for path in sorted((OUT / 'evaluation').glob('*_maps.pt')):
        d = torch.load(path, map_location='cpu', weights_only=False)
        q = int(d['query'][0]); mask = d['mask'][0].reshape(15, 20).numpy(); rgb = d['rgb'].numpy()
        observed = rgb[q].copy()
        observed[mask.repeat(14, 0).repeat(14, 1)] = np.array([.485, .456, .406]) * 255
        fig, axes = plt.subplots(2, 4, figsize=(14, 7), constrained_layout=True)
        for ax, picture, title in zip(axes[0], [rgb[0], rgb[12], rgb[q], observed],
                                      ['Reference 0', 'Reference 12', 'Full query: target only', 'Observed query']):
            ax.imshow(picture); ax.set_title(title, fontsize=9)
        variants = ('wide', 's4', 's1', 's1_shuffled')
        errors = [d['errors'][v][0].reshape(15, 20).numpy() for v in variants]
        vmax = float(np.quantile(np.stack(errors), .98))
        for ax, variant, e in zip(axes[1], variants, errors):
            im = ax.imshow(e, vmin=0, vmax=vmax, cmap='magma')
            ax.contour(mask.astype(float), [.5], colors='cyan', linewidths=.6)
            ax.set_title(f'{variant}: hidden {e[mask].mean():.4f}\nvisible {e[~mask].mean():.4f}; all {e.mean():.4f}', fontsize=9)
        fig.colorbar(im, ax=list(axes[1]), shrink=.8, label='Normalized token MSE; common scale')
        for ax in axes.flat:
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(d['episode']['scene'] + ' | ' + d['split'] + '\nFirst query/mask, seed 20261110', fontsize=11)
        output = path.with_name(d['episode']['id'] + '.png')
        fig.savefig(output, dpi=110); plt.close(fig); outputs.append(str(output))
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    colors = {'wide': '#85868e', 's4': '#d27616', 's1': '#007d79'}
    for variant in VARIANTS:
        curves = []
        for seed in SEEDS:
            rows = [json.loads(x) for x in (model_path(variant, seed).parent / 'training.jsonl').read_text().splitlines()]
            curves.append(np.array([r['loss'] for r in rows]).reshape(20, 100).mean(-1))
        axes[0].plot(np.arange(100, 2001, 100), np.mean(curves, 0), color=colors[variant], label=variant)
    axes[0].set(xlabel='Step (same 2000-step budget)', ylabel='Training token MSE', title='Same inputs, masks and initial S1/S4 weights')
    axes[0].legend()
    splits = ['used_clip40', 'used_clip80', 'primary_both_clips']
    for ax, metric in zip(axes[1:], ('all_patch_mse', 'hidden_mse')):
        for i, variant in enumerate(('s4', 's1')):
            means, lows, highs = [], [], []
            for split in splits:
                s = stats[split]; detail = s['comparisons'][metric]['wide_minus_' + variant]
                denominator = s['summary']['wide'][metric]
                means.append(detail['relative_reduction_percent'])
                low, high = np.asarray(detail['paired_scene_seed_bootstrap_95']) / denominator * 100
                lows.append(means[-1] - low); highs.append(high - means[-1])
            x = np.arange(3) + (i - .5) * .32
            ax.bar(x, means, .3, color=colors[variant], label=variant)
            ax.errorbar(x, means, yerr=[lows, highs], fmt='none', color='black', capsize=3, lw=.8)
        ax.axhline(0, color='black', lw=.8)
        ax.set_xticks(range(3), ['Clip40', 'Clip80', 'Both (12 scenes)'])
        ax.set(title=metric + ': memory vs Wide', ylabel='Error reduction (%)\nPositive = memory benefit')
        ax.legend()
    output = OUT / 'comparison.png'; fig.savefig(output, dpi=140); plt.close(fig); outputs.append(str(output))
    write_json(OUT / 'figure_manifest.json', {'figures': outputs, 'images_reviewed': False})


def main():
    torch.set_num_threads(4)
    stats = analysis(); figures(stats)
    for split in ('used_clip40', 'used_clip80', 'primary_both_clips'):
        detail = stats[split]
        print(json.dumps({'split': split, 'summary': detail['summary'],
                          'comparisons': detail['comparisons']['all_patch_mse'],
                          'interventions': {v: {c: r['relative_error_increase_percent'] for c, r in cs.items()}
                                            for v, cs in detail['interventions']['all_patch_mse'].items()}}), flush=True)


if __name__ == '__main__':
    main()
