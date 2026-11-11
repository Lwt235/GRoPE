"""Paired context-target comparisons, teacher diagnostics and fixed figures."""
import json

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from experiments.gaussian_memory_attention_train import write_json
from experiments.gaussian_memory_coverage_report import estimate
from experiments.gaussian_memory_context_data import OUT, SEEDS, VARIANTS, model_path

METRICS = ('hidden_mse', 'visible_mse', 'all_patch_mse', 'full_query_context_mse')
PAIRS = (('wide', 'spatial'), ('content', 'spatial'), ('wide', 'content'))


def matrix(rows, regime, variant, metric, scenes, condition='normal'):
    result = np.empty((len(scenes), len(SEEDS)))
    for i, scene in enumerate(scenes):
        for j, seed in enumerate(SEEDS):
            values = [r[metric] for r in rows if r['regime'] == regime and r['variant'] == variant
                      and r['scene'] == scene and (r['seed'] == seed or variant == 'backbone')
                      and r.get('condition', 'normal') == condition]
            assert values, (regime, variant, metric, scene, seed, condition)
            result[i, j] = np.mean(values)
    return result


def analysis():
    scores = json.loads((OUT / 'evaluation/scores.json').read_text())
    result = {}
    for split in sorted({r['split'] for r in scores['rows']}) + ['primary_both_clips']:
        def selected(row):
            return row['split'] in ('used_clip40', 'used_clip80') if split == 'primary_both_clips' else row['split'] == split
        rows = [r for r in scores['rows'] if selected(r)]
        full = [r for r in scores['full_query'] if selected(r)]
        scenes = sorted({r['scene'] for r in rows})
        summary = {'context': {v: {} for v in VARIANTS}, 'single': {v: {} for v in ('wide', 'spatial')}, 'backbone': {}}
        comparisons, interventions, target_training_effect, gain_change = {}, {}, {}, {}
        for metric in METRICS:
            source = full if metric == 'full_query_context_mse' else rows
            values = {(r, v): matrix(source, r, v, metric, scenes)
                      for r, variants in [('context', VARIANTS), ('single', ('wide', 'spatial'))] for v in variants}
            for (regime, variant), value in values.items():
                summary[regime][variant][metric] = float(value.mean())
            if metric != 'full_query_context_mse':
                summary['backbone'][metric] = float(matrix(source, 'none', 'backbone', metric, scenes).mean())
            comparisons[metric], target_training_effect[metric] = {}, {}
            for left, right in PAIRS:
                gain = values[('context', left)] - values[('context', right)]
                detail = estimate(gain, scenes)
                detail['relative_reduction_percent'] = float(100 * gain.mean() / values[('context', left)].mean())
                comparisons[metric][left + '_minus_' + right] = detail
            for variant in ('wide', 'spatial'):
                gain = values[('single', variant)] - values[('context', variant)]
                detail = estimate(gain, scenes)
                detail['relative_reduction_percent'] = float(100 * gain.mean() / values[('single', variant)].mean())
                target_training_effect[metric][variant] = detail
            delta = (values[('context', 'wide')] - values[('context', 'spatial')] -
                     values[('single', 'wide')] + values[('single', 'spatial')])
            gain_change[metric] = estimate(delta, scenes)
            if metric == 'full_query_context_mse':
                continue
            interventions[metric] = {}
            for variant in ('content', 'spatial'):
                interventions[metric][variant] = {}
                for condition in ('removed', 'shuffled'):
                    perturbed = matrix(rows, 'context', variant, metric, scenes, condition)
                    normal = values[('context', variant)]
                    damage = perturbed - normal
                    detail = estimate(damage, scenes)
                    detail.update(positive_means_damage=True, normal_mse=float(normal.mean()),
                                  perturbed_mse=float(perturbed.mean()),
                                  relative_error_increase_percent=float(100 * damage.mean() / normal.mean()))
                    interventions[metric][variant][condition] = detail
        aux = {}
        for regime, variants in [('context', VARIANTS), ('single', ('wide', 'spatial'))]:
            aux[regime] = {v: {'masked_original_s1_mse': float(matrix(rows, regime, v, 'original_s1_all_mse', scenes).mean()),
                              'full_query_s1_change_mse': float(matrix(full, regime, v, 'full_query_s1_change_mse', scenes).mean())}
                           for v in variants}
        td = [r for r in scores['teacher_diagnostics'] if selected(r)]
        teacher = {}
        for key in ('real_reference_vs_repeat_query_mse', 'spatially_centered_reference_change_mse',
                    'global_reference_change_mse', 's1_vs_context_target_mse', 's1_vs_repeat_query_target_mse'):
            teacher[key] = float(np.mean([np.mean([r[key] for r in td if r['scene'] == scene]) for scene in scenes]))
        teacher['spatially_varying_fraction_of_reference_change'] = teacher['spatially_centered_reference_change_mse'] / teacher['real_reference_vs_repeat_query_mse']
        backbone = [r for r in rows if r['variant'] == 'backbone']
        result[split] = {'summary': summary, 'comparisons': comparisons, 'interventions': interventions,
                         'target_training_effect_on_same_context_target': target_training_effect,
                         'spatial_over_wide_gain_change_on_same_context_target': gain_change,
                         'original_s1_auxiliary': aux, 'teacher_diagnostics': teacher,
                         'scenes': scenes, 'episodes': len(backbone), 'masked_samples': sum(r['query_count'] for r in backbone),
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
                                      ['Reference 0', 'Reference 12', 'Full query: teacher only', 'Observed student query']):
            ax.imshow(picture); ax.set_title(title, fontsize=9)
        variants = ('wide', 'content', 'spatial', 'spatial_shuffled')
        errors = [d['errors']['context_' + v][0].reshape(15, 20).numpy() for v in variants]
        vmax = float(np.quantile(np.stack(errors), .98))
        for ax, variant, e in zip(axes[1], variants, errors):
            im = ax.imshow(e, vmin=0, vmax=vmax, cmap='magma')
            ax.contour(mask.astype(float), [.5], colors='cyan', linewidths=.6)
            ax.set_title(f'{variant}: hidden {e[mask].mean():.4f}\nvisible {e[~mask].mean():.4f}; all {e.mean():.4f}', fontsize=9)
        fig.colorbar(im, ax=list(axes[1]), shrink=.8, label='Context-target token MSE; common scale')
        for ax in axes.flat:
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(d['episode']['scene'] + ' | ' + d['split'] + '\nFirst query/mask, seed 20261110', fontsize=11)
        output = path.with_name(d['episode']['id'] + '.png')
        fig.savefig(output, dpi=110); plt.close(fig); outputs.append(str(output))
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    colors = {'wide': '#85868e', 'content': '#d27616', 'spatial': '#007d79'}
    for variant in VARIANTS:
        curves = []
        for seed in SEEDS:
            rows = [json.loads(x) for x in (model_path(variant, seed).parent / 'training.jsonl').read_text().splitlines()]
            curves.append(np.array([r['loss'] for r in rows]).reshape(20, 100).mean(-1))
        axes[0].plot(np.arange(100, 2001, 100), np.mean(curves, 0), color=colors[variant], label=variant)
    axes[0].set(xlabel='Step (same 2000-step budget)', ylabel='Context-target token MSE', title='Same input, target, masks and budget')
    axes[0].legend()
    splits = ['used_clip40', 'used_clip80', 'primary_both_clips']
    for ax, (left, right) in zip(axes[1:], (('wide', 'spatial'), ('content', 'spatial'))):
        means, lows, highs = [], [], []
        for split in splits:
            s = stats[split]; detail = s['comparisons']['all_patch_mse'][left + '_minus_' + right]
            denominator = s['summary']['context'][left]['all_patch_mse']
            means.append(detail['relative_reduction_percent'])
            low, high = np.asarray(detail['paired_scene_seed_bootstrap_95']) / denominator * 100
            lows.append(means[-1] - low); highs.append(high - means[-1])
        ax.bar(np.arange(3), means, .6, color=colors[right])
        ax.errorbar(np.arange(3), means, yerr=[lows, highs], fmt='none', color='black', capsize=3, lw=.8)
        ax.axhline(0, color='black', lw=.8)
        ax.set_xticks(range(3), ['Clip40', 'Clip80', 'Both (12 scenes)'])
        ax.set(title=right + ' vs ' + left, ylabel='All-patch error reduction (%)\nPositive = spatial reader benefit')
    output = OUT / 'comparison.png'; fig.savefig(output, dpi=140); plt.close(fig); outputs.append(str(output))
    write_json(OUT / 'figure_manifest.json', {'figures': outputs, 'review_record': 'review.json'})


def main():
    torch.set_num_threads(4)
    stats = analysis(); figures(stats)
    for split in ('used_clip40', 'used_clip80', 'primary_both_clips'):
        d = stats[split]
        print(json.dumps({'split': split, 'summary': d['summary'], 'comparisons': d['comparisons']['all_patch_mse'],
                          'teacher_diagnostics': d['teacher_diagnostics']}), flush=True)


if __name__ == '__main__':
    main()
