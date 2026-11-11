"""Scene-balanced development analysis; no checkpoint or outcome selection."""
import json

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from experiments.gaussian_memory_attention_train import write_json
from experiments.gaussian_memory_surface_data import OUT, VARIANTS, SEEDS, model_path
from experiments.gaussian_memory_coverage_report import estimate

METRICS = ('log_mse', 'hidden_log_mse', 'visible_log_mse', 'rmse_m', 'absrel')
PAIRS = (('local', 'gaussian'), ('fixed', 'gaussian'), ('center', 'gaussian'), ('local', 'center'), ('local', 'fixed'))


def matrix(rows, variant, metric, scenes, condition='normal'):
    result = []
    for scene in scenes:
        line = []
        for seed in SEEDS:
            values = [r[metric] for r in rows if r['scene'] == scene and r['variant'] == variant
                      and r['seed'] == seed and r.get('condition', 'normal') == condition and r[metric] is not None]
            assert values, (scene, seed, variant, condition, metric)
            line.append(float(np.mean(values)))
        result.append(line)
    return np.array(result)


def analysis():
    data = json.loads((OUT / 'evaluation/scores.json').read_text()); result = {}
    splits = sorted({r['split'] for r in data['rows']}) + ['primary_both_clips']
    for split in splits:
        select = lambda r: r['split'] in ('used_clip40', 'used_clip80') if split == 'primary_both_clips' else r['split'] == split
        rows = [r for r in data['rows'] if select(r)]; full = [r for r in data['full_query'] if select(r)]
        consistency = [r for r in data['consistency'] if select(r)]
        scenes = sorted({r['scene'] for r in rows}); summary = {v: {} for v in VARIANTS}; comparisons = {}
        all_metrics = list(METRICS) + ['full_log_mse', 'consistency_residual_xyz_mse']
        for metric in all_metrics:
            source, key = ((full, 'log_mse') if metric == 'full_log_mse' else
                           (consistency, 'residual_xyz_mse') if metric == 'consistency_residual_xyz_mse' else (rows, metric))
            values = {v: matrix(source, v, key, scenes) for v in VARIANTS}
            for v in VARIANTS: summary[v][metric] = float(values[v].mean())
            comparisons[metric] = {}
            for left, right in PAIRS:
                s = estimate(values[left] - values[right], scenes)
                s['relative_reduction_percent'] = float(100 * s['mean_gain'] / values[left].mean())
                comparisons[metric][left + '_minus_' + right] = s
        interventions = {}
        for v in VARIANTS[1:]:
            interventions[v] = {}
            for condition in ('removed', 'swapped', 'shuffled', 'initial_geometry', 'wrong_camera'):
                interventions[v][condition] = {}
                for metric in ('log_mse', 'hidden_log_mse', 'visible_log_mse'):
                    normal = matrix(rows, v, metric, scenes); changed = matrix(rows, v, metric, scenes, condition)
                    s = estimate(changed - normal, scenes)
                    s['relative_error_increase_percent'] = float(100 * s['mean_gain'] / normal.mean())
                    s['interpretation'] = 'positive means intervention harms prediction'
                    interventions[v][condition][metric] = s
        baselines = {}
        for v in ('seed_center', 'seed_gaussian', 'train_mean'):
            baselines[v] = {metric: float(np.mean([np.mean([r[metric] for r in rows if r['variant'] == v and r['scene'] == scene])
                                                  for scene in scenes])) for metric in METRICS}
        result[split] = {'summary': summary, 'comparisons': comparisons, 'interventions': interventions,
                         'baselines': baselines, 'scenes': scenes, 'episodes': len({r['episode'] for r in rows}),
                         'confirmation_claim': False, 'scene_balanced': True,
                         'consistency_pairs_per_episode': {r['episode']: r['pairs'] for r in consistency},
                         'gt_pair_distance_m': float(np.mean([r['gt_pair_distance_mean_m'] for r in consistency]))}
    write_json(OUT / 'analysis.json', result)
    return result


def figures(stats):
    figures = []; seed = SEEDS[0]
    for rid in ('fit_00', 'coverage_04', 'transfer_00', 'transfer_08', 'fidelity_00', 'fidelity_07', 'fidelity_11'):
        d = torch.load(OUT / 'evaluation' / (rid + '_maps.pt'), map_location='cpu', weights_only=False)
        q = int(d['query'][0]); mask = d['mask'][0].reshape(15, 20).numpy()
        valid = d['valid'][0].reshape(15, 20).numpy(); target = d['target'][0].exp().reshape(15, 20).numpy()
        rgb = d['rgb'].numpy(); observed = rgb[q].copy(); observed[mask.repeat(14, 0).repeat(14, 1)] = [.485*255, .456*255, .406*255]
        fig, ax = plt.subplots(3, 4, figsize=(14, 10), constrained_layout=True)
        for a, picture, title in zip(ax[0, :3], [rgb[0], rgb[12], observed], ['Reference 0', 'Reference 12', 'Observed query']):
            a.imshow(picture); a.set_title(title)
        a = ax[0, 3]; im = a.imshow(np.where(valid, target, np.nan), vmin=.1, vmax=6, cmap='viridis'); a.set_title('Sensor target: metres')
        fig.colorbar(im, ax=list(ax[:2].flat), shrink=.6, label='Depth (m); values above 6 m saturated for display only')
        errors = []
        for a, v in zip(ax[1], VARIANTS):
            pred = d['predictions'][f'{v}_{seed}_normal'][0].reshape(15, 20).numpy()
            a.imshow(np.exp(pred), vmin=.1, vmax=6, cmap='viridis'); a.contour(mask.astype(float), [.5], colors='white', linewidths=.5)
            error = (pred - np.log(target))**2; errors.append(np.where(valid, error, np.nan))
            a.set_title(f'{v}: log MSE {error[valid].mean():.4f}', fontsize=10)
        limit = max(float(np.nanquantile(np.stack(errors), .95)), 1e-3)
        for a, v, error in zip(ax[2], VARIANTS, errors):
            im = a.imshow(error, vmin=0, vmax=limit, cmap='magma'); a.set_title(v + ': squared log-depth error', fontsize=9)
        fig.colorbar(im, ax=list(ax[2]), shrink=.7, label='Common error scale')
        for a in ax.flat: a.set_xticks([]); a.set_yticks([])
        fig.suptitle(d['episode']['scene'] + ' | ' + d['split'] + '\nFirst heldout query/mask, seed 20261110')
        path = OUT / 'evaluation' / (rid + '.png'); fig.savefig(path, dpi=110); plt.close(fig); figures.append(str(path))
    colors = {'local': '#707981', 'fixed': '#b48b32', 'center': '#286ba8', 'gaussian': '#079078'}
    fig, ax = plt.subplots(1, 3, figsize=(16, 5.5), constrained_layout=True)
    for v in VARIANTS:
        curves = []
        for seed in SEEDS:
            rows = [json.loads(s) for s in (model_path(v, seed).parent / 'training.jsonl').read_text().splitlines()]
            curves.append(np.array([r['loss'] for r in rows]).reshape(20, 100).mean(-1))
        ax[0].plot(np.arange(100, 2001, 100), np.mean(curves, 0), color=colors[v], label=v)
    ax[0].set(title='Fixed-budget training', xlabel='Step', ylabel='Sensor log-depth MSE'); ax[0].legend()
    main = stats['primary_both_clips']; names = ['local_minus_gaussian', 'fixed_minus_gaussian', 'center_minus_gaussian', 'local_minus_center']
    for i, name in enumerate(names):
        row = main['comparisons']['log_mse'][name]; left = name.split('_minus_')[0]
        denominator = main['summary'][left]['log_mse']; lo, hi = np.array(row['paired_scene_seed_bootstrap_95']) / denominator * 100
        value = row['relative_reduction_percent']; ax[1].barh(i, value, color='#079078')
        ax[1].errorbar(value, i, xerr=[[value-lo], [hi-value]], color='black', capsize=3)
    ax[1].axvline(0, color='black', lw=.7); ax[1].set_yticks(range(4), ['Gaussian / Local', 'Gaussian / Fixed', 'Gaussian / Center', 'Center / Local'])
    ax[1].set(title='12 non-training places, both clips', xlabel='Log-depth error reduction (%)')
    changes = main['comparisons']['log_mse']['local_minus_gaussian']['scene_gain']; scenes = main['scenes']
    ax[2].barh(np.arange(len(scenes)), [changes[s] for s in scenes], color=['#079078' if changes[s]>0 else '#b85e64' for s in scenes])
    ax[2].set_yticks(range(len(scenes)), [s.split('/')[0] for s in scenes], fontsize=8)
    ax[2].axvline(0, color='black', lw=.7); ax[2].set(title='Each place; no exclusion', xlabel='Local error minus Gaussian error')
    path = OUT / 'comparison.png'; fig.savefig(path, dpi=140); plt.close(fig); figures.append(str(path))
    write_json(OUT / 'figure_manifest.json', {'figures': figures})


def main():
    torch.set_num_threads(4); stats = analysis(); figures(stats)
    main = stats['primary_both_clips']
    print(json.dumps({'primary_summary': main['summary'],
                      'primary_comparisons': {k: {x: v[x] for x in ('mean_gain', 'relative_reduction_percent', 'paired_scene_seed_bootstrap_95', 'paired_seed_gain', 'scenes_better')}
                                              for k, v in main['comparisons']['log_mse'].items()}}))


if __name__ == '__main__': main()
