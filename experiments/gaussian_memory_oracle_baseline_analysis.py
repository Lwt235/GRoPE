"""Describe the predeclared zero-training controls without selecting a model."""
import json

import numpy as np
import torch

from experiments.gaussian_memory_attention_train import write_json
from experiments.gaussian_memory_coverage_report import estimate
from experiments.gaussian_memory_oracle_data import OUT, SEEDS, checked
from experiments.gaussian_memory_oracle_report import matrix
from experiments.gaussian_memory_setup import sha256


def values(rows, case, metric, scenes):
    if '_seed_' not in case:
        return matrix(rows, case, metric, scenes)
    geometry, variant = case.split('_', 1)
    mean = np.array([np.mean([r[metric] for r in rows if r['scene'] == scene
                             and r['read_geometry'] == geometry and r['variant'] == variant])
                     for scene in scenes])
    # Broadcasting a deterministic control pairs it with each trained seed;
    # it does not create additional independent baseline observations.
    return np.broadcast_to(mean[:, None], (len(scenes), len(SEEDS)))


def main():
    checked(); torch.set_num_threads(4)
    data = json.loads((OUT / 'evaluation/scores.json').read_text())['rows']
    pairs = [('predicted_seed_center', 'sensor_seed_center'),
             ('predicted_seed_gaussian', 'sensor_seed_gaussian'),
             ('local', 'sensor_seed_center'), ('sensor_gaussian', 'sensor_seed_center'),
             ('sensor_center', 'sensor_seed_center'), ('sensor_fixed', 'sensor_seed_center')]
    results = {}
    for split in sorted({r['split'] for r in data}) + ['primary_both_clips']:
        rows = [r for r in data if r['condition'] == 'normal' and
                (r['split'] in ('used_clip40', 'used_clip80') if split == 'primary_both_clips' else r['split'] == split)]
        scenes = sorted({r['scene'] for r in rows}); stats = {}
        for metric in ('log_mse', 'hidden_log_mse', 'visible_log_mse', 'rmse_m', 'absrel'):
            stats[metric] = {}
            for left, right in pairs:
                a, b = values(rows, left, metric, scenes), values(rows, right, metric, scenes)
                effect = estimate(a - b, scenes)
                effect.update(relative_reduction_percent=100 * effect['mean_gain'] / a.mean(),
                              control_mean=float(a.mean()), candidate_mean=float(b.mean()))
                stats[metric][left + '_minus_' + right] = effect
        baseline_change = values(rows, 'predicted_seed_center', 'log_mse', scenes) - values(rows, 'sensor_seed_center', 'log_mse', scenes)
        learned_change = values(rows, 'predicted_gaussian', 'log_mse', scenes) - values(rows, 'sensor_gaussian', 'log_mse', scenes)
        stats['geometry_gain_direct_minus_learned'] = estimate(baseline_change - learned_change, scenes)
        results[split] = stats
    checks = []
    for rid in ('fit_00', 'transfer_08', 'fidelity_07'):
        maps = torch.load(OUT / 'evaluation' / (rid + '_maps.pt'), map_location='cpu', weights_only=False)
        truth, valid = maps['target'].numpy().astype('float64'), maps['valid'].numpy()
        for geometry in ('predicted', 'sensor'):
            for variant in ('seed_center', 'seed_gaussian'):
                prediction = maps['predictions'][f'none_{geometry}_{variant}'].numpy().astype('float64')
                mse = (((prediction - truth) ** 2 * valid).sum(-1) / valid.sum(-1)).mean()
                original = next(r for r in data if r['episode'] == rid and r['read_geometry'] == geometry and r['variant'] == variant)
                diff = abs(float(mse) - original['log_mse']); assert diff < 1e-6
                checks.append(dict(episode=rid, geometry=geometry, variant=variant, numpy_score_difference=diff))
    result = {'complete': True, 'predeclared_controls': True, 'new_training_or_prediction': False,
              'interpretation': 'Post-score paired description of all predeclared baseline controls; no selection or fresh confirmation. Deterministic controls are repeated only for pairing with trained seeds.',
              'source_sha256': sha256(__file__), 'scores_sha256': sha256(OUT / 'evaluation/scores.json'),
              'comparisons': results, 'independent_score_checks': checks}
    write_json(OUT / 'baseline_analysis.json', result)
    print(json.dumps({pair: {k: value[k] for k in ('relative_reduction_percent', 'paired_scene_seed_bootstrap_95', 'paired_seed_gain', 'scenes_better')}
                      for pair, value in results['primary_both_clips']['log_mse'].items()}))


if __name__ == '__main__':
    main()
