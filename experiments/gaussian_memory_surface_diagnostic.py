"""Frozen-output, post-result decomposition; no learning or prediction correction."""
import json

import numpy as np
import torch

from experiments.gaussian_memory_attention_train import write_json
from experiments.gaussian_memory_setup import sha256
from experiments.gaussian_memory_surface_data import OUT, SEEDS, VARIANTS, checked, load_record
from experiments.gaussian_memory_surface_evaluate import world_points, consistency
from experiments.gaussian_memory_surface_report import matrix
from experiments.gaussian_memory_coverage_report import estimate


def decompose(pred, target, valid):
    error = pred.double() - target.double(); weight = valid.double(); n = weight.sum(-1)
    bias = (error * weight).sum(-1) / n
    mse = (error.square() * weight).sum(-1) / n
    centered = ((error-bias[:, None]).square() * weight).sum(-1) / n
    gradients = {}
    for key, value in [('prediction', pred), ('target', target)]:
        a = value.reshape(-1, 15, 20).double(); v = valid.reshape(-1, 15, 20)
        dy = (a[:, 1:] - a[:, :-1]).square(); dx = (a[:, :, 1:] - a[:, :, :-1]).square()
        vy = v[:, 1:] & v[:, :-1]; vx = v[:, :, 1:] & v[:, :, :-1]
        energy = ((dy * vy).sum((1, 2)) + (dx * vx).sum((1, 2))) / (vy.sum((1, 2)) + vx.sum((1, 2)))
        gradients[key + '_gradient_energy'] = float(energy.mean())
    return {'log_mse': float(mse.mean()), 'bias_squared': float(bias.square().mean()),
            'centered_mse': float(centered.mean()), 'signed_bias': float(bias.mean()),
            'accounting_max_diff': float((mse - bias.square() - centered).abs().max()), **gradients}


@torch.inference_mode()
def main():
    plan = checked(); torch.set_num_threads(4); rows = []; consistency_rows = []
    for meta in plan['catalog']:
        data = load_record(meta); raw, target = data['raw'], data['target']
        maps = torch.load(OUT / 'evaluation' / (meta['id'] + '_maps.pt'), map_location='cpu', weights_only=False)
        base = {'episode': meta['id'], 'scene': meta['row']['scene'], 'split': meta['eval_split']}
        qs = maps['full_query']; indices = [maps['query'].tolist().index(q) for q in qs]
        camera, K, unit = raw['c2w'][qs], raw['K'][qs], raw['diagnostics']['unit_meters']
        y, valid = target['log_depth_m'][qs], target['valid'][qs]
        gt = world_points(y.exp(), camera, K, unit); pairs = maps['correspondences']
        for name, pred in maps['predictions'].items():
            if name in ('seed_center', 'seed_gaussian', 'train_mean'):
                variant, seed, condition = name, None, 'normal'
            else:
                variant, seed, condition = name.split('_', 2); seed = int(seed)
            rows.append(dict(base, variant=variant, seed=seed, condition=condition, input='masked',
                             **decompose(pred, maps['target'], maps['valid'])))
            consistency_rows.append(dict(base, variant=variant, seed=seed, condition=condition,
                **consistency(pred[indices], gt, pairs, camera, K, unit)))
        for name, pred in maps['full_predictions'].items():
            variant, seed = name.split('_')
            rows.append(dict(base, variant=variant, seed=int(seed), condition='normal', input='full',
                             **decompose(pred, y, valid)))
    main_rows = [r for r in rows if r['split'] in ('used_clip40', 'used_clip80')]
    main_consistency = [r for r in consistency_rows if r['split'] in ('used_clip40', 'used_clip80')]
    scenes = sorted({r['scene'] for r in main_rows}); summary = {}; comparisons = {}
    keys = ('log_mse', 'bias_squared', 'centered_mse', 'signed_bias', 'prediction_gradient_energy', 'target_gradient_energy')
    for source in ('masked', 'full'):
        selected = [r for r in main_rows if r['input'] == source and r['condition'] == 'normal']
        summary[source] = {v: {key: float(matrix(selected, v, key, scenes).mean()) for key in keys} for v in VARIANTS}
        comparisons[source] = {}
        for metric in ('bias_squared', 'centered_mse'):
            values = {v: matrix(selected, v, metric, scenes) for v in VARIANTS}
            comparisons[source][metric] = {}
            for left, right in [('local', 'gaussian'), ('fixed', 'gaussian'), ('center', 'gaussian')]:
                result = estimate(values[left]-values[right], scenes)
                result['relative_reduction_percent'] = 100 * result['mean_gain'] / values[left].mean()
                comparisons[source][metric][left+'_minus_'+right] = result
    coherence = {}; interventions = {}
    for v in VARIANTS:
        conditions = ['normal'] if v == 'local' else ['normal','removed','swapped','shuffled','initial_geometry','wrong_camera']
        normal = matrix(main_consistency, v, 'residual_xyz_mse', scenes)
        coherence[v] = {}; interventions[v] = {}
        for c in conditions:
            value = matrix(main_consistency, v, 'residual_xyz_mse', scenes, c)
            coherence[v][c] = float(value.mean())
            result = estimate(value-normal, scenes); result['relative_error_increase_percent'] = 100 * result['mean_gain'] / normal.mean()
            interventions[v][c] = result
    for v in ('seed_center','seed_gaussian','train_mean'):
        coherence[v] = {'normal': float(np.mean([np.mean([r['residual_xyz_mse'] for r in main_consistency if r['variant']==v and r['scene']==s]) for s in scenes]))}
    accounting = max(r['accounting_max_diff'] for r in rows); assert accounting < 1e-10
    write_json(OUT / 'frozen_diagnostic.json', {'prospective_addendum_sha256': sha256(OUT / 'frozen_diagnostic_plan.md'),
               'source_sha256': sha256(__file__), 'predictions_modified': False, 'training_added': False,
               'post_result_diagnostic': True, 'confirmation_claim': False, 'summary': summary, 'comparisons': comparisons,
               'masked_first_sample_consistency': coherence, 'consistency_interventions': interventions,
               'accounting_max_diff': accounting, 'rows': rows, 'consistency_rows': consistency_rows})
    print(json.dumps({'summary': summary, 'masked_consistency': coherence,
                      'decomposition': {source: {k: {name: {z: value[z] for z in ('relative_reduction_percent','paired_scene_seed_bootstrap_95','scenes_better')}
                                                      for name, value in vals.items()} for k, vals in metrics.items()}
                                        for source, metrics in comparisons.items()}}))


if __name__ == '__main__': main()
