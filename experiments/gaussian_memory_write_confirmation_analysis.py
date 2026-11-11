"""Unchanged metric definitions, scene-paired new-place estimates and bounded checks."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.gaussian_memory_attention_train import write_json
from experiments.gaussian_memory_setup import sha256
from experiments.gaussian_memory_coverage_report import estimate
from experiments.gaussian_memory_joint_write_analysis import METRICS, PAIRS, matrix
from experiments.gaussian_memory_write_confirmation import OUT, RAW, CACHE, ACCESS, INVENTORY, checked
from experiments import gaussian_memory_joint_write as joint
from experiments import gaussian_memory_summary_data as summary


def analysis():
    plan = checked(); data = json.loads((OUT / 'evaluation/scores.json').read_text())
    rows, full = data['rows'], data['full_query']; scenes = sorted({r['scene'] for r in rows})
    means = {k: {} for k in joint.KINDS}; comparisons = {}; interventions = {}; updates = {}
    for metric in METRICS:
        source, key = (full, metric.removeprefix('full_')) if metric.startswith('full_') else (rows, metric)
        values = {kind: matrix(source, kind, key, scenes) for kind in joint.KINDS}
        assert all(np.isfinite(value).all() for value in values.values())
        for kind, value in values.items():
            means[kind][metric] = float(value.mean())
        comparisons[metric] = {}
        for left, right in PAIRS:
            effect = estimate(values[left] - values[right], scenes)
            effect['relative_reduction_percent'] = 100 * effect['mean_gain'] / values[left].mean()
            comparisons[metric][left + '_minus_' + right] = effect
    for kind in joint.NEW:
        normal = matrix(rows, kind, 'log_mse', scenes); interventions[kind] = {}
        for condition in ('removed', 'shuffled', 'wrong_camera', 'initial_geometry', 'initial_memory'):
            effect = estimate(matrix(rows, kind, 'log_mse', scenes, condition) - normal, scenes)
            effect['relative_error_increase_percent'] = 100 * effect['mean_gain'] / normal.mean()
            interventions[kind][condition] = effect
        updates[kind] = {key: float(matrix(data['updates'], kind, key, scenes).mean()) for key in
                         ('center_displacement', 'center_displacement_m', 'covariance_relative_change',
                          'opacity_abs_change', 'feature_delta_rms')}
    per_source = {}
    for prefix in plan['quotas']:
        subset = [s for s in scenes if s.startswith(prefix + '_')]
        if subset:
            per_source[prefix] = {'places': len(subset), 'summary':
                {k: {m: float(matrix(rows, k, m, subset).mean()) for m in ('log_mse', 'bias_squared', 'centered_mse')}
                 for k in joint.KINDS}}
    result = {'summary': means, 'comparisons': comparisons, 'interventions': interventions, 'updates': updates,
              'scenes': scenes, 'places': len(scenes), 'planned_places': plan['places'], 'per_source': per_source,
              'confirmation_claim': True, 'freshness_scope': plan['fresh_scope'], 'historical_data_pooled': False}
    write_json(OUT / 'analysis.json', result)
    print(json.dumps({'places': len(scenes), 'means': {k: v['log_mse'] for k, v in means.items()},
        'main_comparisons': {k: {f: v[f] for f in ('relative_reduction_percent', 'paired_scene_seed_bootstrap_95',
            'paired_seed_gain', 'scenes_better', 'scenes_total')} for k, v in comparisons['log_mse'].items()},
        'space_comparisons': {k: comparisons['centered_mse'][k] for k in
                              ('gaussian_minus_full_gaussian', 'large_local_minus_full_gaussian')}}), flush=True)


def independent_reads(catalog, maps_dir, memory_dir, plan, old=False):
    reads = []; scores = []; seed = joint.SEEDS[0]
    with torch.inference_mode():
        for meta in catalog:
            d = summary.load_record(meta); raw, rec = d['raw'], d['rec']; rid = meta['id']
            stored = torch.load(maps_dir / (rid + '_maps.pt'), map_location='cpu', weights_only=False)
            keep = torch.tensor([r == 'evaluation' for r in rec['roles']]); q = rec['query'][keep]
            x = rec['input_tokens'][keep]
            for kind in joint.KINDS:
                net, _ = joint.make_models(kind, plan); net.eval()
                net.load_state_dict(torch.load(joint.model_path(kind, seed), map_location='cpu', weights_only=False)['model'])
                memory = (torch.load(memory_dir / f'{rid}_{kind}_{seed}.pt', map_location='cpu', weights_only=False)['memory']
                          if kind in joint.NEW else d['memory'])
                p = net.read(x, memory, raw['c2w'][q], raw['K'][q], raw['diagnostics']['unit_meters'])
                saved = stored['predictions'][f'{kind}_{seed}_normal']
                diff = float((p - saved).abs().max()); assert diff < 2e-5
                reads.append({'episode': rid, 'kind': kind, 'seed': seed, 'old_sentinel': old, 'max_difference': diff})
                error = saved.numpy().astype('float64') - stored['target'].numpy().astype('float64')
                valid = stored['valid'].numpy()
                independent = float(np.mean([np.mean(e[v] ** 2) for e, v in zip(error, valid)]))
                scores.append({'episode': rid, 'kind': kind, 'seed': seed, 'old_sentinel': old, 'numpy_log_mse': independent})
    return reads, scores


def checks():
    plan = checked(); torch.set_num_threads(4)
    cache = json.loads((OUT / 'cache_identity.json').read_text()); scores = json.loads((OUT / 'evaluation/scores.json').read_text())
    download = json.loads((OUT / 'download_complete.json').read_text()); cohort = json.loads((OUT / 'cohort.json').read_text())
    n = len(cache['catalog']); assert n == len(download['usable_cohort'])
    assert n + len(cohort['metadata_failures']) + len(download['unavailable']) == 20
    assert len(scores['rows']) == n * 69 and len(scores['full_query']) == n * 24
    assert len(scores['memory_sha256']) == n * 9
    for group in (cache['sha256'], scores['memory_sha256'], scores['maps_sha256']):
        for name, digest in group.items():
            assert sha256(name) == digest, name
    fresh = {m['row']['scene'].split('/')[0] for m in cache['catalog']}
    assert fresh <= set(plan['places']) and not fresh & set(plan['excluded_places'])
    assert all(set(m['row']['reference']).isdisjoint(m['row']['heldout_query']) for m in cache['catalog'])
    assert all(not m['row']['training_query'] for m in cache['catalog'])
    assert all(np.isfinite(r['log_mse']) for r in scores['rows'] + scores['full_query'])
    assert all(r['split'] == 'fresh_confirmation' for r in scores['rows'] + scores['full_query'])
    probes = [cache['catalog'][i] for i in sorted({0, n // 2, n - 1})]
    reads, independent = independent_reads(probes, OUT / 'evaluation', OUT / 'memories', plan)
    old_plan = json.loads((joint.OUT / 'plan.json').read_text())
    probes = [next(m for m in old_plan['catalog'] if m['id'] == rid) for rid in ('fit_00', 'confirm_00', 'confirm_15')]
    old_reads, old_independent = independent_reads(probes, joint.OUT / 'evaluation', joint.OUT / 'memories', plan, True)
    old_scores = json.loads((joint.OUT / 'evaluation/scores.json').read_text())
    differences = []
    for r in independent + old_independent:
        source = old_scores if r['old_sentinel'] else scores
        target = next(s for s in source['rows'] if s['kind'] == r['kind'] and s['episode'] == r['episode'] and
                      s['seed'] == r['seed'] and s['condition'] == 'normal')
        delta = abs(r['numpy_log_mse'] - target['log_mse']); assert delta < 2e-6
        differences.append(delta)
    result = {'complete': True, 'models_unchanged': len(plan['models']), 'new_fits': 0,
              'planned_places': 20, 'evaluated_places': n, 'zero_exclusions': n == 20,
              'new_directory_identities_disjoint': True, 'physical_alias_independence_guaranteed': False,
              'saved_memories': len(scores['memory_sha256']), 'queries': sum(r['queries'] for r in scores['counts']),
              'masked_inputs': sum(r['masked_inputs'] for r in scores['counts']),
              'independent_reads': reads, 'old_sentinel_reads': old_reads,
              'independent_CPU_max_difference': max(r['max_difference'] for r in reads),
              'old_sentinel_max_difference': max(r['max_difference'] for r in old_reads),
              'independent_numpy_max_difference': max(differences),
              'accounting_max_difference': max(r['accounting_max_diff'] for r in scores['rows'] + scores['full_query'] if r['condition'] == 'normal'),
              'minimum_covariance_eigenvalue': min(r['minimum_eigenvalue'] for r in scores['updates']),
              'source_sha256': sha256(__file__)}
    write_json(OUT / 'checks.json', result)
    access = json.loads(ACCESS.read_text()); access.update(evaluation_complete=True, evaluated_places=sorted(fresh),
        cache_identity_sha256=sha256(OUT / 'cache_identity.json'), scores_sha256=sha256(OUT / 'evaluation/scores.json'))
    write_json(ACCESS, access)
    previous_inventory = json.loads(INVENTORY.read_text())
    remaining = sorted(set(previous_inventory['remaining_directory_candidates']) - set(plan['places']))
    write_json(OUT.parents[1] / 'research_data_usage/sun3d_remaining_after_shared_writer_confirmation_20261110.json',
        {'excluded_union': sorted(set(plan['excluded_places']) | set(plan['places'])),
         'remaining_directory_candidates': remaining, 'remaining_candidate_count': len(remaining),
         'source_sha256': {str(ACCESS): sha256(ACCESS), str(OUT / 'plan.json'): sha256(OUT / 'plan.json')},
         'physical_alias_independence_not_guaranteed': True})
    print(json.dumps({k: v for k, v in result.items() if k not in ('independent_reads', 'old_sentinel_reads')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=('analysis', 'checks'))
    globals()[parser.parse_args().stage]()
