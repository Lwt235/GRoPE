"""Paired training-state effects, independent PE utility and state range."""
import argparse
import json

import numpy as np

from experiments.gaussian_memory_state_data import OUT, SEEDS, QUERIES, POLICIES, load_plan, write_json
from experiments.gaussian_memory_incremental_analysis import cohorts, case_matrix
from experiments.gaussian_memory_bound_analysis import compare_records


def read_records(phase, case, catalog):
    result = {}
    for meta in catalog:
        path = OUT / 'evaluation' / phase / case / f'{meta["id"]}.json'
        record = json.loads(path.read_text()); assert record['complete'], path
        result[meta['id']] = record['rows']
    return result


def cases_complete(phase, cases, catalog):
    return all((OUT / 'evaluation' / phase / case / f'{m["id"]}.json').is_file() for case in cases for m in catalog)


def state_records(policy, catalog):
    records = {}
    for kind in ('content', 'soft'):
        records[f'{kind}_dynamictrain'] = [read_records('read', f'{kind}_{s}_{policy}', catalog) for s in SEEDS]
        records[f'{kind}_statictrain'] = [read_records('reused', f'{kind}_{s}_{policy}', catalog) for s in SEEDS]
    records['local'] = [read_records('reused', f'local_{s}', catalog) for s in SEEDS]
    records['native'] = [read_records('reused', 'native', catalog)] * len(SEEDS)
    if policy != 'rebuild_clean' and cases_complete('read', [f'soft_center_{s}_{policy}' for s in SEEDS], catalog):
        records['soft_center_dynamictrain'] = [read_records('read', f'soft_center_{s}_{policy}', catalog) for s in SEEDS]
    return records


def interaction(records, catalog, mode, metric):
    values = {k: case_matrix(records[k], catalog, metric, mode)[0] for k in (
        'content_dynamictrain', 'soft_dynamictrain', 'content_statictrain', 'soft_statictrain')}
    delta = ((values['content_dynamictrain'] - values['soft_dynamictrain'])
             - (values['content_statictrain'] - values['soft_statictrain']))
    assert np.isfinite(delta).all(); places, seeds = delta.shape
    rng = np.random.default_rng(20261110)
    pi = rng.integers(places, size=(5000, places)); si = rng.integers(seeds, size=(5000, seeds))
    boot = delta[pi[:, :, None], si[:, None, :]].mean((1, 2))
    return {'definition': '(C_dynamic-S_dynamic)-(C_static-S_static), same evaluation state',
            'positive_means': 'state-matched training increases the Soft advantage over C',
            'mean_gain': float(delta.mean()), 'paired_seed_gain': delta.mean(0).tolist(),
            'paired_place_seed_95': np.quantile(boot, [.025, .975]).tolist(),
            'place_gain': delta.mean(1).tolist(), 'places': places, 'seeds': seeds}


def analyze_policy(policy):
    plan = load_plan(); records = state_records(policy, plan['catalog'])
    pairs = [('content_dynamictrain', 'soft_dynamictrain'), ('content_statictrain', 'content_dynamictrain'),
             ('soft_statictrain', 'soft_dynamictrain'), ('content_statictrain', 'soft_statictrain'),
             ('content_statictrain', 'soft_dynamictrain'),
             ('native', 'content_dynamictrain'), ('native', 'soft_dynamictrain'),
             ('local', 'content_dynamictrain'), ('local', 'soft_dynamictrain')]
    if 'soft_center_dynamictrain' in records:
        pairs += [('soft_center_dynamictrain', 'soft_dynamictrain'), ('content_dynamictrain', 'soft_center_dynamictrain')]
    result = {'policy': policy, 'all_data_previously_used': True,
              'center_evaluated': 'soft_center_dynamictrain' in records, 'cohorts': {}}
    for group, subset in cohorts(plan).items():
        g = {'places': sorted({m['place'] for m in subset}), 'clips': len(subset), 'modes': {}}
        for mode in ('mixed', 'full', 'masked'):
            metrics = ('log_mse', 'centered_log_mse', 'bias_squared', 'absrel', 'rmse_m')
            if mode == 'masked': metrics += ('hidden_log_mse', 'visible_log_mse')
            g['modes'][mode] = {}
            for metric in metrics:
                value = compare_records(records, subset, pairs, mode, metric)
                value['training_state_interaction'] = interaction(records, subset, mode, metric)
                g['modes'][mode][metric] = value
        result['cohorts'][group] = g
    result['per_query_primary51'] = {mode: [{
        'query': q, **compare_records(records, cohorts(plan)['primary51'], pairs, mode, 'log_mse',
                                     filter_row=lambda row, q=q: row['query'] == q)} for q in QUERIES]
        for mode in ('mixed', 'full', 'masked')}
    write_json(OUT / f'analysis_{policy}.json', result)
    main = result['cohorts']['primary51']['modes']['mixed']['log_mse']
    print(json.dumps({'policy': policy, 'primary_means': main['means'],
                      'pe_relative_reduction_percent': main['comparisons']['content_dynamictrain_minus_soft_dynamictrain']['relative_reduction_percent']}), flush=True)
    return result


def conditional_decision():
    path = OUT / 'analysis_corrective_clean.json'
    main = json.loads(path.read_text())['cohorts']['primary51']['modes']['mixed']['log_mse']
    comparison = main['comparisons']['content_dynamictrain_minus_soft_dynamictrain']
    triggered = comparison['relative_reduction_percent'] >= 1. and sum(x > 0 for x in comparison['paired_seed_gain']) >= 2
    value = {'center_control_triggered': bool(triggered), 'comparison': comparison,
             'rule': load_plan()['center_trigger'], 'scope': 'minimum development signal for three extra fits, not method success',
             'decision': 'run three dynamic Soft-center fits and Static/Corrective evaluation' if triggered else 'do not run Center: prespecified minimum signal unmet'}
    target = OUT / 'conditional_decision.json'
    if target.exists(): assert json.loads(target.read_text()) == value
    else: write_json(target, value)
    print(json.dumps({'center_control_triggered': bool(triggered)}), flush=True)
    return value


def state_range():
    plan = load_plan(); records = {}; pairs = []
    for kind in ('content', 'soft'):
        for policy in POLICIES:
            records[f'{kind}_{policy}'] = [read_records('read', f'{kind}_{s}_{policy}', plan['catalog']) for s in SEEDS]
        pairs += [(f'{kind}_static', f'{kind}_corrective_clean'), (f'{kind}_static', f'{kind}_rebuild_clean')]
    result = {'all_models_trained_on_corrective_states': True, 'cohorts': {}}
    for group, subset in cohorts(plan).items():
        result['cohorts'][group] = {mode: {metric: compare_records(records, subset, pairs, mode, metric)
            for metric in ('log_mse', 'centered_log_mse', 'bias_squared')} for mode in ('mixed', 'full', 'masked')}
    write_json(OUT / 'state_range_analysis.json', result)
    return result


def interventions():
    plan = load_plan(); catalog = plan['catalog']; seed = SEEDS[0]
    records = {'soft': [read_records('read', f'soft_{seed}_corrective_clean', catalog)]}
    for name in ('content_permuted', 'uniform_prior'):
        records[name] = [read_records('intervention', f'soft_{seed}_{name}', catalog)]
    pairs = [(name, 'soft') for name in ('content_permuted', 'uniform_prior')]
    result = {'fixed_seed': seed, 'state': 'corrective_clean', 'scope': 'within-model dependence, not utility',
              'positive_gain_definition': 'perturbed error minus intact error; relative denominator is perturbed error', 'cohorts': {}}
    for group, subset in cohorts(plan).items():
        result['cohorts'][group] = {mode: {metric: compare_records(records, subset, pairs, mode, metric)
            for metric in ('log_mse', 'centered_log_mse', 'bias_squared')} for mode in ('mixed', 'full', 'masked')}
    write_json(OUT / 'intervention_analysis.json', result)
    return result


def retention():
    plan = load_plan(); catalog = plan['catalog']; records = {}; pairs = []
    for kind in ('content', 'soft'):
        data = read_records('retention', f'{kind}_{SEEDS[0]}', catalog)
        for ti in (0, 1, 4):
            records[f'{kind}_t{ti}'] = [{rid: [r for r in rows if r['state_key'][1] == ti] for rid, rows in data.items()}]
        pairs += [(f'{kind}_t0', f'{kind}_t1'), (f'{kind}_t1', f'{kind}_t4')]
    pairs += [(f'content_t{ti}', f'soft_t{ti}') for ti in (0, 1, 4)]
    result = {'fixed_seed': SEEDS[0], 'query': 5, 'condition': 0, 'retrospective': True, 'cohorts': {}}
    for group, subset in cohorts(plan).items():
        result['cohorts'][group] = {metric: compare_records(records, subset, pairs, 'full', metric)
                                   for metric in ('log_mse', 'centered_log_mse', 'bias_squared')}
    write_json(OUT / 'retention_analysis.json', result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=('primary', 'all', 'decision'))
    args = p.parse_args()
    if args.stage == 'primary': analyze_policy('corrective_clean'); conditional_decision()
    elif args.stage == 'decision': conditional_decision()
    else:
        for policy in POLICIES: analyze_policy(policy)
        conditional_decision(); state_range(); interventions(); retention()
