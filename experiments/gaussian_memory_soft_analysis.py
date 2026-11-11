"""Place-paired results and prespecified conditional control for soft_read1."""
import argparse
import json

import numpy as np

from experiments.gaussian_memory_setup import sha256
from experiments.gaussian_memory_soft_data import OUT, PARENT_OUT, SEEDS, QUERIES, load_plan, write_json
from experiments.gaussian_memory_incremental_analysis import cohorts, case_matrix, matched_gain
from experiments.gaussian_memory_bound_analysis import compare_records, favorable


def read_records(phase, case, catalog):
    result = {}
    for meta in catalog:
        path = OUT / 'evaluation' / phase / case / f'{meta["id"]}.json'
        record = json.loads(path.read_text()); assert record['complete'], path
        result[meta['id']] = record['rows']
    return result


def cases_complete(phase, cases, catalog):
    return all((OUT / 'evaluation' / phase / case / f'{meta["id"]}.json').is_file()
               for case in cases for meta in catalog)


def static_analysis():
    plan = load_plan(); catalog = plan['catalog']; groups = cohorts(plan)
    records = {k: [read_records('static', f'{k}_{s}', catalog) for s in SEEDS]
               for k in ('local', 'content', 'coverage', 'bound', 'soft')}
    native = read_records('static', 'native', catalog); records['native'] = [native] * len(SEEDS)
    center = cases_complete('static', [f'soft_center_{s}' for s in SEEDS], catalog)
    if center: records['soft_center'] = [read_records('static', f'soft_center_{s}', catalog) for s in SEEDS]
    pairs = [('content', 'soft'), ('local', 'soft'), ('native', 'soft'), ('bound', 'soft'),
             ('coverage', 'soft'), ('local', 'content'), ('native', 'content')]
    if center: pairs += [('soft_center', 'soft'), ('content', 'soft_center')]
    result = {'all_data_previously_used': True, 'center_evaluated': center, 'cohorts': {}}
    for group, subset in groups.items():
        g = {'places': sorted({m['place'] for m in subset}), 'clips': len(subset), 'modes': {}}
        for mode in ('mixed', 'full', 'masked'):
            metrics = ('log_mse', 'centered_log_mse', 'bias_squared', 'absrel', 'rmse_m')
            if mode == 'masked': metrics += ('hidden_log_mse', 'visible_log_mse')
            g['modes'][mode] = {metric: compare_records(records, subset, pairs, mode, metric) for metric in metrics}
        result['cohorts'][group] = g
    main = result['cohorts']['primary51']['modes']['mixed']['log_mse']
    result['static_center_trigger_condition'] = favorable(main['comparisons']['content_minus_soft'])
    result['conditional_decision_note'] = 'Final Center decision also considers Corrective; no promotion claim.'
    write_json(OUT / 'static_analysis.json', result)
    print(json.dumps({'static_analysis_complete': True, 'primary': main}), flush=True)
    return result


def integrated_analysis():
    plan = load_plan(); catalog = plan['catalog']; groups = cohorts(plan)
    result = {'cohorts': {}, 'readers_fixed': True, 'states_unchanged': True,
              'scope': 'Current matched C and Soft architectures, same preexisting causal states; all previously used data'}
    records = {}; pairs = []
    for k in ('content', 'soft'):
        records[f'{k}_static'] = [read_records('static', f'{k}_{s}', catalog) for s in SEEDS]
        for policy in ('corrective_clean', 'rebuild_clean'):
            records[f'{k}_{policy}'] = [read_records('integrated', f'{k}_{s}_{policy}', catalog) for s in SEEDS]
        pairs += [(f'{k}_static', f'{k}_corrective_clean'), (f'{k}_static', f'{k}_rebuild_clean'),
                  (f'{k}_rebuild_clean', f'{k}_corrective_clean')]
    pairs += [(f'content_{p}', f'soft_{p}') for p in ('static', 'corrective_clean', 'rebuild_clean')]
    center = cases_complete('integrated', [f'soft_center_{s}_corrective_clean' for s in SEEDS], catalog)
    if center:
        records['soft_center_corrective_clean'] = [read_records('integrated', f'soft_center_{s}_corrective_clean', catalog) for s in SEEDS]
        pairs += [('soft_center_corrective_clean', 'soft_corrective_clean'),
                  ('content_corrective_clean', 'soft_center_corrective_clean')]
    result['center_evaluated'] = center
    for group, subset in groups.items():
        result['cohorts'][group] = {mode: {metric: compare_records(records, subset, pairs, mode, metric)
            for metric in ('log_mse', 'centered_log_mse', 'bias_squared')}
            for mode in ('mixed', 'full', 'masked')}
    result['causal_primary51'] = {mode: [
        {'write_index': ti, 'query': q, **compare_records(records, groups['primary51'], pairs, mode,
                                                       filter_row=lambda row, q=q: row['query'] == q)}
        for ti, q in enumerate(QUERIES, 1)] for mode in ('mixed', 'full', 'masked')}
    main = result['cohorts']['primary51']['mixed']['log_mse']
    result['corrective_center_trigger_condition'] = favorable(main['comparisons']['content_corrective_clean_minus_soft_corrective_clean'])
    write_json(OUT / 'integrated_analysis.json', result)
    print(json.dumps({'integrated_analysis_complete': True, 'primary': main}), flush=True)
    return result


def conditional_decision():
    static = json.loads((OUT / 'static_analysis.json').read_text())
    dynamic = json.loads((OUT / 'integrated_analysis.json').read_text())
    a = static['cohorts']['primary51']['modes']['mixed']['log_mse']['comparisons']['content_minus_soft']
    b = dynamic['cohorts']['primary51']['mixed']['log_mse']['comparisons']['content_corrective_clean_minus_soft_corrective_clean']
    triggered = favorable(a) or favorable(b)
    result = {'center_control_triggered': bool(triggered), 'static_condition': favorable(a),
              'corrective_condition': favorable(b), 'static_comparison': a, 'corrective_comparison': b,
              'rule': load_plan()['center_trigger'], 'scope': 'developmental control selection, not method promotion',
              'decision': 'run three fixed Soft-center fits and Static/Corrective evaluations' if triggered
                          else 'do not run Center: both prespecified mean-and-seed-direction conditions unmet'}
    target = OUT / 'conditional_decision.json'
    if target.exists(): assert json.loads(target.read_text()) == result
    else: write_json(target, result)
    print(json.dumps({'center_control_triggered': triggered}), flush=True)
    return result


def intervention_analysis():
    plan = load_plan(); catalog = plan['catalog']; seed = SEEDS[0]
    records = {'soft': [read_records('static', f'soft_{seed}', catalog)]}
    for name in ('content_permuted', 'uniform_prior'):
        records[name] = [read_records('intervention', f'soft_{seed}_{name}', catalog)]
    pairs = [(name, 'soft') for name in ('content_permuted', 'uniform_prior')]
    result = {'fixed_seed': seed, 'scope': 'within-model dependence, not matched-training utility',
              'positive_gain_definition': 'perturbed error minus intact error; relative denominator is perturbed error', 'cohorts': {}}
    for group, subset in cohorts(plan).items():
        result['cohorts'][group] = {mode: {metric: compare_records(records, subset, pairs, mode, metric)
            for metric in ('log_mse', 'centered_log_mse', 'bias_squared')}
            for mode in ('mixed', 'full', 'masked')}
    result['trained_interface'] = json.loads((OUT / 'binding_interface.json').read_text())
    write_json(OUT / 'intervention_analysis.json', result)
    return result


def retention_analysis():
    plan = load_plan(); catalog = plan['catalog']; records = {}; pairs = []
    for k in ('content', 'soft'):
        data = read_records('retention', f'{k}_{SEEDS[0]}', catalog)
        for t in (0, 1, 4):
            records[f'{k}_t{t}'] = [{rid: [r for r in rows if r['state_key'][1] == t] for rid, rows in data.items()}]
        pairs += [(f'{k}_t0', f'{k}_t1'), (f'{k}_t1', f'{k}_t4')]
    pairs += [(f'content_t{t}', f'soft_t{t}') for t in (0, 1, 4)]
    result = {'fixed_seed': SEEDS[0], 'query': 5, 'condition': 0, 'state_indices': [0, 1, 4],
              'retrospective': True, 'not_causal_time_performance': True, 'cohorts': {}}
    for group, subset in cohorts(plan).items():
        result['cohorts'][group] = {metric: compare_records(records, subset, pairs, 'full', metric)
                                   for metric in ('log_mse', 'centered_log_mse', 'bias_squared')}
    write_json(OUT / 'retention_analysis.json', result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=('static', 'integrated', 'decision', 'interventions', 'retention'))
    args = p.parse_args()
    {'static': static_analysis, 'integrated': integrated_analysis, 'decision': conditional_decision,
     'interventions': intervention_analysis, 'retention': retention_analysis}[args.stage]()
