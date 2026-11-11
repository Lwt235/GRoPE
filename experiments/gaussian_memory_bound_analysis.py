"""Place-paired utility, binding dependence, and fixed-reader state results."""
import argparse
import json

import numpy as np

from experiments.gaussian_memory_setup import sha256
from experiments.gaussian_memory_bound_data import OUT, BASE_OUT, SEEDS, KINDS, QUERIES, load_plan, write_json
from experiments.gaussian_memory_incremental_analysis import cohorts, case_matrix, matched_gain
from experiments.gaussian_memory_incremental_states import AMPLITUDES, track_name


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


def compare_records(records, catalog, pairs, mode='mixed', metric='log_mse', filter_row=None):
    values = {k: case_matrix(v, catalog, metric, mode, filter_row)[0] for k, v in records.items()}
    return {'means': {k: float(v.mean()) for k, v in values.items()},
            'comparisons': {f'{a}_minus_{b}': matched_gain(values[a], values[b]) for a, b in pairs}}


def favorable(comparison):
    return bool(comparison['mean_gain'] > 0 and sum(x > 0 for x in comparison['paired_seed_gain']) >= 2)


def static_analysis():
    plan = load_plan(); groups = cohorts(plan); catalog = plan['catalog']
    records = {k: [read_records('static', f'{k}_{s}', catalog) for s in SEEDS] for k in KINDS}
    native = read_records('static', 'native', catalog); records['native'] = [native] * len(SEEDS)
    center_present = cases_complete('static', [f'center_{s}' for s in SEEDS], catalog)
    if center_present:
        records['center'] = [read_records('static', f'center_{s}', catalog) for s in SEEDS]
    pairs = [('coverage', 'bound'), ('local', 'bound'), ('content', 'bound'), ('content', 'coverage'),
             ('native', 'bound'), ('native', 'local'), ('local', 'content'), ('local', 'coverage')]
    if center_present:
        pairs += [('center', 'bound'), ('coverage', 'center'), ('local', 'center')]
    result = {'all_data_previously_used': True, 'center_evaluated': center_present, 'cohorts': {}}
    for group, subset in groups.items():
        g = {'places': sorted({m['place'] for m in subset}), 'clips': len(subset), 'modes': {}}
        for mode in ('mixed', 'full', 'masked'):
            metrics = ('log_mse', 'centered_log_mse', 'bias_squared', 'absrel', 'rmse_m')
            if mode == 'masked': metrics += ('hidden_log_mse', 'visible_log_mse')
            g['modes'][mode] = {metric: compare_records(records, subset, pairs, mode, metric) for metric in metrics}
        result['cohorts'][group] = g
    comparisons = result['cohorts']['primary51']['modes']['mixed']['log_mse']['comparisons']
    result['center_control_triggered'] = favorable(comparisons['coverage_minus_bound'])
    result['error_probe_triggered'] = bool(result['center_control_triggered'] and favorable(comparisons['local_minus_bound']))
    result['conditional_decisions'] = {
        'center': 'run three matched Center fits' if result['center_control_triggered'] else 'do not run: B/A mean and seed-direction condition unmet',
        'error_and_retention': 'run first-seed B native probes on existing trajectories' if result['error_probe_triggered']
            else 'do not run: joint B/A and B/L condition unmet; previous correction/retention negatives remain applicable to the previous reader only'}
    write_json(OUT / 'static_analysis.json', result)
    print(json.dumps({'static_analysis_complete': True,
                      'center_control_triggered': result['center_control_triggered'],
                      'error_probe_triggered': result['error_probe_triggered'],
                      'primary_means': result['cohorts']['primary51']['modes']['mixed']['log_mse']['means']}), flush=True)
    return result


def intervention_analysis():
    plan = load_plan(); catalog = plan['catalog']; seed = SEEDS[0]
    records = {'bound': [read_records('static', f'bound_{seed}', catalog)]}
    for name in ('content_permuted', 'coverage_prior'):
        records[name] = [read_records('intervention', f'bound_{seed}_{name}', catalog)]
    # Positive differences mean the perturbed input has larger error than intact B.
    pairs = [(name, 'bound') for name in ('content_permuted', 'coverage_prior')]
    result = {'fixed_seed': seed, 'scope': 'within-model dependence, not matched-training comparative utility', 'cohorts': {}}
    for group, subset in cohorts(plan).items():
        result['cohorts'][group] = {mode: {metric: compare_records(records, subset, pairs, mode, metric)
            for metric in ('log_mse', 'centered_log_mse', 'bias_squared')}
            for mode in ('mixed', 'full', 'masked')}
    result['trained_interface'] = json.loads((OUT / 'binding_interface.json').read_text())
    write_json(OUT / 'intervention_analysis.json', result)
    return result


def integrated_analysis():
    plan = load_plan(); catalog = plan['catalog']; groups = cohorts(plan)
    result = {'cohorts': {}, 'readers_fixed': True, 'new_fits': 0, 'states_unchanged': True,
              'state_analysis_reused_from': str(BASE_OUT / 'state_analysis.json'),
              'state_analysis_sha256': sha256(BASE_OUT / 'state_analysis.json')}
    records = {}; pairs = []
    for k in ('coverage', 'bound'):
        records[f'{k}_static'] = [read_records('static', f'{k}_{s}', catalog) for s in SEEDS]
        for policy in ('corrective_clean', 'rebuild_clean'):
            records[f'{k}_{policy}'] = [read_records('integrated', f'{k}_{s}_{policy}', catalog) for s in SEEDS]
        pairs += [(f'{k}_static', f'{k}_corrective_clean'), (f'{k}_static', f'{k}_rebuild_clean'),
                  (f'{k}_rebuild_clean', f'{k}_corrective_clean')]
    pairs += [('coverage_corrective_clean', 'bound_corrective_clean'), ('coverage_rebuild_clean', 'bound_rebuild_clean')]
    for group, subset in groups.items():
        result['cohorts'][group] = {mode: {metric: compare_records(records, subset, pairs, mode, metric)
            for metric in ('log_mse', 'centered_log_mse', 'bias_squared')}
            for mode in ('mixed', 'full', 'masked')}
    result['causal_primary51'] = {mode: [
        {'write_index': ti, 'query': q, **compare_records(records, groups['primary51'], pairs, mode,
                                                      filter_row=lambda row, q=q: row['query'] == q)}
        for ti, q in enumerate(QUERIES, 1)] for mode in ('mixed', 'full', 'masked')}
    write_json(OUT / 'integrated_analysis.json', result)
    print(json.dumps({'integrated_analysis_complete': True}), flush=True)
    return result


def error_analysis():
    decision = json.loads((OUT / 'static_analysis.json').read_text())
    if not decision['error_probe_triggered']:
        result = {'triggered': False, 'reason': decision['conditional_decisions']['error_and_retention'],
                  'new_reader_correction_or_retention_claim': False,
                  'previous_negative_results': str(BASE_OUT / 'integrated_analysis.json')}
        write_json(OUT / 'error_analysis.json', result)
        return result
    plan = load_plan(); catalog = plan['catalog']; primary = cohorts(plan)['primary51']
    case = f'bound_{SEEDS[0]}'
    error = read_records('error_probe', case, catalog); retention = read_records('retention', case, catalog)
    summaries = {}; error_curves = {}
    for amp in AMPLITUDES:
        arrays = {st: case_matrix([error], primary,
            filter_row=lambda r, n=track_name(st, amp): r['state_key'][0] == n)[0]
            for st in ('static', 'average', 'corrective')}
        summaries[str(amp)] = {'means': {k: float(x.mean()) for k, x in arrays.items()},
            'static_minus_corrective': matched_gain(arrays['static'], arrays['corrective']),
            'average_minus_corrective': matched_gain(arrays['average'], arrays['corrective'])}
        selected = [m for m in primary if any(r.get('selected_region_log_mse') is not None for r in error[m['id']])]
        region = {st: case_matrix([error], selected, metric='selected_region_log_mse',
            filter_row=lambda r, n=track_name(st, amp): r['state_key'][0] == n)[0]
            for st in ('static', 'average', 'corrective')}
        summaries[str(amp)]['fixed_selected_region'] = {
            'clips': len(selected), 'places': len({m['place'] for m in selected}),
            'means': {k: float(x.mean()) for k, x in region.items()},
            'static_minus_corrective': matched_gain(region['static'], region['corrective']),
            'average_minus_corrective': matched_gain(region['average'], region['corrective'])}
        error_curves[str(amp)] = {st: [float(case_matrix([error], primary,
            filter_row=lambda r, n=track_name(st, amp), t=ti: r['state_key'] == [n, t])[0].mean())
            for ti in range(1, 5)] for st in ('static', 'average', 'corrective')}
    retained = {}; curves = {}
    for policy in ('static_clean', 'average_clean', 'corrective_clean', 'rebuild_clean'):
        arrays = [case_matrix([retention], primary, filter_row=lambda r, n=policy, t=ti: r['state_key'] == [n, t])[0]
                  for ti in range(5)]
        retained[policy] = {'after_first_write': float(arrays[1].mean()), 'after_last_write': float(arrays[4].mean()),
                            'first_minus_last': matched_gain(arrays[1], arrays[4])}
        curves[policy] = [float(x.mean()) for x in arrays]
    result = {'triggered': True, 'fixed_seed': SEEDS[0], 'condition': 'full', 'amplitudes': summaries,
              'retrospective_retention': retained, 'retention_curves': curves, 'error_causal_curves': error_curves,
              'scope': 'single-model mechanism diagnostic on previously used data'}
    write_json(OUT / 'error_analysis.json', result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=('static', 'interventions', 'integrated', 'error'))
    args = p.parse_args()
    {'static': static_analysis, 'interventions': intervention_analysis,
     'integrated': integrated_analysis, 'error': error_analysis}[args.stage]()
