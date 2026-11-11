"""Matched before/after, PE and historical-information comparisons."""
import json

import numpy as np

from experiments.gaussian_memory_native_anchor_data import (
    OUT, PARENT, FAMILIES, POLICIES, SEEDS, QUERIES, load_plan, write_json)
from experiments.gaussian_memory_incremental_analysis import case_matrix
from experiments.gaussian_memory_bound_analysis import compare_records
from experiments.gaussian_memory_history_analysis import interaction

MODES = ('mixed', 'full', 'masked')
METRICS = ('log_mse', 'centered_log_mse', 'bias_squared', 'absrel', 'rmse_m')
MODEL_PAIRS = [('content_dynamic', 'soft_dynamic'), ('content_static', 'soft_dynamic'),
    ('center_dynamic', 'soft_dynamic'), ('soft_static', 'soft_dynamic'),
    ('content_static', 'content_dynamic'), ('content_static', 'soft_static'),
    ('native', 'soft_dynamic'), ('local', 'soft_dynamic'),
    ('native', 'content_static'), ('local', 'content_static')]


def metrics(mode):
    return METRICS + (('hidden_log_mse', 'visible_log_mse') if mode == 'masked' else ())


def read_records(root, phase, case, catalog):
    out = {}
    for m in catalog:
        r = json.loads((root / 'evaluation' / phase / case / f'{m["id"]}.json').read_text())
        assert r['complete']; out[m['id']] = r['rows']
    return out


def main():
    plan = load_plan(); catalog = plan['catalog']; early = plan['early_catalog']
    place_names = sorted({m['place'] for m in catalog})
    versions = {}; record_versions = {}; all_versions = {}; ranges = {}; retention = {}; interventions = {}
    for version, root in [('raw', PARENT), ('anchored', OUT)]:
        policies = {}; all_records = {}; record_versions[version] = {}
        for policy in POLICIES:
            records = {f: [read_records(root, 'read', f'{f}_{s}_{policy}', catalog) for s in SEEDS] for f in FAMILIES}
            for f, r in records.items(): all_records[f'{f}_{policy}'] = r
            records['native'] = [read_records(root, 'reused', 'native', catalog)] * len(SEEDS)
            records['local'] = [read_records(root, 'reused' if version == 'raw' else 'local', f'local_{s}', catalog) for s in SEEDS]
            record_versions[version][policy] = records
            modes = {mode: {metric: compare_records(records, catalog, MODEL_PAIRS, mode, metric)
                           for metric in metrics(mode)} for mode in MODES}
            queries = {mode: [{'query': q, **compare_records(records, catalog, MODEL_PAIRS, mode, 'log_mse',
                filter_row=lambda r, q=q: r['query'] == q)} for q in QUERIES] for mode in MODES}
            policies[policy] = {'modes': modes, 'per_query': queries,
                'place_by_seed_log_mse': {f: case_matrix(r, catalog)[0].tolist() for f, r in records.items()}}
        versions[version] = policies; all_versions[version] = all_records
        pairs = []
        for f in FAMILIES:
            pairs += [(f'{f}_recent_{p}', f'{f}_history_{p}') for p in ('corrective', 'rebuild')]
            pairs += [(f'{f}_{h}_rebuild', f'{f}_{h}_corrective') for h in ('recent', 'history')]
        ranges[version] = {}
        for mode in MODES:
            ranges[version][mode] = {}
            for metric in metrics(mode):
                value = compare_records(all_records, catalog, pairs, mode, metric)
                value['history_gain_interaction'] = {p: interaction(all_records, catalog, mode, metric, p)
                                                    for p in ('corrective', 'rebuild')}
                ranges[version][mode][metric] = value
        ret_records = {}; ret_pairs = []
        for f in FAMILIES:
            for name, phase in [('early', 'retention_baseline'), ('late', 'retention')]:
                ret_records[f'{f}_{name}'] = [read_records(root, phase, f'{f}_{s}', early) for s in SEEDS]
            ret_pairs.append((f'{f}_early', f'{f}_late'))
        retention[version] = {metric: compare_records(ret_records, early, ret_pairs, 'full', metric) for metric in METRICS}
        inter = {'soft': [read_records(root, 'read', f'soft_dynamic_{SEEDS[0]}_history_corrective', catalog)]}
        for name in ('content_permuted', 'uniform_prior'):
            inter[name] = [read_records(root, 'intervention', f'soft_dynamic_{SEEDS[0]}_{name}', catalog)]
        interventions[version] = {mode: {metric: compare_records(inter, catalog,
            [('content_permuted', 'soft'), ('uniform_prior', 'soft')], mode, metric)
            for metric in ('log_mse', 'centered_log_mse', 'bias_squared')} for mode in MODES}
    before_after = {}
    for policy in POLICIES:
        records = {f'{f}_{v}': r for v in ('raw', 'anchored') for f, r in record_versions[v][policy].items()}
        pairs = [(f'{f}_raw', f'{f}_anchored') for f in (*FAMILIES, 'local')]
        before_after[policy] = {mode: {metric: compare_records(records, catalog, pairs, mode, metric)
                                     for metric in metrics(mode)} for mode in MODES}
    write_json(OUT / 'main_analysis.json', {'complete': True, 'places': place_names, 'versions': versions,
        'scope': 'All 12 previously used, availability-selected places and three fixed seeds; no new confirmation.'})
    write_json(OUT / 'before_after_analysis.json', {'complete': True, 'policies': before_after,
        'positive_means': 'Fixed anchor reduces error for the same model, state and input.'})
    write_json(OUT / 'history_range_analysis.json', {'complete': True, 'versions': ranges, 'places': place_names})
    write_json(OUT / 'retention_analysis.json', {'complete': True, 'versions': retention,
        'query': 11, 'condition': 0, 'retrospective': True, 'seeds': list(SEEDS)})
    write_json(OUT / 'intervention_analysis.json', {'complete': True, 'versions': interventions,
        'fixed_seed': SEEDS[0], 'scope': 'Within-model dependence, not comparative utility.'})
    print(json.dumps({'complete': True, 'anchored_history_corrective_means':
                      versions['anchored']['history_corrective']['modes']['mixed']['log_mse']['means'],
                      'soft_history_corrective_anchor_effect': before_after['history_corrective']['mixed']['log_mse']['comparisons']['soft_dynamic_raw_minus_soft_dynamic_anchored']}), flush=True)


if __name__ == '__main__': main()
