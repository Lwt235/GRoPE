"""Place/seed-paired history, PE, update, dependence and retention comparisons."""
import json
import numpy as np

from experiments.gaussian_memory_history_data import (
    OUT, FAMILIES, POLICIES, SEEDS, QUERIES, load_plan, write_json)
from experiments.gaussian_memory_incremental_analysis import case_matrix
from experiments.gaussian_memory_bound_analysis import compare_records


def read_records(phase, case, catalog):
    result = {}
    for meta in catalog:
        path = OUT / 'evaluation' / phase / case / f'{meta["id"]}.json'
        record = json.loads(path.read_text()); assert record['complete'], path
        result[meta['id']] = record['rows']
    return result


def interaction(records, catalog, mode, metric, policy):
    keys = [f'{f}_{h}_{policy}' for f in ('soft_dynamic', 'content_dynamic', 'content_static') for h in ('recent', 'history')]
    values = {k: case_matrix(records[k], catalog, metric, mode)[0] for k in keys}
    results = {}
    for control in ('content_dynamic', 'content_static'):
        delta = ((values[f'soft_dynamic_recent_{policy}'] - values[f'soft_dynamic_history_{policy}'])
                 - (values[f'{control}_recent_{policy}'] - values[f'{control}_history_{policy}']))
        places, seeds = delta.shape; rng = np.random.default_rng(20261110)
        pi = rng.integers(places, size=(5000, places)); si = rng.integers(seeds, size=(5000, seeds))
        boot = delta[pi[:, :, None], si[:, None, :]].mean((1, 2))
        results[control] = {'definition': f'(Soft_recent-Soft_history)-({control}_recent-{control}_history)',
            'positive_means': 'Soft benefits more from history than the content control',
            'mean_gain': float(delta.mean()), 'paired_seed_gain': delta.mean(0).tolist(),
            'paired_place_seed_95': np.quantile(boot, [.025, .975]).tolist(),
            'place_gain': delta.mean(1).tolist(), 'places': places, 'seeds': seeds}
    return results


def main():
    plan = load_plan(); catalog = plan['catalog']; all_records = {}; policy_results = {}
    model_pairs = [('content_dynamic', 'soft_dynamic'), ('content_static', 'soft_dynamic'),
                   ('center_dynamic', 'soft_dynamic'), ('soft_static', 'soft_dynamic'),
                   ('content_static', 'content_dynamic'), ('native', 'soft_dynamic'),
                   ('local', 'soft_dynamic'), ('native', 'content_static'), ('local', 'content_static')]
    for policy in POLICIES:
        records = {f: [read_records('read', f'{f}_{s}_{policy}', catalog) for s in SEEDS] for f in FAMILIES}
        for fam, rows in records.items(): all_records[f'{fam}_{policy}'] = rows
        records['native'] = [read_records('reused', 'native', catalog)] * len(SEEDS)
        records['local'] = [read_records('reused', f'local_{s}', catalog) for s in SEEDS]
        modes = {}
        for mode in ('mixed', 'full', 'masked'):
            metrics = ('log_mse', 'centered_log_mse', 'bias_squared', 'absrel', 'rmse_m')
            if mode == 'masked': metrics += ('hidden_log_mse', 'visible_log_mse')
            modes[mode] = {metric: compare_records(records, catalog, model_pairs, mode, metric) for metric in metrics}
        per_query = {mode: [{'query': q, **compare_records(records, catalog, model_pairs, mode, 'log_mse',
                     filter_row=lambda row, q=q: row['query'] == q)} for q in QUERIES] for mode in ('mixed', 'full', 'masked')}
        matrices = {fam: case_matrix(rows, catalog)[0].tolist() for fam, rows in records.items()}
        policy_results[policy] = {'modes': modes, 'per_query': per_query, 'place_by_seed_log_mse': matrices}
    places = sorted({m['place'] for m in catalog})
    write_json(OUT / 'main_analysis.json', {'complete': True, 'places': places, 'clips': 12,
        'all_data_previously_used': True, 'policies': policy_results,
        'scope': 'Availability-selected old12 subset, no new training or confirmation; all fixed models reported.'})
    comparisons = []
    for fam in FAMILIES:
        comparisons.extend([(f'{fam}_recent_{p}', f'{fam}_history_{p}') for p in ('corrective', 'rebuild')])
        comparisons.extend([(f'{fam}_{h}_rebuild', f'{fam}_{h}_corrective') for h in ('recent', 'history')])
    ranges = {}
    for mode in ('mixed', 'full', 'masked'):
        metrics = ('log_mse', 'centered_log_mse', 'bias_squared', 'absrel', 'rmse_m')
        if mode == 'masked': metrics += ('hidden_log_mse', 'visible_log_mse')
        ranges[mode] = {}
        for metric in metrics:
            item = compare_records(all_records, catalog, comparisons, mode, metric)
            item['history_gain_interaction'] = {p: interaction(all_records, catalog, mode, metric, p) for p in ('corrective', 'rebuild')}
            ranges[mode][metric] = item
    write_json(OUT / 'history_range_analysis.json', {'complete': True, 'places': places, 'modes': ranges})
    intervention_records = {'soft': [read_records('read', f'soft_dynamic_{SEEDS[0]}_history_corrective', catalog)]}
    for name in ('content_permuted', 'uniform_prior'):
        intervention_records[name] = [read_records('intervention', f'soft_dynamic_{SEEDS[0]}_{name}', catalog)]
    inter = {mode: {metric: compare_records(intervention_records, catalog,
        [('content_permuted', 'soft'), ('uniform_prior', 'soft')], mode, metric)
        for metric in ('log_mse', 'centered_log_mse', 'bias_squared')} for mode in ('mixed', 'full', 'masked')}
    write_json(OUT / 'intervention_analysis.json', {'complete': True, 'fixed_seed': SEEDS[0], 'modes': inter,
        'scope': 'Within-model first-seed dependence, not comparative utility; positive gain means intervention increases error.'})
    early_catalog = plan['early_catalog']; retention_records = {}; pairs = []
    for fam in FAMILIES:
        for name, phase in [('early', 'retention_baseline'), ('late', 'retention')]:
            retention_records[f'{fam}_{name}'] = [read_records(phase, f'{fam}_{s}', early_catalog) for s in SEEDS]
        pairs.append((f'{fam}_early', f'{fam}_late'))
    ret = {metric: compare_records(retention_records, early_catalog, pairs, 'full', metric)
           for metric in ('log_mse', 'centered_log_mse', 'bias_squared', 'absrel', 'rmse_m')}
    write_json(OUT / 'retention_analysis.json', {'complete': True, 'all_three_seeds': True, 'retrospective': True,
        'query': 11, 'condition': 0, 'metrics': ret, 'scope': 'Identical early query revisited after late writes; separate from causal primary.'})
    diagnostics = {p: json.loads((OUT / 'states' / p / 'diagnostics.json').read_text()) for p in [m['id'] for m in catalog]}
    summary = {}
    for policy in POLICIES:
        summary[policy] = {'by_query': [{key: float(np.mean([v[policy][i][key] for v in diagnostics.values()]))
            for key in ('query', 'query_mean_coverage', 'query_covered_fraction', 'query_mass_on_early_source_slots',
                        'slots_with_early_sources', 'slots_with_late_sources', 'slots_with_both',
                        'memory_bytes', 'recent_buffer_bytes', 'offline_observation_bytes')} for i in range(4)]}
        updates = [json.loads((OUT / 'states' / m['id'] / 'updates.json').read_text())[policy] for m in catalog]
        events = [e for v in updates for e in v['initialization_events'] + v['updates']]
        summary[policy]['events_totals'] = {key: int(sum(e.get(key, 0) for e in events)) for key in
            ('matched_patches', 'fused_slots', 'free_space_conflict_slots', 'unsupported_candidates',
             'capacity_dropped', 'local_rebuilds', 'new_region_insertions')}
        summary[policy]['update_compute_seconds'] = float(sum(v['initialization_seconds'] + sum(e['seconds'] for e in v['updates']) for v in updates))
    write_json(OUT / 'state_diagnostics.json', {'complete': True, 'policies': summary, 'by_clip': diagnostics,
        'early_mass_scope': 'Mass on slots retaining any early provenance, including mixed early/late slots; not isolated early-content attribution.'})
    counts = {}; costs = {}; peak = 0.
    for phase in ('read', 'intervention', 'retention'):
        records = [json.loads(p.read_text()) for p in (OUT / 'evaluation' / phase).glob('*/*.json')]
        counts[phase] = sum(len(r['rows']) for r in records)
        costs[phase] = {'records': len(records), 'case_seconds': sum(r['seconds'] for r in records),
                        'inference_seconds': sum(r['inference_seconds'] for r in records)}
        peak = max(peak, max(r['peak_allocated_GiB'] for r in records))
    assert counts == {'read': 8640, 'intervention': 288, 'retention': 180}
    write_json(OUT / 'runtime_costs.json', {'complete': True, 'new_training_steps': 0, 'new_download_bytes': 0,
        'new_predictions': counts, 'phases': costs, 'peak_allocated_GiB': peak,
        'state_preparation_seconds': sum(json.loads((OUT / 'states' / m['id'] / 'complete.json').read_text())['seconds'] for m in catalog),
        'scope': 'Shared GPU worker times; not exclusive throughput benchmark. Case time excludes clip loading; inference time includes CPU/GPU transfer. State bytes include provenance and lineage; offline Prefix observations reported separately.'})
    print(json.dumps({'analysis_complete': True, 'history_corrective_means': policy_results['history_corrective']['modes']['mixed']['log_mse']['means'],
                      'history_gain': ranges['mixed']['log_mse']['comparisons']['soft_dynamic_recent_corrective_minus_soft_dynamic_history_corrective']}), flush=True)


if __name__ == '__main__': main()
