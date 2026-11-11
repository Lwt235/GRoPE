"""Place-matched comparisons, state diagnostics, and independent output scoring."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.gaussian_memory_setup import sha256
from experiments.gaussian_memory_incremental_data import OUT, CACHE, SEEDS, KINDS, QUERIES, load_plan, write_json, sensor_target
from experiments.gaussian_memory_incremental_states import STATE_OUT, AMPLITUDES, track_name
from experiments.gaussian_memory_view_data import rectangle


def cohorts(plan):
    primary = [m for m in plan['catalog'] if m['eval_split'] in ('used_clip40', 'used_clip80', 'confirmation', 'used_writer19')]
    out = {'primary51': primary,
           'old12': [m for m in primary if m['eval_split'] in ('used_clip40', 'used_clip80')],
           'used_summary20': [m for m in primary if m['eval_split'] == 'confirmation'],
           'used_writer19': [m for m in primary if m['eval_split'] == 'used_writer19'],
           'training12': [m for m in plan['catalog'] if m['training']]}
    assert len({m['place'] for m in primary}) == 51
    return out


def matched_gain(baseline, proposed, bootstrap_seed=20261110):
    """Both arrays [places,seeds], paired throughout; positive = lower error."""
    assert baseline.shape == proposed.shape and np.isfinite(baseline).all() and np.isfinite(proposed).all()
    diff = baseline - proposed
    P, S = diff.shape; rng = np.random.default_rng(bootstrap_seed)
    pi = rng.integers(P, size=(5000, P)); si = rng.integers(S, size=(5000, S))
    boot = diff[pi[:, :, None], si[:, None, :]].mean((1, 2))
    return {'baseline_mean': float(baseline.mean()), 'proposed_mean': float(proposed.mean()),
            'mean_gain': float(diff.mean()), 'relative_reduction_percent': float(diff.mean() / max(baseline.mean(), 1e-12) * 100),
            'paired_place_seed_95': np.quantile(boot, [.025, .975]).tolist(),
            'paired_seed_gain': diff.mean(0).tolist(), 'places_better': int((diff.mean(1) > 0).sum()),
            'places': P, 'seeds': S, 'place_gain': diff.mean(1).tolist()}


def read_records(phase, case, catalog):
    out = {}
    for meta in catalog:
        path = OUT / 'evaluation' / phase / case / (meta['id'] + '.json')
        r = json.loads(path.read_text()); assert r['complete'], path
        out[meta['id']] = r['rows']
    return out


def case_matrix(all_seed_records, catalog, metric='log_mse', mode='mixed', filter_row=None):
    places = sorted({m['place'] for m in catalog})
    result = np.empty((len(places), len(all_seed_records)))
    for pi, place in enumerate(places):
        for si, records in enumerate(all_seed_records):
            clips = []
            for m in catalog:
                if m['place'] != place: continue
                vals = []
                for r in records[m['id']]:
                    if mode == 'full' and r['condition'] != 0: continue
                    if mode == 'masked' and r['condition'] == 0: continue
                    if filter_row is not None and not filter_row(r): continue
                    if r.get(metric) is not None: vals.append(r[metric])
                assert vals, (m['id'], metric, mode)
                clips.append(float(np.mean(vals)))
            result[pi, si] = np.mean(clips)
    return result, places


def static_analysis():
    plan = load_plan(); groups = cohorts(plan)
    records = {k: [read_records('static', f'{k}_{s}', plan['catalog']) for s in SEEDS] for k in KINDS}
    native = read_records('static', 'native', plan['catalog']); records['native'] = [native] * 3
    center_present = ((OUT / 'static_center_all_complete.json').exists()
                      or all((OUT / f'static_center_{s}_complete.json').exists() for s in SEEDS))
    if center_present:
        records['center'] = [read_records('static', f'center_{s}', plan['catalog']) for s in SEEDS]
    result = {'all_data_previously_used': True, 'center_evaluated': center_present, 'cohorts': {}}
    for group, catalog in groups.items():
        g = {'places': sorted({m['place'] for m in catalog}), 'clips': len(catalog), 'modes': {}}
        for mode in ('mixed', 'full', 'masked'):
            metrics = ('log_mse', 'centered_log_mse', 'bias_squared', 'absrel', 'rmse_m')
            if mode == 'masked': metrics += ('hidden_log_mse', 'visible_log_mse')
            modes = {}
            for metric in metrics:
                values = {k: case_matrix(v, catalog, metric, mode)[0] for k, v in records.items()}
                pairs = [('local', 'combined'), ('content', 'combined'), ('geometry', 'combined'),
                         ('native', 'combined'), ('native', 'local'), ('local', 'content'), ('local', 'geometry')]
                if center_present: pairs.append(('center', 'combined'))
                modes[metric] = {'means': {k: float(v.mean()) for k, v in values.items()},
                    'comparisons': {f'{a}_minus_{b}': matched_gain(values[a], values[b]) for a, b in pairs}}
            g['modes'][mode] = modes
        result['cohorts'][group] = g
    pe = result['cohorts']['primary51']['modes']['mixed']['log_mse']['comparisons']['content_minus_combined']
    result['center_control_triggered'] = bool(pe['mean_gain'] > 0 and sum(x > 0 for x in pe['paired_seed_gain']) >= 2)
    result['center_control_decision'] = ('run fixed three-seed center diagnostic' if result['center_control_triggered']
                                          else 'do not add shape-control training; PE increment criterion unmet')
    write_json(OUT / 'static_analysis.json', result)
    print(json.dumps({'center_control_triggered': result['center_control_triggered'],
                      'primary_C_minus_CG': pe}, ensure_ascii=False), flush=True)
    return result


def integrated_analysis():
    plan = load_plan(); result = {'cohorts': {}, 'readers_fixed': True, 'new_fits': 0}
    all_records = {}
    for k in ('content', 'combined'):
        all_records[f'{k}_static'] = [read_records('static', f'{k}_{s}', plan['catalog']) for s in SEEDS]
        for policy in ('corrective_clean', 'rebuild_clean'):
            all_records[f'{k}_{policy}'] = [read_records('integrated', f'{k}_{s}_{policy}', plan['catalog']) for s in SEEDS]
    for group, catalog in cohorts(plan).items():
        result['cohorts'][group] = {}
        for mode in ('mixed', 'full', 'masked'):
            outputs = {}
            for metric in ('log_mse', 'centered_log_mse', 'bias_squared'):
                arrays = {k: case_matrix(v, catalog, metric, mode)[0] for k, v in all_records.items()}
                pairs = []
                for kind in ('content', 'combined'):
                    pairs += [(f'{kind}_static', f'{kind}_corrective_clean'),
                              (f'{kind}_static', f'{kind}_rebuild_clean'),
                              (f'{kind}_rebuild_clean', f'{kind}_corrective_clean')]
                outputs[metric] = {'means': {k: float(x.mean()) for k, x in arrays.items()},
                    'comparisons': {a + '_minus_' + b: matched_gain(arrays[a], arrays[b]) for a, b in pairs}}
            result['cohorts'][group][mode] = outputs
    # Error/retention probes are single-model diagnostics, kept separate from the three-seed result.
    error = read_records('error_probe', 'combined_20261110', plan['catalog'])
    retention = read_records('retention', 'combined_20261110', plan['catalog'])
    primary = cohorts(plan)['primary51']; error_summary = {}
    for amp in AMPLITUDES:
        arrays = {}
        for st in ('static', 'average', 'corrective'):
            name = track_name(st, amp)
            arrays[st] = case_matrix([error], primary, filter_row=lambda r, n=name: r['state_key'][0] == n)[0]
        error_summary[str(amp)] = {'means': {k: float(x.mean()) for k, x in arrays.items()},
                                  'static_minus_corrective': matched_gain(arrays['static'], arrays['corrective']),
                                  'average_minus_corrective': matched_gain(arrays['average'], arrays['corrective'])}
        region_catalog = [m for m in primary if any(r.get('selected_region_log_mse') is not None for r in error[m['id']])]
        region_arrays = {st: case_matrix([error], region_catalog, metric='selected_region_log_mse',
                                        filter_row=lambda r, n=track_name(st, amp): r['state_key'][0] == n)[0]
                         for st in ('static', 'average', 'corrective')}
        error_summary[str(amp)]['fixed_selected_region'] = {
            'clips': len(region_catalog), 'means': {k: float(x.mean()) for k, x in region_arrays.items()},
            'static_minus_corrective': matched_gain(region_arrays['static'], region_arrays['corrective']),
            'average_minus_corrective': matched_gain(region_arrays['average'], region_arrays['corrective'])}
    retained = {}
    for policy in ('static_clean', 'average_clean', 'corrective_clean', 'rebuild_clean'):
        a, _ = case_matrix([retention], primary, filter_row=lambda r, n=policy: r['state_key'] == [n, 1])
        b, _ = case_matrix([retention], primary, filter_row=lambda r, n=policy: r['state_key'] == [n, 4])
        retained[policy] = {'after_first_write': float(a.mean()), 'after_last_write': float(b.mean()),
                            'first_minus_last': matched_gain(a, b)}
    result.update(error_probe_first_seed_full=error_summary, retrospective_retention=retained)
    revisit = json.loads((OUT / 'revisit_diagnostics.json').read_text())['clips']
    strata = {}
    for label, flag in [('some_potential_revisit', True), ('no_potential_revisit', False)]:
        selected = [m for m in primary if (revisit[m['id']]['writes'][-1]['revisited_slots'] > 0) == flag]
        group = {'clips': len(selected), 'places': len({m['place'] for m in selected}), 'last_query_only': True, 'amplitudes': {}}
        for amp in AMPLITUDES:
            values = {st: case_matrix([error], selected, filter_row=lambda r, n=track_name(st, amp): r['state_key'] == [n, 4])[0]
                      for st in ('static', 'average', 'corrective')}
            group['amplitudes'][str(amp)] = {'means': {k: float(x.mean()) for k, x in values.items()},
                'static_minus_corrective': matched_gain(values['static'], values['corrective']),
                'average_minus_corrective': matched_gain(values['average'], values['corrective'])}
        strata[label] = group
    result['revisit_stratification_first_seed'] = strata
    # Preserve the actual four causal query times, beyond an average over them.
    causal = {}
    for mode in ('mixed', 'full', 'masked'):
        causal[mode] = []
        for ti, q in enumerate(QUERIES, 1):
            values = {key: case_matrix(records, primary, mode=mode, filter_row=lambda r, q=q: r['query'] == q)[0]
                      for key, records in all_records.items()}
            pairs = [(f'{k}_static', f'{k}_corrective_clean') for k in ('content', 'combined')]
            pairs += [(f'{k}_rebuild_clean', f'{k}_corrective_clean') for k in ('content', 'combined')]
            causal[mode].append({'write_index': ti, 'query': q, 'means': {k: float(v.mean()) for k, v in values.items()},
                'comparisons': {a + '_minus_' + b: matched_gain(values[a], values[b]) for a, b in pairs}})
    result['causal_primary51'] = causal
    retention_curves = {}
    for policy in ('static_clean', 'average_clean', 'corrective_clean', 'rebuild_clean'):
        retention_curves[policy] = [float(case_matrix([retention], primary,
            filter_row=lambda r, n=policy, t=ti: r['state_key'] == [n, t])[0].mean()) for ti in range(5)]
    result['retention_curves_first_seed'] = retention_curves
    error_curves = {}
    for amp in AMPLITUDES:
        error_curves[str(amp)] = {}
        for strategy in ('static', 'average', 'corrective'):
            name = track_name(strategy, amp)
            error_curves[str(amp)][strategy] = [float(case_matrix([error], primary,
                filter_row=lambda r, n=name, t=ti: r['state_key'] == [n, t])[0].mean()) for ti in range(1, 5)]
    result['error_causal_curves_first_seed'] = error_curves
    write_json(OUT / 'integrated_analysis.json', result)
    print(json.dumps({'integrated_analysis_complete': True}), flush=True)


def state_analysis():
    plan = load_plan(); primary = cohorts(plan)['primary51']
    output = {'data_previously_used': True, 'geometry_is_diagnostic': True, 'cohorts': {},
              'support_metric_revision': json.loads((STATE_OUT / 'support_metric_revision.json').read_text()),
              'legacy_metrics_not_geometric_accuracy': ['geometry_log_mse_on_fixed_support', 'paired_clean_geometry_gap']}
    for group, catalog in cohorts(plan).items():
        by_policy = {}; events = {}
        for meta in catalog:
            scores = json.loads((STATE_OUT / meta['id'] / 'probes.json').read_text())
            support = json.loads((STATE_OUT / meta['id'] / 'support_probes.json').read_text())
            for policy, trajectory in scores.items():
                for ti, state in enumerate(trajectory):
                    for qi, query in enumerate(state['queries']):
                        assert query['query'] == support[policy][ti]['queries'][qi]['query']
                        query.update(support[policy][ti]['queries'][qi])
            logs = json.loads((STATE_OUT / meta['id'] / 'updates.json').read_text())
            for policy, trajectory in scores.items():
                by_policy.setdefault(policy, {})[meta['id']] = trajectory
                events.setdefault(policy, []).append({name: sum(x.get(name, 0) for x in logs[policy])
                    for name in ('local_rebuilds', 'new_region_insertions', 'capacity_dropped', 'free_space_conflict_slots')})
        place_order = sorted({m['place'] for m in catalog}); summaries = {}
        metrics = ('geometry_log_mse_on_fixed_support', 'retained_fixed_support_fraction', 'valid_patch_coverage',
                   'new_coverage_fraction', 'external_probe_log_mse', 'external_probe_centered_mse', 'paired_clean_geometry_gap',
                   'geometry_log_mse_common_support', 'common_support_patches', 'common_support_fraction_of_initial',
                   'paired_clean_geometry_gap_common_support')
        for policy, data in by_policy.items():
            summary = {'causal_query': {}, 'fixed_early_query': {}, 'valid_place_counts': {},
                       'events_mean_per_clip': {k: float(np.mean([e[k] for e in events[policy]])) for k in events[policy][0]}}
            for metric in metrics:
                for mode in ('causal_query', 'fixed_early_query'):
                    times = []; counts = []
                    for ti in range(1, 5):
                        qi = ti - 1 if mode == 'causal_query' else 0
                        values = []
                        for place in place_order:
                            vals = [data[m['id']][ti]['queries'][qi].get(metric) for m in catalog if m['place'] == place]
                            vals = [v for v in vals if v is not None]
                            if vals: values.append(float(np.mean(vals)))
                        times.append(float(np.mean(values)) if values else None)
                        counts.append(len(values))
                    summary[mode][metric] = times
                    summary['valid_place_counts'].setdefault(mode, {})[metric] = counts
            summaries[policy] = summary
        output['cohorts'][group] = {'places': place_order, 'policies': summaries}
    write_json(OUT / 'state_analysis.json', output)
    scores = {m['id']: json.loads((STATE_OUT / m['id'] / 'support_probes.json').read_text()) for m in primary}
    final = {}
    for amp in AMPLITUDES:
        places = sorted({m['place'] for m in primary})
        available = [place for place in places if any(
            scores[m['id']][track_name('static', amp)][4]['queries'][3]['geometry_log_mse_common_support'] is not None
            for m in primary if m['place'] == place)]
        arrays = {}
        for strategy in ('static', 'average', 'corrective'):
            values = []
            for place in available:
                rows = [scores[m['id']][track_name(strategy, amp)][4]['queries'][3]['geometry_log_mse_common_support']
                        for m in primary if m['place'] == place]
                values.append(np.mean([v for v in rows if v is not None]))
            arrays[strategy] = np.asarray(values)[:, None]
        final[str(amp)] = {'places_with_common_support': available,
            'means': {k: float(v.mean()) for k, v in arrays.items()},
            'average_minus_corrective': matched_gain(arrays['average'], arrays['corrective']),
            'static_minus_corrective': matched_gain(arrays['static'], arrays['corrective'])}
    write_json(OUT / 'state_last_query_accuracy.json', final)
    print(json.dumps({'state_analysis_complete': True, 'primary_places': len({m['place'] for m in primary})}), flush=True)


def independent_checks():
    """Re-score exported native predictions in float64 NumPy using original labels."""
    plan = load_plan(); metadata = {m['id']: m for m in plan['catalog']}
    targets = {}; max_error = {}; count = 0; phase_counts = {}
    for path in sorted((OUT / 'evaluation').glob('*/*/*.json')):
        record = json.loads(path.read_text())
        if 'prediction_path' not in record: continue
        rid = path.stem
        if rid not in targets:
            targets[rid] = {q: tuple(x.numpy() for x in sensor_target(metadata[rid]['row']['frames'][q])) for q in QUERIES}
        prediction_path = Path(record['prediction_path'])
        assert sha256(prediction_path) == record['prediction_sha256'], path
        export = torch.load(prediction_path, map_location='cpu', weights_only=False)
        assert len(export['prediction_native']) == len(record['rows'])
        identity = json.loads((CACHE / rid / 'identity.json').read_text())
        scales = {(d['window'][-1], d['condition']): d.get('scale') for d in identity['diagnostics'] if d['kind'] == 'query'}
        for index, row in enumerate(record['rows']):
            q, c = row['query'], row['condition']; assert export['sample_keys'][index] == [q, c]
            target, valid = targets[rid][q]; target = target.astype('float64')
            pred = export['prediction_native'][index].numpy().astype('float64')
            scale = scales[q, c]
            residual = np.log(np.maximum(pred, 1e-6)) - np.log(target)
            if scale is not None: residual += np.log(scale)
            r = residual[valid]
            calc = {'centered_log_mse': float(np.mean((r - r.mean())**2))}
            if scale is not None:
                error = pred[valid] * scale - target[valid]
                mask = np.zeros((210, 280), dtype=bool) if c == 0 else rectangle(rid, q, c - 1).numpy().reshape(15, 20).repeat(14, 0).repeat(14, 1)
                calc.update(log_mse=float(np.mean(r*r)), bias_squared=float(r.mean()**2),
                            absrel=float(np.mean(np.abs(error)/target[valid])), rmse_m=float(np.sqrt(np.mean(error*error))))
                for name, keep in [('hidden_log_mse', valid & mask), ('visible_log_mse', valid & ~mask)]:
                    if keep.any(): calc[name] = float(np.mean(residual[keep]**2))
                if row.get('selected_region_log_mse') is not None:
                    region = torch.load(STATE_OUT / rid / 'selected_region.pt', map_location='cpu', weights_only=False)['query_patch_masks'][QUERIES.index(q)].numpy()
                    keep = valid & region.reshape(15, 20).repeat(14, 0).repeat(14, 1)
                    calc['selected_region_log_mse'] = float(np.mean(residual[keep]**2))
            for metric, value in calc.items():
                delta = abs(value - row[metric]); max_error[metric] = max(max_error.get(metric, 0.), delta)
                assert np.isclose(value, row[metric], rtol=1e-5, atol=1e-5), (str(path), index, metric, value, row[metric])
            count += 1; phase_counts[path.parents[1].name] = phase_counts.get(path.parents[1].name, 0) + 1
    matched = []
    for seed in SEEDS:
        kinds = list(KINDS) + (['center'] if (OUT / f'center_{seed}/status.json').exists() else [])
        initial = [torch.load(OUT / f'{k}_{seed}/initial.pt', map_location='cpu', weights_only=False) for k in kinds]
        assert all(all(torch.equal(a['model'][n], initial[0]['model'][n]) for n in a['model']) for a in initial)
        assert len({a['schedule_sha256'] for a in initial}) == 1
        logs = [[json.loads(x) for x in (OUT / f'{k}_{seed}/training.jsonl').read_text().splitlines()] for k in kinds]
        assert all(len(x) == 2000 for x in logs)
        assert all(all(a[i]['samples'] == logs[0][i]['samples'] for i in range(2000)) for a in logs)
        matched.append({'seed': seed, 'kinds': kinds, 'initialization_equal': True, 'all_2000_sampling_steps_equal': True})
    result = {'complete': True, 'exported_predictions_recomputed': count, 'by_phase': phase_counts,
              'float64_numpy_max_abs_difference': max_error, 'matched_training': matched,
              'interface_checks': json.loads((OUT / 'preflight/interface_checks.json').read_text()),
              'new_confirmation': False}
    write_json(OUT / 'independent_checks.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=('static', 'integrated', 'states', 'checks'))
    args = parser.parse_args(); torch.set_num_threads(4)
    {'static': static_analysis, 'integrated': integrated_analysis, 'states': state_analysis, 'checks': independent_checks}[args.stage]()
