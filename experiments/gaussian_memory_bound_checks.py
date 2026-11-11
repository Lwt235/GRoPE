"""Independent native scoring and matched-fit provenance for bound_read1."""
import json
from pathlib import Path

import numpy as np
import torch

from experiments.gaussian_memory_setup import ROOT, sha256
from experiments.gaussian_memory_bound_data import OUT, CACHE, BASE_CACHE, SEEDS, KINDS, QUERIES, load_plan, write_json
from experiments.gaussian_memory_bound_train import source_identity
from experiments.gaussian_memory_incremental_data import sensor_target
from experiments.gaussian_memory_incremental_states import STATE_OUT
from experiments.gaussian_memory_view_data import rectangle


def expected_cases(decision):
    kinds = list(KINDS) + (['center'] if decision['center_control_triggered'] else [])
    result = {'static': [(f'{k}_{s}', 12) for k in kinds for s in SEEDS],
              'intervention': [(f'bound_{SEEDS[0]}_{name}', 12) for name in ('content_permuted', 'coverage_prior')],
              'integrated': [(f'{k}_{s}_{policy}', 12) for k in ('coverage', 'bound') for s in SEEDS
                             for policy in ('corrective_clean', 'rebuild_clean')]}
    if decision['error_probe_triggered']:
        result.update(error_probe=[(f'bound_{SEEDS[0]}', 52)], retention=[(f'bound_{SEEDS[0]}', 20)])
    return kinds, result


def training_checks(kinds):
    identities = source_identity(); matched = []; summaries = []
    for relative, digest in identities.items():
        assert sha256(OUT / 'source_snapshot' / relative) == digest
    for seed in SEEDS:
        initial = []; logs = []
        for kind in kinds:
            folder = OUT / f'{kind}_{seed}'
            status = json.loads((folder / 'status.json').read_text())
            assert status['complete'] and status['steps'] == 2000
            assert status['checkpoint_sha256'] == sha256(folder / 'step_2000.pt')
            model = torch.load(folder / 'step_2000.pt', map_location='cpu', weights_only=False)
            assert model['source_sha256'] == identities and model['kind'] == kind and model['seed'] == seed
            initial.append(torch.load(folder / 'initial.pt', map_location='cpu', weights_only=False))
            rows = [json.loads(line) for line in (folder / 'training.jsonl').read_text().splitlines()]
            assert [r['step'] for r in rows] == list(range(1, 2001))
            assert all(np.isfinite(r['loss']) and np.isfinite(r['gradient_norm']) for r in rows)
            logs.append(rows)
            # Sum elapsed segments if a fit resumed; do not mislabel worker sums as wall time.
            segments = []; previous = 0.
            for row in rows:
                if row['elapsed_s'] < previous:
                    segments.append(previous)
                previous = row['elapsed_s']
            segments.append(previous)
            summaries.append({**status, 'training_loop_seconds': float(sum(segments)),
                              'elapsed_segments': len(segments)})
        assert all(a['source_sha256'] == identities for a in initial)
        assert all(set(a['model']) == set(initial[0]['model']) for a in initial)
        assert all(all(torch.equal(a['model'][n], initial[0]['model'][n]) for n in a['model']) for a in initial)
        assert len({a['schedule_sha256'] for a in initial}) == 1
        assert all(all(a[i]['samples'] == logs[0][i]['samples'] for i in range(2000)) for a in logs)
        matched.append({'seed': seed, 'kinds': kinds, 'initialization_equal': True, 'all_2000_sampling_steps_equal': True})
    summary = {'fits': len(summaries), 'steps': sum(x['steps'] for x in summaries),
               'sum_training_loop_seconds': sum(x['training_loop_seconds'] for x in summaries),
               'sum_worker_hours': sum(x['training_loop_seconds'] for x in summaries) / 3600,
               'peak_allocated_GiB': max(x['peak_GiB'] for x in summaries),
               'per_fit': summaries, 'timing_scope': 'shared GPU training loop sums; excludes queue wait, extraction, evaluation; not exclusive throughput or elapsed wall time'}
    write_json(OUT / 'training_summary.json', summary)
    return matched


def main():
    plan = load_plan(); decision = json.loads((OUT / 'static_analysis.json').read_text())
    kinds, expected = expected_cases(decision); matched = training_checks(kinds)
    metadata = {m['id']: m for m in plan['catalog']}; targets = {}; source_cache = {}; identities = {}
    max_error = {}; count = 0; phase_counts = {}; record_counts = {}
    protocol_hash = sha256(OUT / 'evaluation_protocol.json')
    evaluate_hash = sha256(ROOT / 'experiments/gaussian_memory_bound_evaluate.py')
    for phase, cases in expected.items():
        for case, rows_expected in cases:
            for rid in metadata:
                path = OUT / 'evaluation' / phase / case / f'{rid}.json'
                record = json.loads(path.read_text())
                assert record['complete'] and len(record['rows']) == rows_expected
                assert record['evaluation_protocol_sha256'] == protocol_hash
                assert record['evaluation_source_sha256'] == evaluate_hash
                if rid not in targets:
                    targets[rid] = {q: tuple(x.numpy() for x in sensor_target(metadata[rid]['row']['frames'][q])) for q in QUERIES}
                    identity = json.loads((BASE_CACHE / rid / 'identity.json').read_text())
                    identities[rid] = identity
                identity = identities[rid]
                source = record['sources']
                assert source['causal_cache_identity_sha256'] == sha256(BASE_CACHE / rid / 'identity.json')
                assert source['initial_sha256'] == identity['sha256']['initial.pt']
                if 'state_trajectories_sha256' in source:
                    state_path = STATE_OUT / rid / 'trajectories.pt'
                    if str(state_path) not in source_cache:
                        source_cache[str(state_path)] = sha256(state_path)
                    assert source['state_trajectories_sha256'] == source_cache[str(state_path)]
                if 'prior_identity_sha256' in source:
                    prior_path = CACHE / rid / ('center_identity.json' if case.startswith('center_') else 'identity.json')
                    assert source['prior_identity_sha256'] == sha256(prior_path)
                checkpoint_case = '_'.join(case.split('_')[:2])
                checkpoint_path = OUT / checkpoint_case / 'step_2000.pt'
                if str(checkpoint_path) not in source_cache:
                    source_cache[str(checkpoint_path)] = sha256(checkpoint_path)
                assert record['checkpoint_sha256'] == source_cache[str(checkpoint_path)]
                prediction_path = Path(record['prediction_path'])
                assert sha256(prediction_path) == record['prediction_sha256']
                export = torch.load(prediction_path, map_location='cpu', weights_only=False)
                assert len(export['prediction_native']) == len(record['rows'])
                scales = {(d['window'][-1], d['condition']): d.get('scale') for d in identity['diagnostics'] if d['kind'] == 'query'}
                for index, row in enumerate(record['rows']):
                    q, c = row['query'], row['condition']
                    assert row['id'] == rid and export['sample_keys'][index] == [q, c]
                    assert export['state_keys'][index] == row.get('state_key')
                    target, valid = targets[rid][q]; target = target.astype('float64')
                    pred = export['prediction_native'][index].numpy().astype('float64')
                    assert np.isfinite(pred).all() and (pred > 0).all()
                    assert int(valid.sum()) == row['valid_pixels']
                    scale = scales[q, c]; assert (scale is not None) == row['calibration_valid']
                    residual = np.log(np.maximum(pred, 1e-6)) - np.log(target)
                    if scale is not None: residual += np.log(scale)
                    r = residual[valid]
                    calc = {'centered_log_mse': float(np.mean((r - r.mean())**2))}
                    if scale is not None:
                        error = pred[valid] * scale - target[valid]
                        mask = np.zeros((210, 280), dtype=bool) if c == 0 else rectangle(rid, q, c - 1).numpy().reshape(15, 20).repeat(14, 0).repeat(14, 1)
                        calc.update(log_mse=float(np.mean(r*r)), bias_squared=float(r.mean()**2),
                                    absrel=float(np.mean(np.abs(error) / target[valid])), rmse_m=float(np.sqrt(np.mean(error*error))))
                        for name, keep in [('hidden_log_mse', valid & mask), ('visible_log_mse', valid & ~mask)]:
                            if keep.any(): calc[name] = float(np.mean(residual[keep]**2))
                        if row.get('selected_region_log_mse') is not None:
                            regions = torch.load(STATE_OUT / rid / 'selected_region.pt', map_location='cpu', weights_only=False)['query_patch_masks']
                            keep = valid & regions[QUERIES.index(q)].numpy().reshape(15, 20).repeat(14, 0).repeat(14, 1)
                            assert int(keep.sum()) == row['selected_region_valid_pixels']
                            calc['selected_region_log_mse'] = float(np.mean(residual[keep]**2))
                    for metric, value in calc.items():
                        delta = abs(value - row[metric]); max_error[metric] = max(max_error.get(metric, 0.), delta)
                        assert np.isclose(value, row[metric], rtol=1e-5, atol=1e-5), (str(path), index, metric, value, row[metric])
                    count += 1; phase_counts[phase] = phase_counts.get(phase, 0) + 1
                record_counts[phase] = record_counts.get(phase, 0) + 1
            print(json.dumps({'independently_checked': f'{phase}/{case}', 'predictions': count}), flush=True)
    native_count = 0
    for rid in metadata:
        record = json.loads((OUT / 'evaluation/static/native' / f'{rid}.json').read_text())
        assert sha256(record['reused_from']) == record['reused_record_sha256']
        native_count += len(record['rows'])
    result = {'complete': True, 'exported_predictions_recomputed': count, 'by_phase': phase_counts,
              'records_by_phase': record_counts, 'reused_native_rows': native_count,
              'float64_numpy_max_abs_difference': max_error, 'matched_training': matched,
              'scientific_sources_match_snapshot': True, 'interface_checks': json.loads((OUT / 'preflight/interface_checks.json').read_text()),
              'new_confirmation': False, 'no_row_or_clip_exclusions': True}
    write_json(OUT / 'independent_checks.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    torch.set_num_threads(4); main()
