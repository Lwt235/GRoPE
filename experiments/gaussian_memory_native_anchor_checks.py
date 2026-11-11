"""Independent sensor scoring and bounded NumPy transform verification."""
import json
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from experiments.gaussian_memory_native_anchor_data import (
    ROOT, OUT, PARENT, BASE_CACHE, COUNTS, SEEDS, load_plan, sha256, write_json)
from experiments.gaussian_memory_incremental_data import sensor_target
from experiments.gaussian_memory_view_data import rectangle


@lru_cache(maxsize=None)
def digest(path): return sha256(path)


def independent_score(pred, target, valid, hidden, scale):
    assert np.isfinite(pred).all() and (pred > 0).all()
    residual = np.log(np.maximum(pred, 1e-6)) - np.log(target)
    if scale is not None: residual += np.log(scale)
    r = residual[valid]; mean = r.mean()
    calc = {'centered_log_mse': float(np.mean((r - mean)**2))}
    if scale is not None:
        error = pred[valid] * scale - target[valid]
        calc.update(log_mse=float(np.mean(r*r)), bias_squared=float(mean*mean),
                    absrel=float(np.mean(np.abs(error) / target[valid])),
                    rmse_m=float(np.sqrt(np.mean(error*error))))
        for key, keep in [('hidden_log_mse', valid & hidden), ('visible_log_mse', valid & ~hidden)]:
            calc[key] = float(np.mean(residual[keep]**2)) if keep.any() else None
    return calc, float(mean)


def main():
    plan = load_plan(); phash = digest(OUT / 'plan.json')
    assert digest(PARENT / 'plan.json') == plan['parent_plan_sha256']
    assert digest(PARENT / 'phase_completion.json') == plan['parent_completion_sha256']
    assert digest(PARENT / 'method_decision.json') == plan['parent_decision_sha256']
    for name, expected in plan['source_sha256'].items():
        assert digest(ROOT / name) == digest(OUT / 'source_snapshot' / name) == expected
    pair_by_id = {rid: p for p in plan['pairs'] for rid in (p['id'], p['early_id'])}
    counts = Counter(); max_score = {}; max_shift = 0.; max_transform_rel = 0.
    max_anchor_mean = 0.; max_spatial = 0.; num_numpy = 0; native_rows = 0
    bias_rows = []
    for meta in plan['catalog'] + plan['early_catalog']:
        rid = meta['id']; tasks = [t for t in plan['records'] if t['id'] == rid]
        keys = {tuple(k) for t in tasks for k in t['sample_keys']}
        identity_path = BASE_CACHE / rid / 'identity.json'
        identity_hash = digest(identity_path)
        assert identity_hash == pair_by_id[rid]['inputs'][rid]['identity_sha256']
        identity = json.loads(identity_path.read_text())
        scales = {(d['window'][-1], d['condition']): d.get('scale') for d in identity['diagnostics'] if d['kind'] == 'query'}
        targets = {}; natives = {}; masks = {}
        for q, c in sorted(keys):
            frame = meta['row']['frames'][q]
            assert digest(frame['depth_path']) == frame['depth_sha256']
            if q not in targets:
                targets[q] = tuple(x.numpy() for x in sensor_target(frame))
            path = BASE_CACHE / rid / f'q{q}_c{c}.pt'
            assert digest(path) == identity['sha256'][path.name]
            sample = torch.load(path, map_location='cpu', weights_only=False)
            mask = np.zeros((210, 280), bool) if c == 0 else rectangle(rid, q, c - 1).numpy().reshape(15, 20).repeat(14, 0).repeat(14, 1)
            assert np.array_equal(mask, sample['hidden'].numpy())
            # Valid labels are read only by this scoring function, never by the transform.
            assert np.array_equal(targets[q][0], sample['target'].numpy())
            assert np.array_equal(targets[q][1], sample['valid'].numpy())
            assert sample['metric_scale'] == scales[q, c]
            natives[q, c] = sample['query_depth_native'].numpy().astype('float64')
            masks[q, c] = mask
            del sample
        for task in tasks:
            assert digest(task['path']) == task['sha256']
            old = json.loads(Path(task['path']).read_text())
            path = OUT / 'evaluation' / task['phase'] / task['case'] / f'{rid}.json'
            record = json.loads(path.read_text())
            assert record['complete'] and record['plan_sha256'] == phash
            assert record['parent_record_sha256'] == task['sha256']
            assert record['parent_prediction_indices'] == task['indices']
            assert record['checkpoint_sha256'] == task['checkpoint_sha256']
            assert record['native_cache_identity_sha256'] == identity_hash
            assert digest(task['prediction_path']) == record['parent_prediction_sha256'] == task['prediction_sha256']
            assert digest(record['prediction_path']) == record['prediction_sha256']
            export = torch.load(record['prediction_path'], map_location='cpu', weights_only=False)
            assert export['sample_keys'] == task['sample_keys']
            assert len(record['rows']) == len(export['prediction_native']) == len(task['indices'])
            # Small fixed subset: first seed, the first early/late pair, query11 full/mask0.
            formula_subset = rid in ('fidelity_00', 'transfer_00') and str(SEEDS[0]) in task['case']
            parent_export = torch.load(task['prediction_path'], map_location='cpu', weights_only=False) if formula_subset else None
            for i, row in enumerate(record['rows']):
                q, c = task['sample_keys'][i]; target, valid = targets[q]
                assert (row['id'], row['query'], row['condition']) == (rid, q, c)
                assert int(valid.sum()) == row['valid_pixels']
                assert row['calibration_valid'] == (scales[q, c] is not None)
                assert row.get('state_key') == old['rows'][i].get('state_key')
                assert row.get('retrospective_retention') == old['rows'][i].get('retrospective_retention')
                pred = export['prediction_native'][i].numpy().astype('float64')
                native = natives[q, c]; hidden = masks[q, c]; keep = ~hidden
                assert row['anchor_unmasked_pixels'] == int(keep.sum())
                delta = float(export['anchor_log_shift'][i]); assert row['anchor_log_shift'] == delta
                calc, mean = independent_score(pred, target.astype('float64'), valid, hidden, scales[q, c])
                for metric, value in calc.items():
                    if value is None:
                        assert row[metric] is None; continue
                    error = abs(value - row[metric]); max_score[metric] = max(max_score.get(metric, 0.), error)
                    assert np.isclose(value, row[metric], rtol=1e-5, atol=1e-5), (path, i, metric, value, row[metric])
                spatial = abs(calc['centered_log_mse'] - old['rows'][i]['centered_log_mse'])
                max_spatial = max(max_spatial, spatial); assert spatial < 1e-5
                mean_change = abs(np.mean((np.log(pred) - np.log(native))[keep]))
                max_anchor_mean = max(max_anchor_mean, mean_change); assert mean_change < 1e-5
                if formula_subset and q == 11 and c in (0, 1):
                    parent_index = task['indices'][i]
                    assert parent_export['sample_keys'][parent_index] == [q, c]
                    raw = parent_export['prediction_native'][parent_index].numpy().astype('float64')
                    # Independent algebraic form, without production transform/helper.
                    independent_delta = np.mean(np.log(raw[keep] / native[keep]))
                    independent_pred = raw * np.exp(-independent_delta)
                    max_shift = max(max_shift, abs(delta - independent_delta))
                    err = float(np.max(np.abs(pred - independent_pred) / independent_pred))
                    max_transform_rel = max(max_transform_rel, err)
                    assert abs(delta - independent_delta) < 1e-5 and err < 1e-5
                    num_numpy += 1
                bias_rows.append({'phase': task['phase'], 'case': task['case'], 'id': rid,
                    'query': q, 'condition': c, 'delta': delta, 'anchored_signed_log_bias': mean,
                    'raw_signed_log_bias_from_constant_shift': mean + delta,
                    'raw_bias_squared': old['rows'][i]['bias_squared'], 'anchored_bias_squared': calc.get('bias_squared'),
                    'anchor_mean_absolute_error': mean_change})
                counts[task['phase']] += 1
        if rid.startswith('fidelity_'):
            info = next(n for n in plan['native_records'] if n['id'] == rid)
            assert digest(info['path']) == info['sha256']
            native_record = json.loads((OUT / 'evaluation/reused/native' / f'{rid}.json').read_text())
            assert native_record['rows'] == json.loads(Path(info['path']).read_text())['rows']
            for row in native_record['rows']:
                q, c = row['query'], row['condition']; target, valid = targets[q]
                calc, _ = independent_score(natives[q, c], target.astype('float64'), valid, masks[q, c], scales[q, c])
                for metric, value in calc.items():
                    if value is None: assert row[metric] is None
                    else: assert np.isclose(value, row[metric], rtol=1e-5, atol=1e-5)
                native_rows += 1
        print(json.dumps({'checked_clip': rid, 'transformed_scores': sum(counts.values())}), flush=True)
    assert dict(counts) == COUNTS and native_rows == 144 and num_numpy > 0
    with (OUT / 'bias_diagnostics.jsonl').open('w') as f:
        for row in bias_rows: f.write(json.dumps(row, allow_nan=False) + '\n')
    result = {'complete': True, 'transformed_scores_recomputed': sum(counts.values()),
        'by_phase': dict(counts), 'native_rows_recomputed_and_reused_unchanged': native_rows,
        'raw_sensor_targets_and_query_cache_hashes_verified': True, 'no_exclusions': True,
        'source_and_parent_identities_unchanged': True, 'numpy_formula_examples': num_numpy,
        'numpy_formula_max_relative_output_error': max_transform_rel, 'numpy_max_delta_error': max_shift,
        'float64_score_max_absolute_difference': max_score,
        'max_unmasked_mean_log_change': max_anchor_mean, 'max_centered_score_change': max_spatial,
        'interpretation': 'Centered invariance is an algebraic implementation property, not new spatial benefit.',
        'new_confirmation': False}
    write_json(OUT / 'independent_checks.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    torch.set_num_threads(4)
    main()
