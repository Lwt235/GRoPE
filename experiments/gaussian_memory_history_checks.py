"""Independent raw-label scoring and causal source audit for the fixed history test."""
import json
from pathlib import Path

import numpy as np
import torch

from experiments.gaussian_memory_history_data import (
    ROOT, OUT, CACHE, BASE_CACHE, STATE_READ, STATE_OUT, FAMILIES, POLICIES, SEEDS, QUERIES,
    WRITES, load_plan, sha256, write_json)
from experiments.gaussian_memory_incremental_data import sensor_target, depth_source
from experiments.gaussian_memory_view_data import rectangle


def main():
    plan = load_plan(); hashes = {}
    def digest(path):
        path = str(path)
        if path not in hashes: hashes[path] = sha256(path)
        return hashes[path]
    plan_hash = digest(OUT / 'plan.json')
    evaluator_hash = digest(ROOT / 'experiments/gaussian_memory_history_evaluate.py')
    for name, expected in plan['source_sha256'].items():
        assert digest(ROOT / name) == digest(OUT / 'source_snapshot' / name) == expected
    for meta in plan['checkpoints'].values(): assert digest(meta['path']) == meta['sha256']
    assert json.loads((OUT / 'update_compatibility.json').read_text())['complete']
    geometry = json.loads((STATE_READ / 'history_geometry_preparation.json').read_text())
    assert geometry['complete']
    state_checks = []
    for pair in plan['pairs']:
        rid = pair['id']; a = pair['early_id']
        assert max(pair['frames_early']) < min(pair['frames_late'])
        assert digest(pair['pose_path']) == pair['pose_sha256']
        marker = json.loads((OUT / 'states' / rid / 'complete.json').read_text())
        assert marker['complete'] and marker['plan_sha256'] == plan_hash
        assert marker['query_frames_never_written'] and not marker['labels_used_for_updates']
        assert marker['late_initialization_available_at'] == 15
        assert digest(marker['trajectories_path']) == marker['trajectories_sha256']
        assert digest(OUT / 'states' / rid / 'diagnostics.json') == marker['diagnostics_sha256']
        assert digest(STATE_OUT / a / 'trajectories.pt') == pair['early_trajectories_sha256']
        tracks = torch.load(marker['trajectories_path'], map_location='cpu', weights_only=False)
        original = torch.load(STATE_OUT / a / 'trajectories.pt', map_location='cpu', weights_only=False)['corrective_clean']['states'][4]
        for key, value in original.items():
            actual = tracks['early_corrective_end'][key]
            assert torch.equal(value, actual[:, :12] if key in ('sources', 'conflicts') else actual)
        assert not tracks['early_corrective_end']['sources'][:, 12:].any()
        for policy in POLICIES:
            assert len(tracks[policy]['states']) == 5
            for ti, memory in enumerate(tracks[policy]['states']):
                allowed = {12, 13, 14, 15} | {12 + x for x in WRITES[:ti]}
                if policy.startswith('history'): allowed |= {0, 1, 2, 3, *WRITES}
                for key in ('sources', 'conflicts'):
                    assert memory[key].shape == (256, 24)
                    actual = set(memory[key].any(0).nonzero().flatten().tolist())
                    assert actual.issubset(allowed), (rid, policy, ti, key, actual)
                    if ti: assert all(f < 12 + QUERIES[ti - 1] for f in actual)
                assert memory['features'].shape == (256, 32)
                assert all(torch.isfinite(v).all() for v in memory.values())
                assert torch.linalg.eigvalsh(memory['covars']).min() > 0
            assert len(tracks[policy]['recent_final']) <= 4
        state_checks.append({'id': rid, 'all_four_policies_causal': True, 'early_baseline_exact': True,
                             'recent_has_no_early_source_slots': True, 'fixed_slots': 256})
    metadata = {m['id']: m for m in plan['catalog'] + plan['early_catalog']}
    pair_by_id = {name: pair for pair in plan['pairs'] for name in (pair['id'], pair['early_id'])}
    targets = {}; identities = {}; scales = {}
    for rid, meta in metadata.items():
        pair = pair_by_id[rid]; path = BASE_CACHE / rid / 'identity.json'
        assert digest(path) == pair['inputs'][rid]['identity_sha256']
        identity = json.loads(path.read_text()); identities[rid] = identity
        scales[rid] = {(d['window'][-1], d['condition']): d.get('scale') for d in identity['diagnostics'] if d['kind'] == 'query'}
        queries = QUERIES if rid == pair['id'] else (11,)
        targets[rid] = {q: tuple(x.numpy() for x in sensor_target(meta['row']['frames'][q])) for q in queries}
        for frame in meta['row']['frames'][:12]:
            assert digest(frame['rgb_path']) == frame['rgb_sha256']
        for name, expected in pair['inputs'][rid]['files'].items(): assert digest(BASE_CACHE / rid / name) == expected
        for q in queries:
            for c in (range(3) if rid == pair['id'] else (0,)):
                name = f'q{q}_c{c}.pt'; assert digest(BASE_CACHE / rid / name) == identity['sha256'][name]
    cases = {'read': [(f'{f}_{s}_{p}', f, s, p) for f in FAMILIES for s in SEEDS for p in POLICIES],
             'intervention': [(f'soft_dynamic_{SEEDS[0]}_{n}', 'soft_dynamic', SEEDS[0], 'history_corrective') for n in ('content_permuted', 'uniform_prior')],
             'retention': [(f'{f}_{s}', f, s, 'history_corrective') for f in FAMILIES for s in SEEDS]}
    total = 0; counts = {}; record_counts = {}; max_error = {}
    for phase, phase_cases in cases.items():
        for case, family, seed, policy in phase_cases:
            catalog = plan['early_catalog'] if phase == 'retention' else plan['catalog']
            for meta in catalog:
                rid = meta['id']; pair = pair_by_id[rid]
                path = OUT / 'evaluation' / phase / case / f'{rid}.json'; record = json.loads(path.read_text())
                expected_keys = [(11, 0)] if phase == 'retention' else [(q, c) for q in QUERIES for c in range(3)]
                assert record['complete'] and len(record['rows']) == len(expected_keys)
                assert record['plan_sha256'] == plan_hash and record['evaluator_sha256'] == evaluator_hash
                assert record['checkpoint_sha256'] == plan['checkpoints'][f'{family}_{seed}']['sha256']
                sources = record['sources']; identity = identities[rid]
                assert sources['causal_cache_identity_sha256'] == digest(BASE_CACHE / rid / 'identity.json')
                assert sources['initial_sha256'] == identity['sha256']['initial.pt']
                assert sources['history_trajectories_sha256'] == digest(CACHE / 'states' / pair['id'] / 'trajectories.pt')
                assert digest(record['prediction_path']) == record['prediction_sha256']
                export = torch.load(record['prediction_path'], map_location='cpu', weights_only=False)
                assert len(export['prediction_native']) == len(expected_keys)
                for index, row in enumerate(record['rows']):
                    q, c = expected_keys[index]; ti = 4 if phase == 'retention' else QUERIES.index(q) + 1
                    assert row['id'] == rid and (row['query'], row['condition']) == (q, c)
                    assert row['state_key'] == export['state_keys'][index] == [policy, ti]
                    assert export['sample_keys'][index] == [q, c]
                    if phase == 'retention': assert row['retrospective_retention']
                    target, valid = targets[rid][q]; target = target.astype('float64')
                    pred = export['prediction_native'][index].numpy().astype('float64')
                    assert np.isfinite(pred).all() and (pred > 0).all()
                    assert int(valid.sum()) == row['valid_pixels']
                    scale = scales[rid][q, c]; assert (scale is not None) == row['calibration_valid']
                    residual = np.log(np.maximum(pred, 1e-6)) - np.log(target)
                    if scale is not None: residual += np.log(scale)
                    r = residual[valid]; calc = {'centered_log_mse': float(np.mean((r - r.mean())**2))}
                    if scale is not None:
                        error = pred[valid] * scale - target[valid]
                        mask = np.zeros((210, 280), bool) if c == 0 else rectangle(rid, q, c - 1).numpy().reshape(15, 20).repeat(14, 0).repeat(14, 1)
                        calc.update(log_mse=float(np.mean(r*r)), bias_squared=float(r.mean()**2),
                            absrel=float(np.mean(np.abs(error) / target[valid])), rmse_m=float(np.sqrt(np.mean(error*error))))
                        for metric, keep in [('hidden_log_mse', valid & mask), ('visible_log_mse', valid & ~mask)]:
                            if keep.any(): calc[metric] = float(np.mean(residual[keep]**2))
                    for metric, value in calc.items():
                        error = abs(value - row[metric]); max_error[metric] = max(max_error.get(metric, 0.), error)
                        assert np.isclose(value, row[metric], rtol=1e-5, atol=1e-5), (path, index, metric, value, row[metric])
                    total += 1; counts[phase] = counts.get(phase, 0) + 1
                record_counts[phase] = record_counts.get(phase, 0) + 1
            print(json.dumps({'checked': f'{phase}/{case}', 'predictions': total}), flush=True)
    assert counts == {'read': 8640, 'intervention': 288, 'retention': 180}
    reused = json.loads((OUT / 'baseline_reuse.json').read_text()); reused_rows = 0
    for item in reused['records']:
        assert digest(item['source']) == item['sha256']
        old = json.loads(Path(item['source']).read_text())
        record = json.loads((OUT / 'evaluation' / item['phase'] / item['case'] / f'{item["id"]}.json').read_text())
        expected = [old['rows'][i] for i in record['parent_prediction_indices']] if item['phase'] == 'retention_baseline' else old['rows']
        assert record['rows'] == expected and len(expected) == item['rows']
        assert record['plan_sha256'] == plan_hash and record['reused_record_sha256'] == item['sha256']
        reused_rows += len(expected)
    assert reused_rows == 756 and total == 9108
    result = {'complete': True, 'exported_predictions_recomputed': total, 'by_phase': counts,
        'records_by_phase': record_counts, 'reused_parent_rows': reused_rows,
        'float64_numpy_max_abs_difference': max_error, 'source_states': state_checks,
        'fixed_checkpoints_verified': 15, 'scientific_sources_match_snapshot': True,
        'query_and_label_boundaries_checked': True, 'no_row_or_clip_exclusions': True,
        'new_confirmation': False}
    write_json(OUT / 'independent_checks.json', result); print(json.dumps(result), flush=True)


if __name__ == '__main__':
    torch.set_num_threads(4)
    main()
