"""Independent source-mask projection and exported prediction scoring."""
import json
from pathlib import Path

import numpy as np
import torch

from experiments.gaussian_memory_visibility_data import (
    ROOT, OUT, CACHE, BASE_CACHE, EARLY, LATE, CURRENT, FAMILIES, SEEDS,
    load_plan, sha256, write_json)
from experiments.gaussian_memory_incremental_data import sensor_target
from experiments.gaussian_memory_view_data import rectangle


def cloud_numpy(depth, valid, camera, K):
    yy, xx = np.mgrid[:210, :280]
    uv = np.stack((xx + .5, yy + .5, np.ones_like(xx)), -1)
    xyz = (uv @ np.linalg.inv(K).T) * depth[..., None]
    xyz = xyz[valid]
    homogeneous = np.concatenate((xyz, np.ones((len(xyz), 1))), axis=1)
    return (homogeneous @ camera.T)[:, :3]


def index_numpy(points, camera, K):
    homogeneous = np.concatenate((points, np.ones((len(points), 1))), axis=1)
    # Independent full homogeneous inverse, rather than transpose/translation.
    xyz = (homogeneous @ np.linalg.inv(camera).T)[:, :3]
    p = xyz @ K.T
    with np.errstate(divide='ignore', invalid='ignore'):
        uv = p[:, :2] / p[:, 2:]
    keep = (np.isfinite(xyz).all(-1) & (xyz[:, 2] > 1e-4) & (uv[:, 0] >= 0)
            & (uv[:, 0] < 280) & (uv[:, 1] >= 0) & (uv[:, 1] < 210))
    return keep, np.floor(uv[keep]).astype(int), xyz[keep, 2]


def raster_numpy(points, camera, K):
    _, uv, depth = index_numpy(points, camera, K)
    buf = np.full(210 * 280, np.inf)
    np.minimum.at(buf, uv[:, 1] * 280 + uv[:, 0], depth)
    return np.where(np.isfinite(buf), buf, 0).reshape(210, 280)


def main():
    torch.set_num_threads(4); plan = load_plan(); plan_sha = sha256(OUT / 'plan.json')
    for p, expected in plan['input_identities'].items(): assert sha256(p) == expected, p
    for p, expected in plan['source_sha256'].items():
        assert sha256(ROOT / p) == expected == sha256(OUT / 'source_snapshot' / p), p
    for rec in plan['checkpoints'].values(): assert sha256(rec['path']) == rec['sha256']
    maximum = {}; mask_checks = []; score_count = 0; baseline_rows = 0; baseline_max = 0.; state_count = 0
    old_bound = ROOT / 'results/gaussian_memory/bound_read1/evaluation/static'
    for meta in plan['catalog']:
        rid = meta['id']; folder = CACHE / rid
        source = json.loads((OUT / 'inputs' / f'{rid}.json').read_text())
        assert source['complete'] and source['plan_sha256'] == plan_sha
        assert source['query_frame_never_written'] and not source['sensor_geometry_in_candidate_states']
        for name, expected in source['files'].items(): assert sha256(folder / name) == expected
        row = meta['row']; K = np.asarray(row['K'], dtype=np.float64); cameras = np.asarray(row['c2w_metric'], dtype=np.float64)
        roi = rectangle(rid, 11, 0).numpy().reshape(15, 20).repeat(14, 0).repeat(14, 1)
        early_clouds = []
        for f in EARLY:
            d, v = sensor_target(row['frames'][f])
            early_clouds.append(cloud_numpy(d.numpy().astype(float), v.numpy(), cameras[f], K[f]))
        early = np.concatenate(early_clouds)
        keep, uv, _ = index_numpy(early, cameras[11], K[11])
        selected = early[np.flatnonzero(keep)[roi[uv[:, 1], uv[:, 0]]]]
        expected_masks = np.zeros((12, 300), bool)
        for f in EARLY + LATE:
            _, uv, _ = index_numpy(selected, cameras[f], K[f])
            expected_masks[f, (uv[:, 1] // 14) * 20 + uv[:, 0] // 14] = True
        expected_masks[11] = rectangle(rid, 11, 0).numpy()
        saved = torch.load(folder / 'masks.pt', map_location='cpu', weights_only=False)
        actual_masks = saved['patch_masks'].numpy()
        mismatch = int(np.count_nonzero(expected_masks != actual_masks))
        assert mismatch == 0, (rid, 'mask', mismatch)
        assert len(selected) == source['diagnostic']['selected_early_points']
        recent, remaining = [], []
        for f in LATE:
            d, v = sensor_target(row['frames'][f]); d, v = d.numpy().astype(float), v.numpy()
            mask = expected_masks[f].reshape(15, 20).repeat(14, 0).repeat(14, 1)
            recent.append(cloud_numpy(d, v, cameras[f], K[f]))
            remaining.append(cloud_numpy(d, v & ~mask, cameras[f], K[f]))
        check_maps = {'early': raster_numpy(early, cameras[11], K[11]),
                      'recent_full': raster_numpy(np.concatenate(recent), cameras[11], K[11]),
                      'recent_persistent': raster_numpy(np.concatenate(remaining), cameras[11], K[11])}
        raster_checks = {}
        for name, independent in check_maps.items():
            original = saved['sensor_support_maps'][name].numpy()
            disagreement = int(np.count_nonzero((original > 0) != (independent > 0)))
            valid = original > 0
            rel = float(np.max(np.abs(original[valid] - independent[valid]) / original[valid])) if valid.any() else 0.
            assert disagreement == 0 and rel < 1e-8, (rid, name, disagreement, rel)
            raster_checks[name] = {'support_mismatch_pixels': disagreement, 'max_relative_depth_difference': rel}
        mask_checks.append({'id': rid, 'mask_mismatch_patches': mismatch, 'selected_early_points': len(selected), 'projection_checks': raster_checks})
        states = torch.load(folder / 'states.pt', map_location='cpu', weights_only=False)
        for key, state in states.items():
            allowed = set(LATE) | (set(EARLY) if key.endswith('/history') else set())
            used = set(state['sources'].any(0).nonzero().flatten().tolist())
            assert used.issubset(allowed) and state['features'].shape == (256, 32)
            assert all(torch.isfinite(v).all() for v in state.values())
            assert torch.linalg.eigvalsh(state['covars']).min() > 0
            state_count += 1
        for w in source['windows']:
            assert w['all_context_copies_masked']
            assert w['mask_patch_counts'] == [int(expected_masks[f].sum()) for f in w['frames']]
            if w['kind'] == 'write': assert 11 not in w['frames']
            else: assert w['frames'] == list(CURRENT)
        target, valid = sensor_target(row['frames'][11]); target = target.numpy().astype(float); valid = valid.numpy()
        samples = {c: torch.load(folder / f'q11_c{c}.pt', map_location='cpu', weights_only=False) for c in range(3)}
        scale = plan['inputs'][rid]['query_scale_full']
        for c, sample in samples.items():
            assert np.array_equal(sample['target'].numpy(), target)
            assert np.array_equal(sample['valid'].numpy(), valid)
            assert np.array_equal(sample['hidden'].numpy(), roi)
            assert sample['metric_scale'] == scale
            if c < 2:
                old = torch.load(BASE_CACHE / rid / f'q11_c{c}.pt', map_location='cpu', weights_only=False)
                for key in ('prefix', 'early4', 'local_content', 'query_depth_native', 'K', 'c2w', 'rgb'):
                    assert torch.equal(sample[key], old[key])
        rec = json.loads((OUT / 'evaluation' / f'{rid}.json').read_text())
        assert rec['complete'] and rec['plan_sha256'] == plan_sha
        assert rec['input_record_sha256'] == sha256(OUT / 'inputs' / f'{rid}.json')
        assert sha256(rec['prediction_path']) == rec['prediction_sha256']
        export = torch.load(rec['prediction_path'], map_location='cpu', weights_only=False)
        expected_keys = {(c, 'native', None, None) for c in range(3)}
        expected_keys |= {(c, 'local', s, None) for c in range(3) for s in SEEDS}
        expected_keys |= {(c, f, s, p) for c in range(3) for f in FAMILIES for s in SEEDS for p in ('recent', 'history')}
        actual_keys = [tuple(k) for k in export['keys']]
        assert set(actual_keys) == expected_keys and len(actual_keys) == 102
        for i, r in enumerate(rec['rows']):
            c = r['condition']; family = r['family']; seed = r['seed']
            assert tuple(export['keys'][i]) == (c, family, seed, r['policy'])
            pred = export['prediction_native'][i].numpy().astype(float)
            assert np.isfinite(pred).all() and (pred > 0).all()
            residual = np.log(np.maximum(pred, 1e-6)) + np.log(scale) - np.log(target)
            x = residual[valid]; error = pred[valid] * scale - target[valid]
            calc = {'log_mse': float(np.mean(x*x)), 'centered_log_mse': float(np.mean((x-x.mean())**2)),
                    'bias_squared': float(x.mean()**2), 'absrel': float(np.mean(np.abs(error) / target[valid])),
                    'rmse_m': float(np.sqrt(np.mean(error**2))),
                    'hidden_log_mse': float(np.mean(residual[valid & roi]**2)),
                    'visible_log_mse': float(np.mean(residual[valid & ~roi]**2))}
            for key, value in calc.items():
                delta = abs(value - r[key]); maximum[key] = max(maximum.get(key, 0.), delta)
                assert np.isclose(value, r[key], rtol=1e-5, atol=1e-5), (rid, i, key, value, r[key])
            assert r['valid_pixels'] == int(valid.sum())
            if family == 'native': assert np.array_equal(pred, samples[c]['query_depth_native'].numpy())
            if family == 'local' and c < 2:
                old_rec = json.loads((old_bound / f'local_{seed}' / f'{rid}.json').read_text())
                assert sha256(old_rec['prediction_path']) == old_rec['prediction_sha256']
                old_pred = torch.load(old_rec['prediction_path'], map_location='cpu', weights_only=False)
                j = old_pred['sample_keys'].index([11, c])
                difference = np.abs(pred - old_pred['prediction_native'][j].numpy())
                baseline_max = max(baseline_max, float(difference.max()))
                np.testing.assert_allclose(pred, old_pred['prediction_native'][j].numpy(), rtol=2e-4, atol=2e-5)
                baseline_rows += 1
            score_count += 1
        print(json.dumps({'checked': rid, 'predictions': score_count, 'mask_mismatch_patches': mismatch}), flush=True)
    assert score_count == 1224 and state_count == 48 and baseline_rows == 72
    write_json(OUT / 'independent_checks.json', {'complete': True,
        'predictions_independently_scored': score_count, 'max_score_difference': maximum,
        'mask_and_sensor_projection_checks': mask_checks, 'states_checked': state_count,
        'unchanged_local_control_predictions_checked': baseline_rows, 'old_local_max_absolute_prediction_difference': baseline_max,
        'all_sources_and_checkpoints_identified': True, 'all12places_retained': True,
        'query_labels_only_for_scoring': True, 'early_sensor_only_for_artificial_task_construction': True,
        'shared_original_full_query_scale': True, 'new_confirmation': False,
        'scope': 'Independent homogeneous-coordinate mask/projection construction and NumPy scoring. No expansion into integrator or numerical precision gates.'})


if __name__ == '__main__': main()
