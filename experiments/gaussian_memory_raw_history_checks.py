"""Independent projection examples, complete count/score checks and source identities."""
import json
from pathlib import Path

import numpy as np
import torch

from experiments.gaussian_memory_raw_history_data import (
    ROOT, OUT, CACHE, HISTORY, VARIANTS, QUERIES, SOURCE_FRAMES, load_plan, sha256, write_json)
from experiments.gaussian_memory_incremental_data import sensor_target
from experiments.gaussian_memory_view_data import rectangle


def numpy_projection(meta, frames, transform, query_camera, query_K, variant):
    saved = torch.load(CACHE / 'sources' / f'{meta["id"]}.pt', map_location='cpu', weights_only=False)['source_depth_m']
    y, x = np.mgrid[:210, :280]; pixels = np.stack((x + .5, y + .5, np.ones_like(x)), -1)
    result = np.full(210 * 280, np.inf); to_query = np.linalg.inv(query_camera)
    for f in frames:
        sensor, valid = sensor_target(meta['row']['frames'][f]); sensor = sensor.numpy().astype('float64'); valid = valid.numpy()
        pred = saved[f].numpy().astype('float64'); pv = np.isfinite(pred) & (pred > .1) & (pred < 10.)
        mask = valid if variant.startswith('sensor') else pv
        if variant.endswith('_matched'): mask = valid & pv
        depth = sensor if variant.startswith('sensor') else pred
        K = np.asarray(meta['row']['K'][f]); camera = transform @ np.asarray(meta['row']['c2w_metric'][f])
        rays = np.linalg.solve(K, pixels.reshape(-1, 3).T).T
        points = rays * depth.reshape(-1, 1)
        homogeneous = np.concatenate((points[mask.ravel()], np.ones((int(mask.sum()), 1))), 1)
        local = (to_query @ camera @ homogeneous.T).T[:, :3]
        front = np.isfinite(local).all(1) & (local[:, 2] > 1e-4); local = local[front]
        image = (query_K @ local.T).T; uv = image[:, :2] / image[:, 2:]
        keep = (uv[:, 0] >= 0) & (uv[:, 0] < 280) & (uv[:, 1] >= 0) & (uv[:, 1] < 210)
        pixel = np.floor(uv[keep]).astype(int)
        np.minimum.at(result, pixel[:, 1] * 280 + pixel[:, 0], local[keep, 2])
    result[~np.isfinite(result)] = 0
    return result.reshape(210, 280)


def region(rid, q, condition, name):
    if name == 'all': return np.ones((210, 280), bool)
    mask = rectangle(rid, q, condition - 1).numpy().reshape(15, 20).repeat(14, 0).repeat(14, 1)
    return mask if name == 'hidden' else ~mask


def main():
    plan = load_plan(); catalog = {m['id']: m for m in plan['catalog']}; max_error = {}; checked = 0; projections = []
    for name, digest in plan['source_sha256'].items(): assert sha256(ROOT / name) == digest
    assert sha256(ROOT / 'docs/gaussian_memory_raw_history_information_plan.md') == plan['document_sha256']
    assert sha256(HISTORY / 'plan.json') == plan['parent_history_plan_sha256']
    for meta in plan['catalog']:
        r = json.loads((OUT / 'sources' / f'{meta["id"]}.json').read_text())
        assert r['complete'] and sha256(r['path']) == r['sha256']
        assert r['plan_sha256'] == sha256(OUT / 'plan.json') and len(r['replayed_windows']) == 5
        for f in SOURCE_FRAMES:
            frame = meta['row']['frames'][f]
            assert sha256(frame['depth_path']) == frame['depth_sha256']
            assert sha256(frame['rgb_path']) == frame['rgb_sha256']
    for pair in plan['pairs']:
        rid = pair['id']; meta = catalog[rid]
        r = json.loads((OUT / 'evaluation' / f'{rid}.json').read_text())
        assert r['complete'] and sha256(r['map_path']) == r['map_sha256']
        maps = torch.load(r['map_path'], map_location='cpu', weights_only=False)
        targets = {}
        parent_diag = json.loads((HISTORY / 'states' / rid / 'diagnostics.json').read_text())
        for ti, q in enumerate(QUERIES):
            frame = meta['row']['frames'][q]; assert sha256(frame['depth_path']) == frame['depth_sha256']
            target, valid = sensor_target(frame); targets[q] = (target.numpy().astype('float64'), valid.numpy())
            assert maps[q]['recent_sources'] == [f for f in SOURCE_FRAMES if f < q]
            assert maps[q]['early_sources'] == list(SOURCE_FRAMES)
            for p, state in maps[q]['state'].items():
                err = abs(float(state['coverage'].mean()) - parent_diag[p][ti]['query_mean_coverage'])
                assert err < 1e-5, (rid, q, p, err)
                max_error['parent_state_mean_coverage'] = max(max_error.get('parent_state_mean_coverage', 0.), err)
            if rid in ('fidelity_00', 'fidelity_01') and q in (5, 11):
                T = np.asarray(pair['common_from_late_metric']); camera = T @ np.asarray(meta['row']['c2w_metric'][q]); K = np.asarray(meta['row']['K'][q])
                for v in ('sensor', 'prediction'):
                    a = numpy_projection(catalog[pair['early_id']], SOURCE_FRAMES, np.eye(4), camera, K, v)
                    b = numpy_projection(meta, maps[q]['recent_sources'], T, camera, K, v)
                    combined = np.minimum(np.where(a > 0, a, np.inf), np.where(b > 0, b, np.inf)); combined[~np.isfinite(combined)] = 0
                    for name, expected in [('early', a), ('recent', b), ('history', combined)]:
                        actual = maps[q]['raw'][v][name]; common = (expected > 0) & (actual > 0)
                        mismatch = int(np.count_nonzero((expected > 0) != (actual > 0)))
                        rel = np.abs(actual[common] - expected[common]) / expected[common]
                        q999 = float(np.quantile(rel, .999)) if len(rel) else 0.
                        assert mismatch / 58800 < .001 and q999 < 1e-4, (rid, q, v, name, mismatch, q999)
                        projections.append({'id': rid, 'query': q, 'variant': v, 'range': name,
                            'support_mismatch_pixels': mismatch, 'relative_error_p999': q999,
                            'relative_error_max': float(rel.max()) if len(rel) else 0.})
        def check(key, value, old):
            nonlocal checked
            if value is None: assert old[key] is None; return
            err = abs(float(value) - old[key]); max_error[key] = max(max_error.get(key, 0.), err)
            assert np.isclose(value, old[key], rtol=1e-7, atol=1e-8), (rid, key, value, old[key])
        for row in r['raw_rows'] + r['state_rows']:
            q = row['query']; target, valid = targets[q]; reg = region(rid, q, row['condition'], row['region'])
            d = maps[q]['raw'][row['variant']][row['range']] if 'variant' in row else maps[q]['state'][row['policy']]['depth_m']
            s = d > 0; known = valid & reg; selected = s & known
            good = selected & (d >= target * .9) & (d <= target * 1.1)
            denom = int(known.sum()); n = int(selected.sum())
            assert row['region_pixels'] == int(reg.sum()) and row['valid_pixels'] == denom and row['supported_valid_pixels'] == n
            for key, a, b in [('support_fraction', (s & reg).sum(), reg.sum()), ('valid_support_fraction', n, denom),
                ('consistent_fraction', good.sum(), denom), ('consistency_given_support', good.sum(), n),
                ('foreground_conflict_fraction', (selected & (d < target * .9)).sum(), denom),
                ('background_fraction', (selected & (d > target * 1.1)).sum(), denom)]:
                check(key, a / b if b else None, row)
            check('supported_absrel', float(np.mean(np.abs(d[selected] / target[selected] - 1))) if n else None, row)
            check('supported_log_mse', float(np.mean((np.log(d[selected]) - np.log(target[selected]))**2)) if n else None, row)
            checked += 1
        for row in r['history_rows']:
            q = row['query']; target, valid = targets[q]; reg = region(rid, q, row['condition'], row['region']); known = valid & reg
            recent, hist = [maps[q]['raw'][row['variant']][name] for name in ('recent', 'history')]
            rs, hs = recent > 0, hist > 0; novel = hs & ~rs; n = int((novel & known).sum()); denom = int(known.sum())
            rg = rs & (recent >= target * .9) & (recent <= target * 1.1); hg = hs & (hist >= target * .9) & (hist <= target * 1.1)
            assert row['new_valid_support_pixels'] == n and np.all(~rs | hs)
            entries = [('new_support_fraction', (novel & reg).sum(), reg.sum()), ('overlap_support_fraction', (rs & hs & reg).sum(), reg.sum()),
                ('new_valid_support_fraction', n, denom), ('new_consistent_support_fraction', (novel & known & hg).sum(), denom),
                ('new_support_consistency', (novel & known & hg).sum(), n),
                ('consistency_net_gain', int((hg & known).sum()) - int((rg & known).sum()), denom),
                ('consistency_added_fraction', (hg & ~rg & known).sum(), denom), ('consistency_lost_fraction', (rg & ~hg & known).sum(), denom)]
            for key, a, b in entries: check(key, a / b if b else None, row)
            check('new_support_absrel', float(np.mean(np.abs(hist[novel & known] / target[novel & known] - 1))) if n else None, row)
            checked += 1
        for row in r['retention_rows']:
            q = row['query']; target, valid = targets[q]; reg = region(rid, q, row['condition'], row['region']); known = valid & reg
            recent, hist = [maps[q]['raw'][row['variant']][name] for name in ('recent', 'history')]
            novel = (hist > 0) & (recent == 0) & known
            useful = novel & (hist >= target * .9) & (hist <= target * 1.1)
            state = maps[q]['state'][row['policy']]['depth_m']; present = state > 0
            good = present & (state >= target * .9) & (state <= target * 1.1)
            n = int(useful.sum()); denom = int(known.sum()); assert n == row['raw_new_consistent_pixels']
            for key, a, b in [('state_covers_raw_new_fraction', (present & useful).sum(), n),
                ('state_consistent_on_raw_new_fraction', (good & useful).sum(), n), ('raw_new_consistent_fraction', n, denom),
                ('retained_consistent_fraction_all_valid', (good & useful).sum(), denom),
                ('state_covers_any_raw_new_fraction', (present & novel).sum(), novel.sum())]:
                check(key, a / b if b else None, row)
            checked += 1
        print(json.dumps({'checked': rid, 'rows': checked}), flush=True)
    assert checked == 12096 and len(projections) == 24
    result = {'complete': True, 'score_rows_independently_checked': checked, 'numpy_projection_examples': projections,
        'max_scalar_difference': max_error, 'all_source_and_target_identities_verified': True,
        'all48queries_and144conditions_preserved': True, 'current_queries_never_source_frames': True,
        'new_training_steps': 0, 'new_confirmation': False,
        'scope': 'Independent homogeneous-coordinate NumPy projections on2fixedpairs; all exported count/error tables recomputed. No broad numerical-gate expansion.'}
    write_json(OUT / 'independent_checks.json', result); print(json.dumps({'complete': True, 'rows': checked, 'projection_examples': len(projections)}), flush=True)


if __name__ == '__main__':
    torch.set_num_threads(4); main()
