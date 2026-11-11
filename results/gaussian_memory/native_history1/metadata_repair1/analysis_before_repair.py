"""Independent native-depth scoring, source projection and paired place analysis."""
import argparse
import json
import time

import numpy as np
import torch

from experiments.gaussian_memory_native_data import (
    ROOT, OUT, CACHE, DEST, HISTORY, RECENT, EXTRA, CONDITIONS, ARMS, read, load, sha256, write_json)
from experiments.gaussian_memory_incremental_data import sensor_target

METRICS = ('hidden_log_mse', 'log_mse', 'visible_log_mse', 'hidden_centered_log_mse',
           'hidden_bias_squared', 'centered_log_mse', 'bias_squared', 'absrel', 'hidden_absrel', 'rmse_m')


def resize_numpy(depth, h=210, w=280):
    H, W = depth.shape
    y = np.clip((np.arange(h) + .5) * H / h - .5, 0, H - 1)
    x = np.clip((np.arange(w) + .5) * W / w - .5, 0, W - 1)
    iy, ix = np.floor(y).astype(int), np.floor(x).astype(int)
    jy, jx = np.minimum(iy + 1, H - 1), np.minimum(ix + 1, W - 1)
    fy, fx = (y - iy)[:, None], (x - ix)[None]
    return ((1 - fy) * ((1 - fx) * depth[iy[:, None], ix] + fx * depth[iy[:, None], jx]) +
            fy * ((1 - fx) * depth[jy[:, None], ix] + fx * depth[jy[:, None], jx]))


def score_numpy(depth, sample):
    target = sample['target'].numpy().astype(float); valid, hidden = sample['valid'].numpy(), sample['hidden'].numpy()
    residual = np.log(np.maximum(depth, 1e-6) / target)
    values, missing = residual[valid], residual[valid & hidden]
    error = depth[valid] - target[valid]; he = depth[valid & hidden] - target[valid & hidden]
    return {'log_mse': float(np.mean(values ** 2)), 'hidden_log_mse': float(np.mean(missing ** 2)),
        'visible_log_mse': float(np.mean(residual[valid & ~hidden] ** 2)),
        'centered_log_mse': float(np.var(values)), 'bias_squared': float(np.mean(values) ** 2),
        'hidden_centered_log_mse': float(np.var(missing)), 'hidden_bias_squared': float(np.mean(missing) ** 2),
        'log_bias': float(np.mean(values)), 'absrel': float(np.mean(np.abs(error) / target[valid])),
        'hidden_absrel': float(np.mean(np.abs(he) / target[valid & hidden])), 'rmse_m': float(np.sqrt(np.mean(error ** 2)))}


def cloud_numpy(depth, valid, camera, K):
    y, x = np.indices(depth.shape); uv = np.stack((x + .5, y + .5, np.ones_like(x)), -1)
    local = (uv @ np.linalg.inv(K).T) * depth[..., None]
    return local[valid] @ camera[:3, :3].T + camera[:3, 3]


def raster_numpy(points, camera, K):
    local = (points - camera[:3, 3]) @ camera[:3, :3]
    local = local[np.isfinite(local).all(1) & (local[:, 2] > 1e-4)]
    uvw = local @ K.T; uv = uvw[:, :2] / uvw[:, 2:]
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < 280) & (uv[:, 1] >= 0) & (uv[:, 1] < 210)
    xy = np.floor(uv[inside]).astype(int)
    depth = np.full(210 * 280, np.inf)
    np.minimum.at(depth, xy[:, 1] * 280 + xy[:, 0], local[inside, 2])
    return np.where(np.isfinite(depth), depth, 0).reshape(210, 280)


def support_numpy(meta, sample):
    row = meta['row']; cameras, K = np.asarray(row['c2w_metric'], float), np.asarray(row['K'], float)
    full, masked = {}, {}
    for f in HISTORY:
        d, valid = sensor_target(row['frames'][f]); d, valid = d.numpy().astype(float), valid.numpy()
        full[f] = cloud_numpy(d, valid, cameras[f], K[f])
        masked[f] = cloud_numpy(d, valid & ~sample['metric_masks'][f].numpy(), cameras[f], K[f])
    def project(points): return raster_numpy(points, cameras[11], K[11])
    def merge(a, b): return np.where((a > 0) & (b > 0), np.minimum(a, b), np.maximum(a, b))
    result = {}
    for condition in CONDITIONS:
        recent = project(np.concatenate([masked[f] if condition == 'persistent' and f != 0 else full[f] for f in RECENT]))
        extra = project(np.concatenate([full[f] for f in EXTRA]))
        em = project(np.concatenate([masked[f] for f in EXTRA]))
        result[condition] = {'recent': recent, 'extra': extra, 'history': merge(recent, extra), 'history_mask': merge(recent, em)}
    return result


def check():
    started = time.monotonic(); plan = read(OUT / 'plan.json')
    assert sha256(plan['model_path']) == plan['model_sha256']
    for source, digest in plan['source_sha256'].items():
        assert sha256(ROOT / source) == digest == sha256(OUT / 'source_snapshot' / source)
    for source, digest in plan['official_source_sha256'].items(): assert sha256(DEST / source) == digest
    differences = {k: 0. for k in METRICS}; maximum_resize = 0.; projections = []; outputs = 0
    for meta in plan['catalog']:
        rid = meta['id']; rec = read(OUT / 'evaluation' / f'{rid}.json'); sample = load(CACHE / rid / 'inputs.pt')
        assert rec['complete'] and rec['plan_sha256'] == sha256(OUT / 'plan.json')
        assert sha256(rec['prediction_path']) == rec['prediction_sha256']
        assert rec['input_sha256'] == sha256(CACHE / rid / 'inputs.pt') == plan['records'][rid]['sha256']
        for name, digest in plan['records'][rid]['identities'].items(): assert sha256(name) == digest
        saved = load(rec['prediction_path']); assert len(saved['keys']) == len(rec['rows']) == 8
        expected_maps = support_numpy(meta, sample)
        max_projection_disagreement = 0; max_consistency_disagreement = 0
        for condition in CONDITIONS:
            for name, expected in expected_maps[condition].items():
                actual = saved['support_maps'][condition][name].numpy()
                disagreement = int(np.count_nonzero(~np.isclose(actual, expected, rtol=1e-9, atol=1e-9)))
                max_projection_disagreement = max(max_projection_disagreement, disagreement)
                target = sample['target'].numpy(); valid = sample['valid'].numpy()
                good_expected = (expected > 0) & (np.abs(expected - target) <= .1 * target) & valid
                good_actual = (actual > 0) & (np.abs(actual - target) <= .1 * target) & valid
                max_consistency_disagreement = max(max_consistency_disagreement, int(np.count_nonzero(good_expected != good_actual)))
        assert max_consistency_disagreement < 6, (rid, max_consistency_disagreement)
        for i, row in enumerate(rec['rows']):
            assert saved['keys'][i] == [row['condition'], row['arm']]
            native = saved['depth_native'][i].numpy().astype(float); assert np.isfinite(native).all() and (native > 0).all()
            independent = resize_numpy(native); metric = saved['depth_metric'][i].numpy().astype(float)
            max_resize = float(np.max(np.abs(independent - metric))); maximum_resize = max(maximum_resize, max_resize)
            assert np.allclose(independent, metric, rtol=2e-5, atol=1e-5), (rid, max_resize)
            # Score the saved metric export independently; also verify its resampling above.
            expected = score_numpy(metric, sample)
            for name in METRICS:
                error = abs(row[name] - expected[name]); differences[name] = max(differences[name], error)
                assert np.isclose(row[name], expected[name], rtol=1e-10, atol=1e-10), (rid, name, error)
            assert np.isclose(row['hidden_log_mse'], row['hidden_centered_log_mse'] + row['hidden_bias_squared'], atol=1e-12)
            outputs += 1
        for c in rec['cache_records']:
            if c['arm'] == 'history_drop':
                assert c['removal']['kept_frame_ids'] == list(RECENT)
                assert c['removal']['kept_camera_source_ids'] == [f for f in RECENT for _ in range(4)]
                assert c['removal']['aggregator_removed_entries_per_layer'] == 3
                assert c['removal']['camera_removed_entries_per_layer'] == 12
        projections.append({'id': rid, 'max_projection_pixels_different': max_projection_disagreement,
                             'max_consistent_pixels_different': max_consistency_disagreement})
        print(json.dumps({'checked': rid, 'outputs': 8}), flush=True)
    assert outputs == 96
    write_json(OUT / 'independent_checks.json', {'complete': True, 'predictions_scored': outputs,
        'metric_max_difference': differences, 'independent_resize_max_abs': maximum_resize,
        'source_projection_checks': projections, 'model_and_source_identities_unchanged': True,
        'checker_sha256': sha256(ROOT / 'experiments/gaussian_memory_native_analysis.py'),
        'seconds': time.monotonic() - started})


def paired(baseline, candidate, places):
    a, b = np.asarray(baseline, float), np.asarray(candidate, float); difference = a - b
    rng = np.random.default_rng(20261111); index = rng.integers(len(places), size=(5000, len(places)))
    aa, bb = a[index].mean(1), b[index].mean(1)
    return {'baseline_mean': float(a.mean()), 'candidate_mean': float(b.mean()), 'gain': float(difference.mean()),
        'relative_gain_percent': float(100 * difference.mean() / a.mean()) if a.mean() else None,
        'paired_95': np.quantile(aa - bb, [.025, .975]).tolist(),
        'relative_gain_percent_95': np.quantile(100 * (aa - bb) / aa, [.025, .975]).tolist() if (aa > 0).all() else None,
        'positive_places': int((difference > 0).sum()), 'places': len(places),
        'place_gain': dict(zip(places, difference.tolist())), 'definition': 'baseline minus candidate; positive means lower error'}


def analyze():
    plan = read(OUT / 'plan.json'); catalog = sorted(plan['catalog'], key=lambda m: m['place']); places = [m['place'] for m in catalog]
    records = [read(OUT / 'evaluation' / f'{m["id"]}.json') for m in catalog]
    conditions = {}; matrices = {}
    specs = [('history_vs_recent', 'recent', 'history'), ('history_vs_history_mask', 'history_mask', 'history'),
             ('history_vs_history_drop', 'history_drop', 'history'), ('history_mask_vs_recent', 'recent', 'history_mask'),
             ('history_drop_vs_recent', 'recent', 'history_drop')]
    for condition in CONDITIONS:
        conditions[condition] = {}; matrices[condition] = {}
        for metric in METRICS:
            data = {arm: np.asarray([next(r[metric] for r in rec['rows'] if r['condition'] == condition and r['arm'] == arm) for rec in records]) for arm in ARMS}
            matrices[condition][metric] = {k: v.tolist() for k, v in data.items()}
            conditions[condition][metric] = {'means': {k: float(v.mean()) for k, v in data.items()},
                'comparisons': {key: paired(data[base], data[cand], places) for key, base, cand in specs}}
    interaction = {}
    for metric in METRICS:
        full = np.asarray(matrices['full'][metric]['recent']) - np.asarray(matrices['full'][metric]['history'])
        persistent = np.asarray(matrices['persistent'][metric]['recent']) - np.asarray(matrices['persistent'][metric]['history'])
        result = paired(persistent, full, places)
        for key in ('relative_gain_percent', 'relative_gain_percent_95'): result.pop(key)
        result['definition'] = 'Persistent history gain minus full-input history gain, absolute metric units.'
        interaction[metric] = result
    support = {}; costs = []
    for condition in CONDITIONS:
        support[condition] = {}
        for region in ('roi', 'all'):
            rows = [next(r for r in rec['support'] if r['condition'] == condition and r['region'] == region) for rec in records]
            values = [r['history_gain']['new_consistent_support_fraction'] for r in rows]
            support[condition][region] = {'new_consistent_fraction_mean': float(np.mean(values)),
                'zero_new_consistent_places': [places[i] for i, v in enumerate(values) if not v],
                'new_consistent_fraction_by_place': dict(zip(places, values)),
                'raw_rows': rows}
    for rec in records: costs += [{**c, 'place': rec['place']} for c in rec['cache_records']]
    workers = [read(p) for p in [OUT / 'pilot_worker.json', OUT / 'worker.json'] if p.exists()]
    result = {'complete': True, 'experiment': 'native_history1', 'places': places, 'conditions': conditions,
        'matrices': matrices, 'visibility_interactions': interaction, 'source_support': support,
        'cache_costs': costs, 'cost': {'worker_seconds': [w['seconds'] for w in workers],
            'clip_inference_seconds': sum(r['seconds'] for r in records), 'peak_GiB': max(r['peak_allocated_GiB'] for r in records),
            'outputs': 96, 'new_training_steps': 0, 'model_bytes': __import__('pathlib').Path(plan['model_path']).stat().st_size,
            'scope': 'Depth plus camera execution; point and tracking heads are stored but not evaluated except official replay.'},
        'checkpoint_count': 1, 'fresh_confirmation': False,
        'analysis_source_sha256': sha256(ROOT / 'experiments/gaussian_memory_native_analysis.py')}
    write_json(OUT / 'analysis.json', result)
    print(json.dumps({'primary': conditions['persistent']['hidden_log_mse']['comparisons'],
                      'source_support': {c: {r: v['new_consistent_fraction_mean'] for r, v in d.items()} for c, d in support.items()},
                      'cost': result['cost']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['check', 'analyze']); args = parser.parse_args()
    torch.set_num_threads(4); globals()[args.action]()
