"""Old-development-only sensor interventions; never a deployable input exporter.

The sealed Round7 observation and label arrays remain unchanged. Sensor-derived
oracle arrays are separately serialized and hashed, and never filter the query
loss population. There is intentionally no arbitrary dataset or test CLI.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np

from experiments.round5_rgbd_data import STATUS_NAMES
from experiments.round6_region_data import (
    _load_cache, _project_gt_pixels, load_sensor_depth, make_grid,
    read_scene_selection, resize_pixel_coordinates, sample_nearest_depth,
)
from experiments.round7_depth_data import query_labels


ROOT = Path(__file__).resolve().parents[1]
IMAGE_HW = (144, 192)
STRIDE = 8
ORACLE_FIELDS = (
    'oracle_indices', 'oracle_weights', 'oracle_valid',
    'k_oracle_value_depth', 'k_sensor_fraction',
)
DIAGNOSTIC_FIELDS = (
    'oracle_pixel', 'oracle_status', 'route_clipped',
    'direct_corresponding_pred_depth', 'direct_corresponding_sensor_depth',
)
ALLOWLIST = {
    'harvard_c3/hv_c3_1': ('old', 'train', 192),
    'harvard_c5/hv_c5_1': ('old', 'train', 192),
    'harvard_c6/hv_c6_1': ('old', 'train', 192),
    'harvard_c8/hv_c8_3': ('old', 'val', 64),
    'brown_bm_4/brown_bm_4': ('bm4', 'val', 64),
}
DATASETS = {
    'old': dict(raw='data/round5_sun3d', selection='data/round5_sun3d/selection.json',
                cache='results/round5_inputs/cache', base='results/round7_depth_old'),
    'bm4': dict(raw='data/round6_sun3d', selection='data/round6_sun3d/selection_v2.json',
                cache='results/round6_inputs/cache', base='results/round7_depth_bm4'),
}
SOURCES = (
    'experiments/round8_oracle_data.py', 'experiments/round7_depth_data.py',
    'experiments/round6_region_data.py', 'experiments/round5_rgbd_data.py',
)
CONTRACT = dict(
    schema='round8_oracle_data_v1', diagnostic_only=True, test_access=False,
    query_population='unchanged Round7 positive-finite native-nearest sensor query centers',
    correspondence='native sensor projection of stride8 query center; depth tolerance .05 + .05 * key sensor z',
    correspondence_value='bilinear four stride8 feature centers; continuous grid index (pixel-3.5)/8; border clamp',
    invalid_correspondence='oracle_valid false; training model must retain original predicted attention',
    source_value='nearest-native sensor depth at every 192x144 pixel when valid, otherwise original same-pixel RGB depth',
    source_value_transform='clamp mixed pixel depth .1..20, backproject with resized K, mean each 8x8 3D patch, key-to-query transform, z clamp .1..20',
    source_holes='per-pixel prediction fallback, never zero fill or drop an entire patch',
    fraction='sensor-valid pixel fraction of each source 8x8 patch; diagnostic only',
    geometry_and_visual_features='unchanged sealed Round7 inputs, including predicted covariance',
    no_filtering='correspondence validity or source sensor fraction never changes query labels or masks',
    route_clipped='valid corresponding pixel lies outside convex hull of stride8 cell centers; bilinear weights clamp to center hull',
    direct_predicted_value='diagnostic only: pixel-bilinear original RGB predicted depth at continuous key projection with image-border clamp; continuous resized-K ray; key-to-query transform',
    direct_sensor_value='diagnostic only: nearest native key sensor depth at continuous key projection; continuous native-K ray; key-to-query transform; constrained by correspondence visibility tolerance',
    direct_value_fallback='source depth and final query z clamp .1..20; oracle-invalid rows retain original sealed q_depth; exclude fallback rows only from explicitly eligible diagnostic summaries, not task loss',
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bilinear_correspondence(pixels, valid, image_hw=IMAGE_HW, stride=STRIDE):
    """Sparse four-neighbor weights; repeated border indices require scatter-add.

    Invalid rows contain safe zero indices and zero weights. Their weights must
    not replace the model attention. No Gaussian target distribution is used.
    """
    pixels = np.asarray(pixels, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    h, w = image_hw
    if (pixels.ndim != 2 or pixels.shape[1] != 2 or valid.shape != (len(pixels),)
            or stride <= 0 or h % stride or w % stride):
        raise ValueError('invalid bilinear input shape or stride')
    if not np.isfinite(pixels[valid]).all():
        raise ValueError('a valid correspondence must have finite coordinates')
    inside = ((pixels[:, 0] >= -.5) & (pixels[:, 0] < w-.5)
              & (pixels[:, 1] >= -.5) & (pixels[:, 1] < h-.5))
    if not inside[valid].all():
        raise ValueError('a valid correspondence must be inside the image')
    xy = (np.where(valid[:, None], pixels, 0.) - (stride-1)/2) / stride
    xy = np.clip(xy, (0., 0.), (w//stride-1, h//stride-1))
    lower = np.floor(xy).astype(np.int64)
    upper = np.minimum(lower+1, (w//stride-1, h//stride-1))
    frac = xy-lower
    x0, y0 = lower.T
    x1, y1 = upper.T
    fx, fy = frac.T
    indices = np.stack((y0*(w//stride)+x0, y0*(w//stride)+x1,
                        y1*(w//stride)+x0, y1*(w//stride)+x1), axis=-1)
    weights = np.stack(((1-fx)*(1-fy), fx*(1-fy), (1-fx)*fy, fx*fy), axis=-1)
    return (np.where(valid[:, None], indices, 0).astype(np.int64),
            np.where(valid[:, None], weights, 0.).astype(np.float32))


def oracle_correspondence(qdepth, kdepth, Kq_native, Kk_native, c2wq, c2wk,
                          image_hw=IMAGE_HW, stride=STRIDE):
    pixels, valid, status = _project_gt_pixels(
        make_grid(stride, image_hw).centers, qdepth, kdepth,
        Kq_native, Kk_native, c2wq, c2wk, image_hw, .05, .05)
    indices, weights = bilinear_correspondence(pixels, valid, image_hw, stride)
    h, w = image_hw
    lower = (stride-1)/2
    clipped = valid & ((pixels[:, 0] < lower) | (pixels[:, 0] > w-1-lower)
                       | (pixels[:, 1] < lower) | (pixels[:, 1] > h-1-lower))
    return dict(oracle_indices=indices, oracle_weights=weights, oracle_valid=valid,
                oracle_pixel=pixels, oracle_status=status, route_clipped=clipped)


def bilinear_depth(depth, pixels):
    """Zero-based pixel bilinear interpolation with explicit image-border clamp."""
    depth = np.nan_to_num(np.asarray(depth, np.float64), nan=10., posinf=20., neginf=.1).clip(.1, 20)
    pixels = np.asarray(pixels, np.float64)
    if depth.ndim != 2 or pixels.ndim != 2 or pixels.shape[1] != 2 or not np.isfinite(pixels).all():
        raise ValueError('invalid pixel-bilinear depth inputs')
    h, w = depth.shape
    xy = np.clip(pixels, (0., 0.), (w-1, h-1))
    lower = np.floor(xy).astype(np.int64)
    upper = np.minimum(lower+1, (w-1, h-1))
    fx, fy = (xy-lower).T
    x0, y0 = lower.T
    x1, y1 = upper.T
    return ((1-fx)*(1-fy)*depth[y0, x0] + fx*(1-fy)*depth[y0, x1]
            + (1-fx)*fy*depth[y1, x0] + fx*fy*depth[y1, x1])


def direct_corresponding_depths(pixels, valid, predicted_depth, native_depth,
                               K_resized, K_native, key_c2w, query_c2w, q_fallback):
    """Dense-value diagnostics only; neither returned field is a model input."""
    valid = np.asarray(valid, bool)
    pixels = np.asarray(pixels, np.float64)
    safe = np.where(valid[:, None], pixels, 0.)
    native_pixels = resize_pixel_coordinates(safe, predicted_depth.shape, native_depth.shape)
    sensor, known = sample_nearest_depth(native_depth, native_pixels)
    if not known[valid].all():
        raise ValueError('visible oracle row must have valid key sensor depth')
    values = {}
    for name, xy, depth, K in (
        ('pred', safe, bilinear_depth(predicted_depth, safe), K_resized),
        ('sensor', native_pixels, np.where(known, sensor, 1.).clip(.1, 20), K_native),
    ):
        rays = np.column_stack((xy, np.ones(len(xy)))) @ np.linalg.inv(np.asarray(K, np.float64)).T
        z = transformed_value_depth(rays*depth[:, None], key_c2w, query_c2w)
        values['direct_corresponding_'+name+'_depth'] = np.where(valid, z, q_fallback).astype(np.float32)
    return values


def sensor_patch_points(native_depth, predicted_depth, K_resized, stride=STRIDE):
    """Sensor/prediction pixel mixture and means in the source camera frame."""
    pred = np.asarray(predicted_depth, dtype=np.float64)
    if pred.ndim != 2 or pred.shape[0] % stride or pred.shape[1] % stride:
        raise ValueError('predicted depth must be a stride-divisible 2D array')
    h, w = pred.shape
    pixels = make_grid(1, (h, w)).centers
    sensor, known = sample_nearest_depth(
        native_depth, resize_pixel_coordinates(pixels, (h, w), native_depth.shape))
    pred = np.nan_to_num(pred, nan=10., posinf=20., neginf=.1).clip(.1, 20)
    mixed = np.where(known, sensor, pred.reshape(-1)).clip(.1, 20)
    K = np.asarray(K_resized, dtype=np.float64)
    if K.shape != (3, 3) or not np.isfinite(K).all():
        raise ValueError('invalid resized camera matrix')
    rays = np.column_stack((pixels, np.ones(len(pixels)))) @ np.linalg.inv(K).T
    points = (rays*mixed[:, None]).reshape(h//stride, stride, w//stride, stride, 3)
    mean3 = points.mean((1, 3)).reshape(-1, 3)
    fraction = known.reshape(h//stride, stride, w//stride, stride).mean((1, 3)).reshape(-1)
    return mean3, fraction.astype(np.float32)


def transformed_value_depth(mean3, key_c2w, query_c2w):
    transform = np.linalg.inv(np.asarray(query_c2w, np.float64)) @ np.asarray(key_c2w, np.float64)
    points = np.asarray(mean3, np.float64) @ transform[:3, :3].T + transform[:3, 3]
    if not np.isfinite(points).all():
        raise ValueError('nonfinite sensor-derived 3D points')
    return points[:, 2].clip(.1, 20).astype(np.float32)


def selected_rows(groups):
    """Only hard-coded old development scenes can pass this entry point."""
    if not groups or len(set(groups)) != len(groups) or not set(groups) <= set(DATASETS):
        raise ValueError('groups must be distinct fixed old development datasets')
    selected = []
    for group in groups:
        spec = {k: ROOT/v for k, v in DATASETS[group].items()}
        selection = json.loads(spec['selection'].read_text())
        rows = {row['scene']: row for row in selection['scenes']}
        for scene, (expected_group, split, count) in ALLOWLIST.items():
            if expected_group != group:
                continue
            row = rows[scene]
            if row['split'] != split or len(row['pairs']) != count or split not in ('train', 'val'):
                raise ValueError('changed development selection identity or pair count')
            dest = spec['base']/split/scene.replace('/', '__')
            manifest = json.loads((dest/'manifest.json').read_text())
            if (manifest['scene'] != scene or manifest['split'] != split
                    or manifest['pairs'] != row['pairs']
                    or manifest['selection_sha256'] != sha(spec['selection'])):
                raise ValueError('Round7 base and native selection mismatch')
            for filename, name in [('inputs.npz', 'input_sha256'), ('labels.npz', 'label_sha256')]:
                if sha(dest/filename) != manifest[name]:
                    raise ValueError('changed sealed Round7 base arrays')
            selected.append((spec, row, dest, manifest))
    return selected


def export(out, groups=('old', 'bm4')):
    out = Path(out)
    if out.exists():
        raise ValueError('refuse overwrite '+str(out))
    selected = selected_rows(groups)
    hashes = {p: sha(ROOT/p) for p in SOURCES}
    frozen = dict(contract=CONTRACT, source_hashes=hashes, groups=list(groups),
                  scenes=[dict(scene=row['scene'], split=row['split'], pairs=row['pairs'],
                    selection_sha256=sha(spec['selection']),
                    base_manifest_sha256=sha(base/'manifest.json'),
                    base_input_sha256=m['input_sha256'], base_label_sha256=m['label_sha256'])
                    for spec, row, base, m in selected])
    out.mkdir(parents=True)
    (out/'frozen_export_protocol.json').write_text(json.dumps(frozen, indent=2))
    for path in SOURCES:
        dest = out/'source'/path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/path, dest)
    started = time.time()
    summaries = []
    for spec, row, base, manifest in selected:
        scene = row['scene']
        dest = out/row['split']/scene.replace('/', '__')
        dest.mkdir(parents=True)
        frames = read_scene_selection(spec['raw'], row)
        depths, caches, means, fractions, provenance = {}, {}, {}, {}, []
        needed = sorted({i for pair in row['pairs'] for i in pair})
        keys = {k for q, k in row['pairs']}
        for index in needed:
            frame = frames[index]
            depths[index] = load_sensor_depth(frame.depth_path)
            path = spec['cache']/scene/f'{index:07d}.npz'
            cache, sidecar = _load_cache(path, frame, row)
            # Features are never recomputed or mixed into the oracle arrays.
            caches[index] = {k: cache[k] for k in ('predicted_depth', 'K', 'c2w')}
            if index in keys:
                means[index], fractions[index] = sensor_patch_points(
                    depths[index], cache['predicted_depth'], cache['K'])
            provenance.append(dict(index=index, native_depth_path=str(frame.depth_path),
                native_depth_sha256=sha(frame.depth_path), cache_path=str(path),
                cache_sha256=sha(path), cache_sidecar_sha256=sha(path.with_suffix('.json')),
                rgb_sha256=sidecar['producer']['rgb_sha256']))
        arrays = {f: [] for f in ORACLE_FIELDS+DIAGNOSTIC_FIELDS}
        with np.load(base/'labels.npz', allow_pickle=False) as saved:
            labels = {f: saved[f] for f in ('target', 'valid', 'boundary')}
        with np.load(base/'inputs.npz', allow_pickle=False) as saved:
            base_value = saved['k_value_depth']
            q_fallback = saved['q_depth']
        rows = []
        for pi, (qi, ki) in enumerate(row['pairs']):
            # Recompute solely to verify the original all-query population.
            recomputed = query_labels(depths[qi])
            for field in recomputed:
                np.testing.assert_array_equal(recomputed[field], labels[field][pi])
            q, k = frames[qi], frames[ki]
            pair = oracle_correspondence(depths[qi], depths[ki], q.K_native, k.K_native,
                                          q.c2w, k.c2w)
            pair['k_oracle_value_depth'] = transformed_value_depth(means[ki], k.c2w, q.c2w)
            pair['k_sensor_fraction'] = fractions[ki]
            pair.update(direct_corresponding_depths(pair['oracle_pixel'], pair['oracle_valid'],
                caches[ki]['predicted_depth'], depths[ki], caches[ki]['K'], k.K_native,
                k.c2w, q.c2w, q_fallback[pi]))
            # Verify the interpolation/fallback frame convention against sealed V.
            pred_points, _ = sensor_patch_points(np.zeros_like(depths[ki]),
                caches[ki]['predicted_depth'], caches[ki]['K'])
            np.testing.assert_allclose(transformed_value_depth(pred_points, k.c2w, q.c2w),
                                       base_value[pi], rtol=1e-6, atol=2e-6)
            for field in arrays:
                arrays[field].append(pair[field])
            count = np.bincount(pair['oracle_status'].astype(np.int64), minlength=len(STATUS_NAMES))
            valid = labels['valid'][pi]
            rows.append(dict(q_index=qi, k_index=ki, pair_id=f'{qi}:{ki}',
                query_valid_count=int(valid.sum()), oracle_valid_count=int(pair['oracle_valid'].sum()),
                query_valid_oracle_valid_count=int((valid & pair['oracle_valid']).sum()),
                key_sensor_fraction_mean=float(fractions[ki].mean()),
                key_sensor_full_count=int((fractions[ki] == 1).sum()),
                key_sensor_empty_count=int((fractions[ki] == 0).sum()),
                route_clipped_count=int(pair['route_clipped'].sum()),
                statuses={name: int(count[i]) for i, name in STATUS_NAMES.items()}))
        np.savez_compressed(dest/'oracle.npz', **{f: np.stack(v) for f, v in arrays.items()})
        total = sum(r['query_valid_count'] for r in rows)
        covered = sum(r['query_valid_oracle_valid_count'] for r in rows)
        m = dict(scene=scene, split=row['split'], pairs=row['pairs'], diagnostic_only=True,
            oracle_sha256=sha(dest/'oracle.npz'), exporter_sha256=hashes[SOURCES[0]],
            frozen_protocol_sha256=sha(out/'frozen_export_protocol.json'),
            base_path=str(base), base_manifest_sha256=sha(base/'manifest.json'),
            base_input_sha256=manifest['input_sha256'], base_label_sha256=manifest['label_sha256'],
            selection_path=str(spec['selection']), selection_sha256=sha(spec['selection']),
            source_hashes=hashes, sensor_and_cache_sources=provenance, pair_rows=rows,
            summary=dict(total_positions=len(rows)*432, query_valid_count=total,
                oracle_covered_query_valid_count=covered,
                oracle_coverage_of_query_valid=covered/total,
                key_sensor_fraction_mean=float(np.mean([r['key_sensor_fraction_mean'] for r in rows]))))
        (dest/'manifest.json').write_text(json.dumps(m, indent=2))
        summaries.append(dict(scene=scene, split=row['split'], pairs=len(rows),
            oracle_sha256=m['oracle_sha256'], manifest_sha256=sha(dest/'manifest.json'), **m['summary']))
        print('EXPORTED', scene, len(rows), 'query-valid correspondence coverage', round(covered/total, 6), flush=True)
    for path, digest in hashes.items():
        if sha(ROOT/path) != digest:
            raise ValueError('export source changed during execution: '+path)
    (out/'completion.json').write_text(json.dumps(dict(summaries=summaries,
        seconds=time.time()-started, frozen_protocol_sha256=sha(out/'frozen_export_protocol.json')),
        indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'results/round8_oracle_data')
    parser.add_argument('--groups', nargs='+', choices=tuple(DATASETS), default=list(DATASETS))
    args = parser.parse_args()
    export(args.out, args.groups)


if __name__ == '__main__':
    main()
