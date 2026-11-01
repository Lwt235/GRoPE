"""Region-area RGB-D supervision and independent RGB-predicted Gaussian inputs.

Every 192x144 source pixel center contributes equal *source-image area* mass.
GT projection/visibility is supervision only. No predicted-geometry quality
chooses a label or evaluation row. Forward support is continuous pixel-footprint
+ 3D scatter of frozen RGB-predicted depth, not optimized 3DGS primitive R/S.
All forward 2D coordinates share the fixed reference unit 8 pixels, even when
the two images use different token strides. Frozen Round5 files are untouched.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import time
from typing import Any

import numpy as np

from experiments.round5_rgbd_data import (
    DEFAULT_IMAGE_HW, STATUS_NAMES, VALID, QUERY_DEPTH_MISSING, BEHIND_CAMERA,
    OUTSIDE_IMAGE, KEY_DEPTH_MISSING, DEPTH_INCONSISTENT, NONFINITE_PROJECTION,
    NEAREST_TIE_EPSILON, _K, _pose, token_grid, resize_pixel_coordinates,
    sample_nearest_depth, project_points_and_jacobian, load_sensor_depth,
    read_sun3d_intrinsics, read_sun3d_extrinsics, Sun3DFrame, _listing,
    sun3d_timestamp, resize_intrinsics, sha256,
)


REFERENCE_STRIDE = 8
BASE_FEATURE_STRIDE = 8
ALLOWED_STRIDES = (8, 16)


@dataclass(frozen=True)
class Grid:
    image_hw: tuple[int, int]
    stride: int
    centers: np.ndarray
    bounds: np.ndarray  # [xmin,ymin,xmax,ymax], half-open zero-based support.

    @property
    def token_count(self):
        return len(self.centers)


def make_grid(stride=8, image_hw=DEFAULT_IMAGE_HW) -> Grid:
    centers = token_grid(image_hw, stride)
    bounds = np.concatenate([centers-stride/2, centers+stride/2], axis=-1)
    return Grid(tuple(image_hw), int(stride), centers, bounds)


def containing_cells(pixels, grid: Grid):
    """Hard containing cell = nearest regular-grid center, with stable boundary ties."""
    pixels = np.asarray(pixels, np.float64)
    if pixels.ndim != 2 or pixels.shape[1] != 2:
        raise ValueError("pixels must have shape [N,2]")
    h, w = grid.image_hw
    finite = np.isfinite(pixels).all(-1)
    safe = np.where(np.isfinite(pixels), pixels, 0)
    inside = finite & (safe[:, 0] >= -.5) & (safe[:, 0] < w-.5)
    inside &= (safe[:, 1] >= -.5) & (safe[:, 1] < h-.5)
    xy = np.floor((safe+.5+NEAREST_TIE_EPSILON)/grid.stride)
    xy = np.clip(xy, [0, 0], [w//grid.stride-1, h//grid.stride-1]).astype(np.int64)
    indices = xy[:, 1]*(w//grid.stride)+xy[:, 0]
    return np.where(inside, indices, -1), inside


def _project_gt_pixels(pixels, qdepth, kdepth, Kq, Kk, c2wq, c2wk,
                       image_hw, absolute_tolerance, relative_tolerance):
    """Float64 version of frozen Round5 visibility, at arbitrary source pixels."""
    qdepth, kdepth = np.asarray(qdepth), np.asarray(kdepth)
    if qdepth.ndim != 2 or kdepth.ndim != 2:
        raise ValueError("native sensor depths must be 2D meters arrays")
    if not np.isfinite([absolute_tolerance, relative_tolerance]).all() or min(absolute_tolerance, relative_tolerance)<0:
        raise ValueError("depth tolerances must be finite and nonnegative")
    Kq, Kk = _K(Kq), _K(Kk)
    transform = np.linalg.inv(_pose(c2wk))@_pose(c2wq)
    qnative = resize_pixel_coordinates(pixels, image_hw, qdepth.shape)
    z, known = sample_nearest_depth(qdepth, qnative)
    rays = np.column_stack([qnative, np.ones(len(pixels))])@np.linalg.inv(Kq).T
    qpoints = rays*np.where(known, z, 0)[:, None]
    kpoints = qpoints@transform[:3, :3].T+transform[:3, 3]
    knative, _, projected = project_points_and_jacobian(kpoints, Kk)
    target = resize_pixel_coordinates(knative, kdepth.shape, image_hw)
    _, inside = containing_cells(target, make_grid(1, image_hw))
    kz, kknown = sample_nearest_depth(kdepth, knative)
    consistent = np.abs(kpoints[:, 2]-kz) <= absolute_tolerance+relative_tolerance*kz
    status = np.full(len(pixels), VALID, np.uint8)
    active = known.copy()
    status[~active] = QUERY_DEPTH_MISSING
    for condition, reason in [(np.isfinite(kpoints).all(-1), NONFINITE_PROJECTION),
                              (projected, BEHIND_CAMERA), (inside, OUTSIDE_IMAGE),
                              (kknown, KEY_DEPTH_MISSING), (consistent, DEPTH_INCONSISTENT)]:
        rejected = active & ~condition
        status[rejected] = reason
        active &= condition
    target = np.where((known & projected)[:, None], target, np.nan)
    return target, active, status


@dataclass(frozen=True)
class RegionSupervision:
    soft_labels: np.ndarray
    target_counts: np.ndarray
    gt_valid: np.ndarray
    visible_fraction: np.ndarray
    sample_count: np.ndarray
    counts_by_reason: np.ndarray
    point_gt_valid: np.ndarray
    point_pixcontinuous: np.ndarray
    point_key_index: np.ndarray
    metadata: dict[str, Any]


def region_supervision(query_depth_native, key_depth_native, K_query_native,
                       K_key_native, c2w_query, c2w_key, q_stride=8, k_stride=8,
                       image_hw=DEFAULT_IMAGE_HW, absolute_tolerance=.05,
                       relative_tolerance=.05) -> RegionSupervision:
    """Push uniform source-image pixel mass into hard target token cells.

    ALL resized 192x144 integer pixel centers are used: 64/256 samples per
    8/16-pixel query patch. Native sensor depth is nearest-sampled, with exactly
    the Round5 consistency tolerance. Normalize visible mass only; gt_valid is
    visible_count>0 with no quality/visibility-fraction cutoff. All counts and
    fractions belong exclusively to supervision.
    """
    qgrid, kgrid = make_grid(q_stride, image_hw), make_grid(k_stride, image_hw)
    h, w = image_hw
    yy, xx = np.mgrid[:h, :w]
    pixels = np.stack([xx, yy], -1).reshape(-1, 2).astype(np.float64)
    qindex, inside = containing_cells(pixels, qgrid)
    if not inside.all():
        raise ValueError("source pixel grid is not fully covered")
    target, visible, status = _project_gt_pixels(pixels, query_depth_native, key_depth_native,
        K_query_native, K_key_native, c2w_query, c2w_key, image_hw,
        absolute_tolerance, relative_tolerance)
    kindex, target_inside = containing_cells(target, kgrid)
    if np.any(visible & ~target_inside):
        raise ValueError("visible projection outside target grid")
    counts = np.bincount(qindex[visible]*kgrid.token_count+kindex[visible],
        minlength=qgrid.token_count*kgrid.token_count).reshape(qgrid.token_count, kgrid.token_count).astype(np.int32)
    samples = np.bincount(qindex, minlength=qgrid.token_count).astype(np.int32)
    visible_count = counts.sum(-1)
    valid = visible_count>0
    soft = np.divide(counts, visible_count[:, None], out=np.zeros(counts.shape, np.float64),
                     where=visible_count[:, None]>0).astype(np.float32)
    reasons = np.bincount(qindex*len(STATUS_NAMES)+status,
        minlength=qgrid.token_count*len(STATUS_NAMES)).reshape(qgrid.token_count, len(STATUS_NAMES)).astype(np.int32)
    np.testing.assert_array_equal(reasons.sum(-1), samples)
    np.testing.assert_array_equal(reasons[:, VALID], visible_count)

    # Independent center-point target: never substitute the visible region centroid.
    point_xy, point_valid, _ = _project_gt_pixels(qgrid.centers, query_depth_native,
        key_depth_native, K_query_native, K_key_native, c2w_query, c2w_key, image_hw,
        absolute_tolerance, relative_tolerance)
    point_key, _ = containing_cells(point_xy, kgrid)
    point_key = np.where(point_valid, point_key, -1).astype(np.int32)
    metadata = {"q_stride": int(q_stride), "k_stride": int(k_stride), "image_hw": list(image_hw),
        "query_tokens": qgrid.token_count, "key_tokens": kgrid.token_count,
        "area_measure": "uniform source-image pixel-center measure, not 3D surface area",
        "sampling": "every resized image pixel center once; native sensor nearest floor(x+.5+1e-9)",
        "target_assignment": "hard half-open containing key cell; 1e-9 pixel guard at exact boundaries",
        "normalization": "visible source mass only; no visibility-fraction or predicted-quality cutoff",
        "absolute_tolerance_m": absolute_tolerance, "relative_tolerance": relative_tolerance,
        "depth_consistency": "abs(projected_z-sensor_z)<=absolute+relative*sensor_z",
        "reason_columns": [STATUS_NAMES[i] for i in range(len(STATUS_NAMES))],
        "point_target": "independent GT sample at query patch center; continuous target in 192x144 pixels",
        "secondary_pck_threshold_pixels": int(k_stride),
        "total_source_samples": len(pixels), "total_visible_samples": int(visible_count.sum())}
    return RegionSupervision(soft, counts, valid, (visible_count/samples).astype(np.float32),
        samples, reasons, point_valid, point_xy.astype(np.float32), point_key, metadata)


def _patches(values, stride):
    h, w = values.shape[:2]
    make_grid(stride, (h, w))
    trailing = values.shape[2:]
    return values.reshape(h//stride, stride, w//stride, stride, *trailing).transpose(
        0, 2, 1, 3, *range(4, values.ndim+2)).reshape(-1, stride**2, *trailing)


def pool_features(features, stride=8, image_hw=DEFAULT_IMAGE_HW):
    """Frozen 8px features, or 2x2 area average followed by L2 for 16px tokens."""
    base = make_grid(BASE_FEATURE_STRIDE, image_hw)
    features = np.asarray(features)
    if features.ndim != 2 or len(features) != base.token_count or not np.isfinite(features).all():
        raise ValueError("expected finite [base_token_count,channels] frozen RGB features")
    if stride not in ALLOWED_STRIDES:
        raise ValueError("forward feature strides supported here are 8 and 16")
    if stride == BASE_FEATURE_STRIDE:
        return features.astype(np.float32, copy=True)
    h, w = image_hw
    dense = features.astype(np.float64).reshape(h//8, w//8, -1)
    pooled = _patches(dense, stride//8).mean(1)
    norm = np.linalg.norm(pooled, axis=-1, keepdims=True)
    return (pooled/np.maximum(norm, 1e-12)).astype(np.float32)


def fit_predicted_support(predicted_depth, K, stride=8):
    """3D point scatter plus within-pixel physical area; no sensor/GT inputs.

    Depth is constant within each 1x1 pixel footprint. Its image-plane covariance
    is I/12, backprojected by z*inv(K)[:,:2]. This adds the missing within-pixel
    variance exactly for a constant-depth plane without double-counting patch
    scatter. It is support geometry, not calibrated depth uncertainty.
    """
    raw = np.asarray(predicted_depth, np.float64)
    if raw.ndim != 2:
        raise ValueError("predicted depth must be a resized 2D array")
    h, w = raw.shape
    make_grid(stride, raw.shape)
    K = _K(K)
    z = np.nan_to_num(raw, nan=10., posinf=20., neginf=.1).clip(.1, 20.)
    yy, xx = np.mgrid[:h, :w]
    rays = np.stack([xx, yy, np.ones_like(xx)], -1)@np.linalg.inv(K).T
    points = _patches(rays*z[..., None], stride)
    mean = points.mean(1)
    centered = points-mean[:, None]
    scatter = np.einsum('nsi,nsj->nij', centered, centered)/stride**2
    A = np.linalg.inv(K)[:, :2]
    z2 = _patches(z[..., None]**2, stride).mean(1)[:, 0]
    footprint = z2[:, None, None]*(A@A.T)[None]/12
    diagnostics = {"nonfinite_depth_pixels": int((~np.isfinite(raw)).sum()),
        "depth_pixels_clipped": int(((np.isfinite(raw)) & ((raw<.1)|(raw>20))).sum()),
        "source_pixels": int(raw.size), "tokens": len(mean),
        "within_pixel_footprint": "continuous 1x1 I/12 backprojected at RGB-predicted z",
        "gt_or_sensor_inputs_used": False}
    return mean, scatter+footprint, diagnostics


def projected_support(mean, covariance, source_c2w, target_c2w, target_K,
                      reference_stride=REFERENCE_STRIDE):
    """Linearized cross-view Gaussian with fixed finite Round5-style fallbacks."""
    transform = np.linalg.inv(_pose(target_c2w))@_pose(source_c2w)
    mean = np.asarray(mean, np.float64)@transform[:3, :3].T+transform[:3, 3]
    covariance = transform[:3, :3]@np.asarray(covariance, np.float64)@transform[:3, :3].T
    safe = mean.copy()
    safe[:, 2] = np.maximum(safe[:, 2], .1)
    pixels, J, _ = project_points_and_jacobian(safe, target_K)
    projected = J@covariance@J.transpose(0, 2, 1)/reference_stride**2
    values, vectors = np.linalg.eigh((projected+projected.transpose(0, 2, 1))/2)
    projected = (vectors*values.clip(0, 256)[:, None, :])@vectors.transpose(0, 2, 1)
    xy = pixels/reference_stride
    diagnostics = {"behind_or_near": int((mean[:, 2]<.1).sum()),
        "covariance_clipped": int((values.max(-1)>256).sum()),
        "mean_clipped": int((np.abs(xy)>128).any(-1).sum()),
        "fallback": "z>=.1m; projected mean clipped +/-128 reference units; cov eigenvalues clipped[0,256]; no row removal"}
    return xy.clip(-128, 128).astype(np.float32), projected.astype(np.float32), diagnostics


@dataclass(frozen=True)
class PreparedFrame:
    features: np.ndarray
    mean3: np.ndarray
    cov3: np.ndarray
    K: np.ndarray
    c2w: np.ndarray
    stride: int
    image_hw: tuple[int, int]
    diagnostics: dict[str, Any]


def prepare_frame(cache, stride=8) -> PreparedFrame:
    """Reads only frozen RGB features/predicted depth and shared camera metadata."""
    depth = np.asarray(cache['predicted_depth'])
    if depth.shape != DEFAULT_IMAGE_HW:
        raise ValueError("Round6 cached RGB prediction must cover full 192x144 image")
    features = pool_features(cache['features'], stride, depth.shape)
    mean, covariance, diagnostics = fit_predicted_support(depth, cache['K'], stride)
    return PreparedFrame(features, mean, covariance, _K(cache['K']).copy(),
        _pose(cache['c2w']).copy(), int(stride), tuple(depth.shape), diagnostics)


def make_pair_inputs(q, k, q_stride=8, k_stride=8):
    """Return exactly the model's ten forward fields plus diagnostics; no GT args."""
    q = q if isinstance(q, PreparedFrame) else prepare_frame(q, q_stride)
    k = k if isinstance(k, PreparedFrame) else prepare_frame(k, k_stride)
    if q.stride != q_stride or k.stride != k_stride or q.image_hw != k.image_hw:
        raise ValueError("prepared frames disagree with requested grids")
    inputs = {'q': q.features.copy(), 'k': k.features.copy()}
    diagnostics = {'query_prediction': q.diagnostics, 'key_prediction': k.diagnostics}
    for side, frame in [('q', q), ('k', k)]:
        grid = make_grid(frame.stride, frame.image_hw)
        inputs[side+'mean'] = (grid.centers/REFERENCE_STRIDE).astype(np.float32)
        inputs[side+'cov'] = np.broadcast_to(np.eye(2)*(frame.stride**2/(12*REFERENCE_STRIDE**2)),
            (grid.token_count, 2, 2)).astype(np.float32).copy()
    for name, source, target in [('k_to_q', k, q), ('q_to_k', q, k)]:
        mean, covariance, diag = projected_support(source.mean3, source.cov3,
            source.c2w, target.c2w, target.K)
        inputs[name], inputs[name+'cov'] = mean, covariance
        diagnostics[name] = diag
    if not all(np.isfinite(array).all() for array in inputs.values()):
        raise ValueError("nonfinite forward input after fixed prediction fallback")
    return inputs, diagnostics


def _load_cache(path, frame, scene_row):
    sidecar = json.loads(path.with_suffix('.json').read_text())
    if sha256(path) != sidecar['cache_sha256']:
        raise ValueError("frozen RGB cache hash mismatch")
    producer = sidecar['producer']
    if (producer['rgb_sha256'] != sha256(frame.rgb_path)
            or producer['K_sha256'] != scene_row['K_sha256']
            or producer['pose_sha256'] != scene_row['pose_sha256']
            or producer['original_index'] != frame.original_index):
        raise ValueError("frozen RGB cache provenance mismatch")
    with np.load(path, allow_pickle=False) as values:
        cache = {key: values[key].copy() for key in ['features','predicted_depth','K','c2w']}
    np.testing.assert_array_equal(cache['c2w'], frame.c2w)
    from PIL import Image
    with Image.open(frame.rgb_path) as image:
        expected_K = resize_intrinsics(frame.K_native, (image.height, image.width))
    np.testing.assert_allclose(cache['K'], expected_K, atol=1e-12, rtol=0)
    return cache, sidecar


def read_scene_selection(raw_root, row):
    """Native SUN3D reader accepting an external selection manifest's scene row.

    This is the Round5 format contract, generalized without editing its frozen
    loader or forcing a new selection into raw_root/selection.json.
    """
    directory = Path(raw_root)/row['scene']
    K_path, pose_path = directory/'intrinsics.txt', Path(raw_root)/row['pose_file']
    if sha256(K_path) != row['K_sha256'] or sha256(pose_path) != row['pose_sha256']:
        raise ValueError('native camera metadata hash mismatch')
    K, poses = read_sun3d_intrinsics(K_path), read_sun3d_extrinsics(pose_path)
    rgbs, depths = _listing(directory/'image_index.html','jpg'), _listing(directory/'depth_index.html','png')
    if len(rgbs) != row['rgb_count'] or len(poses) != len(rgbs) or len(depths) != row['depth_count']:
        raise ValueError('full original image/depth/pose count mismatch')
    timestamps = np.array([sun3d_timestamp(name) for name in depths], np.int64)
    frames = {}
    for item in row['frames']:
        index = int(item['index'])
        if index < 0 or index >= len(rgbs) or rgbs[index] != item['rgb'] or index in frames:
            raise ValueError('selected RGB original index mismatch or duplicate')
        gaps = np.abs(timestamps-sun3d_timestamp(item['rgb']))
        nearest = int(gaps.argmin())
        if depths[nearest] != item['depth'] or gaps[nearest] != item['time_gap_us']:
            raise ValueError('depth timestamp association mismatch')
        frames[index] = Sun3DFrame(f"{row['scene']}/{item['rgb']}", index,
            directory/'image'/item['rgb'], directory/'depth'/item['depth'],
            K.copy(), poses[index].copy(), int(gaps[nearest]))
    if any(q not in frames or k not in frames for q,k in row['pairs']):
        raise ValueError('pair refers to an unselected sparse frame')
    return frames


def _authorize_splits(args, selection_path):
    if 'test' not in args.splits:
        return None
    if not args.allow_test or args.protocol_lock is None:
        raise ValueError('test requires --allow-test and a frozen --protocol-lock')
    lock = json.loads(args.protocol_lock.read_text())
    if (lock.get('selection_sha256') != sha256(selection_path)
            or lock.get('exporter_sha256') != sha256(__file__)
            or lock.get('test_export_authorized') is not True):
        raise ValueError('test protocol lock must authorize these exact selection/exporter hashes')
    return {'path':str(args.protocol_lock),'sha256':sha256(args.protocol_lock)}


def export(args):
    """Default old train/val. Heldout test requires an explicit immutable lock."""
    started = time.time()
    if args.limit_pairs <= 0 or len(set(args.strides)) != len(args.strides):
        raise ValueError('positive pair limit and unique strides are required')
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("refusing to overwrite region export; use a new empty output")
    selection_path = args.selection or args.data_root/'selection.json'
    lock = _authorize_splits(args, selection_path)
    selection = json.loads(selection_path.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = {'scope': 'train/val development' if lock is None else 'protocol-locked heldout test export',
        'strides': args.strides, 'reference_stride': REFERENCE_STRIDE, 'limit_pairs_per_scene': args.limit_pairs,
        'pair_rule': 'first selected pairs in frozen old selection, no GT/prediction-quality selection',
        'selection_sha256': sha256(selection_path), 'source_sha256': sha256(__file__),
        'round5_gt_helper_sha256': sha256(Path(__file__).with_name('round5_rgbd_data.py')),
        'test_protocol_lock': lock, 'raw_root':str(args.data_root), 'cache_root':str(args.cache_root),
        'forward_support': 'RGB-predicted point scatter + continuous within-pixel footprint; no optimized 3DGS R/S',
        'native_footprint': 'continuous square stride^2/(12*8^2) I in common8px units',
        'model_blur_floor': 'not added here; model owns it', 'sensor_labels_forward_mask': False}
    (args.output/'frozen_export_protocol.json').write_text(json.dumps(protocol, indent=2))
    shutil.copy2(__file__,args.output/'source_snapshot.py')
    shutil.copy2(Path(__file__).with_name('round5_rgbd_data.py'),args.output/'round5_gt_helper_snapshot.py')
    summaries = []
    for row in selection['scenes']:
        if row['split'] not in args.splits:
            continue
        if row['split'] not in ['train','val'] and lock is None:
            raise ValueError("new test export is forbidden before independent protocol freeze")
        frames = read_scene_selection(args.data_root, row)
        pairs = row['pairs'][:args.limit_pairs]
        selected = sorted(set(index for pair in pairs for index in pair))
        prepared, provenance = {}, {}
        for index in selected:
            path = args.cache_root/row['scene']/f'{index:07d}.npz'
            cache, sidecar = _load_cache(path, frames[index], row)
            provenance[str(path)] = sidecar
            for stride in args.strides:
                prepared[index, stride] = prepare_frame(cache, stride)
        inputs, diagnostics = {}, {}
        for qs in args.strides:
            for ks in args.strides:
                key = f'q{qs}_k{ks}'
                inputs[key], diagnostics[key] = [], []
                for qi, ki in pairs:
                    value, diag = make_pair_inputs(prepared[qi, qs], prepared[ki, ks], qs, ks)
                    inputs[key].append(value)
                    diagnostics[key].append(diag)
        # No sensor depth has been read before every forward input above exists.
        depth_cache = {index: load_sensor_depth(frames[index].depth_path) for index in selected}
        labels = {key: [] for key in inputs}
        for qi, ki in pairs:
            for qs in args.strides:
                for ks in args.strides:
                    labels[f'q{qs}_k{ks}'].append(region_supervision(depth_cache[qi], depth_cache[ki],
                        frames[qi].K_native, frames[ki].K_native, frames[qi].c2w, frames[ki].c2w, qs, ks))
        label_fields = ['soft_labels','target_counts','gt_valid','visible_fraction','sample_count',
                        'counts_by_reason','point_gt_valid','point_pixcontinuous','point_key_index']
        for key in inputs:
            destination = args.output/row['split']/row['scene'].replace('/','__')/key
            destination.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(destination/'inputs.npz', **{field: np.stack([value[field] for value in inputs[key]]) for field in inputs[key][0]})
            np.savez_compressed(destination/'labels.npz', **{field: np.stack([getattr(value, field) for value in labels[key]]) for field in label_fields})
            summary = {'scene': row['scene'], 'split': row['split'], 'grid': key, 'pairs': pairs,
                'input_sha256': sha256(destination/'inputs.npz'), 'label_sha256': sha256(destination/'labels.npz'),
                'label_metadata': labels[key][0].metadata, 'projection_diagnostics': diagnostics[key],
                'rgb_cache_provenance': provenance,
                'sensor_depth_sha256': {str(frames[index].depth_path): sha256(frames[index].depth_path) for index in selected},
                'gt_valid_queries': int(sum(value.gt_valid.sum() for value in labels[key])),
                'point_gt_valid_queries': int(sum(value.point_gt_valid.sum() for value in labels[key])),
                'visible_source_pixel_fraction': float(np.mean([value.visible_fraction.mean() for value in labels[key]]))}
            (destination/'manifest.json').write_text(json.dumps(summary, indent=2))
            summaries.append({key: value for key, value in summary.items() if key not in ['rgb_cache_provenance','sensor_depth_sha256','projection_diagnostics']})
            print(row['scene'], key, len(pairs), 'pairs', summary['gt_valid_queries'], 'area-valid queries', flush=True)
    if sha256(__file__) != protocol['source_sha256'] or sha256(selection_path) != protocol['selection_sha256']:
        raise ValueError("export source/selection changed during run")
    if not summaries:
        raise ValueError('no matching scenes in selection')
    (args.output/'summary.json').write_text(json.dumps(summaries, indent=2))
    (args.output/'completion.json').write_text(json.dumps({'elapsed_seconds':time.time()-started,
        'scene_grid_conditions':len(summaries), 'pairs_across_conditions':sum(len(row['pairs']) for row in summaries),
        'summary_sha256':sha256(args.output/'summary.json'), 'source_sha256':protocol['source_sha256']},indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root','--raw-root', dest='data_root', type=Path, default=Path('data/round5_sun3d'))
    parser.add_argument('--selection', type=Path)
    parser.add_argument('--cache-root', type=Path, default=Path('results/round5_inputs/cache'))
    parser.add_argument('--output','--out-root', dest='output', type=Path, default=Path('results/round6_region_inputs'))
    parser.add_argument('--splits', nargs='+', choices=['train','val','test'], default=['train','val'])
    parser.add_argument('--allow-test',action='store_true')
    parser.add_argument('--protocol-lock',type=Path)
    parser.add_argument('--strides', nargs='+', type=int, choices=ALLOWED_STRIDES, default=[8,16])
    parser.add_argument('--limit-pairs', type=int, default=64)
    export(parser.parse_args())
