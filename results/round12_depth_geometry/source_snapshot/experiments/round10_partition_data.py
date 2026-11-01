"""Old-development observations and independent labels for token partition probes.

The forward archive is written before any sensor image is opened. Every source
image pixel has unit image-area mass; no sensor visibility enters its support.
Rectangular partitions change which actual pixels belong to a token. Gaussian
moments after projection and projection of Gaussian 3D moments are separate
functions because perspective projection does not commute with moment fitting.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from experiments.round5_rgbd_data import (
    DEFAULT_IMAGE_HW, NEAREST_TIE_EPSILON, STATUS_NAMES, _K, _pose,
    load_sensor_depth, project_points_and_jacobian,
)
from experiments.round6_region_data import (
    _load_cache, _project_gt_pixels, read_scene_selection,
)

ROOT = Path(__file__).resolve().parents[1]
PARTITIONS = ("g8", "g16", "mixed")
OLD_SCENES = {
    "harvard_c3/hv_c3_1": "train", "harvard_c5/hv_c5_1": "train",
    "harvard_c6/hv_c6_1": "train", "harvard_c8/hv_c8_3": "val",
    "brown_bm_4/brown_bm_4": "val",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pixel_centers(image_hw=DEFAULT_IMAGE_HW):
    h, w = image_hw
    yy, xx = np.mgrid[:h, :w]
    return np.stack([xx, yy], -1).reshape(-1, 2).astype(np.float64)


def partition_bounds(name, image_hw=DEFAULT_IMAGE_HW):
    """Integer-index half-open bounds; corresponding physical edges are -.5.

    Mixed partitions cycle through four deterministic 16x16 macroblock types:
    intact 16x16, four 8x8, two 16x8, two 8x16. They use no image information.
    """
    h, w = image_hw
    if min(h, w) <= 0 or h % 16 or w % 16:
        raise ValueError("image dimensions must be positive multiples of 16")
    if name in ("g8", "g16"):
        s = int(name[1:])
        bounds = [[x, y, x+s, y+s] for y in range(0, h, s) for x in range(0, w, s)]
    elif name == "mixed":
        bounds = []
        for y in range(0, h, 16):
            for x in range(0, w, 16):
                kind = (x//16 + y//16) % 4
                wh = ((16, 16), (8, 8), (16, 8), (8, 16))[kind]
                for dy in range(0, 16, wh[1]):
                    for dx in range(0, 16, wh[0]):
                        bounds.append([x+dx, y+dy, x+dx+wh[0], y+dy+wh[1]])
    else:
        raise ValueError("unsupported partition: " + str(name))
    bounds = np.asarray(bounds, np.int32)
    partition_index(bounds, image_hw)
    return bounds


def partition_index(bounds, image_hw=DEFAULT_IMAGE_HW):
    """Validate exact integer-pixel coverage and return a dense token lookup."""
    raw = np.asarray(bounds)
    if raw.ndim != 2 or raw.shape[1] != 4 or not np.isfinite(raw).all():
        raise ValueError("bounds must be finite [tokens,4]")
    if not np.array_equal(raw, raw.astype(np.int64)):
        raise ValueError("partition edges must have integer pixel indices")
    h, w = image_hw
    lookup = np.full((h, w), -1, np.int32)
    for i, (x0, y0, x1, y1) in enumerate(raw.astype(np.int64)):
        if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
            raise ValueError("invalid rectangle")
        if (lookup[y0:y1, x0:x1] >= 0).any():
            raise ValueError("overlapping partition rectangles")
        lookup[y0:y1, x0:x1] = i
    if (lookup < 0).any():
        raise ValueError("partition does not cover every source pixel")
    return lookup


def native_support(bounds):
    bounds = np.asarray(bounds, np.float64)
    wh = bounds[:, 2:]-bounds[:, :2]
    centers = (bounds[:, :2]+bounds[:, 2:]-1)/2
    covariance = np.zeros((len(bounds), 2, 2), np.float64)
    covariance[:, 0, 0] = wh[:, 0]**2/12
    covariance[:, 1, 1] = wh[:, 1]**2/12
    return centers, covariance, wh.prod(-1)


def containing_partition(points, bounds, image_hw=DEFAULT_IMAGE_HW):
    lookup = partition_index(bounds, image_hw)
    points = np.asarray(points, np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape [N,2]")
    h, w = image_hw
    finite = np.isfinite(points).all(-1)
    safe = np.where(np.isfinite(points), points, 0)
    inside = finite & (safe[:, 0] >= -.5) & (safe[:, 0] < w-.5)
    inside &= (safe[:, 1] >= -.5) & (safe[:, 1] < h-.5)
    ij = np.floor(safe+.5+NEAREST_TIE_EPSILON)
    ij = np.clip(ij, [0, 0], [w-1, h-1]).astype(np.int64)
    return np.where(inside, lookup[ij[:, 1], ij[:, 0]], -1), inside


def partition_moments(centers, within_cov, bounds, image_hw=DEFAULT_IMAGE_HW):
    """Exact first two mixture moments of every actual pixel, with area mass 1."""
    lookup = partition_index(bounds, image_hw).ravel()
    centers = np.asarray(centers, np.float64)
    within_cov = np.asarray(within_cov, np.float64)
    n, d = centers.shape
    if n != len(lookup) or within_cov.shape != (n, d, d):
        raise ValueError("one mean and covariance per source image pixel required")
    if not np.isfinite(centers).all() or not np.isfinite(within_cov).all():
        raise ValueError("forward means and covariances must be finite")
    means, covariance = [], []
    for i in range(len(bounds)):
        selected = lookup == i
        m = centers[selected].mean(0)
        delta = centers[selected]-m
        means.append(m)
        covariance.append(within_cov[selected].mean(0)+delta.T@delta/selected.sum())
    return np.asarray(means), np.asarray(covariance)


def pool_base_features(features, bounds, image_hw=DEFAULT_IMAGE_HW, normalize=True):
    """Area-pool the same frozen 8px feature field for every true partition."""
    h, w = image_hw
    features = np.asarray(features, np.float64)
    if features.ndim != 2 or len(features) != (h//8)*(w//8):
        raise ValueError("expected one frozen feature per base8 cell")
    if not np.isfinite(features).all():
        raise ValueError("nonfinite base feature")
    dense = features.reshape(h//8, w//8, -1).repeat(8, 0).repeat(8, 1)
    partition_index(bounds, image_hw)
    pooled = np.asarray([dense[y0:y1, x0:x1].mean((0, 1)) for x0,y0,x1,y1 in bounds])
    if normalize:
        pooled /= np.maximum(np.linalg.norm(pooled, axis=-1, keepdims=True), 1e-12)
    return pooled.astype(np.float32)


def project_finite_support(points3, covariance3, K):
    """Fixed Round6-style finite fallback, expressed in physical pixel units."""
    points3 = np.asarray(points3, np.float64)
    covariance3 = np.asarray(covariance3, np.float64)
    safe = points3.copy()
    safe[:, 2] = np.maximum(safe[:, 2], .1)
    xy, J, _ = project_points_and_jacobian(safe, K)
    cov = J@covariance3@J.transpose(0, 2, 1)
    values, vectors = np.linalg.eigh((cov+cov.transpose(0, 2, 1))/2)
    cov = (vectors*values.clip(0, 256*64)[:, None, :])@vectors.transpose(0, 2, 1)
    return xy.clip(-128*8, 128*8), cov, points3[:, 2] >= .1


def dense_observations(query_cache, key_cache):
    """Only RGB-predicted depth/features and supplied cameras are accepted."""
    depth = np.asarray(query_cache['predicted_depth'], np.float64)
    if depth.shape != DEFAULT_IMAGE_HW:
        raise ValueError("full 192x144 RGB-predicted depth required")
    z = np.nan_to_num(depth, nan=10., posinf=20., neginf=.1).clip(.1, 20.).ravel()
    Kq, Kk = _K(query_cache['K']), _K(key_cache['K'])
    transform = np.linalg.inv(_pose(key_cache['c2w']))@_pose(query_cache['c2w'])
    pixels = pixel_centers(depth.shape)
    rays = np.column_stack([pixels, np.ones(len(pixels))])@np.linalg.inv(Kq).T
    points = rays*z[:, None]
    A = np.linalg.inv(Kq)[:, :2]
    within = z[:, None, None]**2*(A@A.T)[None]/12
    key_points = points@transform[:3, :3].T+transform[:3, 3]
    key_cov = transform[:3, :3]@within@transform[:3, :3].T
    xy, cov, in_front = project_finite_support(key_points, key_cov, Kk)
    return dict(q_to_k_xy=xy, q_to_k_pixel_cov=cov, q_to_k_in_front=in_front,
        q_points3=points, q_pixel_cov3=within, q_to_k_transform=transform,
        q_K=Kq.copy(), k_K=Kk.copy(), q_c2w=_pose(query_cache['c2w']).copy(),
        k_c2w=_pose(key_cache['c2w']).copy(),
        q_features=np.asarray(query_cache['features']).copy(),
        k_features=np.asarray(key_cache['features']).copy(),
        q_predicted_depth=np.asarray(query_cache['predicted_depth']).copy(),
        k_predicted_depth=np.asarray(key_cache['predicted_depth']).copy())


def projected_3d_moments(inputs, bounds, image_hw=DEFAULT_IMAGE_HW):
    means, cov = partition_moments(inputs['q_points3'], inputs['q_pixel_cov3'], bounds, image_hw)
    transform = inputs['q_to_k_transform']
    means = means@transform[:3, :3].T+transform[:3, 3]
    cov = transform[:3, :3]@cov@transform[:3, :3].T
    return project_finite_support(means, cov, inputs['k_K'])


def choose_pairs(row, query_count=8):
    """Metadata-only evenly spaced query ordinals, retaining both old offsets."""
    if row['scene'] not in OLD_SCENES or row['split'] != OLD_SCENES[row['scene']]:
        raise ValueError("only the five allowlisted old development scans are permitted")
    if query_count < 1:
        raise ValueError("query_count must be positive")
    queries = list(dict.fromkeys(q for q, _ in row['pairs']))
    if query_count > len(queries):
        raise ValueError("query_count exceeds available old queries")
    indices = np.linspace(0, len(queries)-1, query_count).astype(np.int64)
    selected = {queries[i] for i in indices}
    pairs = [[int(q), int(k)] for q, k in row['pairs'] if q in selected]
    if len(pairs) != query_count*2 or any(k-q not in (15,60) for q,k in pairs):
        raise ValueError("expected exactly both offset15/60 pairs per chosen query")
    return pairs


def source_hashes():
    paths = [Path(__file__), ROOT/'experiments/round6_region_data.py',
             ROOT/'experiments/round5_rgbd_data.py']
    return {str(path.resolve()): sha(path) for path in paths}


def export(out, query_count=8):
    out = Path(out)
    if out.exists():
        raise ValueError("refuse overwrite: " + str(out))
    sources = source_hashes()
    groups = [dict(raw_root=ROOT/'data/round5_sun3d',
                   selection=ROOT/'data/round5_sun3d/selection.json',
                   cache_root=ROOT/'results/round5_inputs/cache'),
              dict(raw_root=ROOT/'data/round6_sun3d',
                   selection=ROOT/'data/round6_sun3d/selection_v2.json',
                   cache_root=ROOT/'results/round6_inputs/cache')]
    jobs = []
    for group in groups:
        selection = json.loads(group['selection'].read_text())
        for row in selection['scenes']:
            if row['scene'] not in OLD_SCENES:
                continue
            pairs = choose_pairs(row, query_count)
            jobs.append((group, row, pairs))
    if {row['scene'] for _,row,_ in jobs} != set(OLD_SCENES) or len(jobs) != 5:
        raise ValueError("old development cohort is incomplete or duplicated")
    out.mkdir(parents=True)
    started = time.time()
    partitions = {}
    for name in PARTITIONS:
        bounds = partition_bounds(name)
        centers, covariance, area = native_support(bounds)
        partitions.update({name+'_bounds':bounds, name+'_centers':centers,
            name+'_covariance':covariance, name+'_area':area,
            name+'_pixel_index':partition_index(bounds),
            name+'_parent_g16':(centers[:, 1].astype(np.int64)//16)*(DEFAULT_IMAGE_HW[1]//16)
                               +centers[:, 0].astype(np.int64)//16})
    np.savez_compressed(out/'partitions.npz', **partitions)
    contract = dict(source_hashes=sources, scope='old-development only; no fresh confirmation',
        old_scene_splits=OLD_SCENES, query_count=query_count,
        pair_rule='linspace old query ordinal endpoints, integer floor; preserve both offset15/60',
        image_hw=list(DEFAULT_IMAGE_HW), coordinate_units='pixels, zero-based centers',
        area_measure='uniform source image area; every actual pixel has mass 1',
        partitions_sha256=sha(out/'partitions.npz'), sensor_used_for_forward=False,
        finite_fallback='depth nan10,+inf20,-inf.1; clip[.1,20]; transformed z floor.1; xy clip +/-1024px; covariance eigenvalues[0,16384]px^2',
        in_front='transformed predicted pixel z>=.1m before finite fallback; forward diagnostic only',
        rgb='same PIL BOX 192x144 sRGB/255 as frozen feature producer',
        moments='dense projection mixture and 3D mixture followed by projection are separate',
        native_covariance='continuous rectangle diag(width^2,height^2)/12; physical edges index-.5',
        input_phase_complete_before_sensor_reads=True,
        source_selections={str(g['selection']):sha(g['selection']) for g in groups})
    (out/'contract.json').write_text(json.dumps(contract, indent=2))
    pending, input_files = [], []
    # All scenes' forward archives are complete before the first sensor open.
    for group, row, pairs in jobs:
        frames = read_scene_selection(group['raw_root'], row)
        chosen = sorted({i for pair in pairs for i in pair})
        caches, rgb, provenance = {}, {}, {}
        for index in chosen:
            cache_path = group['cache_root']/row['scene']/f'{index:07d}.npz'
            caches[index], sidecar = _load_cache(cache_path, frames[index], row)
            with Image.open(frames[index].rgb_path) as image:
                rgb[index] = np.asarray(image.convert('RGB').resize((192,144), Image.Resampling.BOX), np.float32)/255
            provenance[str(index)] = dict(cache_path=str(cache_path), cache_sha256=sha(cache_path),
                sidecar_path=str(cache_path.with_suffix('.json')), sidecar_sha256=sha(cache_path.with_suffix('.json')),
                rgb_path=str(frames[index].rgb_path), rgb_sha256=sidecar['producer']['rgb_sha256'])
        for pair_index, (qi,ki) in enumerate(pairs):
            dest = out/row['split']/row['scene'].replace('/', '__')/f'{pair_index:03d}_{qi:07d}_{ki:07d}'
            dest.mkdir(parents=True)
            inputs = dense_observations(caches[qi], caches[ki])
            inputs.update(q_rgb=rgb[qi], k_rgb=rgb[ki])
            np.savez_compressed(dest/'inputs.npz', **inputs)
            input_files.append(dict(path=str(dest/'inputs.npz'), sha256=sha(dest/'inputs.npz')))
            metadata = dict(scene=row['scene'], split=row['split'], pair=[qi,ki], pair_ordinal=pair_index,
                source_selection=str(group['selection']), source_selection_sha256=sha(group['selection']),
                camera_files={str(group['raw_root']/row['scene']/'intrinsics.txt'):row['K_sha256'],
                              str(group['raw_root']/row['pose_file']):row['pose_sha256']},
                source_index_files={str(group['raw_root']/row['scene']/name):sha(group['raw_root']/row['scene']/name)
                                    for name in ('image_index.html','depth_index.html')},
                input_sha256=sha(dest/'inputs.npz'), rgb_cache_sources={str(i):provenance[str(i)] for i in (qi,ki)},
                forward_uses_sensor=False, predicted_in_front_count=int(inputs['q_to_k_in_front'].sum()))
            pending.append((dest, metadata, frames[qi], frames[ki]))
        print('FORWARD', row['scene'], len(pairs), flush=True)
    (out/'forward_complete.json').write_text(json.dumps(dict(inputs=input_files,
        sensor_depth_opened=False, source_hashes=sources, seconds=time.time()-started), indent=2))
    manifests = []
    pixels = pixel_centers()
    for dest, metadata, qframe, kframe in pending:
        qdepth, kdepth = load_sensor_depth(qframe.depth_path), load_sensor_depth(kframe.depth_path)
        args = (qdepth,kdepth,qframe.K_native,kframe.K_native,qframe.c2w,kframe.c2w,DEFAULT_IMAGE_HW,.05,.05)
        xy, visible, status = _project_gt_pixels(pixels, *args)
        labels = dict(gt_xy=xy, gt_visible=visible, gt_status=status)
        for name in PARTITIONS:
            centers, _, _ = native_support(partitions[name+'_bounds'])
            pxy, pvalid, pstatus = _project_gt_pixels(centers, *args)
            labels.update({name+'_point_xy':pxy, name+'_point_visible':pvalid, name+'_point_status':pstatus})
        np.savez_compressed(dest/'labels.npz', **labels)
        metadata.update(label_sha256=sha(dest/'labels.npz'),
            sensor_sources={str(qframe.depth_path):sha(qframe.depth_path), str(kframe.depth_path):sha(kframe.depth_path)},
            gt_visible_count=int(visible.sum()), gt_status_counts=np.bincount(status,minlength=len(STATUS_NAMES)).tolist(),
            sensor_visibility='Round6 native nearest sensor; abs(z-sensor)<=.05+.05*sensor, positive depth/inframe',
            contract_sha256=sha(out/'contract.json'))
        (dest/'manifest.json').write_text(json.dumps(metadata, indent=2))
        manifests.append(dict(path=str(dest/'manifest.json'), sha256=sha(dest/'manifest.json'),
                              scene=metadata['scene'], split=metadata['split'], pair=metadata['pair']))
    if source_hashes() != sources or any(sha(p) != d for p,d in contract['source_selections'].items()):
        raise ValueError('data producer or source selection changed during export')
    completion = dict(manifests=manifests, contract_sha256=sha(out/'contract.json'),
        forward_complete_sha256=sha(out/'forward_complete.json'), source_hashes=sources,
        pair_count=len(manifests), seconds=time.time()-started, fresh_test_read=False)
    (out/'completion.json').write_text(json.dumps(completion, indent=2))
    print('DONE', len(manifests), round(completion['seconds'],2), flush=True)
    return completion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--query-count', type=int, default=8)
    args = parser.parse_args()
    export(args.out, args.query_count)


if __name__ == '__main__':
    main()
