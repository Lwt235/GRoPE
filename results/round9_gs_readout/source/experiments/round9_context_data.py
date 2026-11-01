"""Context-only SIFT initialization for the Round 9 Gaussian color-read experiment.

Only the provided camera metadata is inherited from full-scene SfM. Old point
coordinates, colors, keypoints, tracks and Gaussian checkpoints are never loaded.
Target image bytes are not opened by this exporter.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

DATASETS = {
    name: Path('/home/liwt/miniColmap/data/lerf_ovs') / name / 'dataset_posed_all.json'
    for name in ('teatime', 'waldo_kitchen')
}
WIDTH, HEIGHT = 192, 144
SIFT_EDGE, MAX_FEATURES, MAX_POINTS = 640, 2000, 12000
RATIO, MAX_REPROJECTION, MIN_ANGLE = .75, 2., 1.
VOXEL_WIDTH = 1e-4  # Units of the context-camera extent, after normalization.


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def scaled_intrinsics(K, original_size, new_size):
    sx, sy = np.asarray(new_size) / np.asarray(original_size)
    return np.array([[sx, 0., (sx - 1) / 2], [0., sy, (sy - 1) / 2],
                     [0., 0., 1.]]) @ np.asarray(K)


def select_ids(image_records, posed_ids):
    """Name order, old development first two thirds, 32 uniform selected views."""
    names = {int(row['id']): row['name'] for row in image_records}
    ordered = sorted((int(i) for i in posed_ids), key=lambda i: names[i])
    blocks = np.array_split(ordered, 3)
    development = np.concatenate(blocks[:2]).astype(np.int64)
    if len(development) < 32:
        raise ValueError('Round 9 requires at least 32 old-development cameras')
    selected = development[np.linspace(0, len(development) - 1, 32, dtype=np.int64)]
    target = selected[np.arange(32) % 4 == 3]
    context = selected[np.arange(32) % 4 != 3]
    if len(set(selected)) != 32 or set(context) & set(target):
        raise AssertionError('Invalid split')
    return {'all_name_order': ordered, 'development_ids': development.tolist(),
            'sealed_third_block_ids': [int(i) for i in blocks[2]],
            'selected_ids': selected.tolist(), 'context_ids': context.tolist(),
            'target_ids': target.tolist()}


def normalize_context_cameras(R, t):
    centers = -np.einsum('nji,nj->ni', R, t)
    center = centers.mean(0)
    scale = 1.1 * float(np.linalg.norm(centers - center, axis=1).max())
    if not np.isfinite(scale) or scale <= 1e-8:
        raise ValueError('Degenerate context camera extent')
    normalized_t = (np.einsum('nij,j->ni', R, center) + t) / scale
    return center, scale, normalized_t


def read_context_image(frame, allowed_ids, target_ids):
    """Check membership before opening an image; targets are forbidden here."""
    if frame['id'] not in allowed_ids or frame['id'] in target_ids:
        raise ValueError('Attempted to read a non-context image')
    path = Path(frame['image_path'])
    digest = sha256(path)
    with Image.open(path) as image:
        rgb = np.asarray(image.convert('RGB'))
    if rgb.shape[:2] != (frame['original_height'], frame['original_width']):
        raise ValueError('Image dimensions do not match calibration')
    return rgb, digest


def bilinear_rgb(rgb, pixels):
    h, w = rgb.shape[:2]
    xy = np.clip(pixels, [0., 0.], [w - 1., h - 1.])
    lo = np.floor(xy).astype(np.int64)
    hi = np.minimum(lo + 1, [w - 1, h - 1])
    dx, dy = (xy - lo).T
    return (rgb[lo[:, 1], lo[:, 0]] * ((1 - dx) * (1 - dy))[:, None]
            + rgb[lo[:, 1], hi[:, 0]] * (dx * (1 - dy))[:, None]
            + rgb[hi[:, 1], lo[:, 0]] * ((1 - dx) * dy)[:, None]
            + rgb[hi[:, 1], hi[:, 0]] * (dx * dy)[:, None])


def triangulate_matches(x1, x2, K1, K2, R1, t1, R2, t2):
    """Known-camera DLT with positive-depth, two-view error and angle checks."""
    import cv2
    if len(x1) == 0:
        return np.empty((0, 3)), np.zeros(0, bool), {'ratio_matches': 0, 'accepted': 0}
    P1 = K1 @ np.column_stack((R1, t1))
    P2 = K2 @ np.column_stack((R2, t2))
    homogeneous = cv2.triangulatePoints(P1, P2, x1.T, x2.T).T
    safe = np.abs(homogeneous[:, 3]) > 1e-12
    points = np.full((len(x1), 3), np.nan)
    points[safe] = homogeneous[safe, :3] / homogeneous[safe, 3, None]
    c1, c2 = points @ R1.T + t1, points @ R2.T + t2
    positive = safe & np.isfinite(points).all(1) & (c1[:, 2] > 1e-8) & (c2[:, 2] > 1e-8)
    with np.errstate(invalid='ignore', divide='ignore'):
        h1, h2 = c1 @ K1.T, c2 @ K2.T
        e1 = np.linalg.norm(h1[:, :2] / h1[:, 2, None] - x1, axis=1)
        e2 = np.linalg.norm(h2[:, :2] / h2[:, 2, None] - x2, axis=1)
        rays1, rays2 = points + R1.T @ t1, points + R2.T @ t2
        cosine = np.einsum('ni,ni->n', rays1, rays2)
        cosine /= np.linalg.norm(rays1, axis=1) * np.linalg.norm(rays2, axis=1)
        angle = np.rad2deg(np.arccos(np.clip(cosine, -1, 1)))
    reprojection = positive & (e1 <= MAX_REPROJECTION) & (e2 <= MAX_REPROJECTION)
    accepted = reprojection & np.isfinite(angle) & (angle >= MIN_ANGLE)
    return points, accepted, {'ratio_matches': len(x1), 'positive_depth': int(positive.sum()),
                             'reprojection_pass': int(reprojection.sum()),
                             'accepted': int(accepted.sum())}


def deduplicate_and_limit(points, colors, maximum=MAX_POINTS, voxel_width=VOXEL_WIDTH):
    """First representative per tiny voxel, then deterministic uniform subsampling."""
    if not len(points) or not np.isfinite(points).all() or not np.isfinite(colors).all():
        raise ValueError('Empty or nonfinite initialization')
    voxels = np.floor(points / voxel_width).astype(np.int64)
    _, first = np.unique(voxels, axis=0, return_index=True)
    first.sort()
    unique_count = len(first)
    if unique_count > maximum:
        first = first[np.linspace(0, unique_count - 1, maximum, dtype=np.int64)]
    return points[first], colors[first], unique_count


def export_scene(name, output_root=Path('results/round9_context_data')):
    import cv2
    start = time.monotonic()
    cv2.setNumThreads(1)
    cv2.setRNGSeed(20261030)
    dataset_path = DATASETS[name]
    document = json.loads(dataset_path.read_text())
    if set(document) != {'cameras', 'images', 'pose_convention', 'poses'}:
        raise ValueError('Expected camera-only posed dataset, without inherited geometry')
    if document['pose_convention'] != 'world_to_camera':
        raise ValueError('Unsupported pose convention')
    cameras = {int(row['id']): row for row in document['cameras']}
    images = {int(row['id']): row for row in document['images']}
    poses = {int(row['image_id']): row for row in document['poses']}
    split = select_ids(document['images'], poses)
    context_ids, target_ids = split['context_ids'], split['target_ids']
    allowed, forbidden = set(context_ids), set(target_ids)
    R = np.asarray([poses[i]['R'] for i in context_ids], np.float64)
    world_t = np.asarray([poses[i]['t'] for i in context_ids], np.float64)
    center, scale, t = normalize_context_cameras(R, world_t)
    normalization = {'center': center.tolist(), 'scale': scale,
                     'source': 'selected_context_camera_extent',
                     'formula': 'x_normalized=(x_world-center)/scale'}

    def camera_record(i):
        image, pose = images[i], poses[i]
        camera = cameras[image['camera_id']]
        K = scaled_intrinsics(camera['K'], (camera['width'], camera['height']), (WIDTH, HEIGHT))
        rotation = np.asarray(pose['R'], np.float64)
        translation = (rotation @ center + np.asarray(pose['t'], np.float64)) / scale
        path = (dataset_path.parent / image['path']).resolve()
        if not path.is_relative_to(dataset_path.parent.resolve()):
            raise ValueError('Image path escapes original dataset directory')
        return {'id': i, 'name': image['name'], 'K': K.tolist(), 'R': rotation.tolist(),
                't': translation.tolist(), 'image_path': str(path), 'width': WIDTH,
                'height': HEIGHT, 'original_width': camera['width'],
                'original_height': camera['height']}

    context_records = [camera_record(i) for i in context_ids]
    target_records = [{**camera_record(i), 'split': 'calibration' if j % 2 == 0 else 'gate'}
                      for j, i in enumerate(target_ids)]
    output = Path(output_root) / name
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Refusing to overwrite {output}')
    output.mkdir(parents=True, exist_ok=True)
    sift = cv2.SIFT_create(nfeatures=MAX_FEATURES)
    data, original_hashes = [], {}
    for frame in context_records:
        raw, digest = read_context_image(frame, allowed, forbidden)
        original_hashes[str(frame['image_path'])] = digest
        ratio = min(1., SIFT_EDGE / max(raw.shape[:2]))
        sift_size = (round(raw.shape[1] * ratio), round(raw.shape[0] * ratio))
        small = cv2.resize(raw, sift_size, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
        keys, descriptors = sift.detectAndCompute(gray, None)
        xy = np.asarray([k.pt for k in keys], np.float64).reshape(-1, 2)
        if len(keys) > MAX_FEATURES:
            keep = np.argsort([-k.response for k in keys], kind='stable')[:MAX_FEATURES]
            xy, descriptors = xy[keep], descriptors[keep]
        K_sift = scaled_intrinsics(frame['K'], (WIDTH, HEIGHT), sift_size)
        rgb = cv2.resize(raw, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        data.append({'xy': xy, 'descriptors': descriptors, 'K_sift': K_sift,
                     'rgb': rgb, 'sift_size': sift_size})
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    all_points, all_colors, pair_records = [], [], []
    for i in range(len(context_ids)):
        for offset in (1, 2, 3):
            j = i + offset
            if j >= len(context_ids):
                continue
            a, b = data[i], data[j]
            matches = ([] if a['descriptors'] is None or b['descriptors'] is None
                       else matcher.knnMatch(a['descriptors'], b['descriptors'], k=2))
            chosen = [m for pair in matches if len(pair) == 2
                      for m, n in [pair] if m.distance < RATIO * n.distance]
            x1 = np.asarray([a['xy'][m.queryIdx] for m in chosen], np.float64).reshape(-1, 2)
            x2 = np.asarray([b['xy'][m.trainIdx] for m in chosen], np.float64).reshape(-1, 2)
            points, valid, stats = triangulate_matches(x1, x2, a['K_sift'], b['K_sift'],
                                                       R[i], t[i], R[j], t[j])
            if valid.any():
                p1 = (x1[valid] + .5) * (np.array([WIDTH, HEIGHT]) / a['sift_size']) - .5
                p2 = (x2[valid] + .5) * (np.array([WIDTH, HEIGHT]) / b['sift_size']) - .5
                all_points.append(points[valid])
                all_colors.append((bilinear_rgb(a['rgb'], p1) + bilinear_rgb(b['rgb'], p2)) * .5)
            pair_records.append({'first_id': context_ids[i], 'second_id': context_ids[j], **stats})
    if not all_points:
        raise ValueError('No valid context-only triangulations')
    raw_points, raw_colors = np.concatenate(all_points), np.concatenate(all_colors)
    points, colors, unique_count = deduplicate_and_limit(raw_points, raw_colors)
    if len(points) < 100:
        raise ValueError(f'Insufficient context initialization: {len(points)} points')
    print(f'{name}: accepted={len(raw_points)}, unique={unique_count}, initialization={len(points)}', flush=True)
    np.savez_compressed(output / 'context.npz', points=points.astype(np.float32),
                        colors=colors.astype(np.float32), rgb=np.stack([d['rgb'] for d in data]),
                        K=np.asarray([f['K'] for f in context_records], np.float64), R=R, t=t,
                        ids=np.asarray(context_ids, np.int64))
    target_document = {'frames': target_records, 'normalization': normalization,
                       'target_rgb_read': False, 'camera_calibration_provenance': 'provided full-image SfM'}
    (output / 'target_cameras.json').write_text(json.dumps(target_document, indent=2) + '\n')
    manifest = {
        'scene': name, 'scope': 'old-development images; context-only pixels and triangulation',
        'normalization': normalization, 'split': split, 'context_frames': context_records,
        'source_dataset': str(dataset_path), 'source_dataset_sha256': sha256(dataset_path),
        'source_image_sha256': original_hashes, 'exporter_sha256': sha256(__file__),
        'context_archive_sha256': sha256(output / 'context.npz'),
        'target_cameras_sha256': sha256(output / 'target_cameras.json'),
        'context_rgb_reads': len(original_hashes), 'target_rgb_reads': 0,
        'old_geometry_keypoints_tracks_read': False,
        'camera_calibration_provenance': 'provided full-image SfM; not context-only camera estimation',
        'settings': {'width': WIDTH, 'height': HEIGHT, 'sift_max_edge': SIFT_EDGE,
                     'max_features': MAX_FEATURES, 'neighbor_offsets': [1, 2, 3], 'ratio': RATIO,
                     'reprojection_limit_sift_pixels': MAX_REPROJECTION,
                     'minimum_ray_angle_degrees': MIN_ANGLE, 'voxel_width_normalized': VOXEL_WIDTH,
                     'maximum_initialization_points': MAX_POINTS, 'sift_opencv_version': cv2.__version__,
                     'color': 'mean bilinear context RGB from both observations, sRGB'},
        'statistics': {'features_by_frame': [len(d['xy']) for d in data],
                       'candidate_pairs': len(pair_records), 'valid_triangulations': len(raw_points),
                       'unique_voxels': unique_count, 'initialization_points': len(points),
                       'point_quantiles_normalized': np.quantile(points, [0, .01, .5, .99, 1], axis=0).tolist()},
        'pairs': pair_records, 'seconds': time.monotonic() - start,
    }
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenes', nargs='+', choices=tuple(DATASETS), default=list(DATASETS))
    parser.add_argument('--out', type=Path, default=Path('results/round9_context_data'))
    args = parser.parse_args()
    for name in args.scenes:
        result = export_scene(name, args.out)
        print(json.dumps({'scene': name, 'points': result['statistics']['initialization_points'],
                          'seconds': result['seconds']}), flush=True)


if __name__ == '__main__':
    main()
