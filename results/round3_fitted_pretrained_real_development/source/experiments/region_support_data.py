"""Dense visible-region transfer on independently partitioned rendered views.

Labels count actual rendered surface pixels reprojected into target rectangles;
no Gaussian kernel, fitted covariance, or primitive descriptor generates labels.
The target is the distribution *conditional on visible query area*. Completely
invisible rows have label_valid=False and a zero target, not a dustbin label.

API: make_dataset(seed, scenes, condition='oracle', cache=None) -> (data, stats).
Each data tensor starts with [scenes, ...]. There are 48 regions per view. q/k
are 108-dimensional RGB features; mq/mk and cq/ck are world-coordinate moments.
pixq, k_to_q, projected_*cov are in patch units (pixel coordinates / 12).
projected_qcov is the exact covariance of the query rectangle's pixel centers,
known without geometry. The key covariance is fitted independently in its view
and projected into the query camera. A .005^2 world-coordinate diagonal floor
and a 1e-6 patch-coordinate projection floor are explicit numerical conventions.

Both views independently jitter interior boundaries of an 8x6 rectangular grid.
oracle: integer jitter +/-2 px; grid_ood: +/-4 px; wide: oracle grid with the
existing 1.8x camera-baseline intervention. These ranges are fixed in advance.
"""
import numpy as np
import torch

from experiments.rendered_scene import WIDTH, HEIGHT, PATCH, FOCAL, COLS, ROWS, pair


CONDITIONS = ['oracle', 'grid_ood', 'wide']
GEOMETRY_SAMPLES = 5
GEOMETRY_FLOOR = .005 ** 2
PROJECTED_FLOOR = 1e-6
VISIBILITY_DEPTH_TOLERANCE = .06


def grid_rectangles(rng, jitter=2):
    """Integer [x0,y0,x1,y1) pixel rectangles, with independent view streams.

    Pixel coordinates are centers: the continuous boundary of [x0,x1) is
    [x0-.5,x1-.5). Even the OOD partition has width/height >=4 pixels.
    """
    if jitter not in (0, 2, 4):
        raise ValueError('predefined jitter must be 0, 2, or 4')
    xs = np.arange(COLS + 1, dtype=np.int64) * PATCH
    ys = np.arange(ROWS + 1, dtype=np.int64) * PATCH
    xs[1:-1] += rng.integers(-jitter, jitter + 1, COLS - 1)
    ys[1:-1] += rng.integers(-jitter, jitter + 1, ROWS - 1)
    rects = np.asarray([(xs[x], ys[y], xs[x+1], ys[y+1])
                        for y in range(ROWS) for x in range(COLS)])
    assert np.all(rects[:, 2:] - rects[:, :2] >= 4)
    return rects


def rectangle_moments(rectangles):
    """Known pixel-center mean/covariance in patch units, not fitted geometry."""
    sizes = rectangles[:, 2:] - rectangles[:, :2]
    centers = (rectangles[:, :2] + rectangles[:, 2:] - 1) / 2
    covariance = np.zeros((len(rectangles), 2, 2), dtype=np.float64)
    covariance[:, 0, 0] = (sizes[:, 0] ** 2 - 1) / 12 / PATCH ** 2
    covariance[:, 1, 1] = (sizes[:, 1] ** 2 - 1) / 12 / PATCH ** 2
    return centers / PATCH, covariance


def pixel_region_map(rectangles):
    result = np.full((HEIGHT, WIDTH), -1, dtype=np.int64)
    for index, (x0, y0, x1, y1) in enumerate(rectangles):
        if np.any(result[y0:y1, x0:x1] >= 0):
            raise ValueError('rectangles must not overlap')
        result[y0:y1, x0:x1] = index
    if np.any(result < 0):
        raise ValueError('rectangles must cover the image')
    return result


def _bilinear(image, x, y):
    x = np.clip(x, 0, WIDTH-1)
    y = np.clip(y, 0, HEIGHT-1)
    x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    x1, y1 = np.minimum(x0+1, WIDTH-1), np.minimum(y0+1, HEIGHT-1)
    wx, wy = (x-x0)[..., None], (y-y0)[..., None]
    return ((1-wx)*(1-wy)*image[y0, x0] + wx*(1-wy)*image[y0, x1]
            + (1-wx)*wy*image[y1, x0] + wx*wy*image[y1, x1])


def region_features(rgb, rectangles):
    """Resize each actual RGB crop to 12x12 then average 2x2 -> 108 values.

    Fixed bilinear sampling depends only on that crop's image and rectangle.
    It does not consult depth, fitted moments, surface IDs, or transfer labels.
    """
    features = []
    for x0, y0, x1, y1 in rectangles:
        xx = x0 - .5 + (np.arange(12) + .5) * (x1-x0) / 12
        yy = y0 - .5 + (np.arange(12) + .5) * (y1-y0) / 12
        # Crop-border replication prevents sampling another region's RGB.
        xx, yy = np.clip(xx, x0, x1-1), np.clip(yy, y0, y1-1)
        x, y = np.meshgrid(xx, yy)
        patch = _bilinear(rgb, x, y)
        features.append(patch.reshape(6, 2, 6, 2, 3).mean((1, 3)).reshape(-1))
    return np.asarray(features, dtype=np.float32)


def fit_region_geometry(view, rectangles, floor=GEOMETRY_FLOOR):
    """Population moments from an independent 5x5 stratified surface sample.

    The image is already a deterministic nearest-surface raycast. Sampling the
    integer pixel buffer avoids interpolating across occlusion discontinuities.
    This is support covariance, not prediction-error uncertainty.
    """
    means, covariances, normals, purities = [], [], [], []
    for x0, y0, x1, y1 in rectangles:
        xx = np.floor(x0 + (np.arange(GEOMETRY_SAMPLES)+.5)
                      * (x1-x0)/GEOMETRY_SAMPLES).astype(np.int64)
        yy = np.floor(y0 + (np.arange(GEOMETRY_SAMPLES)+.5)
                      * (y1-y0)/GEOMETRY_SAMPLES).astype(np.int64)
        x, y = np.meshgrid(xx, yy)
        points = view['points'][y, x].reshape(-1, 3).astype(np.float64)
        mean = points.mean(0)
        centered = points - mean
        cov = centered.T @ centered / len(points) + np.eye(3) * floor
        _, eigenvectors = np.linalg.eigh(cov)
        means.append(mean)
        covariances.append(cov)
        normals.append(eigenvectors[:, 0])
        _, counts = np.unique(view['surface'][y0:y1, x0:x1], return_counts=True)
        purities.append(counts.max()/counts.sum())
    return {'mean': np.asarray(means), 'cov': np.asarray(covariances),
            'normal': np.asarray(normals), 'purity': np.asarray(purities)}


def project_geometry(mean, covariance, query_view):
    """First-order Gaussian projection into the query camera in patch units."""
    rotation, translation = query_view['rotation'], query_view['translation']
    camera_mean = mean @ rotation.T + translation
    x, y, z = camera_mean.T
    if not np.all(z > .05):
        raise ValueError('mean support must be in front of the query camera')
    pixel_mean = camera_mean[:, :2] / z[:, None] * FOCAL
    pixel_mean += np.asarray([WIDTH/2, HEIGHT/2])
    jac = np.zeros((len(mean), 2, 3), dtype=np.float64)
    jac[:, 0, 0], jac[:, 1, 1] = FOCAL/z, FOCAL/z
    jac[:, 0, 2], jac[:, 1, 2] = -FOCAL*x/z**2, -FOCAL*y/z**2
    jac = jac @ rotation
    projected = jac @ covariance @ jac.transpose(0, 2, 1) / PATCH**2
    projected += np.eye(2) * PROJECTED_FLOOR
    return pixel_mean/PATCH, projected


def visible_region_labels(query_view, key_view, query_rectangles, key_rectangles):
    """Empirical dense pixel-area transfer; Gaussian geometry is not an input.

    Every actual query pixel is reprojected. The nearest target-pixel surface
    and depth validate visibility. Membership uses actual target rectangles.
    No target-region-center surface rule is applied. Visibility is label-only.
    Returns conditional soft targets, validity mask, and raw visible fraction.
    """
    count = len(query_rectangles)
    query_ids = pixel_region_map(query_rectangles).reshape(-1)
    key_ids = pixel_region_map(key_rectangles)
    points = query_view['points'].reshape(-1, 3).astype(np.float64)
    camera_points = points @ key_view['rotation'].T + key_view['translation']
    depth = camera_points[:, 2]
    uv = camera_points[:, :2] / depth[:, None] * FOCAL
    uv += np.asarray([WIDTH/2, HEIGHT/2])
    inside = ((depth > .05) & (uv[:, 0] >= -.5) & (uv[:, 0] < WIDTH-.5)
              & (uv[:, 1] >= -.5) & (uv[:, 1] < HEIGHT-.5))
    # Pixel centers are integers, so cell boundaries occur at half-integers.
    px = np.clip(np.floor(uv[:, 0]+.5).astype(np.int64), 0, WIDTH-1)
    py = np.clip(np.floor(uv[:, 1]+.5).astype(np.int64), 0, HEIGHT-1)
    visible = (inside
               & (np.abs(depth-key_view['depth'][py, px]) < VISIBILITY_DEPTH_TOLERANCE)
               & (query_view['surface'].reshape(-1) == key_view['surface'][py, px]))
    mapped = key_ids[py, px]
    bins = query_ids[visible]*count + mapped[visible]
    counts = np.bincount(bins, minlength=count*count).reshape(count, count)
    visible_counts = counts.sum(-1)
    areas = np.bincount(query_ids, minlength=count)
    valid = visible_counts > 0
    soft = counts / np.maximum(visible_counts[:, None], 1)
    fraction = visible_counts / areas
    return soft.astype(np.float32), valid, fraction.astype(np.float32)


def make_example(scene_seed, condition='oracle', cache=None, geometry_floor=GEOMETRY_FLOOR):
    if condition not in CONDITIONS:
        raise ValueError(condition)
    wide = condition == 'wide'
    cache_key = (scene_seed, wide)
    if cache is not None and cache_key in cache:
        a, b, _ = cache[cache_key]
    else:
        a, b, gt = pair(scene_seed, wide)
        if cache is not None:
            cache[cache_key] = (a, b, gt)
    jitter = 4 if condition == 'grid_ood' else 2
    rq = grid_rectangles(np.random.default_rng(scene_seed+77101), jitter)
    rk = grid_rectangles(np.random.default_rng(scene_seed+77203), jitter)
    qgeometry = fit_region_geometry(a, rq, geometry_floor)
    kgeometry = fit_region_geometry(b, rk, geometry_floor)
    pixq, query_cov = rectangle_moments(rq)
    fitted_query_pixel, fitted_query_cov = project_geometry(qgeometry['mean'], qgeometry['cov'], a)
    key_pixel, key_cov = project_geometry(kgeometry['mean'], kgeometry['cov'], a)
    soft, valid, fraction = visible_region_labels(a, b, rq, rk)
    values = {'q': region_features(a['rgb'], rq), 'k': region_features(b['rgb'], rk),
              'mq': qgeometry['mean'], 'mk': kgeometry['mean'],
              'cq': qgeometry['cov'], 'ck': kgeometry['cov'],
              'nq': qgeometry['normal'], 'nk': kgeometry['normal'],
              'pixq': pixq, 'k_to_q': key_pixel,
              'projected_qcov': query_cov, 'projected_kcov': key_cov,
              'projected_fitted_qcov': fitted_query_cov,
              'fitted_pixq': fitted_query_pixel,
              'rectq': rq, 'rectk': rk,
              'soft_labels': soft, 'label_valid': valid,
              'visible_fraction': fraction}
    data = {key: torch.as_tensor(value, dtype=torch.bool if key == 'label_valid' else torch.float32)
            for key, value in values.items()}
    widths = np.concatenate([rq[:, 2:]-rq[:, :2], rk[:, 2:]-rk[:, :2]])
    stats = {'scene_seed': int(scene_seed), 'condition': condition,
             'region_label': 'dense pixel-area distribution conditional on visibility; no dustbin',
             'geometry_floor_world_variance': float(geometry_floor),
             'projected_floor_patch_variance': PROJECTED_FLOOR,
             'geometry_samples_per_axis': GEOMETRY_SAMPLES,
             'label_query_pixels': WIDTH*HEIGHT,
             'mean_visible_fraction': float(fraction.mean()),
             'invalid_query_fraction': float((~valid).mean()),
             'mean_label_entropy': (float(-(soft*np.log(np.maximum(soft, 1e-30))).sum(-1)[valid].mean())
                                    if valid.any() else None),
             'mean_query_surface_purity': float(qgeometry['purity'].mean()),
             'mean_key_surface_purity': float(kgeometry['purity'].mean()),
             'minimum_region_side_pixels': int(widths.min()),
             'maximum_region_side_pixels': int(widths.max()),
             'mean_projected_key_cov_trace': float(np.trace(key_cov, axis1=-2, axis2=-1).mean()),
             'query_projection_center_error_patch_units': float(np.linalg.norm(fitted_query_pixel-pixq, axis=-1).mean())}
    assert all(torch.isfinite(value).all() for value in data.values())
    return data, stats


def make_dataset(seed, scenes, condition='oracle', cache=None):
    if scenes < 1:
        raise ValueError('scenes must be positive')
    examples = [make_example(seed*1000+index, condition, cache) for index in range(scenes)]
    data = {key: torch.stack([example[0][key] for example in examples]) for key in examples[0][0]}
    return data, [example[1] for example in examples]
