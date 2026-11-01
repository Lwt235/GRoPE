"""Pure R38 input construction: no archive access, model, or task labels.

The factor follows R23's explicit scatter/within-pixel QR. The independent
center follows R13 up to target-camera 3D, before either target-z or screen
clipping. The old two-dimensional covariance controls only the inherited
content temperature, not the new three-dimensional Gaussian.
"""
from __future__ import annotations

import math

import numpy as np  # Keep the accepted NumPy-before-Torch import order.
import torch

IMAGE_HW = (144, 192)
N_QUERY, N_KEY, CHANNELS = 48, 432, 128
INPUT_FIELDS = ('q_points3', 'q_pixel_cov3', 'q_K', 'k_K',
                'q_to_k_transform', 'q_predicted_depth')
CACHE_FIELDS = ('g24__a', 'g24__anchor_cov', 'g24__tau32',
                'g24__q_features', 'g8__k_features', 'g8__bounds_px')
# These are provenance references, deliberately not executable imports of the
# old pipelines. The joint contract must bind their current accepted bytes.
PROVENANCE_SOURCES = (
    'experiments/round23_ray_applicability.py',
    'experiments/round13_mixture_mechanism.py',
    'experiments/round25_student_screen.py',
    'experiments/round28_rms_applicability.py',
)
GEOMETRY_FIELDS = (
    'query_bounds_px', 'source_pixel_count', 'source_within_basis',
    'source_mean', 'source_factor', 'source_covariance',
    'source_covariance_direct', 'target_mean', 'target_factor',
    'target_covariance', 'homogeneous_transform', 'mean_homogeneous',
    'factor_homogeneous', 'covariance_homogeneous', 'anchor_query_xy',
    'anchor_depth', 'anchor_source', 'anchor_target', 'anchor_homogeneous',
    'anchor_strict_front', 'anchor_delta', 'anchor_covariance',
    'anchored_factor', 'camera_rotation_diagnostics',
)
CONTENT_FIELDS = (
    'old_a2', 'old_anchor_covariance2', 'old_covariance_asymmetry_normalized',
    'tau32', 'q_features32', 'k_features32', 'key_bounds_px', 'key_bounds',
    'content32', 'bias32', 'content_bias64',
)
BANK_FIELDS = GEOMETRY_FIELDS + CONTENT_FIELDS


def native_bounds(size):
    """Return the fixed raster order; only the declared q24/k8 are allowed."""
    if type(size) is not int or size not in (8, 24):
        raise ValueError('only native q24 and k8 partitions are declared')
    height, width = IMAGE_HW
    return np.asarray([[x, y, x+size, y+size]
                       for y in range(0, height, size)
                       for x in range(0, width, size)], np.int64)


def _array(value, dtype, shape, name, *, finite=True):
    if (not isinstance(value, np.ndarray) or value.dtype != np.dtype(dtype)
            or value.shape != shape or (finite and not np.isfinite(value).all())):
        raise ValueError('dtype/shape/finite mismatch: '+name)
    return value


def validate_observations(observations):
    """Validate the six-field boundary without changing supplied cameras."""
    if set(observations) != set(INPUT_FIELDS):
        raise ValueError('exact six-field predicted-input allowlist required')
    height, width = IMAGE_HW
    expected = dict(q_points3=(height*width, 3),
                    q_pixel_cov3=(height*width, 3, 3), q_K=(3, 3),
                    k_K=(3, 3), q_to_k_transform=(4, 4))
    for name, shape in expected.items():
        _array(observations[name], np.float64, shape, name)
    depth = observations['q_predicted_depth']
    if (not isinstance(depth, np.ndarray) or depth.shape != IMAGE_HW
            or not np.issubdtype(depth.dtype, np.floating)):
        raise ValueError('floating predicted depth raster required')
    # NaN/Inf in predicted depth are legal under the inherited R13 cleanup.
    for name in ('q_K', 'k_K'):
        matrix = observations[name]
        if not np.array_equal(matrix[2], [0., 0., 1.]) or np.linalg.det(matrix) == 0:
            raise ValueError('standard invertible pinhole intrinsics required')
    transform = observations['q_to_k_transform']
    if not np.array_equal(transform[3], [0., 0., 0., 1.]):
        raise ValueError('standard homogeneous camera transform required')
    rotation = transform[:3, :3]
    return np.asarray([np.max(abs(rotation @ rotation.T-np.eye(3))),
                       np.linalg.det(rotation)], np.float64)


def center_anchor(observations, bounds):
    """R13 center-ray arithmetic, ending at target3D before both clamps."""
    depth = np.nan_to_num(np.asarray(observations['q_predicted_depth'], np.float64),
                         nan=10., posinf=20., neginf=.1).clip(.1, 20.)
    points = (bounds[:, :2]+bounds[:, 2:]-1)/2
    ij = np.floor(points).astype(np.int64)
    frac = points-ij
    z = np.zeros(len(points), np.float64)
    for dy in (0, 1):
        for dx in (0, 1):
            weight = (frac[:, 0] if dx else 1-frac[:, 0])*(frac[:, 1] if dy else 1-frac[:, 1])
            z += depth[ij[:, 1]+dy, ij[:, 0]+dx]*weight
    rays = np.column_stack([points, np.ones(len(points))]) @ np.linalg.inv(observations['q_K']).T
    source = rays*z[:, None]
    transform = observations['q_to_k_transform']
    target = source @ transform[:3, :3].T+transform[:3, 3]
    return dict(anchor_query_xy=points, anchor_depth=z, anchor_source=source,
                anchor_target=target, anchor_strict_front=target[:, 2] > 0.)


def augment_factor(mean, factor, anchor):
    """Compress [F, mu-a] with QR, preserving deficient rank without jitter."""
    _array(mean, np.float64, (3,), 'mean')
    _array(factor, np.float64, (3, 3), 'factor')
    _array(anchor, np.float64, (3,), 'anchor')
    delta = mean-anchor
    columns = np.concatenate((factor, delta[:, None]), axis=1)
    _, upper = np.linalg.qr(columns.T, mode='reduced')
    result = upper.T
    if not np.isfinite(result).all():
        raise ValueError('nonfinite augmented QR factor; no repair')
    return result


def prepare_geometry(observations):
    """Construct all 48 supports; no front/FOV/visibility/rank filtering."""
    diagnostics = validate_observations(observations)
    height, width = IMAGE_HW
    bounds = native_bounds(24)
    point_grid = observations['q_points3'].reshape(height, width, 3)
    covariance_grid = observations['q_pixel_cov3'].reshape(height, width, 3, 3)
    basis = np.linalg.solve(observations['q_K'], np.eye(3))[:, :2]/math.sqrt(12.)
    rotation = observations['q_to_k_transform'][:3, :3]
    translation = observations['q_to_k_transform'][:3, 3]
    homogeneous = np.eye(3, dtype=np.float64)
    homogeneous[:2, :2] = observations['k_K'][:2, :2]/8.
    homogeneous[:2, 2] = observations['k_K'][:2, 2]/8.
    collected = {name: [] for name in ('source_mean', 'source_factor',
        'source_covariance', 'source_covariance_direct', 'target_mean',
        'target_factor', 'target_covariance', 'mean_homogeneous',
        'factor_homogeneous', 'covariance_homogeneous')}
    for x0, y0, x1, y1 in bounds:
        points = point_grid[y0:y1, x0:x1].reshape(-1, 3)
        count = len(points)
        mean = points.mean(axis=0)
        within = points[:, 2, None, None]*basis[None]
        columns = np.concatenate(((points-mean).T,
            within.transpose(1, 0, 2).reshape(3, -1)), axis=1)/math.sqrt(count)
        _, upper = np.linalg.qr(columns.T, mode='reduced')
        source_factor = upper.T
        source_covariance = columns @ columns.T
        scatter = points-mean
        # Independent form retained as evidence only. It never replaces the
        # R23 factor; the input auditor checks the cached within-pixel moments.
        direct = (scatter.T @ scatter/count
                  + covariance_grid[y0:y1, x0:x1].reshape(-1, 3, 3).mean(axis=0))
        target_mean = rotation @ mean+translation
        target_factor = rotation @ source_factor
        full_factor = homogeneous @ target_factor
        values = dict(source_mean=mean, source_factor=source_factor,
            source_covariance=source_covariance, source_covariance_direct=direct,
            target_mean=target_mean, target_factor=target_factor,
            target_covariance=rotation @ source_covariance @ rotation.T,
            mean_homogeneous=homogeneous @ target_mean,
            factor_homogeneous=full_factor,
            covariance_homogeneous=full_factor @ full_factor.T)
        for name, value in values.items():
            collected[name].append(value)
    result = {name: np.stack(values) for name, values in collected.items()}
    result.update(center_anchor(observations, bounds))
    # Preserve per-query matrix-vector order, matching the inherited mapping.
    anchor = np.stack([homogeneous @ a for a in result['anchor_target']])
    delta = result['mean_homogeneous']-anchor
    augmented = np.stack([augment_factor(m, f, a) for m, f, a in zip(
        result['mean_homogeneous'], result['factor_homogeneous'], anchor)])
    result.update(query_bounds_px=bounds,
        source_pixel_count=np.full(N_QUERY, 24*24, np.int64),
        source_within_basis=basis, homogeneous_transform=homogeneous,
        anchor_homogeneous=anchor, anchor_delta=delta,
        anchor_covariance=result['covariance_homogeneous']+delta[:, :, None]*delta[:, None, :],
        anchored_factor=augmented, camera_rotation_diagnostics=diagnostics)
    if set(result) != set(GEOMETRY_FIELDS) or not all(np.isfinite(v).all() for v in result.values()):
        raise ValueError('nonfinite prepared geometry; no repair')
    return result


def prepare_content(cache):
    """Preserve R25's exact whole-matrix float32 content/temperature product."""
    expected = {
        'g24__a': (np.float64, (N_QUERY, 2)),
        'g24__anchor_cov': (np.float64, (N_QUERY, 2, 2)),
        'g24__tau32': (np.float32, (N_QUERY,)),
        'g24__q_features': (np.float32, (N_QUERY, CHANNELS)),
        'g8__k_features': (np.float32, (N_KEY, CHANNELS)),
        'g8__bounds_px': (np.int64, (N_KEY, 4)),
    }
    if set(cache) != set(CACHE_FIELDS):
        raise ValueError('exact six-field cache allowlist required')
    for name, (dtype, shape) in expected.items():
        _array(cache[name], dtype, shape, name)
    covariance = cache['g24__anchor_cov']
    scale = np.maximum(1., np.max(np.abs(covariance), axis=(1, 2)))
    normalized = covariance/scale[:, None, None]
    asymmetry = np.max(np.abs(normalized-normalized.transpose(0, 2, 1)), axis=(1, 2))
    if (asymmetry > 1e-12).any():
        raise ValueError('archived covariance exceeds original normalized symmetry tolerance')
    diagonal = torch.from_numpy(covariance).diagonal(dim1=-2, dim2=-1)
    h2 = torch.tensor(.25**2, dtype=torch.float64)
    tau_scale = torch.maximum(diagonal.amax(-1), h2)
    relative_h2 = h2/tau_scale
    expected_tau = (10.*(relative_h2/(relative_h2+(diagonal/tau_scale[:, None]).sum(-1)))).float().numpy()
    if not np.isfinite(expected_tau).all() or not np.array_equal(expected_tau, cache['g24__tau32']):
        raise ValueError('archived trace temperature disagrees with fixed formula')
    if not np.array_equal(cache['g8__bounds_px'], native_bounds(8)):
        raise ValueError('native key cell identity/order changed')
    q, k = cache['g24__q_features'], cache['g8__k_features']
    with torch.no_grad():
        content = torch.from_numpy(q) @ torch.from_numpy(k).T
        bias = torch.from_numpy(cache['g24__tau32'])[:, None]*content
    if not bool(torch.isfinite(content).all() and torch.isfinite(bias).all()):
        raise ValueError('nonfinite full-matrix float32 content/bias')
    return dict(old_a2=cache['g24__a'].copy(), old_anchor_covariance2=covariance.copy(),
        old_covariance_asymmetry_normalized=asymmetry,
        tau32=cache['g24__tau32'].copy(), q_features32=q.copy(), k_features32=k.copy(),
        key_bounds_px=cache['g8__bounds_px'].copy(),
        key_bounds=(cache['g8__bounds_px'].astype(np.float64)-.5)/8.,
        content32=content.numpy(), bias32=bias.numpy(), content_bias64=bias.double().numpy())


def prepare_inputs(observations, cache):
    """Return a flat, fully archivable ndarray bank, without any I/O."""
    result = {**prepare_geometry(observations), **prepare_content(cache)}
    if set(result) != set(BANK_FIELDS):
        raise ValueError('prepared pair bank schema differs')
    return result


def model_inputs(bank, index):
    """Extract only the model's four inputs, with independent owned buffers."""
    if type(index) is not int or not 0 <= index < N_QUERY:
        raise ValueError('query index must be an integer in the full fixed population')
    if set(bank) != set(BANK_FIELDS):
        raise ValueError('prepared pair bank schema differs')
    fields = {'mu': ('anchor_homogeneous', (N_QUERY, 3)),
              'factor': ('anchored_factor', (N_QUERY, 3, 3)),
              'bounds': ('key_bounds', (N_KEY, 4)),
              'content_bias': ('content_bias64', (N_QUERY, N_KEY))}
    result = {}
    for output, (name, shape) in fields.items():
        value = _array(bank[name], np.float64, shape, name)
        result[output] = (value if output == 'bounds' else value[index]).copy()
    return result
