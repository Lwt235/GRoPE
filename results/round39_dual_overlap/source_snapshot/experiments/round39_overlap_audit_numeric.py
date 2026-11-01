"""Independent, pure-array R39 geometry and Gaussian-overlap arithmetic.

No experiment/model imports, archive access, integration or labels.  The full
population and provenance are enforced by the caller.  The two entry points
below do not mutate their inputs and perform no learned calibration or repair.
"""
from __future__ import annotations

import math

import numpy as np


IMAGE_HW = (144, 192)
OBSERVATION_FIELDS = frozenset((
    'q_predicted_depth', 'k_predicted_depth', 'q_K', 'k_K', 'q_to_k_transform'))
ROLES = ('full', 'point', 'isotropic')
COVARIANCE_NORMALIZED_TOLERANCE = 1e-12
QUERY_MASS = 9.
KEY_MASS = 1.
FLOOR_DENSITY = 1e-30


def _finite_array(value, shape, name, dtype=np.float64):
    result = np.asarray(value)
    if result.shape != shape or result.dtype != np.dtype(dtype):
        raise ValueError('shape/dtype mismatch: ' + name)
    if not np.isfinite(result).all():
        raise ValueError('nonfinite values: ' + name)
    return result


def _clean_depth(value, name):
    supplied = np.asarray(value)
    if supplied.shape != IMAGE_HW or not np.issubdtype(supplied.dtype, np.floating):
        raise ValueError('floating 144 by 192 predicted depth required: ' + name)
    depth = supplied.astype(np.float64, copy=True)
    depth[np.isnan(depth)] = 10.
    depth[np.isposinf(depth)] = 20.
    depth[np.isneginf(depth)] = .1
    return np.minimum(np.maximum(depth, .1), 20.)


def _side(depth, intrinsics, patch_size):
    """Patch central covariance plus mean-to-anchor displacement identity."""
    height, width = IMAGE_HW
    inverse = np.linalg.inv(intrinsics)
    if not np.isfinite(inverse).all():
        raise ValueError('intrinsic inversion overflow')
    rows, columns = np.indices(IMAGE_HW, dtype=np.float64)
    pixels = np.stack((columns, rows, np.ones(IMAGE_HW)), axis=-1)
    rays = np.einsum('ij,hwj->hwi', inverse, pixels)
    points = rays * depth[..., None]
    basis = inverse[:, :2]
    pixel_covariance_unit = basis @ basis.T / 12.
    anchors, covariances = [], []
    for y0 in range(0, height, patch_size):
        for x0 in range(0, width, patch_size):
            cx = x0 + (patch_size - 1) / 2.
            cy = y0 + (patch_size - 1) / 2.
            ix, iy = math.floor(cx), math.floor(cy)
            fx, fy = cx - ix, cy - iy
            # Explicit bilinear interpolation in two independent stages.
            top = (1. - fx) * float(depth[iy, ix]) + fx * float(depth[iy, ix + 1])
            bottom = (1. - fx) * float(depth[iy + 1, ix]) + fx * float(depth[iy + 1, ix + 1])
            center_depth = (1. - fy) * top + fy * bottom
            anchor = center_depth * (inverse @ np.asarray([cx, cy, 1.], np.float64))
            patch = points[y0:y0 + patch_size, x0:x0 + patch_size].reshape(-1, 3)
            centroid = np.mean(patch, axis=0)
            displacement = centroid - anchor
            scatter = np.cov(patch, rowvar=False, bias=True)
            patch_depth = depth[y0:y0 + patch_size, x0:x0 + patch_size]
            mean_depth_squared = float(np.mean(patch_depth * patch_depth))
            covariance = (scatter + np.outer(displacement, displacement)
                          + mean_depth_squared * pixel_covariance_unit)
            anchors.append(anchor)
            covariances.append(covariance)
    squared_pixel_scale = depth * depth * float(np.sum(basis * basis)) / 2.
    result = (np.stack(anchors), np.stack(covariances), squared_pixel_scale)
    if not all(np.isfinite(value).all() for value in result):
        raise ValueError('nonfinite independently reconstructed geometry')
    return result


def audit_prepare(obs):
    """Return q/k anchors, anchored covariances and scalar gamma, all float64.

    Exactly five observation fields are accepted.  q outputs are in the key
    camera; k outputs stay in that camera.  No front/FOV filtering is applied.
    gamma is four times the median of BOTH full-image pixel-scale populations,
    with each pixel contributing one value before the single median operation.
    """
    if set(obs) != OBSERVATION_FIELDS:
        raise ValueError('exact five-field predicted observation boundary required')
    q_depth = _clean_depth(obs['q_predicted_depth'], 'q_predicted_depth')
    k_depth = _clean_depth(obs['k_predicted_depth'], 'k_predicted_depth')
    q_K = _finite_array(obs['q_K'], (3, 3), 'q_K')
    k_K = _finite_array(obs['k_K'], (3, 3), 'k_K')
    transform = _finite_array(obs['q_to_k_transform'], (4, 4), 'q_to_k_transform')
    if (not np.array_equal(q_K[2], [0., 0., 1.])
            or not np.array_equal(k_K[2], [0., 0., 1.])
            or not np.array_equal(transform[3], [0., 0., 0., 1.])):
        raise ValueError('standard pinhole and homogeneous camera matrices required')
    rotation, translation = transform[:3, :3], transform[:3, 3]
    if (np.max(np.abs(rotation @ rotation.T - np.eye(3))) > 1e-6
            or abs(float(np.linalg.det(rotation)) - 1.) > 1e-6):
        raise ValueError('supplied camera mapping must be rigid; no projection to a rotation')
    q_anchor_source, q_cov_source, q_scale = _side(q_depth, q_K, 24)
    k_anchor, k_cov, k_scale = _side(k_depth, k_K, 8)
    q_anchor = np.stack([rotation @ point + translation for point in q_anchor_source])
    q_cov = np.stack([rotation @ covariance @ rotation.T for covariance in q_cov_source])
    ell_squared = float(np.median(np.concatenate((q_scale.ravel(), k_scale.ravel()))))
    gamma = np.asarray(4. * ell_squared, np.float64)
    if not math.isfinite(float(gamma)) or float(gamma) <= 0:
        raise ValueError('positive finite Gaussian bandwidth squared required')
    result = dict(q_anchor=q_anchor, q_cov=q_cov, k_anchor=k_anchor, k_cov=k_cov, gamma=gamma)
    if not all(np.isfinite(value).all() for value in result.values()):
        raise ValueError('nonfinite mapped geometry; no repair')
    return result


def _covariances(value, count, name):
    covariance = _finite_array(value, (count, 3, 3), name)
    scale = np.maximum(1., np.max(np.abs(covariance), axis=(1, 2)))
    normalized_asymmetry = np.max(np.abs(covariance - covariance.transpose(0, 2, 1)), axis=(1, 2)) / scale
    if np.any(normalized_asymmetry > COVARIANCE_NORMALIZED_TOLERANCE):
        raise ValueError('covariance asymmetry exceeds numerical admission: ' + name)
    symmetric = (covariance + covariance.transpose(0, 2, 1)) * .5
    eigenvalues = np.linalg.eigvalsh(symmetric)
    if np.any(eigenvalues[:, 0] / scale < -COVARIANCE_NORMALIZED_TOLERANCE):
        raise ValueError('covariance is not positive semidefinite: ' + name)
    # The symmetric copy is used only for PSD admission.  Preserve the
    # supplied support; symmetrize the summed S explicitly below.
    return covariance


def audit_scores(qanchor, qcov, kanchor, kcov, gamma, content, role='full'):
    """Independent eigensystem evaluation of the declared Gaussian kernel.

    q mass=576/64=9 and k mass=64/64=1 are included in lograw.  The only
    floor is their product times 1e-30, added once before the original content
    bias.  Returned keys are lograw/logfloored/logp/p/winner; winner is int32.
    Small synthetic populations are supported; the formal caller requires
    all 48 by 432 entries per pair and all 48 pairs.
    """
    if role not in ROLES:
        raise ValueError('only full/point/isotropic are declared')
    qanchor, kanchor = np.asarray(qanchor), np.asarray(kanchor)
    if qanchor.ndim != 2 or kanchor.ndim != 2 or not len(qanchor) or not len(kanchor):
        raise ValueError('nonempty anchor matrices required')
    nq, nk = len(qanchor), len(kanchor)
    qanchor = _finite_array(qanchor, (nq, 3), 'qanchor')
    kanchor = _finite_array(kanchor, (nk, 3), 'kanchor')
    qcov = _covariances(qcov, nq, 'qcov')
    kcov = _covariances(kcov, nk, 'kcov')
    gamma = _finite_array(gamma, (), 'gamma')
    bandwidth_squared = float(gamma)
    if bandwidth_squared <= 0:
        raise ValueError('strictly positive gamma required')
    supplied_content = np.asarray(content)
    if (supplied_content.shape != (nq, nk)
            or supplied_content.dtype not in (np.dtype(np.float32), np.dtype(np.float64))
            or not np.isfinite(supplied_content).all()):
        raise ValueError('finite original float32 bias or its float64 cast required')
    bias = supplied_content.astype(np.float64, copy=False)
    identity = np.eye(3, dtype=np.float64)
    if role == 'point':
        qcov, kcov = np.zeros_like(qcov), np.zeros_like(kcov)
    elif role == 'isotropic':
        qcov = np.trace(qcov, axis1=1, axis2=2)[:, None, None] / 3. * identity
        kcov = np.trace(kcov, axis1=1, axis2=2)[:, None, None] / 3. * identity
    lograw = np.empty((nq, nk), np.float64)
    log_mass_product = math.log(QUERY_MASS * KEY_MASS)
    log_gamma = math.log(bandwidth_squared)
    for index in range(nq):
        matrices = bandwidth_squared * identity + qcov[index] + kcov
        matrices = .5 * (matrices + matrices.swapaxes(-1, -2))
        eigenvalues, eigenvectors = np.linalg.eigh(matrices)
        if not np.isfinite(eigenvalues).all() or np.any(eigenvalues <= 0):
            raise ValueError('positive definite summed covariance required; no jitter')
        differences = qanchor[index] - kanchor
        coordinates = np.einsum('kij,ki->kj', eigenvectors, differences)
        quadratic = np.sum((coordinates * coordinates) / eigenvalues, axis=1)
        log_determinant = np.sum(np.log(eigenvalues), axis=1)
        lograw[index] = .5 * (3. * log_gamma - log_determinant - quadratic) + log_mass_product
    if not np.isfinite(lograw).all():
        raise ValueError('nonfinite raw overlap score; no salvage by the floor')
    log_floor = log_mass_product + math.log(FLOOR_DENSITY)
    logfloored = np.logaddexp(lograw, log_floor)
    weighted = logfloored + bias
    if not np.isfinite(weighted).all():
        raise ValueError('nonfinite content-weighted logits')
    logp = np.empty_like(weighted)
    for index, scores in enumerate(weighted):
        largest = float(np.max(scores))
        shifted_sum = math.fsum(math.exp(float(score) - largest) for score in scores)
        log_normalizer = largest + math.log(shifted_sum)
        logp[index] = scores - log_normalizer
    with np.errstate(under='ignore'):
        probability = np.exp(logp)
    result = dict(lograw=lograw, logfloored=logfloored, logp=logp, p=probability,
                  winner=np.argmax(probability, axis=1).astype(np.int32))
    if not all(np.isfinite(value).all() for value in result.values()):
        raise ValueError('nonfinite overlap/normalization arithmetic; no repair')
    if any(abs(math.fsum(map(float, row)) - 1.) > 2e-12 for row in probability):
        raise ValueError('independent probability normalization exceeds 2e-12')
    return result
