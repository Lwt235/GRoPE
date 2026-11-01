"""Independent Schur components of saved R39 Gaussian overlap inputs.

Pure NumPy/math only: no experiment/model imports, file access, labels,
probability construction or data-dependent parameter selection.
"""
from __future__ import annotations

import math

import numpy as np


ROLES = ('full', 'point', 'isotropic')
COVARIANCE_TOLERANCE = 1e-12
BASIS_TOLERANCE = 1e-10
FIELDS = ('td', 'tv', 'rd', 'rv', 's', 'residual', 'radial_delta')


def _finite(value, shape, name):
    result = np.asarray(value)
    if result.dtype != np.float64 or result.shape != shape or not np.isfinite(result).all():
        raise ValueError('finite float64 shape mismatch: ' + name)
    return result


def _covariance(value, count, role, name):
    covariance = _finite(value, (count, 3, 3), name)
    magnitude = np.maximum(1., np.max(np.abs(covariance), axis=(-2, -1)))
    transpose = covariance.swapaxes(-2, -1)
    asymmetry = np.max(np.abs(covariance - transpose), axis=(-2, -1)) / magnitude
    if np.any(asymmetry > COVARIANCE_TOLERANCE):
        raise ValueError('support symmetry admission failed: ' + name)
    eigenvalues = np.linalg.eigvalsh(.5 * (covariance + transpose))
    if np.any(eigenvalues[:, 0] / magnitude < -COVARIANCE_TOLERANCE):
        raise ValueError('support PSD admission failed: ' + name)
    if role == 'point':
        return np.zeros_like(covariance)
    if role == 'isotropic':
        return np.trace(covariance, axis1=-2, axis2=-1)[:, None, None] * np.eye(3) / 3.
    return covariance


def _radial_basis(key_anchors):
    """Stable Householder, with final radial row equal to +r transpose.

    With sign determined by r_z, v=r+sign*e_z avoids cancellation.  The
    reflection maps r to -sign*e_z; multiplying its third row by -sign
    restores the declared radial direction.  No QR/cross-product basis is
    shared with a producer.  Singular rays or failed checks raise errors.
    """
    bases, directions = [], []
    identity = np.eye(3, dtype=np.float64)
    ez = np.asarray([0., 0., 1.], np.float64)
    for point in key_anchors:
        scale = float(np.max(np.abs(point)))
        if scale == 0:
            raise ValueError('key anchor at the camera center has no radial direction')
        scaled = point / scale
        radius = math.hypot(*map(float, scaled))
        direction = scaled / radius
        sign = 1. if direction[2] >= 0 else -1.
        vector = direction + sign * ez
        denominator = math.fsum(float(v) * float(v) for v in vector)
        if not math.isfinite(denominator) or denominator <= 0:
            raise ValueError('invalid Householder vector')
        basis = identity - (2. / denominator) * np.outer(vector, vector)
        basis[2] *= -sign
        orthogonality_error = float(np.max(np.abs(basis @ basis.T - identity)))
        radial_row_error = float(np.max(np.abs(basis[2] - direction)))
        if (not np.isfinite(basis).all() or orthogonality_error > BASIS_TOLERANCE
                or radial_row_error > BASIS_TOLERANCE):
            raise ValueError('independent radial basis check failed; no repair')
        bases.append(basis)
        directions.append(direction)
    return np.stack(bases), np.stack(directions)


def audit_components(q_anchor, q_cov, k_anchor, k_cov, gamma, role='full'):
    """Return seven float64 [n_query,n_key] Schur component matrices.

    td/tv are transverse distance/volume log terms; rd/rv are conditional
    radial distance/volume log terms.  Their sum plus log(9) reconstructs
    the declared raw overlap score.  s is the conditional radial variance,
    residual its conditional signed displacement, and radial_delta the
    original signed displacement along the normalized key-anchor ray.

    No absolute depth, front/FOV, opacity or visibility filtering occurs.
    Input/population provenance and reconstruction comparison belong to the
    caller; this function never constructs or changes a probability matrix.
    """
    if role not in ROLES:
        raise ValueError('undeclared R39 role')
    queries, keys = np.asarray(q_anchor), np.asarray(k_anchor)
    if queries.ndim != 2 or keys.ndim != 2 or len(queries) == 0 or len(keys) == 0:
        raise ValueError('nonempty query/key anchor matrices required')
    nq, nk = len(queries), len(keys)
    queries = _finite(queries, (nq, 3), 'q_anchor')
    keys = _finite(keys, (nk, 3), 'k_anchor')
    qcov = _covariance(q_cov, nq, role, 'q_cov')
    kcov = _covariance(k_cov, nk, role, 'k_cov')
    bandwidth = float(_finite(gamma, (), 'gamma'))
    if bandwidth <= 0:
        raise ValueError('strictly positive gamma required')

    basis, directions = _radial_basis(keys)
    matrix = qcov[:, None] + kcov[None] + bandwidth * np.eye(3)
    matrix = .5 * (matrix + matrix.swapaxes(-2, -1))
    rotated = basis[None] @ matrix @ basis[None].swapaxes(-2, -1)
    delta = queries[:, None] - keys[None]
    displacement = np.einsum('kij,qkj->qki', basis, delta)
    direct_radial = np.einsum('ki,qki->qk', directions, delta)
    radial = displacement[..., 2]
    radial_error = np.abs(radial - direct_radial) / np.maximum(
        1., np.maximum(np.abs(radial), np.abs(direct_radial)))
    if not np.isfinite(rotated).all() or np.any(radial_error > BASIS_TOLERANCE):
        raise ValueError('rotated covariance/radial sign check failed')

    a00 = rotated[..., 0, 0]
    a01 = .5 * (rotated[..., 0, 1] + rotated[..., 1, 0])
    a11 = rotated[..., 1, 1]
    a_scale = np.maximum(np.maximum(np.abs(a00), np.abs(a01)), np.abs(a11))
    if not np.isfinite(a_scale).all() or np.any(a_scale <= 0):
        raise ValueError('positive finite transverse scale required')
    aa, ab, bb = a00 / a_scale, a01 / a_scale, a11 / a_scale
    determinant = aa * bb - ab * ab
    if np.any(aa <= 0) or np.any(bb <= 0) or np.any(determinant <= 0):
        raise ValueError('positive definite transverse block required; no jitter')
    inverse00 = (bb / determinant) / a_scale
    inverse01 = (-ab / determinant) / a_scale
    inverse11 = (aa / determinant) / a_scale
    # Transforming a symmetric matrix may leave floating antisymmetry.  The
    # off-diagonal entries below are its symmetric part, with no PSD repair.
    b0 = .5 * (rotated[..., 0, 2] + rotated[..., 2, 0])
    b1 = .5 * (rotated[..., 1, 2] + rotated[..., 2, 1])
    c = rotated[..., 2, 2]
    d0, d1 = displacement[..., 0], displacement[..., 1]
    inverse_b0 = inverse00 * b0 + inverse01 * b1
    inverse_b1 = inverse01 * b0 + inverse11 * b1
    inverse_d0 = inverse00 * d0 + inverse01 * d1
    inverse_d1 = inverse01 * d0 + inverse11 * d1
    conditional_variance = c - (b0 * inverse_b0 + b1 * inverse_b1)
    residual = radial - (b0 * inverse_d0 + b1 * inverse_d1)
    if not np.isfinite(conditional_variance).all() or np.any(conditional_variance <= 0):
        raise ValueError('positive finite Schur complement required; no clipping')
    log_gamma = math.log(bandwidth)
    log_determinant = 2. * np.log(a_scale) + np.log(determinant)
    result = dict(
        td=-.5 * (d0 * inverse_d0 + d1 * inverse_d1),
        tv=log_gamma - .5 * log_determinant,
        rd=-.5 * np.square(residual / np.sqrt(conditional_variance)),
        rv=.5 * (log_gamma - np.log(conditional_variance)),
        s=conditional_variance,
        residual=residual,
        radial_delta=radial,
    )
    if set(result) != set(FIELDS) or not all(
            value.shape == (nq, nk) and value.dtype == np.float64
            and np.isfinite(value).all() for value in result.values()):
        raise ValueError('nonfinite Schur component arithmetic; no repair')
    return result
