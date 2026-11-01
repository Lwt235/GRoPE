"""Pure NumPy transverse/radial decomposition of the frozen R39 kernel.

No files, labels, producer imports, probabilities, or model execution occur in
this module. The caller owns the frozen four-CPU-thread runtime and requires
all 48x432 entries and all three roles; small pure synthetic banks are allowed.

API: schur_components(q_anchor, q_cov, k_anchor, k_cov, gamma, role='full').
Returns exactly td, tv, rd, rv, s, residual, radial_delta, each float64 [Nq,Nk].
Here gamma is the R39 *variance* multiplying I, not its square root. Differences
are q_anchor-k_anchor; radial_delta is their signed projection onto the key
ray, and residual is that radial difference after transverse conditioning.
For the original fixed R39 masses, td+tv+rd+rv+log(9) reconstructs lograw.
"""
from __future__ import annotations

import numpy as np


ROLES = ('full', 'point', 'isotropic')
FIELDS = ('td', 'tv', 'rd', 'rv', 's', 'residual', 'radial_delta')
COVARIANCE_NORMALIZED_TOLERANCE = 1e-12


def _finite_array(value, shape, name):
    value = np.asarray(value)
    if value.dtype != np.float64 or value.shape != shape or not np.isfinite(value).all():
        raise ValueError('finite original float64 '+name+' with shape '+str(shape)+' required')
    return value


def _covariance_role(value, count, role, name):
    covariance = _finite_array(value, (count, 3, 3), name)
    scale = np.maximum(1., np.max(np.abs(covariance), axis=(1, 2)))
    asymmetry = np.max(np.abs(covariance-covariance.transpose(0, 2, 1)), axis=(1, 2))/scale
    if not np.isfinite(asymmetry).all() or np.any(asymmetry > COVARIANCE_NORMALIZED_TOLERANCE):
        raise ValueError('support symmetry admission failed: '+name)
    symmetric = .5*(covariance+covariance.transpose(0, 2, 1))
    if not np.isfinite(symmetric).all():
        raise ValueError('nonfinite symmetric support admission copy: '+name)
    eigenvalues = np.linalg.eigvalsh(symmetric)
    if (not np.isfinite(eigenvalues).all()
            or np.any(eigenvalues[:, 0]/scale < -COVARIANCE_NORMALIZED_TOLERANCE)):
        raise ValueError('support PSD admission failed; no eigenvalue repair: '+name)
    # The symmetric copy above is only for admission. Preserve the original
    # supports until the summed S is symmetrized, as in the R39 contract.
    if role == 'point':
        return np.zeros_like(covariance)
    if role == 'isotropic':
        return np.trace(covariance, axis1=1, axis2=2)[:, None, None]*np.eye(3)/3.
    return covariance


def _key_basis(key):
    # Stable k/||k|| without squaring very large or very small physical units.
    scale = np.max(np.abs(key), axis=1)
    if np.any(scale == 0):
        raise ValueError('zero key anchor has no declared radial direction')
    scaled = key/scale[:, None]
    ray = scaled/np.sqrt(np.sum(scaled*scaled, axis=1))[:, None]
    # np.argmin retains the first coordinate axis for exact equal minima.
    axis = np.eye(3, dtype=np.float64)[np.argmin(np.abs(ray), axis=1)]
    u = np.cross(ray, axis)
    norm = np.sqrt(np.sum(u*u, axis=1))
    if not np.isfinite(norm).all() or np.any(norm <= 0):
        raise ValueError('declared transverse basis is not representable')
    u = u/norm[:, None]
    v = np.cross(ray, u)
    basis = np.stack((u, v, ray), axis=-1)
    if (not np.isfinite(basis).all()
            or np.max(np.abs(basis.swapaxes(-1, -2) @ basis-np.eye(3))) > 1e-10
            or np.max(np.abs(basis[:, :, 2]-ray)) > 1e-10):
        raise ValueError('nonfinite or nonorthogonal key-ray basis')
    return basis


def schur_components(q_anchor, q_cov, k_anchor, k_cov, gamma, role='full'):
    """Decompose every supplied pair, retaining signed radial diagnostics.

    Schur subtraction can lose precision even for a mathematically positive
    matrix. Nonpositive/nonfinite results fail explicitly; no jitter, clipping,
    population filtering, or alternative numerical method is attempted.
    """
    if role not in ROLES:
        raise ValueError('only full/point/isotropic roles are declared')
    q, k = np.asarray(q_anchor), np.asarray(k_anchor)
    if q.ndim != 2 or k.ndim != 2 or not len(q) or not len(k):
        raise ValueError('nonempty query/key anchor banks required')
    nq, nk = len(q), len(k)
    q = _finite_array(q, (nq, 3), 'query anchor')
    k = _finite_array(k, (nk, 3), 'key anchor')
    qc = _covariance_role(q_cov, nq, role, 'query covariance')
    kc = _covariance_role(k_cov, nk, role, 'key covariance')
    gamma = _finite_array(gamma, (), 'gamma')
    if gamma <= 0:
        raise ValueError('strictly positive gamma required')
    basis = _key_basis(k)
    # Retain the original R39 producer's addition and symmetrization order.
    summed = qc[:, None]+kc[None]+float(gamma)*np.eye(3)
    summed = .5*(summed+summed.swapaxes(-1, -2))
    difference = q[:, None]-k[None]
    if not np.isfinite(summed).all() or not np.isfinite(difference).all():
        raise ValueError('summed covariance or anchor difference overflow')
    left = basis.swapaxes(-1, -2)[None]
    local = np.matmul(left, np.matmul(summed, basis[None]))
    delta = np.matmul(left, difference[..., None])[..., 0]
    if not np.isfinite(local).all() or not np.isfinite(delta).all():
        raise ValueError('nonfinite transverse/radial coordinates')
    transverse = local[..., :2, :2]
    coupling = local[..., :2, 2]
    radial_variance = local[..., 2, 2]
    d_transverse, d_radial = delta[..., :2], delta[..., 2]
    # Cholesky evaluates all A^{-1} contractions via the same factor. It is
    # not an eigenvalue repair. A or s outside the numerical domain fails.
    lower = np.linalg.cholesky(transverse)
    white_delta = np.linalg.solve(lower, d_transverse[..., None])[..., 0]
    white_coupling = np.linalg.solve(lower, coupling[..., None])[..., 0]
    schur = radial_variance-np.sum(white_coupling*white_coupling, axis=-1)
    residual = d_radial-np.sum(white_coupling*white_delta, axis=-1)
    if not np.isfinite(schur).all() or np.any(schur <= 0):
        raise ValueError('Schur variance must be strictly positive; no repair')
    log_gamma = np.log(float(gamma))
    result = dict(
        td=-.5*np.sum(white_delta*white_delta, axis=-1),
        tv=log_gamma-np.sum(np.log(np.diagonal(lower, axis1=-2, axis2=-1)), axis=-1),
        rd=-.5*(residual*residual)/schur,
        rv=.5*(log_gamma-np.log(schur)),
        s=schur, residual=residual, radial_delta=d_radial.copy())
    if set(result) != set(FIELDS) or any(
            value.dtype != np.float64 or value.shape != (nq, nk)
            or not np.isfinite(value).all() for value in result.values()):
        raise ValueError('nonfinite or incomplete Schur component matrices')
    return result
