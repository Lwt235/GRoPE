"""Fixed R40 Gaussian half-ray position bias, using frozen R39 geometry.

This pure NumPy/SciPy module has no data loading or producer dependencies.
The caller supplies the unchanged physical camera-frame anchors, supports,
gamma (a variance), and original content bias, and owns the four-CPU runtime.
Every supplied query/key pair is evaluated, including rear-facing queries.

The key anchor defines only a ray direction. Its covariance and gamma remain
the supplied R39 quantities; neither is recomputed from that direction.
"""
from __future__ import annotations

import numpy as np
from scipy.special import log_ndtr, logsumexp


ROLES = ('full', 'point', 'isotropic')
FIELDS = ('lograw', 'logfloored', 'logp', 'p', 'winner', 'T', 'log_front')
COVARIANCE_NORMALIZED_TOLERANCE = 1e-12
BASIS_TOLERANCE = 1e-10


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
    # Admission uses a symmetric copy, while role construction retains the
    # original input until the summed S is symmetrized, as in R39.
    if role == 'point':
        return np.zeros_like(covariance)
    if role == 'isotropic':
        return np.trace(covariance, axis1=1, axis2=2)[:, None, None]*np.eye(3)/3.
    return covariance


def _key_basis(key):
    scale = np.max(np.abs(key), axis=1)
    if np.any(scale == 0):
        raise ValueError('zero key anchor has no declared ray direction')
    scaled = key/scale[:, None]
    ray = scaled/np.sqrt(np.sum(scaled*scaled, axis=1))[:, None]
    # The first least-aligned coordinate axis resolves exact ties.
    axis = np.eye(3, dtype=np.float64)[np.argmin(np.abs(ray), axis=1)]
    transverse_u = np.cross(ray, axis)
    norm = np.sqrt(np.sum(transverse_u*transverse_u, axis=1))
    if not np.isfinite(norm).all() or np.any(norm <= 0):
        raise ValueError('declared transverse basis is not representable')
    transverse_u = transverse_u/norm[:, None]
    transverse_v = np.cross(ray, transverse_u)
    basis = np.stack((transverse_u, transverse_v, ray), axis=-1)
    if (not np.isfinite(basis).all()
            or np.max(np.abs(basis.swapaxes(-1, -2) @ basis-np.eye(3))) > BASIS_TOLERANCE
            or np.max(np.abs(basis[:, :, 2]-ray)) > BASIS_TOLERANCE):
        raise ValueError('nonfinite or nonorthogonal key-ray basis')
    return basis


def score(q_anchor, q_cov, k_anchor, k_cov, gamma, content, role='full'):
    """Score the complete supplied pair bank with the fixed half-ray kernel.

    Returns the fields in FIELDS. All matrices are float64 [Nq,Nk]; winner
    is int32 [Nq] and uses the first maximum of the materialized probability.
    T is the transverse log factor; log_front=log(Phi(m/sqrt(s))). Internally,
    m is the signed conditional radial mean relative to the camera origin,
    and s is its Schur variance. Neither is returned. The fixed raw mass is
    9*exp(T)*Phi(m/sqrt(s)).

    Small synthetic banks are supported. Formal population/order checks are
    the caller's responsibility. Invalid numerical domains fail explicitly;
    no jitter, clipping, finite integration cutoff, or population filtering
    is applied.
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
    content = np.asarray(content)
    if (content.dtype not in (np.dtype('float32'), np.dtype('float64'))
            or content.shape != (nq, nk) or not np.isfinite(content).all()):
        raise ValueError('finite original float32/float64 content matrix required')
    content = content.astype(np.float64, copy=False)

    basis = _key_basis(k)
    # Preserve R39's addition and final symmetrization order.
    summed = qc[:, None]+kc[None]+float(gamma)*np.eye(3)
    summed = .5*(summed+summed.swapaxes(-1, -2))
    if not np.isfinite(summed).all():
        raise ValueError('nonfinite summed covariance')
    left = basis.swapaxes(-1, -2)[None]
    local = np.matmul(left, np.matmul(summed, basis[None]))
    # Project q from the camera origin directly. Subtracting k first would
    # introduce cancellation and the wrong radial origin for the half-ray.
    coordinates = np.matmul(left, q[:, None, :, None])[..., 0]
    if not np.isfinite(local).all() or not np.isfinite(coordinates).all():
        raise ValueError('nonfinite transverse/radial coordinates')
    transverse = local[..., :2, :2]
    coupling = local[..., :2, 2]
    radial_variance = local[..., 2, 2]
    d_transverse, radial_mean = coordinates[..., :2], coordinates[..., 2]
    lower = np.linalg.cholesky(transverse)
    white_delta = np.linalg.solve(lower, d_transverse[..., None])[..., 0]
    white_coupling = np.linalg.solve(lower, coupling[..., None])[..., 0]
    schur = radial_variance-np.sum(white_coupling*white_coupling, axis=-1)
    conditional_mean = radial_mean-np.sum(white_coupling*white_delta, axis=-1)
    if not np.isfinite(schur).all() or np.any(schur <= 0):
        raise ValueError('Schur variance must be strictly positive; no repair')
    if not np.isfinite(conditional_mean).all():
        raise ValueError('nonfinite conditional radial mean')
    transverse_log = (np.log(float(gamma))
                      - np.sum(np.log(np.diagonal(lower, axis1=-2, axis2=-1)), axis=-1)
                      - .5*np.sum(white_delta*white_delta, axis=-1))
    standardized_mean = conditional_mean/np.sqrt(schur)
    if not np.isfinite(standardized_mean).all():
        raise ValueError('conditional radial standardization overflow')
    log_front = log_ndtr(standardized_mean)
    lograw = np.log(9.)+transverse_log+log_front
    # Apply the original mass floor exactly once, then the original content.
    logfloored = np.logaddexp(lograw, np.log(9.e-30))
    logits = logfloored+content
    logp = logits-logsumexp(logits, axis=1, keepdims=True)
    with np.errstate(under='ignore'):
        p = np.exp(logp)
    result = dict(lograw=lograw, logfloored=logfloored, logp=logp, p=p,
                  winner=p.argmax(axis=1).astype(np.int32),
                  T=transverse_log, log_front=log_front)
    if set(result) != set(FIELDS) or any(
            value.dtype != np.float64 or value.shape != (nq, nk)
            or not np.isfinite(value).all()
            for name, value in result.items() if name != 'winner'):
        raise ValueError('nonfinite or incomplete half-ray output matrices')
    if np.max(np.abs(p.sum(axis=1)-1.)) > 2e-12:
        raise ValueError('probability normalization failure')
    return result
