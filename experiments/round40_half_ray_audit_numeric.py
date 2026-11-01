"""Independent 3D precision calculation of the fixed R40 half-ray kernel.

Pure NumPy/SciPy array arithmetic: no experiment/model imports, archive access,
labels, geometry reconstruction or parameter selection.  The caller enforces
the saved population and provenance.  Inputs are not modified or repaired.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.special import erfc, erfcx


ROLES = ('full', 'point', 'isotropic')
COVARIANCE_NORMALIZED_TOLERANCE = 1e-12
QUERY_MASS = 9.
KEY_MASS = 1.
FLOOR_DENSITY = 1e-30


def _finite_array(value, shape, name):
    result = np.asarray(value)
    if result.shape != shape or result.dtype != np.dtype(np.float64):
        raise ValueError('float64 shape/dtype mismatch: ' + name)
    if not np.isfinite(result).all():
        raise ValueError('nonfinite values: ' + name)
    return result


def _covariances(value, count, name):
    covariance = _finite_array(value, (count, 3, 3), name)
    scale = np.maximum(1., np.max(np.abs(covariance), axis=(1, 2)))
    transposed = covariance.swapaxes(-1, -2)
    asymmetry = np.max(np.abs(covariance - transposed), axis=(1, 2)) / scale
    if np.any(asymmetry > COVARIANCE_NORMALIZED_TOLERANCE):
        raise ValueError('covariance symmetry admission failed: ' + name)
    # Admission does not replace the supplied covariance with a repaired one.
    symmetric = .5 * (covariance + transposed)
    eigenvalues = np.linalg.eigvalsh(symmetric)
    if (not np.isfinite(eigenvalues).all()
            or np.any(eigenvalues[:, 0] / scale < -COVARIANCE_NORMALIZED_TOLERANCE)):
        raise ValueError('covariance PSD admission failed: ' + name)
    return covariance


def _unit_rays(anchors):
    scales = np.max(np.abs(anchors), axis=1)
    if np.any(scales == 0.):
        raise ValueError('zero key anchor does not define a viewing ray')
    scaled = anchors / scales[:, None]
    lengths = np.sqrt(np.sum(scaled * scaled, axis=1))
    rays = scaled / lengths[:, None]
    if (not np.isfinite(rays).all()
            or np.max(np.abs(np.sum(rays * rays, axis=1) - 1.)) > 1e-10):
        raise ValueError('invalid key unit-ray arithmetic')
    return rays


def _log_normal_cdf(arguments):
    """Independent log Phi; erfcx keeps the negative tail in log space."""
    x = np.asarray(arguments, dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError('nonfinite normal-CDF argument')
    answer = np.empty_like(x)
    negative = x <= 0.
    root_two = math.sqrt(2.)
    with np.errstate(over='raise', invalid='raise', divide='raise', under='ignore'):
        tail = x[negative]
        answer[negative] = (-.5 * tail * tail
                            + np.log(erfcx(-tail / root_two)) - math.log(2.))
        answer[~negative] = np.log1p(-.5 * erfc(x[~negative] / root_two))
    if not np.isfinite(answer).all() or np.any(answer > 0.):
        raise ValueError('nonfinite or positive log-CDF; no tail clipping')
    return answer


def audit_scores(q_anchor, q_cov, k_anchor, k_cov, gamma, content, role='full'):
    """Return six float64 matrices and one int32 winner row vector.

    Keys are lograw/logfloored/logp/p/winner/T/log_front.  T is the normalized
    whole-line log integral without log mass; log_front is the half-ray factor.
    Query/key masses are 9/1.  The one mass-weighted floor precedes the original
    content bias and one softmax.  All queries, including those behind the key
    camera, are retained.  Small synthetic populations are supported here;
    formal population requirements belong to the caller.

    The ray starts at the key camera origin.  q_anchor is therefore used
    directly, never q_anchor-k_anchor.  Key anchor magnitude does not enter
    the calculation except to obtain its unit direction.
    """
    if role not in ROLES:
        raise ValueError('only full/point/isotropic are declared')
    q_anchor, k_anchor = np.asarray(q_anchor), np.asarray(k_anchor)
    if (q_anchor.ndim != 2 or k_anchor.ndim != 2
            or not len(q_anchor) or not len(k_anchor)):
        raise ValueError('nonempty query and key anchor matrices required')
    nq, nk = len(q_anchor), len(k_anchor)
    q_anchor = _finite_array(q_anchor, (nq, 3), 'q_anchor')
    k_anchor = _finite_array(k_anchor, (nk, 3), 'k_anchor')
    q_cov = _covariances(q_cov, nq, 'q_cov')
    k_cov = _covariances(k_cov, nk, 'k_cov')
    bandwidth_squared = float(_finite_array(gamma, (), 'gamma'))
    if bandwidth_squared <= 0.:
        raise ValueError('strictly positive gamma required')
    supplied_content = np.asarray(content)
    if (supplied_content.shape != (nq, nk)
            or supplied_content.dtype not in (np.dtype(np.float32), np.dtype(np.float64))
            or not np.isfinite(supplied_content).all()):
        raise ValueError('finite original float32 bias or its float64 cast required')
    bias = supplied_content.astype(np.float64, copy=False)
    rays = _unit_rays(k_anchor)
    identity = np.eye(3, dtype=np.float64)
    if role == 'point':
        q_cov, k_cov = np.zeros_like(q_cov), np.zeros_like(k_cov)
    elif role == 'isotropic':
        q_cov = np.trace(q_cov, axis1=1, axis2=2)[:, None, None] / 3. * identity
        k_cov = np.trace(k_cov, axis1=1, axis2=2)[:, None, None] / 3. * identity

    transverse = np.empty((nq, nk), dtype=np.float64)
    log_front = np.empty_like(transverse)
    log_gamma = math.log(bandwidth_squared)
    for index, ray in enumerate(rays):
        covariance = bandwidth_squared * identity + q_cov + k_cov[index]
        covariance = .5 * (covariance + covariance.swapaxes(-1, -2))
        if not np.isfinite(covariance).all():
            raise ValueError('nonfinite summed covariance')
        factor = np.linalg.cholesky(covariance)
        repeated_ray = np.broadcast_to(ray, (nq, 3))
        # L^-1 vectors give the action of the full 3D precision quadratic.
        white_ray = np.linalg.solve(factor, repeated_ray[..., None])[..., 0]
        white_query = np.linalg.solve(factor, q_anchor[..., None])[..., 0]
        precision_along_ray = np.sum(white_ray * white_ray, axis=1)
        beta = np.sum(white_ray * white_query, axis=1)
        if (not np.isfinite(precision_along_ray).all()
                or np.any(precision_along_ray <= 0.) or not np.isfinite(beta).all()):
            raise ValueError('invalid precision along ray; no regularization')
        best_position = beta / precision_along_ray
        perpendicular = q_anchor - best_position[:, None] * ray
        white_perpendicular = np.linalg.solve(factor, perpendicular[..., None])[..., 0]
        # This residual form avoids subtracting beta^2/c from q^T S^-1 q.
        quadratic = np.sum(white_perpendicular * white_perpendicular, axis=1)
        log_determinant = 2. * np.sum(np.log(np.diagonal(factor, axis1=1, axis2=2)), axis=1)
        transverse[:, index] = (log_gamma
                                - .5 * (log_determinant + np.log(precision_along_ray))
                                - .5 * quadratic)
        front_argument = beta / np.sqrt(precision_along_ray)
        log_front[:, index] = _log_normal_cdf(front_argument)

    log_mass_product = math.log(QUERY_MASS * KEY_MASS)
    lograw = log_mass_product + transverse + log_front
    if not np.isfinite(transverse).all() or not np.isfinite(lograw).all():
        raise ValueError('nonfinite half-ray score; no salvage by the floor')
    logfloored = np.logaddexp(lograw, log_mass_product + math.log(FLOOR_DENSITY))
    logits = logfloored + bias
    if not np.isfinite(logits).all():
        raise ValueError('nonfinite content-weighted logits')
    logp = np.empty_like(logits)
    for index, scores in enumerate(logits):
        largest = float(np.max(scores))
        shifted_total = math.fsum(math.exp(float(value) - largest) for value in scores)
        logp[index] = scores - (largest + math.log(shifted_total))
    with np.errstate(under='ignore'):
        probability = np.exp(logp)
    result = dict(lograw=lograw, logfloored=logfloored, logp=logp, p=probability,
                  winner=np.argmax(probability, axis=1).astype(np.int32),
                  T=transverse, log_front=log_front)
    if not all(np.isfinite(value).all() for value in result.values()):
        raise ValueError('nonfinite normalization arithmetic; no repair')
    if any(abs(math.fsum(map(float, row)) - 1.) > 2e-12 for row in probability):
        raise ValueError('independent probability normalization exceeds 2e-12')
    return result
