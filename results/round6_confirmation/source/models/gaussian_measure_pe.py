"""Known Gaussian density-overlap position bias with explicit key measure.

This is an additive relative position operator, not a new probability kernel.
The log mass has coefficient one, including when density_strength differs from
one.  A single bandwidth variance is added to the *pair* covariance.
"""
import math

import torch


def _isotropic(cov):
    eye = torch.eye(2, dtype=cov.dtype, device=cov.device)
    return cov.diagonal(dim1=-2, dim2=-1).mean(-1)[..., None, None] * eye


def _fixed_covariance(variance, mean):
    value = torch.as_tensor(variance, dtype=mean.dtype, device=mean.device)
    # A batch-only variance [B] applies to all of that batch's tokens.
    if value.ndim == mean.ndim - 2 and value.ndim:
        value = value.unsqueeze(-1)
    value = torch.broadcast_to(value, mean.shape[:-1])
    return value[..., None, None] * torch.eye(2, dtype=mean.dtype, device=mean.device)


def _logdet2(cov):
    """Scaled symmetric 2x2 determinant; requires a positive determinant."""
    a, c = cov[..., 0, 0], cov[..., 1, 1]
    b = (cov[..., 0, 1] + cov[..., 1, 0]) * .5
    scale = torch.maximum(a.abs(), torch.maximum(b.abs(), c.abs()))
    scale = scale.clamp_min(torch.finfo(cov.dtype).tiny)
    aa, bb, cc = a / scale, b / scale, c / scale
    determinant = aa * cc - bb.square()
    return 2 * scale.log() + determinant.log(), (aa, bb, cc, scale, determinant)


def gaussian_measure_log_bias(qmean, qcov, kmean, kcov, *, key_mass=None,
                              method='full', bandwidth=.25,
                              fixed_variance=1/12, density_strength=1.):
    """Return ``log(A_k) + strength * log N(qmean-kmean; 0, S)``.

    Means have shape ``[..., tokens, 2]`` and covariances
    ``[..., tokens, 2, 2]``; output is ``[..., queries, keys]``.
    ``S=Cq+Ck+bandwidth**2 I``. Controls ``point``, ``fixed``,
    ``isotropic`` and ``full`` change only C; means and key mass are shared.
    Fixed variance accepts a scalar or ``(query_variance, key_variance)``;
    each may be scalar, batch-only or per-token. If no mass is supplied,
    ``12 sqrt(det(original Ck))`` is an affine uniform-cell area convention,
    not a general area estimator. Supply positive mass for degenerate Ck.

    Tensor bandwidth/strength retain gradients. Inputs must be finite,
    covariances symmetric positive semidefinite and bandwidth positive;
    positive mass is required. Float16/bfloat16 inputs compute in float32;
    float64 inputs permit an independent high-precision reference. Extremely
    ill-conditioned covariance beyond the work dtype's precision is outside
    the numerical domain (no hidden covariance clipping is applied).
    """
    if method not in ('point', 'fixed', 'isotropic', 'full'):
        raise ValueError('unknown Gaussian position control: ' + str(method))
    if qmean.shape[-1] != 2 or kmean.shape[-1] != 2:
        raise ValueError('means must be two dimensional')
    if qcov.shape != qmean.shape[:-1] + (2, 2) or kcov.shape != kmean.shape[:-1] + (2, 2):
        raise ValueError('mean/covariance shape mismatch')
    dtype = torch.float64 if any(x.dtype == torch.float64 for x in (qmean, qcov, kmean, kcov)) else torch.float32
    output_dtype = qmean.dtype
    qmean, qcov, kmean, kcov = (x.to(dtype) for x in (qmean, qcov, kmean, kcov))
    width = torch.as_tensor(bandwidth, dtype=dtype, device=qmean.device)
    strength = torch.as_tensor(density_strength, dtype=dtype, device=qmean.device)
    if width.numel() != 1 or strength.numel() != 1:
        raise ValueError('bandwidth and density_strength must be scalar')
    # Validate Python scalars without a device synchronization in training.
    if isinstance(bandwidth, (int, float)) and (not math.isfinite(bandwidth) or bandwidth <= 0):
        raise ValueError('bandwidth must be finite and positive')
    if key_mass is None:
        logdet, _ = _logdet2(kcov)
        log_mass = math.log(12.) + .5 * logdet
    else:
        mass = torch.as_tensor(key_mass, dtype=dtype, device=qmean.device)
        log_mass = torch.broadcast_to(mass, kmean.shape[:-1]).log()
    if method == 'point':
        qcov, kcov = torch.zeros_like(qcov), torch.zeros_like(kcov)
    elif method == 'isotropic':
        qcov, kcov = _isotropic(qcov), _isotropic(kcov)
    elif method == 'fixed':
        qvar, kvar = fixed_variance if isinstance(fixed_variance, (tuple, list)) else (fixed_variance, fixed_variance)
        qcov, kcov = _fixed_covariance(qvar, qmean), _fixed_covariance(kvar, kmean)
    eye = torch.eye(2, dtype=dtype, device=qmean.device)
    pair_cov = qcov.unsqueeze(-3) + kcov.unsqueeze(-4) + width.square() * eye
    logdet, (a, b, c, scale, determinant) = _logdet2(pair_cov)
    delta = qmean.unsqueeze(-2) - kmean.unsqueeze(-3)
    dx, dy = delta.unbind(-1)
    mahalanobis = (c * dx.square() - 2 * b * dx * dy + a * dy.square()) / (scale * determinant)
    log_density = -.5 * (mahalanobis + logdet) - math.log(2 * math.pi)
    return (log_mass.unsqueeze(-2) + strength * log_density).to(output_dtype)
