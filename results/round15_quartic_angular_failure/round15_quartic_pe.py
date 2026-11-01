"""Covariance-matched 2D radial quartic density and positive rectangle mass.

This is a fixed generalized-Gaussian moment closure, not Gaussian convolution.
Adding h^2 I specifies the *total covariance* of the quartic distribution;
zero input covariance therefore gives a quartic, not Gaussian, point blur.
No task labels, calibration, floors, FOV normalization or content enter here.
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import torch


@lru_cache(None)
def _legendre(order):
    return np.polynomial.legendre.leggauss(order)


def _inputs(qmean, qcov, cell_bounds, bandwidth):
    if qmean.ndim < 2 or qmean.shape[-1] != 2 or qcov.shape != qmean.shape[:-1] + (2, 2):
        raise ValueError('invalid mean/covariance shapes')
    if not qmean.numel() or cell_bounds.ndim != 2 or cell_bounds.shape[-1] != 4 or not len(cell_bounds):
        raise ValueError('queries and rectangles must be nonempty')
    mean, cov, boxes = (v.to(device=qmean.device, dtype=torch.float64)
                        for v in (qmean, qcov, cell_bounds))
    h = torch.as_tensor(bandwidth, device=mean.device, dtype=mean.dtype)
    if h.numel() != 1 or not bool(torch.isfinite(h).all()) or bool((h < 0).any()):
        raise ValueError('bandwidth must be finite and nonnegative')
    if not all(bool(torch.isfinite(v).all()) for v in (mean, cov, boxes)):
        raise ValueError('inputs must be finite')
    if not bool((boxes[:, 2:] > boxes[:, :2]).all()):
        raise ValueError('rectangle sides must be positive')
    scale = cov.abs().amax(dim=(-2, -1)).clamp_min(1.)
    if not bool(((cov-cov.transpose(-1, -2)).abs() <= 1.e-12*scale[..., None, None]).all()):
        raise ValueError('covariance must be symmetric')
    if not bool((torch.linalg.eigvalsh(cov) >= -1.e-12*scale[..., None]).all()):
        raise ValueError('covariance must be positive semidefinite')
    effective = cov + h.square()*torch.eye(2, device=mean.device, dtype=mean.dtype)
    factor, info = torch.linalg.cholesky_ex(effective)
    if bool((info != 0).any()):
        raise ValueError('effective covariance must be positive definite')
    return mean, effective, boxes, factor


def quartic_log_density(points, mean, covariance):
    """Broadcasted log density for an already positive-definite total covariance.

    For arbitrary invertible A, this density transforms with 1/|det A| when
    mean and total covariance transform by A. An isotropic h fixed in new
    coordinates is not affine covariant and must not be added a second time.
    """
    points, mean, covariance = (v.to(dtype=torch.float64, device=mean.device)
                                for v in (points, mean, covariance))
    delta = points-mean
    solved = torch.linalg.solve(covariance, delta.unsqueeze(-1)).squeeze(-1)
    q = (delta*solved).sum(-1)
    _, logdet = torch.linalg.slogdet(covariance)
    return -q.square()/(4.*math.pi)-2.*math.log(math.pi)-.5*logdet


def _radial_interval(lo, hi):
    # At r=10 the survival is below the smallest float64 subnormal. Clipping
    # first also avoids overflow/NaN gradients on rays nearly parallel to an
    # edge. This is a machine-underflow boundary, not a tail approximation at
    # any representable probability of this radial family.
    a = lo.clamp(0., 10.).square()/(2.*math.sqrt(math.pi))
    b = hi.clamp(0., 10.).square()/(2.*math.sqrt(math.pi))
    return torch.where(a < .5, torch.erf(b)-torch.erf(a),
                       torch.erfc(a)-torch.erfc(b)).clamp_min(0.)


def _angular_mass(mean, factor, boxes, nodes, weights):
    # All arrays retain query/key axes. The four vertex angles delimit the
    # visible angular window and ray edge switches; four parallel angles
    # additionally delimit every denominator sign change. Repeated angles
    # merely produce zero-width segments. Sorting is differentiable almost
    # everywhere, and no decisions are based on labels or probability size.
    low = boxes[None, :, :2]-mean[:, None, :]
    high = boxes[None, :, 2:]-mean[:, None, :]
    vx = torch.stack((low[..., 0], high[..., 0], high[..., 0], low[..., 0]), -1)
    vy = torch.stack((low[..., 1], low[..., 1], high[..., 1], high[..., 1]), -1)
    l00 = factor[:, 0, 0]
    l10 = factor[:, 1, 0]
    l11 = factor[:, 1, 1]
    zx = vx/l00[:, None, None]
    zy = (vy-l10[:, None, None]*zx)/l11[:, None, None]
    at_origin = (zx == 0.) & (zy == 0.)
    zx = torch.where(at_origin, torch.ones_like(zx), zx)
    zy = torch.where(at_origin, torch.zeros_like(zy), zy)
    vertex = torch.remainder(torch.atan2(zy, zx), 2.*math.pi)
    py = torch.remainder(torch.atan2(-l10, l11), 2.*math.pi)
    parallel = torch.stack((torch.full_like(py, math.pi/2.),
                            torch.full_like(py, 3.*math.pi/2.), py,
                            torch.remainder(py+math.pi, 2.*math.pi)), -1)
    parallel = parallel[:, None, :].expand(-1, len(boxes), -1)
    endpoints = torch.zeros_like(vertex[..., :2])
    endpoints[..., 1] = 2.*math.pi
    angles = torch.cat((vertex, parallel, endpoints), -1).sort(-1).values
    half = (angles[..., 1:]-angles[..., :-1])*.5
    middle = (angles[..., 1:]+angles[..., :-1])*.5
    theta = middle[..., None]+half[..., None]*nodes
    co, si = torch.cos(theta), torch.sin(theta)
    dx = l00[:, None, None, None]*co
    dy = l10[:, None, None, None]*co+l11[:, None, None, None]*si
    # A nonzero-width GL interval has no exactly parallel interior node.
    # Degenerate intervals can, so replace only numerical zero denominators
    # by signed tiny values. Their angular contribution is identically zero.
    tiny = torch.finfo(mean.dtype).tiny
    dx = torch.where(dx.abs() < tiny, torch.full_like(dx, tiny), dx)
    dy = torch.where(dy.abs() < tiny, torch.full_like(dy, tiny), dy)
    rx0, rx1 = low[..., 0, None, None]/dx, high[..., 0, None, None]/dx
    ry0, ry1 = low[..., 1, None, None]/dy, high[..., 1, None, None]/dy
    lower = torch.maximum(torch.minimum(rx0, rx1), torch.minimum(ry0, ry1)).clamp_min(0.)
    upper = torch.minimum(torch.maximum(rx0, rx1), torch.maximum(ry0, ry1))
    value = _radial_interval(lower, upper)
    value = torch.where(upper > lower, value, torch.zeros_like(value))
    return ((value*weights).sum(-1)*half).sum(-1)/(2.*math.pi)


def quartic_cell_mass(qmean, qcov, cell_bounds, *, bandwidth=.5,
                      quadrature_order=32, query_chunk_size=16,
                      key_chunk_size=64):
    """Positive raw float64 rectangle probabilities, shape [..., Q, K].

    Bounds are [xmin,ymin,xmax,ymax], in the same units as means. The angular
    method has nine (possibly zero-width) subintervals, each with the fixed GL
    order. It has no cell-size restriction, but accuracy at arbitrary extreme
    condition numbers is not guaranteed by this API. Numeric preflight must
    state its finite tested domain and compare FOV-normalized probabilities.
    """
    if quadrature_order < 4 or query_chunk_size < 1 or key_chunk_size < 1:
        raise ValueError('invalid quadrature or chunk size')
    mean, _, boxes, factor = _inputs(qmean, qcov, cell_bounds, bandwidth)
    shape = mean.shape[:-1]
    mean, factor = mean.reshape(-1, 2), factor.reshape(-1, 2, 2)
    n, w = _legendre(int(quadrature_order))
    nodes, weights = (torch.as_tensor(v.copy(), device=mean.device, dtype=mean.dtype) for v in (n, w))
    rows = []
    for qi in range(0, len(mean), query_chunk_size):
        columns = []
        for ki in range(0, len(boxes), key_chunk_size):
            columns.append(_angular_mass(mean[qi:qi+query_chunk_size],
                factor[qi:qi+query_chunk_size], boxes[ki:ki+key_chunk_size], nodes, weights))
        rows.append(torch.cat(columns, -1))
    return torch.cat(rows, 0).reshape(*shape, len(boxes))


def quartic_cell_mass_cartesian(qmean, qcov, cell_bounds, *, bandwidth=.5,
                                quadrature_order=32, query_chunk_size=16,
                                key_chunk_size=32):
    """Independent tensor-product GL formulation for numerical comparisons.

    Cost is order squared per query/key. A single box interval can miss narrow
    peaks; this is a comparison implementation, not an automatic fallback.
    """
    if quadrature_order < 4 or query_chunk_size < 1 or key_chunk_size < 1:
        raise ValueError('invalid quadrature or chunk size')
    mean, covariance, boxes, _ = _inputs(qmean, qcov, cell_bounds, bandwidth)
    shape = mean.shape[:-1]
    mean, covariance = mean.reshape(-1, 2), covariance.reshape(-1, 2, 2)
    n, w = _legendre(int(quadrature_order))
    nodes, weights = (torch.as_tensor(v.copy(), device=mean.device, dtype=mean.dtype) for v in (n, w))
    product = weights[:, None]*weights[None, :]
    rows = []
    for qi in range(0, len(mean), query_chunk_size):
        mu, cov = mean[qi:qi+query_chunk_size], covariance[qi:qi+query_chunk_size]
        inv = torch.linalg.inv(cov)
        norm = math.pi**2*torch.linalg.det(cov).sqrt()
        columns = []
        for ki in range(0, len(boxes), key_chunk_size):
            rect = boxes[ki:ki+key_chunk_size]
            mid, half = (rect[:, :2]+rect[:, 2:])*.5, (rect[:, 2:]-rect[:, :2])*.5
            x = mid[:, 0, None]+half[:, 0, None]*nodes
            y = mid[:, 1, None]+half[:, 1, None]*nodes
            dx = x[None, :, :, None]-mu[:, None, 0, None, None]
            dy = y[None, :, None, :]-mu[:, None, 1, None, None]
            q = inv[:, None, 0, 0, None, None]*dx.square()+2*inv[:, None, 0, 1, None, None]*dx*dy+inv[:, None, 1, 1, None, None]*dy.square()
            mass = (torch.exp(-q.square()/(4.*math.pi))*product).sum((-1,-2))
            columns.append(mass*half.prod(-1)[None]/norm[:, None])
        rows.append(torch.cat(columns, -1))
    return torch.cat(rows, 0).reshape(*shape, len(boxes))
