"""Gaussian mass of actual rectangular cells, with positive quadrature.

This is a known Gaussian integration operator, used as a relative positional
bias. It does not give a finite exact Q/K feature map. Unlike replacing a cell
indicator by an area-scaled Gaussian, integration of the actual cell is
additive under key subdivision (up to the stated quadrature error).

Forward inputs are means/covariances and native cell bounds in one shared
coordinate system. No labels, depth visibility, content or values are used.
"""
from __future__ import annotations

from functools import lru_cache
import math

import numpy as np
import torch


@lru_cache(maxsize=8)
def _legendre(order: int):
    return np.polynomial.legendre.leggauss(order)


def _normal_interval(lower, upper):
    """Positive normal interval mass without subtracting near-one CDFs."""
    root2 = math.sqrt(2.)
    positive = .5 * (torch.erfc(lower / root2) - torch.erfc(upper / root2))
    negative = .5 * (torch.erfc(-upper / root2) - torch.erfc(-lower / root2))
    crossing = .5 * (torch.erf(upper / root2) - torch.erf(lower / root2))
    return torch.where(lower >= 0, positive,
                       torch.where(upper <= 0, negative, crossing)).clamp_min(0.)


def _mass_block(mean, covariance, bounds, nodes, weights, tail_sigma):
    sx = covariance[:, 0, 0].sqrt()[:, None]
    sy = covariance[:, 1, 1].sqrt()[:, None]
    rho = covariance[:, 0, 1, None] / (sx * sy)
    residual = (1. - rho.square()).sqrt()
    xlo = ((bounds[None, :, 0] - mean[:, None, 0]) / sx).clamp(-tail_sigma, tail_sigma)
    xhi = ((bounds[None, :, 2] - mean[:, None, 0]) / sx).clamp(-tail_sigma, tail_sigma)
    ylo = (bounds[None, :, 1] - mean[:, None, 1]) / sy
    yhi = (bounds[None, :, 3] - mean[:, None, 1]) / sy

    # Break at ordinary normal-density scales and the two conditional CDF
    # transitions. This remains accurate for correlations close to +/-1,
    # where one unsplit Gauss-Legendre rule can miss a narrow transition.
    safe_rho = torch.where(rho.abs() > 1.e-10, rho, torch.ones_like(rho))
    offsets = mean.new_tensor([-8., 0., 8.])
    transitions = torch.cat([
        (ylo[..., None] + residual[..., None] * offsets) / safe_rho[..., None],
        (yhi[..., None] + residual[..., None] * offsets) / safe_rho[..., None],
    ], dim=-1)
    transitions = torch.where((rho.abs() > 1.e-10)[..., None], transitions, xlo[..., None])
    normal_breaks = mean.new_tensor([-8., -4., 0., 4., 8.]).expand(*xlo.shape, -1)
    interior = torch.cat([transitions, normal_breaks], dim=-1)
    interior = torch.minimum(torch.maximum(interior, xlo[..., None]), xhi[..., None])
    edges = torch.cat([xlo[..., None], interior, xhi[..., None]], dim=-1).sort(-1).values
    mid = (edges[..., :-1] + edges[..., 1:]) * .5
    half = (edges[..., 1:] - edges[..., :-1]) * .5
    x = mid[..., None] + half[..., None] * nodes
    lo = (ylo[..., None, None] - rho[..., None, None] * x) / residual[..., None, None]
    hi = (yhi[..., None, None] - rho[..., None, None] * x) / residual[..., None, None]
    conditional = _normal_interval(lo, hi)
    density = torch.exp(-.5 * x.square()) / math.sqrt(2. * math.pi)
    return (half * (conditional * density * weights).sum(-1)).sum(-1)


def gaussian_cell_mass(qmean, qcov, cell_bounds, *, bandwidth=.25,
                       quadrature_order=24, query_chunk_size=16,
                       key_chunk_size=128, tail_sigma=12.):
    """Integrate ``N(qmean, qcov + bandwidth**2 I)`` over each native cell.

    qmean: [..., queries, 2]; qcov: [..., queries, 2, 2].
    cell_bounds: [keys, 4], columns xmin, ymin, xmax, ymax, in the SAME
    units as means/covariances. For pixel indices [a,b), physical support is
    [a-.5,b-.5), divided by the reference stride if means use stride units.
    Output: [..., queries, keys], raw nonnegative masses, always float64.

    Uses deterministic, differentiable positive conditional integration with
    12 interval segments and ``quadrature_order`` nodes per segment (many
    segments have zero length). Standardized integration is truncated at
    +/-12 by default: omitted absolute mass is at most 2*Phi(-12) < 4e-33.
    PSD input covariance plus strictly positive bandwidth is required; the
    resulting correlation must be strictly less than one in float64.
    There is no probability floor, FOV normalization, area multiplier or
    strength exponent here. Mass already contains physical cell area.

    Quadrature error is separate from the Gaussian support approximation.
    The default was audited against independent SciPy integration; callers
    may increase order to assess numerical error without changing the model.
    No guarantee is made for arbitrarily ill-conditioned, unrepresentable
    covariance or floating-point intervals narrower than machine resolution.
    """
    if qmean.ndim < 2 or qmean.shape[-1] != 2 or qcov.shape != qmean.shape[:-1] + (2, 2):
        raise ValueError('invalid mean/covariance shapes')
    if cell_bounds.ndim != 2 or cell_bounds.shape[-1] != 4 or not len(cell_bounds):
        raise ValueError('cell_bounds must be nonempty [keys,4]')
    if quadrature_order < 8 or query_chunk_size < 1 or key_chunk_size < 1:
        raise ValueError('invalid quadrature or chunk size')
    if not math.isfinite(float(tail_sigma)) or tail_sigma < 8:
        raise ValueError('tail_sigma must be finite and at least eight')
    qm, qc, boxes = (x.to(device=qmean.device, dtype=torch.float64) for x in (qmean, qcov, cell_bounds))
    width = torch.as_tensor(bandwidth, device=qm.device, dtype=qm.dtype)
    if width.numel() != 1 or not bool(torch.isfinite(width).all()) or not bool((width > 0).all()):
        raise ValueError('bandwidth must be one finite positive scalar')
    if not all(bool(torch.isfinite(x).all()) for x in (qm, qc, boxes)):
        raise ValueError('all inputs must be finite')
    if not bool((boxes[:, 2:] > boxes[:, :2]).all()):
        raise ValueError('cell widths and heights must be positive')
    scale = qc.abs().amax().clamp_min(1.)
    if not bool(((qc - qc.transpose(-1, -2)).abs() <= 1.e-12 * scale).all()):
        raise ValueError('covariance must be symmetric')
    if not bool((torch.linalg.eigvalsh(qc) >= -1.e-12 * scale).all()):
        raise ValueError('covariance must be positive semidefinite')
    covariance = qc + width.square() * torch.eye(2, device=qm.device, dtype=qm.dtype)
    variance_product = covariance[..., 0, 0] * covariance[..., 1, 1]
    if not bool((variance_product > covariance[..., 0, 1].square()).all()):
        raise ValueError('regularized covariance is numerically singular')
    shape = qm.shape[:-1]
    qm = qm.reshape(-1, 2)
    covariance = covariance.reshape(-1, 2, 2)
    n, w = _legendre(int(quadrature_order))
    nodes = torch.as_tensor(n.copy(), device=qm.device, dtype=qm.dtype)
    weights = torch.as_tensor(w.copy(), device=qm.device, dtype=qm.dtype)
    rows = []
    for qi in range(0, len(qm), query_chunk_size):
        columns = []
        for ki in range(0, len(boxes), key_chunk_size):
            columns.append(_mass_block(qm[qi:qi + query_chunk_size],
                covariance[qi:qi + query_chunk_size], boxes[ki:ki + key_chunk_size],
                nodes, weights, tail_sigma))
        rows.append(torch.cat(columns, dim=-1))
    return torch.cat(rows, dim=0).reshape(*shape, len(boxes))


def normalized_cell_probabilities(masses, *, floor=0.):
    """Condition Gaussian cell mass on the supplied FOV partition.

    A positive floor is optional and explicit. It perturbs exact subdivision
    consistency and should only be applied under a predeclared numerical
    policy shared by controls. Empty/underflow rows are rejected.
    """
    if not math.isfinite(floor) or floor < 0:
        raise ValueError('floor must be finite and nonnegative')
    if not bool(torch.isfinite(masses).all()) or not bool((masses >= 0).all()):
        raise ValueError('masses must be finite and nonnegative')
    mass = masses.clamp_min(floor)
    denominator = mass.sum(-1, keepdim=True)
    if not bool((denominator > 0).all()):
        raise ValueError('zero FOV mass; caller must declare a fallback')
    return mass / denominator
