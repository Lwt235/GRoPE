"""Native-cell Gaussian integration specialized to the Round10 numeric domain.

This uses the same known rectangle integral as round10_cell_integral, with
one positive Gauss-Legendre rule per physical x interval. Native cell side
lengths <= 2 and Gaussian bandwidth >= .25 ensure the conditional standard
deviation is at least .25: no interval spans more than 8 such deviations.
This is a bounded-domain numerical implementation, not a new PE operator.
"""
from __future__ import annotations

import math

import torch

from models.round10_cell_integral import _legendre, _normal_interval


def gaussian_cell_mass_fast(qmean, qcov, cell_bounds, *, bandwidth=.25,
                            quadrature_order=32, query_chunk_size=32,
                            key_chunk_size=128):
    """Return raw float64 cell probabilities, without floor or normalization.

    Shapes follow gaussian_cell_mass. Means/bounds use stride-8 coordinate
    units; cell side lengths must be <= 2 and bandwidth must be >= .25.
    The covariance must be symmetric PSD. This bounded domain admits a simple
    physical-coordinate quadrature without truncating global normal tails.
    Underflow of truly tiny masses can still occur; downstream numerical
    floors/fallbacks must be predeclared and reported by the experiment.

    A no-grad diagonal-covariance case uses exact separable normal interval
    probabilities, so point/isotropic controls need not pay for 2D quadrature.
    With gradients enabled, quadrature preserves the derivative with respect
    to zero off-diagonal covariance as well.
    """
    if qmean.ndim < 2 or qmean.shape[-1] != 2 or qcov.shape != qmean.shape[:-1] + (2, 2):
        raise ValueError('invalid mean/covariance shapes')
    if qmean.numel() == 0 or cell_bounds.ndim != 2 or cell_bounds.shape[-1] != 4 or not len(cell_bounds):
        raise ValueError('queries and cell bounds must be nonempty')
    if quadrature_order < 8 or query_chunk_size < 1 or key_chunk_size < 1:
        raise ValueError('invalid quadrature or chunk size')
    qm, qc, boxes = (x.to(device=qmean.device, dtype=torch.float64) for x in (qmean, qcov, cell_bounds))
    width = torch.as_tensor(bandwidth, device=qm.device, dtype=qm.dtype)
    if width.numel() != 1 or not bool(torch.isfinite(width).all()) or not bool((width >= .25).all()):
        raise ValueError('fast bounded-domain integration requires bandwidth >= .25')
    if not all(bool(torch.isfinite(x).all()) for x in (qm, qc, boxes)):
        raise ValueError('all inputs must be finite')
    wh = boxes[:, 2:] - boxes[:, :2]
    if not bool(((wh > 0) & (wh <= 2. + 1.e-12)).all()):
        raise ValueError('fast bounded-domain integration requires cell sides in (0,2]')
    scale = qc.abs().amax(-1).amax(-1).clamp_min(1.)
    if not bool(((qc - qc.transpose(-1, -2)).abs() <= 1.e-12 * scale[..., None, None]).all()):
        raise ValueError('covariance must be symmetric')
    if not bool((torch.linalg.eigvalsh(qc) >= -1.e-12 * scale[..., None]).all()):
        raise ValueError('covariance must be positive semidefinite')
    covariance = qc + width.square() * torch.eye(2, device=qm.device, dtype=qm.dtype)
    condvar = covariance[..., 1, 1] - covariance[..., 0, 1].square() / covariance[..., 0, 0]
    if not bool((condvar > 0).all()):
        raise ValueError('regularized covariance is numerically singular')
    shape = qm.shape[:-1]
    qm, covariance = qm.reshape(-1, 2), covariance.reshape(-1, 2, 2)
    no_grad = not any(x.requires_grad for x in (qm, covariance, boxes, width))
    diagonal = no_grad and bool((covariance[:, 0, 1] == 0).all())
    n, w = _legendre(int(quadrature_order))
    nodes = torch.as_tensor(n.copy(), device=qm.device, dtype=qm.dtype)
    weights = torch.as_tensor(w.copy(), device=qm.device, dtype=qm.dtype)
    rows = []
    for qi in range(0, len(qm), query_chunk_size):
        mean = qm[qi:qi + query_chunk_size]
        cov = covariance[qi:qi + query_chunk_size]
        sx = cov[:, 0, 0].sqrt()[:, None]
        coefficient = cov[:, 0, 1, None] / cov[:, 0, 0, None]
        residual = (cov[:, 1, 1] - cov[:, 0, 1].square() / cov[:, 0, 0]).sqrt()[:, None]
        columns = []
        for ki in range(0, len(boxes), key_chunk_size):
            rect = boxes[ki:ki + key_chunk_size]
            if diagonal:
                xlo = (rect[None, :, 0] - mean[:, None, 0]) / sx
                xhi = (rect[None, :, 2] - mean[:, None, 0]) / sx
                ylo = (rect[None, :, 1] - mean[:, None, 1]) / residual
                yhi = (rect[None, :, 3] - mean[:, None, 1]) / residual
                mass = _normal_interval(xlo, xhi) * _normal_interval(ylo, yhi)
            else:
                midpoint = (rect[:, 0] + rect[:, 2]) * .5
                half = (rect[:, 2] - rect[:, 0]) * .5
                x = midpoint[:, None] + half[:, None] * nodes
                centered = x[None] - mean[:, None, 0, None]
                standard = centered / sx[..., None]
                conditional_mean = mean[:, None, 1, None] + coefficient[..., None] * centered
                lo = (rect[None, :, 1, None] - conditional_mean) / residual[..., None]
                hi = (rect[None, :, 3, None] - conditional_mean) / residual[..., None]
                conditional = _normal_interval(lo, hi)
                density = torch.exp(-.5 * standard.square()) / (math.sqrt(2.*math.pi) * sx[..., None])
                mass = half[None] * (conditional * density * weights).sum(-1)
            columns.append(mass)
        rows.append(torch.cat(columns, dim=-1))
    return torch.cat(rows, dim=0).reshape(*shape, len(boxes))
