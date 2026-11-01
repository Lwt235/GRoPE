"""Independent special-law references for the bounded R22 synthetic preflight.

No production integrator is imported. These routines implement the frozen
canonical distributions only, not arbitrary singular/full-rank Gaussians.
Two orders give convergence evidence, not a rigorous integration-error bound.
"""
from __future__ import annotations

import time
import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy.polynomial.hermite_e import hermegauss
from scipy.special import log_ndtr, ndtr


ORDERS = (64, 96)
KINDS = ('point', 'constant_depth', 'radial', 'offset_rank1', 'origin_plane', 'origin_isotropic',
         'origin_correlated', 'tilted_plane')


def normal_interval(lower, upper):
    """Stable standard-normal interval probability, including either tail."""
    lower, upper = np.broadcast_arrays(np.asarray(lower, np.float64), np.asarray(upper, np.float64))
    if np.any(lower > upper) or np.isnan(lower).any() or np.isnan(upper).any():
        raise ValueError('ordered non-NaN Gaussian interval required')
    with np.errstate(invalid='ignore'):
        a, b = log_ndtr(lower), log_ndtr(upper)
        left = np.exp(b) * -np.expm1(a - b)
        a, b = log_ndtr(-upper), log_ndtr(-lower)
        right = np.exp(b) * -np.expm1(a - b)
    result = np.where(lower >= 0., right, left)
    # Endpoints both infinite on the same side are exact zero intervals.
    result = np.where(lower == upper, 0., result)
    if np.any(result < 0) or not np.isfinite(result).all():
        raise ValueError('nonfinite independent Gaussian interval')
    return result


def point_rectangle(point, bounds, h):
    point, bounds = np.asarray(point, np.float64), np.asarray(bounds, np.float64)
    return normal_interval((bounds[:, 0] - point[0]) / h, (bounds[:, 2] - point[0]) / h) * normal_interval(
        (bounds[:, 1] - point[1]) / h, (bounds[:, 3] - point[1]) / h)


def front_mass(mu, factor):
    variance = float(np.dot(factor[2], factor[2]))
    return float(ndtr(mu[2] / np.sqrt(variance))) if variance else float(mu[2] > 0.)


def _equal(actual, expected, name):
    if not np.allclose(actual, expected, atol=1e-13, rtol=1e-13):
        raise ValueError('unsupported canonical ' + name)


def _inputs(mu, factor, bounds, H, offset, policy, kind, order, reference_policy):
    if any(isinstance(x, np.ndarray) and x.dtype != np.float64 for x in (mu, factor, bounds, H, offset)):
        raise ValueError('float64 input arrays required')
    mu, factor, bounds, H, offset = [np.asarray(x, np.float64) for x in (mu, factor, bounds, H, offset)]
    if (mu.shape != (3,) or factor.ndim != 2 or factor.shape[0] != 3 or factor.shape[1] > 3
            or bounds.ndim != 2 or bounds.shape[1] != 4 or not len(bounds) or H.shape != (2, 2)
            or offset.shape != (2,) or not all(np.isfinite(a).all() for a in (mu, factor, bounds, H, offset))
            or np.any(bounds[:, 2:] <= bounds[:, :2])):
        raise ValueError('finite canonical Gaussian/cells required')
    for i, box in enumerate(bounds):
        overlap = np.minimum(box[2:], bounds[i + 1:, 2:]) - np.maximum(box[:2], bounds[i + 1:, :2])
        if np.any(np.all(overlap > 0., axis=1)):
            raise ValueError('disjoint native cells required')
    rectangle = np.r_[bounds[:, :2].min(0), bounds[:, 2:].max(0)]
    if not np.isclose(np.sum(np.prod(bounds[:, 2:] - bounds[:, :2], axis=1)), np.prod(rectangle[2:] - rectangle[:2]), rtol=1e-13, atol=0.):
        raise ValueError('native cells must partition their rectangular FOV')
    if kind not in KINDS or order not in ORDERS or reference_policy['orders'] != list(ORDERS):
        raise ValueError('unfrozen reference kind/order')
    scale = float(H[0, 0]); _equal(H, np.eye(2) * scale, 'screen transform')
    if scale <= 0 or not np.isfinite(policy['h']) or policy['h'] <= 0:
        raise ValueError('positive isotropic screen units and smoothing required')
    bounds = (bounds - np.r_[offset, offset]) / scale
    return mu, factor, bounds, float(policy['h']) / scale


def _gaussian_rectangle(mean, covariance, bounds, order):
    """Native rectangle integral after blur, by independent 1D conditioning."""
    sx = np.sqrt(covariance[0, 0]); slope = covariance[1, 0] / covariance[0, 0]
    sy = np.sqrt(covariance[1, 1] - covariance[1, 0] ** 2 / covariance[0, 0])
    nodes, weights = leggauss(order)
    low, high = bounds[:, 0], bounds[:, 2]
    x = (low[:, None] + high[:, None]) / 2 + (high - low)[:, None] / 2 * nodes
    density = np.exp(-.5 * ((x - mean[0]) / sx) ** 2) / (np.sqrt(2 * np.pi) * sx)
    cy = mean[1] + slope * (x - mean[0])
    interval = normal_interval((bounds[:, 1, None] - cy) / sy, (bounds[:, 3, None] - cy) / sy)
    return np.sum(density * interval * weights, axis=1) * (high - low) / 2


def _solid_angle_rectangle(bounds):
    """Unconditional Z>0 mass for a zero-mean isotropic 3D Gaussian."""
    def corner(x, y):
        return np.arctan2(x * y, np.sqrt(1 + x * x + y * y)) / (4 * np.pi)
    return corner(bounds[..., 2], bounds[..., 3]) - corner(bounds[..., 0], bounds[..., 3]) - corner(
        bounds[..., 2], bounds[..., 1]) + corner(bounds[..., 0], bounds[..., 1])


def _latent_solid_angle_rectangle(bounds, factor):
    """Zero-mean fullrank ray-cone mass via uniform latent-sphere area."""
    x0, y0, x1, y1 = np.moveaxis(bounds, -1, 0)
    ones = np.ones_like(x0)
    corners = np.stack((np.stack((x0, y0, ones), -1), np.stack((x1, y0, ones), -1),
                        np.stack((x1, y1, ones), -1), np.stack((x0, y1, ones), -1)), axis=-2)
    latent = np.linalg.solve(factor, corners.reshape(-1, 3).T).T.reshape(corners.shape)
    latent = latent / np.linalg.norm(latent, axis=-1, keepdims=True)
    def triangle(a, b, c):
        numerator = abs(np.sum(a * np.cross(b, c), axis=-1))
        denominator = 1 + np.sum(a * b, axis=-1) + np.sum(b * c, axis=-1) + np.sum(c * a, axis=-1)
        return 2 * np.arctan2(numerator, denominator)
    return (triangle(latent[..., 0, :], latent[..., 1, :], latent[..., 2, :])
            + triangle(latent[..., 0, :], latent[..., 2, :], latent[..., 3, :])) / (4 * np.pi)


def _panels(bounds, h, extent, fraction, pole, support_sign, maximum):
    low, high = float(bounds[:, 0].min() - extent * h), float(bounds[:, 2].max() + extent * h)
    if support_sign > 0: low = max(low, pole)
    if support_sign < 0: high = min(high, pole)
    if high <= low:
        return np.empty((0, 2))
    count = int(np.ceil((high - low) / (h * fraction)))
    if count > maximum:
        raise ValueError('independent fixed panel budget exceeded')
    edges = np.linspace(low, high, count + 1)
    return np.column_stack((edges[:-1], edges[1:]))


def integrate(mu, factor, bounds, H, offset, policy, kind, order, reference_policy):
    """Return raw positive-depth mass; the caller owns floor/normalization.

    ``analytic_tail.raw_fov_upper`` bounds only the omitted blur tail in the
    projected-coordinate reference. It excludes quadrature/roundoff error.
    """
    started = time.perf_counter()
    mu, factor, bounds, h = _inputs(mu, factor, bounds, H, offset, policy, kind, order, reference_policy)
    covariance = factor @ factor.T; positive = front_mass(mu, factor)
    n, panels, evaluations, tail = len(bounds), 0, 0, 0.
    if kind == 'point':
        if factor.shape[1] != 0:
            raise ValueError('finite-rank point-limit is not an exact point reference')
        raw = positive * point_rectangle(mu[:2] / mu[2], bounds, h) if positive else np.zeros(n)
        method = 'exact-positive-depth-point-and-Gaussian-blur'; evaluations = n
    elif kind == 'radial':
        if factor.shape[1] != 1 or factor[2, 0] == 0.:
            raise ValueError('one radial depth factor required')
        point = factor[:2, 0] / factor[2, 0]
        _equal(mu[:2], point * mu[2], 'radial mean')
        raw = positive * point_rectangle(point, bounds, h)
        method = 'analytic-front-mass-times-radial-point-blur'; evaluations = n
    elif kind == 'constant_depth':
        if np.any(factor[2] != 0.) or mu[2] <= 0.:
            raise ValueError('positive constant-depth plane required')
        raw = _gaussian_rectangle(mu[:2] / mu[2], covariance[:2, :2] / mu[2] ** 2 + h ** 2 * np.eye(2), bounds, order)
        method = 'projected-Gaussian-independent-1D-rectangle'; evaluations = n * order
    elif kind == 'origin_plane':
        _equal(mu, np.zeros(3), 'origin plane mean'); _equal(covariance, np.diag([1., 0., 1.]), 'origin plane covariance')
        nodes, weights = hermegauss(order); noise = h * nodes
        xmass = (np.arctan(bounds[:, 2, None] - noise) - np.arctan(bounds[:, 0, None] - noise)) / (2 * np.pi)
        raw = (xmass @ weights / np.sqrt(2 * np.pi)) * normal_interval(bounds[:, 1] / h, bounds[:, 3] / h)
        method = 'front-sub-Cauchy-CDF-and-1D-blur-GH'; evaluations = n * order
    elif kind in ('origin_isotropic', 'origin_correlated'):
        _equal(mu, np.zeros(3), 'isotropic origin mean')
        if kind == 'origin_isotropic':
            if covariance[0, 0] <= 0.:
                raise ValueError('positive isotropic covariance required')
            _equal(covariance / covariance[0, 0], np.eye(3), 'isotropic origin covariance')
        elif factor.shape != (3, 3) or np.linalg.det(factor) == 0.:
            raise ValueError('invertible zero-mean fullrank factor required')
        nodes, weights = hermegauss(order)
        noise = h * np.stack(np.meshgrid(nodes, nodes, indexing='ij'), axis=-1).reshape(-1, 2)
        weights2 = np.outer(weights, weights).ravel() / (2 * np.pi)
        shifted = bounds[:, None, :] - np.concatenate((noise, noise), axis=1)[None]
        angular = _solid_angle_rectangle(shifted) if kind == 'origin_isotropic' else _latent_solid_angle_rectangle(shifted, factor)
        raw = angular @ weights2
        method = ('ray-solid-angle-rectangle-and-2D-blur-GH' if kind == 'origin_isotropic'
                  else 'latent-sphere-two-triangle-solid-angle-and-2D-blur-GH')
        evaluations = n * order ** 2
    else:
        if kind == 'offset_rank1':
            if factor.shape[1] != 1 or factor[2, 0] == 0. or mu[1] != 0. or factor[1, 0] != 0.:
                raise ValueError('canonical offset rank-one support required')
            beta = factor[0, 0] / factor[2, 0]; alpha = mu[0] - beta * mu[2]
            if alpha == 0.:
                raise ValueError('non-radial rank-one offset required')
            pole, sign = beta, np.sign(alpha)
        else:
            _equal(mu, [0., 0., 1.], 'tilted mean')
            _equal(covariance, [[1., 0., .5], [0., .25, 0.], [.5, 0., .25]], 'tilted covariance')
            pole, sign = 2., -1
        boxes = _panels(bounds, h, reference_policy['blur_extent'], reference_policy['panel_width_h_fraction'],
                        pole, sign, reference_policy['max_panels'])
        panels = len(boxes); nodes, weights = leggauss(order)
        u = ((boxes[:, :1] + boxes[:, 1:]) / 2 + (boxes[:, 1:] - boxes[:, :1]) / 2 * nodes).ravel()
        quadrature = ((boxes[:, 1:] - boxes[:, :1]) / 2 * weights).ravel()
        wx = normal_interval((bounds[:, 0, None] - u) / h, (bounds[:, 2, None] - u) / h)
        if kind == 'offset_rank1':
            z = alpha / (u - beta); sigma = abs(factor[2, 0])
            density = np.exp(-.5 * ((z - mu[2]) / sigma) ** 2) / (np.sqrt(2 * np.pi) * sigma) * abs(alpha) / (u - beta) ** 2
            wy = normal_interval(bounds[:, 1, None] / h, bounds[:, 3, None] / h)
            method = 'offset-line-projected-u-inverse-Jacobian-GL'
        else:
            z = 1 / (1 - .5 * u); s = z * u
            density = np.exp(-.5 * s ** 2) / np.sqrt(2 * np.pi) / (1 - .5 * u) ** 2
            sy = np.sqrt((.5 / z) ** 2 + h ** 2)
            wy = normal_interval(bounds[:, 1, None] / sy, bounds[:, 3, None] / sy)
            method = 'tilted-plane-projected-u-Jacobian-GL-and-analytic-y'
        raw = np.sum(wx * wy * (density * quadrature)[None], axis=1)
        tail = float(2 * ndtr(-reference_policy['blur_extent']) * positive)
        evaluations = n * panels * order
    raw = np.asarray(raw, np.float64)
    if raw.shape != (n,) or not np.isfinite(raw).all() or np.any(raw < 0):
        raise ValueError('independent reference produced invalid raw mass')
    return dict(raw_mass=raw, front_mass=positive, method=method, order=order,
        work=dict(cell_node_evaluations=evaluations, panels=panels, wall_seconds=time.perf_counter() - started),
        analytic_tail=dict(raw_fov_upper=tail, scope='omitted blur tail only; excludes quadrature and roundoff'),
        error_estimates=dict(rigorous_total_error_bound=False, policy='compare frozen independent orders; no automatic promotion'))
