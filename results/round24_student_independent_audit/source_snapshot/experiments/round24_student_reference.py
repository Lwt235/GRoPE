"""Independent conditional-t5 native-rectangle reference for fixed Student t4.

The production Cartesian density/tiler/normalizer is never imported. Shared
dependencies are NumPy linear algebra and SciPy's elementary Student CDF.
Two fixed orders provide convergence evidence, not a rigorous error bound.
"""
from __future__ import annotations
from functools import lru_cache
import math
import time
import numpy as np
from scipy.special import stdtr


class ReferenceFailure(RuntimeError):
    def __init__(self, error, work, partial):
        super().__init__(type(error).__name__ + ': ' + str(error))
        self.reference_work = dict(cell_node_evaluations=work)
        self.partial_raw_mass = np.asarray(partial, np.float64)


@lru_cache(None)
def _legendre(order):
    if order not in (64, 96):
        raise ValueError('unfrozen independent reference order')
    return np.polynomial.legendre.leggauss(order)


def _centered_t5(x):
    """F5(x)-1/2 near zero, without subtracting CDF values near 1/2."""
    t = np.asarray(x, np.float64) / np.sqrt(5.)
    return (np.arctan(t) + t / (1 + t * t) + 2 * t / (3 * (1 + t * t) ** 2)) / np.pi


def t5_interval(lower, upper):
    lower, upper = np.broadcast_arrays(np.asarray(lower, np.float64), np.asarray(upper, np.float64))
    if np.isnan(lower).any() or np.isnan(upper).any() or np.any(lower > upper):
        raise ValueError('ordered non-NaN t5 interval required')
    result = np.empty(lower.shape, np.float64)
    central = (abs(lower) <= 1.) & (abs(upper) <= 1.)
    result[central] = _centered_t5(upper[central]) - _centered_t5(lower[central])
    right = (lower >= 0.) & ~central
    left = (upper <= 0.) & ~central & ~right
    for mask, a, b in ((right, lower, upper), (left, -upper, -lower)):
        if np.any(mask):
            small, large = stdtr(5., -b[mask]), stdtr(5., -a[mask])
            with np.errstate(divide='ignore', invalid='ignore'):
                value = np.exp(np.log(large)) * -np.expm1(np.log(small) - np.log(large))
            value = np.where(a[mask] == b[mask], 0., value)
            result[mask] = value
    middle = ~(central | left | right)
    result[middle] = stdtr(5., upper[middle]) - stdtr(5., lower[middle])
    if not np.isfinite(result).all() or np.any(result < 0.) or np.any(result > 1.):
        raise ValueError('unresolved independent t5 interval; no clipping/repair')
    return result


def validate(mu, covariance, bounds, content_bias, policy):
    for name, value, shape in (('mu', mu, (2,)), ('covariance', covariance, (2, 2))):
        value = np.asarray(value)
        if value.dtype != np.float64 or value.shape != shape or not np.isfinite(value).all():
            raise ValueError('finite float64 ' + name + ' required')
    mu, covariance, bounds = map(np.asarray, (mu, covariance, bounds))
    if bounds.dtype != np.float64 or bounds.ndim != 2 or bounds.shape[1] != 4 or not len(bounds) or not np.isfinite(bounds).all():
        raise ValueError('finite float64 native cells required')
    bias = np.zeros(len(bounds), np.float64) if content_bias is None else np.asarray(content_bias)
    if bias.dtype != np.float64 or bias.shape != (len(bounds),) or not np.isfinite(bias).all():
        raise ValueError('finite native bias vector required')
    if not np.array_equal(covariance, covariance.T):
        raise ValueError('exactly symmetric C required; no symmetrization')
    eigen = np.linalg.eigvalsh(covariance)
    if eigen[0] < -policy['covariance_admission_relative_tolerance'] * max(1., float(np.max(abs(covariance)))):
        raise ValueError('C outside declared covariance admission tolerance')
    if policy['degrees_of_freedom'] != 4 or not math.isfinite(policy['h']) or policy['h'] <= 0:
        raise ValueError('fixed t4 and positive h required')
    total = covariance + policy['h'] ** 2 * np.eye(2)
    if not np.isfinite(total).all():
        raise ValueError('finite S required')
    factor = np.linalg.cholesky(total)  # Admission only; no density solve.
    if not np.isfinite(factor).all():
        raise ValueError('finite strict Cholesky of S required')
    if np.any(bounds[:, 2:] <= bounds[:, :2]):
        raise ValueError('positive native rectangle sides required')
    fov = np.r_[bounds[:, :2].min(0), bounds[:, 2:].max(0)]
    for i, box in enumerate(bounds):
        overlap = np.minimum(box[2:], bounds[i + 1:, 2:]) - np.maximum(box[:2], bounds[i + 1:, :2])
        if np.any(np.all(overlap > 0, axis=1)):
            raise ValueError('disjoint native cells required')
    area = np.prod(bounds[:, 2:] - bounds[:, :2], axis=1)
    if not np.isclose(math.fsum(map(float, area)), np.prod(fov[2:] - fov[:2]), atol=0., rtol=1e-12):
        raise ValueError('cells must tile their rectangular FOV')
    return mu, total, bounds, fov, eigen


def _rectangle(mean, total, box, h, order, panel_side):
    count = int(np.ceil((box[2] - box[0]) / (panel_side * h)))
    edges = np.linspace(box[0], box[2], count + 1)
    nodes, weights = _legendre(order)
    half = (edges[1:] - edges[:-1]) / 2
    x = ((edges[1:] + edges[:-1])[:, None] / 2 + half[:, None] * nodes).ravel()
    weight = (half[:, None] * weights).ravel()
    sx = np.sqrt(total[0, 0]); slope = total[1, 0] / total[0, 0]
    residual = total[1, 1] - total[1, 0] * slope
    if not np.isfinite(residual) or residual <= 0:
        raise ValueError('conditional t5 scale lost positivity; no repair')
    u = (x - mean[0]) / sx
    marginal = 3. / (4 * np.sqrt(2.) * sx) * np.exp(-2.5 * np.log1p(u * u / 2))
    location = mean[1] + slope * (x - mean[0])
    scale = np.sqrt(residual * (2 + u * u) / 5)
    interval = t5_interval((box[1] - location) / scale, (box[3] - location) / scale)
    value = math.fsum(map(float, marginal * interval * weight))
    return value, count, len(x)


def integrate(mu, covariance, bounds, content_bias=None, *, policy, reference_policy, order, deadline=None):
    started = time.perf_counter()
    if reference_policy['orders'] != [64, 96] or reference_policy['maximum_panel_side_in_h'] != 4.:
        raise ValueError('unfrozen reference quadrature policy')
    _legendre(order)
    mean, total, bounds, fov, eigen = validate(mu, covariance, bounds, content_bias, policy)
    masses, native_panels, evaluations = [], 0, 0
    try:
        for box in bounds:
            if deadline is not None and time.perf_counter() >= deadline:
                raise RuntimeError('fixed independent reference phase deadline exceeded')
            # A vectorized x-node batch is charged before evaluating it, so
            # failures do not erase attempted numerical work.
            panels = int(np.ceil((box[2] - box[0]) / (reference_policy['maximum_panel_side_in_h'] * policy['h'])))
            evaluations += panels * order
            value, actual_panels, work = _rectangle(mean, total, box, policy['h'], order, reference_policy['maximum_panel_side_in_h'])
            if actual_panels != panels or work != panels * order:
                raise ValueError('independent reference work ledger differs')
            masses.append(value); native_panels += panels
        if deadline is not None and time.perf_counter() >= deadline:
            raise RuntimeError('fixed independent reference phase deadline exceeded')
        fov_panels = int(np.ceil((fov[2] - fov[0]) / (reference_policy['maximum_panel_side_in_h'] * policy['h'])))
        evaluations += fov_panels * order
        direct, actual_panels, work = _rectangle(mean, total, fov, policy['h'], order, reference_policy['maximum_panel_side_in_h'])
        if actual_panels != fov_panels or work != fov_panels * order:
            raise ValueError('independent direct-FOV work ledger differs')
    except Exception as error:
        raise ReferenceFailure(error, evaluations, masses) from error
    masses = np.asarray(masses, np.float64)
    if not np.isfinite(masses).all() or np.any(masses < 0) or not np.isfinite(direct) or direct < 0:
        raise ReferenceFailure(ValueError('invalid independent native mass'), evaluations, masses)
    return dict(raw_mass=masses, raw_fov_mass=direct, order=order,
        method='independent-t4-marginal-times-conditional-t5-CDF-1D-GL',
        native_panel_count=native_panels, fov_panel_count=fov_panels,
        work=dict(cell_node_evaluations=evaluations), wall_seconds=time.perf_counter() - started,
        admitted_covariance_min_eigenvalue=float(eigen[0]), tail_truncation_used=False,
        quadrature_error_bound_available=False)
