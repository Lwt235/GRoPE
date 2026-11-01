"""Independent synthetic references for geometry-conditioned Gaussian attention.

No production R18 module is imported. The primary reference is tensor-product
Gauss--Legendre integration on [-14, 14]^2 in standard-normal coordinates.
The independent T=0 marginal uses scipy.integrate.quad on a one-dimensional
conditional-Gaussian rectangle formula. Neither route clips/repairs covariance,
mass, or probabilities, nor substitutes a fallback when integration fails.

The reported tail bounds are maximum absolute *component* bounds, not L1 and
not bounds on quadrature error. One uses the computed truncated geometry
normalizer; a separate certified expression uses only the positive area floor
and the analytic standard-normal tail bound. Quadrature convergence remains a
separate requirement. No numerical reference is executed on module import.
"""
from __future__ import annotations

from functools import lru_cache
import math
from numbers import Integral, Real
import warnings

import numpy as np
from scipy.integrate import IntegrationWarning, quad
from scipy.special import erfc


STANDARD_BOUND = 14.
SUPPORTED_ORDERS = (384, 512)
MAX_CHUNK_SIZE = 2048


def _scalar(value, name, *, nonnegative=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(name + ' must be a real scalar')
    result = float(value)
    if not math.isfinite(result) or (result < 0 if nonnegative else result <= 0):
        raise ValueError(name + ' has an invalid range')
    return result


def _array(value, name):
    if np.iscomplexobj(value):
        raise ValueError(name + ' must be real')
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(name + ' must be numeric') from error
    if not np.isfinite(result).all():
        raise ValueError(name + ' must be finite')
    return result


def _geometry_inputs(mean, covariance, bounds, bandwidth, floor_density):
    mean = _array(mean, 'mean')
    covariance = _array(covariance, 'covariance')
    bounds = _array(bounds, 'bounds')
    single = mean.ndim == 1
    if single:
        if mean.shape != (2,) or covariance.shape != (2, 2):
            raise ValueError('single query requires mean[2] and covariance[2,2]')
        mean, covariance = mean[None], covariance[None]
    elif (mean.ndim != 2 or mean.shape[1] != 2 or not len(mean)
          or covariance.shape != (len(mean), 2, 2)):
        raise ValueError('batched query requires mean[Q,2] and covariance[Q,2,2]')
    if bounds.ndim != 2 or bounds.shape[1] != 4 or not len(bounds):
        raise ValueError('nonempty bounds[K,4] in xmin,ymin,xmax,ymax order required')
    if not np.array_equal(covariance, covariance.swapaxes(-1, -2)):
        raise ValueError('covariance must be symmetric; no symmetrization is applied')
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if not np.isfinite(eigenvalues).all() or not np.isfinite(eigenvectors).all():
        raise ValueError('covariance eigendecomposition must be finite')
    if (eigenvalues < 0).any():
        raise ValueError('covariance must be PSD; negative eigenvalues are not clipped')
    if (bounds[:, 2:] <= bounds[:, :2]).any():
        raise ValueError('key rectangles must have positive width and height')
    # Disjoint interiors imply sum_j rectangle_mass_j(x) <= 1. Touching
    # boundaries have zero Gaussian measure and remain legal.
    for index, rectangle in enumerate(bounds[:-1]):
        others = bounds[index + 1:]
        overlaps = (np.minimum(rectangle[2:], others[:, 2:])
                    > np.maximum(rectangle[:2], others[:, :2])).all(axis=-1)
        if overlaps.any():
            raise ValueError('key rectangles must have disjoint interiors')
    bandwidth = _scalar(bandwidth, 'bandwidth')
    if not math.isfinite(bandwidth * bandwidth) or bandwidth * bandwidth == 0:
        raise ValueError('bandwidth square must be finite and positive')
    floor_density = _scalar(floor_density, 'floor_density')
    with np.errstate(over='raise', invalid='raise', under='ignore'):
        try:
            areas = (bounds[:, 2:] - bounds[:, :2]).prod(axis=-1)
            floor = floor_density * areas
            total_floor = float(floor.sum())
        except FloatingPointError as error:
            raise ValueError('rectangle areas and floors must be finite') from error
    if (not np.isfinite(areas).all() or not np.isfinite(floor).all()
            or (floor <= 0).any() or not math.isfinite(total_floor) or total_floor > 1):
        raise ValueError('positive representable area floors with total <= 1 required')
    return mean, covariance, bounds, bandwidth, floor, eigenvalues, eigenvectors, single


def _normal_interval(low, high):
    """Normal CDF difference from erfc; choose tails without subtracting 1-1."""
    low, high = np.broadcast_arrays(np.asarray(low, dtype=np.float64),
                                    np.asarray(high, dtype=np.float64))
    if np.isnan(low).any() or np.isnan(high).any() or (high < low).any():
        raise ValueError('ordered, non-NaN standardized interval required')
    root_two = math.sqrt(2.)
    result = np.empty(low.shape, dtype=np.float64)
    positive = low >= 0
    negative = (~positive) & (high <= 0)
    crossing = ~(positive | negative)
    result[positive] = .5 * (erfc(low[positive] / root_two)
                              - erfc(high[positive] / root_two))
    result[negative] = .5 * (erfc(-high[negative] / root_two)
                              - erfc(-low[negative] / root_two))
    result[crossing] = .5 * (erfc(low[crossing] / root_two)
                              - erfc(high[crossing] / root_two))
    if not np.isfinite(result).all() or (result < 0).any() or (result > 1).any():
        raise ValueError('invalid normal interval mass; no clipping is applied')
    return result


def _rectangle_mass(centers, bounds, bandwidth, floor):
    with np.errstate(over='raise', invalid='raise', divide='raise', under='ignore'):
        try:
            low = (bounds[None, :, :2] - centers[:, None, :]) / bandwidth
            high = (bounds[None, :, 2:] - centers[:, None, :]) / bandwidth
            mass = _normal_interval(low, high).prod(axis=-1) + floor[None]
        except FloatingPointError as error:
            raise ValueError('standardized rectangles or masses overflowed') from error
    if not np.isfinite(mass).all() or (mass <= 0).any():
        raise ValueError('floored rectangle mass must be finite and positive')
    return mass


@lru_cache(maxsize=2)
def _standard_quadrature(order):
    nodes, weights = np.polynomial.legendre.leggauss(order)
    z = STANDARD_BOUND * nodes
    weighted_density = (STANDARD_BOUND * weights
                        * np.exp(-.5 * z * z) / math.sqrt(2. * math.pi))
    grid = np.stack(np.meshgrid(z, z, indexing='ij'), axis=-1).reshape(-1, 2)
    product_weights = np.multiply.outer(weighted_density, weighted_density).reshape(-1)
    grid.setflags(write=False)
    product_weights.setflags(write=False)
    return grid, product_weights


def conditional_gl_reference(mean, covariance, bounds, content, bandwidth,
                             temperature, *, order=384, chunk_size=2048,
                             floor_density=1.e-30):
    """Return P, Zg and unnormalized numerators using independent GL384/GL512.

    Shapes: mean[2], covariance[2,2], content[K], bounds[K,4], or batched
    mean[Q,2], covariance[Q,2,2], content[Q,K] with shared bounds. Coordinates
    and rectangle areas use the same units. C=0 and exact rank-one PSD C are
    handled by the same eigen-coordinate quadrature, with no special fallback.

    Since A <= 1 + sum(floor) <= 2, excluded square probability <= tail, and
    0 <= omitted numerator_j <= omitted Zg, each component's truncation error
    is <= omitted_Zg/(inside_Zg+omitted_Zg) <= 2*tail/inside_Zg. The reported
    ``amplified_tail_bound`` substitutes the computed inside_Zg. Separately,
    A >= total_floor gives inside_Zg >= total_floor*(1-tail), so
    ``certified_amplified_tail_bound`` does not rely on numerical quadrature
    for its denominator. Both exclude integration error, which still requires
    a separate convergence check.
    """
    (means, covariances, boxes, bandwidth, floor, eigenvalues,
     eigenvectors, single) = _geometry_inputs(mean, covariance, bounds, bandwidth,
                                            floor_density)
    del covariances
    scores = _array(content, 'content')
    if single:
        if scores.shape != (len(boxes),):
            raise ValueError('single-query content must have shape [K]')
        scores = scores[None]
    elif scores.shape != (len(means), len(boxes)):
        raise ValueError('batched content must have shape [Q,K]')
    if (np.abs(scores) > 1).any():
        raise ValueError('declared content domain is [-1,1]')
    temperature = _scalar(temperature, 'temperature', nonnegative=True)
    if isinstance(order, (bool, np.bool_)) or not isinstance(order, Integral) or order not in SUPPORTED_ORDERS:
        raise ValueError('independent reference order must be 384 or 512')
    if (isinstance(chunk_size, (bool, np.bool_)) or not isinstance(chunk_size, Integral)
            or not 1 <= chunk_size <= MAX_CHUNK_SIZE):
        raise ValueError('chunk_size must be an integer in [1,2048]')
    standard, weights = _standard_quadrature(int(order))
    raw_quadrature_mass = float(weights.sum())
    numerators = np.zeros((len(means), len(boxes)), dtype=np.float64)
    marginal_masses = np.zeros_like(numerators)
    normalizers = np.empty(len(means), dtype=np.float64)
    probabilities = np.empty_like(numerators)
    for query in range(len(means)):
        factor = eigenvectors[query] * np.sqrt(eigenvalues[query])[None]
        # A very large finite T may underflow non-maximal exp terms to zero.
        # This is the declared stable exponential, not clipping or a fallback.
        with np.errstate(over='ignore', under='ignore', invalid='raise'):
            exponential = np.exp(temperature * (scores[query] - scores[query].max()))
        for start in range(0, len(weights), int(chunk_size)):
            end = start + int(chunk_size)
            centers = means[query] + standard[start:end] @ factor.T
            if not np.isfinite(centers).all():
                raise ValueError('transformed quadrature coordinates overflowed')
            mass = _rectangle_mass(centers, boxes, bandwidth, floor)
            geometry_sum = mass.sum(axis=-1)
            tilted = mass * exponential[None]
            local_normalizer = tilted.sum(axis=-1)
            if (not np.isfinite(geometry_sum).all() or not np.isfinite(local_normalizer).all()
                    or (geometry_sum <= 0).any() or (local_normalizer <= 0).any()):
                raise ValueError('positive finite local normalizers required')
            local_probability = tilted / local_normalizer[:, None]
            weighted_geometry = weights[start:end] * geometry_sum
            numerators[query] += (weighted_geometry[:, None] * local_probability).sum(axis=0)
            marginal_masses[query] += (weights[start:end, None] * mass).sum(axis=0)
        normalizers[query] = marginal_masses[query].sum()
        if not math.isfinite(float(normalizers[query])) or normalizers[query] <= 0:
            raise ValueError('truncated geometry normalizer must be finite and positive')
        probabilities[query] = numerators[query] / normalizers[query]
    if (not np.isfinite(probabilities).all() or (probabilities < 0).any()
            or not np.isfinite(numerators).all()):
        raise ValueError('invalid integrated probabilities; no normalization repair')
    tail_bound = float(4. * .5 * erfc(STANDARD_BOUND / math.sqrt(2.)))
    amplified_tail_bound = 2. * tail_bound / normalizers
    total_floor = float(floor.sum())
    certified_geometry_lower_bound = total_floor * (1. - tail_bound)
    certified_amplified_tail_bound = 2. * tail_bound / certified_geometry_lower_bound
    return dict(P=probabilities[0] if single else probabilities,
                Zg=float(normalizers[0]) if single else normalizers,
                numerator=numerators[0] if single else numerators,
                geometry_mass=marginal_masses[0] if single else marginal_masses,
                raw_quadrature_mass=raw_quadrature_mass,
                tail_bound=tail_bound,
                amplified_tail_bound=float(amplified_tail_bound[0]) if single else amplified_tail_bound,
                certified_amplified_tail_bound=certified_amplified_tail_bound,
                certified_geometry_lower_bound=certified_geometry_lower_bound,
                certified_tail_bound_basis='A>=total_floor; normal_square_mass>=1-tail_bound',
                tail_bound_norm='maximum_absolute_component',
                tail_bound_excludes_quadrature_error=True,
                raw_quadrature_nodes=len(weights), order=int(order),
                standard_normal_bound=STANDARD_BOUND,
                total_area_floor=total_floor)


def gaussian_marginal_reference(mean, covariance, bounds, bandwidth, *,
                                epsabs=1.e-13, epsrel=1.e-12,
                                floor_density=1.e-30):
    """Full-domain marginal for T=0 from independent conditional 1D quad.

    Computes each rectangle under N(mean, C+h^2 I), then adds exactly one
    density*area floor. Cholesky here is a conditional-normal parameterization,
    not a covariance repair. Before quad, divide the integrand by the maximum
    marginal x density on the original rectangle interval; multiply the result
    and error estimate back afterward. Thus epsabs does not silently discard
    a tiny but relevant visible tail solely because its marginal x mass is
    far below epsabs. The quad error is an estimate, not a certificate.
    """
    (means, covariances, boxes, bandwidth, floor, _,
     _, single) = _geometry_inputs(mean, covariance, bounds, bandwidth, floor_density)
    epsabs = _scalar(epsabs, 'epsabs')
    epsrel = _scalar(epsrel, 'epsrel')
    raw = np.empty((len(means), len(boxes)), dtype=np.float64)
    error_estimates = np.empty_like(raw)
    for query, (center, covariance) in enumerate(zip(means, covariances)):
        convolved = covariance + bandwidth * bandwidth * np.eye(2)
        if not np.isfinite(convolved).all():
            raise ValueError('convolved covariance must be finite')
        try:
            factor = np.linalg.cholesky(convolved)
        except np.linalg.LinAlgError as error:
            raise ValueError('convolved covariance is numerically non-positive-definite') from error
        sigma_x, cross, sigma_y_given_x = factor[0, 0], factor[1, 0], factor[1, 1]
        for key, rectangle in enumerate(boxes):
            lower = float((rectangle[0] - center[0]) / sigma_x)
            upper = float((rectangle[2] - center[0]) / sigma_x)
            if not math.isfinite(lower) or not math.isfinite(upper):
                raise ValueError('finite standardized x limits required')
            closest_to_zero = lower if lower > 0 else upper if upper < 0 else 0.
            maximum_log_x_density = (-.5 * closest_to_zero * closest_to_zero
                                     - .5 * math.log(2. * math.pi))
            maximum_x_density = math.exp(maximum_log_x_density)

            def integrand(z):
                conditional_mean_y = center[1] + cross * z
                interval = _normal_interval((rectangle[1] - conditional_mean_y) / sigma_y_given_x,
                                            (rectangle[3] - conditional_mean_y) / sigma_y_given_x)
                scaled_x_density = math.exp(-.5 * (z - closest_to_zero) * (z + closest_to_zero))
                return scaled_x_density * float(interval)

            # Expose the normal-density core and conditional boundary
            # transitions to quad; the original interval is never truncated.
            points = [-14., -8., -4., 0., 4., 8., 14.]
            if cross != 0:
                for y in (rectangle[1], rectangle[3]):
                    transition = (y - center[1]) / cross
                    spread = sigma_y_given_x / abs(cross)
                    points.extend((transition - 8 * spread, transition, transition + 8 * spread))
            points = sorted(set(float(z) for z in points if math.isfinite(float(z)) and lower < z < upper))
            with warnings.catch_warnings():
                warnings.simplefilter('error', IntegrationWarning)
                scaled_value, scaled_error = quad(integrand, lower, upper, epsabs=epsabs,
                                                 epsrel=epsrel, points=points, limit=256)
            value, error = scaled_value * maximum_x_density, scaled_error * maximum_x_density
            if not math.isfinite(value) or not math.isfinite(error) or value < 0 or value > 1:
                raise ValueError('invalid conditional rectangle integral; no clipping')
            raw[query, key], error_estimates[query, key] = value, error
    floored = raw + floor[None]
    normalizers = floored.sum(axis=-1)
    if not np.isfinite(normalizers).all() or (normalizers <= 0).any():
        raise ValueError('positive finite full-domain geometry normalizer required')
    probability = floored / normalizers[:, None]
    return dict(P=probability[0] if single else probability,
                Zg=float(normalizers[0]) if single else normalizers,
                mass=raw[0] if single else raw,
                floored_mass=floored[0] if single else floored,
                quad_error=error_estimates[0] if single else error_estimates,
                quad_error_is_estimate=True, epsabs=epsabs, epsrel=epsrel,
                quad_integrand_scaling='maximum_marginal_x_density_on_original_interval',
                total_area_floor=float(floor.sum()))
