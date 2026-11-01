"""R22 numerical proposal: a positive-depth Gaussian measure on image cells.

No labels, content, case-kind dispatch, covariance jitter, or learned parameters.
Only the omitted normal-depth tail has an analytic bound. Adaptive quadrature
errors are estimates, and are deliberately reported separately.
"""
from __future__ import annotations

import math
import time
import warnings

import numpy as np
from scipy.integrate import IntegrationWarning, quad, quad_vec
from scipy.special import log_ndtr, ndtr


class NumericalFailure(RuntimeError):
    pass


def normal_interval(lower, upper):
    """Stable standard-normal interval probability, without clipping."""
    lo, hi = np.broadcast_arrays(np.asarray(lower, np.float64), np.asarray(upper, np.float64))
    if np.isnan(lo).any() or np.isnan(hi).any() or (lo > hi).any():
        raise ValueError('ordered non-NaN normal bounds required')
    out = np.empty(lo.shape, np.float64)
    left = hi <= 0
    right = (lo >= 0) & ~left
    middle = ~(left | right)
    for mask, a, b in ((left, lo, hi), (right, -hi, -lo)):
        if np.any(mask):
            la, lb = log_ndtr(a[mask]), log_ndtr(b[mask])
            with np.errstate(invalid='ignore'):
                value = np.exp(lb) * -np.expm1(la - lb)
            value = np.where(a[mask] == b[mask], 0., value)
            out[mask] = value
    out[middle] = ndtr(hi[middle]) - ndtr(lo[middle])
    if not np.isfinite(out).all() or (out < 0).any() or (out > 1).any():
        raise NumericalFailure('invalid normal interval; no clipping')
    return out


def _array(value, shape, name):
    if isinstance(value, np.ndarray) and value.dtype != np.float64:
        raise ValueError(name + ' must use float64')
    value = np.asarray(value, dtype=np.float64)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(name + ' has an invalid shape or nonfinite entry')
    return value


def prepare(mu, factor, bounds, H, offset, policy):
    mu = _array(mu, (3,), 'mu')
    factor = np.asarray(factor, dtype=np.float64) if not isinstance(factor, np.ndarray) else factor
    if factor.ndim != 2 or factor.shape[0] != 3 or not 0 <= factor.shape[1] <= 3:
        raise ValueError('factor must have shape (3,r), 0<=r<=3')
    factor = _array(factor, factor.shape, 'factor')
    H = _array(np.eye(2) if H is None else H, (2, 2), 'H')
    offset = _array(np.zeros(2) if offset is None else offset, (2,), 'offset')
    if np.linalg.det(H) == 0:
        raise ValueError('invertible screen projection required')
    bounds = np.asarray(bounds, dtype=np.float64) if not isinstance(bounds, np.ndarray) else bounds
    if bounds.ndim != 2 or bounds.shape[1] != 4 or not len(bounds):
        raise ValueError('nonempty (N,4) rectangle bounds required')
    bounds = _array(bounds, bounds.shape, 'bounds')
    widths = bounds[:, 2:] - bounds[:, :2]
    if (widths <= 0).any():
        raise ValueError('strictly positive cell areas required')
    areas = widths.prod(axis=1)
    for i in range(len(bounds)):
        overlap = np.minimum(bounds[i, 2:], bounds[i+1:, 2:]) - np.maximum(bounds[i, :2], bounds[i+1:, :2])
        if np.any(np.all(overlap > 0, axis=1)):
            raise ValueError('native cells must not overlap')
    fov = np.r_[bounds[:, :2].min(axis=0), bounds[:, 2:].max(axis=0)]
    if not math.isclose(math.fsum(map(float, areas)), float(np.prod(fov[2:]-fov[:2])), rel_tol=1e-13, abs_tol=0.):
        raise ValueError('cells must partition their rectangular FOV')
    if not all(math.isfinite(float(policy[k])) and float(policy[k]) > 0 for k in
               ('h', 'floor_density', 'depth_extent', 'outer_epsabs', 'outer_epsrel', 'inner_epsabs', 'inner_epsrel')):
        raise ValueError('positive finite numerical policy required')
    for k in ('outer_limit', 'inner_limit', 'max_depth_evaluations', 'max_inner_evaluations', 'max_breakpoints'):
        if type(policy[k]) is not int or policy[k] <= 0:
            raise ValueError('positive integer budget required: ' + k)
    result = dict(mu=mu, factor=factor, H=H, offset=offset, bounds=bounds, areas=areas,
                  all_bounds=np.vstack((bounds, fov)), fov=fov)
    lz = factor[2]
    sd = float(np.linalg.norm(lz))
    result['depth_sd'] = sd
    if not math.isfinite(sd):
        raise ValueError('finite depth norm required')
    if sd == 0:
        if np.any(lz != 0):
            raise ValueError('nonzero depth factor underflowed; no rank threshold')
        return result
    # A factor-space orthogonal complement retains PSD without Schur subtraction.
    basis, _ = np.linalg.qr((lz / sd)[:, None], mode='complete')
    beta = (factor[:2] @ (lz / sd)) / sd
    conditional_factor = H @ (factor[:2] @ basis[:, 1:])
    base = H @ beta + offset
    slope = H @ (mu[:2] - beta * mu[2])
    if not all(np.isfinite(x).all() for x in (conditional_factor, base, slope)):
        raise ValueError('conditional geometry overflow')
    result.update(base=base, slope=slope, conditional_factor=conditional_factor,
                  a=-float(mu[2])/sd)
    return result


def geometry_breakpoints(prepared, policy, lower, upper):
    """Depth roots depend on the projected mean trajectory and all cell edges."""
    points = [float(t) for t in policy['density_breakpoints'] if lower < float(t) < upper]
    bounds = prepared['bounds']
    for d in range(2):
        for edge in np.unique(bounds[:, [d, d+2]]):
            for k in policy['geometry_sigma_offsets']:
                denominator = float(edge) + float(k)*float(policy['h']) - prepared['base'][d]
                if denominator == 0:
                    continue
                z = float(prepared['slope'][d] / denominator)
                if z > 0 and math.isfinite(z):
                    t = z / prepared['depth_sd'] + prepared['a']
                    if lower < t < upper:
                        points.append(float(t))
    points = sorted(set(points))
    if len(points) > policy['max_breakpoints']:
        raise NumericalFailure('geometry breakpoint budget exceeded')
    return points


def _count(work, key, limit):
    if work.get('_deadline') is not None and time.perf_counter() >= work['_deadline']:
        raise NumericalFailure('production phase wall budget exceeded')
    if work[key] >= limit:
        raise NumericalFailure(key + ' budget exceeded')
    work[key] += 1


def _rectangle_masses(mean, image_factor, boxes, policy, work):
    """Positive conditional-x integral; no stochastic multivariate CDF."""
    work['rectangle_calls'] += len(boxes)
    sx = float(np.linalg.norm(image_factor[0]))
    sy = float(np.linalg.norm(image_factor[1]))
    if not (math.isfinite(sx) and math.isfinite(sy) and sx > 0 and sy > 0):
        raise NumericalFailure('invalid blurred image factor')
    cross_cov = float(image_factor[0] @ image_factor[1])
    if cross_cov == 0:
        raw = normal_interval((boxes[:, :2]-mean)/[sx, sy], (boxes[:, 2:]-mean)/[sx, sy]).prod(axis=1)
        work['diagonal_rectangle_calls'] += len(boxes)
        return raw, np.zeros(len(boxes), np.float64)
    direction = image_factor[0] / sx
    cross = float(image_factor[1] @ direction)
    conditional_sd = float(np.linalg.norm(image_factor[1]-cross*direction))
    if not math.isfinite(conditional_sd) or conditional_sd <= 0:
        raise NumericalFailure('conditional image standard deviation lost; no jitter')
    raw, error = [], []
    for box in boxes:
        lower, upper = (box[[0, 2]]-mean[0])/sx
        closest = float(lower if lower > 0 else upper if upper < 0 else 0.)
        scale = math.exp(-.5*closest*closest-.5*math.log(2*math.pi))
        if scale == 0:
            raise NumericalFailure('inner marginal scale underflow')
        points = [-14., -8., -4., 0., 4., 8., 14.]
        if cross != 0:
            for edge in box[[1, 3]]:
                root = float((edge-mean[1])/cross)
                spread = conditional_sd/abs(cross)
                points.extend((root-8*spread, root, root+8*spread))
        points = sorted(set(float(t) for t in points if math.isfinite(t) and lower < t < upper))

        def integrand(t):
            _count(work, 'inner_evaluations', policy['max_inner_evaluations'])
            center = mean[1] + cross*t
            width = float(normal_interval((box[1]-center)/conditional_sd, (box[3]-center)/conditional_sd))
            return math.exp(-.5*(t-closest)*(t+closest))*width

        with warnings.catch_warnings():
            warnings.simplefilter('error', IntegrationWarning)
            result = quad(integrand, float(lower), float(upper), points=points,
                          epsabs=policy['inner_epsabs'], epsrel=policy['inner_epsrel'],
                          limit=policy['inner_limit'], full_output=1)
        if len(result) != 3:
            raise NumericalFailure('inner quadrature did not converge: ' + str(result[3]))
        value, estimate, info = result
        work['inner_intervals'] += int(info['last'])
        raw.append(value*scale); error.append(estimate*scale)
    raw, error = np.asarray(raw), np.asarray(error)
    if not np.isfinite(raw).all() or not np.isfinite(error).all() or (raw < 0).any() or (raw > 1).any():
        raise NumericalFailure('invalid rectangle mass; no clipping')
    return raw, error


def integrate(mu, factor, bounds, H=None, offset=None, *, policy, deadline=None):
    """Return raw submass first; floor is applied only after depth rescaling."""
    p = prepare(mu, factor, bounds, H, offset, policy)
    work = dict(depth_evaluations=0, inner_evaluations=0, rectangle_calls=0,
                diagonal_rectangle_calls=0, inner_intervals=0, outer_intervals=0,
                _deadline=deadline)
    common = dict(status='failed', passed=False, work=work, h=float(policy['h']),
                  floor_density=float(policy['floor_density']), quadrature_error_is_estimate=True,
                  tail_bound_excludes_quadrature_and_roundoff=True, factor_columns=p['factor'].shape[1])
    blur = float(policy['h'])*np.eye(2)
    n = len(p['all_bounds'])
    try:
        if deadline is not None and time.perf_counter() >= deadline:
            raise NumericalFailure('production phase wall budget exceeded')
        if p['depth_sd'] == 0:
            front = float(p['mu'][2] > 0)
            log_front = 0. if front else None
            if front:
                center = p['H'] @ (p['mu'][:2]/p['mu'][2]) + p['offset']
                image_factor = np.column_stack((p['H'] @ (p['factor'][:2]/p['mu'][2]), blur))
                masses, errors = _rectangle_masses(center, image_factor, p['all_bounds'], policy, work)
            else:
                masses = errors = np.zeros(n, np.float64)
            tail = outer_error = 0.
            points = []
            depth_interval = None
            exact_zero = not bool(front)
        else:
            log_front = float(log_ndtr(-p['a']))
            front = math.exp(log_front)
            extent = float(policy['depth_extent'])
            lower, upper = max(-extent, p['a']), extent
            # This conservative certificate applies to the union of disjoint
            # cells, not N independent copies of the omitted depth event.
            tail = float(2*ndtr(-extent))
            exact_zero = False
            depth_interval = [float(lower), float(upper)]
            if lower >= upper:
                masses = errors = np.zeros(n, np.float64)
                outer_error = 0.
                points = []
            else:
                points = geometry_breakpoints(p, policy, lower, upper)

                def integrand(t):
                    _count(work, 'depth_evaluations', policy['max_depth_evaluations'])
                    z = p['depth_sd']*(t-p['a'])
                    if not math.isfinite(z) or z <= 0:
                        raise NumericalFailure('positive depth lost to roundoff')
                    mean = p['base'] + p['slope']/z
                    image_factor = np.column_stack((p['conditional_factor']/z, blur))
                    raw, inner_error = _rectangle_masses(mean, image_factor, p['all_bounds'], policy, work)
                    scaled_density = math.exp(-.5*t*t-.5*math.log(2*math.pi)-log_front)
                    return np.r_[raw, inner_error]*scaled_density

                values, estimate, info = quad_vec(integrand, lower, upper,
                    epsabs=policy['outer_epsabs'], epsrel=policy['outer_epsrel'], norm='max',
                    limit=policy['outer_limit'], points=points, quadrature='gk21', workers=1,
                    full_output=True)
                work['outer_intervals'] = len(info.intervals)
                if not info.success:
                    raise NumericalFailure('outer quadrature did not converge: ' + str(info.message))
                # Numerical rescaling is undone BEFORE the native area floor.
                masses = values[:n]*front
                outer_error = float(estimate*front)
                errors = values[n:]*front + outer_error
        if not np.isfinite(masses).all() or (masses < 0).any() or not np.isfinite(errors).all():
            raise NumericalFailure('invalid integrated raw mass')
        raw = masses[:-1]
        floor = float(policy['floor_density'])*p['areas']
        floored = raw + floor
        denominator = float(np.sum(floored, dtype=np.float64))
        if not math.isfinite(denominator) or denominator <= 0:
            raise NumericalFailure('nonpositive normalization')
        probability = floored/denominator
        total_floor = float(np.sum(floor, dtype=np.float64))
        common.update(status='complete', passed=True, raw_mass=raw, raw_fov_mass=float(masses[-1]),
            raw_mass_sum=float(np.sum(raw, dtype=np.float64)), probability=probability,
            floored_mass=floored, normalizer=denominator, positive_mass=front,
            log_positive_mass=log_front, positive_mass_underflow=(front == 0 and not exact_zero),
            exact_zero_positive_mass=exact_zero, zero_raw_cells=int(np.sum(raw == 0)),
            total_area_floor=total_floor, depth_tail_bound=tail,
            certified_tail_amplification=2*tail/total_floor,
            quadrature_error_estimate=errors[:-1], fov_quadrature_error_estimate=float(errors[-1]),
            outer_error_estimate=outer_error, depth_interval=depth_interval,
            depth_breakpoints=points, integrand_scaling='positive-depth probability, multiplied back before floor')
    except (NumericalFailure, IntegrationWarning, FloatingPointError, OverflowError) as error:
        common['failure_reason'] = str(error)
    work.pop('_deadline', None)
    return common
