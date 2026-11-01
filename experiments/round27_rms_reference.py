"""Independent R27 homogeneous-depth reference with fixed standard-X panels.

RMS normalization and a latent SVD nullspace are independently implemented.
The reference never imports a producer, density evaluator, fixture or contract.
NumPy/SciPy elementary operations are shared dependencies. Analytic truncation
bounds below exclude adaptive quadrature, SVD and floating-point errors.
"""
from __future__ import annotations

from functools import lru_cache
import math
import time

import numpy as np
from scipy.integrate import quad_vec
from scipy.special import erf, log_ndtr, logsumexp, ndtr

LOG_SQRT_2PI = .5 * math.log(2. * math.pi)


class NodeBudget:
    """One shared, monotonically charged phase budget, never silently reset."""
    def __init__(self, maximum):
        if type(maximum) is not int or maximum < 0:
            raise ValueError('nonnegative integer shared node budget required')
        self.maximum = maximum
        self.used = 0

    @property
    def remaining_nodes(self):
        return self.maximum - self.used

    def charge(self, count):
        if type(count) is not int or count < 0 or count > self.remaining_nodes:
            raise RuntimeError('fixed whole-reference inner-node budget exceeded')
        self.used += count


class ReferenceFailure(RuntimeError):
    def __init__(self, error, work, arrays=None, metadata=None):
        super().__init__(type(error).__name__ + ': ' + str(error))
        self.reference_work = dict(work)
        self.available_arrays = dict(arrays or {})
        self.partial_raw_mass = self.available_arrays.get('partial_raw_mass', np.empty(0, np.float64))
        self.reference_metadata = dict(metadata or {})


class _Ledger:
    def __init__(self, reference_policy, deadline, budget):
        self.policy, self.deadline, self.budget = reference_policy, deadline, budget
        self.work = dict(depth_evaluations=0, inner_x_nodes=0, narrow_y_pdf_nodes=0,
                         elementary_inner_nodes=0, cell_node_evaluations=0, inner_panels=0, outer_intervals=0,
                         outer_geometry_breakpoints=0)

    def check_time(self):
        if self.deadline is not None and time.perf_counter() >= self.deadline:
            raise RuntimeError('fixed independent reference phase deadline exceeded')

    def depth(self):
        self.check_time()
        if self.work['depth_evaluations'] >= self.policy['maximum_depth_evaluations']:
            raise RuntimeError('fixed independent depth evaluation budget exceeded')
        self.work['depth_evaluations'] += 1

    def nodes(self, count, kind):
        self.check_time()
        if self.work['elementary_inner_nodes'] + count > self.policy['max_call_cell_node_evaluations']:
            raise RuntimeError('fixed per-call independent inner-node budget exceeded')
        self.budget.charge(count)
        self.work['elementary_inner_nodes'] += count
        self.work['cell_node_evaluations'] += count
        self.work[kind] += count


@lru_cache(None)
def _legendre(order):
    if order not in (16, 32, 48):
        raise ValueError('only frozen independent GL16/32/48 orders are allowed')
    return np.polynomial.legendre.leggauss(order)


def _finite_float64(value, shape, name):
    value = np.asarray(value)
    if value.dtype != np.float64 or value.shape != shape or not np.isfinite(value).all():
        raise ValueError('finite float64 ' + str(shape) + ' required: ' + name)
    return value


def validate(mu, factor, bounds, content_bias, policy):
    mu = _finite_float64(mu, (3,), 'mu')
    factor = _finite_float64(factor, (3, 3), 'factor')
    bounds = np.asarray(bounds)
    if (bounds.dtype != np.float64 or bounds.ndim != 2 or bounds.shape[1:] != (4,)
            or not len(bounds) or not np.isfinite(bounds).all() or np.any(bounds[:, 2:] <= bounds[:, :2])):
        raise ValueError('finite positive float64 native rectangles required')
    bias = np.zeros(len(bounds), np.float64) if content_bias is None else _finite_float64(content_bias, (len(bounds),), 'content_bias')
    if not math.isfinite(policy['h']) or policy['h'] <= 0 or not math.isfinite(policy['floor_density']) or policy['floor_density'] <= 0:
        raise ValueError('positive finite bandwidth and single floor required')
    for index, box in enumerate(bounds):
        overlap = np.minimum(box[2:], bounds[index+1:, 2:]) - np.maximum(box[:2], bounds[index+1:, :2])
        if np.any(np.all(overlap > 0, axis=1)):
            raise ValueError('native rectangles must be disjoint')
    fov = np.r_[bounds[:, :2].min(0), bounds[:, 2:].max(0)]
    areas = np.prod(bounds[:, 2:] - bounds[:, :2], axis=1)
    if not np.isclose(math.fsum(map(float, areas)), np.prod(fov[2:] - fov[:2]), rtol=1e-12, atol=0):
        raise ValueError('native cells must tile enclosing rectangular FOV')
    return mu, factor, bounds, bias, fov


def prepare(mu, factor, h):
    """Independent SVD conditioning, with hI retained as explicit factor columns."""
    depth_scale = max(abs(float(mu[2])), float(np.max(np.abs(factor[2]))))
    stochastic = bool(np.any(factor[2] != 0))
    if depth_scale == 0:
        return dict(structural_branch='zero_depth', structural_zero=True, rms_depth=0.,
                    normalized_depth_mean=0., normalized_depth_std=0., front_mass=0., log_front_mass=None)
    if not stochastic and mu[2] < 0:
        # The raw measure is zero regardless of the XY scale. Do not overflow
        # an irrelevant XY normalization before taking this exact branch.
        return dict(structural_branch='nonpositive_constant_depth', structural_zero=True,
                    rms_depth=abs(float(mu[2])), normalized_depth_mean=-1.,
                    normalized_depth_std=0., front_mass=0., log_front_mass=None)
    scaled_depth = np.r_[mu[2] / depth_scale, factor[2] / depth_scale]
    relative_rms = math.hypot(*map(float, scaled_depth))
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        normalized_mu = (mu / depth_scale) / relative_rms
        normalized_factor = (factor / depth_scale) / relative_rms
    if not np.isfinite(normalized_mu).all() or not np.isfinite(normalized_factor).all():
        raise FloatingPointError('nonfinite RMS-normalized parameters')
    rms = depth_scale * relative_rms
    if not math.isfinite(rms):
        raise FloatingPointError('unrepresentable RMS depth scalar')
    m = float(normalized_mu[2])
    sigma = math.hypot(*map(float, normalized_factor[2]))
    if stochastic and sigma == 0:
        raise FloatingPointError('stochastic depth lost in RMS normalization; no deterministic replacement')
    if not stochastic and m <= 0:
        return dict(structural_branch='nonpositive_constant_depth', structural_zero=True,
                    rms_depth=rms, normalized_depth_mean=m, normalized_depth_std=0., front_mass=0., log_front_mass=None)
    if stochastic:
        direction = normalized_factor[2] / sigma
        _, _, vh = np.linalg.svd(normalized_factor[2:3], full_matrices=True)
        null = vh[1:].T
        base = normalized_factor[:2] @ null
        beta = normalized_factor[:2] @ direction
        branch = 'stochastic_depth'
        with np.errstate(over='ignore'):
            log_front = float(log_ndtr(m / sigma))
        front = math.exp(log_front) if math.isfinite(log_front) else 0.
        if front <= 0:
            raise FloatingPointError('positive front probability is unrepresentable')
    else:
        base, beta, branch, log_front, front = normalized_factor[:2], np.zeros(2), 'positive_constant_depth', 0., 1.
    conditional_factor = np.column_stack((base, h * np.eye(2)))
    sx = math.hypot(*map(float, conditional_factor[0]))
    marginal_sy = math.hypot(*map(float, conditional_factor[1]))
    direction_x = conditional_factor[0] / sx
    k = float(np.dot(conditional_factor[1], direction_x))
    residual = conditional_factor[1] - k * direction_x
    sy = math.hypot(*map(float, residual))
    if not all(math.isfinite(v) and v > 0 for v in (sx, marginal_sy, sy)) or not np.isfinite(beta).all() or not math.isfinite(k):
        raise FloatingPointError('conditional Gaussian factor is not finite positive')
    return dict(structural_branch=branch, structural_zero=False, rms_depth=rms,
        normalized_depth_mean=m, normalized_depth_std=sigma, front_mass=front, log_front_mass=log_front,
        xy_mean=normalized_mu[:2], beta=beta, sx=sx, marginal_sy=marginal_sy,
        conditional_y_coefficient=k, conditional_y_residual_std=sy,
        conditional_factor=conditional_factor)


def normal_interval(midpoint, halfwidth, *, reference_policy, ledger=None):
    """Normal interval from its center/positive halfwidth; no tiny CDF subtraction."""
    mid, half = np.broadcast_arrays(np.asarray(midpoint, np.float64), np.asarray(halfwidth, np.float64))
    if not np.isfinite(mid).all() or not np.isfinite(half).all() or np.any(half < 0):
        raise FloatingPointError('finite normal midpoint and nonnegative halfwidth required')
    result = np.zeros(mid.shape, np.float64)
    with np.errstate(over='ignore', invalid='ignore'):
        narrow = (half > 0) & (half * (np.abs(mid) + half) <= reference_policy['narrow_cdf_exponent_span'])
    if narrow.any():
        count = int(np.count_nonzero(narrow)) * reference_policy['narrow_cdf_order']
        if ledger is not None:
            ledger.nodes(count, 'narrow_y_pdf_nodes')
        nodes, weights = _legendre(reference_policy['narrow_cdf_order'])
        x = mid[narrow, None] + half[narrow, None] * nodes
        with np.errstate(over='ignore'):
            result[narrow] = np.exp(np.log(half[narrow]) + logsumexp(np.log(weights)[None] - .5*x*x - LOG_SQRT_2PI, axis=1))
    broad = (half > 0) & ~narrow
    if broad.any():
        lo, hi = mid[broad] - half[broad], mid[broad] + half[broad]
        values = np.empty(lo.shape)
        negative, positive = hi <= 0, lo >= 0
        crossing = ~(negative | positive)
        if negative.any():
            upper, lower = log_ndtr(hi[negative]), log_ndtr(lo[negative])
            values[negative] = np.exp(upper) * (-np.expm1(lower-upper))
        if positive.any():
            upper, lower = log_ndtr(-lo[positive]), log_ndtr(-hi[positive])
            values[positive] = np.exp(upper) * (-np.expm1(lower-upper))
        if crossing.any():
            values[crossing] = .5 * (erf(hi[crossing] / math.sqrt(2.)) - erf(lo[crossing] / math.sqrt(2.)))
        result[broad] = values
    if not np.isfinite(result).all() or np.any(result < 0):
        raise FloatingPointError('invalid independent normal interval')
    return result


def standard_x_grid(reference_policy):
    """Prospective R27 fixed grid; independent of cells, scores and results."""
    if reference_policy['inner_x_cutoff'] != 32. or reference_policy['standard_x_panel_width'] != 2.:
        raise ValueError('fixed standard-X cutoff32 and maximum panel width2 required')
    return np.arange(-32., 33., 2., dtype=np.float64)


def _rectangle(location, z, box, prepared, order, reference_policy, ledger):
    sx, ky, sy = prepared['sx'], prepared['conditional_y_coefficient'], prepared['conditional_y_residual_std']
    xmid = (z * ((box[0] + box[2]) / 2.) - location[0]) / sx
    xhalf = z * ((box[2] - box[0]) / 2.) / sx
    cutoff = reference_policy['inner_x_cutoff']
    lo, hi = max(-cutoff, xmid-xhalf), min(cutoff, xmid+xhalf)
    if hi < lo or xhalf <= 0 or lo >= cutoff or hi <= -cutoff:
        return 0.
    # Preserve a positive halfwidth even when finite endpoints round together.
    inside = xmid-xhalf >= -cutoff and xmid+xhalf <= cutoff
    if hi == lo:
        if not inside:
            return 0.
        panels = [(xmid, xhalf)]
    else:
        cuts = {float(lo), float(hi)}
        cuts.update(float(point) for point in standard_x_grid(reference_policy) if lo < point < hi)
        if ky != 0:
            for edge in (box[1], box[3]):
                for c in reference_policy['inner_geometry_breaks']:
                    root = (z * edge - location[1] - c * sy) / ky
                    if math.isfinite(root) and lo < root < hi:
                        cuts.add(float(root))
        cuts = sorted(cuts)
        panels = [((a+b)/2., (b-a)/2.) for a, b in zip(cuts, cuts[1:])]
        if len(panels) == 1 and inside:
            panels = [(float(xmid), float(xhalf))]
    nodes, weights = _legendre(order)
    values = []
    for middle, half in panels:
        ledger.nodes(order, 'inner_x_nodes')
        ledger.work['inner_panels'] += 1
        u = middle + half * nodes
        ymid = (z * ((box[1] + box[3]) / 2.) - location[1] - ky*u) / sy
        yhalf = z * ((box[3] - box[1]) / 2.) / sy
        probability = normal_interval(ymid, yhalf, reference_policy=reference_policy, ledger=ledger)
        values.append(math.fsum(map(float, half * weights * np.exp(-.5*u*u-LOG_SQRT_2PI) * probability)))
    return math.fsum(values)


def _rectangles(location, z, boxes, prepared, order, reference_policy, ledger):
    """Vectorized ragged-panel version of the independent Gaussian rectangles.

    Every cell's own exact interior roots are retained. This changes batching,
    not the GL order, panel rule, cutoff, narrow-CDF branch or mathematical law.
    """
    sx, ky, sy = prepared['sx'], prepared['conditional_y_coefficient'], prepared['conditional_y_residual_std']
    cutoff = reference_policy['inner_x_cutoff']
    xmid = (z * ((boxes[:, 0]+boxes[:, 2])/2.) - location[0]) / sx
    xhalf = z * ((boxes[:, 2]-boxes[:, 0])/2.) / sx
    if not np.isfinite(xmid).all() or not np.isfinite(xhalf).all():
        raise FloatingPointError('nonfinite standardized homogeneous rectangle')
    low, high = xmid-xhalf, xmid+xhalf
    lo, hi = np.maximum(-cutoff, low), np.minimum(cutoff, high)
    active = (xhalf > 0) & (hi >= lo) & (lo < cutoff) & (hi > -cutoff)
    inside = (low >= -cutoff) & (high <= cutoff)
    regular = active & (hi > lo)
    columns = [np.where(regular, lo, np.inf), np.where(regular, hi, np.inf)]
    for point in standard_x_grid(reference_policy):
        columns.append(np.where(regular & (lo < point) & (point < hi), point, np.inf))
    if ky != 0:
        for axis in (1, 3):
            for c in reference_policy['inner_geometry_breaks']:
                roots = (z * boxes[:, axis]-location[1]-c*sy) / ky
                columns.append(np.where(regular & np.isfinite(roots) & (lo < roots) & (roots < hi), roots, np.inf))
    cuts = np.sort(np.column_stack(columns), axis=1)
    a, b = cuts[:, :-1], cuts[:, 1:]
    valid = np.isfinite(a) & np.isfinite(b) & (b > a)
    owners = np.nonzero(valid)[0]
    middle, half = (a[valid]+b[valid])/2., (b[valid]-a[valid])/2.
    if len(owners):
        counts = np.bincount(owners, minlength=len(boxes))
        preserve = inside[owners] & (counts[owners] == 1)
        middle[preserve], half[preserve] = xmid[owners[preserve]], xhalf[owners[preserve]]
    collapsed = np.flatnonzero(active & inside & (hi == lo))
    if len(collapsed):
        owners = np.r_[owners, collapsed]
        middle, half = np.r_[middle, xmid[collapsed]], np.r_[half, xhalf[collapsed]]
    result = np.zeros(len(boxes), np.float64)
    if not len(owners):
        return result
    ledger.nodes(int(len(owners)*order), 'inner_x_nodes')
    ledger.work['inner_panels'] += int(len(owners))
    nodes, weights = _legendre(order)
    u = middle[:, None]+half[:, None]*nodes
    ymid = (z*((boxes[owners, 1]+boxes[owners, 3])/2.)[:, None]-location[1]-ky*u)/sy
    yhalf = (z*((boxes[owners, 3]-boxes[owners, 1])/2.)/sy)[:, None]
    probability = normal_interval(ymid, yhalf, reference_policy=reference_policy, ledger=ledger)
    panel = np.sum(half[:, None]*weights*np.exp(-.5*u*u-LOG_SQRT_2PI)*probability, axis=1)
    result = np.bincount(owners, weights=panel, minlength=len(boxes))
    if not np.isfinite(result).all() or np.any(result < 0):
        raise FloatingPointError('invalid batched independent Gaussian rectangle')
    return result


def outer_breaks(prepared, boxes, lower, upper, reference_policy):
    m, sigma = prepared['normalized_depth_mean'], prepared['normalized_depth_std']
    points = set()
    for axis, sd in ((0, prepared['sx']), (1, prepared['marginal_sy'])):
        edges = np.unique(boxes[:, [axis, axis+2]])
        mu, beta = float(prepared['xy_mean'][axis]), float(prepared['beta'][axis])
        for edge in edges:
            scale = max(1., abs(float(edge)), abs(mu), abs(beta), sd)
            denominator = sigma * (edge/scale) - beta/scale
            if denominator == 0:
                continue
            for c in reference_policy['outer_geometry_breaks']:
                root = (mu/scale - m*(edge/scale) + c*(sd/scale)) / denominator
                if math.isfinite(root) and lower < root < upper:
                    points.add(float(root))
    return sorted(points)


def _probabilities(raw, bounds, bias, floor_density):
    floor = np.prod(bounds[:, 2:]-bounds[:, :2], axis=1) * floor_density
    if not np.isfinite(floor).all() or np.any(floor <= 0):
        raise FloatingPointError('unrepresentable positive native area floor')
    floored = raw + floor
    logits = np.log(floored) + bias
    return np.exp(logits-logsumexp(logits)), floored, floor


def truncation_bounds(prepared, raw, bounds, bias, floor, reference_policy):
    if prepared['structural_zero']:
        return dict(depth_omitted_mass=0., inner_x_omitted_mass=0., native_raw_l1_absolute=0.,
                    direct_fov_absolute=0., raw_l1_over_reference_fov=None,
                    probability_max_absolute=0., structural_zero_exemption=True)
    cutoff = reference_policy['depth_cutoff']
    if prepared['normalized_depth_std'] == 0:
        depth_tail = 0.
    else:
        threshold = -prepared['normalized_depth_mean'] / prepared['normalized_depth_std']
        lower_tail = float(ndtr(-cutoff)-ndtr(threshold)) if threshold < -cutoff else 0.
        depth_tail = lower_tail + float(ndtr(-max(cutoff, threshold)))
    inner_tail = float(2.*ndtr(-reference_policy['inner_x_cutoff'])) * prepared['front_mass']
    total = depth_tail + inner_tail
    raw_sum = math.fsum(map(float, raw))
    weights = np.exp(bias-bias.max())
    weighted_floor = math.fsum(map(float, floor*weights))
    if weighted_floor <= 0:
        raise FloatingPointError('bias-aware area-floor lower bound is unrepresentable')
    return dict(depth_omitted_mass=depth_tail, inner_x_omitted_mass=inner_tail,
        native_raw_l1_absolute=total, direct_fov_absolute=total,
        native_union_bound=True, raw_l1_over_reference_fov=total/raw_sum if raw_sum > 0 else None,
        raw_relative_denominator='computed retained FOV mass, not an interval enclosure',
        probability_max_absolute=min(1., 2.*total/weighted_floor),
        structural_zero_exemption=False, excludes_quadrature_and_roundoff=True)


def integrate(mu, factor, bounds, content_bias=None, *, policy, reference_policy, order,
              deadline=None, budget=None):
    started = time.perf_counter()
    budget = budget if budget is not None else NodeBudget(reference_policy['max_total_cell_node_evaluations'])
    ledger = _Ledger(reference_policy, deadline, budget)
    available, metadata = {}, {}
    try:
        if reference_policy['orders'] != [32, 48] or order not in reference_policy['orders']:
            raise ValueError('fixed independent reference orders 32/48 required')
        mu, factor, bounds, bias, fov = validate(mu, factor, bounds, content_bias, policy)
        prepared = prepare(mu, factor, policy['h'])
        metadata = {n: v for n, v in prepared.items() if not isinstance(v, np.ndarray)}
        boxes = np.vstack((bounds, fov))
        raw, direct, error_estimate = np.zeros(len(bounds), np.float64), 0., 0.
        if prepared['structural_zero']:
            pass
        elif prepared['normalized_depth_std'] == 0:
            values = _rectangles(prepared['xy_mean'], 1., boxes, prepared, order, reference_policy, ledger)
            raw, direct = values[:-1], float(values[-1])
        else:
            m, sigma = prepared['normalized_depth_mean'], prepared['normalized_depth_std']
            positive_threshold = -m/sigma
            lower = max(-reference_policy['depth_cutoff'], positive_threshold)
            upper = reference_policy['depth_cutoff']
            if lower >= upper:
                available['partial_raw_mass'] = raw.copy()
                raise FloatingPointError('positive-depth support lies beyond the fixed reference cutoff')
            points = outer_breaks(prepared, boxes, lower, upper, reference_policy)
            ledger.work['outer_geometry_breakpoints'] = len(points)
            if len(points)+1 > reference_policy['maximum_depth_intervals']:
                raise RuntimeError('fixed reference initial depth interval budget exceeded')
            def integrand(zeta):
                ledger.depth()
                z = m + sigma*zeta
                if z <= 0:
                    if zeta <= positive_threshold:
                        return np.zeros(len(boxes), np.float64)
                    raise FloatingPointError('positive interior depth lost by cancellation/underflow')
                location = prepared['xy_mean'] + prepared['beta']*zeta
                values = _rectangles(location, z, boxes, prepared, order, reference_policy, ledger)
                available['last_conditional_mass'] = values
                weight = math.exp(-.5*zeta*zeta-LOG_SQRT_2PI-prepared['log_front_mass'])
                return values*weight
            tolerance = reference_policy['outer_relative_tolerances'][reference_policy['orders'].index(order)]
            result, outer_error, info = quad_vec(integrand, lower, upper,
                epsabs=reference_policy['outer_absolute_tolerance'], epsrel=tolerance,
                norm=reference_policy['outer_norm'], workers=reference_policy['outer_workers'],
                quadrature=reference_policy['outer_quadrature'], points=points,
                limit=reference_policy['maximum_depth_intervals'], full_output=True)
            result = np.asarray(result, np.float64)*prepared['front_mass']
            raw, direct = result[:-1], float(result[-1])
            available['partial_raw_mass'] = raw.copy()
            available['partial_raw_fov_mass'] = np.asarray([direct])
            ledger.work['outer_intervals'] = len(info.intervals)
            error_estimate = float(outer_error)*prepared['front_mass']
            if not info.success or len(info.intervals) > reference_policy['maximum_depth_intervals']:
                raise RuntimeError('independent adaptive depth quadrature failed: ' + info.message)
        available['partial_raw_mass'] = raw.copy()
        available['partial_raw_fov_mass'] = np.asarray([direct])
        if not np.isfinite(raw).all() or np.any(raw < 0) or not math.isfinite(direct) or direct < 0:
            raise FloatingPointError('invalid independent native or direct FOV mass')
        if not prepared['structural_zero'] and (np.any(raw <= 0) or direct <= 0):
            raise FloatingPointError('nonpositive mathematical-positive reference is unresolved')
        probability, floored, floor = _probabilities(raw, bounds, bias, policy['floor_density'])
        available.update(probability=probability, floored_mass=floored)
        tails = truncation_bounds(prepared, raw, bounds, bias, floor, reference_policy)
        ledger.check_time()
        normalization = abs(math.fsum(map(float, probability))-1.)
        direct_error = abs(math.fsum(map(float, raw))-direct)
        tail_passed = bool(prepared['structural_zero'] or
            (tails['raw_l1_over_reference_fov'] is not None and tails['raw_l1_over_reference_fov'] <= 1e-10
             and tails['probability_max_absolute'] <= 1e-8))
        passed = bool(tail_passed and normalization <= 2e-12 and direct_error <= 1e-8)
        result = dict(raw_mass=raw, probability=probability, floored_mass=floored,
            raw_fov_mass=direct, order=order, method='independent-SVD-homogeneous-depth-Gaussian-rectangle',
            work=dict(ledger.work), wall_seconds=time.perf_counter()-started,
            outer_error_estimate=error_estimate, quadrature_error_bound_available=False,
            tail_truncation_used=not prepared['structural_zero'], truncation_bounds=tails,
            passed=passed, status='complete' if passed else 'failed',
            normalization_error=normalization, direct_fov_additivity_error=direct_error, **metadata)
        if not passed:
            result['failure_reason'] = 'independent tail, normalization or direct-FOV gate failed; arrays retained'
        return result
    except Exception as error:
        raise ReferenceFailure(error, ledger.work, available, metadata) from error
