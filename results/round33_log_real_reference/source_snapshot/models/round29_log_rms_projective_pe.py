"""R29 same RMS law with canonical log native-cell mass.

One declared XY Gaussian regularizer precedes perspective projection. This is
not fixed screen Gaussian convolution. GL error bounds below exclude every
floating-point density/linear-algebra error and do not certify relative error.
"""
from __future__ import annotations

from functools import lru_cache
import math
import time

import numpy as np
from scipy.special import erfcx, log_ndtr, ndtr


class NumericalFailure(RuntimeError):
    pass


@lru_cache(None)
def _legendre():
    return np.polynomial.legendre.leggauss(24)


def _array(value, shape, name):
    value = np.asarray(value)
    if value.dtype != np.float64 or value.shape != shape or not np.isfinite(value).all():
        raise ValueError('finite float64 '+name+' with prescribed shape required')
    return value


def validate(mu, factor, bounds, content_bias, policy):
    mu = _array(mu, (3,), 'mean')
    factor = _array(factor, (3, 3), 'Gaussian factor')
    bounds = np.asarray(bounds)
    if bounds.ndim != 2 or bounds.shape[1:] != (4,) or not len(bounds):
        raise ValueError('nonempty (N,4) native bounds required')
    bounds = _array(bounds, bounds.shape, 'bounds')
    bias = np.zeros(len(bounds), np.float64) if content_bias is None else _array(content_bias, (len(bounds),), 'content bias')
    fixed = dict(quadrature_order=24, maximum_tile_side_in_h=2.,
        negative_tail_threshold=10., negative_tail_series_terms=20,
        max_negative_tail_relative_remainder=1e-10,
        projection_regularization='rms_depth_xy_gaussian')
    if any(policy.get(k) != v for k, v in fixed.items()):
        raise ValueError('fixed RMS-depth/GL24/2h/tail policy required')
    h, floor = float(policy['h']), float(policy['floor_density'])
    if not (math.isfinite(h) and h > 0 and math.isfinite(floor) and floor > 0):
        raise ValueError('positive finite h and area floor required')
    if not math.isfinite(h*h) or h*h == 0:
        raise ValueError('positive finite representable squared bandwidth required')
    budget = policy['max_cell_node_evaluations']
    if type(budget) is not int or budget <= 0:
        raise ValueError('positive integer cell-node budget required')
    sides = bounds[:, 2:]-bounds[:, :2]
    areas = np.prod(sides, axis=1)
    if not np.isfinite(sides).all() or not np.isfinite(areas).all() or (sides <= 0).any() or (areas <= 0).any():
        raise ValueError('finite positive native rectangle sides and areas required')
    fov = np.r_[bounds[:, :2].min(0), bounds[:, 2:].max(0)]
    if not math.isfinite(float(areas.sum())) or not math.isfinite(float(np.prod(fov[2:]-fov[:2]))):
        raise ValueError('finite representable total area required')
    overlap = np.minimum(bounds[:, None, 2:], bounds[None, :, 2:])-np.maximum(bounds[:, None, :2], bounds[None, :, :2])
    duplicated = (overlap > 0).all(-1)
    np.fill_diagonal(duplicated, False)
    if duplicated.any() or abs(float(areas.sum())-float(np.prod(fov[2:]-fov[:2]))) > 1e-12*max(1., float(areas.sum())):
        raise ValueError('native rectangles must tile one rectangular FOV')
    return dict(mu=mu, factor=factor, bounds=bounds, bias=bias, areas=areas,
                fov=fov, h=h, floor_density=floor)


def _positive_qr_factor(columns):
    """Covariance factor from rectangular columns; no Schur subtraction."""
    if not np.isfinite(columns).all():
        raise NumericalFailure('nonfinite conditional factor columns')
    _, upper = np.linalg.qr(columns.T, mode='reduced')
    diagonal = np.diag(upper)
    if not np.isfinite(upper).all() or (diagonal == 0).any():
        raise NumericalFailure('conditional QR factor is not representable')
    upper = np.where(diagonal < 0, -1., 1.)[:, None]*upper
    return upper.T.copy()


def prepare(mu, factor, bounds, content_bias, policy):
    return _prepare_validated(validate(mu, factor, bounds, content_bias, policy))


def _prepare_validated(p):
    source_z = p['factor'][2]
    rms = math.hypot(float(p['mu'][2]), *map(float, source_z))
    p.update(rms_depth=rms, normalized_mean=None, normalized_factor=None,
             normalized_depth_mean=None, normalized_depth_std=None)
    if not math.isfinite(rms):
        raise NumericalFailure('RMS depth exceeds representable range')
    if rms == 0:
        if p['mu'][2] != 0 or np.any(source_z != 0):
            raise NumericalFailure('nonzero depth RMS underflowed')
        p.update(structural_branch='zero_depth_rms', structural_zero=True, front_mass=0.)
        return p
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        mean, normalized = p['mu']/rms, p['factor']/rms
    if not np.isfinite(mean).all() or not np.isfinite(normalized).all():
        raise NumericalFailure('RMS-normalized input is not finite')
    sigma = math.hypot(*map(float, normalized[2]))
    p.update(normalized_mean=mean, normalized_factor=normalized,
             normalized_depth_mean=float(mean[2]), normalized_depth_std=sigma)
    if not np.any(source_z != 0):
        if sigma != 0:
            raise NumericalFailure('deterministic-depth identity changed')
        if mean[2] <= 0:
            p.update(structural_branch='nonpositive_constant_depth', structural_zero=True, front_mass=0.)
            return p
        p.update(structural_branch='positive_constant_depth_gaussian', structural_zero=False,
            front_mass=1., beta=np.zeros(2, np.float64),
            conditional_factor=_positive_qr_factor(np.column_stack((normalized[:2], p['h']*np.eye(2)))))
        return p
    if sigma == 0 or not math.isfinite(sigma):
        raise NumericalFailure('nonzero depth variation lost in RMS normalization')
    axis = normalized[2]/sigma
    # QR provides an orthogonal complement to the declared latent depth axis.
    basis, _ = np.linalg.qr(axis[:, None], mode='complete')
    beta = normalized[:2] @ axis
    residual = normalized[:2] @ basis[:, 1:]
    conditional = _positive_qr_factor(np.column_stack((residual, p['h']*np.eye(2))))
    with np.errstate(over='ignore', divide='ignore'):
        depth_ratio = float(np.float64(mean[2])/np.float64(sigma))
    front = float(ndtr(depth_ratio))
    if not math.isfinite(front) or front <= 0:
        raise NumericalFailure('strictly positive front mass is not representable')
    p.update(structural_branch='variable_depth', structural_zero=False, front_mass=front,
        beta=beta, conditional_factor=conditional,
        conditional_basis_orthogonality_error=float(np.max(np.abs(basis.T@basis-np.eye(3)))))
    return p


def _solve_lower(factor, value):
    x = value[..., 0]/factor[0, 0]
    y = (value[..., 1]-factor[1, 0]*x)/factor[1, 1]
    return np.stack((x, y), axis=-1)


def negative_log_k(q, policy):
    """Fixed erfcx/20-term branches; series bound is exact-arithmetic only."""
    q = np.asarray(q, np.float64)
    if not np.isfinite(q).all() or (q < 0).any():
        raise NumericalFailure('finite nonnegative negative-tail argument required')
    logs = np.empty_like(q)
    series = q >= policy['negative_tail_threshold']
    moderate = ~series
    if moderate.any():
        z = q[moderate]
        value = (1.+z*z)*math.sqrt(math.pi/2.)*erfcx(z/math.sqrt(2.))-z
        if not np.isfinite(value).all() or (value <= 0).any():
            raise NumericalFailure('erfcx negative-tail difference is not positive finite')
        logs[moderate] = np.log(value)
    max_log_bound, lower_min, upper_max = None, None, None
    if series.any():
        z = q[series]
        inverse_square = (1./z)**2
        term = np.full_like(z, 2.)
        lower = term.copy()
        for k in range(1, 20):
            term *= ((2*k+2)*(2*k+1)/(2.*k))*inverse_square
            lower += term if k % 2 == 0 else -term
        if not np.isfinite(lower).all() or (lower <= 0).any():
            raise NumericalFailure('negative-tail alternating lower bound is not positive')
        # For 20 terms (k=0..19), the positive omitted k=20 term bounds
        # the mathematical remainder; work in logs to retain tiny bounds.
        log_coefficient = math.lgamma(43)-20.*math.log(2.)-math.lgamma(21)
        log_relative = log_coefficient-40.*np.log(z)-np.log(lower)
        if not np.isfinite(log_relative).all() or (log_relative > math.log(policy['max_negative_tail_relative_remainder'])).any():
            raise NumericalFailure('negative-tail series relative remainder exceeds policy')
        max_log_bound = float(np.max(log_relative))
        with np.errstate(under='ignore'):
            scaled_upper = lower+np.exp(log_coefficient-40.*np.log(z))
        lower_min, upper_max = float(lower.min()), float(scaled_upper.max())
        logs[series] = -3.*np.log(z)+np.log(lower)
    return logs, dict(erfcx_nodes=int(moderate.sum()), series_nodes=int(series.sum()),
        maximum_log_relative_remainder=max_log_bound,
        minimum_scaled_series_lower_bound=lower_min, maximum_scaled_series_upper_bound=upper_max,
        interval_scope='K(q)*q^3; mathematical series interval, not a floating-point enclosure')


def log_density(points, p, policy, diagnostics):
    """Evaluate the declared unconditioned law, not divided by front mass."""
    y = np.asarray(points, np.float64)
    if y.ndim != 2 or y.shape[1] != 2 or not np.isfinite(y).all():
        raise NumericalFailure('finite two-dimensional density nodes required')
    if p['structural_zero']:
        return np.full(len(y), -np.inf, np.float64)
    mean, factor = p['normalized_mean'], p['conditional_factor']
    m, sigma = p['normalized_depth_mean'], p['normalized_depth_std']
    logdet = math.log(float(factor[0, 0]))+math.log(float(factor[1, 1]))
    with np.errstate(over='ignore', under='ignore', invalid='ignore', divide='ignore'):
        if p['structural_branch'] == 'positive_constant_depth_gaussian':
            whitened = _solve_lower(factor, m*y-mean[:2])
            radius = np.sum(whitened*whitened, axis=1)
            log_density = -math.log(2.*math.pi)-logdet+2.*math.log(m)-.5*radius
            diagnostics['constant_depth_nodes'] += len(y)
        else:
            beta = p['beta']
            v = _solve_lower(factor, sigma*y-beta)
            u = _solve_lower(factor, sigma*mean[:2]-m*beta)
            root_d = np.hypot(1., np.hypot(v[:, 0], v[:, 1]))
            w = np.sum((v/root_d[:, None])*u, axis=1)+m/root_d
            d = m*y-mean[:2]
            perpendicular = _solve_lower(factor, d)/root_d[:, None]
            determinant = sigma*(y[:, 0]*mean[1]-y[:, 1]*mean[0])+beta[0]*d[:, 1]-beta[1]*d[:, 0]
            third = ((determinant/factor[0, 0])/factor[1, 1])/root_d
            delta = np.sum(perpendicular*perpendicular, axis=1)+third*third
            if not all(np.isfinite(a).all() for a in (v, u, root_d, w, delta)):
                raise NumericalFailure('conditional density geometry is not representable')
            log_density = np.empty(len(y), np.float64)
            positive = w >= 0
            if positive.any():
                wp = w[positive]
                t = wp/sigma
                log_w = np.where(wp > 0, np.log(wp), -np.inf)
                first = np.logaddexp(2.*math.log(sigma), 2.*log_w)+log_ndtr(t)
                second = math.log(sigma)+log_w-.5*t*t-.5*math.log(2.*math.pi)
                numerator = np.logaddexp(first, second)
                log_density[positive] = -.5*delta[positive]+numerator-math.log(2.*math.pi)-logdet-3.*np.log(root_d[positive])
                diagnostics['positive_branch_nodes'] += int(positive.sum())
            if (~positive).any():
                q = -w[~positive]/sigma
                log_k, report = negative_log_k(q, policy)
                c = (math.hypot(float(u[0]), float(u[1]), m)/sigma)**2
                log_density[~positive] = 2.*math.log(sigma)-.5*c+log_k-1.5*math.log(2.*math.pi)-logdet-3.*np.log(root_d[~positive])
                diagnostics['negative_branch_nodes'] += int((~positive).sum())
                diagnostics['negative_erfcx_nodes'] += report['erfcx_nodes']
                diagnostics['negative_series_nodes'] += report['series_nodes']
                bound = report['maximum_log_relative_remainder']
                old = diagnostics['maximum_log_negative_tail_relative_remainder']
                if bound is not None:
                    diagnostics['maximum_log_negative_tail_relative_remainder'] = bound if old is None else max(bound, old)
                    lower = report['minimum_scaled_series_lower_bound']
                    upper = report['maximum_scaled_series_upper_bound']
                    old_lower = diagnostics['minimum_scaled_negative_tail_series_lower_bound']
                    old_upper = diagnostics['maximum_scaled_negative_tail_series_upper_bound']
                    diagnostics['minimum_scaled_negative_tail_series_lower_bound'] = lower if old_lower is None else min(lower, old_lower)
                    diagnostics['maximum_scaled_negative_tail_series_upper_bound'] = upper if old_upper is None else max(upper, old_upper)
    if not np.isfinite(log_density).all():
        raise NumericalFailure('positive log-density is nonfinite; no repair')
    high, low = float(log_density.max()), float(log_density.min())
    old_high, old_low = diagnostics['maximum_log_density'], diagnostics['minimum_log_density']
    diagnostics['maximum_log_density'] = high if old_high is None else max(high, old_high)
    diagnostics['minimum_log_density'] = low if old_low is None else min(low, old_low)
    return log_density


def integration_tiles(bounds, maximum_side, maximum_tiles):
    tiles, owners = [], []
    for owner, box in enumerate(bounds):
        counts_float = np.ceil((box[2:]-box[:2])/maximum_side)
        if not np.isfinite(counts_float).all() or (counts_float <= 0).any() or (counts_float > maximum_tiles).any():
            raise NumericalFailure('tile subdivision exceeds representable work budget')
        nx, ny = map(int, counts_float)
        if len(tiles)+nx*ny > maximum_tiles:
            raise NumericalFailure('tile allocation exceeds fixed cell-node budget')
        step = (box[2:]-box[:2])/np.asarray([nx, ny])
        for iy in range(ny):
            for ix in range(nx):
                low = box[:2]+np.asarray([ix, iy])*step
                high = box[:2]+np.asarray([ix+1, iy+1])*step
                tiles.append(np.r_[low, high]); owners.append(owner)
    return np.asarray(tiles, np.float64), np.asarray(owners, np.int64)


LSE_SCOPES = ('tile_nodes', 'cell_tiles', 'fov_tiles', 'native_total',
              'floor', 'floor_total', 'floored_total', 'content_normalization')


def new_lse_diagnostics():
    return dict(by_scope={name:dict(sum_count=0, underflow_sum_count=0, underflow_term_count=0,
        maximum_log_scaled_omitted_bound=None, maximum_log_absolute_omitted_bound=None,
        maximum_log_relative_omitted_bound=None) for name in LSE_SCOPES},
        exact_arithmetic_local_omission_only=True, is_pipeline_error_bound=False,
        excludes_addition_absorption_density_quadrature_and_roundoff=True,
        scope='Per-scope maxima of individual positive-sum bounds; not accumulated pipeline error. '
              'None means no relative-exp underflow occurred, not certified zero error.')


def positive_logsum(log_terms, diagnostics, scope):
    """Reduce the final axis; -inf is an explicit exact-zero/empty identity.

    No magnitude threshold is used. Only a hardware exp result of zero for
    a finite relative log term is counted. The local exact-arithmetic omitted
    scaled sum is bounded by count*exp(maximum dropped relative exponent).
    """
    terms = np.asarray(log_terms, np.float64)
    if terms.ndim == 0 or terms.shape[-1] == 0 or np.isnan(terms).any() or np.isposinf(terms).any():
        raise NumericalFailure('valid nonempty positive log-sum terms required')
    report = diagnostics['by_scope'][scope]
    report['sum_count'] += int(np.prod(terms.shape[:-1], dtype=np.int64))
    maximum = np.max(terms, axis=-1, keepdims=True)
    finite = np.isfinite(terms)
    with np.errstate(invalid='ignore', over='ignore', under='ignore', divide='ignore'):
        shifted = np.where(finite, terms-maximum, -np.inf)
        if not np.isfinite(shifted[finite]).all():
            raise NumericalFailure('finite relative log exponent is not representable')
        scaled = np.exp(shifted)
        total = np.sum(scaled, axis=-1)
        result = maximum[..., 0]+np.log(total)
    lost = finite & (scaled == 0)
    counts = np.sum(lost, axis=-1)
    affected = counts > 0
    report['underflow_term_count'] += int(np.sum(counts))
    report['underflow_sum_count'] += int(np.count_nonzero(affected))
    if np.any(affected):
        largest = np.max(np.where(lost, shifted, -np.inf), axis=-1)
        scaled_bound = np.log(counts[affected].astype(np.float64))+largest[affected]
        absolute_bound = maximum[..., 0][affected]+scaled_bound
        relative_bound = scaled_bound-np.log(total[affected])
        if not all(np.isfinite(v).all() for v in (scaled_bound, absolute_bound, relative_bound)):
            raise NumericalFailure('local log omitted-mass bound is not representable')
        for key,values in (('maximum_log_scaled_omitted_bound',scaled_bound),
                           ('maximum_log_absolute_omitted_bound',absolute_bound),
                           ('maximum_log_relative_omitted_bound',relative_bound)):
            value, previous = float(np.max(values)), report[key]
            report[key] = value if previous is None else max(previous,value)
    if np.isnan(result).any() or np.isposinf(result).any():
        raise NumericalFailure('positive log-sum is not representable')
    return result


def _integrate_log_tiles(tiles, owners, p, policy, deadline, work, density_report,
                        lse_report, result, accumulated, scope):
    nodes, weights = _legendre()
    # Exactly the old positive tensor weights and physical GL nodes.
    log_weight = np.log(weights[:, None]*weights[None, :]).reshape(1,-1)
    for start in range(0,len(tiles),32):
        if deadline is not None and time.perf_counter() >= deadline:
            raise NumericalFailure('production phase wall budget exceeded')
        batch = tiles[start:start+32]
        charged = len(batch)*24*24
        if work['cell_node_evaluations']+charged > policy['max_cell_node_evaluations']:
            raise NumericalFailure('cell-node evaluation budget exceeded')
        work['cell_node_evaluations'] += charged
        middle, half = (batch[:,:2]+batch[:,2:])/2., (batch[:,2:]-batch[:,:2])/2.
        x = middle[:,0,None]+half[:,0,None]*nodes
        y = middle[:,1,None]+half[:,1,None]*nodes
        points = np.stack(np.broadcast_arrays(x[:,:,None],y[:,None,:]),axis=-1).reshape(-1,2)
        values = log_density(points,p,policy,density_report).reshape(len(batch),24*24)
        jacobian = np.prod(half,axis=1)
        if not np.isfinite(jacobian).all() or (jacobian<=0).any():
            raise NumericalFailure('positive tile Jacobian is not representable')
        log_mass = positive_logsum(values+log_weight,lse_report,'tile_nodes')+np.log(jacobian)
        if not np.isfinite(log_mass).all():
            raise NumericalFailure('positive tile log mass is not finite')
        for owner,value in zip(owners[start:start+len(batch)],log_mass):
            result[owner] = positive_logsum(np.asarray([result[owner],value]),lse_report,
                'cell_tiles' if scope=='native' else 'fov_tiles')
            accumulated[owner] += 1
            work[scope+'_tiles_completed'] += 1


def materialize_raw(log_raw, structural_zero):
    """Optional indexed display values; canonical comparisons use log mass."""
    logs = np.asarray(log_raw,np.float64)
    if structural_zero:
        if not np.isneginf(logs).all():
            raise NumericalFailure('structural zero must retain exact zero log representation')
        return np.arange(len(logs),dtype=np.int64),np.zeros(len(logs),np.float64),0
    if not np.isfinite(logs).all():
        raise NumericalFailure('positive canonical raw log mass must be finite')
    with np.errstate(over='ignore',under='ignore',invalid='ignore'):
        ordinary = np.exp(logs)
    if not np.isfinite(ordinary).all():
        raise NumericalFailure('ordinary raw materialization overflow')
    indices = np.flatnonzero(ordinary>0).astype(np.int64)
    values = ordinary[indices]
    subnormal = int(np.count_nonzero(values < np.finfo(np.float64).tiny))
    return indices,values,subnormal


def integrate(mu, factor, bounds, content_bias=None, *, policy, deadline=None):
    p0 = validate(mu,factor,bounds,content_bias,policy)
    count = len(p0['bounds'])
    raw, direct = np.full(count,-np.inf,np.float64),np.full(1,-np.inf,np.float64)
    accumulated, accumulated_fov = np.zeros(count,np.int64),np.zeros(1,np.int64)
    completed, completed_fov = np.zeros(count,bool),np.zeros(1,bool)
    work = dict(cell_node_evaluations=0,native_tiles_completed=0,fov_tiles_completed=0)
    diagnostics = dict(constant_depth_nodes=0,positive_branch_nodes=0,negative_branch_nodes=0,
        negative_erfcx_nodes=0,negative_series_nodes=0,maximum_log_density=None,minimum_log_density=None,
        maximum_log_negative_tail_relative_remainder=None,
        minimum_scaled_negative_tail_series_lower_bound=None,
        maximum_scaled_negative_tail_series_upper_bound=None,
        series_interval_scope='K(q)*q^3; mathematical series interval, not a floating-point enclosure')
    lse = new_lse_diagnostics()
    common = dict(status='failed',passed=False,work=work,density_diagnostics=diagnostics,lse_diagnostics=lse,
        h=p0['h'],floor_density=p0['floor_density'],quadrature_order=24,maximum_tile_side_in_h=2.,
        projection_regularization='rms_depth_xy_gaussian',canonical_raw_representation='log_mass',
        ordinary_materialization_is_canonical=False,quadrature_error_bound_available=False,
        exact_arithmetic_gl_bound_available=True,tail_truncation_used=False,
        independent_numerical_verification_required=True)
    totals, total_fov = None,None
    try:
        p = _prepare_validated(p0)
        common.update(rms_depth=p['rms_depth'],normalized_depth_mean=p['normalized_depth_mean'],
            normalized_depth_std=p['normalized_depth_std'],front_mass=p['front_mass'],
            structural_branch=p['structural_branch'],structural_zero=p['structural_zero'])
        if deadline is not None and time.perf_counter() >= deadline:
            raise NumericalFailure('production phase wall budget exceeded')
        area = float(np.sum(p['areas']))
        bound = 0. if p['structural_zero'] else area/(math.pi*p['h']**2*math.comb(48,24))
        if not p['structural_zero'] and (not math.isfinite(bound) or bound<=0):
            raise NumericalFailure('positive analytic quadrature bound is not representable')
        common['analytic_truncation_bound'] = dict(raw_native_l1_absolute=bound,raw_direct_fov_absolute=bound,
            exact_arithmetic_only=True,excludes_density_evaluation_and_roundoff=True,
            is_relative_error_certificate=False,source='docs/round26_rms_quadrature_derivation.md')
        if p['structural_zero']:
            common.update(native_tile_count=0,fov_tile_count=0)
            completed[:],completed_fov[:] = True,True
        else:
            maximum_side=policy['maximum_tile_side_in_h']*p['h']
            maximum_tiles=policy['max_cell_node_evaluations']//(24*24)
            tiles,owners=integration_tiles(p['bounds'],maximum_side,maximum_tiles)
            fov_tiles,fov_owners=integration_tiles(p['fov'][None],maximum_side,maximum_tiles)
            totals=np.bincount(owners,minlength=count);total_fov=np.asarray([len(fov_tiles)],np.int64)
            common.update(native_tile_count=len(tiles),fov_tile_count=len(fov_tiles))
            _integrate_log_tiles(tiles,owners,p,policy,deadline,work,diagnostics,lse,raw,accumulated,'native')
            completed[:] = accumulated==totals
            _integrate_log_tiles(fov_tiles,fov_owners,p,policy,deadline,work,diagnostics,lse,direct,accumulated_fov,'fov')
            completed_fov[:] = accumulated_fov==total_fov
            if not np.isfinite(raw).all() or not np.isfinite(direct).all():
                raise NumericalFailure('positive completed native/direct log mass is not finite')
        # Keep the original ordinary floor-product admission boundary.
        floor = p['floor_density']*p['areas']
        if not np.isfinite(floor).all() or (floor<=0).any():
            raise NumericalFailure('positive native area floor is not representable')
        log_floor=np.log(floor)
        log_floored=positive_logsum(np.column_stack((raw,log_floor)),lse,'floor')
        log_raw_sum=float(positive_logsum(raw,lse,'native_total'))
        log_floor_sum=float(positive_logsum(log_floor,lse,'floor_total'))
        log_normalizer=float(positive_logsum(log_floored,lse,'floored_total'))
        logits=log_floored+p['bias']
        log_content=float(positive_logsum(logits,lse,'content_normalization'))
        log_probability=logits-log_content
        with np.errstate(over='ignore',under='ignore',invalid='ignore'):
            probability=np.exp(log_probability)
        if (not np.isfinite(log_floored).all() or not np.isfinite(log_probability).all()
                or not np.isfinite(probability).all() or np.any(probability<0)):
            raise NumericalFailure('invalid log-floor/native probability normalization')
        indices,ordinary,subnormal=materialize_raw(raw,p['structural_zero'])
        common.update(status='complete',passed=True,log_raw_mass=raw,log_raw_fov_mass=direct,
            log_floored_mass=log_floored,log_probability=log_probability,probability=probability,
            materialized_raw_indices=indices,materialized_raw_mass=ordinary,
            log_raw_mass_sum=None if p['structural_zero'] else log_raw_sum,
            log_normalizer=log_normalizer,log_content_normalizer=log_content,log_total_area_floor=log_floor_sum,
            ordinary_raw_unmaterialized_count=count-len(indices),ordinary_raw_subnormal_count=subnormal,
            zero_raw_cells=count if p['structural_zero'] else 0,all_raw_mass_zero=p['structural_zero'],
            area_prior_used=bool(p['structural_zero'] and np.all(p['bias']==p['bias'][0])),
            raw_mass_below_total_floor=bool(log_raw_sum<log_floor_sum),
            max_normalization_error=abs(float(np.sum(probability))-1.),argmax_index=int(np.argmax(probability)),
            argmax_exact_ties=int(np.sum(probability==np.max(probability))),
            argmax_near_ties_at_1e_12=int(np.sum(np.max(probability)-probability<=1e-12)))
    except (NumericalFailure,np.linalg.LinAlgError,FloatingPointError,OverflowError) as error:
        if totals is not None:completed[:] = accumulated==totals
        if total_fov is not None:completed_fov[:] = accumulated_fov==total_fov
        common.update(failure_reason=type(error).__name__+': '+str(error),
            partial_log_raw_mass=raw,partial_log_raw_fov_mass=direct,
            partial_completed_cells=completed,partial_completed_fov=completed_fov,
            partial_accumulated_tiles=accumulated,partial_accumulated_fov_tiles=accumulated_fov)
    return common


