"""R34 independent full-finite-X Gaussian conditioning in canonical log mass.

SVD preparation and physical depth geometry derive from the independent R27
reference. Inner X has no probability cutoff. A fixed initial-midpoint pilot
sets per-component outer scales once; its maximum is not an upper bound.
Only narrow-normal GL16 is changed to GL8 relative to R32; the exponent
span .25, initial partition coalescing and all other mathematics remain unchanged.
No production preparation, projected density or normalization is imported.
NumPy/SciPy elementary operations are common dependencies. Tail and local-loss
bounds below exclude quadrature, input error and floating-point roundoff.
"""
from __future__ import annotations
from functools import lru_cache
import math
import time
import numpy as np
from scipy.integrate import quad_vec
from scipy.special import erf,log_ndtr,logsumexp
LOG_SQRT_2PI=.5*math.log(2*math.pi)
LSE_SCOPES=('narrow_normal_nodes','inner_panel_nodes','rectangle_panels','native_total',
            'floor','floor_total','floored_total','content_normalization')

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
    def __init__(self,policy,deadline,budget):
        self.policy,self.deadline,self.budget=policy,deadline,budget
        self.work=dict(depth_evaluations=0,pilot_depth_evaluations=0,quadrature_depth_evaluations=0,
            inner_x_nodes=0,narrow_y_pdf_nodes=0,elementary_inner_nodes=0,cell_node_evaluations=0,
            inner_panels=0,outer_intervals=0,outer_geometry_breakpoints=0)
        self.maximum_full_x_panel_width=0.
        self.normal_logcdf_scalar_evaluations=0
        self.loss=dict(by_scope={name:dict(sum_count=0,underflow_sum_count=0,underflow_term_count=0,
            maximum_log_scaled_omitted_bound=None,maximum_log_absolute_omitted_bound=None,
            maximum_log_relative_omitted_bound=None) for name in LSE_SCOPES},
            exact_arithmetic_local_omission_only=True,is_pipeline_error_bound=False,
            excludes_addition_absorption_density_quadrature_and_roundoff=True)

    def check_time(self):
        if self.deadline is not None and time.perf_counter()>=self.deadline:
            raise RuntimeError('fixed independent reference phase deadline exceeded')

    def depth(self,pilot=False):
        self.check_time()
        if self.work['depth_evaluations']>=self.policy['maximum_depth_evaluations']:
            raise RuntimeError('fixed independent depth-evaluation budget exceeded')
        self.work['depth_evaluations']+=1
        self.work['pilot_depth_evaluations' if pilot else 'quadrature_depth_evaluations']+=1

    def allocation(self,count):
        self.check_time()
        if (type(count) is not int or count<0 or count>self.budget.remaining_nodes
                or self.work['elementary_inner_nodes']+count>self.policy['max_call_cell_node_evaluations']):
            raise RuntimeError('full-X partition exceeds remaining fixed inner-node budget before allocation')

    def nodes(self,count,kind):
        self.allocation(count);self.budget.charge(count)
        self.work['elementary_inner_nodes']+=count;self.work['cell_node_evaluations']+=count
        self.work[kind]+=count


def _record_losses(terms,owners,maxima,scaled,totals,ledger,scope):
    report=ledger.loss['by_scope'][scope];groups=len(maxima)
    report['sum_count']+=groups
    finite=np.isfinite(terms);lost=finite&(scaled==0)
    counts=np.bincount(owners,weights=lost.astype(np.int64),minlength=groups).astype(np.int64)
    affected=counts>0
    report['underflow_sum_count']+=int(affected.sum());report['underflow_term_count']+=int(counts.sum())
    if np.any(affected):
        relative=terms-maxima[owners]
        largest=np.full(groups,-np.inf);np.maximum.at(largest,owners[lost],relative[lost])
        bound=np.log(counts[affected])+largest[affected]
        values=(bound,maxima[affected]+bound,bound-np.log(totals[affected]))
        for key,value in zip(('maximum_log_scaled_omitted_bound','maximum_log_absolute_omitted_bound','maximum_log_relative_omitted_bound'),values):
            if not np.isfinite(value).all():raise FloatingPointError('local omitted-term log bound is nonfinite')
            high=float(value.max());old=report[key];report[key]=high if old is None else max(old,high)


def _log_group_sum(terms,owners,count,ledger,scope):
    terms=np.asarray(terms,np.float64);owners=np.asarray(owners,np.int64)
    if terms.ndim!=1 or owners.shape!=terms.shape or np.isnan(terms).any() or np.isposinf(terms).any():
        raise FloatingPointError('invalid independent positive log-sum inputs')
    maxima=np.full(count,-np.inf);np.maximum.at(maxima,owners,terms)
    finite=np.isfinite(terms)
    with np.errstate(invalid='ignore',under='ignore'):
        shifted=np.where(finite,terms-maxima[owners],-np.inf);scaled=np.exp(shifted)
    if not np.isfinite(shifted[finite]).all():raise FloatingPointError('relative log exponent is not representable')
    totals=np.bincount(owners,weights=scaled,minlength=count)
    with np.errstate(divide='ignore'):result=maxima+np.log(totals)
    _record_losses(terms,owners,maxima,scaled,totals,ledger,scope)
    if np.isnan(result).any() or np.isposinf(result).any():raise FloatingPointError('invalid positive log-sum result')
    return result


def _log_sum(terms,ledger,scope):
    terms=np.asarray(terms,np.float64)
    if terms.ndim==1:return float(_log_group_sum(terms,np.zeros(len(terms),np.int64),1,ledger,scope)[0])
    if terms.ndim!=2 or not terms.shape[1]:raise ValueError('nonempty one/two-dimensional positive sum required')
    return _log_group_sum(terms.ravel(),np.repeat(np.arange(len(terms)),terms.shape[1]),len(terms),ledger,scope)


def normal_log_interval(midpoint,halfwidth,*,reference_policy,ledger):
    """Positive finite interval via narrow GL8 or reflected log-CDF difference."""
    mid,half=np.broadcast_arrays(np.asarray(midpoint,np.float64),np.asarray(halfwidth,np.float64))
    if not np.isfinite(mid).all() or not np.isfinite(half).all() or np.any(half<=0):
        raise FloatingPointError('normal interval requires finite midpoint and positive halfwidth')
    result=np.empty(mid.shape,np.float64)
    with np.errstate(over='ignore',invalid='ignore'):
        narrow=half*(np.abs(mid)+half)<=reference_policy['narrow_cdf_exponent_span']
    if np.any(narrow):
        ledger.nodes(int(narrow.sum())*8,'narrow_y_pdf_nodes')
        nodes,weights=_legendre(8);m=mid[narrow,None];delta=half[narrow,None]*nodes
        with np.errstate(over='ignore',invalid='ignore'):
            logs=-.5*m*m-m*delta-.5*delta*delta-LOG_SQRT_2PI+np.log(weights)[None]
        if not np.isfinite(logs).all():raise FloatingPointError('narrow normal log density is nonfinite')
        result[narrow]=np.log(half[narrow])+_log_sum(logs,ledger,'narrow_normal_nodes')
    broad=~narrow
    if np.any(broad):
        lo,hi=mid[broad]-half[broad],mid[broad]+half[broad]
        if not np.isfinite(lo).all() or not np.isfinite(hi).all() or np.any(lo>=hi):
            raise FloatingPointError('broad positive normal interval endpoints are unresolved')
        negative,positive=hi<=0,lo>=0;crossing=~(negative|positive);value=np.empty(lo.shape)
        for mask,upper_arg,lower_arg in ((negative,hi,lo),(positive,-lo,-hi)):
            if np.any(mask):
                ledger.normal_logcdf_scalar_evaluations+=2*int(mask.sum())
                upper,lower=log_ndtr(upper_arg[mask]),log_ndtr(lower_arg[mask]);difference=lower-upper
                if np.any(difference>=0) or not np.isfinite(upper).all() or not np.isfinite(lower).all():
                    raise FloatingPointError('positive normal log-CDF difference is unresolved')
                value[mask]=upper+np.log(-np.expm1(difference))
        if np.any(crossing):
            probability=.5*(erf(hi[crossing]/math.sqrt(2))-erf(lo[crossing]/math.sqrt(2)))
            value[crossing]=np.log(probability)
        result[broad]=value
    if not np.isfinite(result).all():raise FloatingPointError('positive normal interval log mass is nonfinite')
    return result


def _rectangles(location,z,boxes,prepared,order,reference_policy,ledger):
    """Ragged full-X global-width2 panels and original conditional-Y roots."""
    sx,ky,sy=prepared['sx'],prepared['conditional_y_coefficient'],prepared['conditional_y_residual_std']
    if not math.isfinite(z) or z<=0:raise FloatingPointError('strict positive homogeneous depth required')
    xmid=(z*((boxes[:,0]+boxes[:,2])/2)-location[0])/sx
    xhalf=z*((boxes[:,2]-boxes[:,0])/2)/sx
    low,high=xmid-xhalf,xmid+xhalf;width=reference_policy['standard_x_panel_width']
    if (not np.isfinite(xmid).all() or not np.isfinite(xhalf).all() or np.any(xhalf<=0)
            or not np.isfinite(low).all() or not np.isfinite(high).all() or np.any(high<low)):
        raise FloatingPointError('positive full-X interval is not representable')
    regular=high>low;collapsed=~regular
    if np.any(collapsed&(2*xhalf>width)):
        raise FloatingPointError('rounded full-X interval cannot represent width2 subdivision')
    first=[];counts=[]
    for a,b,is_regular in zip(low,high,regular):
        start=math.floor(float(a)/width)+1 if is_regular else 0
        stop=math.ceil(float(b)/width)-1 if is_regular else -1
        first.append(start);counts.append(max(0,stop-start+1))
    minimum_panels=sum(counts)+len(boxes)
    ledger.allocation(int(minimum_panels*order))
    if any(c and np.spacing(max(abs(float(a)),abs(float(b))))>width for a,b,c in zip(low,high,counts)):
        raise FloatingPointError('global width2 grid spacing is not representable')
    count_array=np.asarray(counts,np.int64);grid_owners=np.repeat(np.arange(len(boxes)),count_array)
    if len(grid_owners):
        offsets=np.arange(len(grid_owners))-np.repeat(np.cumsum(count_array)-count_array,count_array)
        grid=np.asarray([float(first[int(owner)]+int(offset))*width for owner,offset in zip(grid_owners,offsets)],np.float64)
    else:grid=np.empty(0,np.float64)
    owner_parts=[np.flatnonzero(regular),np.flatnonzero(regular),grid_owners]
    point_parts=[low[regular],high[regular],grid]
    if ky!=0:
        for axis in (1,3):
            for c in reference_policy['inner_geometry_breaks']:
                roots=(z*boxes[:,axis]-location[1]-c*sy)/ky
                active=regular&np.isfinite(roots)&(low<roots)&(roots<high)
                owner_parts.append(np.flatnonzero(active));point_parts.append(roots[active])
    owners=np.concatenate(owner_parts);points=np.concatenate(point_parts)
    # The full-X grid was bounded before allocation. Duplicate geometry roots
    # are discarded before charging the exact panel nodes below.
    ledger.check_time()
    permutation=np.lexsort((points,owners));owners,points=owners[permutation],points[permutation]
    valid=(owners[1:]==owners[:-1])&(points[1:]>points[:-1])
    panel_owners=owners[:-1][valid];a,b=points[:-1][valid],points[1:][valid]
    middle,half=(a+b)/2,(b-a)/2
    nper=np.bincount(panel_owners,minlength=len(boxes));single=nper[panel_owners]==1
    middle[single],half[single]=xmid[panel_owners[single]],xhalf[panel_owners[single]]
    if np.any(collapsed):
        idx=np.flatnonzero(collapsed);panel_owners=np.r_[panel_owners,idx];middle=np.r_[middle,xmid[idx]];half=np.r_[half,xhalf[idx]]
    if not len(panel_owners) or np.any(half<=0) or np.any(2*half>width):
        raise FloatingPointError('full-X partition failed positive maximum-width contract')
    ledger.nodes(int(len(panel_owners)*order),'inner_x_nodes');ledger.work['inner_panels']+=len(panel_owners)
    ledger.maximum_full_x_panel_width=max(ledger.maximum_full_x_panel_width,float(2*half.max()))
    nodes,weights=_legendre(order);delta=half[:,None]*nodes;u=middle[:,None]+delta
    ymid=(z*((boxes[panel_owners,1]+boxes[panel_owners,3])/2)[:,None]-location[1]-ky*u)/sy
    yhalf=(z*((boxes[panel_owners,3]-boxes[panel_owners,1])/2)/sy)[:,None]
    logy=normal_log_interval(ymid,yhalf,reference_policy=reference_policy,ledger=ledger)
    with np.errstate(over='ignore',invalid='ignore'):
        logphi=-.5*middle[:,None]**2-middle[:,None]*delta-.5*delta**2-LOG_SQRT_2PI
    if not np.isfinite(logphi).all():raise FloatingPointError('full-X normal log density is nonfinite')
    panel=np.log(half)+_log_sum(np.log(weights)[None]+logphi+logy,ledger,'inner_panel_nodes')
    result=_log_group_sum(panel,panel_owners,len(boxes),ledger,'rectangle_panels')
    if not np.isfinite(result).all():raise FloatingPointError('positive full-X rectangle log mass is unresolved')
    return result

def _legendre(order):
    if order not in (8, 32, 48):
        raise ValueError('only frozen independent GL8/32/48 orders are allowed')
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


def coalesce_depth_edges(original_edges,*,available=None,metadata=None):
    """Remove only unrepresentable-midpoint partition hints, never domain.

    The exact R31 midpoint expression is used. Evidence is updated during the
    walk, including before a whole-domain failure. The two physical endpoints
    are always selected from their original array, without reconstruction.
    """
    available={} if available is None else available
    metadata={} if metadata is None else metadata
    original=np.asarray(original_edges)
    available['original_depth_edges']=original.copy()
    dropped=[];kept=[]
    def publish():
        indices=kept+([len(original)-1] if original.ndim==1 and len(original)>=2 else [])
        retained=original[np.asarray(indices,np.int64)].copy() if original.ndim==1 else np.empty(0,np.float64)
        available['initial_depth_edges']=retained
        available['dropped_depth_edge_indices']=np.asarray([r['original_index'] for r in dropped],np.int64)
        available['dropped_depth_edge_values']=np.asarray([r['value'] for r in dropped],np.float64)
        metadata['depth_edge_coalescing']=dict(rule='left_to_right_strict_midpoint_then_upper_backtrack',
            original_interior_count=max(0,len(original)-2) if original.ndim else 0,
            retained_interior_count=max(0,len(retained)-2),dropped_count=len(dropped),dropped=list(dropped))
        return retained
    publish()
    if (original.dtype!=np.float64 or original.ndim!=1 or len(original)<2
            or not np.isfinite(original).all() or np.any(np.diff(original)<=0)):
        raise ValueError('original depth edges must be finite float64, strictly sorted and unique with interior cuts inside endpoints')
    kept.append(0);publish()
    def strict(a,b):
        midpoint=(float(a)+float(b))/2.
        return a<midpoint<b
    for index in range(1,len(original)-1):
        if strict(original[kept[-1]],original[index]):kept.append(index)
        else:dropped.append(dict(original_index=index,value=float(original[index]),reason='no_strict_midpoint_from_previous_retained'))
        publish()
    while not strict(original[kept[-1]],original[-1]):
        if len(kept)==1:
            publish()
            raise FloatingPointError('whole physical depth domain has no strict representable midpoint; endpoints unchanged')
        index=kept.pop();dropped.append(dict(original_index=index,value=float(original[index]),reason='no_strict_midpoint_to_upper'))
        publish()
    retained=publish()
    return retained


def _log_difference(a,b):
    if b==-np.inf:return float(a)
    if a==b:return -np.inf
    if not a>b:raise FloatingPointError('positive log difference is unresolved')
    return float(a+math.log(-math.expm1(b-a)))


def _diagnostic_exp(value):
    if value==-np.inf:return 0.
    if value>math.log(np.finfo(np.float64).max):return None
    return math.exp(value)


def _json_log(value):
    return None if value==-np.inf else float(value)


def _normalize(log_raw,bounds,bias,policy,ledger):
    floor=policy['floor_density']*np.prod(bounds[:,2:]-bounds[:,:2],axis=1)
    if not np.isfinite(floor).all() or np.any(floor<=0):
        raise FloatingPointError('original positive finite ordinary area-floor product is required')
    log_floor=np.log(floor)
    log_floored=_log_sum(np.column_stack((log_raw,log_floor)),ledger,'floor')
    log_floor_sum=_log_sum(log_floor,ledger,'floor_total')
    log_normalizer=_log_sum(log_floored,ledger,'floored_total')
    logits=log_floored+bias;log_content=_log_sum(logits,ledger,'content_normalization')
    log_p=logits-log_content
    with np.errstate(under='ignore'):p=np.exp(log_p)
    if not np.isfinite(log_floored).all() or not np.isfinite(log_p).all() or not np.isfinite(p).all():
        raise FloatingPointError('independent log floor or probability is nonfinite')
    return dict(log_floored_mass=log_floored,log_probability=log_p,probability=p),dict(
        log_floor_mass=log_floor,log_total_area_floor=log_floor_sum,log_normalizer=log_normalizer,
        log_content_normalizer=log_content)


def truncation_bounds(prepared,log_raw,bias,log_floor,reference_policy):
    zero=prepared['structural_zero'];sigma=prepared['normalized_depth_std']
    log_tail=-np.inf
    if not zero and sigma!=0:
        cutoff=reference_policy['depth_cutoff']
        with np.errstate(over='ignore',divide='ignore'):
            threshold=float(-np.float64(prepared['normalized_depth_mean'])/np.float64(sigma))
        lower=-np.inf
        if threshold < -cutoff:lower=_log_difference(float(log_ndtr(-cutoff)),float(log_ndtr(threshold)))
        upper=float(log_ndtr(-max(cutoff,threshold)))
        log_tail=float(logsumexp([lower,upper]))
    log_fov=float(logsumexp(log_raw))
    log_ratio=None if zero else (log_tail-log_fov)
    weighted_floor=float(logsumexp(log_floor+bias-float(bias.max())))
    if not math.isfinite(weighted_floor):raise FloatingPointError('bias-aware log floor denominator is nonfinite')
    log_p=-np.inf if log_tail==-np.inf else min(0.,math.log(2.)+log_tail-weighted_floor)
    return dict(depth_omitted_mass=_diagnostic_exp(log_tail),log_depth_omitted_mass=_json_log(log_tail),
        native_raw_l1_absolute=_diagnostic_exp(log_tail),log_native_raw_l1_absolute=_json_log(log_tail),
        direct_fov_absolute=_diagnostic_exp(log_tail),log_direct_fov_absolute=_json_log(log_tail),
        raw_l1_over_reference_fov=None if zero else _diagnostic_exp(log_ratio),
        log_raw_l1_over_reference_fov=None if zero else _json_log(log_ratio),
        probability_max_absolute=_diagnostic_exp(log_p),log_probability_max_absolute=_json_log(log_p),
        native_union_bound=True,inner_x_truncation_used=False,structural_zero_exemption=zero,
        raw_relative_denominator='computed retained FOV mass, not an interval enclosure',
        excludes_quadrature_and_roundoff=True)


def narrow_normal_local_bound():
    """Local exact-arithmetic GL8 remainder on the conservative .5 envelope.

    The computed value is ordinary-float diagnostic evaluation of the design
    formula. The stated conservative ceiling is a mathematical remainder only:
    neither roundoff, adaptive outer integration nor a pipeline bound is added.
    """
    degree=16;envelope=.5
    derivative=math.exp(envelope)*math.factorial(degree)*sum(
        envelope**(degree-k)/(2**k*math.factorial(k)*math.factorial(degree-2*k))
        for k in range(9))
    coefficient=2**17*math.factorial(8)**4/(17*math.factorial(16)**3)
    relative=coefficient*derivative/(2*math.exp(-envelope))
    return dict(order=8,branch_exponent_span=.25,proof_envelope=envelope,
        computed_relative_remainder=relative,conservative_relative_remainder_bound=1.88e-13,
        exact_arithmetic_only=True,includes_floating_point_error=False,is_pipeline_error_bound=False)


def _check_policy(reference_policy,order):
    fixed=dict(orders=[32,48],standard_x_panel_width=2.,depth_cutoff=18.,narrow_cdf_order=8,
        narrow_cdf_exponent_span=.25,outer_quadrature='gk21',outer_norm='max',outer_workers=1,
        x_domain='complete_finite_cell_interval',outer_scaling='fixed_initial_interval_midpoint_pilots',
        pilot_reselection_allowed=False,initial_depth_partition='strict_midpoint_greedy_coalescing')
    if any(reference_policy.get(k)!=v for k,v in fixed.items()) or order not in (32,48) or 'inner_x_cutoff' in reference_policy:
        raise ValueError('fixed full-X, midpoint-pilot and GL32/48 reference policy required')


def integrate(mu,factor,bounds,content_bias=None,*,policy,reference_policy,order,budget=None,deadline=None):
    started=time.perf_counter();budget=budget if budget is not None else NodeBudget(reference_policy['max_total_cell_node_evaluations'])
    ledger=_Ledger(reference_policy,deadline,budget);available={};metadata={}
    try:
        _check_policy(reference_policy,order)
        mu,factor,bounds,bias,fov=validate(mu,factor,bounds,content_bias,policy)
        prepared=prepare(mu,factor,policy['h']);count=len(bounds);boxes=np.vstack((bounds,fov))
        metadata={k:v for k,v in prepared.items() if not isinstance(v,np.ndarray)}
        log_raw=np.full(count,-np.inf,np.float64);log_direct=np.full(1,-np.inf,np.float64)
        computed=np.full(count+1,prepared['structural_zero'],bool)
        available.update(partial_log_raw_mass=log_raw.copy(),partial_log_raw_fov_mass=log_direct.copy(),partial_raw_computed=computed.copy())
        scaled_error=None;convergence=True
        outer_report=dict(evaluation_vectors=0,component_evaluations=0,total_underflow_terms=0,
            local_node_omission_only=True,is_continuous_integral_or_pipeline_error_bound=False,
            scale_reselected=False,scale_is_certified_upper_bound=False)
        metadata.update(method='independent-SVD-fullX-log-conditional-pilot',order=order,
            narrow_normal_quadrature_bound=narrow_normal_local_bound(),
            scaled_outer_error_estimate=None,outer_convergence_success=None,outer_exp_diagnostics=outer_report)
        if prepared['structural_zero']:pass
        elif prepared['normalized_depth_std']==0:
            values=_rectangles(prepared['xy_mean'],1.,boxes,prepared,order,reference_policy,ledger)
            log_raw,log_direct=values[:-1],values[-1:].copy();computed[:]=True
        else:
            m,sigma=prepared['normalized_depth_mean'],prepared['normalized_depth_std']
            threshold=-m/sigma;lower=max(-reference_policy['depth_cutoff'],threshold);upper=reference_policy['depth_cutoff']
            if lower>=upper:raise FloatingPointError('positive depth event lies outside fixed physical cutoff')
            points=outer_breaks(prepared,boxes,lower,upper,reference_policy)
            original_edges=np.asarray([lower]+points+[upper],np.float64)
            edges=coalesce_depth_edges(original_edges,available=available,metadata=metadata)
            points=edges[1:-1].tolist()
            ledger.work['outer_geometry_breakpoints']=len(points)
            if len(edges)-1>reference_policy['maximum_depth_intervals']:
                raise RuntimeError('fixed initial depth-interval budget exceeded')
            pilots=(edges[:-1]+edges[1:])/2
            if np.any(pilots<=edges[:-1]) or np.any(pilots>=edges[1:]):
                raise FloatingPointError('strict interior pilot midpoint is not representable')
            available['initial_depth_edges']=edges;available['pilot_zeta']=pilots
            pilot_logs=np.full((len(pilots),count+1),-np.inf,np.float64)
            pilot_complete=np.zeros(len(pilots),bool)
            def conditional_log(zeta,pilot=False):
                ledger.depth(pilot=pilot);z=m+sigma*zeta
                if z<=0:
                    if zeta<=threshold:return np.full(count+1,-np.inf,np.float64)
                    raise FloatingPointError('strict positive interior depth lost by cancellation/underflow')
                location=prepared['xy_mean']+prepared['beta']*zeta
                value=_rectangles(location,z,boxes,prepared,order,reference_policy,ledger)
                available['last_conditional_log_mass']=value.copy()
                result=value-.5*zeta*zeta-LOG_SQRT_2PI
                if not np.isfinite(result).all():raise FloatingPointError('positive conditional log integrand is nonfinite')
                return result
            for index,zeta in enumerate(pilots):
                pilot_logs[index]=conditional_log(float(zeta),pilot=True);pilot_complete[index]=True
                available['pilot_log_integrand']=pilot_logs.copy();available['pilot_completed']=pilot_complete.copy()
            scale=np.max(pilot_logs,axis=0)
            if not np.isfinite(scale).all():raise FloatingPointError('fixed pilot scale is nonfinite')
            available['outer_log_scale']=scale.copy()
            lost=np.zeros(count+1,np.int64);largest_lost=np.full(count+1,-np.inf);largest=np.full(count+1,-np.inf)
            available.update(outer_underflow_counts=lost.copy(),outer_max_dropped_log_shift=largest_lost.copy(),outer_max_log_shift=largest.copy())
            def integrand(zeta):
                logf=conditional_log(zeta)
                if np.isneginf(logf).all():return np.zeros(count+1,np.float64)
                shift=logf-scale
                if not np.isfinite(shift).all():raise FloatingPointError('fixed outer-scale log shift is nonfinite')
                with np.errstate(over='ignore',under='ignore'):values=np.exp(shift)
                if not np.isfinite(values).all():raise FloatingPointError('fixed pilot outer-scale exponential overflow; no reselection')
                missing=values==0;lost[:]+=missing;largest_lost[:]=np.maximum(largest_lost,np.where(missing,shift,-np.inf));largest[:]=np.maximum(largest,shift)
                outer_report['evaluation_vectors']+=1;outer_report['component_evaluations']+=count+1
                outer_report['total_underflow_terms']+=int(missing.sum())
                available.update(outer_underflow_counts=lost.copy(),outer_max_dropped_log_shift=largest_lost.copy(),outer_max_log_shift=largest.copy())
                return values
            tolerance=reference_policy['outer_relative_tolerances'][reference_policy['orders'].index(order)]
            scaled,scaled_error,info=quad_vec(integrand,lower,upper,epsabs=reference_policy['outer_absolute_tolerance'],
                epsrel=tolerance,norm=reference_policy['outer_norm'],workers=reference_policy['outer_workers'],
                quadrature=reference_policy['outer_quadrature'],points=points,limit=reference_policy['maximum_depth_intervals'],full_output=True)
            scaled=np.asarray(scaled,np.float64);scaled_error=float(scaled_error);ledger.work['outer_intervals']=len(info.intervals)
            metadata.update(scaled_outer_error_estimate=scaled_error if math.isfinite(scaled_error) else None,
                outer_convergence_success=bool(info.success))
            available['outer_scaled_integral']=scaled.copy();convergence=bool(info.success)
            valid=np.isfinite(scaled)&(scaled>0);computed=valid.copy()
            rawlogs=np.full(count+1,-np.inf)
            rawlogs[valid]=np.log(scaled[valid])+scale[valid]
            log_raw,log_direct=rawlogs[:-1],rawlogs[-1:].copy()
            available.update(partial_log_raw_mass=log_raw.copy(),partial_log_raw_fov_mass=log_direct.copy(),partial_raw_computed=computed.copy())
            if not np.all(valid) or not np.isfinite(rawlogs).all():
                raise FloatingPointError('positive-law scaled integral is nonpositive/nonfinite or restoration failed')
            if not math.isfinite(scaled_error) or scaled_error<0:raise FloatingPointError('scaled-vector outer error estimate is nonfinite')
            if not info.success or len(info.intervals)>reference_policy['maximum_depth_intervals']:
                raise RuntimeError('independent scaled outer quadrature failed: '+info.message)
        available.update(partial_log_raw_mass=log_raw.copy(),partial_log_raw_fov_mass=log_direct.copy(),partial_raw_computed=computed.copy())
        normalized,normmeta=_normalize(log_raw,bounds,bias,policy,ledger)
        available.update(normalized)
        log_total=_log_sum(log_raw,ledger,'native_total')
        tail=truncation_bounds(prepared,log_raw,bias,normmeta['log_floor_mass'],reference_policy)
        norm=abs(math.fsum(map(float,normalized['probability']))-1.)
        difference=-np.inf if prepared['structural_zero'] else _log_difference(max(log_total,float(log_direct[0])),min(log_total,float(log_direct[0])))
        ratio=tail['log_raw_l1_over_reference_fov'];pbound=tail['log_probability_max_absolute']
        tail_passed=bool(prepared['structural_zero'] or ((ratio is None or ratio<=math.log(1e-10)) and (pbound is None or pbound<=math.log(1e-8))))
        passed=bool(tail_passed and norm<=2e-12 and difference<=math.log(1e-8))
        ledger.check_time()
        result=dict(log_raw_mass=log_raw,log_raw_fov_mass=log_direct,**normalized,
            method='independent-SVD-fullX-log-conditional-pilot',reference_algorithm_is_unchanged_R27=False,
            order=order,passed=passed,status='complete' if passed else 'failed',work=dict(ledger.work),
            scaled_outer_error_estimate=scaled_error,scaled_outer_error_scope='scaled vector max-norm estimate; not unscaled raw or total error certificate',
            outer_convergence_success=convergence,outer_exp_diagnostics=outer_report,
            lse_diagnostics=ledger.loss,maximum_full_x_panel_width=ledger.maximum_full_x_panel_width,
            normal_logcdf_scalar_evaluations=ledger.normal_logcdf_scalar_evaluations,
            tail_truncation_used=bool(not prepared['structural_zero'] and prepared['normalized_depth_std']!=0),
            inner_x_truncation_used=False,quadrature_error_bound_available=False,truncation_bounds=tail,
            normalization_error=norm,direct_fov_log_absolute_error=_json_log(difference),
            direct_fov_absolute_error_materialized_diagnostic=_diagnostic_exp(difference),
            log_raw_mass_sum=None if prepared['structural_zero'] else log_total,
            log_total_area_floor=normmeta['log_total_area_floor'],log_normalizer=normmeta['log_normalizer'],
            log_content_normalizer=normmeta['log_content_normalizer'],wall_seconds=time.perf_counter()-started,
            **{k:v for k,v in metadata.items() if k not in ('method','order','scaled_outer_error_estimate','outer_convergence_success','outer_exp_diagnostics')})
        if prepared['normalized_depth_std']!=0 and not prepared['structural_zero']:
            for name in ('initial_depth_edges','pilot_zeta','pilot_log_integrand','outer_log_scale','outer_scaled_integral',
                         'outer_underflow_counts','outer_max_dropped_log_shift','outer_max_log_shift',
                         'original_depth_edges','dropped_depth_edge_indices','dropped_depth_edge_values'):
                result[name]=available[name]
        if not passed:result['failure_reason']='independent physical depth-tail, normalization or direct-FOV gate failed; computed logs retained'
        return result
    except (Exception,KeyboardInterrupt) as error:
        metadata.update(lse_diagnostics=ledger.loss,maximum_full_x_panel_width=ledger.maximum_full_x_panel_width,
            normal_logcdf_scalar_evaluations=ledger.normal_logcdf_scalar_evaluations)
        raise ReferenceFailure(error,ledger.work,available,metadata) from error

