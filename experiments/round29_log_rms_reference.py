"""R29 references: immutable R27 quadrature and restricted analytic log-CDF.

Only the declared positive constant-depth diagonal branch uses the analytic
reference. NumPy/SciPy elementary functions are shared dependencies; neither
production preparation, density nor normalization is imported.
"""
from __future__ import annotations

import math
import time
import numpy as np
from scipy.special import log_ndtr, logsumexp

from experiments import round27_rms_reference as OLD

NodeBudget = OLD.NodeBudget
ReferenceFailure = OLD.ReferenceFailure


def probabilities(log_raw, bounds, bias, floor_density):
    """Independent once-only native area floor and bias in log coordinates."""
    log_raw,bounds,bias=map(np.asarray,(log_raw,bounds,bias))
    n=len(bounds)
    if (log_raw.dtype!=np.float64 or log_raw.shape!=(n,) or np.isnan(log_raw).any()
            or np.isposinf(log_raw).any() or bounds.dtype!=np.float64 or bounds.shape!=(n,4)
            or not np.isfinite(bounds).all() or bias.dtype!=np.float64 or bias.shape!=(n,)
            or not np.isfinite(bias).all() or not math.isfinite(floor_density) or floor_density<=0):
        raise ValueError('valid canonical log raw mass, native bounds and finite bias required')
    sides=bounds[:,2:]-bounds[:,:2]
    if np.any(sides<=0):raise ValueError('positive native rectangle sides required')
    floor=np.prod(sides,axis=1)*floor_density
    if not np.isfinite(floor).all() or np.any(floor<=0):
        raise FloatingPointError('original positive finite ordinary area floor product required')
    log_floor=np.log(floor)
    log_floored=np.logaddexp(log_raw,log_floor)
    logits=log_floored+bias;log_z=float(logsumexp(logits))
    if not np.isfinite(log_floored).all() or not math.isfinite(log_z):
        raise FloatingPointError('nonfinite independent log floor or normalizer')
    log_p=logits-log_z;p=np.exp(log_p)
    return dict(log_floored_mass=log_floored,log_probability=log_p,probability=p,
        log_floor_mass=log_floor,log_content_normalizer=log_z)


def log_normal_interval(lower,upper):
    """Two log-CDF values with positive-tail reflection and expm1 subtraction.

    This does not pretend a rounded-equal CDF pair resolves a positive interval:
    such an input fails explicitly. No density quadrature or rescue is used.
    """
    lower,upper=np.broadcast_arrays(np.asarray(lower,np.float64),np.asarray(upper,np.float64))
    if not np.isfinite(lower).all() or not np.isfinite(upper).all() or np.any(lower>=upper):
        raise ValueError('finite strictly ordered standardized endpoints required')
    reflect=lower>=0
    high=log_ndtr(np.where(reflect,-lower,upper))
    low=log_ndtr(np.where(reflect,-upper,lower))
    difference=low-high
    if not np.isfinite(high).all() or not np.isfinite(low).all() or np.any(difference>=0):
        raise FloatingPointError('positive analytic normal interval unresolved by log-CDF difference')
    result=high+np.log(-np.expm1(difference))
    if not np.isfinite(result).all():raise FloatingPointError('nonfinite analytic interval log mass')
    return result


def integrate_old(mu,factor,bounds,content_bias=None,*,policy,reference_policy,order,budget=None,deadline=None):
    """Execute the unchanged R27 reference with unchanged admission/work rules."""
    result=OLD.integrate(mu,factor,bounds,content_bias,policy=policy,reference_policy=reference_policy,
        order=order,budget=budget,deadline=deadline)
    try:
        raw=result['raw_mass'];zero=result['structural_zero']
        if not np.isfinite(raw).all() or not math.isfinite(result['raw_fov_mass']):
            raise FloatingPointError('unchanged R27 reference returned invalid ordinary mass')
        if (zero and (np.any(raw!=0) or result['raw_fov_mass']!=0)) or (not zero and (np.any(raw<=0) or result['raw_fov_mass']<=0)):
            raise FloatingPointError('unchanged R27 reference did not resolve its exact-zero/positive ordinary mass')
        log_raw=np.full(raw.shape,-np.inf,np.float64) if zero else np.log(raw)
        log_direct=-np.inf if zero else math.log(result['raw_fov_mass'])
        result.update(log_raw_mass=log_raw,log_raw_fov_mass=np.asarray([log_direct]),
            raw_structural_zero=np.full(raw.shape,zero,np.bool_),analytic_cdf_evaluations=0,
            reference_route='unchanged_r27_two_order')
        return result
    except Exception as error:
        raise ReferenceFailure(error,result['work'],{k:v for k,v in result.items() if isinstance(v,np.ndarray)},
            {k:v for k,v in result.items() if not isinstance(v,np.ndarray)}) from error


def integrate_analytic(mu,factor,bounds,content_bias=None,*,policy,deadline=None):
    """Single exact-law CDF reference for positive constant-Z diagonal Gaussians.

    Only this restricted law is admitted; the caller separately restricts the
    route to the three predeclared cases. Log-CDF floating-point error has no
    claimed rigorous bound. Direct FOV is evaluated as its own rectangle.
    """
    work=dict(cell_node_evaluations=0,analytic_cdf_evaluations=0)
    available={};started=time.perf_counter()
    try:
        mu,factor,bounds=np.asarray(mu),np.asarray(factor),np.asarray(bounds)
        if (mu.dtype!=np.float64 or mu.shape!=(3,) or not np.isfinite(mu).all()
                or factor.dtype!=np.float64 or factor.shape!=(3,3) or not np.isfinite(factor).all()
                or bounds.dtype!=np.float64 or bounds.ndim!=2 or bounds.shape[1]!=4
                or not len(bounds) or not np.isfinite(bounds).all() or np.any(bounds[:,2:]<=bounds[:,:2])):
            raise ValueError('finite float64 Gaussian and nonempty rectangle arrays required')
        bias=np.zeros(len(bounds),np.float64) if content_bias is None else np.asarray(content_bias)
        if bias.dtype!=np.float64 or bias.shape!=(len(bounds),) or not np.isfinite(bias).all():
            raise ValueError('finite native bias required')
        h=float(policy['h'])
        if not math.isfinite(h) or h<=0 or not np.all(factor[2]==0) or mu[2]<=0:
            raise ValueError('analytic route admits positive exact constant depth only')
        xy=factor[:2]/float(mu[2]);mean=mu[:2]/float(mu[2])
        cross=math.fsum(float(x)*float(y) for x,y in zip(xy[0],xy[1]))
        if cross!=0:raise ValueError('analytic route admits exact diagonal screen covariance only')
        scales=np.asarray([math.hypot(h,*map(float,row)) for row in xy])
        if not np.isfinite(mean).all() or not np.isfinite(scales).all():
            raise FloatingPointError('nonfinite analytic screen parameters')
        if deadline is not None and time.perf_counter()>=deadline:raise RuntimeError('fixed reference phase deadline exceeded')
        fov=np.r_[bounds[:,:2].min(0),bounds[:,2:].max(0)]
        boxes=np.vstack((bounds,fov))
        lower=(boxes[:,:2]-mean)/scales;upper=(boxes[:,2:]-mean)/scales
        if not np.isfinite(lower).all() or not np.isfinite(upper).all() or np.any(lower>=upper):
            raise FloatingPointError('standardized analytic intervals are not representable')
        work['analytic_cdf_evaluations']=4*len(boxes)
        log_mass=log_normal_interval(lower,upper).sum(axis=1)
        available.update(log_raw_mass=log_mass[:-1],log_raw_fov_mass=log_mass[-1:].copy())
        normalized=probabilities(log_mass[:-1],bounds,bias,policy['floor_density'])
        available.update({k:v for k,v in normalized.items() if isinstance(v,np.ndarray)})
        if deadline is not None and time.perf_counter()>=deadline:raise RuntimeError('fixed reference phase deadline exceeded')
        return dict(**available,log_content_normalizer=normalized['log_content_normalizer'],
            raw_structural_zero=np.zeros(len(bounds),np.bool_),structural_zero=False,
            method='independent-constant-depth-diagonal-normal-logCDF-product',reference_route='analytic_single_call',
            passed=True,status='complete',work=work,analytic_cdf_evaluations=work['analytic_cdf_evaluations'],
            quadrature_order=None,order=None,quadrature_error_bound_available=False,
            tail_truncation_used=False,wall_seconds=time.perf_counter()-started)
    except Exception as error:
        raise ReferenceFailure(error,work,available,dict(reference_route='analytic_single_call')) from error
