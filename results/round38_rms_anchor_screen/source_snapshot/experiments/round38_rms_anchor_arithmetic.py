"""Unchanged pure log arithmetic checks copied from accepted R30 audit.
No model/reference import, archive decode, or probability integration.
"""
import math
import numpy as np
from scipy.special import logsumexp, ndtr
LSE_SCOPES=('tile_nodes','cell_tiles','fov_tiles','native_total','floor','floor_total','floored_total','content_normalization')

def structural_branch(mu,factor):
    rms=math.hypot(float(mu[2]),*map(float,factor[2]))
    if rms==0:return 'zero_depth_rms',True
    if np.all(factor[2]==0):
        return ('nonpositive_constant_depth',True) if mu[2]<=0 else ('positive_constant_depth_gaussian',False)
    return 'variable_depth',False

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

def finite_or_none(value):
    return float(value) if math.isfinite(float(value)) else None

def exp_diagnostic(value):
    if value==-np.inf:return 0.
    if value>math.log(np.finfo(np.float64).max):return None
    return math.exp(float(value))

def log_absolute_difference(a,b):
    """Log |exp(a)-exp(b)| without ordinary raw materialization."""
    a,b=np.broadcast_arrays(np.asarray(a,np.float64),np.asarray(b,np.float64))
    if np.isnan(a).any() or np.isnan(b).any() or np.isposinf(a).any() or np.isposinf(b).any():
        raise ValueError('canonical log masses may be finite or exact negative infinity only')
    high=np.maximum(a,b);result=np.full(a.shape,-np.inf,np.float64)
    different=a!=b
    result[different]=high[different]+np.log(-np.expm1(-np.abs(a[different]-b[different])))
    return result

def validate_logs(log_raw,structural_zero):
    log_raw=np.asarray(log_raw)
    if log_raw.dtype!=np.float64 or log_raw.ndim!=1 or not len(log_raw):
        raise ValueError('nonempty float64 canonical native log mass required')
    if structural_zero:
        if not np.isneginf(log_raw).all():raise ValueError('structural zero must have exact -inf log mass')
    elif not np.isfinite(log_raw).all():
        raise ValueError('positive nonstructural raw mass must have finite canonical logs')
    return log_raw

def direct_fov_check(log_raw,log_direct,structural_zero,absolute_gate):
    raw=validate_logs(log_raw,structural_zero);direct=np.asarray(log_direct)
    if direct.dtype!=np.float64 or direct.shape!=(1,):raise ValueError('one canonical direct FOV log required')
    validate_logs(direct,structural_zero)
    error=float(log_absolute_difference(float(logsumexp(raw)),float(direct[0])))
    return dict(passed=bool(error<=math.log(absolute_gate)),log_absolute_error=finite_or_none(error),
        absolute_error_materialized_diagnostic=exp_diagnostic(error))

def materialization_check(bank,structural_zero):
    raw=validate_logs(bank['log_raw_mass'],structural_zero)
    with np.errstate(under='ignore',over='ignore'):ordinary=np.exp(raw)
    if not np.isfinite(ordinary).all() or not np.isfinite(bank['materialized_raw_mass']).all():
        raise ValueError('ordinary materialization overflow is a failed admission, never valid packed output')
    indices=np.arange(len(raw),dtype=np.int64) if structural_zero else np.flatnonzero(ordinary>0)
    values=np.zeros(len(raw),np.float64) if structural_zero else ordinary[indices]
    if (bank['materialized_raw_indices'].dtype!=np.int64 or bank['materialized_raw_mass'].dtype!=np.float64
            or not np.array_equal(bank['materialized_raw_indices'],indices)
            or not np.array_equal(bank['materialized_raw_mass'],values)):
        raise ValueError('indexed ordinary materialization must retain exactly exp(lograw)>0, or structural zeros')
    return dict(ordinary_raw_unmaterialized_count=0 if structural_zero else len(raw)-len(indices),
        ordinary_raw_subnormal_count=int(np.count_nonzero((values>0)&(values<np.finfo(np.float64).tiny))))

def local_lse_diagnostic(terms):
    """Reconstruct local-loss metadata for saved log inputs, not density nodes."""
    terms=np.asarray(terms,np.float64)
    if terms.ndim==1:terms=terms[None,:]
    maximum=np.max(terms,axis=-1,keepdims=True);finite=np.isfinite(terms)
    with np.errstate(invalid='ignore',under='ignore'):
        shifted=np.where(finite,terms-maximum,-np.inf);scaled=np.exp(shifted)
    lost=finite&(scaled==0);counts=np.sum(lost,axis=-1);affected=counts>0
    result=dict(sum_count=len(terms),underflow_sum_count=int(affected.sum()),underflow_term_count=int(counts.sum()),
        maximum_log_scaled_omitted_bound=None,maximum_log_absolute_omitted_bound=None,maximum_log_relative_omitted_bound=None)
    if np.any(affected):
        biggest=np.max(np.where(lost,shifted,-np.inf),axis=-1)[affected]
        bound=np.log(counts[affected])+biggest
        retained=np.asarray([math.fsum(map(float,row)) for row in scaled[affected]])
        result.update(maximum_log_scaled_omitted_bound=float(bound.max()),
            maximum_log_absolute_omitted_bound=float((maximum[:,0][affected]+bound).max()),
            maximum_log_relative_omitted_bound=float((bound-np.log(retained)).max()))
    return result

def lse_check(row,bank,zero,native_tiles,fov_tiles,tolerance):
    report=row['lse_diagnostics'];scopes=report['by_scope']
    if (set(scopes)!=set(LSE_SCOPES) or report['exact_arithmetic_local_omission_only'] is not True
            or report['is_pipeline_error_bound'] is not False
            or report['excludes_addition_absorption_density_quadrature_and_roundoff'] is not True):
        raise ValueError('LSE diagnostics must retain local-only scope')
    n=len(bank['log_raw_mass']);counts=dict(tile_nodes=native_tiles+fov_tiles,cell_tiles=native_tiles,fov_tiles=fov_tiles,
        native_total=1,floor=n,floor_total=1,floored_total=1,content_normalization=1)
    sizes=dict(tile_nodes=576,cell_tiles=2,fov_tiles=2,native_total=n,floor=2,floor_total=n,floored_total=n,content_normalization=n)
    fields={'sum_count','underflow_sum_count','underflow_term_count','maximum_log_scaled_omitted_bound',
            'maximum_log_absolute_omitted_bound','maximum_log_relative_omitted_bound'}
    for name,r in scopes.items():
        if set(r)!=fields or r['sum_count']!=counts[name] or any(type(r[k]) is not int or r[k]<0 for k in ('sum_count','underflow_sum_count','underflow_term_count')):
            raise ValueError('local positive sum counts differ')
        if not (0<=r['underflow_sum_count']<=r['sum_count'] and r['underflow_sum_count']<=r['underflow_term_count']<=r['underflow_sum_count']*(sizes[name]-1)):
            raise ValueError('impossible positive-sum underflow counts')
        bounds=[r[k] for k in ('maximum_log_scaled_omitted_bound','maximum_log_absolute_omitted_bound','maximum_log_relative_omitted_bound')]
        if not r['underflow_term_count']:
            if any(v is not None for v in bounds):raise ValueError('no lost terms requires null local bounds')
        elif not all(isinstance(v,(float,int)) and math.isfinite(v) for v in bounds):
            raise ValueError('lost positive terms require finite local log bounds')
        elif bounds[2]>bounds[0]+tolerance or bounds[0]>math.log(max(1,sizes[name]-1))+math.log(np.nextafter(0.,1.))+tolerance:
            raise ValueError('local omitted bound incompatible with exp-underflow event')
    floor=np.prod(bank['input__bounds'][:,2:]-bank['input__bounds'][:,:2],axis=1)*row['floor_density']
    logfloor=np.log(floor)
    saved_inputs=dict(native_total=bank['log_raw_mass'],floor=np.column_stack((bank['log_raw_mass'],logfloor)),
        floor_total=logfloor,floored_total=bank['log_floored_mass'],
        content_normalization=bank['log_floored_mass']+bank['input__content_bias'])
    for name,terms in saved_inputs.items():
        expected=local_lse_diagnostic(terms)
        for key,value in expected.items():
            actual=scopes[name][key]
            if value is None or isinstance(value,int):
                if actual!=value:raise ValueError('saved-input LSE local loss diagnostic differs: '+name+'/'+key)
            elif actual is None or abs(actual-value)>tolerance:
                raise ValueError('saved-input LSE local log bound differs: '+name+'/'+key)
    return dict(passed=True,saved_input_scopes_independently_recomputed=list(saved_inputs),
        node_and_tile_scopes_not_numerically_replayed=['tile_nodes','cell_tiles','fov_tiles'],
        local_omitted_bound_is_pipeline_certificate=False)

def production_check(row,bank,policy,gates):
    if row['passed'] is not True:return dict(passed=False,status='production_failed',failure_reason=row.get('failure_reason'))
    mu,factor,bounds,bias=[bank['input__'+n] for n in ('mu','factor','bounds','content_bias')]
    branch,zero=structural_branch(mu,factor);raw=validate_logs(bank['log_raw_mass'],zero)
    expected=probabilities(raw,bounds,bias,policy['floor_density'])
    p=bank['probability'];logp=bank['log_probability'];lf=bank['log_floored_mass']
    if (not np.isfinite(p).all() or np.any(p<0) or not np.isfinite(logp).all() or not np.isfinite(lf).all()):
        raise ValueError('finite completed log floor, log probability and ordinary probability required')
    rms=math.hypot(float(mu[2]),*map(float,factor[2]));m=None if rms==0 else float(mu[2]/rms)
    sigma=None if rms==0 else math.hypot(*map(float,factor[2]/rms))
    front=0. if zero else (1. if sigma==0 else float(ndtr(m/sigma)))
    fov=np.r_[bounds[:,:2].min(0),bounds[:,2:].max(0)];maxside=policy['h']*policy['maximum_tile_side_in_h']
    native=0 if zero else sum(int(np.prod(np.ceil((box[2:]-box[:2])/maxside).astype(np.int64))) for box in bounds)
    fv=0 if zero else int(np.prod(np.ceil((fov[2:]-fov[:2])/maxside).astype(np.int64)))
    work=dict(cell_node_evaluations=(native+fv)*24**2,native_tiles_completed=native,fov_tiles_completed=fv)
    material=materialization_check(bank,zero);direct=direct_fov_check(raw,bank['log_raw_fov_mass'],zero,gates['direct_fov_additivity_absolute'])
    diag=row['density_diagnostics'];series_keys=('maximum_log_negative_tail_relative_remainder','minimum_scaled_negative_tail_series_lower_bound','maximum_scaled_negative_tail_series_upper_bound')
    nodekeys=('constant_depth_nodes','positive_branch_nodes','negative_branch_nodes','negative_erfcx_nodes','negative_series_nodes')
    if any(type(diag[k]) is not int or diag[k]<0 for k in nodekeys):raise ValueError('density branch counts invalid')
    if diag['negative_series_nodes']:
        rel,lower,upper=[diag[k] for k in series_keys]
        series=all(math.isfinite(v) for v in (rel,lower,upper)) and 0<lower<=upper and rel<=math.log(policy['max_negative_tail_relative_remainder'])
    else:series=all(diag[k] is None for k in series_keys)
    density_range=(diag['maximum_log_density'] is None and diag['minimum_log_density'] is None) if zero else (
        math.isfinite(diag['maximum_log_density']) and math.isfinite(diag['minimum_log_density']) and diag['minimum_log_density']<=diag['maximum_log_density'])
    bound=0. if zero else float(np.prod(bounds[:,2:]-bounds[:,:2],axis=1).sum())/(math.pi*policy['h']**2*math.comb(48,24))
    trunc=row['analytic_truncation_bound'];tol=gates['normalization_absolute']
    logsum=None if zero else float(logsumexp(raw));logfloor=float(logsumexp(expected['log_floor_mass']))
    metadata=bool(row['structural_branch']==branch and row['structural_zero'] is zero and series and density_range
        and np.isclose(row['front_mass'],front,rtol=2e-13,atol=0) and row['rms_depth']==rms
        and row['normalized_depth_mean']==m and row['normalized_depth_std']==sigma
        and row['quadrature_order']==24 and row['maximum_tile_side_in_h']==2. and row['h']==policy['h'] and row['floor_density']==policy['floor_density']
        and row['projection_regularization']=='rms_depth_xy_gaussian' and row['canonical_raw_representation']=='log_mass'
        and row['ordinary_materialization_is_canonical'] is False and row['tail_truncation_used'] is False
        and row['quadrature_error_bound_available'] is False and row['exact_arithmetic_gl_bound_available'] is True
        and row['native_tile_count']==native and row['fov_tile_count']==fv and row['work']==work
        and trunc['exact_arithmetic_only'] is True and trunc['excludes_density_evaluation_and_roundoff'] is True
        and trunc['is_relative_error_certificate'] is False and trunc['raw_native_l1_absolute']==bound and trunc['raw_direct_fov_absolute']==bound
        and diag['constant_depth_nodes']+diag['positive_branch_nodes']+diag['negative_branch_nodes']==work['cell_node_evaluations']
        and diag['negative_erfcx_nodes']+diag['negative_series_nodes']==diag['negative_branch_nodes']
        and (row['log_raw_mass_sum'] is None if zero else (isinstance(row['log_raw_mass_sum'],(float,int)) and abs(row['log_raw_mass_sum']-logsum)<=tol))
        and abs(row['log_total_area_floor']-logfloor)<=tol and abs(row['log_normalizer']-float(logsumexp(lf)))<=tol
        and abs(row['log_content_normalizer']-expected['log_content_normalizer'])<=tol
        and row['ordinary_raw_unmaterialized_count']==material['ordinary_raw_unmaterialized_count']
        and row['ordinary_raw_subnormal_count']==material['ordinary_raw_subnormal_count']
        and row['zero_raw_cells']==(len(raw) if zero else 0) and row['all_raw_mass_zero'] is zero
        and row['area_prior_used'] is bool(zero and np.all(bias==bias[0]))
        and row['raw_mass_below_total_floor'] is bool(zero or logsum<logfloor)
        and row['argmax_index']==int(p.argmax()) and row['argmax_exact_ties']==int(np.count_nonzero(p==p.max()))
        and row['argmax_near_ties_at_1e_12']==int(np.count_nonzero(p.max()-p<=1e-12)))
    local=lse_check(row,bank,zero,native,fv,tol)
    norm=abs(math.fsum(map(float,p))-1.);pe=float(np.max(abs(p-expected['probability'])))
    lfe=float(np.max(abs(lf-expected['log_floored_mass'])));lpe=float(np.max(abs(logp-expected['log_probability'])))
    with np.errstate(under='ignore'):saved_exp_error=float(np.max(abs(np.exp(logp)-p)))
    passed=bool(metadata and direct['passed'] and local['passed'] and work['cell_node_evaluations']<=policy['max_cell_node_evaluations']
        and max(norm,pe,lfe,lpe,saved_exp_error)<=tol)
    return dict(passed=passed,status='checked',metadata_consistent=metadata,structural_branch=branch,structural_zero=zero,
        normalization_error=norm,same_raw_probability_error=pe,log_floor_arithmetic_error=lfe,log_probability_arithmetic_error=lpe,
        saved_probability_exp_error=saved_exp_error,direct_fov=direct,lse_diagnostic_check=local,**material,
        log_raw_mass_sum=logsum,log_total_area_floor=logfloor,raw_below_total_floor=bool(zero or logsum<logfloor),
        all_raw_mass_zero=zero,exact_maximizers=np.flatnonzero(p==p.max()).tolist(),work_expected=work)

def partial_check(row,bank,policy):
    names={n for n in bank if not n.startswith('input__')}
    expected={'partial_log_raw_mass','partial_log_raw_fov_mass','partial_completed_cells','partial_completed_fov',
              'partial_accumulated_tiles','partial_accumulated_fov_tiles'}
    if not names:return dict(passed=True,available_partial_arrays=[])
    if names!=expected:raise ValueError('returned partial log integration requires all masks and tile counts')
    for logkey,countkey,maskkey in (('partial_log_raw_mass','partial_accumulated_tiles','partial_completed_cells'),
                                   ('partial_log_raw_fov_mass','partial_accumulated_fov_tiles','partial_completed_fov')):
        logs,counts,mask=bank[logkey],bank[countkey],bank[maskkey]
        if np.isnan(logs).any() or np.isposinf(logs).any() or np.any(counts<0):raise ValueError('invalid partial log sum/count')
        if not np.array_equal(np.isfinite(logs),counts>0):raise ValueError('partial finite sums must distinguish accumulated from uncomputed cells')
        if np.any(mask&(counts==0)) and not row.get('structural_zero',False):raise ValueError('positive completed cell needs accumulated tiles')
    if (int(bank['partial_accumulated_tiles'].sum())!=row['work']['native_tiles_completed']
            or int(bank['partial_accumulated_fov_tiles'].sum())!=row['work']['fov_tiles_completed']
            or row['work']['cell_node_evaluations']>policy['max_cell_node_evaluations']):
        raise ValueError('partial tile work differs or exceeds fixed budget')
    zero=structural_branch(bank['input__mu'],bank['input__factor'])[1]
    if zero:
        all_masks=np.r_[bank['partial_completed_cells'],bank['partial_completed_fov']]
        if (any(np.any(bank[n]) for n in ('partial_accumulated_tiles','partial_accumulated_fov_tiles'))
                or not (np.all(all_masks) or not np.any(all_masks))):
            raise ValueError('structural-zero partial state must be wholly before or after its exact branch')
    else:
        bounds=bank['input__bounds'];maxside=policy['h']*policy['maximum_tile_side_in_h']
        totals=np.asarray([int(np.prod(np.ceil((box[2:]-box[:2])/maxside))) for box in bounds],np.int64)
        fov=np.r_[bounds[:,:2].min(0),bounds[:,2:].max(0)]
        ftotal=np.asarray([int(np.prod(np.ceil((fov[2:]-fov[:2])/maxside)))],np.int64)
        for counts,masks,expected in ((bank['partial_accumulated_tiles'],bank['partial_completed_cells'],totals),
                (bank['partial_accumulated_fov_tiles'],bank['partial_completed_fov'],ftotal)):
            if np.any(counts>expected) or not np.array_equal(masks,counts==expected):
                raise ValueError('partial completion masks must match independently derived full tile counts')
    return dict(passed=True,available_partial_arrays=sorted(names),
        completed_native_cells=int(bank['partial_completed_cells'].sum()),
        finite_partial_native_cells=int(np.isfinite(bank['partial_log_raw_mass']).sum()),
        incomplete_finite_native_cells=int(np.count_nonzero(np.isfinite(bank['partial_log_raw_mass'])&~bank['partial_completed_cells'])))
