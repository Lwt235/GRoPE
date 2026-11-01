"""R32 independent sealed-input, delivery and log-arithmetic audit.

Imports no numerical reference integrator or production model. Accepted R29
raw/P evidence is decoded only after the global new-reference seal. Shared C32
helpers perform provenance/runtime operations only. Log comparison arithmetic
is retained from the accepted independent R29 audit without importing it.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import shutil
import time
import numpy as np
from scipy.special import logsumexp, log_ndtr, ndtr
from experiments import round32_log_conditional_contract as C

SCOPE=dict(real_inputs_read=False,labels_read=False,general_numerical_accuracy_established=False,
           efficacy_tested=False,candidate_selected=False,scientific_gate_passed=False,goal_complete=False)
DEFAULT_RUN=C.ROOT/'results/round32_log_conditional_preflight'
DEFAULT_OUT=C.ROOT/'results/round32_log_conditional_independent_audit'
INPUT_FIELDS=('input__mu','input__factor','input__bounds','input__content_bias')
APPROXIMATE_TIE_TOLERANCE=1e-12

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

def compare_log_raw(actual,reference,bounds,bias,floor_density,gates,*,reference_comparison=False,
                    archived_probability=None,reference_probability=None,structural_zero=False):
    actual=validate_logs(actual,structural_zero);reference=validate_logs(reference,structural_zero)
    if actual.shape!=reference.shape:raise ValueError('native log mass shapes differ')
    pa=probabilities(actual,bounds,bias,floor_density)['probability']
    pr=probabilities(reference,bounds,bias,floor_density)['probability']
    if archived_probability is not None:
        pa=np.asarray(archived_probability)
        if pa.dtype!=np.float64 or pa.shape!=actual.shape or not np.isfinite(pa).all() or np.any(pa<0):
            raise ValueError('finite archived native probability required')
    if reference_probability is not None:
        pr=np.asarray(reference_probability)
        if pr.dtype!=np.float64 or pr.shape!=reference.shape or not np.isfinite(pr).all() or np.any(pr<0):
            raise ValueError('finite archived reference probability required')
    log_a=float(logsumexp(actual));log_r=float(logsumexp(reference))
    log_diff=log_absolute_difference(actual,reference)
    if structural_zero:log_fov_relative=log_l1_relative=-np.inf
    else:
        log_fov_relative=float(log_absolute_difference(log_a,log_r))-log_r
        log_l1_relative=float(logsumexp(log_diff))-log_r
    relative_gate=gates['reference_raw_relative'] if reference_comparison else gates['raw_fov_relative']
    l1_gate=gates['reference_raw_relative'] if reference_comparison else gates['raw_l1_over_reference_fov']
    p_gate=gates['reference_probability_max_absolute'] if reference_comparison else gates['probability_max_absolute']
    pe=float(np.max(abs(pa-pr)))
    log_cell_relative=log_diff-reference if not structural_zero else np.full(actual.shape,-np.inf)
    return dict(passed=bool((structural_zero or (log_fov_relative<=math.log(relative_gate) and log_l1_relative<=math.log(l1_gate))) and pe<=p_gate),
        structural_zero=structural_zero,raw_relative_applicable=not structural_zero,
        log_raw_fov_actual=finite_or_none(log_a),log_raw_fov_reference=finite_or_none(log_r),
        raw_fov_actual_materialized_diagnostic=exp_diagnostic(log_a),raw_fov_reference_materialized_diagnostic=exp_diagnostic(log_r),
        raw_fov_relative=None if structural_zero else exp_diagnostic(log_fov_relative),
        raw_l1_over_reference_fov=None if structural_zero else exp_diagnostic(log_l1_relative),
        log_raw_fov_relative=finite_or_none(log_fov_relative),log_raw_l1_over_reference_fov=finite_or_none(log_l1_relative),
        maximum_probability_error=pe,per_cell_log_absolute_error=[finite_or_none(x) for x in log_diff],
        per_cell_relative_error=[None if structural_zero else exp_diagnostic(x) for x in log_cell_relative],
        per_cell_log_relative_error=[finite_or_none(x) for x in log_cell_relative],
        zero_reference_cells=len(reference) if structural_zero else 0,actual_argmax=int(pa.argmax()),reference_argmax=int(pr.argmax()),
        argmax_changed=bool(pa.argmax()!=pr.argmax()),actual_exact_maximizers=np.flatnonzero(pa==pa.max()).tolist(),
        reference_exact_maximizers=np.flatnonzero(pr==pr.max()).tolist(),
        actual_approximate_maximizers=np.flatnonzero(abs(pa-pa.max())<=APPROXIMATE_TIE_TOLERANCE).tolist(),
        reference_approximate_maximizers=np.flatnonzero(abs(pr-pr.max())<=APPROXIMATE_TIE_TOLERANCE).tolist(),
        approximate_tie_absolute_tolerance=APPROXIMATE_TIE_TOLERANCE)

def direct_fov_check(log_raw,log_direct,structural_zero,absolute_gate):
    raw=validate_logs(log_raw,structural_zero);direct=np.asarray(log_direct)
    if direct.dtype!=np.float64 or direct.shape!=(1,):raise ValueError('one canonical direct FOV log required')
    validate_logs(direct,structural_zero)
    error=float(log_absolute_difference(float(logsumexp(raw)),float(direct[0])))
    return dict(passed=bool(error<=math.log(absolute_gate)),log_absolute_error=finite_or_none(error),
        absolute_error_materialized_diagnostic=exp_diagnostic(error))


def scope_check(value):
    if any(value.get(k) is not False for k in SCOPE):
        raise ValueError('seven explicit false synthetic-only scope flags required')
    for key in ('synthetic_inputs_read','independent_synthetic_reference'):
        if type(value.get(key)) is not bool:raise ValueError('explicit actual synthetic stage flags required')


def read_arrays(path,hashes,fields=None,on_decode=None):
    with np.load(path,allow_pickle=False) as archive:
        names=list(hashes) if fields is None else list(fields)
        if fields is None and set(archive.files)!=set(hashes):raise ValueError('unlisted archive fields')
        result={}
        for name in names:
            result[name]=archive[name]
            if on_decode is not None:on_decode()
    if {k:C.array_sha(v) for k,v in result.items()}!=hashes:raise ValueError('canonical array bytes differ')
    return result


def input_check(source,actual,spec):
    if set(source)!=set(INPUT_FIELDS) or set(actual)!=set(INPUT_FIELDS):raise ValueError('exact four synthetic inputs required')
    fixture=spec['fixture'];n=len(fixture['bounds'])
    shapes=dict(input__mu=(3,),input__factor=(3,3),input__bounds=(n,4),input__content_bias=(n,))
    for name,shape in shapes.items():
        expected=np.asarray(fixture[name.removeprefix('input__')],np.float64)
        for value in (source[name],actual[name]):
            if value.dtype!=np.float64 or value.shape!=shape or not np.isfinite(value).all():
                raise ValueError('finite exact input schema differs: '+name)
            if C.array_sha(value)!=spec['array_sha256'][name] or C.array_sha(value)!=C.array_sha(expected):
                raise ValueError('descriptor/source/copied input bytes differ: '+name)
    if spec['id']!=fixture['id'] or fixture['kind']!='base':raise ValueError('base descriptor identity differs')
    return dict(passed=True,canonical_input_bytes_exact=True,descriptor_bytes_exact=True)


def complete_inventory(directory):
    done=C.read(directory/'completion.json');C.verify_inventory(directory,done['files'],exclude=('completion.json',));return done


def ordered_calls(specs):
    return [(s['id'],o,s['id']+'__order'+str(o)) for s in specs for o in (32,48)]


def verify_producer_lock(run):
    lock=C.read(run/'lock.json');scope_check(lock)
    bindings,specs,oracles=C.inherited_evidence()
    expected=dict(version=1,runtime=C.check_runtime(),policy=C.POLICY,reference_policy=C.REFERENCE_POLICY,gates=C.GATES,
        bindings=bindings,case_ids=[s['id'] for s in specs],case_count=23,
        ordered_call_ids=[x[2] for x in ordered_calls(specs)],call_count=46,
        query_specs_sha256=C.sha(run/'query_specs.json'),oracle_specs_sha256=C.sha(run/'oracle_specs.json'),
        synthetic_inputs_read=False,independent_synthetic_reference=False,**SCOPE)
    if (C.sha(run/'lock.json')!=(run/'lock.sha256').read_text().strip()
            or any(lock.get(k)!=v for k,v in expected.items())
            or C.read(run/'query_specs.json')!=specs or C.read(run/'oracle_specs.json')!=oracles):
        raise ValueError('frozen sources/old evidence/population/runtime differs')
    if set(lock['source_hashes'])!=set(C.source_names()):raise ValueError('full local source closure differs')
    for n,h in lock['source_hashes'].items():
        if C.sha(C.ROOT/n)!=h or C.sha(run/'source_snapshot'/n)!=h:raise ValueError('current or frozen source changed')
    return lock


def freeze(run,out):
    if out.exists() or any((run/n).exists() for n in ('prepared','prepared_complete.json','forward','completion.json')):
        raise ValueError('independent freeze must precede new synthetic decoding')
    lock=verify_producer_lock(run)
    if Path(lock['audit_root']).resolve()!=out.resolve():raise ValueError('audit destination differs')
    out.mkdir(parents=True)
    for n,h in lock['source_hashes'].items():
        target=out/'source_snapshot'/n;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(C.ROOT/n,target)
        if C.sha(target)!=h:raise ValueError('source changed during independent freeze')
    C.write(out/'lock.json',dict(created_unix=time.time(),run=str(run.resolve()),producer_root=str(run.resolve()),
        producer_lock_sha256=C.sha(run/'lock.json'),query_specs_sha256=lock['query_specs_sha256'],
        oracle_specs_sha256=lock['oracle_specs_sha256'],source_hashes=lock['source_hashes'],runtime=lock['runtime'],
        synthetic_inputs_read=False,independent_synthetic_reference=False,**SCOPE))
    (out/'lock.sha256').write_text(C.sha(out/'lock.json')+'\n')


def joint_locks(run,out):
    lock=verify_producer_lock(run);design=C.read(out/'lock.json');scope_check(design)
    expected=dict(run=str(run.resolve()),producer_root=str(run.resolve()),producer_lock_sha256=C.sha(run/'lock.json'),
        query_specs_sha256=lock['query_specs_sha256'],oracle_specs_sha256=lock['oracle_specs_sha256'],
        source_hashes=lock['source_hashes'],runtime=lock['runtime'],synthetic_inputs_read=False,
        independent_synthetic_reference=False,**SCOPE)
    if (C.sha(out/'lock.json')!=(out/'lock.sha256').read_text().strip() or any(design.get(k)!=v for k,v in expected.items())
            or not lock['created_unix']<=design['created_unix']<=time.time()):raise ValueError('independent joint lock differs')
    for n,h in lock['source_hashes'].items():
        if C.sha(out/'source_snapshot'/n)!=h:raise ValueError('independent source snapshot changed')
    return lock,design


def verify_prepared(run,out):
    lock,design=joint_locks(run,out);done=complete_inventory(run/'prepared')
    seal=C.read(run/'prepared_complete.json');rows=C.read(run/'prepared/records.json');specs=C.read(run/'query_specs.json')
    expected=dict(producer_lock_sha256=C.sha(run/'lock.json'),audit_lock_sha256=C.sha(out/'lock.json'),
        prepared_records_sha256=C.sha(run/'prepared/records.json'),prepared_completion_sha256=C.sha(run/'prepared/completion.json'),
        query_specs_sha256=lock['query_specs_sha256'],case_ids=[s['id'] for s in specs],case_count=23,
        independent_synthetic_reference=False,**SCOPE)
    if any(seal.get(k)!=v for k,v in expected.items()) or [r['id'] for r in rows]!=expected['case_ids']:
        raise ValueError('complete ordered23 prepared seal required')
    start=C.read(run/'prepared/started.json')
    for stage in (start,seal,done):scope_check(stage)
    if (not design['created_unix']<=seal['started_unix']==start['started_unix']<=seal['finished_unix']
            or start['producer_lock_sha256']!=expected['producer_lock_sha256']
            or start['audit_lock_sha256']!=expected['audit_lock_sha256']
            or done['complete_case_inventory'] is not True or done['all_inputs_prepared']!=seal['all_inputs_prepared']
            or seal['all_inputs_prepared']!=all(r['passed'] for r in rows)
            or seal['predictions_executed'] is not False or seal['reference_calls_executed'] is not False
            or start['synthetic_inputs_read'] is not False
            or seal['synthetic_inputs_read'] is not any(r['synthetic_inputs_read'] for r in rows)
            or done['synthetic_inputs_read'] is not seal['synthetic_inputs_read']
            or any(s['independent_synthetic_reference'] is not False for s in (start,seal,done))):
        raise ValueError('prepared chronology or actual scope differs')
    if set(done['files'])!={'started.json','records.json'}|{'arrays/'+s['id']+'.npz' for s in specs}:
        raise ValueError('prepared inventory differs')
    for row,spec in zip(rows,specs):
        scope_check(row)
        if (row['source_archive_path']!=spec['source_archive_path'] or row['source_archive_sha256']!=spec['source_archive_sha256']
                or row['policy_overrides']!=spec['policy_overrides']
                or row['archive_path']!='prepared/arrays/'+spec['id']+'.npz'
                or C.sha(run/row['archive_path'])!=row['archive_sha256'] or type(row['passed']) is not bool
                or row['status']!=('prepared' if row['passed'] else 'failed')
                or row['independent_synthetic_reference'] is not False
                or not math.isfinite(row['preparation_seconds']) or row['preparation_seconds']<0):
            raise ValueError('prepared row identity/status/time differs')
        if row['passed'] and (set(row['array_sha256'])!=set(INPUT_FIELDS) or not row['synthetic_inputs_read']):
            raise ValueError('prepared success requires four actually decoded fields')
        if not row['passed'] and not row.get('failure_reason'):raise ValueError('failed preparation reason required')
    return lock,seal,rows,specs


def failure(directory,error,started,decoded=False,reference=False):
    C.write(directory/'failure.json',dict(error_type=type(error).__name__,message=str(error),started_unix=started,
        finished_unix=time.time(),synthetic_inputs_read=decoded,independent_synthetic_reference=reference,**SCOPE))
    C.write(directory/'completion.json',dict(files={n:h for n,h in C.inventory(directory).items() if n!='completion.json'},
        delivery_integrity_passed=False,numerical_preflight_passed=False,finished_unix=time.time(),
        synthetic_inputs_read=decoded,independent_synthetic_reference=reference,**SCOPE))


def input_audit(run,out):
    directory=out/'input_audit'
    if directory.exists() or (out/'input_verification.json').exists():raise ValueError('preserve prior input audit')
    directory.mkdir();started=time.time();C.write(directory/'started.json',dict(started_unix=started,
        synthetic_inputs_read=False,independent_synthetic_reference=False,**SCOPE))
    decoded={'value':False};checks=[]
    def note():decoded['value']=True;local['value']=True
    try:
        lock,seal,rows,specs=verify_prepared(run,out)
        if started<seal['finished_unix'] or (run/'forward').exists():raise ValueError('input audit must follow global preparation and precede calls')
        for row,spec in zip(rows,specs):
            local={'value':False}
            try:
                if not row['passed']:raise ValueError(row['failure_reason'])
                if C.sha(spec['source_archive_path'])!=spec['source_archive_sha256']:raise ValueError('old source archive changed')
                old=read_arrays(spec['source_archive_path'],spec['array_sha256'],INPUT_FIELDS,note)
                actual=read_arrays(run/row['archive_path'],row['array_sha256'],on_decode=note)
                check=input_check(old,actual,spec)
            except Exception as error:check=dict(passed=False,failure_reason=type(error).__name__+': '+str(error))
            checks.append(dict(id=spec['id'],prepared_archive_sha256=row['archive_sha256'],**check,
                synthetic_inputs_read=local['value'],independent_synthetic_reference=False,**SCOPE))
            C.write(directory/'checks.json',checks)
        joint_locks(run,out)
        passed=len(checks)==23 and all(r['passed'] for r in checks)
        verification=dict(passed=passed,checked_cases=len(checks),producer_lock_sha256=C.sha(run/'lock.json'),
            audit_lock_sha256=C.sha(out/'lock.json'),producer_prepared_complete_sha256=C.sha(run/'prepared_complete.json'),
            producer_prepared_completion_sha256=C.sha(run/'prepared/completion.json'),query_specs_sha256=lock['query_specs_sha256'],
            source_prepared_hashes={s['source_archive_path']:s['source_archive_sha256'] for s in specs},
            all_input_bytes_exact=passed,started_unix=started,finished_unix=time.time(),synthetic_inputs_read=decoded['value'],
            independent_synthetic_reference=False,**SCOPE)
        C.write(out/'input_verification.json',verification)
        C.write(directory/'completion.json',dict(passed=passed,files=C.inventory(directory),
            verification_sha256=C.sha(out/'input_verification.json'),finished_unix=time.time(),
            synthetic_inputs_read=decoded['value'],independent_synthetic_reference=False,**SCOPE))
        if not passed:raise RuntimeError('input identity failed; no new reference authorized')
        return verification
    except (Exception,KeyboardInterrupt) as error:
        if not (directory/'completion.json').exists():failure(directory,error,started,decoded['value'])
        raise


def verify_input_audit(run,out):
    lock,seal,rows,specs=verify_prepared(run,out);done=complete_inventory(out/'input_audit')
    verification=C.read(out/'input_verification.json');scope_check(verification);scope_check(done)
    expected=dict(passed=True,checked_cases=23,producer_lock_sha256=C.sha(run/'lock.json'),audit_lock_sha256=C.sha(out/'lock.json'),
        producer_prepared_complete_sha256=C.sha(run/'prepared_complete.json'),
        producer_prepared_completion_sha256=C.sha(run/'prepared/completion.json'),query_specs_sha256=lock['query_specs_sha256'],
        source_prepared_hashes={s['source_archive_path']:s['source_archive_sha256'] for s in specs},all_input_bytes_exact=True,
        synthetic_inputs_read=True,independent_synthetic_reference=False,**SCOPE)
    if (any(verification.get(k)!=v for k,v in expected.items()) or done['passed'] is not True
            or done['verification_sha256']!=C.sha(out/'input_verification.json')
            or done['independent_synthetic_reference'] is not False or done['synthetic_inputs_read'] is not True
            or not seal['finished_unix']<=verification['started_unix']<=verification['finished_unix']<=done['finished_unix']):
        raise ValueError('complete independent input audit required')
    checks=C.read(out/'input_audit/checks.json')
    if ([r['id'] for r in checks]!=[s['id'] for s in specs] or not all(r['passed'] and r['canonical_input_bytes_exact']
            and r['descriptor_bytes_exact'] and r['synthetic_inputs_read'] for r in checks)):
        raise ValueError('all23 exact descriptor/input checks required')
    for row in checks:
        scope_check(row)
        if row['independent_synthetic_reference'] is not False:raise ValueError('input audit cannot execute references')
    return lock,seal,rows,specs,verification


WORK_KEYS={'cell_node_evaluations','elementary_inner_nodes','inner_x_nodes','narrow_y_pdf_nodes','inner_panels',
           'depth_evaluations','outer_intervals','outer_geometry_breakpoints','pilot_depth_evaluations','quadrature_depth_evaluations'}
SUCCESS_ARRAYS={'log_raw_mass','log_raw_fov_mass','log_floored_mass','log_probability','probability'}
OUTER_ARRAYS={'initial_depth_edges','pilot_zeta','pilot_log_integrand','outer_log_scale','outer_scaled_integral'}


def independent_depth(mu,factor):
    """Scalar depth law only; no reference Gaussian preparation/density/integration."""
    rms=math.hypot(float(mu[2]),*map(float,factor[2]))
    if rms==0:return dict(structural_branch='zero_depth',structural_zero=True,rms_depth=0.,
        normalized_depth_mean=0.,normalized_depth_std=0.,front_mass=0.,log_front_mass=None)
    m=float(mu[2]/rms);sigma=math.hypot(*map(float,factor[2]/rms))
    zero=sigma==0 and m<=0
    branch='nonpositive_constant_depth' if zero else ('positive_constant_depth' if sigma==0 else 'stochastic_depth')
    lf=None if zero else (0. if sigma==0 else float(log_ndtr(m/sigma)))
    front=0. if zero else math.exp(lf)
    return dict(structural_branch=branch,structural_zero=zero,rms_depth=rms,
        normalized_depth_mean=m,normalized_depth_std=sigma,front_mass=front,log_front_mass=lf)


def work_check(work,policy,*,require_budget=True):
    if (set(work)!=WORK_KEYS or any(type(v) is not int or v<0 for v in work.values())
            or work['cell_node_evaluations']!=work['elementary_inner_nodes']
            or work['cell_node_evaluations']!=work['inner_x_nodes']+work['narrow_y_pdf_nodes']
            or work['depth_evaluations']!=work['pilot_depth_evaluations']+work['quadrature_depth_evaluations']
            or (require_budget and (work['cell_node_evaluations']>policy['max_call_cell_node_evaluations']
                or work['depth_evaluations']>policy['maximum_depth_evaluations']
                or work['outer_intervals']>policy['maximum_depth_intervals']))):
        raise ValueError('complete charged reference work/budget arithmetic differs')
    return True


def log_same(actual,expected,tolerance=2e-12):
    if expected is None:return actual is None
    return isinstance(actual,(float,int)) and not isinstance(actual,bool) and math.isfinite(actual) and abs(actual-expected)<=tolerance


def physical_depth_tail(info,cutoff):
    """Log positive-depth mass omitted outside [-cutoff,cutoff], once per FOV."""
    if info['structural_zero'] or info['normalized_depth_std']==0:return -np.inf
    threshold=-info['normalized_depth_mean']/info['normalized_depth_std']
    right=float(log_ndtr(-max(cutoff,threshold)))
    left=-np.inf
    if threshold < -cutoff:
        high=float(log_ndtr(-cutoff));low=float(log_ndtr(threshold))
        left=high+math.log(-math.expm1(low-high)) if low<high else -np.inf
    return float(np.logaddexp(left,right))


COALESCING_ARRAYS={'original_depth_edges','dropped_depth_edge_indices','dropped_depth_edge_values'}
COALESCING_RULE='left_to_right_strict_midpoint_then_upper_backtrack'


def expected_depth_partition(original):
    """Independent order-preserving partition walk; no density or integrator."""
    edges=np.asarray(original)
    if (edges.dtype!=np.float64 or edges.ndim!=1 or len(edges)<2
            or not np.isfinite(edges).all() or np.any(edges[1:]<=edges[:-1])):
        raise ValueError('finite strictly increasing original depth partition required')
    def interior(left,right):
        midpoint=(float(left)+float(right))/2.
        return math.isfinite(midpoint) and left<midpoint<right
    kept=[0];dropped=[]
    for index in range(1,len(edges)-1):
        if interior(edges[kept[-1]],edges[index]):kept.append(index)
        else:dropped.append(dict(original_index=index,value=float(edges[index]),reason='no_strict_midpoint_from_previous_retained'))
    while len(kept)>1 and not interior(edges[kept[-1]],edges[-1]):
        index=kept.pop();dropped.append(dict(original_index=index,value=float(edges[index]),reason='no_strict_midpoint_to_upper'))
    whole_valid=interior(edges[kept[-1]],edges[-1])
    kept.append(len(edges)-1)
    retained=edges[np.asarray(kept,np.int64)].copy()
    indices=np.asarray([v['original_index'] for v in dropped],np.int64)
    values=edges[indices].copy()
    report=dict(rule=COALESCING_RULE,original_interior_count=len(edges)-2,
        retained_interior_count=len(retained)-2,dropped_count=len(dropped),dropped=dropped)
    return dict(initial_depth_edges=retained,dropped_depth_edge_indices=indices,
        dropped_depth_edge_values=values,depth_edge_coalescing=report,strict_midpoints_exist=whole_valid)


def coalescing_check(row,bank,info,reference_policy,*,require_success=True):
    variable=info['structural_branch']=='stochastic_depth';present=set(bank)&COALESCING_ARRAYS
    if not variable:
        if present or row.get('depth_edge_coalescing') is not None:
            raise ValueError('nonvariable branch cannot invent depth coalescing')
        return dict(passed=True,applicable=False,dropped_count=0)
    if present!=COALESCING_ARRAYS or 'initial_depth_edges' not in bank:
        raise ValueError('original, retained and dropped depth evidence required')
    original=bank['original_depth_edges'];expected=expected_depth_partition(original)
    for key in ('initial_depth_edges','dropped_depth_edge_indices','dropped_depth_edge_values'):
        if C.array_sha(bank[key])!=C.array_sha(expected[key]):raise ValueError('deterministic depth coalescing differs: '+key)
    if row.get('depth_edge_coalescing')!=expected['depth_edge_coalescing']:
        raise ValueError('depth coalescing index/value/reason/count records differ')
    if row['work']['outer_geometry_breakpoints']!=expected['depth_edge_coalescing']['retained_interior_count']:
        raise ValueError('outer work must report retained interior cuts')
    cutoff=reference_policy['depth_cutoff'];threshold=-info['normalized_depth_mean']/info['normalized_depth_std']
    if not np.isclose(original[0],max(-cutoff,threshold),rtol=2e-13,atol=0) or original[-1]!=cutoff:
        raise ValueError('coalescing changed original positive-depth domain endpoints')
    if require_success and not expected['strict_midpoints_exist']:
        raise ValueError('whole domain lacks a representable strict midpoint')
    return dict(passed=True,applicable=True,original_interior_count=len(original)-2,
        retained_interior_count=len(bank['initial_depth_edges'])-2,dropped_count=len(bank['dropped_depth_edge_indices']),
        complete_original_domain_preserved=True,strict_midpoints_exist=expected['strict_midpoints_exist'],
        original_geometry_root_generation_not_recomputed=True)


def scale_check(row,bank,info,reference_policy,tolerance):
    n=len(bank['input__bounds']);variable=info['structural_branch']=='stochastic_depth'
    coalescing=coalescing_check(row,bank,info,reference_policy)
    present=set(bank)&OUTER_ARRAYS
    if not variable:
        if present or row['scaled_outer_error_estimate'] is not None or row['outer_convergence_success'] is not True:
            raise ValueError('constant/structural branch must not invent outer work or scales')
        if any(row['work'][k] for k in ('depth_evaluations','pilot_depth_evaluations','quadrature_depth_evaluations','outer_intervals','outer_geometry_breakpoints')):
            raise ValueError('nonvariable depth must not report pilot/outer work')
        return dict(passed=True,variable_depth=False,scale_restoration_error=0.,pilot_count=0,coalescing=coalescing)
    if present!=OUTER_ARRAYS:raise ValueError('complete fixed-pilot scale evidence required')
    edges,zeta,pilot,scale,scaled=[bank[k] for k in ('initial_depth_edges','pilot_zeta','pilot_log_integrand','outer_log_scale','outer_scaled_integral')]
    k=len(zeta)
    if (any(x.dtype!=np.float64 for x in (edges,zeta,pilot,scale,scaled)) or edges.shape!=(k+1,) or zeta.shape!=(k,)
            or pilot.shape!=(k,n+1) or scale.shape!=(n+1,) or scaled.shape!=(n+1,) or k<1
            or any(not np.isfinite(x).all() for x in (edges,zeta,pilot,scale,scaled))
            or np.any(np.diff(edges)<=0) or np.any(scaled<=0)
            or row['work']['pilot_depth_evaluations']!=k or row['work']['outer_geometry_breakpoints']!=k-1
            or row['work']['quadrature_depth_evaluations']<=0 or row['work']['outer_intervals']<k):
        raise ValueError('pilot/scaled-integral shapes, positivity or charged work differ')
    cutoff=reference_policy['depth_cutoff'];threshold=-info['normalized_depth_mean']/info['normalized_depth_std']
    if not np.isclose(edges[0],max(-cutoff,threshold),rtol=2e-13,atol=0) or edges[-1]!=cutoff:
        raise ValueError('fixed physical positive-depth integration interval differs')
    midpoint=(edges[:-1]+edges[1:])/2.
    if C.array_sha(zeta)!=C.array_sha(midpoint) or np.any(zeta<=threshold) or not np.array_equal(scale,np.max(pilot,axis=0)):
        raise ValueError('pilot midpoint or fixed per-component maximum scale changed')
    if row['outer_convergence_success'] is not True or not math.isfinite(row['scaled_outer_error_estimate']) or row['scaled_outer_error_estimate']<0:
        raise ValueError('successful outer convergence and finite scaled error required')
    restored=np.log(scaled)+scale;actual=np.r_[bank['log_raw_mass'],bank['log_raw_fov_mass']]
    error=float(np.max(abs(restored-actual)))
    return dict(passed=bool(error<=tolerance),variable_depth=True,scale_restoration_error=error,pilot_count=k,
        scaled_error_is_unscaled_raw_certificate=False,front_mass_not_applied_to_restoration=True,coalescing=coalescing)


def outer_loss_check(row,bank,info):
    report=row['outer_exp_diagnostics'];n=len(bank['input__bounds']);variable=info['structural_branch']=='stochastic_depth'
    if (report['local_node_omission_only'] is not True or report['is_continuous_integral_or_pipeline_error_bound'] is not False
            or report['scale_reselected'] is not False or report['scale_is_certified_upper_bound'] is not False):
        raise ValueError('outer exponential omissions must remain local diagnostics')
    for key in ('evaluation_vectors','component_evaluations','total_underflow_terms'):
        if type(report[key]) is not int or report[key]<0:raise ValueError('nonnegative outer loss counters required')
    present=set(bank)&OUTER_DIAGNOSTIC_ARRAYS
    if not variable:
        if present or any(report[k] for k in ('evaluation_vectors','component_evaluations','total_underflow_terms')):
            raise ValueError('nonvariable branch cannot invent outer diagnostics')
        return dict(passed=True,evaluation_vectors=0,total_underflow_terms=0,local_only=True)
    if present!=OUTER_DIAGNOSTIC_ARRAYS:raise ValueError('complete outer loss arrays required')
    counts,maximum_lost,maximum=[bank[n] for n in ('outer_underflow_counts','outer_max_dropped_log_shift','outer_max_log_shift')]
    evaluations=report['evaluation_vectors']
    if (counts.dtype!=np.int64 or counts.shape!=(n+1,) or np.any(counts<0) or np.any(counts>evaluations)
            or maximum_lost.dtype!=np.float64 or maximum_lost.shape!=(n+1,)
            or maximum.dtype!=np.float64 or maximum.shape!=(n+1,) or not np.isfinite(maximum).all()
            or not 0<evaluations<=row['work']['quadrature_depth_evaluations']
            or report['component_evaluations']!=evaluations*(n+1)
            or report['total_underflow_terms']!=sum(map(int,counts))
            or not np.isneginf(maximum_lost[counts==0]).all() or not np.isfinite(maximum_lost[counts>0]).all()
            or np.any(maximum_lost>maximum)):
        raise ValueError('outer local underflow evidence/counts differ')
    with np.errstate(under='ignore'):
        if np.any(np.exp(maximum_lost[counts>0])!=0.):raise ValueError('reported underflow shift is representable')
    return dict(passed=True,evaluation_vectors=evaluations,total_underflow_terms=report['total_underflow_terms'],
        local_only=True,continuous_integral_or_pipeline_certificate=False)


def diagnostic_matches(actual,logvalue):
    expected=exp_diagnostic(logvalue)
    if expected is None:return actual is None
    return isinstance(actual,(float,int)) and not isinstance(actual,bool) and math.isfinite(actual) and math.isclose(actual,expected,rel_tol=3e-12,abs_tol=0.)


def tail_check(row,bank,info,policy,reference_policy,gates):
    tail=physical_depth_tail(info,reference_policy['depth_cutoff']);zero=info['structural_zero'];raw=bank['log_raw_mass']
    lr=-np.inf if zero else tail-float(logsumexp(raw))
    bounds,bias=bank['input__bounds'],bank['input__content_bias'];floor=np.prod(bounds[:,2:]-bounds[:,:2],axis=1)*policy['floor_density']
    weighted=float(logsumexp(np.log(floor)+bias-bias.max()))
    lp=-np.inf if tail==-np.inf else min(0.,math.log(2.)+tail-weighted)
    actual=row['truncation_bounds'];pairs=(('depth_omitted_mass',tail),('native_raw_l1_absolute',tail),('direct_fov_absolute',tail),
        ('raw_l1_over_reference_fov',lr),('probability_max_absolute',lp))
    metadata=bool(actual['native_union_bound'] is True and actual['inner_x_truncation_used'] is False
        and actual['structural_zero_exemption'] is zero and actual['excludes_quadrature_and_roundoff'] is True)
    for name,value in pairs:
        metadata=metadata and log_same(actual['log_'+name],finite_or_none(value))
        if name=='raw_l1_over_reference_fov' and zero:metadata=metadata and actual[name] is None
        else:metadata=metadata and diagnostic_matches(actual[name],value)
    passed=bool(metadata and (zero or (lr<=math.log(gates['truncation_raw_l1_relative']) and lp<=math.log(gates['truncation_probability_absolute']))))
    return dict(passed=passed,metadata_consistent=metadata,log_depth_omitted_mass=finite_or_none(tail),
        log_raw_l1_over_reference_fov=finite_or_none(lr),log_probability_max_absolute=finite_or_none(lp),
        bound_excludes_quadrature_and_roundoff=True,raw_relative_denominator_is_computed_not_enclosed=True)


LSE_SCOPES=('narrow_normal_nodes','inner_panel_nodes','rectangle_panels','native_total','floor','floor_total','floored_total','content_normalization')


def local_lse_loss(terms):
    terms=np.asarray(terms,np.float64);finite=terms[np.isfinite(terms)]
    if not len(finite):return dict(sum_count=1,underflow_sum_count=0,underflow_term_count=0,
        maximum_log_scaled_omitted_bound=None,maximum_log_absolute_omitted_bound=None,maximum_log_relative_omitted_bound=None)
    peak=float(finite.max());shift=finite-peak
    with np.errstate(under='ignore'):values=np.exp(shift)
    lost=values==0;n=int(lost.sum());result=dict(sum_count=1,underflow_sum_count=int(n>0),underflow_term_count=n,
        maximum_log_scaled_omitted_bound=None,maximum_log_absolute_omitted_bound=None,maximum_log_relative_omitted_bound=None)
    if n:
        bound=math.log(n)+float(shift[lost].max());retained=math.log(float(values.sum()))
        result.update(maximum_log_scaled_omitted_bound=bound,maximum_log_absolute_omitted_bound=bound+peak,
                      maximum_log_relative_omitted_bound=bound-retained)
    return result


def combine_loss_reports(reports):
    result={k:sum(r[k] for r in reports) for k in ('sum_count','underflow_sum_count','underflow_term_count')}
    for k in ('maximum_log_scaled_omitted_bound','maximum_log_absolute_omitted_bound','maximum_log_relative_omitted_bound'):
        values=[r[k] for r in reports if r[k] is not None];result[k]=max(values) if values else None
    return result


def lse_check(row,bank,policy,tolerance):
    report=row['lse_diagnostics'];scopes=report['by_scope']
    if (set(scopes)!=set(LSE_SCOPES) or report['is_pipeline_error_bound'] is not False
            or report['exact_arithmetic_local_omission_only'] is not True
            or report['excludes_addition_absorption_density_quadrature_and_roundoff'] is not True):
        raise ValueError('local-only LSE scope differs')
    for name,value in scopes.items():
        if (any(type(value[k]) is not int or value[k]<0 for k in ('sum_count','underflow_sum_count','underflow_term_count'))
                or value['underflow_sum_count']>value['sum_count'] or value['underflow_sum_count']>value['underflow_term_count']
                or bool(value['underflow_sum_count'])!=bool(value['underflow_term_count'])):
            raise ValueError('local LSE counts invalid')
        for key in ('maximum_log_scaled_omitted_bound','maximum_log_absolute_omitted_bound','maximum_log_relative_omitted_bound'):
            if value['underflow_term_count']:
                if not isinstance(value[key],(float,int)) or not math.isfinite(value[key]):raise ValueError('local omitted term log bound required')
            elif value[key] is not None:raise ValueError('no-loss sum must not invent a lost-term bound')
    raw=bank['log_raw_mass'];b=bank['input__bounds'];bias=bank['input__content_bias'];floor=np.log(np.prod(b[:,2:]-b[:,:2],axis=1)*policy['floor_density'])
    expected=dict(native_total=local_lse_loss(raw),floor=combine_loss_reports([local_lse_loss(np.asarray([x,y])) for x,y in zip(raw,floor)]),
        floor_total=local_lse_loss(floor),floored_total=local_lse_loss(bank['log_floored_mass']),
        content_normalization=local_lse_loss(bank['log_floored_mass']+bias))
    for name,value in expected.items():
        for key,ref in value.items():
            actual=scopes[name][key]
            if ref is None or isinstance(ref,int):good=actual==ref
            else:good=actual is not None and abs(actual-ref)<=tolerance
            if not good:raise ValueError('independent saved-input LSE report differs: '+name+'/'+key)
    return dict(passed=True,saved_input_scopes_independently_recomputed=list(expected),
        inner_integral_scopes_not_replayed=['narrow_normal_nodes','inner_panel_nodes','rectangle_panels'],
        local_bounds_are_total_error_certificate=False)


def reference_check(row,bank,policy,reference_policy,gates):
    """Recompute depth metadata, scaling, floor/P and physical tail; no integral."""
    if not row['passed']:return dict(passed=False,status='reference_failed',failure_reason=row.get('failure_reason'))
    n=len(bank['input__bounds']);info=independent_depth(bank['input__mu'],bank['input__factor']);zero=info['structural_zero']
    raw=validate_logs(bank['log_raw_mass'],zero);validate_logs(bank['log_raw_fov_mass'],zero)
    metadata=bool(row['method']=='independent-SVD-fullX-log-conditional-pilot' and row['order'] in (32,48)
        and row['reference_algorithm_is_unchanged_R27'] is False and row['inner_x_truncation_used'] is False
        and row['quadrature_error_bound_available'] is False
        and row['tail_truncation_used'] is (info['structural_branch']=='stochastic_depth')
        and isinstance(row['maximum_full_x_panel_width'],(float,int)) and math.isfinite(row['maximum_full_x_panel_width'])
        and 0<=row['maximum_full_x_panel_width']<=reference_policy['standard_x_panel_width']+2e-12
        and type(row['normal_logcdf_scalar_evaluations']) is int and row['normal_logcdf_scalar_evaluations']>=0)
    for name,value in info.items():
        actual=row.get(name)
        if isinstance(value,(str,bool)) or value is None:metadata=metadata and actual==value
        else:metadata=metadata and isinstance(actual,(float,int)) and math.isfinite(actual) and math.isclose(actual,value,rel_tol=2e-12,abs_tol=0.)
    work_check(row['work'],reference_policy)
    expected=probabilities(raw,bank['input__bounds'],bank['input__content_bias'],policy['floor_density'])
    for key in ('log_floored_mass','log_probability','probability'):
        if bank[key].dtype!=np.float64 or bank[key].shape!=(n,) or not np.isfinite(bank[key]).all():raise ValueError('finite complete probability output schema required')
    if np.any(bank['probability']<0):raise ValueError('negative saved probability')
    errors={key:float(np.max(abs(bank[key]-expected[key]))) for key in ('log_floored_mass','log_probability','probability')}
    norm=abs(math.fsum(map(float,bank['probability']))-1.)
    with np.errstate(under='ignore'):exp_error=float(np.max(abs(np.exp(bank['log_probability'])-bank['probability'])))
    tol=gates['normalization_absolute'];direct=direct_fov_check(raw,bank['log_raw_fov_mass'],zero,gates['direct_fov_additivity_absolute'])
    scale=scale_check(row,bank,info,reference_policy,tol);tail=tail_check(row,bank,info,policy,reference_policy,gates)
    local=lse_check(row,bank,policy,tol);outer=outer_loss_check(row,bank,info)
    floor=expected['log_floor_mass']
    for key,value in dict(log_raw_mass_sum=None if zero else float(logsumexp(raw)),
            log_total_area_floor=float(logsumexp(floor)),log_normalizer=float(logsumexp(bank['log_floored_mass'])),
            log_content_normalizer=expected['log_content_normalizer']).items():
        metadata=metadata and log_same(row[key],value,tol)
    metadata=metadata and abs(row['normalization_error']-norm)<=tol
    declared_direct=-np.inf if zero else float(log_absolute_difference(row['log_raw_mass_sum'],bank['log_raw_fov_mass'][0]))
    metadata=metadata and log_same(row['direct_fov_log_absolute_error'],finite_or_none(declared_direct),tol)
    metadata=metadata and diagnostic_matches(row['direct_fov_absolute_error_materialized_diagnostic'],declared_direct)
    if zero:metadata=metadata and all(v==0 for v in row['work'].values())
    passed=bool(metadata and max(norm,exp_error,*errors.values())<=tol and direct['passed'] and scale['passed'] and tail['passed'] and local['passed'] and outer['passed'])
    return dict(passed=passed,status='checked',metadata_consistent=bool(metadata),structural_zero=zero,
        normalization_error=norm,saved_probability_exp_error=exp_error,arithmetic_errors=errors,direct_fov=direct,
        scale_restoration=scale,depth_tail=tail,lse_diagnostics=local,outer_exp_diagnostics=outer,independent_probability_integral_replayed=False)


def verify_forward(run,out):
    lock,prepared,preprows,specs,input_verification=verify_input_audit(run,out)
    forward=run/'forward';done=complete_inventory(forward);rootdone=complete_inventory(run)
    seal=C.read(run/'forward_complete.json');rows=C.read(forward/'records.json');calls=ordered_calls(specs)
    expected=dict(producer_lock_sha256=C.sha(run/'lock.json'),audit_lock_sha256=C.sha(out/'lock.json'),
        producer_prepared_complete_sha256=C.sha(run/'prepared_complete.json'),
        input_verification_sha256=C.sha(out/'input_verification.json'),query_specs_sha256=lock['query_specs_sha256'],
        oracle_specs_sha256=lock['oracle_specs_sha256'],forward_records_sha256=C.sha(forward/'records.json'),
        forward_completion_sha256=C.sha(forward/'completion.json'),case_ids=[s['id'] for s in specs],
        ordered_call_ids=[x[2] for x in calls],case_count=23,call_count=46,all_case_attempts_retained=True,
        all_available_partial_arrays_retained=True,**SCOPE)
    if any(seal.get(k)!=v for k,v in expected.items()) or [(r['id'],r['order'],r['call_id']) for r in rows]!=calls:
        raise ValueError('complete ordered46 new-reference seal required')
    start=C.read(forward/'started.json')
    for stage in (seal,start,done,rootdone):scope_check(stage)
    phase=seal['phase_wall_seconds'];total=seal['total_cell_node_evaluations']
    wall_passed=math.isfinite(phase) and phase>=0 and phase<=lock['reference_policy']['max_phase_wall_seconds']
    node_passed=type(total) is int and 0<=total<=lock['reference_policy']['max_total_cell_node_evaluations'] and all(r['reference_work_consistent'] for r in rows)
    completed=wall_passed and node_passed and all(r['attempted'] and r['passed'] for r in rows)
    if (not math.isfinite(phase) or phase<0 or start['started_unix']!=seal['started_unix']
            or any(start[k]!=expected[k] for k in ('producer_lock_sha256','audit_lock_sha256','input_verification_sha256'))
            or not input_verification['finished_unix']<=seal['started_unix']<=seal['phase_finished_unix']<=seal['finished_unix']
            or rootdone['forward_complete_sha256']!=C.sha(run/'forward_complete.json')
            or done['complete_case_inventory'] is not True or rootdone['complete_case_inventory'] is not True
            or start['synthetic_inputs_read'] is not False or start['independent_synthetic_reference'] is not False):
        raise ValueError('reference phase chronology or completion differs')
    for value in (seal,done,rootdone):
        for key,expected_value in dict(phase_wall_passed=wall_passed,phase_node_budget_passed=node_passed,
                reference_calls_completed=completed,total_cell_node_evaluations=total,
                synthetic_inputs_read=any(r['synthetic_inputs_read'] for r in rows),
                independent_synthetic_reference=any(r['attempted'] for r in rows)).items():
            if value[key]!=expected_value:raise ValueError('reference phase resources/actual status differs: '+key)
    files={'started.json','records.json','call_starts.jsonl','reference_results.jsonl'}|{'arrays/'+c[2]+'.npz' for c in calls}
    if set(done['files'])!=files:raise ValueError('reference inventory membership differs')
    rootfiles={'lock.json','lock.sha256','query_specs.json','oracle_specs.json','prepared_complete.json','forward_complete.json'}
    rootfiles|={'source_snapshot/'+n for n in lock['source_hashes']}|{'prepared/'+n for n in C.read(run/'prepared/completion.json')['files']}
    rootfiles|={'forward/'+n for n in files}|{'prepared/completion.json','forward/completion.json'}
    if set(rootdone['files'])!=rootfiles:raise ValueError('complete producer inventory membership differs')
    starts=[json.loads(s) for s in (forward/'call_starts.jsonl').read_text().splitlines()]
    journal=[json.loads(s) for s in (forward/'reference_results.jsonl').read_text().splitlines()]
    if ([r['call_id'] for r in starts]!=[r['call_id'] for r in rows if r['attempted']]
            or [r['call_id'] for r in journal]!=expected['ordered_call_ids'] or seal['attempted_calls']!=len(starts)):
        raise ValueError('actual-reference and result journals differ')
    starts={r['call_id']:r for r in starts};prepmap={r['id']:r for r in preprows};specmap={s['id']:s for s in specs}
    previous=seal['started_unix'];used=0
    for row,record in zip(rows,journal):
        scope_check(row);scope_check(record);prep=prepmap[row['id']];spec=specmap[row['id']]
        if (row['prepared_archive_path']!=prep['archive_path'] or row['prepared_archive_sha256']!=prep['archive_sha256']
                or row['policy_overrides']!=spec['policy_overrides'] or row['archive_path']!='forward/arrays/'+row['call_id']+'.npz'
                or C.sha(run/row['archive_path'])!=row['archive_sha256']
                or not previous<=row['started_unix']<=row['finished_unix']<=seal['phase_finished_unix']):
            raise ValueError('reference row identity/archive/chronology differs')
        previous=row['finished_unix']
        for key in ('budget_used_before','budget_used_after','charged_cell_node_evaluations'):
            if type(row[key]) is not int or row[key]<0:raise ValueError('actual budget ledger must be nonnegative integer')
        if (row['budget_used_before']!=used or row['budget_used_after']-used!=row['charged_cell_node_evaluations']
                or row['reference_work_consistent'] is not (row['work'].get('cell_node_evaluations',0)==row['charged_cell_node_evaluations'])):
            raise ValueError('actual shared node ledger continuity differs')
        used=row['budget_used_after']
        if row['attempted']:
            s=starts[row['call_id']];scope_check(s)
            if (s['id']!=row['id'] or s['order']!=row['order'] or s['budget_used_before']!=row['budget_used_before']
                    or not row['started_unix']<=s['started_unix']<=row['finished_unix']
                    or not s['synthetic_inputs_read'] or not s['independent_synthetic_reference']):
                raise ValueError('actual reference-start evidence differs')
        if (record['available_array_sha256']!=row['array_sha256']
                or any(row.get(k)!=v for k,v in record.items() if k!='available_array_sha256')):
            raise ValueError('reference result/work/available-array journal changed')
    if used!=total:raise ValueError('total charged nodes differ from retained ordered ledger')
    return lock,seal,rows,preprows,specs


OUTER_DIAGNOSTIC_ARRAYS={'outer_underflow_counts','outer_max_dropped_log_shift','outer_max_log_shift'}


def validate_call(row,bank,prepared,reference_policy):
    scope_check(row)
    if (type(row.get('passed')) is not bool or type(row.get('attempted')) is not bool
            or row['status']!=('complete' if row['passed'] else 'failed')
            or row['independent_synthetic_reference'] is not row['attempted']
            or not isinstance(row.get('work'),dict)):
        raise ValueError('explicit coherent call outcome/scope/work required')
    if not row['passed'] and (not isinstance(row.get('failure_reason'),str) or not row['failure_reason'].strip()):
        raise ValueError('failed reference must retain reason')
    for name in ('wall_seconds','call_and_input_load_seconds'):
        if not isinstance(row.get(name),(float,int)) or isinstance(row[name],bool) or not math.isfinite(row[name]) or row[name]<0:
            raise ValueError('actual finite nonnegative call time required')
    if type(row.get('peak_rss_kib_process')) is not int or row['peak_rss_kib_process']<0:raise ValueError('process high-water RSS required')
    inputs=set(bank)&set(INPUT_FIELDS);outputs=set(bank)-set(INPUT_FIELDS)
    if inputs and inputs!=set(INPUT_FIELDS):raise ValueError('retained input copies must contain all four fields')
    if (inputs or row['attempted']) and not row['synthetic_inputs_read']:raise ValueError('actual decoded input scope required')
    for name in inputs:
        if C.array_sha(bank[name])!=C.array_sha(prepared[name]):raise ValueError('reference input bytes changed after input audit')
    if not row['attempted']:
        if outputs or any(row['work'].values()) or row['charged_cell_node_evaluations'] or row['passed']:
            raise ValueError('unattempted reference cannot invent work/results')
    else:work_check(row['work'],reference_policy,require_budget=row['passed'])
    n=len(prepared['input__bounds']);info=independent_depth(prepared['input__mu'],prepared['input__factor'])
    partials={'partial_log_raw_mass','partial_log_raw_fov_mass','partial_raw_computed','pilot_completed','last_conditional_log_mass'}
    allowed=SUCCESS_ARRAYS|OUTER_ARRAYS|OUTER_DIAGNOSTIC_ARRAYS|COALESCING_ARRAYS|partials
    if not outputs<=allowed:raise ValueError('unlisted available reference array: '+str(outputs-allowed))
    if row['passed']:
        expected=SUCCESS_ARRAYS|(OUTER_ARRAYS|OUTER_DIAGNOSTIC_ARRAYS|COALESCING_ARRAYS if info['structural_branch']=='stochastic_depth' else set())
        if not row['attempted'] or inputs!=set(INPUT_FIELDS) or outputs!=expected:
            raise ValueError('complete branch-specific reference success schema required')
    for name in outputs:
        a=bank[name]
        dtype=np.int64 if name in ('outer_underflow_counts','dropped_depth_edge_indices') else (np.bool_ if name in ('partial_raw_computed','pilot_completed') else np.float64)
        if a.dtype!=dtype:raise ValueError('reference array dtype differs: '+name)
        if name in SUCCESS_ARRAYS or name in ('partial_log_raw_mass','partial_log_raw_fov_mass'):
            shape=(1,) if name in ('log_raw_fov_mass','partial_log_raw_fov_mass') else (n,)
        elif name in ('outer_log_scale','outer_scaled_integral','outer_underflow_counts','outer_max_dropped_log_shift','outer_max_log_shift','last_conditional_log_mass','partial_raw_computed'):shape=(n+1,)
        elif name in ('initial_depth_edges','original_depth_edges','dropped_depth_edge_indices','dropped_depth_edge_values','pilot_zeta','pilot_completed'):
            if a.ndim!=1:raise ValueError('pilot/edge vector required')
            shape=a.shape
        else:
            if a.ndim!=2 or a.shape[1]!=n+1:raise ValueError('pilot matrix width differs')
            shape=a.shape
        if a.shape!=shape:raise ValueError('reference available shape differs: '+name)
        # Failed arithmetic can produce NaN/inf/negative scaled values; retain
        # their actual bytes as failure evidence, never validate them as a result.
        if row['passed'] and a.dtype==np.float64 and (np.isnan(a).any() or np.isposinf(a).any()):
            raise ValueError('successful reference has noncanonical logs: '+name)
        if name=='outer_underflow_counts' and np.any(a<0):raise ValueError('negative outer loss count')
    if 'partial_raw_computed' in bank:
        if not {'partial_log_raw_mass','partial_log_raw_fov_mass'}<=outputs:raise ValueError('partial raw mask requires both raw vectors')
        logs=np.r_[bank['partial_log_raw_mass'],bank['partial_log_raw_fov_mass']]
        if not np.isneginf(logs[~bank['partial_raw_computed']]).all():raise ValueError('uncomputed raw entries must remain absent -inf')
    if 'pilot_completed' in bank:
        if 'pilot_zeta' not in bank or bank['pilot_completed'].shape!=bank['pilot_zeta'].shape or 'pilot_log_integrand' not in bank:
            raise ValueError('partial pilot mask shape differs')
        if bank['pilot_log_integrand'].shape[0]!=len(bank['pilot_completed']):raise ValueError('partial pilot matrix height differs')
        if not np.isneginf(bank['pilot_log_integrand'][~bank['pilot_completed']]).all():raise ValueError('uncomputed pilot rows must remain absent -inf')
    coalescing=None
    if COALESCING_ARRAYS<=outputs and 'initial_depth_edges' in bank and row.get('depth_edge_coalescing') is not None:
        original=bank['original_depth_edges']
        original_valid=(original.ndim==1 and len(original)>=2 and np.isfinite(original).all() and np.all(original[1:]>original[:-1]))
        if row['passed'] or original_valid:
            coalescing=coalescing_check(row,bank,info,reference_policy,require_success=row['passed'])
        else:
            coalescing=dict(passed=False,valid_original_partition=False,retained_failure_evidence=True)
    return dict(delivery_integrity_passed=True,attempted=row['attempted'],reference_completed=row['passed'],
        available_output_arrays=sorted(outputs),reference_work_consistent=row['reference_work_consistent'],coalescing=coalescing)


def read_oracle(spec,on_decode=None):
    if C.sha(spec['source_archive_path'])!=spec['source_archive_sha256']:raise ValueError('accepted old oracle archive changed')
    wanted=('log_raw_mass','log_raw_fov_mass','probability')
    fields={n:spec['source_fields'][n] for n in wanted};hashes={fields[n]:spec['array_sha256'][n] for n in wanted}
    stored=read_arrays(spec['source_archive_path'],hashes,fields.values(),on_decode)
    return {n:stored[field] for n,field in fields.items()}


def compare_saved(actual,reference,prepared,policy,gates,strict):
    info=independent_depth(prepared['input__mu'],prepared['input__factor'])
    return compare_log_raw(actual['log_raw_mass'],reference['log_raw_mass'],prepared['input__bounds'],prepared['input__content_bias'],
        policy['floor_density'],gates,reference_comparison=strict,archived_probability=actual['probability'],
        reference_probability=reference['probability'],structural_zero=info['structural_zero'])


def comparison_plan(specs,oracles):
    result=[]
    for spec in specs:
        name=spec['id'];result.append(dict(id=name,kind='new_orders',actual_key=(name,32),reference_key=(name,48)))
        for order in (32,48):
            for old in (o for o in oracles if o['id']==name):
                result.append(dict(id=name,kind='new_vs_old_reference',new_order=order,old_order=old['order'],
                    actual_key=(name,order),reference_key=(name,old['order'])))
            result.append(dict(id=name,kind='old_production_vs_new',new_order=order,actual_key=name,reference_key=(name,order)))
    return result


def audit(run,out):
    if any((out/n).exists() for n in ('audit_started.json','verification.json','summary.json','completion.json')):
        raise ValueError('preserve every prior final audit attempt')
    started=time.time();C.write(out/'audit_started.json',dict(started_unix=started,synthetic_inputs_read=False,
        independent_synthetic_reference=False,reference_integrations_executed_by_auditor=0,**SCOPE))
    decoded={'value':False};local={'value':False};reference=False;checks=[];comparisons=[]
    def note():decoded['value']=True;local['value']=True
    try:
        lock,seal,rows,preprows,specs=verify_forward(run,out)
        if started<seal['finished_unix']:raise ValueError('final audit/old output decode must follow global reference seal')
        reference=bool(seal['attempted_calls']);prepmap={r['id']:r for r in preprows};specmap={s['id']:s for s in specs}
        prepared={};new={};checked={}
        for row in rows:
            local={'value':False}
            key=(row['id'],row['order']);spec=specmap[row['id']]
            try:
                prep=prepmap[row['id']]
                if row['id'] not in prepared:prepared[row['id']]=read_arrays(run/prep['archive_path'],prep['array_sha256'],on_decode=note)
                bank=read_arrays(run/row['archive_path'],row['array_sha256'],on_decode=note);new[key]=bank
                check=validate_call(row,bank,prepared[row['id']],lock['reference_policy'])
                policy=dict(lock['policy'],**spec['policy_overrides'])
                arithmetic=reference_check(row,bank,policy,lock['reference_policy'],lock['gates'])
                check.update(arithmetic=arithmetic,arithmetic_passed=arithmetic['passed'])
            except Exception as error:check=dict(delivery_integrity_passed=False,arithmetic_passed=False,
                failure_reason=type(error).__name__+': '+str(error))
            checked[key]=check
            checks.append(dict(id=row['id'],order=row['order'],call_id=row['call_id'],**check,
                synthetic_inputs_read=local['value'],independent_synthetic_reference=row['attempted'],**SCOPE))
            C.write(out/'checks.json',checks)
        oracles=C.read(run/'oracle_specs.json');old={};production={};oracle_failures=[]
        for spec in oracles:
            try:old[(spec['id'],spec['order'])]=read_oracle(spec,note)
            except Exception as error:oracle_failures.append(dict(id=spec['id'],order=spec['order'],failure_reason=type(error).__name__+': '+str(error)))
        for spec in specs:
            try:
                if C.sha(spec['source_archive_path'])!=spec['source_archive_sha256']:raise ValueError('accepted production archive changed')
                wanted=('log_raw_mass','log_raw_fov_mass','probability')
                production[spec['id']]=read_arrays(spec['source_archive_path'],{n:spec['production_array_sha256'][n] for n in wanted},wanted,note)
            except Exception as error:oracle_failures.append(dict(id=spec['id'],kind='old_production',failure_reason=type(error).__name__+': '+str(error)))
        C.write(out/'old_output_delivery_failures.json',oracle_failures)
        for declaration in comparison_plan(specs,oracles):
            description={k:v for k,v in declaration.items() if k not in ('actual_key','reference_key')}
            akey,rkey=declaration['actual_key'],declaration['reference_key'];kind=declaration['kind']
            try:
                actual=production[akey] if kind=='old_production_vs_new' else new[akey]
                ref=old[rkey] if kind=='new_vs_old_reference' else new[rkey]
                policy=dict(lock['policy'],**specmap[declaration['id']]['policy_overrides'])
                result=compare_saved(actual,ref,prepared[declaration['id']],policy,lock['gates'],kind!='old_production_vs_new')
                newkeys=[akey] if kind=='new_vs_old_reference' else ([rkey] if kind=='old_production_vs_new' else [akey,rkey])
                result['all_new_call_checks_passed']=all(checked[k].get('arithmetic_passed',False) and checked[k].get('delivery_integrity_passed',False) for k in newkeys)
                result['passed']=bool(result['passed'] and result['all_new_call_checks_passed'])
            except Exception as error:result=dict(passed=False,failure_reason=type(error).__name__+': '+str(error))
            comparisons.append(dict(**description,**result,synthetic_inputs_read=False,synthetic_arrays_used=True,
                independent_synthetic_reference=reference,reference_integrations_executed_by_auditor=0,**SCOPE))
            C.write(out/'comparisons.json',comparisons)
        inherited=dict(identities=C.read(C.OLD_AUDIT/'identities.json'),near_constant=C.read(C.OLD_AUDIT/'near_constant_limit.json'),
            identities_sha256=C.sha(C.OLD_AUDIT/'identities.json'),near_constant_sha256=C.sha(C.OLD_AUDIT/'near_constant_limit.json'),
            new_identity_calls=0,new_near_constant_calls=0,inherited_only=True)
        C.write(out/'inherited_checks.json',inherited);joint_locks(run,out)
        integrity=len(checks)==46 and all(r['delivery_integrity_passed'] for r in checks) and not oracle_failures
        passed=bool(integrity and seal['reference_calls_completed'] and len(comparisons)==155
            and all(r['arithmetic_passed'] for r in checks) and all(r['passed'] for r in comparisons)
            and len(inherited['identities'])==6 and all(r['passed'] for r in inherited['identities']) and inherited['near_constant']['passed'])
        summary=dict(delivery_integrity_passed=integrity,numerical_preflight_passed=passed,base_case_count=23,reference_call_count=46,
            actual_reference_calls=seal['attempted_calls'],calls_not_attempted=46-seal['attempted_calls'],
            reference_completed_calls=sum(r['passed'] for r in rows),new_order_comparisons=23,new_old_reference_comparisons=86,
            old_production_new_reference_comparisons=46,comparison_count=len(comparisons),comparison_failures=sum(not r['passed'] for r in comparisons),
            reference_check_failures=sum(not r['arithmetic_passed'] for r in checks),inherited_identity_count=6,inherited_near_constant_only=True,
            reference_phase_wall_seconds=seal['phase_wall_seconds'],reference_cell_node_evaluations=seal['total_cell_node_evaluations'],
            phase_wall_passed=seal['phase_wall_passed'],phase_node_budget_passed=seal['phase_node_budget_passed'],
            reference_integrations_executed_by_auditor=0,started_unix=started,finished_unix=time.time(),
            synthetic_inputs_read=decoded['value'],independent_synthetic_reference=reference,**SCOPE)
        C.write(out/'summary.json',summary)
        verification=dict(delivery_integrity_passed=integrity,numerical_preflight_passed=passed,producer_lock_sha256=C.sha(run/'lock.json'),
            audit_lock_sha256=C.sha(out/'lock.json'),producer_completion_sha256=C.sha(run/'completion.json'),
            producer_forward_complete_sha256=C.sha(run/'forward_complete.json'),input_verification_sha256=C.sha(out/'input_verification.json'),
            summary_sha256=C.sha(out/'summary.json'),checks_sha256=C.sha(out/'checks.json'),comparisons_sha256=C.sha(out/'comparisons.json'),
            reference_integrations_executed_by_auditor=0,finished_unix=time.time(),synthetic_inputs_read=decoded['value'],independent_synthetic_reference=reference,**SCOPE)
        C.write(out/'verification.json',verification)
        C.write(out/'completion.json',dict(files=C.inventory(out),verification_sha256=C.sha(out/'verification.json'),
            delivery_integrity_passed=integrity,numerical_preflight_passed=passed,finished_unix=time.time(),
            synthetic_inputs_read=decoded['value'],independent_synthetic_reference=reference,**SCOPE))
        return verification
    except (Exception,KeyboardInterrupt) as error:
        failure(out,error,started,decoded['value'],reference);raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('freeze','input-audit','audit'))
    parser.add_argument('--run',type=Path,default=DEFAULT_RUN);parser.add_argument('--out',type=Path,default=DEFAULT_OUT)
    args=parser.parse_args();{'freeze':freeze,'input-audit':input_audit,'audit':audit}[args.command](args.run.resolve(),args.out.resolve())


if __name__=='__main__':main()
