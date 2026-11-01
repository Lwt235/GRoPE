"""One prospectively frozen replay of R35's failed call, with observation only."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[key]='4'
import sys,time,json,hashlib,math
from pathlib import Path
import numpy as np
ROOT=Path('/home/liwt/GRoPE');sys.path.insert(0,str(ROOT))
from experiments import round34_log_narrow_reference as REF
OUT=ROOT/'results/round35_roundoff_postfailure_diagnostic_20261101'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+'\n')
def load(p):
    with np.load(p,allow_pickle=False) as f:return {k:f[k].copy() for k in f.files}
def main():
    started=time.time();before=read(OUT/'before_execution.json')
    assert not (OUT/'diagnostic.json').exists()
    for path,digest in before['files'].items():assert sha(path)==digest,path
    root=read(ROOT/'results/round35_log_real_root_acceptance_20261101.json')
    assert root['accepted_for_failure_interpretation'] and not root['numerical_preflight_passed']
    run=ROOT/'results/round35_log_real_reference'
    rows=read(run/'forward/records.json');r32,r48=rows[20:22]
    assert r48['call_id']==before['fixed_call_id'] and r48['call_index']==21 and r48['order']==48
    assert r32['passed'] and not r48['passed'] and r48['attempted']
    assert r32['id']==r48['id']
    a32,a48=load(run/r32['archive_path']),load(run/r48['archive_path'])
    bank=load(run/r48['prepared_archive_path']);lock=read(run/'lock.json')
    for k in ('mu','factor','bounds','content_bias'):
        name='input__'+k
        assert np.array_equal(bank[name],a48[name]) and bank[name].tobytes()==a48[name].tobytes()
    policy=dict(lock['policy'],**r48['policy_overrides']);refpolicy=lock['reference_policy']
    assert refpolicy['max_call_cell_node_evaluations']==250000000
    budget=REF.NodeBudget(250000000);captured={};original_quad=REF.quad_vec
    def observe(*args,**kwargs):
        result=original_quad(*args,**kwargs)
        integral,error,info=result
        captured.update(integral=np.asarray(integral).copy(),intervals=info.intervals.copy(),
            interval_integrals=info.integrals.copy(),interval_errors=info.errors.copy())
        captured.update(status=int(info.status),success=bool(info.success),message=info.message,
            neval=int(info.neval),error=float(error),epsrel=kwargs['epsrel'],epsabs=kwargs['epsabs'],norm=kwargs['norm'])
        np.savez_compressed(OUT/'observed_outer_arrays.npz',**{k:v for k,v in captured.items() if isinstance(v,np.ndarray)})
        return result
    REF.quad_vec=observe;attempt_started=time.perf_counter()
    try:
        REF.integrate(**{k:bank['input__'+k] for k in ('mu','factor','bounds','content_bias')},
            policy=policy,reference_policy=refpolicy,order=48,budget=budget,deadline=time.perf_counter()+180)
    except REF.ReferenceFailure as error:
        failed=error
    else:raise AssertionError('unchanged failed call unexpectedly returned')
    finally:REF.quad_vec=original_quad
    elapsed=time.perf_counter()-attempt_started
    np.savez_compressed(OUT/'replayed_failure_arrays.npz',**failed.available_arrays)
    assert failed.reference_work==r48['work'] and budget.used==r48['charged_cell_node_evaluations']
    equality={k:(k in a48 and v.dtype==a48[k].dtype and v.shape==a48[k].shape and v.tobytes()==a48[k].tobytes()) for k,v in failed.available_arrays.items()}
    assert all(equality.values()),equality
    assert 'ReferenceFailure: '+str(failed)==r48['failure_reason']
    assert captured['status']==2 and captured['error']==r48['scaled_outer_error_estimate']
    assert captured['norm']=='max' and captured['epsrel']==1e-12 and captured['epsabs']==0
    intervals=captured['interval_integrals'];global_i=captured['integral'];errors=captured['interval_errors']
    assert np.isfinite(intervals).all() and np.all(intervals>=0) and np.all(global_i>0)
    norm=float(np.max(global_i));sum_max=math.fsum(float(np.max(x)) for x in intervals)
    vector_sum=np.asarray([math.fsum(map(float,intervals[:,j])) for j in range(intervals.shape[1])])
    vector_max=float(np.max(vector_sum));eps=np.finfo(np.float64).eps
    floor_sum=50*eps*sum_max;floor_vector=50*eps*vector_max
    threshold=captured['epsrel']*norm/8
    global_estimate=math.fsum(map(float,errors));history_round=captured['error']-global_estimate
    restored=a48['outer_log_scale']+np.log(a48['outer_scaled_integral'])
    assert np.array_equal(restored[:-1],a48['partial_log_raw_mass'])
    diagnostic=dict(created_unix=time.time(),fixed_call_id=r48['call_id'],only_one_reference_call=True,
        original_reference_source_unchanged=True,original_quad_vec_return_values_unchanged=True,
        original_failure_exactly_reproduced=True,all_saved_partial_array_bytes_reproduced=equality,
        original_failure_reason=r48['failure_reason'],work=failed.reference_work,charged_nodes=budget.used,
        diagnostic_reference_wall_seconds=elapsed,diagnostic_deadline_seconds=180,
        saved_partial_vs_complete32_max_log_mass_difference=float(np.max(np.abs(restored[:-1]-a32['log_raw_mass']))),
        partial_mass_comparison_is_diagnostic_not_acceptance=True,
        outer={k:v for k,v in captured.items() if not isinstance(v,np.ndarray)},
        max_norm_global_scaled_integral=norm,active_intervals=len(intervals),
        sum_leaf_max_norms=sum_max,max_norm_sum_leaf_vectors=vector_max,
        max_abs_reaccumulation_difference=float(np.max(np.abs(vector_sum-global_i))),
        norm_inflation_ratio=sum_max/norm,
        scalar_leaf_roundoff_floor=floor_sum,vector_leaf_roundoff_floor=floor_vector,
        convergence_threshold=threshold,scalar_floor_over_threshold=floor_sum/threshold,
        vector_floor_over_threshold=floor_vector/threshold,
        sum_saved_interval_error_estimates=global_estimate,
        inferred_historical_rounding_estimate=history_round,
        scalar_floor_prevents_success_at_saved_partition=bool(floor_sum>=threshold),
        error_estimate_is_not_true_error_certificate=True,
        no_new_probability_or_reference_oracle=True,labels_read=False,efficacy_tested=False,
        candidate_selected=False,scientific_gate_passed=False,goal_complete=False)
    write(OUT/'diagnostic.json',diagnostic)
    for path,digest in before['files'].items():assert sha(path)==digest,path
    write(OUT/'completion.json',dict(started_unix=started,finished_unix=time.time(),passed=True,
        original_files_unchanged=True,formal_R35_outcome_unchanged=True,
        files={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='execution.log'}))
    print(json.dumps(diagnostic,indent=2,allow_nan=False))
if __name__=='__main__':main()
