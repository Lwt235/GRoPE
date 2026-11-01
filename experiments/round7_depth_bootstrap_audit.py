"""Independent NumPy/explicit-loop replay of the sealed depth analysis.

This imports neither the analyzer nor trainer/model code. Run only after the
authorized fresh confirmation and main analysis have completed.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


SEEDS=(11,22,33,44,55)
ROLES=('full','fixed','isotropic','point','point_pck','singleview')
METRICS=('AbsRel','MAE','RMSE','log_RMSE','delta1','boundary_AbsRel','boundary_MAE')


def file_sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def independently_aggregate(rows,scenes):
    result={};population=[]
    for scene in scenes:
        items=[r for r in rows if r['scene']==scene]
        if len(items)!=96:raise ValueError('independent audit: expected 96 pairs per scan')
        population.extend((scene,r['q_index'],r['k_index'],r['count'],r['boundary_count']) for r in items)
        count=sum(r['count'] for r in items);boundary=sum(r['boundary_count'] for r in items)
        if count<=0:raise ValueError('independent audit: empty scan')
        def ratio(field,denominator):return sum(r[field] for r in items)/denominator
        result[scene]=dict(AbsRel=ratio('absrel_sum',count),MAE=ratio('abs_sum',count),
            RMSE=math.sqrt(ratio('sq_sum',count)),log_RMSE=math.sqrt(ratio('log_sq_sum',count)),
            delta1=ratio('delta1_count',count),boundary_AbsRel=ratio('boundary_absrel_sum',boundary) if boundary else None,
            boundary_MAE=ratio('boundary_abs_sum',boundary) if boundary else None)
    if len(set((p[0],p[1],p[2]) for p in population))!=480:raise ValueError('independent audit: duplicate/missing identities')
    return result,sorted(population)


def independent_bootstrap(a,b,seed_draws,scene_draws):
    """Accumulate each of the 25 crossed selected cells explicitly."""
    a,b=np.asarray(a,float),np.asarray(b,float)
    if a.shape!=b.shape or a.ndim!=2:raise ValueError('matrix mismatch')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return dict(available=False)
    aa=np.zeros(len(seed_draws));bb=np.zeros(len(seed_draws))
    divisor=seed_draws.shape[1]*scene_draws.shape[1]
    for i in range(seed_draws.shape[1]):
        for j in range(scene_draws.shape[1]):
            aa+=a[seed_draws[:,i],scene_draws[:,j]]/divisor
            bb+=b[seed_draws[:,i],scene_draws[:,j]]/divisor
    mean_a=float(sum(a.flat)/a.size);mean_b=float(sum(b.flat)/b.size)
    difference=mean_a-mean_b
    per_seed=[float(sum(a[i]-b[i])/a.shape[1]) for i in range(a.shape[0])]
    per_scene=[float(sum(a[:,i]-b[:,i])/a.shape[0]) for i in range(a.shape[1])]
    relative=(aa-bb)/bb if np.all(bb>0) else None
    return dict(available=True,candidate_mean=mean_a,control_mean=mean_b,difference=difference,
        relative_difference=difference/mean_b if mean_b>0 else None,
        ci95_difference=np.percentile(aa-bb,[2.5,97.5]).tolist(),
        ci95_relative_difference=np.percentile(relative,[2.5,97.5]).tolist() if relative is not None else None,
        improved_seed_count=sum(x<0 for x in per_seed),per_seed=per_seed,per_scene=per_scene)


def audit(summary_path,out):
    summary_path=Path(summary_path);report=json.loads(summary_path.read_text())
    if tuple(report['seeds'])!=SEEDS or len(report['scenes'])!=5 or set(report['roles'])!=set(ROLES):
        raise ValueError('wrong analysis matrix')
    if report['bootstrap']['draws']!=20000 or report['bootstrap']['seed']!=20261030:
        raise ValueError('wrong frozen bootstrap')
    lock_path=Path(report['verification']['lock_path'])
    if file_sha(lock_path)!=report['verification']['lock_sha256']:raise ValueError('confirmation lock changed')
    lock=json.loads(lock_path.read_text())
    if lock['baseline_role']!=report['baseline_role']:raise ValueError('baseline differs from actual lock')
    scenes=tuple(sorted(report['scenes']));arrays={r:{m:np.full((5,5),np.nan) for m in METRICS} for r in ROLES}
    seen=set();reference_population=None;raw=None;hashes={str(summary_path):file_sha(summary_path),str(lock_path):file_sha(lock_path)}
    for record in report['evaluation_files']:
        path=Path(record['path'])
        if file_sha(path)!=record['sha256']:raise ValueError('test JSON changed')
        hashes[str(path)]=record['sha256']
        payload=json.loads(path.read_text());role=record['role']
        aggregate,population=independently_aggregate(payload['rows'],scenes)
        if reference_population is None:reference_population=population
        elif reference_population!=population:raise ValueError('independent audit: populations differ')
        if role=='baseline_raw':
            if raw is not None:raise ValueError('duplicate raw baseline')
            raw=aggregate;continue
        seed=int(record['seed']);key=(role,seed)
        if role not in ROLES or seed not in SEEDS or key in seen:raise ValueError('duplicate/unknown role-seed cell')
        seen.add(key);si=SEEDS.index(seed)
        for pi,scene in enumerate(scenes):
            for metric in METRICS:
                if aggregate[scene][metric] is not None:arrays[role][metric][si,pi]=aggregate[scene][metric]
    if seen!={(r,s) for r in ROLES for s in SEEDS} or raw is None:raise ValueError('incomplete evaluation files')
    rng=np.random.default_rng(20261030)
    seed_draws=rng.integers(0,5,size=(20000,5));scene_draws=rng.integers(0,5,size=(20000,5))
    checks=[];max_error=0.
    def equal(actual,expected,name):
        nonlocal max_error
        if actual is None or expected is None:
            if actual is not None or expected is not None:raise ValueError(name+' null mismatch')
        else:
            error=float(np.max(np.abs(np.asarray(actual,float)-np.asarray(expected,float))))
            max_error=max(max_error,error)
            if not np.allclose(actual,expected,rtol=1e-11,atol=1e-12,equal_nan=False):
                raise ValueError(name+' differs in independent replay')
        checks.append(name)
    for role in ROLES:
        for metric in METRICS:
            values=arrays[role][metric];model=report['models'][role][metric]
            equal(float(np.nanmean(values)) if np.isfinite(values).any() else None,model['mean'],role+'/'+metric+'/mean')
            for si,seed in enumerate(SEEDS):
                equal(float(np.nanmean(values[si])) if np.isfinite(values[si]).any() else None,model['per_seed'][str(seed)],role+'/'+metric+'/seed'+str(seed))
            for pi,scene in enumerate(scenes):
                equal(float(np.nanmean(values[:,pi])) if np.isfinite(values[:,pi]).any() else None,model['per_scene'][scene],role+'/'+metric+'/'+scene)
    comparisons={}
    for role in ROLES:
        if role=='full':continue
        comparisons[role]={}
        for metric in METRICS:
            independent=independent_bootstrap(arrays['full'][metric],arrays[role][metric],seed_draws,scene_draws)
            original=report['comparisons'][role][metric];comparisons[role][metric]=independent
            if independent['available']!=original['available']:raise ValueError('boundary availability changed')
            if not independent['available']:continue
            for field in ('candidate_mean','control_mean','difference','relative_difference','ci95_difference','ci95_relative_difference','improved_seed_count'):
                equal(independent[field],original[field],role+'/'+metric+'/'+field)
            for si,seed in enumerate(SEEDS):equal(independent['per_seed'][si],original['per_seed_difference'][str(seed)],role+'/'+metric+'/seed_effect'+str(seed))
            for pi,scene in enumerate(scenes):
                equal(independent['per_scene'][pi],original['per_scene_difference'][scene],role+'/'+metric+'/scene_effect'+scene)
                remaining=[i for i in range(5) if i!=pi]
                ca=arrays['full'][metric][:,remaining].mean();cb=arrays[role][metric][:,remaining].mean()
                entry=original['leave_one_scan_out'][scene]
                equal(ca-cb,entry['difference'],role+'/'+metric+'/LOO'+scene)
                equal((ca-cb)/cb if cb>0 else None,entry['relative_difference'],role+'/'+metric+'/LOO_relative'+scene)
    baseline=report['baseline_role'];primary=comparisons[baseline]['AbsRel'];single=comparisons['singleview']['AbsRel'];rmse=comparisons[baseline]['RMSE']
    gate=dict(relative_AbsRel_improvement_at_least_2_percent=primary['relative_difference'] is not None and primary['relative_difference']<=-.02,
        AbsRel_difference_CI_upper_below_zero=primary['ci95_difference'][1]<0,
        at_least_4_of_5_seed_macros_improve=primary['improved_seed_count']>=4,
        better_than_singleview_AbsRel_CI_upper_below_zero=single['ci95_difference'][1]<0,
        RMSE_relative_difference_CI_upper_below_2_percent=rmse['ci95_relative_difference'] is not None and rmse['ci95_relative_difference'][1]<.02)
    if gate!=report['primary_gate']['criteria'] or all(gate.values())!=report['primary_gate']['passed']:
        raise ValueError('independently recomputed gate differs')
    for metric in METRICS:
        vals=[raw[s][metric] for s in scenes if raw[s][metric] is not None]
        equal(sum(vals)/len(vals) if vals else None,report['raw_baseline']['summary']['metrics'][metric],'raw/'+metric)
    result=dict(complete=True,numeric_checks=len(checks),maximum_absolute_numeric_difference=max_error,
        paired_draws=20000,random_seed=20261030,seeds=SEEDS,scenes=scenes,primary_gate=gate,
        primary_gate_passed=all(gate.values()),independent_comparisons=comparisons,
        reference='independent standard sums, within-scan square roots, and explicit accumulation of 25 crossed cells per draw; no analyzer/trainer imports',
        input_sha256=hashes,source_sha256=file_sha(__file__),raw_baseline_seed_replication=False)
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    (out/'audit.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps({k:result[k] for k in ('complete','numeric_checks','maximum_absolute_numeric_difference','primary_gate_passed')},indent=2))
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--summary',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();audit(a.summary,a.out)


if __name__=='__main__':main()
