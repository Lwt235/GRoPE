"""Independent scoring, paired comparisons, budget accounting for coarse_memory_pe1."""
import argparse
import hashlib
import json
import shutil
import numpy as np
import torch
from experiments.gaussian_memory_coarse_data import *
from experiments.gaussian_memory_native_pe_analysis import compare, METRICS


def numpy_scores(pred,cpu):
    pred=np.asarray(pred,dtype=np.float64);target=cpu['target'].numpy().astype(np.float64)
    valid=cpu['valid'].numpy().astype(bool);hidden=cpu['hidden'].numpy().astype(bool)
    assert np.isfinite(pred).all() and (pred>0).all()
    def region(mask):
        mask=valid&mask;r=np.log(np.maximum(pred[mask],1e-6)/target[mask]);err=pred[mask]-target[mask]
        return dict(mse=float(np.mean(r*r)),variance=float(np.var(r)),bias=float(np.mean(r)**2),mean=float(r.mean()),absrel=float(np.mean(abs(err)/target[mask])),rmse=float(np.sqrt(np.mean(err**2))))
    a=region(np.ones_like(valid));h=region(hidden);v=region(~hidden)
    return dict(log_mse=a['mse'],hidden_log_mse=h['mse'],visible_log_mse=v['mse'],centered_log_mse=a['variance'],bias_squared=a['bias'],log_bias=a['mean'],absrel=a['absrel'],hidden_centered_log_mse=h['variance'],hidden_bias_squared=h['bias'],hidden_absrel=h['absrel'],rmse_m=a['rmse'],valid_pixels=int(valid.sum()),hidden_valid_pixels=int((valid&hidden).sum()))


def lock_analysis():
    files=['experiments/gaussian_memory_coarse_analysis.py','experiments/gaussian_memory_coarse_runtime.py','experiments/gaussian_memory_coarse_report.py','experiments/gaussian_memory_native_pe_analysis.py','experiments/gaussian_memory_native_run.py']
    protocol=dict(plan_sha256=sha256(OUT/'plan.json'),source_sha256={s:sha256(ROOT/s) for s in files},primary='256 persistent hidden_centered_log_mse gaussian vs center',practical_interest_percent=1.,selection='final fixed2000, all places; first three examples plus largest primary regression',draws=5000,seed=20261111,development_results_read=False)
    if (OUT/'evaluation_protocol.json').exists():assert read(OUT/'evaluation_protocol.json')==protocol
    else:write_json(OUT/'evaluation_protocol.json',protocol)
    for f in files:
        dest=OUT/'analysis_source_snapshot'/f;dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists():shutil.copy2(ROOT/f,dest)


def check_and_score():
    p=plan();verify(p);rows=[];maxdiff=0.;identities={};targets={}
    for meta in p['catalog']:
        for c in p['conditions']:targets[(meta['id'],c)]=load(input_path(meta['id'],11,c))
    for kind in KINDS:
        for seed in SEEDS:
            folder=OUT/f'{kind}_{seed}';status=read(folder/'status.json');ev=read(OUT/'evaluation'/f'{kind}_{seed}.json')
            assert status['complete'] and status['steps']==2000 and sha256(folder/'step_2000.pt')==ev['checkpoint_sha256']==status['checkpoint_sha256']
            assert sha256(ev['prediction_path'])==ev['prediction_sha256']
            log=[json.loads(x) for x in (folder/'training.jsonl').read_text().splitlines()]
            assert len(log)==2000 and all(np.isfinite(r['loss']) and np.isfinite(r['gradient_norm']) for r in log)
            counts={n:sum(int(key[3])==n for r in log for key in r['samples']) for n in CAPACITIES};assert counts=={256:2000,1024:2000},counts
            samplehash=hashlib.sha256(json.dumps([r['samples'] for r in log]).encode()).hexdigest()
            identities[(kind,seed)]=dict(initial=status['initial_state_sha256'],schedule=status['schedule_sha256'],actual_sample_sha256=samplehash,capacities=counts,checkpoint=ev['checkpoint_sha256'])
            data=load(ev['prediction_path']);assert len(data['keys'])==48==len(ev['rows'])
            assert data['keys']==[list(k) for k in example_keys(p,'development')]
            for key,pred,record in zip(data['keys'],data['depth_metric'],ev['rows']):
                rid,q,c,n=key;assert q==11 and (record['id'],record['condition'],record['capacity'])==(rid,c,n)
                scores=numpy_scores(pred.numpy(),targets[rid,c])
                for metric,value in scores.items():maxdiff=max(maxdiff,abs(record[metric]-value))
                rows.append(dict(id=rid,place=record['place'],condition=c,capacity=n,kind=kind,seed=seed,**scores))
    for seed in SEEDS:
        for field in ('initial','schedule','actual_sample_sha256'):
            assert len({identities[k,seed][field] for k in KINDS})==1,(seed,field)
    # Reused official scores already independently checked in the parent experiment.
    oldrows=read(PARENT_OUT/'independent_rows.json');official=[dict(r,capacity=0) for r in oldrows if r['kind']=='official'];assert len(official)==24
    rows.extend(official)
    ref=read(OUT/'evaluation/references.json');assert sha256(ref['prediction_path'])==ref['prediction_sha256']
    data=load(ref['prediction_path'])
    expected_ref=[[m['id'],c,n,kind] for m in p['catalog'] for c in p['conditions'] for n,kind in [(0,'single_view'),*[(n,'coarse_frozen') for n in CAPACITIES]]]
    assert len(ref['rows'])==len(data['depth_metric'])==72 and data['keys']==expected_ref
    for key,pred,record in zip(data['keys'],data['depth_metric'],ref['rows']):
        rid,c,n,kind=key;s=numpy_scores(pred.numpy(),targets[rid,c]);maxdiff=max(maxdiff,max(abs(s[k]-record[k]) for k in s))
        rows.append(dict(id=rid,place=record['place'],condition=c,capacity=n,kind=kind,seed=None,**s))
    assert maxdiff<1e-10,maxdiff
    write_json(OUT/'independent_rows.json',rows)
    write_json(OUT/'independent_result_checks.json',dict(complete=True,new_adapted=432,new_reference=len(ref['rows']),reused_independently_checked_official=24,rows=len(rows),max_metric_abs=maxdiff,paired={str(s):{k:identities[k,s] for k in KINDS} for s in SEEDS},all_losses_and_gradients_finite=True,plan_sha256=sha256(OUT/'plan.json')))
    return rows


def analyze():
    p=plan();rows=check_and_score();ids=p['evaluation_ids'];places=[m['place'] for m in p['catalog']]
    lookup={(r['id'],r['condition'],r['capacity'],r['kind'],r['seed']):r for r in rows}
    def mat(n,c,kind,metric):
        return np.array([[lookup[rid,c,0 if kind in ('official','single_view') else n,kind,seed if kind in KINDS else None][metric] for rid in ids] for seed in SEEDS])
    pairs=[('gaussian','center'),('gaussian','sphere'),('sphere','center'),('center','coarse_frozen'),('gaussian','coarse_frozen'),('center','official'),('gaussian','official'),('gaussian','single_view')]
    comparisons={};tables={}
    for n in CAPACITIES:
        comparisons[str(n)]={};tables[str(n)]={}
        for c in p['conditions']:
            tables[str(n)][c]={k:{m:float(mat(n,c,k,m).mean()) for m in METRICS} for k in ('official','single_view','coarse_frozen',*KINDS)}
            comparisons[str(n)][c]={f'{a}_vs_{b}':{m:compare(mat(n,c,a,m),mat(n,c,b,m),places,list(SEEDS)) for m in METRICS} for a,b in pairs}
    metric='hidden_centered_log_mse';delta=mat(256,'persistent','gaussian',metric).mean(0)-mat(256,'persistent','center',metric).mean(0)
    worst=sorted(zip(ids,delta),key=lambda x:(-x[1],x[0]))[0][0];selected=list(dict.fromkeys(p['fixed_examples']+[worst]))
    # Capacity interaction is exploratory: first-pass query geometry can also change.
    rng=np.random.default_rng(20261111);si=rng.integers(3,size=(5000,3));pi=rng.integers(len(ids),size=(5000,len(ids)))
    effects={}
    for n in CAPACITIES:
        a=mat(n,'persistent','gaussian',metric);b=mat(n,'persistent','center',metric)
        aa=a[si[:,:,None],pi[:,None,:]].mean((1,2));bb=b[si[:,:,None],pi[:,None,:]].mean((1,2));effects[n]=(bb-aa)/bb*100
    interaction=effects[256]-effects[1024]
    budgets={}
    for n in CAPACITIES:
        rec=[read(OUT/'cache'/(keyname(rid,11,c,n)+'.json')) for rid in ids for c in p['conditions']]
        actual=[r['persistent_bytes'] for r in rec];assert len(set(actual))==1
        # Removing the full source covariance cannot buy one extra 24-layer KV slot.
        possible_saving=(n+40)*3*3*4;extra_slot_bytes=24*2*16*64*2+3*4+8+4+1
        assert possible_saving<extra_slot_bytes
        budgets[str(n)]=dict(actual_persistent_bytes=actual[0],MiB=actual[0]/2**20,source_camera_bytes=rec[0]['camera_bytes'],maximum_center_metadata_saving_bytes=possible_saving,extra_slot_bytes=extra_slot_bytes,extra_center_slot_fits=False,source_build_seconds_median=float(np.median([r['source_build_seconds'] for r in rec])),static_construction_peak_GiB=max(r['peak_GiB'] for r in rec),offline_audit_cache_bytes=sum(sample_path(*k).stat().st_size for k in example_keys(p,'train')+example_keys(p,'development') if k[3]==n))
    statuses=[read(OUT/f'{k}_{s}'/'status.json') for k in KINDS for s in SEEDS]
    result=dict(complete=True,tables=tables,comparisons=comparisons,budgets=budgets,capacity_effect_difference_256_minus_1024_percent_95=np.percentile(interaction,[2.5,97.5]).tolist(),capacity_effect_scope='Paired overall-interface condition; first-pass geometry changes too, not a uniquely isolated covariance cause',worst_id=worst,worst_place=places[ids.index(worst)],fixed_example_ids=selected,bootstrap=p['bootstrap'],training_worker_hours=sum(s['seconds_this_invocation'] for s in statuses)/3600,training_peak_GiB=max(s['peak_GiB'] for s in statuses),new_confirmation=False,all_data_previously_used=True,plan_sha256=sha256(OUT/'plan.json'))
    write_json(OUT/'analysis.json',result);write_json(OUT/'figure_selection.json',dict(fixed=p['fixed_examples'],worst=worst,ids=selected,seed=SEEDS[0],selection='maximum 256 persistent ROI spatial Gaussian minus Center error across paired seeds'))
    print(json.dumps({n:comparisons[n]['persistent']['gaussian_vs_center']['hidden_centered_log_mse'] for n in comparisons}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['lock','analyze']);args=parser.parse_args();lock_analysis() if args.action=='lock' else analyze()
