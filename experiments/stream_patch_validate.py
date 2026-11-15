"""Independent raw-pair bootstrap, inherited W identity and causal ID lineage."""
import os,time,resource
import numpy as np
import torch
from experiments import stream_patch_data as d
from experiments.stream_patch_run import hash_order

def main():
 start=time.monotonic();torch.set_num_threads(2);p=d.plan();s=d.read(d.OUT/'scoring/summary.json');frames=d.read(d.OUT/'scoring/frames.json');pairs=d.read(d.OUT/'scoring/pairs.json');assert len(frames)==3456 and len(pairs)==2268;ids=[m['id'] for m in p['catalog']];vv={a:np.array([np.mean([r['relative_rotation_deg'] for r in pairs if r['arm']==a and r['id']==rid and r['revisit']]) if any(r['arm']==a and r['id']==rid and r['revisit'] for r in pairs) else np.nan for rid in ids]) for a in ('W','U','C')};rng=np.random.default_rng(20261114);draw=rng.integers(0,12,(5000,12));checks={}
 for a,b in [('C','U'),('C','W'),('U','W')]:
  v=vv[b]-vv[a];arr=v[draw];den=np.isfinite(arr).sum(1);means=np.nansum(arr,1)[den>0]/den[den>0];mu=float(np.nanmean(v));ci=np.quantile(means,[.025,.975]);z=s['comparisons']['revisit_coverage_ge5pct'][a+'_vs_'+b]['relative_rotation_deg'];error=max(abs(mu-z['mean']),float(np.max(np.abs(ci-z['interval95']))));assert error<1e-10;checks[a+'_vs_'+b]=dict(mean=mu,interval95=ci.tolist(),raw_row_difference=error)
 copied=0
 for oldname,new in [('frames',frames),('pairs',pairs)]:
  old={(r['id'],r['t']):r for r in d.read(d.OLDOUT/'scoring'/f'{oldname}.json') if r['arm']=='W'}
  for r in new:
   if r['arm']=='W':assert r==old[r['id'],r['t']];copied+=1
 count=0;hash_checks=0
 for m in p['catalog']:
  for arm in ('U','C'):
   v=d.load(d.DATA/'predictions'/f'{arm}_{m["id"]}.pt');prev=np.empty(0,dtype=np.int64)
   for t,row in enumerate(v['retained_IDs']):
    kept=row.numpy().astype(np.int64);available=np.sort(np.r_[prev,np.arange(t*1041,(t+1)*1041)]);assert np.isin(kept,available).all();src=available//1041;patch=available%1041>=5;near=np.isin(src,[0]+list(range(max(1,t-2),t+1)));special=(~patch)&np.isin(src,[0]+list(range(max(1,t-6),t+1)));candidate=available[patch&~near];required=available[special|(patch&near)];assert np.isin(required,kept).all();assert len(kept)-len(required)==min(len(candidate),p['M'])
    if arm=='U':expected=np.sort(np.r_[required,candidate[hash_order(candidate)[:p['M']]]]);assert np.array_equal(expected,kept);hash_checks+=1
    prev=kept;count+=1
 raw=max(abs(r['log_mse']-r['centered_log_mse']-r['bias_squared']) for r in frames if r['log_mse'] is not None)
 d.write(d.OUT/'interface_validation.json',dict(complete=True,raw_pair_bootstrap_checks=checks,inherited_W_metric_rows_identical=copied,causal_source_ID_lineage_frames=count,U_hash_policy_frames=hash_checks,all_recent_special_capacity_invariants=True,no_evicted_ID_reappearance=True,raw_logMSE_decomposition_max_difference=raw,C_GPU_FPS_toy_from_preflight=d.read(d.OUT/'preflight.json')['FP64_GPU_FPS_toy_check'],new_model_calls=0,old_W_pixel_metrics_recomputed=0))
 d.write(d.OUT/'CPU_validation_runtime.json',dict(complete=True,PID=os.getpid(),seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,CUDA_initialized=torch.cuda.is_initialized(),new_model_calls=0));print(dict(validated=True,U_hash_frames=hash_checks,lineage_frames=count),flush=True)
if __name__=='__main__':main()
