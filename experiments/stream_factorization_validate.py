"""Independent raw-row paired contrasts and literal inherited metric identities."""
import os,time,resource
import numpy as np
import torch
from experiments import stream_factorization_data as d
def main():
 start=time.monotonic();p=d.plan();s=d.read(d.OUT/'scoring/summary.json');frames=d.read(d.OUT/'scoring/frames.json');rows=d.read(d.OUT/'scoring/pairs.json');assert len(frames)==5760 and len(rows)==3780
 ids=[m['id'] for m in p['catalog']];values={a:[] for a in ('WW','WF','FW','FF','A')}
 for a in values:
  for rid in ids:
   v=[r['relative_rotation_deg'] for r in rows if r['arm']==a and r['id']==rid and r['revisit']];values[a].append(np.mean(v) if v else np.nan)
 values={a:np.asarray(v) for a,v in values.items()};effects={'WF_vs_WW':values['WW']-values['WF'],'FW_vs_WW':values['WW']-values['FW'],'FF_vs_WW':values['WW']-values['FF'],'A_vs_FF':values['FF']-values['A'],'FF_vs_FW':values['FW']-values['FF'],'interaction':values['FW']-values['FF']-values['WW']+values['WF']};rng=np.random.default_rng(20261114);draw=rng.integers(0,12,size=(5000,12));checks={};maxdiff=0.
 for name,v in effects.items():
  vv=v[draw];num=np.nansum(vv,axis=1);den=np.isfinite(vv).sum(axis=1);ci=np.quantile(num[den>0]/den[den>0],[.025,.975]);mu=float(np.mean(v[np.isfinite(v)]));z=s['interaction']['revisit_coverage_ge5pct']['relative_rotation_deg'] if name=='interaction' else s['comparisons']['revisit_coverage_ge5pct'][name]['relative_rotation_deg'];delta=max(abs(mu-z['mean']),float(np.max(np.abs(ci-z['interval95']))));maxdiff=max(maxdiff,delta);checks[name]=dict(mean=mu,interval95=ci.tolist(),independent_raw_row_max_difference=delta,defined_places=int(np.isfinite(v).sum()))
 decomposition=float(np.nanmax(np.abs(effects['FF_vs_WW']-(effects['WF_vs_WW']+effects['FW_vs_WW']+effects['interaction']))));assert maxdiff<1e-10 and decomposition<1e-10
 oldframes=d.read(d.OLDOUT/'scoring/frames.json');oldpairs=d.read(d.OLDOUT/'scoring/pairs.json');oldindex={(r['id'],r['arm'],r['t']):r for r in oldframes};oldpindex={(r['id'],r['arm'],r['t']):r for r in oldpairs};copied=0
 for source,dest in [(oldindex,frames),(oldpindex,rows)]:
  for r in dest:
   if r['arm'] in ('WW','A'):
    old=source[r['id'],'W' if r['arm']=='WW' else 'A',r['t']];assert {k:v for k,v in r.items() if k!='arm'}=={k:v for k,v in old.items() if k!='arm'};copied+=1
 raw_error=max(abs(r['log_mse']-r['centered_log_mse']-r['bias_squared']) for r in frames if r['log_mse'] is not None and r['centered_log_mse'] is not None)
 c=d.read(d.OUT/'call_accounting.json');assert c['repeated_completed_stream_frames']==0 and c['unique_current_RGB_native_aggregator_calls']==4284
 out=dict(complete=True,raw_pair_bootstrap_checks=checks,linear_identity_joint_equals_read_plus_write_plus_interaction_max_difference=decomposition,bootstrap_max_difference=maxdiff,inherited_WW_A_metric_rows_identical=copied,raw_logMSE_decomposition_max_difference=raw_error,counterexample_rows=len(d.read(d.OUT/'scoring/all_counterexamples.json')),interaction_rows=len(d.read(d.OUT/'scoring/interaction_rows.json')),new_model_calls=0,old_pixel_metrics_recomputed=0,selection_labels_recomputed=0,scientific_thresholds_changed=False)
 d.write(d.OUT/'interface_validation.json',out);d.write(d.OUT/'CPU_validation_runtime.json',dict(complete=True,PID=os.getpid(),seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,CUDA_initialized=torch.cuda.is_initialized()));print(out,flush=True)
if __name__=='__main__':main()
