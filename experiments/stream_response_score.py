"""Independent CPU full-vector scoring; all fixed captures, queries and tails retained."""
import os,time,resource,csv,json
import numpy as np
import torch
from experiments import stream_response_data as d
METHODS=['D','X','K','M','native_fused']
def ratio(n,z):return float(n/z) if z!=0 else None
def stats(x):
 x=np.asarray(x,np.float64);finite=np.isfinite(x);v=x[finite];return dict(n=x.size,nonfinite=int((~finite).sum()),min=float(v.min()) if len(v) else None,mean=float(v.mean()) if len(v) else None,median=float(np.median(v)) if len(v) else None,p95=float(np.percentile(v,95)) if len(v) else None,max=float(v.max()) if len(v) else None,finite_summary_only_nonfinite_counted=True)
def boots(Ea,Eb,draws):
 # Positive: a improves on b. Ratio of equal-place mean errors, never average ratios.
 a=np.asarray(Ea);b=np.asarray(Eb);valid=np.isfinite(a)&np.isfinite(b);a=a[valid];b=b[valid]
 if not len(a) or b.mean()==0:return dict(relative_reduction=None,CI95=None,n=len(a),NA=True)
 ix=draws if len(a)==draws.shape[1] else np.random.default_rng(20261114).integers(0,len(a),(5000,len(a)));den=b[ix].mean(1);num=a[ix].mean(1);good=den!=0;values=(den[good]-num[good])/den[good];ci=np.percentile(values,[2.5,97.5]);return dict(relative_reduction=float((b.mean()-a.mean())/b.mean()),CI95=ci.tolist(),n=len(a),NA=False,bootstrap_zero_denominator=int((~good).sum()),positive_places=int((b>a).sum()),practical_10pct_and_positive_CI=bool((b.mean()-a.mean())/b.mean()>=.1 and ci[0]>0),mean_candidate_E=float(a.mean()),mean_control_E=float(b.mean()))
def main():
 begin=time.monotonic();torch.set_num_threads(2);p=d.plan();assert d.read(d.OUT/'pool_complete.json')['complete'];ids=[x['id'] for x in p['catalog']];places={x['id']:x['place'] for x in p['catalog']};totals={r:np.zeros(5,np.float64) for r in ids};strata={};rows=[];queryrows=[];logrows=[];evrows=[];allmax=[];maxdiff=0.;cases=0;zero_queries=0;source_records=[]
 def add(key,vec):strata[key]=strata.get(key,np.zeros(5,np.float64))+vec
 for rid in ids:
  for t in (31,63,95):
   geom=d.load(d.DATA/'geometry'/f'{rid}_{t:02d}.pt');source_records.append(dict(id=rid,place=places[rid],frame=t,eligible_remote=geom['eligible_remote'],valid_remote=geom['valid_remote'],invalid_remote=geom['invalid_remote'],invalid_fraction=ratio(geom['invalid_remote'],geom['eligible_remote']),coarse_groups=len(geom['coarse']),fine_groups=len(geom['fine']),tree_seconds=geom['tree_seconds']))
   for layer in (6,12,18,24):
    case=f'{rid}_{t:02d}_{layer:02d}';v=d.load(d.DATA/'cases'/f'{case}.pt');a=v['projected'].numpy();assert a.shape==(5,1041,1024);energy=np.einsum('mqd,mqd->mq',a,a,dtype=np.float64);check=v['projected'].double().square().sum(-1).numpy();maxdiff=max(maxdiff,float(np.abs(energy-check).max()));assert np.allclose(energy,check,rtol=1e-12,atol=1e-12);assert np.isfinite(energy).all();sums=energy.sum(1);totals[rid]+=sums;cases+=1;zero_queries+=int((energy[0]==0).sum())
    for label,sl in [('all',slice(None)),('patch',slice(5,None)),('special',slice(0,5)),('camera_token',slice(0,1)),('register',slice(1,5))]:
     ss=energy[:,sl].sum(1)
     for group in [('all',label),('layer',str(layer),label),('time',str(t),label),('place',rid,label),('place_layer',rid,str(layer),label),('place_time',rid,str(t),label)]:add(group,ss)
    logs=v['logs'].double().numpy();assert logs.shape==(10,16,1041)
    for j,m in enumerate(('X','K','M')):
     err=logs[4+2*j]-logs[0];masserr=logs[5+2*j]-logs[3]
     with np.errstate(over='ignore',invalid='ignore'):rel=np.expm1(err)
     logrows.append(dict(id=rid,place=places[rid],frame=t,global_layer=layer,method=m,logZ_error=stats(err),relative_Z_error=stats(rel),remote_mass_error=stats(masserr),exact_remote_mass=stats(logs[3])))
     for h,e in enumerate(v['fit']['explained'][m]):evrows.append(dict(id=rid,frame=t,global_layer=layer,head=h,method=m,**e,key_R2=1-e['key_residual_SSE']/e['key_total_SSE'] if e['key_total_SSE'] else None,value_R2=1-e['value_residual_SSE']/e['value_total_SSE'] if e['value_total_SSE'] else None))
    rows.append(dict(id=rid,place=places[rid],frame=t,global_layer=layer,remote_tokens=v['remote_tokens'],exact_tokens=v['exact_tokens'],absolute_SSE=dict(zip(METHODS,sums.tolist())),E={m:ratio(sums[j],sums[0]) for j,m in enumerate(METHODS)},fit_seconds=v['fit']['seconds'],response_seconds=v['timing']['total_seconds'],encoded_read_seconds=v['timing']['encoded_read_seconds'],native_fused_difference_SSE=float(sums[4])))
    for q in range(1041):
     qq=dict(id=rid,place=places[rid],frame=t,global_layer=layer,query=q,type='patch' if q>=5 else 'special',D_SSE=float(energy[0,q]),X_SSE=float(energy[1,q]),K_SSE=float(energy[2,q]),M_SSE=float(energy[3,q]),native_fused_SSE=float(energy[4,q]),E_X=ratio(energy[1,q],energy[0,q]),E_K=ratio(energy[2,q],energy[0,q]),E_M=ratio(energy[3,q],energy[0,q]),exact_remote_mass_mean=float(logs[3,:,q].mean()),X_minus_K_absolute_SSE=float(energy[1,q]-energy[2,q]));queryrows.append(qq)
     if q==0 or len(allmax)<30:allmax.append(qq)
     elif qq['X_minus_K_absolute_SSE']>min(x['X_minus_K_absolute_SSE'] for x in allmax):allmax.append(qq);allmax=sorted(allmax,key=lambda x:x['X_minus_K_absolute_SSE'],reverse=True)[:30]
    del v,a,check,energy
 perplace=[]
 for rid in ids:
  s=totals[rid];perplace.append(dict(id=rid,place=places[rid],absolute_SSE=dict(zip(METHODS,s.tolist())),E={m:ratio(s[j],s[0]) for j,m in enumerate(METHODS)},NA_zero_denominator=s[0]==0))
 E={m:np.array([np.nan if x['E'][m] is None else x['E'][m] for x in perplace]) for m in METHODS};draws=np.random.default_rng(20261114).integers(0,len(ids),(5000,len(ids)));comparisons={f'{a}_vs_{b}':boots(E[a],E[b],draws) for a,b in [('X','K'),('M','X'),('X','D'),('K','D'),('M','D'),('native_fused','D')]};primary=comparisons['X_vs_K'];primary['spatial_investment_signal']=bool(primary['practical_10pct_and_positive_CI'] and np.nanmean(E['X'])<1);loo=[]
 for i,rid in enumerate(ids):
  mask=np.arange(len(ids))!=i;ex=E['X'][mask];ek=E['K'][mask];loo.append(dict(removed_id=rid,removed_place=places[rid],mean_X=float(np.nanmean(ex)),mean_K=float(np.nanmean(ek)),X_K_relative_reduction=ratio(np.nanmean(ek)-np.nanmean(ex),np.nanmean(ek))))
 stratrows=[]
 for key,s in strata.items():stratrows.append(dict(group=list(key),absolute_SSE=dict(zip(METHODS,s.tolist())),E={m:ratio(s[j],s[0]) for j,m in enumerate(METHODS)},zero_denominator=s[0]==0))
 # Per-stratum paired-place intervals preserve place equal weighting (not pooled-token inference).
 stratcomparisons=[]
 for label in ('all','patch','special','camera_token','register'):
  for grouping,values in [('place',[None]),('place_layer',['6','12','18','24']),('place_time',['31','63','95'])]:
   for val in values:
    ss=[strata[(grouping,rid,label) if val is None else (grouping,rid,val,label)] for rid in ids];ee=np.array([[np.nan if s[0]==0 else s[j]/s[0] for j in range(5)] for s in ss]);stratcomparisons.append(dict(group=grouping,value=val,query_type=label,X_vs_K=boots(ee[:,1],ee[:,2],draws),M_vs_X=boots(ee[:,3],ee[:,1],draws),mean_E={m:float(np.nanmean(ee[:,j])) if np.isfinite(ee[:,j]).any() else None for j,m in enumerate(METHODS)}))
 def clean(v):
  if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
  if isinstance(v,(list,tuple)):return [clean(x) for x in v]
  if isinstance(v,(float,np.floating)) and not np.isfinite(v):return None
  if isinstance(v,np.generic):return v.item()
  return v
 for name,table in [('per_case',rows),('per_place',perplace),('strata',stratrows),('paired_strata',stratcomparisons),('logZ_mass',logrows),('explained_variance',evrows),('source_support',source_records),('leave_one_out',loo),('worst_query_counterexamples',sorted(allmax,key=lambda x:x['X_minus_K_absolute_SSE'],reverse=True)[:30])]:d.write(d.OUT/f'{name}.json',clean(table))
 with (d.OUT/'all_query_metrics.csv').open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=queryrows[0]);writer.writeheader();writer.writerows(queryrows)
 score=dict(complete=True,PID=os.getpid(),cases=cases,places=12,query_vectors=cases*1041,head_queries=cases*1041*16,zero_query_deletion_denominators=zero_queries,zero_place_deletion_denominators=sum(x['NA_zero_denominator'] for x in perplace),means={m:float(np.nanmean(E[m])) if np.isfinite(E[m]).any() else None for m in METHODS},comparisons=comparisons,independent_full_vector_NumPy_Torch_SSE_max_abs=maxdiff,bootstrap=dict(draws=5000,seed=20261114,unit='pairedplace',estimator='ratioofequalplacemeanerrors'),all_queries_retained=True,all_development=True,training_model_GPU_calls=0,CPU_seconds=time.monotonic()-begin,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
 d.write(d.OUT/'scores.json',clean(score));print(json.dumps(clean(score)),flush=True)
if __name__=='__main__':main()
