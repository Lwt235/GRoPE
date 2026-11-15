"""Independent CPU geometry scoring; paired sequence estimates, failures and zeros."""
import os,sys,time,resource,collections
import numpy as np
import torch
from experiments import stream_merge_data as d
from experiments.gaussian_memory_native_analysis import resize_numpy
ARMS=('C','I','G')
COMPARISONS=[('G','I'),('G','C'),('I','C'),('C','W'),('I','W'),('G','W')]
def cache_encode(x):
 if isinstance(x,dict):return {k:cache_encode(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [cache_encode(v) for v in x]
 if isinstance(x,np.ndarray):return cache_encode(x.tolist())
 if isinstance(x,(np.integer,np.bool_)):return x.item()
 if isinstance(x,(float,np.floating)) and not np.isfinite(x):return {'nonfinite_float':str(float(x))}
 return x
def cache_decode(x):
 if isinstance(x,dict):
  if set(x)=={'nonfinite_float'}:return float(x['nonfinite_float'])
  return {k:cache_decode(v) for k,v in x.items()}
 if isinstance(x,list):return [cache_decode(v) for v in x]
 return x
def clean(x):
 if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [clean(v) for v in x]
 if isinstance(x,np.ndarray):return clean(x.tolist())
 if isinstance(x,(np.integer,np.bool_)):return x.item()
 if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
 return x
def mean(x):
 x=[v for v in x if v is not None];return float(np.mean(x)) if x else None
def angle(R):
 if not np.isfinite(R).all():return 180.
 s=np.linalg.norm([R[2,1]-R[1,2],R[0,2]-R[2,0],R[1,0]-R[0,1]])/2;c=(np.trace(R)-1)/2
 return float(np.degrees(np.arctan2(s,c)))
def camera(p):
 p=np.asarray(p,dtype=np.float64);x,y,z,w=p[3:7];norm=x*x+y*y+z*z+w*w
 if not np.isfinite(p).all() or norm<1e-20:return None,None
 s=2/norm;R=np.array([[1-s*(y*y+z*z),s*(x*y-z*w),s*(x*z+y*w)],[s*(x*y+z*w),1-s*(x*x+z*z),s*(y*z-x*w)],[s*(x*z-y*w),s*(y*z+x*w),1-s*(x*x+y*y)]])
 C=np.eye(4);C[:3,:3]=R.T;C[:3,3]=-R.T@p[:3];K=np.array([[259/np.tan(p[8]/2),0,259],[0,196/np.tan(p[7]/2),196],[0,0,1.]])
 transform=np.array([[280/518,0,(280/518-1)/2],[0,210/392,(210/392-1)/2],[0,0,1.]])
 K=transform@K
 if not np.isfinite(K).all() or K[0,0]<=0 or K[1,1]<=0:return None,None
 return C,K
def local_points(depth,K):
 y,x=np.indices((210,280));uv=np.stack((x+.5,y+.5,np.ones_like(x)),-1);return (uv@np.linalg.inv(K).T)*depth[...,None]
def world(points,C):return points@C[:3,:3].T+C[:3,3]
def anchored(points,C0):return (points-C0[:3,3])@C0[:3,:3]
def paired(values):
 if any(v is not None and not np.isfinite(v) for v in values):return dict(mean=None,interval95=None,total_places=len(values),status='Nonfinitefailurekept;nocomplete-caseCI')
 a=np.asarray([np.nan if x is None else x for x in values],float);valid=np.isfinite(a)
 if not valid.any():return dict(mean=None,interval95=None,defined_places=0,total_places=len(a),per_place=values,bootstrap_defined_draws=0)
 rng=np.random.default_rng(20261114);ids=rng.integers(0,len(a),(5000,len(a)));b=a[ids];num=np.nansum(b,1);den=np.isfinite(b).sum(1);sample=num[den>0]/den[den>0]
 return dict(mean=float(a[valid].mean()),interval95=np.quantile(sample,[.025,.975]).tolist(),defined_places=int(valid.sum()),total_places=len(a),per_place=values,improved_places=int((a[valid]>0).sum()),bootstrap_defined_draws=len(sample),leave_one_place_out_means=[mean([v for j,v in enumerate(values) if j!=i]) for i in range(len(a))],bootstrap='5000paired resamples of all12place IDs; undefined places stay NA and are never zero-imputed; CI conditional on defined sampled denominator')
def depth_metrics(pred,target,valid):
 n=int(valid.sum());bad=valid&(~np.isfinite(pred)|(pred<=0))
 if not n:return dict(valid_depth_pixels=0,failed_depth_pixels=0,absrel=None,log_mse=None,centered_log_mse=None,bias_squared=None,RMSE_m=None)
 if bad.any():return dict(valid_depth_pixels=n,failed_depth_pixels=int(bad.sum()),absrel=float('inf'),log_mse=float('inf'),centered_log_mse=float('inf'),bias_squared=float('inf'),RMSE_m=float('inf'))
 p=pred[valid];g=target[valid];r=np.log(p/g);e=p-g
 return dict(valid_depth_pixels=n,failed_depth_pixels=0,absrel=float(np.mean(np.abs(e)/g)),log_mse=float(np.mean(r*r)),centered_log_mse=float(np.var(r)),bias_squared=float(np.mean(r)**2),RMSE_m=float(np.sqrt(np.mean(e*e))))
def main():
 start=time.monotonic();torch.set_num_threads(2);available='--available' in sys.argv;print(dict(CPU_scoring_host_PID=os.getpid(),CUDA_initialized=torch.cuda.is_initialized(),available_only=available),flush=True);assert d.read(d.OUT/'profiling_admission.json')['admitted_before_scoring'];plan=d.plan();assert all((d.OLDOUT/'label_pairs'/f"{m['id']}.json").exists() for m in plan['catalog'])
 if not available:assert d.read(d.OUT/'costs.json')['actual_all_waited']
 sys.path.insert(0,str(d.ROOT/'third_party/streamvggt/src'));from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri
 from scipy.spatial.transform import Rotation
 old_summary=d.read(d.OLDOUT/'scoring/summary.json');frame_rows=[r for r in d.read(d.OLDOUT/'scoring/frames.json') if r['arm']=='W'];pair_rows=[r for r in d.read(d.OLDOUT/'scoring/pairs.json') if r['arm']=='W'];selection=d.read(d.OLDOUT/'scoring/selection.json');max_depth_recompute=0.;max_camera_decode=0.;max_angle_check=0.;max_resize=0.;independent_depth_records=0
 for meta in plan['catalog']:
  rid=meta['id'];cache=d.OUT/'scoring_cache'/f'{rid}.json';receipts=[d.OUT/'completed'/f'{a}_{rid}.json' for a in ARMS]
  if not all(x.exists() for x in receipts):
   assert available,'Final scoring requires all fixed policy outputs'
   continue
  identities={x.stem:d.read(x)['sha256'] for x in receipts}
  if cache.exists():
   cached=cache_decode(d.read(cache));assert cached['prediction_identities']==identities and cached['configuration_sha256']==d.sha(d.OUT/'configuration.json')
   if cached.get('cache_version',1)==1:assert not any(r['failure'] for r in cached['frames']+cached['pairs']),'Legacy finite-only cache cannot hide an actual model failure'
   frame_rows.extend(cached['frames']);pair_rows.extend(cached['pairs']);z=cached['independent'];max_depth_recompute=max(max_depth_recompute,z['depth']);max_camera_decode=max(max_camera_decode,z['camera']);max_angle_check=max(max_angle_check,z['angle']);max_resize=max(max_resize,z['resize']);independent_depth_records+=z['depth_records'];continue
  frame_begin=len(frame_rows);pair_begin=len(pair_rows);count_begin=independent_depth_records;pred={arm:d.load(d.DATA/'predictions'/f'{arm}_{rid}.pt') for arm in ARMS};lab=d.load(d.OLDDATA/'labels'/f'{rid}.pt');GT=np.asarray(meta['c2w_metric'],float);Kgt=np.asarray(meta['K_metric'],float);target=lab['depth'].numpy().astype(float);valid=lab['valid'].numpy();metric={arm:np.stack([resize_numpy(v.numpy().astype(float)) for v in pred[arm]['depth']]) for arm in ARMS};cams={};intr={};points={};GTpoints=[]
  for t in range(96):GTpoints.append(anchored(world(local_points(target[t],Kgt),GT[t]),GT[0]))
  for arm in ARMS:
   cs=[];ks=[];ps=[];poses=pred[arm]['poses'].numpy();official,_=pose_encoding_to_extri_intri(torch.from_numpy(poses).double()[None],(392,518));official=official[0].numpy()
   for t in range(96):
    C,K=camera(poses[t]);cs.append(C);ks.append(K)
    if C is not None:
     ext=np.concatenate((C[:3,:3].T,(-C[:3,:3].T@C[:3,3])[:,None]),1);max_camera_decode=max(max_camera_decode,float(np.max(np.abs(ext-official[t]))))
   cams[arm]=cs;intr[arm]=ks
   for t in range(96):
    m=depth_metrics(metric[arm][t],target[t],valid[t]);C=cs[t];C0=cs[0];failure=C is None or C0 is None or m['failed_depth_pixels']>0
    if C is not None and C0 is not None:
     xyz=anchored(world(local_points(metric[arm][t],ks[t]),C),C0);R=C0[:3,:3].T@C[:3,:3];Rgt=GT[0,:3,:3].T@GT[t,:3,:3];rot=angle(R.T@Rgt);center=(C[:3,3]-C0[:3,3])@C0[:3,:3];gcenter=(GT[t,:3,3]-GT[0,:3,3])@GT[0,:3,:3];ce=float(np.linalg.norm(center-gcenter));error=np.linalg.norm(xyz-GTpoints[t],axis=-1);ge=float(error[valid[t]].mean()) if valid[t].any() else None
    else:xyz=np.full((210,280,3),np.nan);rot=180.;ce=float('inf');ge=float('inf') if valid[t].any() else None
    ps.append(xyz);frame_rows.append(dict(id=rid,arm=arm,t=t,timestamp_us=meta['frames'][t]['timestamp_us'],original_frame_index=meta['frames'][t]['index'],trajectory_rotation_anchor_deg=rot,trajectory_center_error_raw_m=ce,absolute_worldpoint_GT_error_raw_m=ge,failure=failure,**m))
    if m['valid_depth_pixels'] and not m['failed_depth_pixels']:
     vv=torch.from_numpy(metric[arm][t][valid[t]]);gg=torch.from_numpy(target[t][valid[t]]);rr=(vv/gg).log();checks=dict(absrel=((vv-gg).abs()/gg).mean(),log_mse=rr.square().mean(),centered_log_mse=rr.var(unbiased=False),bias_squared=rr.mean().square(),RMSE_m=(vv-gg).square().mean().sqrt());max_depth_recompute=max(max_depth_recompute,max(abs(float(v)-m[k]) for k,v in checks.items()));independent_depth_records+=1
   points[arm]=ps
  # Full-resolution archived native outputs independently resized; compare all
  # unique F/W frames with Torch FP64 bilinear, not another model call.
  for arm in ARMS:
   rr=torch.nn.functional.interpolate(pred[arm]['depth'][:,None].double(),(210,280),mode='bilinear',align_corners=False)[:,0].numpy();max_resize=max(max_resize,float(np.max(np.abs(rr-metric[arm]))))
  for row,mapping in zip(lab['pairs'],lab['maps']):
   t,s=row['t'],row['s'];selected=dict(**row);mask=mapping['consistent'].numpy();source_idx=mapping['source_index'].numpy()[mask];target_idx=np.flatnonzero(mask);assert (source_idx>=0).all();GTrel=GT[t,:3,:3].T@GT[s,:3,:3];GTtrans=(GT[s,:3,3]-GT[t,:3,3])@GT[t,:3,:3];GTnorm=float(np.linalg.norm(GTtrans));GTdelta=GTpoints[s].reshape(-1,3)[source_idx]-GTpoints[t].reshape(-1,3)[target_idx];GTsampling=float(np.linalg.norm(GTdelta,axis=1).mean()) if len(source_idx) else None
   for arm in ARMS:
    a,b=cams[arm][s],cams[arm][t];failure=a is None or b is None
    if not failure:
     rel=b[:3,:3].T@a[:3,:3];rot=angle(rel.T@GTrel);max_angle_check=max(max_angle_check,abs(rot-float(np.degrees(np.linalg.norm(Rotation.from_matrix(rel.T@GTrel).as_rotvec())))));trans=(a[:3,3]-b[:3,3])@b[:3,:3];norm=float(np.linalg.norm(trans));td=None if GTnorm<=plan['near_zero_translation_m'] else (180. if norm<=1e-20 else float(np.degrees(np.arccos(np.clip(np.dot(trans,GTtrans)/(norm*GTnorm),-1,1)))));te=float(np.linalg.norm(trans-GTtrans))
    else:rot=180.;td=None if GTnorm<=plan['near_zero_translation_m'] else 180.;te=float('inf')
    if len(source_idx) and not failure:
     p=points[arm][s].reshape(-1,3)[source_idx];q=points[arm][t].reshape(-1,3)[target_idx];consistency=float(np.linalg.norm(p-q,axis=1).mean());absolute=float((np.linalg.norm(p-GTpoints[s].reshape(-1,3)[source_idx],axis=1).mean()+np.linalg.norm(q-GTpoints[t].reshape(-1,3)[target_idx],axis=1).mean())/2)
    else:consistency=float('inf') if len(source_idx) else None;absolute=float('inf') if len(source_idx) else None
    pair_rows.append(dict(**row,arm=arm,relative_rotation_deg=rot,relative_translation_direction_deg=td,relative_translation_vector_error_raw_m=te,GT_translation_norm_m=GTnorm,GT_translation_near_zero=GTnorm<=plan['near_zero_translation_m'],cross_time_point_consistency_raw_m=consistency,paired_absolute_GT_geometry_error_raw_m=absolute,GT_sampling_point_discrepancy_m=GTsampling,failure=failure))
  d.write(cache,cache_encode(dict(cache_version=2,id=rid,prediction_identities=identities,configuration_sha256=d.sha(d.OUT/'configuration.json'),frames=frame_rows[frame_begin:],pairs=pair_rows[pair_begin:],independent=dict(depth=max_depth_recompute,camera=max_camera_decode,angle=max_angle_check,resize=max_resize,depth_records=independent_depth_records-count_begin))))
  print(dict(scored=rid,revisit_targets=sum(r['revisit'] for r in lab['pairs'])),flush=True)
 if available:
  d.write(d.OUT/'CPU_scoring_available_runtime.json',dict(complete=True,all12_complete=len(list((d.OUT/'scoring_cache').glob('*.json')))==12,scored_place_IDs=[x.stem for x in (d.OUT/'scoring_cache').glob('*.json')],PID=os.getpid(),wall_seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,CUDA_initialized=torch.cuda.is_initialized(),new_model_calls=0,scope='Once-only new C/I/G place metrics cached while GPU queue runs; final all12 equal-place bootstrap/NA unchanged and deferred. Old W rows inherited, no old pixel scoring.'));return
 frame_metrics=['trajectory_rotation_anchor_deg','trajectory_center_error_raw_m','absolute_worldpoint_GT_error_raw_m','absrel','log_mse','centered_log_mse','bias_squared','RMSE_m','failure'];pair_metrics=['relative_rotation_deg','relative_translation_direction_deg','relative_translation_vector_error_raw_m','cross_time_point_consistency_raw_m','paired_absolute_GT_geometry_error_raw_m','failure'];byplace=[]
 for meta in plan['catalog']:
  rid=meta['id']
  for arm in ('W','C','I','G'):
   for group in ['all96frames','all63far_targets','revisit_coverage_ge5pct','nonrevisit_lt5pct']:
    rr=[r for r in (frame_rows if group=='all96frames' else pair_rows) if r['id']==rid and r['arm']==arm and (group not in ['revisit_coverage_ge5pct','nonrevisit_lt5pct'] or r['revisit']==(group=='revisit_coverage_ge5pct'))];keys=frame_metrics if group=='all96frames' else pair_metrics;byplace.append(dict(id=rid,arm=arm,group=group,records=len(rr),**{k:mean([r[k] for r in rr]) for k in keys}))
 primary=[];comparisons={}
 for group in ['all96frames','all63far_targets','revisit_coverage_ge5pct','nonrevisit_lt5pct']:
  keys=frame_metrics if group=='all96frames' else pair_metrics;comparisons[group]={}
  for candidate,baseline in COMPARISONS:
   estimates={}
   for metric in keys:
    delta=[]
    for meta in plan['catalog']:
     a=next(r for r in byplace if r['id']==meta['id'] and r['arm']==candidate and r['group']==group)[metric];b=next(r for r in byplace if r['id']==meta['id'] and r['arm']==baseline and r['group']==group)[metric];delta.append(b-a if a is not None and b is not None and np.isfinite(a) and np.isfinite(b) else None)
    nonfinite_failure=sum(any(v is not None and not np.isfinite(v) for v in [next(r for r in byplace if r['id']==meta['id'] and r['arm']==candidate and r['group']==group)[metric],next(r for r in byplace if r['id']==meta['id'] and r['arm']==baseline and r['group']==group)[metric]]) for meta in plan['catalog'])
    estimates[metric]=dict(mean=None,interval95=None,status='Nonfinite model-failure error retained; no finite-subset CI',failed_places=nonfinite_failure,total_places=12) if nonfinite_failure else paired(delta)
   comparisons[group][candidate+'_vs_'+baseline]=estimates
 strata=[]
 for scope in ['selected_pair_exclusive','all_far_union_exclusive']:
  field='exclusive_stratum' if scope=='selected_pair_exclusive' else 'all_far_union_exclusive_stratum'
  for label in ['0','(0,5%]','(5,20%]','>20%','undefined']:
   for revisit in [False,True]:
    for arm in ('W','C','I','G'):
     rr=[r for r in pair_rows if r[field]==label and r['arm']==arm and (r['revisit'] or not revisit)];ids=sorted({r['id'] for r in rr});strata.append(dict(scope=scope,stratum=label,revisit_only=revisit,arm=arm,records=len(rr),places=len(ids),**{k:mean([mean([r[k] for r in rr if r['id']==rid]) for rid in ids]) for k in pair_metrics}))
 strata_paired=[]
 for scope in ['selected_pair_exclusive','all_far_union_exclusive']:
  field='exclusive_stratum' if scope=='selected_pair_exclusive' else 'all_far_union_exclusive_stratum'
  for label in ['0','(0,5%]','(5,20%]','>20%','undefined']:
   for revisit in [False,True]:
    for candidate,baseline in COMPARISONS:
     effects={}
     for metric in pair_metrics:
      delta=[]
      for meta in plan['catalog']:
       vv=[]
       for arm in (candidate,baseline):
        rr=[r[metric] for r in pair_rows if r['id']==meta['id'] and r['arm']==arm and r[field]==label and (r['revisit'] or not revisit)];vv.append(mean(rr))
       delta.append(vv[1]-vv[0] if all(v is not None for v in vv) else None)
      effects[metric]=paired(delta)
     strata_paired.append(dict(scope=scope,stratum=label,revisit_only=revisit,candidate=candidate,baseline=baseline,effects=effects))
 counter=[];index={(r['id'],r['arm'],r['t']):r for r in pair_rows}
 for a,b in COMPARISONS:
  for r in pair_rows:
   if r['arm']!=a:continue
   base=index[r['id'],b,r['t']];counter.append(dict(candidate=a,baseline=b,id=r['id'],t=r['t'],s=r['s'],revisit=r['revisit'],coverage=r['coverage'],exclusive_fraction=r['selected_pair_exclusive_fraction'],elapsed_seconds=r['elapsed_seconds'],rotation_improvement_deg=base['relative_rotation_deg']-r['relative_rotation_deg']))
 counter.sort(key=lambda r:r['rotation_improvement_deg']);coverage=dict(total_places=12,frames=1152,far_targets=756,revisit_targets=sum(r['revisit'] for r in selection),places_with_revisits=sum(any(r['id']==m['id'] and r['revisit'] for r in selection) for m in plan['catalog']),zero_selected_exclusive=sum(r['selected_pair_exclusive_fraction']==0 for r in selection),zero_selected_covisibility=sum(r['consistent_pixels']==0 for r in selection),per_place=[dict(id=m['id'],revisit_targets=sum(r['revisit'] for r in selection if r['id']==m['id']),zero_exclusive=sum(r['selected_pair_exclusive_fraction']==0 for r in selection if r['id']==m['id']),mean_coverage=mean([r['coverage'] for r in selection if r['id']==m['id']])) for m in plan['catalog']])
 summary=dict(complete=True,background_F_A_inherited=dict(source=str(d.OLDOUT/'scoring/summary.json'),identity='InheritedfixedWscoreidentity;notrehashedorrescored',comparisons=old_summary['comparisons']),primary=comparisons['revisit_coverage_ge5pct']['G_vs_I']['relative_rotation_deg'],comparisons=comparisons,coverage=coverage,independent=dict(unique_depth_records=independent_depth_records,NumPy_Torch_depth_metric_max_difference=max_depth_recompute,NumPy_Torch_FP64_resize_max_difference=max_resize,NumPy_official_Torch_pose_decode_max_difference=max_camera_decode,NumPy_SciPy_relative_rotation_max_difference_degrees=max_angle_check,old_metrics_recomputed=0,inherited_W_frame_metrics=1152,inherited_W_pair_metrics=756,new_model_calls=0),metric_conventions=dict(primary='I_shape=E_I−E_G relative rotation degrees, sequence means then equal-defined-place mean; all12denominators retained',failure='Invalid predicted camera retains maximum180degrotation and explicit failure; zeroGTcoverage undefined, not zero error',translation='GT norm<=1e-6m marks direction NA only; collapsed nonzero-GT prediction retains180degfailure',geometry='Raw native depth/translation units with first-frame rigid gauge alignment only; no per-method fitted scale. Cross-time consistency accompanied by absolute GT point error and GT sampling discrepancy.',strata='Selected best-pair exclusive plus separately reported all-far union exclusive; both subtract frame0/recent7 per-view consistent-support union'),not_fresh=True,scientific_disposition_reserved_for_Astra=True)
 flags={}
 for comparison in [a+'_vs_'+b for a,b in COMPARISONS]:
  rot=comparisons['revisit_coverage_ge5pct'][comparison]['relative_rotation_deg'];dep=comparisons['all96frames'][comparison]['absrel']
  flags[comparison]=dict(practical_rotation=rot['mean'] is not None and rot['mean']>=1 and rot['interval95'][0]>0,obvious_AbsRel_cost=dep['mean'] is not None and -dep['mean']>=.01 and dep['interval95'][1]<0,rotation=rot,AbsRel_improvement=dep,scope='Predefined development investment flags, not automatic overall success/next experiment; OnlyG/Iprimary;G/C,I/C,andallC/I/GvsWprespecifiedsecondary')
 summary['predefined_flags']=flags
 for name,value in [('summary',summary),('frames',frame_rows),('pairs',pair_rows),('selection',selection),('by_place',byplace),('support_strata',strata),('support_strata_paired',strata_paired),('all_counterexamples',counter)]:d.write(d.OUT/'scoring'/f'{name}.json',clean(value))
 d.write(d.OUT/'CPU_scoring_runtime.json',dict(complete=True,PID=os.getpid(),PID_namespace='Hostexecutionviaapprovedtool',wall_seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,CUDA_initialized=torch.cuda.is_initialized(),new_model_calls=0));print(clean(dict(primary=summary['primary'],coverage=coverage,independent=summary['independent'])),flush=True)
if __name__=='__main__':main()
