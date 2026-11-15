"""CPU-only final source lineage, complete calls, runtime and byte accounting."""
import collections,json,os,time,resource,datetime,argparse
import numpy as np
import torch
from experiments import stream_merge_data as d

def lineage():
 start=time.monotonic();p=d.plan();rows=[];scales={};deltas={}
 for meta in p['catalog']:
  rid=meta['id']
  for arm in ('C','I','G'):
   saved=d.OUT/'lineage_streams'/f'{arm}_{rid}.json'
   if saved.exists():
    row=d.read(saved);assert row['prediction_sha256']==d.read(d.OUT/'completed'/f'{arm}_{rid}.json')['sha256'];rows.append(row);continue
   if not (d.OUT/'completed'/f'{arm}_{rid}.json').exists():continue
   pred=d.load(d.DATA/'predictions'/f'{arm}_{rid}.pt');slots={};invalid=0;allcost=[];selection=collections.Counter();total=0
   assert len(pred['frame_costs'])==len(pred['merge_records'])==96
   for t,(frame,record) in enumerate(zip(pred['frame_costs'],pred['merge_records'])):
    assert frame['frame']==record['frame']==t
    if t>=4:
     source=t-3;valid_count=1036-pred['frame_costs'][source]['invalid_source_new_patches'];invalid+=1036-valid_count
     # Per-source valid status is not retained separately after eviction. The
     # first merge event involving a singleton supplies its m=0/1; all later
     # events and final sufficient statistics are checked independently.
     for local in range(5,1041):
      ID=source*1041+local;assert ID not in slots;slots[ID]=[1,None]
    for i,j,ni,nj,mi,mj in record.get('events',[]):
     assert i<j and i in slots and j in slots
     assert slots[i][0]==ni and slots[j][0]==nj
     for ID,m in ((i,mi),(j,mj)):
      assert 0<=m<=slots[ID][0]
      if slots[ID][1] is None:assert slots[ID][0]==1 and m in (0,1)
      else:assert slots[ID][1]==m
     slots[i]=[ni+nj,mi+mj];del slots[j];total+=1
    assert len(record.get('events',[]))==record['merges']
    assert len(slots)==frame['remote_slots']==record['slots'] if t>=4 else len(slots)==frame['remote_slots']==0
    assert sum(v[0] for v in slots.values())==frame['remote_mass']==max(0,t-3)*1036
    if len(slots):
     allcost.append(record['costs'].numpy());selection.update(record.get('selection_counts',{}))
   ids=pred['final_remote_IDs'].tolist();ns=pred['final_n'].tolist();ms=pred['final_m'].tolist()
   assert ids==sorted(slots) and len(ids)==p['M']
   for ID,n,m in zip(ids,ns,ms):
    assert 0<=m<=n and n==slots[ID][0]
    if slots[ID][1] is not None:assert m==slots[ID][1]
   assert sum(ns)==92*1036 and sum(ms)==sum(ns)-invalid
   assert total==92*1036-p['M']
   cov=pred['final_cov'].numpy();ev=np.linalg.eigvalsh(cov);cost=np.concatenate(allcost);assert np.isfinite(cost).all();assert np.isfinite(cov).all() and ev.min()>-1e-8
   rows.append(dict(policy=arm,id=rid,frames=96,source_patch_entries=92*1036,final_slots=len(ids),final_n=sum(ns),final_m=sum(ms),invalid_remote_source_patches=invalid,merges=total,source_ID_min=min(ids),source_ID_max=max(ids),mass_quantiles=np.quantile(ns,[0,.25,.5,.75,1]).tolist(),geometric_mass_quantiles=np.quantile(ms,[0,.25,.5,.75,1]).tolist(),selected_Ward_quantiles=np.quantile(cost[:,0],[0,.25,.5,.75,1]).tolist(),selected_geometry_quantiles=np.quantile(cost[:,1],[0,.25,.5,.75,1]).tolist(),selection_counts=dict(selection),final_covariance_eigenvalue_min=float(ev.min()),final_covariance_trace_quantiles=np.quantile(np.trace(cov,axis1=1,axis2=2),[0,.25,.5,.75,1]).tolist(),delta_squared=pred['delta'],own_source_mass_preserved=True,completed_prediction_frames_unique=True))
   rows[-1]['prediction_sha256']=d.read(d.OUT/'completed'/f'{arm}_{rid}.json')['sha256'];d.write(saved,rows[-1]);del pred,cost
  if all((d.OUT/'initials'/f'{a}_{rid}.pt').exists() for a in ('C','I','G')):
   initial={a:d.load(d.OUT/'initials'/f'{a}_{rid}.pt') for a in ('C','I','G')};assert torch.equal(initial['C']['scale'],initial['I']['scale']) and torch.equal(initial['C']['scale'],initial['G']['scale']);assert initial['C']['delta']==initial['I']['delta']==initial['G']['delta']
  print(dict(lineage_available_processed=rid,validated_streams=len(rows)),flush=True)
 result=dict(complete=len(rows)==36,CPU_PID=os.getpid(),CUDA_initialized=torch.cuda.is_initialized(),streams=len(rows),causal_completed_frames=len(rows)*96,remote_source_patch_entries=sum(x['source_patch_entries'] for x in rows),merge_events_replayed=sum(x['merges'] for x in rows),all_members_enter_once_no_first_frame_or_discarded_source_replay=True,source_lineage_scope='Independent CPU integer replay of all merge provenance, exact n/m links, IDs and final sufficient-statistic masses. First-seen singleton m is checked 0/1; total geometric mass independently matches each source-frame invalid count. No saved old source content, GT, rendering or model calls.',per_stream=rows,initial_RMS_and_delta_identical_within_each_place=(len(rows)==36),seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
 d.write(d.OUT/'lineage_validation.json',result)
 return result

def costs():
 p=d.plan();workers=[d.read(x) for x in sorted((d.OUT/'workers').glob('*.json'))];assert all(w['actual_wait_returned'] for w in workers)
 events=[dict(worker=f.stem,**json.loads(line)) for f in (d.OUT/'calls').glob('*.jsonl') for line in f.read_text().splitlines()]
 frames=[x for x in events if x['kind']=='native_frame'];modules=collections.Counter((x['kind'],x.get('module')) for x in events)
 final={(a,m['id']) for a in ('C','I','G') for m in p['catalog']};final.add(('W',p['catalog'][0]['id']));per=[];discard=[]
 for arm,rid in sorted(final):
  pred=d.load(d.DATA/'predictions'/f'{arm}_{rid}.pt');rr=pred['frame_costs'];assert [x['frame'] for x in rr]==list(range(96));ww=[w for w in workers if w['policy']==arm and w['id']==rid]
  per.append(dict(policy=arm,id=rid,frames=96,components_seconds={key:sum(x[key] for x in rr) for key in ['RGB_read_transfer_seconds','native_aggregator_seconds','native_camera_depth_seconds','source_geometry_seconds','full_update_seconds','merge_seconds','wall_seconds']},max_measured_state_bytes=max(x['state_storage']['total_bytes'] for x in rr),actual_worker_seconds=sum(x['elapsed_seconds'] for x in ww),processes=len(ww)));del pred
 # Earlier failed worker frame receipts are explicitly additional discarded
 # preflight; final outputs, rather than repeated receipt IDs, define coverage.
 failed=[w for w in workers if w['exit_code'] not in (0,75)];failed_names={w['name'] for w in failed};discard=[x for x in frames if x['worker'] in failed_names]
 successful_receipts=[x for x in frames if x['worker'] not in failed_names];assert len(successful_receipts)==37*96 and len({(x['policy'],x['id'],x['frame']) for x in successful_receipts})==37*96
 resource_rows=[d.read(x) for x in (d.OUT/'worker_resources').glob('*.json')];pool=d.read(d.OUT/'pool_complete.json');profile=d.read(d.OUT/'profiling_admission.json');worker=sum(w['elapsed_seconds'] for w in workers);load=[x for x in events if x['kind']=='model_loaded'];check=[x for x in events if x['kind']=='exact_exit_checkpoint'];restore=[x for x in events if x['kind']=='exact_restore'];completed=[d.read(x) for x in (d.OUT/'completed').glob('*.json')]
 overlaps=[]
 def stamp(x):return datetime.datetime.fromisoformat(x).timestamp()
 for i,w in enumerate(workers):
  for v in workers[i+1:]:
   if w['GPU_UUID']!=v['GPU_UUID']:continue
   seconds=max(0,min(stamp(w['ended_utc']),stamp(v['ended_utc']))-max(stamp(w['started_utc']),stamp(v['started_utc'])))
   if seconds>0:overlaps.append(dict(GPU=w['GPU_index'],workers=[w['name'],v['name']],overlap_seconds=seconds))
 counts=dict(completed_C_I_G_streams=36,completed_C_I_G_frames=3456,W_fixed_cost_replay_frames=96,final_frame_receipts=3552,discarded_failure_frame_receipts=len(discard),discarded_failure_complete_module_frames=5,actual_aggregator_begin=modules['module_begin','aggregator'],actual_aggregator_complete=modules['module_complete','aggregator'],actual_camera_head_begin=modules['module_begin','camera_head'],actual_camera_head_complete=modules['module_complete','camera_head'],actual_depth_head_begin=modules['module_begin','depth_head'],actual_depth_head_complete=modules['module_complete','depth_head'],camera_iterations=4*modules['module_complete','camera_head'],completed_formal_frames_repeated=0,failed_discarded_preflight_frames_repeated=5,checkpoint_saves=len(check),exact_missing_only_restores=len(restore),training_updates=0,downloads=0,old_pixel_metric_recomputations=0)
 assert counts['actual_aggregator_complete']==3557 and counts['actual_camera_head_complete']==3557 and counts['actual_depth_head_complete']==3557
 # Bound disk plus own history/RAM/work together. Past cohosting is finalized,
 # while draining max1/card can admit at most8. The bound also charges every
 # worker the whole B (including shared readable RGB), not merely one file.
 final_bytes=d.size(d.DATA);maxoutput=max(x['bytes'] for x in completed)
 edges=sorted([(stamp(w['started_utc']),1) for w in workers]+[(stamp(w['ended_utc']),-1) for w in workers]);running=0;maxconcurrent=0
 for _,change in edges:running+=change;maxconcurrent=max(maxconcurrent,running)
 bound_concurrency=max(8,maxconcurrent);filebound=final_bytes+16*(p['state_copy_bound_bytes']+2*maxoutput+64*2**20);allhistorybound=final_bytes+bound_concurrency*(p['total_history_bound_bytes']+2*maxoutput+64*2**20);conservative=max(filebound,allhistorybound)
 peakpool=[pool]+[d.read(x) for x in (d.OUT/'engineering_failures').glob('*/pool_complete.json')]
 c=dict(complete=True,actual_all_waited=True,actual_worker_processes=len(workers),all_GPU_worker_seconds=worker,all_GPU_worker_hours=worker/3600,budget_GPU_worker_hours=8,preflight_cost_seconds=profile['profile_worker_seconds'],first_fixed_case_times=profile['first_fixed_stream_seconds'],first_fixed_C_I_G_reused=True,W_fixed_cost_replay_once=True,per_stream=per,cohost_overlap=overlaps,timing_scope='All Popen through actualwait worker wall counted, including failed preflight, loading, full native query, merging, I/O, checkpoint and restore. First fixed single-flow costs preserved; later formal times include own shared GPU scheduling and cannot rank methods causal exclusive speed. full_update includes source_geometry; merge_seconds is nested. Frame wall excludes final output save; worker wall includes it.',model_load_body_seconds=sum(x['seconds'] for x in load),checkpoint_IO_seconds=sum(x['seconds'] for x in check),restore_seconds=sum(x['seconds'] for x in restore),checkpoint_records=check,restore_records=restore,failed_workers=failed,failed_worker_detailed_RSS_IO_unavailable=[w['PID'] for w in failed if not (d.OUT/'worker_resources'/f"{w['name']}.json").exists()],discarded_preflight_cost_seconds=sum(w['elapsed_seconds'] for w in failed),discarded_preflight_model_frames=5,peak_NVML_worker_MiB=max(x['peak_NVML_worker_MiB'] for x in peakpool),peak_GPU_allocated_bytes=max(x['peak_GPU_allocated_bytes'] for x in resource_rows),peak_GPU_reserved_bytes=max(x['peak_GPU_reserved_bytes'] for x in resource_rows),peak_RSS_worker_bytes=max(x['peak_RSS_bytes'] for x in resource_rows),worker_resources=resource_rows,DATA_final_bytes=final_bytes,DATA_peak_sampled_bytes=max(x['DATA_peak_sampled_bytes'] for x in peakpool),DATA_conservative_bound_bytes=conservative,DATA_by_kind={x.name:d.size(x) for x in d.DATA.iterdir()},project_peak_sampled_bytes=max(x['project_peak_sampled_bytes'] for x in peakpool),native_model_weight_file_bytes=5026743569,large_old_weights_and_scores_not_rehashed=True,full_query_scope='Current full RGB read/transfer, official encoder/all24global layers, all4camera rounds and depth, own geometry/merge/update and output; no cached reader represented as full query. Source formation and reading occur causally in the real native chain.',transfer_ledger=dict(actual_current_FP32_RGB_H2D_bytes=3557*3*392*518*4,actual_native_FP32_depth_pose_D2H_bytes=3557*(392*518+9)*4,checkpoint_serialized_state_bytes=[x['state_bytes'] for x in check],checkpoint_partial_outputs_bytes=[x['outputs_bytes'] for x in check],scope='Logical mandatory tensor copies, not measured PCIe bandwidth. Per-merge scalar/index and 2-cost D2H copies, geometry scalar/RNG, D2D append/compaction, allocator effects included in actual full-update/frame/worker envelopes; no free transfer claim. Checkpoint full GPU state copies plus disk atomic backing included in B and physical DATA peak.'))
 c.update(actual_max_simultaneous_workers=maxconcurrent,conservative_concurrency_bound=bound_concurrency,DATA_disk_only_16exit_bound_bytes=filebound,DATA_history_GPU_CPU_disk_work_inclusive_bound_bytes=allhistorybound,DATA_bound_scope='Permanent files plus complete per-worker B (all GPU/CPU/disk state, workspace/readable RGB), two output atomic copies and64MiB IO allowance. Max of this and16 simultaneous disk-exit bound. Native common model/activations separately VRAM/RSS, not new DATA. Past cohost max observed, current max1/card admits at most8; this overcounts shared RGB/backing.')
 assert worker<=8*3600 and c['preflight_cost_seconds']<=3600 and conservative<32*2**30 and c['peak_GPU_reserved_bytes']<=40*2**30
 d.write(d.OUT/'call_accounting.json',counts);d.write(d.OUT/'costs.json',c);d.write(d.OUT/'history_bytes.json',dict(capacity=d.read(d.OUT/'byte_capacity.json'),lifecycle=d.read(d.OUT/'history_lifecycle_bound.json'),actual_per_stream=per,checkpoint_records=check,scope='All exact FP32 source keys/BF16 values; remote full FP32 K/V means; integer n/m/ID, full FP64 geometry, all24layers/special/camera4, RMS/delta, readable RGB, workspace, CPU/GPU/disk recovery copies. No teacher/shadow/discarded content backing; output provenance never fed back. Unallocated128KiBmetadata reserve is not optimal frontier evidence.'))
 print(dict(GPU_worker_hours=worker/3600,processes=len(workers),DATA_conservative_GiB=conservative/2**30),flush=True)
 return c

def geometry_reference():
 """One fixed completed G stream; independently reconstruct source statistics."""
 start=time.monotonic();p=d.plan();rid=p['catalog'][0]['id'];rec=d.read(d.OUT/'completed'/f'G_{rid}.json');saved=d.OUT/'geometry_reference.json'
 if saved.exists():assert d.read(saved)['prediction_sha256']==rec['sha256'];return
 pred=d.load(rec['path']);slots={};delta=None;valid_patches=0;points_used=0
 yy,xx=np.indices((392,518));uv=np.stack((xx+.5,yy+.5,np.ones_like(xx)),-1)
 def source(t):
  pose=pred['poses'][t].double().numpy();x,y,z,w=pose[3:7];qnorm=x*x+y*y+z*z+w*w;s=2/qnorm
  R=np.array([[1-s*(y*y+z*z),s*(x*y-z*w),s*(x*z+y*w)],[s*(x*y+z*w),1-s*(x*x+z*z),s*(y*z-x*w)],[s*(x*z-y*w),s*(y*z+x*w),1-s*(x*x+y*y)]])
  fx=259/np.tan(pose[8]/2);fy=196/np.tan(pose[7]/2);K=np.array([[fx,0,259],[0,fy,196],[0,0,1.]])
  dep=pred['depth'][t].double().numpy();xyz=((uv@np.linalg.inv(K).T)*dep[...,None]-pose[:3])@R
  xyz=xyz.reshape(28,14,37,14,3).transpose(0,2,1,3,4).reshape(1036,196,3);zs=dep.reshape(28,14,37,14).transpose(0,2,1,3).reshape(1036,196)
  ok=np.isfinite(xyz).all(-1)&np.isfinite(zs)&(zs>0)&np.isfinite(pose).all();counts=ok.sum(1);m=(counts>0).astype(np.int64);xyz=np.where(ok[...,None],xyz,0);mu=xyz.sum(1)/np.maximum(counts,1)[:,None];center=np.where(ok[...,None],xyz-mu[:,None],0);cov=np.einsum('npi,npj->nij',center,center)/np.maximum(counts,1)[:,None,None]
  scales=[]
  for i in np.flatnonzero(m):
   zmed=np.median(zs[i,ok[i]]);scale=zmed*zmed*(fx**-2+fy**-2)/2
   if np.isfinite(scale) and scale>0:scales.append(scale)
  return mu,cov,m,(float(np.median(scales)) if scales else None),int(counts.sum())
 # Determine first-valid regularizer from source arrival, independent of when
 # that patch later leaves the exact recent portion.
 for t in range(96):
  if delta is None:
   _,_,_,scale,_=source(t)
   if scale is not None:delta=scale
  if t>=4:
   birth=t-3;mu,cov,m,_,npix=source(birth);valid_patches+=int(m.sum());points_used+=npix
   for patch in range(1036):slots[birth*1041+patch+5]=(int(m[patch]),mu[patch],cov[patch])
  for i,j,ni,nj,mi,mj in pred['merge_records'][t].get('events',[]):
   ai,ui,Si=slots[i];aj,uj,Sj=slots[j];assert ai==mi and aj==mj;total=ai+aj;wi=ai/max(total,1);wj=aj/max(total,1);diff=ui-uj;slots[i]=(total,wi*ui+wj*uj,wi*Si+wj*Sj+wi*wj*np.outer(diff,diff));del slots[j]
 ids=pred['final_remote_IDs'].tolist();mu=np.stack([slots[i][1] for i in ids]);cov=np.stack([slots[i][2] for i in ids]);assert [slots[i][0] for i in ids]==pred['final_m'].tolist()
 actualmu=pred['final_mu'].numpy();actualcov=pred['final_cov'].numpy();result=dict(complete=True,id=rid,policy='G',prediction_sha256=rec['sha256'],CPU_PID=os.getpid(),new_model_calls=0,GT_used=False,source_frames=92,source_patches=92*1036,valid_source_patches=valid_patches,valid_backprojected_pixels=points_used,final_mu_max_abs=float(np.max(np.abs(mu-actualmu))),final_mu_relative_L2=float(np.linalg.norm(mu-actualmu)/max(np.linalg.norm(actualmu),1e-30)),final_covariance_max_abs=float(np.max(np.abs(cov-actualcov))),final_covariance_relative_L2=float(np.linalg.norm(cov-actualcov)/max(np.linalg.norm(actualcov),1e-30)),delta_squared_reference=delta,delta_squared_saved=pred['delta'],delta_squared_relative_difference=abs(delta-pred['delta'])/pred['delta'],seconds=time.monotonic()-start,scope='Independent NumPy quaternion/intrinsic decode, all196 valid pixel population statistics and all recorded pooled covariance merges on fixed first G. Saved policy events are held fixed; this is implementation validation, not a new fitted policy or GT geometry claim. Production official pose decode is FP32 before FP64 statistics; independent decode is FP64, so finite camera rounding differences are disclosed without a new numerical materiality gate.')
 d.write(saved,result);print(dict(geometry_reference=result),flush=True)

if __name__=='__main__':
 begin=time.monotonic();torch.set_num_threads(2);a=argparse.ArgumentParser();a.add_argument('--lineage-available',action='store_true');args=a.parse_args();print(dict(CPU_PID=os.getpid(),CUDA_initialized=torch.cuda.is_initialized()),flush=True)
 if not args.lineage_available:costs()
 lineage()
 if not args.lineage_available:geometry_reference()
 d.write(d.OUT/('CPU_lineage_available_runtime.json' if args.lineage_available else 'CPU_account_runtime.json'),dict(complete=True,PID=os.getpid(),seconds=time.monotonic()-begin,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,CUDA_initialized=torch.cuda.is_initialized(),new_model_calls=0))
