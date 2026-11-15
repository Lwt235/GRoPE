"""Offline GT-only covisibility; fixed far-source selection, never model input."""
import os,time,signal,argparse,resource
import numpy as np
import torch
from experiments import stream_history_data as d
from experiments.gaussian_memory_raw_history_run import unproject,zbuffer
STOP=False
def stop(*_):
 global STOP;STOP=True
@torch.inference_mode()
def source_indices(points,flat_indices,camera,K):
 local=(points-camera[:3,3])@camera[:3,:3];valid=torch.isfinite(local).all(-1)&(local[:,2]>1e-4);local=local[valid];fi=flat_indices[valid];uvw=local@K.T;uv=uvw[:,:2]/uvw[:,2:];inside=(uv[:,0]>=0)&(uv[:,0]<280)&(uv[:,1]>=0)&(uv[:,1]<210);xy=uv[inside].floor().long();z=local[inside,2];dst=xy[:,1]*280+xy[:,0];depth=z.new_full((58800,),float('inf'));depth.scatter_reduce_(0,dst,z,reduce='amin',include_self=True);win=z==depth[dst];idx=torch.full((58800,),58800,device=z.device,dtype=torch.long);idx.scatter_reduce_(0,dst[win],fi[inside][win],reduce='amin',include_self=True);return torch.where(idx<58800,idx,-1).reshape(210,280)
def stratum(x):return 'undefined' if x is None else ('0' if x==0 else ('(0,5%]' if x<=.05 else ('(5,20%]' if x<=.2 else '>20%')))
@torch.inference_mode()
def main():
 p=argparse.ArgumentParser();p.add_argument('--rid',required=True);p.add_argument('--name',required=True);args=p.parse_args();start=time.monotonic();signal.signal(signal.SIGTERM,stop);torch.set_num_threads(2);assert torch.cuda.is_available();meta=next(m for m in d.plan()['catalog'] if m['id']==args.rid)
 depths=[];valids=[]
 for f in meta['frames']:
  a,b=d.sensor(f);depths.append(torch.from_numpy(a));valids.append(torch.from_numpy(b))
 D=torch.stack(depths).cuda().double();V=torch.stack(valids).cuda();C=torch.tensor(meta['c2w_metric'],device='cuda',dtype=torch.float64);K=torch.tensor(meta['K_metric'],device='cuda',dtype=torch.float64)
 clouds=[unproject(D[i],V[i],C[i],K) for i in range(96)];flat=[torch.arange(58800,device='cuda')[V[i].flatten()] for i in range(96)];rows=[];candidates=[];maps=[];calls=0;validation=d.read(d.OUT/'label_validation'/f'{args.rid}.json') if (d.OUT/'label_validation'/f'{args.rid}.json').exists() else None
 # Per-target atomic completion permits exact missing-only arithmetic recovery.
 folder=d.DATA/'labels_partial'/args.rid;folder.mkdir(parents=True,exist_ok=True)
 for t in range(33,96):
  path=folder/f'{t:02d}.pt'
  if path.exists():
   old=d.load(path);rows.append(old['row']);candidates.extend(old['candidates']);maps.append(old['maps']);continue
  target=D[t];valid=V[t];den=int(valid.sum());best_count=-1;best_s=1;best_good=None;union=torch.zeros_like(valid)
  def project_good(s):
   nonlocal calls
   z=zbuffer(clouds[s],C[t],K);calls+=1;return valid&(z>0)&((z-target).abs()<=torch.maximum(target*.05,torch.full_like(target,.05)))
  local_candidates=[]
  for s in range(1,t-31):
   good=project_good(s);n=int(good.sum());union|=good;local_candidates.append(dict(t=t,s=s,consistent_pixels=n,valid_target_pixels=den,coverage=n/den if den else None))
   if n>best_count:best_count=n;best_s=s;best_good=good.clone()
  recent=torch.zeros_like(valid);window=[0]+list(range(max(1,t-7),t))
  for s in window:recent|=project_good(s)
  exclusive=best_good&~recent;union_exclusive=union&~recent;coverage=best_count/den if den else None;ex=float(exclusive.sum())/den if den else None;ue=float(union_exclusive.sum())/den if den else None;idx=source_indices(clouds[best_s],flat[best_s],C[t],K);calls+=1
  row=dict(id=args.rid,t=t,s=best_s,sampled_frame_gap=t-best_s,raw_frame_gap=meta['frames'][t]['index']-meta['frames'][best_s]['index'],elapsed_seconds=(meta['frames'][t]['timestamp_us']-meta['frames'][best_s]['timestamp_us'])/1e6,valid_target_pixels=den,consistent_pixels=best_count,coverage=coverage,revisit=coverage is not None and coverage>=.05,selected_pair_exclusive_fraction=ex,all_far_union_exclusive_fraction=ue,exclusive_stratum=stratum(ex),all_far_union_exclusive_stratum=stratum(ue),window_before_current=window,first_frame_excluded_from_candidate=True,tie_break='earliest sampled s')
  mapping=dict(source_index=idx.cpu().int(),consistent=best_good.cpu(),exclusive=exclusive.cpu(),all_far_union_exclusive=union_exclusive.cpu());d.save(path,dict(row=row,candidates=local_candidates,maps=mapping));rows.append(row);candidates.extend(local_candidates);maps.append(mapping)
  if validation is None:
   from experiments.gaussian_memory_native_analysis import cloud_numpy,raster_numpy
   src=best_s;cloud=cloud_numpy(D[src].cpu().numpy(),V[src].cpu().numpy(),C[src].cpu().numpy(),K.cpu().numpy());z=raster_numpy(cloud,C[t].cpu().numpy(),K.cpu().numpy());gt=target.cpu().numpy();vm=valid.cpu().numpy();expected=vm&(z>0)&(np.abs(z-gt)<=np.maximum(.05,.05*gt));validation=dict(fixed_target=t,selected_source=src,independent_NumPy_consistent_mask_disagreements=int(np.count_nonzero(expected!=best_good.cpu().numpy())),NumPy_count=int(expected.sum()),Torch_count=best_count,scope='Independent old NumPy world-unproject/zbuffer helpers, no model/prediction access')
  if validation is not None:d.write(d.OUT/'label_validation'/f'{args.rid}.json',validation)
  d.journal(args.name,kind='GT_label_target_complete',id=args.rid,t=t,candidate_pairs=len(local_candidates),GT_projection_calls_this_segment=calls,model_calls=0)
  d.write(d.OUT/'label_progress'/f'{args.rid}.json',dict(id=args.rid,t=t,PID=os.getpid(),model_calls=0,elapsed_seconds=time.monotonic()-start))
  if STOP:break
 if len(rows)==63:
  result=dict(id=args.rid,configuration_sha256=d.sha(d.OUT/'configuration.json'),depth=torch.stack(depths),valid=torch.stack(valids),pairs=rows,maps=maps,selection_uses_only_GT=True)
  dest=d.DATA/'labels'/f'{args.rid}.pt';d.save(dest,result);d.write(d.OUT/'label_pairs'/f'{args.rid}.json',dict(complete=True,id=args.rid,pairs=rows,all_candidates=candidates,projection_definition='Source valid .1–10m sensor pixels at210x280,+.5centers; forward world projection/floor target pixel/min-z, target-valid agreement max(.05m,.05target_depth). No GT enters prediction.',selected_exclusive_definition='Selected best far source consistent mask minus union of individually consistent frame0/recent7 masks',additional_union_exclusive_definition='Also retain union of all eligible far sources minus same Window union, separately labelled diagnostic',independent_validation=validation,path=str(dest),sha256=d.sha(dest),GPU_projection_calls_this_segment=calls))
  import shutil;shutil.rmtree(folder)
 d.write(d.OUT/'worker_resources'/f'{args.name}.json',dict(PID=os.getpid(),phase='labels',id=args.rid,complete=len(rows)==63,GT_projection_calls=calls,new_model_calls=0,peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),peak_GPU_reserved_bytes=torch.cuda.max_memory_reserved(),peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,worker_body_seconds=time.monotonic()-start))
 print(dict(id=args.rid,label_targets=len(rows),GT_projection_calls=calls,model_calls=0),flush=True)
 if len(rows)<63:raise SystemExit(75)
if __name__=='__main__':main()
