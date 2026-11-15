"""Frozen official Full/Window/common-camera stream; resumable immutable KV chunks."""
import os,sys,time,signal,gc,argparse,resource
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from experiments import stream_history_data as d
STOP=False
def stop(*_):
 global STOP;STOP=True
def sync():torch.cuda.synchronize()
def cpu(x):
 if isinstance(x,torch.Tensor):return x.detach().to('cpu',copy=True).contiguous()
 if isinstance(x,dict):return {k:cpu(v) for k,v in x.items()}
 if isinstance(x,(tuple,list)):return type(x)(cpu(v) for v in x)
 return x
def cuda(x):
 if isinstance(x,torch.Tensor):return x.cuda()
 if isinstance(x,dict):return {k:cuda(v) for k,v in x.items()}
 if isinstance(x,(tuple,list)):return type(x)(cuda(v) for v in x)
 return x
def tensor_bytes(pairs):
 stor={};logical=0
 for pair in pairs:
  if pair is None:continue
  for x in pair:
   logical+=x.numel()*x.element_size();s=x.untyped_storage();stor[(str(x.device),s.data_ptr())]=s.nbytes()
 return dict(logical_bytes=logical,unique_backing_bytes=sum(stor.values()),storages=stor)
def keep_ids(t,capacity=8):return [0]+list(range(max(1,t-capacity+2),t+1)) if t else [0]
def prune(pairs,ids,kept):
 indices=[i for i,f in enumerate(ids) if f in kept]
 if len(indices)==len(ids):return pairs,ids
 return [tuple(x.index_select(2,torch.tensor(indices,device=x.device)) for x in pair) for pair in pairs],[ids[i] for i in indices]
def newstate(model):return dict(full=[None]*24,window=[None]*24,camF=[None]*4,camW=[None]*4,camA=[None]*4,idsW=[],idsCamW=[],idsCamA=[])
def state_cost(state,t):
 result={};union={};position_metadata=1041*2*8+5*8
 for arm,agg,cam in [('F','full','camF'),('A','full','camA'),('W','window','camW')]:
  a=tensor_bytes(state[agg]);b=tensor_bytes(state[cam]);st={**a['storages'],**b['storages']};union.update(st)
  frames=t+1 if arm in ('F','A') else min(t+1,8);camframes=t+1 if arm=='F' else min(t+1,8)
  result[arm]=dict(aggregator_frames=frames,camera_frames=camframes,camera_entries_per_layer=4*camframes,logical_tensor_bytes=a['logical_bytes']+b['logical_bytes'],unique_tensor_backing_bytes=sum(st.values()),necessary_metadata_bytes=position_metadata+8*(frames+4*camframes+1),inference_history_bytes=sum(st.values())+position_metadata+8*(frames+4*camframes+1))
 result['physical_shared_union_tensor_bytes']=sum(union.values());return result
def empty_predictions():return dict(depth_full=[],depth_W=[],poses={k:[] for k in ('F','W','A')},frame_costs=[])
def checkpoint(rid,state,pred,t,chunks,last_saved,plan_sha,name):
 folder=d.DATA/'resume'/rid;folder.mkdir(parents=True,exist_ok=True);old=d.read(folder/'manifest.json') if (folder/'manifest.json').exists() else None
 if t>last_saved:
  path=folder/f'full_{last_saved+1:03d}_{t:03d}.pt';value=[tuple(cpu(x[:,:,last_saved+1:t+1]) for x in pair) for pair in state['full']];d.save(path,value);del value;chunks=chunks+[dict(path=str(path),begin=last_saved+1,end=t,bytes=path.stat().st_size)]
 files={}
 for key in ('window','camF','camW','camA'):
  p=folder/f'{key}_{t:03d}.pt';d.save(p,cpu(state[key]));files[key]=str(p)
 p=folder/f'archive_{t:03d}.pt';d.save(p,pred);files['prediction_archive']=str(p)
 record=dict(complete=True,id=rid,last_frame=t,configuration_sha256=plan_sha,chunks=chunks,files=files,idsW=state['idsW'],idsCamW=state['idsCamW'],idsCamA=state['idsCamA'],CPU_RNG=torch.get_rng_state().tolist(),CUDA_RNG=torch.cuda.get_rng_state().cpu().tolist(),bytes_by_file={k:Path(v).stat().st_size for k,v in files.items()})
 d.write(folder/'manifest.json',record)
 d.write(d.OUT/'checkpoint_records'/f'{rid}_{t:03d}.json',dict(**record,scope='Recovery backing is not free: per-arm own cache files and full chunks count separately from GPU state; prediction_archive is scoring/output-only and never re-read by forward',persistent_cache_file_bytes=sum(x['bytes'] for x in chunks)+sum(n for k,n in record['bytes_by_file'].items() if k!='prediction_archive')))
 if old:
  for p in old['files'].values():
   if p not in files.values():Path(p).unlink(missing_ok=True)
 d.journal(name,kind='complete_checkpoint',id=rid,frame=t,full_delta_begin=last_saved+1,full_delta_end=t,new_full_chunk=t>last_saved,DATA_bytes=d.size(d.DATA))
 return chunks,t
def restore(rid,model,plan_sha,name):
 p=d.DATA/'resume'/rid/'manifest.json'
 if not p.exists():return newstate(model),empty_predictions(),-1,[],-1
 rec=d.read(p);assert rec['configuration_sha256']==plan_sha
 chunks=rec['chunks'];parts=[d.load(x['path']) for x in chunks];state={}
 state['full']=[tuple(torch.cat([a[layer][j] for a in parts],2).cuda() for j in range(2)) for layer in range(24)];del parts
 for key in ('window','camF','camW','camA'):state[key]=cuda(d.load(rec['files'][key]))
 for key in ('idsW','idsCamW','idsCamA'):state[key]=rec[key]
 torch.set_rng_state(torch.tensor(rec['CPU_RNG'],dtype=torch.uint8));torch.cuda.set_rng_state(torch.tensor(rec['CUDA_RNG'],dtype=torch.uint8))
 pred=d.load(rec['files']['prediction_archive']);d.journal(name,kind='restore',id=rid,from_complete_frame=rec['last_frame'],next_frame=rec['last_frame']+1,source='Exact full raw KV chunks, all3camera states, bounded W state, RNG and completed output archive')
 return state,pred,rec['last_frame'],chunks,rec['last_frame']
@torch.inference_mode()
def step(model,image,state,t,capacity=8):
 times={};images=image[None,None].cuda();sync();start=time.monotonic()
 with torch.autocast('cuda',dtype=torch.bfloat16):full,st,state['full']=model.aggregator(images,past_key_values=state['full'],use_cache=True,past_frame_idx=t)
 sync();times['full_aggregator']=time.monotonic()-start;start=time.monotonic()
 with torch.autocast('cuda',dtype=torch.bfloat16):window,stw,state['window']=model.aggregator(images,past_key_values=state['window'],use_cache=True,past_frame_idx=t)
 sync();times['window_aggregator']=time.monotonic()-start;poses={}
 with torch.autocast('cuda',enabled=False):
  for arm,tokens in [('F',full),('A',full),('W',window)]:
   start=time.monotonic();p,state['cam'+arm]=model.camera_head(tokens,past_key_values_camera=state['cam'+arm],use_cache=True);poses[arm]=p[-1][0,0].float().cpu();sync();times['camera_'+arm]=time.monotonic()-start
  start=time.monotonic();depth,_=model.depth_head(full,images=images,patch_start_idx=st);df=depth[0,0,...,0].float().cpu();sync();times['full_depth']=time.monotonic()-start
  start=time.monotonic();depth,_=model.depth_head(window,images=images,patch_start_idx=stw);dw=depth[0,0,...,0].float().cpu();sync();times['window_depth']=time.monotonic()-start
 start=time.monotonic();state['idsW'].append(t);state['idsCamW'].extend([t]*4);state['idsCamA'].extend([t]*4);kept=keep_ids(t,capacity)
 state['window'],state['idsW']=prune(state['window'],state['idsW'],kept)
 for key,ik in [('camW','idsCamW'),('camA','idsCamA')]:state[key],state[ik]=prune(state[key],state[ik],kept)
 assert state['idsW']==kept and state['idsCamW']==state['idsCamA']==[i for i in kept for _ in range(4)]
 assert all(x[0].shape[2]==t+1 for x in state['full'])
 sync();times['postframe_eviction']=time.monotonic()-start
 return df,dw,poses,times
@torch.inference_mode()
def short_full(model,images,name,capacity):
 state=newstate(model);out=[]
 for t,image in enumerate(images):
  # A single full/window path, no redundant A or W heads in this check.
  with torch.autocast('cuda',dtype=torch.bfloat16):tok,st,state['full']=model.aggregator(image[None,None].cuda(),past_key_values=state['full'],use_cache=True,past_frame_idx=t)
  with torch.autocast('cuda',enabled=False):p,state['camF']=model.camera_head(tok,past_key_values_camera=state['camF'],use_cache=True);dep,_=model.depth_head(tok,images=image[None,None].cuda(),patch_start_idx=st)
  out.append(dict(depth=dep[0,0,...,0].float().cpu(),pose=p[-1][0,0].float().cpu()));d.journal(name,kind='preflight_managed_frame',frame=t,capacity=capacity)
  ids=list(range(t+1));state['full'],_=prune(state['full'],ids,keep_ids(t,capacity));state['camF'],_=prune(state['camF'],[j for j in ids for _ in range(4)],keep_ids(t,capacity))
 return out,state
def diffs(a,b):
 x=(a.double()-b.double());return dict(max_abs=float(x.abs().max()),RMS=float(x.square().mean().sqrt()),reference_RMS=float(b.double().square().mean().sqrt()))
@torch.inference_mode()
def preflight(model,meta,name):
 images=torch.stack([d.rgb(f) for f in meta['frames'][:3]])
 from streamvggt.utils.load_fn import load_and_preprocess_images
 original=load_and_preprocess_images([f['rgb_path'] for f in meta['frames'][:3]]);assert torch.equal(images,original)
 own,state=short_full(model,images,name,96);bounded,_=short_full(model,images,name,8)
 with torch.autocast('cuda',dtype=torch.bfloat16):off=model.inference([dict(img=x[None].cuda()) for x in images])
 rows=[]
 for i in range(3):
  rows.append(dict(frame=i,manager_vs_official_depth=diffs(own[i]['depth'],off.ress[i]['depth'][0,...,0].cpu()),manager_vs_official_pose=diffs(own[i]['pose'],off.ress[i]['camera_pose'][0].cpu()),cover_all_Window_depth=diffs(bounded[i]['depth'],own[i]['depth']),cover_all_Window_pose=diffs(bounded[i]['pose'],own[i]['pose'])))
  d.journal(name,kind='preflight_official_frame',frame=i,point_head_also_executed=True)
 del off,bounded
 # All raw entries with their original positions, no merge. Native attention
 # and explicit unordered retrieval use the same SDPA/dtype/norm/RoPE.
 a=model.aggregator.global_blocks[0].attn;N=1041;x=torch.randn(1,N,1024,device='cuda');pos=model.aggregator.position_getter(1,28,37,device=x.device)+1;pos=torch.cat((torch.zeros(1,5,2,device=x.device,dtype=pos.dtype),pos),1)
 with torch.autocast('cuda',dtype=torch.bfloat16):
  native,_=a(x,pos=pos,past_key_values=state['full'][0],use_cache=True)
  q,k,v=a.qkv(x).reshape(1,N,3,a.num_heads,a.head_dim).permute(2,0,3,1,4).unbind(0);k=torch.cat((state['full'][0][0].flatten(2,3),k),2);v=torch.cat((state['full'][0][1].flatten(2,3),v),2);p=pos.repeat(1,4,1);order=torch.arange(k.shape[2]-1,-1,-1,device=x.device)
  q=a.rope(a.q_norm(q),pos);k=a.rope(a.k_norm(k[:,:,order]),p[:,order]);z=F.scaled_dot_product_attention(q,k,v[:,:,order],dropout_p=0);explicit=a.proj_drop(a.proj(z.transpose(1,2).reshape(1,N,1024)))
 comparison=diffs(explicit,native);d.journal(name,kind='preflight_direct_attention',calls=2,all_entries=True,original_positions=True,merged=False)
 assert all(torch.isfinite(v['depth']).all() and torch.isfinite(v['pose']).all() for v in own) and torch.isfinite(explicit).all()
 d.write(d.OUT/'preflight.json',dict(complete=True,preprocessing_exact=True,records=rows,unordered_all_original_position_no_merge=comparison,native_frame_calls=9,direct_attention_calls=2,camera_iterations=4,no_universal_1e5_gate=True,scope='Implementation equivalence only, same dtype and SDPA; finite differences disclosed, no scientific utility conclusion'))
 del state,own,images,original,native,explicit,q,k,v,x;gc.collect();torch.cuda.empty_cache()
def main():
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['profile','formal']);p.add_argument('--rid',required=True);p.add_argument('--name',required=True);args=p.parse_args();start=time.monotonic();signal.signal(signal.SIGTERM,stop);torch.set_num_threads(2);torch.manual_seed(20261114);assert torch.cuda.is_available();torch.cuda.set_per_process_memory_fraction(39*2**30/torch.cuda.get_device_properties(0).total_memory)
 plan=d.plan();meta=next(m for m in plan['catalog'] if m['id']==args.rid);plan_sha=d.sha(d.OUT/'configuration.json');from experiments.gaussian_memory_native_run import model_load
 model=model_load(plan);sync();d.journal(args.name,kind='model_loaded',model_seconds=time.monotonic()-start,CUDA=True,weights_frozen=True)
 for module_name in ('aggregator','camera_head','depth_head','point_head'):
  module=getattr(model,module_name)
  module.register_forward_pre_hook(lambda _m,_a,n=module_name:d.journal(args.name,kind='native_module_begin',module=n))
  module.register_forward_hook(lambda _m,_a,_o,n=module_name:d.journal(args.name,kind='native_module_complete',module=n))
 if args.phase=='profile' and not (d.OUT/'preflight.json').exists():preflight(model,meta,args.name)
 state,pred,last,chunks,last_saved=restore(args.rid,model,plan_sha,args.name);segment_first=last+1;t=last
 for t in range(last+1,96):
  frame_start=time.monotonic();rstart=time.monotonic();image=d.rgb(meta['frames'][t]);input_seconds=time.monotonic()-rstart;df,dw,poses,times=step(model,image,state,t)
  pred['depth_full'].append(df);pred['depth_W'].append(dw)
  for arm in poses:pred['poses'][arm].append(poses[arm])
  cost=state_cost(state,t);times.update(frame=t,input_RGB_seconds=input_seconds,frame_wall_before_checkpoint=time.monotonic()-frame_start,state=cost,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved());pred['frame_costs'].append(times)
  d.journal(args.name,kind='formal_frame',id=args.rid,frame=t,phase=args.phase,aggregator_calls=2,camera_head_calls=3,camera_refinement_rounds=12,depth_head_calls=2,finite_pose=all(bool(torch.isfinite(v).all()) for v in poses.values()),finite_depth=bool(torch.isfinite(df).all() and torch.isfinite(dw).all()))
  if (t+1)%32==0 or STOP:
   q=time.monotonic();chunks,last_saved=checkpoint(args.rid,state,pred,t,chunks,last_saved,plan_sha,args.name);times['checkpoint_seconds']=time.monotonic()-q;d.journal(args.name,kind='checkpoint_IO',id=args.rid,frame=t,seconds=times['checkpoint_seconds'])
  d.write(d.OUT/'progress'/f'{args.rid}.json',dict(id=args.rid,frame=t,phase=args.phase,PID=os.getpid(),name=args.name,state=cost,elapsed_worker_seconds=time.monotonic()-start,DATA_bytes=d.size(d.DATA)))
  if STOP:break
 if t==95:
  result=dict(id=args.rid,configuration_sha256=plan_sha,depth_full=torch.stack(pred['depth_full']),depth_W=torch.stack(pred['depth_W']),poses={k:torch.stack(v) for k,v in pred['poses'].items()},frame_costs=pred['frame_costs'],F_A_depth_shared_exact=True)
  path=d.DATA/'predictions'/f'{args.rid}.pt';d.save(path,result);d.write(d.OUT/'completed'/f'{args.rid}.json',dict(complete=True,id=args.rid,path=str(path),bytes=path.stat().st_size,sha256=d.sha(path),frames=96,configuration_sha256=plan_sha,phase_completed=args.phase,score_not_inspected=args.phase=='profile',final_state=state_cost(state,95)))
  # A completed stream's recovery chunks are owned regenerable temporary state;
  # permanent final predictions and all cost records remain.
  import shutil;shutil.rmtree(d.DATA/'resume'/args.rid)
 d.write(d.OUT/'worker_resources'/f'{args.name}.json',dict(PID=os.getpid(),phase=args.phase,id=args.rid,segment_first_frame=segment_first,last_complete_frame=t,complete=t==95,peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),peak_GPU_reserved_bytes=torch.cuda.max_memory_reserved(),peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,worker_body_seconds=time.monotonic()-start,all_state_on_GPU_no_CPU_offload=True,training_updates=0))
 print(dict(id=args.rid,last=t,complete=t==95,phase=args.phase),flush=True)
 if t<95:raise SystemExit(75)
if __name__=='__main__':main()
