"""Own-history native W/T/D retention, exact causal KV and bounded descriptors."""
import os,time,signal,argparse,gc,resource,itertools,sys
from pathlib import Path
import numpy as np
import torch
from experiments import stream_retention_data as d
from experiments.stream_history_run import cpu,cuda,prune
STOP=False
def stop(*_):
 global STOP;STOP=True
def sync():torch.cuda.synchronize()
def recent(t,n):return [0]+list(range(max(1,t-n+1),t+1))
def choose_remote(policy,remote,c,recent_ids,desc):
 candidates=sorted(set(remote+[c]))
 if len(candidates)<=4:return candidates,None
 def objective(kept):
  if policy=='T':
   locations=[0]+list(kept);return max([b-a for a,b in zip(locations,locations[1:])]+[2*(c-locations[-1])])
  center=list(kept)+recent_ids;z=np.stack([desc[i].double().numpy() for i in candidates]);q=np.stack([desc[i].double().numpy() for i in center]);return float(np.max(np.min(1-z@q.T,axis=1)))
 choices=list(itertools.combinations(candidates,4));chosen=min(choices,key=lambda x:(objective(x),x));return list(chosen),objective(chosen)
def state_new():return dict(kv=[None]*24,cam=[None]*4,ids=[],camids=[],remote=[],arena=torch.zeros(d.METADATA_ARENA_BYTES,dtype=torch.uint8),descids=[])
def desc_pool(state):return state['arena'][:9*1024*4].view(torch.float32).reshape(9,1024)
def samples(state):
 out={}
 for key in ('kv','cam'):
  out[key]=[]
  for pair in state[key]:
   row=[]
   for x in pair:
    v=x.reshape(-1);indices=sorted({0,len(v)//7,len(v)//3,len(v)//2,len(v)-1});row.append(dict(shape=list(x.shape),dtype=str(x.dtype),sample=v[indices].float().cpu().tolist()))
   out[key].append(row)
 return out
def tensor_storage(obj):
 seen=set();stor={};headers=0
 def visit(x):
  nonlocal headers
  if id(x) in seen:return
  seen.add(id(x));headers+=sys.getsizeof(x)
  if isinstance(x,torch.Tensor):
   a=x.untyped_storage();stor[(str(x.device),a.data_ptr())]=a.nbytes()
  elif isinstance(x,dict):
   for k,v in x.items():visit(k);visit(v)
  elif isinstance(x,(tuple,list)):
   for v in x:visit(v)
 visit(obj);return dict(unique_tensor_bytes=sum(stor.values()),Python_object_bytes=headers,total_bytes=sum(stor.values())+headers)
def memory(model,state,t):
 raw=tensor_storage(dict(kv=state['kv'],cam=state['cam']))['unique_tensor_bytes'];measured=tensor_storage(dict(state=state,position=model.aggregator.position_getter.position_cache,RoPE=model.aggregator.rope.frequency_cache,past_frame_idx=t+1));return dict(raw_KV_camera_bytes=raw,**measured)
def record_B(model,state,t):
 assert len(state['ids'])==8 and len(state['camids'])==32
 raw=memory(model,state,t)['raw_KV_camera_bytes'];p=d.plan();B=3*(raw+d.METADATA_ARENA_BYTES)+p['RGB_readable_batch_backing_bytes']+d.SERIALIZATION_ALLOWANCE_BYTES
 d.write(d.OUT/'byte_ceiling.json',dict(registered_utc=d.now(),before_any_T_D_scientific_effect_scoring=True,W_frame=t,W_measured=memory(model,state,t),raw8_frame_KV_camera_bytes=raw,common_metadata_arena_bytes=d.METADATA_ARENA_BYTES,whole_selected_RGB_batch_backing_bytes=p['RGB_readable_batch_backing_bytes'],common_serialization_metadata_allowance_bytes=d.SERIALIZATION_ALLOWANCE_BYTES,B_total_history_bytes=B,formula='3*(W8rawKVcamera+common128KiB arena)+whole12stream readableRGB backing+1MiB Python/nativegrid/ZIP/RNG metadata allowance',bound_scope='Normal forward: at most8retained frames; native current KV and out-of-place eviction can temporarily coexist old9+new8, bounded conservatively by2*9/8 raw8. Exit save/restore: GPU exact8 + CPU exact8 + one disk exact8; no old checkpoint remains in continuing run. Same128KiB arena/recovery for all. Unused pooled descriptor rows zeroed; no ninth persistent frame.',no_extra_frame_capacity=True,weight_current_activations_and_scoring_archives_separate=True))
def empty():return dict(depth=[],poses=[],frame_costs=[])
def save_exit(policy,rid,state,pred,last,name):
 folder=d.DATA/'resume'/f'{policy}_{rid}';folder.mkdir(parents=True,exist_ok=True);assert not (folder/'state.pt').exists(),'Consumed checkpoint must be removed before any continuation'
 begin=time.monotonic();payload=cpu(dict(kv=state['kv'],cam=state['cam'],arena=state['arena']));d.save(folder/'state.pt',payload);del payload
 d.save(folder/'outputs.pt',pred);rec=dict(policy=policy,id=rid,last=last,next_frame=last+1,ids=state['ids'],camids=state['camids'],remote=state['remote'],descids=state['descids'],state_samples=samples(state),CPU_RNG=torch.get_rng_state().tolist(),CUDA_RNG=torch.cuda.get_rng_state().cpu().tolist(),configuration_sha256=d.sha(d.OUT/'configuration.json'),state_file_bytes=(folder/'state.pt').stat().st_size,outputs_archive_bytes=(folder/'outputs.pt').stat().st_size)
 d.write(folder/'manifest.json',rec);d.write(d.OUT/'checkpoint_records'/f'{policy}_{rid}_{last:03d}.json',rec);d.journal(name,kind='complete_exit_checkpoint',policy=policy,id=rid,frame=last,seconds=time.monotonic()-begin,state_file_bytes=rec['state_file_bytes'],outputs_archive_bytes=rec['outputs_archive_bytes'])
def restore(policy,rid,name):
 import shutil
 folder=d.DATA/'resume'/f'{policy}_{rid}'
 if not (folder/'manifest.json').exists():return state_new(),empty(),-1
 rec=d.read(folder/'manifest.json');assert rec['configuration_sha256']==d.sha(d.OUT/'configuration.json');payload=d.load(folder/'state.pt');state=state_new();state['kv']=cuda(payload['kv']);state['cam']=cuda(payload['cam']);state['arena']=payload['arena'];del payload
 for k in ('ids','camids','remote','descids'):state[k]=rec[k]
 assert samples(state)==rec['state_samples'];torch.set_rng_state(torch.tensor(rec['CPU_RNG'],dtype=torch.uint8));torch.cuda.set_rng_state(torch.tensor(rec['CUDA_RNG'],dtype=torch.uint8));pred=d.load(folder/'outputs.pt');assert len(pred['depth'])==rec['next_frame']
 shutil.rmtree(folder);d.journal(name,kind='exact_restore',policy=policy,id=rid,last_complete=rec['last'],next_frame=rec['next_frame'],all28_KV_pairs_56tensors_shapes_dtypes_samples_exact=True,CPU_CUDA_RNG_restored=True,consumed_checkpoint_deleted_before_next_query=True,model_queries_repeated=0)
 return state,pred,rec['last']
@torch.inference_mode()
def step(model,image,state,t,policy,descriptor):
 image=image[None,None].cuda();sync();times={};begin=time.monotonic()
 with torch.autocast('cuda',dtype=torch.bfloat16):tokens,st,state['kv']=model.aggregator(image,past_key_values=state['kv'],use_cache=True,past_frame_idx=t)
 sync();times['aggregator_seconds']=time.monotonic()-begin;begin=time.monotonic()
 with torch.autocast('cuda',enabled=False):pose,state['cam']=model.camera_head(tokens,past_key_values_camera=state['cam'],use_cache=True);pose=pose[-1][0,0].float().cpu();depth,_=model.depth_head(tokens,images=image,patch_start_idx=st);depth=depth[0,0,...,0].float().cpu()
 sync();times['camera_depth_seconds']=time.monotonic()-begin;begin=time.monotonic();oldids=state['ids']+[t];state['camids'].extend([t]*4);objective=None
 if policy=='W':kept=recent(t,7)
 else:
  small=recent(t,3);remote=state['remote']
  if policy=='D':
   desc={i:desc_pool(state)[j].clone() for j,i in enumerate(state['descids'])};desc[t]=descriptor['current'];assert all(torch.isfinite(v).all() for v in desc.values())
  else:desc={}
  if t>=4:remote,objective=choose_remote(policy,remote,t-3,small,desc)
  kept=sorted(set(small+remote));state['remote']=remote
  if policy=='D':
   values=torch.stack([desc[i] for i in kept]);desc_pool(state).zero_();desc_pool(state)[:len(kept)].copy_(values);state['descids']=kept;descriptor.clear()
 state['kv'],state['ids']=prune(state['kv'],oldids,kept);camkeep=recent(t,7);state['cam'],state['camids']=prune(state['cam'],state['camids'],camkeep)
 assert len(kept)<=8 and state['ids']==kept and state['camids']==[i for i in camkeep for _ in range(4)]
 assert all(pair[0].shape[2]==len(kept) for pair in state['kv']);sync();times['selection_eviction_seconds']=time.monotonic()-begin
 return depth,pose,times,objective
@torch.inference_mode()
def stream(model,policy,rid,name,limit=96,profiling=False):
 meta=next(m for m in d.plan()['catalog'] if m['id']==rid);state,pred,last=restore(policy,rid,name);first=last+1;descriptor={};hook=None
 if policy=='D':
  def capture(module,args,out):
   begin=time.monotonic();x=out['x_norm_patchtokens'] if isinstance(out,dict) else out;z=x.float().mean(1);z=torch.nn.functional.normalize(z,dim=-1,eps=1e-12);descriptor['current']=z[0].cpu();sync();d.journal(name,kind='descriptor_from_existing_patch_embed',policy=policy,id=rid,seconds=time.monotonic()-begin,extra_encoder_calls=0)
  hook=model.aggregator.patch_embed.register_forward_hook(capture)
 t=last
 for t in range(first,limit):
  begin=time.monotonic();r=time.monotonic();im=d.rgb(meta['frames'][t]);read=time.monotonic()-r;depth,pose,times,obj=step(model,im,state,t,policy,descriptor);mem=memory(model,state,t);times.update(frame=t,RGB_read_seconds=read,frame_wall_seconds=time.monotonic()-begin,ids=state['ids'].copy(),remote=state['remote'].copy(),selection_objective=obj,memory=mem);pred['depth'].append(depth);pred['poses'].append(pose);pred['frame_costs'].append(times)
  if policy=='W' and t==7 and not (d.OUT/'byte_ceiling.json').exists():record_B(model,state,t)
  if (d.OUT/'byte_ceiling.json').exists():
   B=d.read(d.OUT/'byte_ceiling.json');raw8=B['raw8_frame_KV_camera_bytes'];assert mem['total_bytes']<raw8+d.METADATA_ARENA_BYTES+d.SERIALIZATION_ALLOWANCE_BYTES;assert 3*(raw8+d.METADATA_ARENA_BYTES)+B['whole_selected_RGB_batch_backing_bytes']+d.SERIALIZATION_ALLOWANCE_BYTES==B['B_total_history_bytes']
  d.journal(name,kind='native_frame',policy=policy,id=rid,frame=t,aggregator_calls=1,camera_head_calls=1,camera_iterations=4,depth_head_calls=1,pose_finite=bool(torch.isfinite(pose).all()),depth_finite=bool(torch.isfinite(depth).all()),retained=state['ids'],camframes=sorted(set(state['camids'])))
  d.write(d.OUT/'progress'/f'{policy}_{rid}.json',dict(policy=policy,id=rid,frame=t,PID=os.getpid(),name=name,memory=mem))
  if STOP:break
 if hook:hook.remove()
 if t==95:
  path=d.DATA/'predictions'/f'{policy}_{rid}.pt';result=dict(policy=policy,id=rid,configuration_sha256=d.sha(d.OUT/'configuration.json'),depth=torch.stack(pred['depth']),poses=torch.stack(pred['poses']),frame_costs=pred['frame_costs']);d.save(path,result);d.write(d.OUT/'completed'/f'{policy}_{rid}.json',dict(complete=True,policy=policy,id=rid,path=str(path),bytes=path.stat().st_size,sha256=d.sha(path),frames=96,profiling=profiling,final_memory=memory(model,state,t)))
 else:save_exit(policy,rid,state,pred,t,name)
 d.journal(name,kind='stream_segment_complete',policy=policy,id=rid,first_frame=first,last_frame=t,complete=t==95,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),memory=memory(model,state,t))
 return pred,t

def main():
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['profile','formal']);p.add_argument('--policy',default='W');p.add_argument('--rid',default='fidelity_00');p.add_argument('--name',required=True);a=p.parse_args();signal.signal(signal.SIGTERM,stop);start=time.monotonic();torch.set_num_threads(2);torch.manual_seed(20261114);assert torch.cuda.is_available();torch.cuda.set_per_process_memory_fraction(39*2**30/torch.cuda.get_device_properties(0).total_memory)
 from experiments.gaussian_memory_native_run import model_load
 model=model_load(d.plan());sync();d.journal(a.name,kind='model_loaded',seconds=time.monotonic()-start,CUDA=True,training=False)
 for key in ('aggregator','camera_head','depth_head','point_head'):
  mod=getattr(model,key);mod.register_forward_pre_hook(lambda m,x,k=key:d.journal(a.name,kind='module_begin',module=k));mod.register_forward_hook(lambda m,x,y,k=key:d.journal(a.name,kind='module_complete',module=k))
 complete=False
 if a.phase=='profile':
  W,last=stream(model,'W','fidelity_00',a.name,profiling=True);assert last==95
  old=d.load(d.OLDDATA/'predictions/fidelity_00.pt');replay=dict(depth_max_abs=float((torch.stack(W['depth'])-old['depth_W']).abs().max()),pose_max_abs=float((torch.stack(W['poses'])-old['poses']['W']).abs().max()),native_scores_not_recomputed=True);del old
  eq={}
  for policy in ('T','D'):
   pred,last=stream(model,policy,'fidelity_00',a.name,limit=8,profiling=True);assert last==7;eq[policy]=dict(first8_depth_max_abs=float((torch.stack(pred['depth'])-torch.stack(W['depth'][:8])).abs().max()),first8_pose_max_abs=float((torch.stack(pred['poses'])-torch.stack(W['poses'][:8])).abs().max()),all8history_covered=True,own_state_written=True);del pred;gc.collect();torch.cuda.empty_cache()
  d.write(d.OUT/'preflight.json',dict(complete=True,W_fixed96_cost_replay=replay,T_D_covering_all8=eq,descriptor_encoder_extra_calls=0,formal_T_D_prefixes_reused=True,new_scientific_scores_not_inspected=True));complete=True
 else:
  pred,last=stream(model,a.policy,a.rid,a.name);complete=last==95
 d.write(d.OUT/'worker_resources'/f'{a.name}.json',dict(PID=os.getpid(),phase=a.phase,policy=a.policy,id=a.rid,complete=complete,worker_body_seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),peak_GPU_reserved_bytes=torch.cuda.max_memory_reserved()))
 print(dict(phase=a.phase,policy=a.policy,id=a.rid,complete=complete),flush=True)
 if not complete:raise SystemExit(75)
if __name__=='__main__':main()
