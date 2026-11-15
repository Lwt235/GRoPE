"""Independent causal raw-patch U/C policies with source-preserving native attention."""
import os,time,signal,argparse,gc,resource,types,sys,shutil
import numpy as np
import torch
from experiments import stream_patch_data as d
from experiments.stream_history_run import cpu,cuda,prune
from experiments.stream_retention_run import tensor_storage,samples
STOP=False
def stop(*_):
 global STOP;STOP=True
def sync():torch.cuda.synchronize()
def recent(t,n):return [0]+list(range(max(1,t-n+1),t+1))
def hash_order(ids):
 a=np.asarray(ids,dtype=np.uint64);x=a^np.uint64(20261115)
 with np.errstate(over='ignore'):
  x=x+np.uint64(0x9e3779b97f4a7c15);x=(x^(x>>np.uint64(30)))*np.uint64(0xbf58476d1ce4e5b9);x=(x^(x>>np.uint64(27)))*np.uint64(0x94d049bb133111eb);x=x^(x>>np.uint64(31))
 return np.lexsort((a,x))
def positions(ids):
 j=ids.remainder(1041);p=(j-5).clamp_min(0);v=torch.stack((p//37+1,p%37+1),-1);v[j<5]=0;return v[None]
def ragged_forward(self,x,pos=None,attn_mask=None,past_key_values=None,use_cache=False):
 B,N,C=x.shape;qkv=self.qkv(x).reshape(B,N,3,self.num_heads,self.head_dim).permute(2,0,3,1,4);q,k,v=qkv.unbind(0)
 if past_key_values is not None:
  pk,pv=past_key_values;k=torch.cat((pk.flatten(2,3),k),2);v=torch.cat((pv.flatten(2,3),v),2);pos_k=torch.cat((self._source_positions,pos),1)
 else:pos_k=pos
 raw=(k.unsqueeze(2),v.unsqueeze(2));q,k=self.q_norm(q),self.k_norm(k)
 if self.rope is not None:q=self.rope(q,pos);k=self.rope(k,pos_k)
 y=torch.nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=attn_mask,dropout_p=0.)
 y=self.proj_drop(self.proj(y.transpose(1,2).reshape(B,N,C)));return y,raw

def centers(depth,pose):
 from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri
 z=depth.reshape(28,14,37,14).permute(0,2,1,3).reshape(1036,196).sort(-1).values;z=(z[:,97]+z[:,98])*.5
 ext,K=pose_encoding_to_extri_intri(pose[None,None],(392,518));ext=ext[0,0];K=K[0,0]
 yy,xx=torch.meshgrid(torch.arange(28,device=z.device),torch.arange(37,device=z.device),indexing='ij');uv=torch.stack((xx.flatten()*14+7,yy.flatten()*14+7,torch.ones(1036,device=z.device)),1).float();local=(uv@torch.linalg.inv(K).T)*z[:,None];world=(local-ext[:,3])@ext[:,:3]
 good=torch.isfinite(world).all(1)&torch.isfinite(z)&(z>0)&torch.isfinite(pose).all();out=torch.zeros((1041,3),device=z.device);out[5:]=torch.where(good[:,None],world,torch.zeros_like(world));valid=torch.zeros(1041,dtype=torch.bool,device=z.device);valid[5:]=good;return out,valid

def newstate():return dict(kv=[None]*24,cam=[None]*4,camids=[],ids=torch.empty(0,dtype=torch.int64,device='cuda'),xyz=torch.empty((0,3),dtype=torch.float32,device='cuda'),valid=torch.empty(0,dtype=torch.bool,device='cuda'),arena=torch.zeros(d.ARENA,dtype=torch.uint8,device='cuda'))
def choose(policy,ids,xyz,valid,t):
 frames=ids//1041;patch=ids.remainder(1041)>=5;near=torch.isin(frames,torch.tensor(recent(t,3),device=ids.device));special=~patch&torch.isin(frames,torch.tensor(recent(t,7),device=ids.device));anchors=patch&near;eligible=torch.nonzero(patch&~near).flatten();M=d.plan()['M'];info=dict(candidates=len(eligible),valid_candidates=int(valid[eligible].sum()),invalid_candidates=int((~valid[eligible]).sum()),anchor_centers=int(valid[anchors].sum()))
 if len(eligible)<=M:chosen=eligible
 elif policy=='U':chosen=eligible[torch.from_numpy(hash_order(ids[eligible].cpu().numpy())[:M].copy()).to(ids.device)]
 else:
  from experiments.stream_patch_fps import fps
  good=eligible[valid[eligible]];k=min(M,len(good));chosen=good[fps(xyz[good],xyz[anchors&valid],k)] if k else good
  if len(chosen)<M:
   available=eligible[~torch.isin(eligible,chosen)];order=torch.from_numpy(hash_order(ids[available].cpu().numpy())[:M-len(chosen)].copy()).to(ids.device);chosen=torch.cat((chosen,available[order]))
  info['invalid_fill']=M-k
 keep=torch.nonzero(special|anchors).flatten();keep=torch.cat((keep,chosen)).sort().values;info.update(remote_retained=len(chosen),retained_tokens=len(keep));return keep,info

def fps_check(name):
 from experiments.stream_patch_fps import fps
 x=torch.tensor([[0.,0,0],[1,0,0],[-1,0,0],[0,2,0],[0,-2,0]],device='cuda');a=x[:1];out=fps(x,a,5).cpu().tolist();assert out==[3,4,1,2,0],out
 d.journal(name,kind='deterministic_FPS_interface',actual_GPU_arithmetic_calls=1,centers=5,anchors=1,result=out,FP64_squared_Euclidean=True,source_ID_ties=True,new_model_calls=0)

def memory(model,state):return tensor_storage(dict(state=state,position=model.aggregator.position_getter.position_cache,RoPE=model.aggregator.rope.frequency_cache))
def save_exit(policy,rid,state,pred,t,name):
 folder=d.DATA/'resume'/f'{policy}_{rid}';folder.mkdir(parents=True,exist_ok=True);assert not (folder/'state.pt').exists();s=time.monotonic();payload=cpu(state);d.save(folder/'state.pt',payload);del payload;d.save(folder/'outputs.pt',pred);rec=dict(policy=policy,id=rid,last=t,next_frame=t+1,configuration_sha256=d.sha(d.OUT/'configuration.json'),samples=samples(state),CPU_RNG=torch.get_rng_state().tolist(),CUDA_RNG=torch.cuda.get_rng_state().cpu().tolist(),state_file_bytes=(folder/'state.pt').stat().st_size,outputs_file_bytes=(folder/'outputs.pt').stat().st_size,retained_IDs=state['ids'].cpu().tolist(),center_samples=state['xyz'][::max(1,len(state['ids'])//5)].cpu().tolist())
 d.write(folder/'manifest.json',rec);d.write(d.OUT/'checkpoint_records'/f'{policy}_{rid}_{t:03d}.json',rec);d.journal(name,kind='complete_exit_checkpoint',policy=policy,id=rid,frame=t,seconds=time.monotonic()-s,state_file_bytes=rec['state_file_bytes'],history_total_bound=d.plan()['total_history_bound_bytes'])
def restore(policy,rid,name):
 folder=d.DATA/'resume'/f'{policy}_{rid}'
 if not (folder/'manifest.json').exists():return newstate(),dict(depth=[],poses=[],frame_costs=[],retained_IDs=[]),0
 rec=d.read(folder/'manifest.json');assert rec['configuration_sha256']==d.sha(d.OUT/'configuration.json');payload=d.load(folder/'state.pt');state=cuda(payload);del payload;assert samples(state)==rec['samples'];assert state['ids'].cpu().tolist()==rec['retained_IDs'];assert state['xyz'][::max(1,len(state['ids'])//5)].cpu().tolist()==rec['center_samples'];torch.set_rng_state(torch.tensor(rec['CPU_RNG'],dtype=torch.uint8));torch.cuda.set_rng_state(torch.tensor(rec['CUDA_RNG'],dtype=torch.uint8));pred=d.load(folder/'outputs.pt');shutil.rmtree(folder);d.journal(name,kind='exact_restore',policy=policy,id=rid,next_frame=rec['next_frame'],completed_queries_repeated=0,consumed_checkpoint_deleted=True);return state,pred,rec['next_frame']

@torch.inference_mode()
def stream(model,policy,rid,name,profiling):
 meta=next(m for m in d.plan()['catalog'] if m['id']==rid);state,pred,first=restore(policy,rid,name);native=policy=='W';eq={'depth_max_abs':0.,'pose_max_abs':0.};old=d.load(d.OLDDATA/'predictions'/f'{rid}.pt') if profiling else None
 if not native:
  for block in model.aggregator.global_blocks:
   assert block.attn.fused_attn and not block.attn.training
   block.attn.forward=types.MethodType(ragged_forward,block.attn)
  if policy=='C' and profiling:fps_check(name)
 for t in range(first,96):
  begin=time.monotonic();s=time.monotonic();image=d.rgb(meta['frames'][t])[None,None].cuda();sync();times=dict(RGB_read_seconds=time.monotonic()-s)
  if not native:
   pp=positions(state['ids'])
   for block in model.aggregator.global_blocks:block.attn._source_positions=pp
  s=time.monotonic()
  with torch.autocast('cuda',dtype=torch.bfloat16):tokens,st,raw=model.aggregator(image,past_key_values=list(state['kv']),use_cache=True,past_frame_idx=t)
  sync();times['aggregator_seconds']=time.monotonic()-s;state['kv']=[None]*24;s=time.monotonic()
  with torch.autocast('cuda',enabled=False):pose,cam=model.camera_head(tokens,past_key_values_camera=list(state['cam']),use_cache=True);pose=pose[-1][0,0].float();depth,_=model.depth_head(tokens,images=image,patch_start_idx=st);depth=depth[0,0,...,0].float()
  sync();times['camera_depth_seconds']=time.monotonic()-s;s=time.monotonic();camids=state['camids']+[t]*4;state['cam']=[None]*4;state['cam'],state['camids']=prune(cam,camids,recent(t,7))
  if native:
   ids=state.get('W_ids',[])+[t];state['kv'],state['W_ids']=prune(raw,ids,recent(t,7));info=dict(retained_frames=state['W_ids'])
  else:
   xyz,valid=centers(depth,pose);ids=torch.cat((state['ids'],torch.arange(t*1041,(t+1)*1041,device='cuda')));xyz=torch.cat((state['xyz'],xyz));valid=torch.cat((state['valid'],valid));keep,info=choose(policy,ids,xyz,valid,t);state['kv']=[tuple(x.index_select(3,keep) for x in pair) for pair in raw];state['ids']=ids[keep];state['xyz']=xyz[keep];state['valid']=valid[keep];pred['retained_IDs'].append(state['ids'].cpu().to(torch.int32));del xyz,ids,valid,keep,pp
   for block in model.aggregator.global_blocks:block.attn._source_positions=None
  del raw,cam,tokens;sync();times['center_selection_eviction_seconds']=time.monotonic()-s;dep=depth.cpu();pos=pose.cpu();pred['depth'].append(dep);pred['poses'].append(pos);del depth,pose
  if profiling and (native or t<7):eq['depth_max_abs']=max(eq['depth_max_abs'],float((dep-old['depth_W'][t]).abs().max()));eq['pose_max_abs']=max(eq['pose_max_abs'],float((pos-old['poses']['W'][t]).abs().max()))
  mem=memory(model,state);assert mem['total_bytes']<=d.plan()['state_copy_bound_bytes']+d.ALLOW
  times.update(frame=t,frame_wall_seconds=time.monotonic()-begin,selection=info,memory=mem);pred['frame_costs'].append(times);d.journal(name,kind='native_frame',policy=policy,id=rid,aggregator_calls=1,camera_head_calls=1,camera_iterations=4,depth_head_calls=1,depth_finite=bool(torch.isfinite(dep).all()),pose_finite=bool(torch.isfinite(pos).all()),**times);d.write(d.OUT/'progress'/f'{policy}_{rid}.json',dict(policy=policy,id=rid,frame=t,PID=os.getpid(),name=name,selection=info,memory=mem));print(dict(policy=policy,id=rid,frame=t,seconds=times['frame_wall_seconds']),flush=True)
  if STOP or (profiling and policy=='U' and t==47 and not (d.OUT/'checkpoint_records'/f'U_{rid}_047.json').exists()):break
 del old
 if t==95:
  path=d.DATA/'predictions'/f'{policy}_{rid}.pt';d.save(path,dict(policy=policy,id=rid,depth=torch.stack(pred['depth']),poses=torch.stack(pred['poses']),frame_costs=pred['frame_costs'],retained_IDs=pred['retained_IDs'],configuration_sha256=d.sha(d.OUT/'configuration.json')));d.write(d.OUT/'completed'/f'{policy}_{rid}.json',dict(complete=True,policy=policy,id=rid,path=str(path),bytes=path.stat().st_size,sha256=d.sha(path),frames=96,profiling=profiling,first7_or_W_full_replay=eq if profiling else None))
 else:save_exit(policy,rid,state,pred,t,name)
 return t

def main():
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['profile','formal']);p.add_argument('--policy',required=True);p.add_argument('--rid',required=True);p.add_argument('--name',required=True);a=p.parse_args();signal.signal(signal.SIGTERM,stop);start=time.monotonic();torch.set_num_threads(2);torch.manual_seed(20261114);assert torch.cuda.is_available();torch.cuda.set_per_process_memory_fraction(39*2**30/torch.cuda.get_device_properties(0).total_memory)
 from experiments.gaussian_memory_native_run import model_load
 model=model_load(d.plan());sync();d.journal(a.name,kind='model_loaded',seconds=time.monotonic()-start,CUDA=True,frozen=not any(p.requires_grad for p in model.parameters()))
 for key in ('aggregator','camera_head','depth_head'):
  mod=getattr(model,key);mod.register_forward_pre_hook(lambda m,x,k=key:d.journal(a.name,kind='module_begin',module=k));mod.register_forward_hook(lambda m,x,y,k=key:d.journal(a.name,kind='module_complete',module=k))
 last=stream(model,a.policy,a.rid,a.name,a.phase=='profile');io={k:int(v) for k,v in (line.split(':') for line in open('/proc/self/io'))};d.write(d.OUT/'worker_resources'/f'{a.name}.json',dict(PID=os.getpid(),complete=last==95,last_frame=last,body_seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),peak_GPU_reserved_bytes=torch.cuda.max_memory_reserved(),process_IO_counters=io));print(dict(complete=last==95),flush=True)
 if last!=95:raise SystemExit(75)
if __name__=='__main__':main()
