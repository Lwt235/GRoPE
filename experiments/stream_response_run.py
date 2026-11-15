"""One frozen causal full-F stream, source-only moments at fixed native layers."""
import os,time,signal,argparse,types,resource,shutil,gc
import numpy as np
import torch
from experiments import stream_response_data as d
from experiments import stream_response_moments as mm
from experiments.stream_patch_run import centers
from experiments.stream_history_run import cpu,cuda
from experiments.stream_retention_run import tensor_storage,samples
STOP=False
CONTEXT=None
def stop(*_):
 global STOP;STOP=True

def attention(self,x,pos=None,attn_mask=None,past_key_values=None,use_cache=False):
 B,N,C=x.shape;qkv=self.qkv(x).reshape(B,N,3,self.num_heads,self.head_dim).permute(2,0,3,1,4);q,k,v=qkv.unbind(0);pos_k=pos
 if use_cache:
  k=k.unsqueeze(2);v=v.unsqueeze(2)
  if past_key_values is not None:k=torch.cat([past_key_values[0],k],2);v=torch.cat([past_key_values[1],v],2)
  new_kv=(k,v);frames=k.shape[2];k=k.flatten(2,3);v=v.flatten(2,3)
  if pos_k is not None:pos_k=pos_k.repeat(1,frames,1)
 q,k=self.q_norm(q),self.k_norm(k)
 if self.rope is not None:q=self.rope(q,pos);k=self.rope(k,pos_k)
 assert self.fused_attn and not self.training
 y=torch.nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=attn_mask,dropout_p=0.)
 if CONTEXT['t'] in (31,63,95):
  with torch.autocast('cuda',enabled=False):capture(self,q[0],k[0],v[0],y[0])
 y=self.proj_drop(self.proj(y.transpose(1,2).reshape(B,N,C)))
 return (y,new_kv) if use_cache else y

def capture(module,q,a,v,native):
 ctx=CONTEXT;t=ctx['t'];layer=module._moment_layer;case=f"{ctx['rid']}_{t:02d}_{layer:02d}";path=d.DATA/'cases'/f'{case}.pt';assert not path.exists(),f'No repeated teacher capture allowed: {case}';begin=time.monotonic()
 if ctx.get('tree_t')!=t:
  s=time.monotonic();xyz=ctx['state']['xyz'];valid=ctx['state']['valid'];ids=torch.arange(t*1041,device='cuda');eligible=(ids//1041>=1)&(ids//1041<=t-8)&(ids.remainder(1041)>=5);remote=(eligible&valid).nonzero().flatten();source=xyz[remote];coarse,fine=mm.tree(source.cpu().numpy());ctx.update(tree_t=t,remote=remote,source=source,coarse=coarse,fine=fine);ctx['tree_seconds']=time.monotonic()-s
  tree=dict(t=t,id=ctx['rid'],remote_IDs=remote.cpu(),source_XYZ=source.cpu(),coarse={k:torch.from_numpy(v.copy()).int() for k,v in coarse.items()},fine={k:torch.from_numpy(v.copy()).int() for k,v in fine.items()},valid_remote=len(remote),invalid_remote=int((eligible&~valid).sum()),eligible_remote=int(eligible.sum()),tree_seconds=ctx['tree_seconds'],common_all_four_layers=True,query_or_feature_selection=False)
  d.save(d.DATA/'geometry'/f"{ctx['rid']}_{t:02d}.pt",tree)
 remote=ctx['remote'];states,stats=mm.fit(a[:,remote].float(),v[:,remote].float(),ctx['source'],ctx['coarse'],ctx['fine']);stats.pop('group_source_indices');stats.pop('fine_group_source_indices');records=[];decoded={}
 for method in ('X','K','M'):
  p=d.DATA/'encoded'/f'{case}_{method}.bin';rec=mm.encode(p,states[method],method);decoded[method],rs=mm.decode(p,'cuda');rec['read_CPU_GPU_seconds']=rs
  for key in ('means','counts'):
   assert torch.equal(decoded[method][key],states[method][key]),(method,key)
  if method!='M':assert torch.equal(decoded[method]['slopes'],states[method]['slopes'])
  records.append(rec)
 del states;result,validation=mm.responses(q,a,v,native,module.proj.weight,remote,decoded);first=(ctx['rid']==d.plan()['catalog'][0]['id'] and t==31 and layer==6)
 if first:
  d.save(d.DATA/'validation'/'first_case.pt',dict(q=q[:,:6].cpu(),a=a.cpu(),v=v.cpu(),remote=remote.cpu(),weight=module.proj.weight.float().cpu(),GPU_outputs=validation,case=case,source='actualpostnorm/sourceRoPE,all16heads,first6queries',native_precision=str(q.dtype),projection_bias_not_applied_to_differences=True))
 for rec in records:
  p=d.DATA/'encoded'/f"{case}_{rec['method']}.bin";rec['preserved_first_fixed_case']=first
  if not first:p.unlink()
 result.update(id=ctx['rid'],frame=t,global_layer=layer,fit=stats,encoded=records,teacher_no_approx_feedback=True,actual_query_count=q.shape[1],actual_heads=q.shape[0],tree_seconds=ctx['tree_seconds'],full_teacher_state_bytes=tensor_storage(ctx['state'])['total_bytes'],case_seconds=time.monotonic()-begin,configuration_sha256=ctx['config_sha'])
 d.save(path,result);d.write(d.OUT/'case_receipts'/f'{case}.json',dict(complete=True,id=ctx['rid'],frame=t,global_layer=layer,path=str(path),bytes=path.stat().st_size,sha256=d.sha(path),remote_tokens=len(remote),exact_tokens=result['exact_tokens'],queries=1041,heads=16,case_seconds=result['case_seconds'],fit_seconds=stats['seconds'],response_timing=result['timing'],encoded=records));d.journal(ctx['name'],kind='response_capture',id=ctx['rid'],frame=t,global_layer=layer,queries=1041,heads=16,remote_tokens=len(remote),exact_tokens=result['exact_tokens'],seconds=result['case_seconds'],fit_seconds=stats['seconds'],response_timing=result['timing']);print(dict(case=case,seconds=result['case_seconds'],remote=len(remote)),flush=True)
 del result,decoded;gc.collect()

def checkpoint(state,pred,next_frame,rid,name,reason):
 folder=d.DATA/'resume'/rid;folder.mkdir(parents=True,exist_ok=True);assert not (folder/'state.pt').exists();s=time.monotonic();d.save(folder/'predictions.pt',pred)
 # Stream CUDA tensors directly through torch's serializer: no retained CPU mirror.
 d.save(folder/'state.pt',state);rec=dict(id=rid,next_frame=next_frame,last_completed_frame=next_frame-1,state_bytes=(folder/'state.pt').stat().st_size,prediction_bytes=(folder/'predictions.pt').stat().st_size,configuration_sha256=d.sha(d.OUT/'configuration.json'),samples=samples(state),CPU_RNG=torch.get_rng_state().tolist(),CUDA_RNG=torch.cuda.get_rng_state().cpu().tolist(),reason=reason,no_completed_teacher_frame_repeated=True,write_seconds=time.monotonic()-s);assert rec['state_bytes']<=d.plan()['checkpoint_full_payload_bound_bytes'];d.write(folder/'manifest.json',rec);d.write(d.OUT/'checkpoint_records'/f'{rid}_{next_frame:02d}_{time.time_ns()}.json',rec);d.journal(name,kind='exact_full_F_checkpoint',**rec)

def restore(rid,name):
 folder=d.DATA/'resume'/rid
 if not (folder/'manifest.json').exists():return dict(kv=[None]*24,cam=[None]*4,xyz=torch.empty((0,3),device='cuda'),valid=torch.empty(0,dtype=torch.bool,device='cuda')),dict(depth=[],poses=[],frame_costs=[]),0
 rec=d.read(folder/'manifest.json');assert rec['configuration_sha256']==d.sha(d.OUT/'configuration.json');s=time.monotonic();state=cuda(d.load(folder/'state.pt'));assert samples(state)==rec['samples'];pred=d.load(folder/'predictions.pt');torch.set_rng_state(torch.tensor(rec['CPU_RNG'],dtype=torch.uint8));torch.cuda.set_rng_state(torch.tensor(rec['CUDA_RNG'],dtype=torch.uint8));shutil.rmtree(folder);d.journal(name,kind='exact_full_F_restore',id=rid,next_frame=rec['next_frame'],seconds=time.monotonic()-s,completed_frames_repeated=0,consumed_checkpoint_deleted=True);return state,pred,rec['next_frame']

@torch.inference_mode()
def stream(model,rid,name,until):
 global CONTEXT
 meta=next(x for x in d.plan()['catalog'] if x['id']==rid);state,pred,start=restore(rid,name);CONTEXT=dict(rid=rid,name=name,state=state,config_sha=d.sha(d.OUT/'configuration.json'))
 for layer in (6,12,18,24):
  mod=model.aggregator.global_blocks[layer-1].attn;mod._moment_layer=layer;mod.forward=types.MethodType(attention,mod)
 for t in range(start,96):
  CONTEXT['t']=t;begin=time.monotonic();s=time.monotonic();image=d.rgb(meta['frames'][t])[None,None].cuda();mm.sync();timing=dict(RGB_read_transfer_seconds=time.monotonic()-s);s=time.monotonic()
  with torch.autocast('cuda',dtype=torch.bfloat16):tokens,st,raw=model.aggregator(image,past_key_values=state['kv'],use_cache=True,past_frame_idx=t)
  state['kv']=raw;mm.sync();timing['full_F_aggregator_and_capture_seconds']=time.monotonic()-s;s=time.monotonic()
  with torch.autocast('cuda',enabled=False):pose,cam=model.camera_head(tokens,past_key_values_camera=state['cam'],use_cache=True);pose=pose[-1][0,0].float();depth,_=model.depth_head(tokens,images=image,patch_start_idx=st);depth=depth[0,0,...,0].float()
  state['cam']=cam;mm.sync();timing['full_F_camera_depth_seconds']=time.monotonic()-s;s=time.monotonic();xyz,valid=centers(depth,pose);state['xyz']=torch.cat((state['xyz'],xyz));state['valid']=torch.cat((state['valid'],valid));pred['depth'].append(depth.cpu());pred['poses'].append(pose.cpu());del xyz,valid,depth,pose,tokens,raw,cam,image
  if CONTEXT.get('tree_t')==t:
   for key in ('source','remote','coarse','fine'):CONTEXT.pop(key,None)
  mm.sync();timing.update(frame=t,wall_seconds=time.monotonic()-begin,teacher_state_storage=tensor_storage(state),CUDA_allocated=torch.cuda.memory_allocated(),CUDA_reserved=torch.cuda.memory_reserved());pred['frame_costs'].append(timing);d.journal(name,kind='native_F_frame',id=rid,frame=t,aggregator_calls=1,camera_head_calls=1,camera_iterations=4,depth_head_calls=1,approximate_feedback=False,**{k:v for k,v in timing.items() if k!='frame'});d.write(d.OUT/'progress'/f'{rid}.json',dict(id=rid,frame=t,PID=os.getpid(),name=name,last_native_seconds=timing['wall_seconds'],teacher_state_bytes=timing['teacher_state_storage']['total_bytes']));print(dict(id=rid,frame=t,seconds=timing['wall_seconds']),flush=True);torch.cuda.empty_cache()
  if STOP or t>=until:break
 if t==95:
  path=d.DATA/'predictions'/f'{rid}.pt';d.save(path,dict(depth=torch.stack(pred['depth']),poses=torch.stack(pred['poses']),frame_costs=pred['frame_costs'],id=rid));d.write(d.OUT/'completed'/f'{rid}.json',dict(complete=True,id=rid,frames=96,cases=12,path=str(path),bytes=path.stat().st_size,sha256=d.sha(path),no_approximate_teacher_feedback=True));return True
 checkpoint(state,pred,t+1,rid,name,'external_or_own_SIGTERM' if STOP else 'fixed_prefix_capacity_stage');return False

def main():
 p=argparse.ArgumentParser();p.add_argument('--rid',required=True);p.add_argument('--name',required=True);p.add_argument('--until',type=int,default=95);a=p.parse_args();signal.signal(signal.SIGTERM,stop);start=time.monotonic();torch.set_num_threads(2);torch.manual_seed(20261114);assert torch.cuda.is_available();torch.cuda.set_per_process_memory_fraction(39*2**30/torch.cuda.get_device_properties(0).total_memory)
 from experiments.gaussian_memory_native_run import model_load
 model=model_load(d.plan());mm.sync();d.journal(a.name,kind='model_loaded',seconds=time.monotonic()-start,CUDA=True,frozen=not any(p.requires_grad for p in model.parameters()),GPU_name=torch.cuda.get_device_name())
 for key in ('aggregator','camera_head','depth_head'):
  mod=getattr(model,key);mod.register_forward_pre_hook(lambda m,x,k=key:d.journal(a.name,kind='module_begin',module=k));mod.register_forward_hook(lambda m,x,y,k=key:d.journal(a.name,kind='module_complete',module=k))
 complete=stream(model,a.rid,a.name,a.until);io={k:int(v) for k,v in (line.split(':') for line in open('/proc/self/io'))};d.write(d.OUT/'worker_resources'/f'{a.name}.json',dict(PID=os.getpid(),complete=complete,body_seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),peak_GPU_reserved_bytes=torch.cuda.max_memory_reserved(),process_IO_counters=io));print(dict(complete=complete),flush=True)
 if not complete:raise SystemExit(75)
if __name__=='__main__':main()
