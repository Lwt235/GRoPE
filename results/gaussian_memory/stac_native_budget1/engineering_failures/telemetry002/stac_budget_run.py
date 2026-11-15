"""Fixed real-author S and matched H/W; chunk-prefix snapshots, no GT reads."""
import os,sys,time,signal,argparse,resource,gc
from pathlib import Path
from experiments import stac_budget_data as d
sys.path[:0]=[str(d.AUTHOR),str(d.AUTHOR/'attn-cuda'),str(d.AUTHOR/'merger-cuda'),str(d.ROOT/'third_party/native_history_python')]
import torch
from experiments.stream_history_data import rgb
from experiments.stac_budget_loadcheck import load_author
from experiments import stac_budget_memory as mem
from stac.kv_manager import KVManager
class WindowKV(KVManager):
 def decode_sparse_attn(self,query_states,layer_idx):
  import attn_cuda
  slot=self._to_slot(layer_idx);n=self._offset_hot[slot]
  k=self.key_cache_hot[slot,:,:n].unsqueeze(0).transpose(1,2).contiguous();v=self.value_cache_hot[slot,:,:n].unsqueeze(0).transpose(1,2).contiguous()
  return attn_cuda.flash_attn_bias_colsum(query_states.contiguous(),k,v,return_colsum=False)[0]

STOP=False
ACTIVE_METER=None
def stop(*_):
 global STOP;STOP=True
def sync():torch.cuda.synchronize()

class MovedTensor:
 def __init__(self,tensor):self.tensor=tensor

def migrate(obj,direction,observe):
 """In-place graph migration; preserve each entire storage and all its views."""
 seen={};bases={}
 def convert(x):
  key=id(x)
  if key in seen:return seen[key]
  if direction=='cuda' and isinstance(x,MovedTensor):
   old=x.tensor;st=old.untyped_storage();k=(str(old.device),st.data_ptr())
   if k not in bases:
    byte=torch.empty(0,dtype=torch.uint8).set_(st,0,(st.nbytes(),),(1,));bases[k]=byte.cuda();observe([old,bases])
   base=bases[k];new=torch.empty(0,dtype=old.dtype,device='cuda').set_(base.untyped_storage(),old.storage_offset(),old.size(),old.stride());seen[key]=new;return new
  if isinstance(x,torch.Tensor):
   if direction!='cpu' or x.device.type!='cuda':return x
   st=x.untyped_storage();k=(str(x.device),st.data_ptr())
   if k not in bases:
    byte=torch.empty(0,dtype=torch.uint8,device=x.device).set_(st,0,(st.nbytes(),),(1,));bases[k]=byte.cpu();observe([x,bases])
   base=bases[k];new=torch.empty(0,dtype=x.dtype).set_(base.untyped_storage(),x.storage_offset(),x.size(),x.stride());result=MovedTensor(new);seen[key]=result;return result
  seen[key]=x
  if isinstance(x,dict):
   for k in list(x):x[k]=convert(x[k])
  elif isinstance(x,list):
   for i in range(len(x)):x[i]=convert(x[i])
  elif isinstance(x,tuple):
   seen[key]=tuple(convert(v) for v in x);return seen[key]
  elif not isinstance(x,(torch.nn.Module,type)) and hasattr(x,'__dict__'):
   for k,v in list(vars(x).items()):setattr(x,k,convert(v))
  return x
 return convert(obj)

def state_sample(manager):
 x=manager._vstore._stac_merger.export_state() if hasattr(manager,'_vstore') else vars(manager)
 out={}
 for k,v in x.items():
  if isinstance(v,torch.Tensor) and v.numel():
   flat=v.contiguous().view(torch.uint8).flatten();idx=torch.tensor([0,len(flat)//2,len(flat)-1],device=v.device);out[k]=dict(shape=list(v.shape),dtype=str(v.dtype),sample_bytes=flat[idx].cpu().tolist())
 return out

def checkpoint(ctx,meter,model,next_frame,name,reason):
 meter.detach();folder=d.DATA/'resume'/f"{ctx['arm']}_{ctx['rid']}";folder.mkdir(parents=True,exist_ok=True);path=folder/'state.pt';assert not path.exists()
 ctx.update(next_frame=next_frame,CPU_RNG=torch.get_rng_state(),CUDA_RNG=torch.cuda.get_rng_state(),samples=state_sample(ctx['manager']))
 cpp=mem.snapshot_cpp(ctx['manager']);del cpp
 def observe(extra):
  measured=mem.storage([ctx,extra]);value=measured['total_bytes']+d.plan()['compressed_readable_all12_RGB_bytes']+2**20;meter.peak=max(meter.peak,value)
 migrate(ctx,'cpu',observe);sync();model.aggregator.kv_manager=None;gc.collect();torch.cuda.empty_cache()
 torch.save(ctx,path.with_suffix('.tmp'));path.with_suffix('.tmp').replace(path)
 value=mem.storage(ctx)['total_bytes']+path.stat().st_size+d.plan()['compressed_readable_all12_RGB_bytes']+2**20;meter.peak=max(meter.peak,value)
 record=dict(complete=True,next_frame=next_frame,reason=reason,path=str(path),bytes=path.stat().st_size,history_peak_bytes=meter.peak,consumed_before_next_forward=True,configuration_sha256=d.sha(d.OUT/'configuration.json'))
 d.write(folder/'manifest.json',record);d.write(d.OUT/'checkpoints'/f'{name}.json',record)
 if meter.peak>meter.limit:raise RuntimeError('HARD_HISTORY_BYTES_EXCEEDED_IN_RECOVERY:'+str(meter.peak))

def new_context(model,arm,rid,meta,capacity):
 from stream_session import StreamSession
 from stac.stac_voxel import STACVoxelKV
 from stac.h2o import HeavyHittersKV
 from stac.kv_manager import KVManager
 images=torch.stack([rgb(f) for f in meta['frames']]).cuda();session=StreamSession(model,device='cuda');model.aggregator.clear_kv_mgr()
 kw=dict(dtype=torch.bfloat16,device='cuda',chunk_size=4,recent_size=4,pinned_idx=[0],buffer_size=11,temperature=.9,attn_backend='cuda',subsample_ratio=1.)
 if arm=='S':
  cls=STACVoxelKV;kw.update(hh_size=2,voxel_size=.05,init_voxels=4096,voxel_buf_cap=8,voxel_piv_cap=4,voxel_backend='cuda',sim_threshold=.8,replace_threshold=.8,score_threshold=.2,slab_growth=1024,slab_cap=10000,seg_size=1,retrieval_size=2,allocator='segment',enable_alloc_cpu=False)
 elif arm=='H':cls=HeavyHittersKV;kw.update(hh_size=capacity,buffer_size=min(96,4+1+4+capacity))
 else:cls=WindowKV;kw.update(recent_size=capacity,buffer_size=min(96,4+1+capacity))
 kw['token_per_frame']=1041;model.aggregator.register_kv_mgr(cls,**kw)
 model.set_camhead(False);model.enable_point=arm=='S'
 return dict(arm=arm,rid=rid,manager=model.aggregator.kv_manager,images=images,pose_tokens=[],predictions={k:[] for k in session.predictions_keys},prefix_poses=[],chunks=[],next_frame=0)

@torch.inference_mode()
def main():
 global ACTIVE_METER
 parser=argparse.ArgumentParser();parser.add_argument('--arm',required=True);parser.add_argument('--rid',required=True);parser.add_argument('--name',required=True);parser.add_argument('--phase',default='formal');a=parser.parse_args();signal.signal(signal.SIGTERM,stop);started=time.monotonic();torch.set_num_threads(2);torch.manual_seed(20261115);torch.backends.cuda.matmul.allow_tf32=False;torch.cuda.set_per_process_memory_fraction(40*2**30/torch.cuda.get_device_properties(0).total_memory)
 p=d.plan();meta=next(m for m in p['catalog'] if m['id']==a.rid);capacity=0 if a.arm=='S' else d.read(d.OUT/'capacity_lock.json')['capacity'][a.arm]
 model,loadrec=load_author();model.cuda();model.enable_point=a.arm=='S';sync();d.journal(a.name,kind='model_loaded',PID=os.getpid(),seconds=time.monotonic()-started,**loadrec)
 from stream_session import StreamSession
 folder=d.DATA/'resume'/f'{a.arm}_{a.rid}';resume=folder/'state.pt'
 if resume.exists():
  ctx=torch.load(resume,weights_only=False);saved=d.read(folder/'manifest.json');assert saved['configuration_sha256']==d.sha(d.OUT/'configuration.json')
  peak=saved['history_peak_bytes'];disk=resume.stat().st_size
  def observe(extra):
   nonlocal peak
   peak=max(peak,mem.storage([ctx,extra])['total_bytes']+disk+p['compressed_readable_all12_RGB_bytes']+2**20)
  migrate(ctx,'cuda',observe);mem.restore_cpp(ctx['manager']);model.aggregator.kv_manager=ctx['manager'];[b.attn.set_layer_idx(i) for i,b in enumerate(model.aggregator.global_blocks)];assert all(b.attn.layer_idx==i for i,b in enumerate(model.aggregator.global_blocks));model.set_camhead(False);model.enable_point=a.arm=='S';assert state_sample(ctx['manager'])==ctx['samples'];torch.set_rng_state(ctx.pop('CPU_RNG'));torch.cuda.set_rng_state(ctx.pop('CUDA_RNG'));del ctx['samples'];ctx['restore_backing']=str(resume);d.journal(a.name,kind='exact_restore',next_frame=ctx['next_frame'],history_peak_bytes=peak,no_completed_chunks_repeated=True)
  if peak>16*2**30:raise RuntimeError('HARD_HISTORY_BYTES_EXCEEDED_ON_RESTORE:'+str(peak))
 else:
  # First fixed author's single-frame warmup is already counted by loadcheck.
  if not (a.arm=='S' and a.rid==p['catalog'][0]['id']):
   im=rgb(meta['frames'][0])[None,None].cuda()
   with torch.autocast('cuda',dtype=torch.bfloat16):warm=model(im,mode='full',streaming=True,is_anchor_exist=True)
   del warm,im;sync();d.journal(a.name,kind='warmup',author_model_forward_calls=1)
  ctx=new_context(model,a.arm,a.rid,meta,capacity);peak=0
 session=StreamSession(model,device='cuda');session.pose_tokens_list=ctx['pose_tokens'];session.predictions=ctx['predictions'];session._processed_frames=ctx['next_frame']
 extra={};aux=dict(position=model.aggregator.position_getter.position_cache,rope=vars(model.aggregator.rope))
 meter=mem.HistoryMeter(lambda:dict(context=ctx,aux=aux,extra=extra),a.name,p['compressed_readable_all12_RGB_bytes']+2**20+(Path(ctx['restore_backing']).stat().st_size if 'restore_backing' in ctx else 0));meter.peak=peak;meter.limit=(d.read(d.OUT/'capacity_lock.json')['B_bytes'] if a.arm!='S' else 16*2**30);ACTIVE_METER=meter;mem.instrument(meter,ctx['manager']);meter.wrap(session,'get_pointmap');meter.record('initial_state')
 for key,mod in [('aggregator',model.aggregator),('camera',model.camera_head),('depth',model.depth_head),('point',model.point_head)]:
  mod.register_forward_pre_hook(lambda m,x,k=key:d.journal(a.name,kind='module_begin',module=k))
  mod.register_forward_hook(lambda m,x,y,k=key:d.journal(a.name,kind='module_complete',module=k))
 for t in range(ctx['next_frame'],96,4):
  begin=time.monotonic();chunk=ctx['images'][t:t+4];s=time.monotonic()
  with torch.autocast('cuda',dtype=torch.bfloat16):outputs=model(chunk,mode='full',camera_head_kv_cache_list=None,streaming=True,is_anchor_exist=t==0,timing=True)
  sync();backbone_heads=time.monotonic()-s;s=time.monotonic();cam=session.camera_head_inference(outputs['aggregated_tokens_list']);sync();camera_seconds=time.monotonic()-s
  ctx['predictions']=session.predictions;ctx['prefix_poses'].append(cam['pose_enc'].cpu());meter.record('prefix_camera_archive')
  position_time=retrieval_time=prune_time=0.
  if a.arm=='S':
   s=time.monotonic();xyz,valid=session.get_pointmap(outputs,conf_threshold=2.,special_tokens_size=5,pose_enc=cam['pose_enc'],images=chunk);extra['geometry']=(xyz,valid);model.aggregator.update_kv_mgr_pos(xyz,valid,timing=False);sync();position_time=time.monotonic()-s
   if t>max(11,16):s=time.monotonic();model.aggregator.retrieve_kv_mgr(timing=False,verbose=False,dist_thres=.1,return_buf=True);sync();retrieval_time=time.monotonic()-s
  s=time.monotonic();model.aggregator.prune_kv_mgr(timing=False);sync();prune_time=time.monotonic()-s
  # Author retains all policy-used outputs on CPU; H/W omit the unused point head.
  for key in session.predictions_keys:
   if key in outputs and isinstance(outputs[key],torch.Tensor):
    for j in range(4):session.predictions[key].append(outputs[key][:,j:j+1].cpu())
  del outputs,cam,chunk
  if a.arm=='S':del xyz,valid
  extra.clear();ctx['next_frame']=t+4;session._processed_frames=t+4;sync();record=meter.record('chunk_commit');cost=dict(frame_start=t,submitted_prefix=t+4,backbone_and_heads_seconds=backbone_heads,camera_prefix_seconds=camera_seconds,position_seconds=position_time,retrieval_seconds=retrieval_time,prune_merge_seconds=prune_time,chunk_wall_seconds=time.monotonic()-begin,history_peak_bytes=meter.peak,current_history=record['current']);ctx['chunks'].append(cost)
  if 'restore_backing' in ctx:
   backing=Path(ctx.pop('restore_backing'));meter.common-=backing.stat().st_size;backing.unlink();backing.with_name('manifest.json').unlink();backing.parent.rmdir();d.write(d.OUT/'recovery_validation.json',dict(checkpoint_complete=True,restore_complete=True,first_new_submitted_prefix=t+4,completed_chunks_repeated=0,full_tensor_samples_equal=True,layer_indices_all24=True));d.journal(a.name,kind='recovery_consumed_after_new_commit',prefix=t+4)
  d.journal(a.name,kind='chunk_commit',arm=a.arm,id=a.rid,author_model_forward_calls=1,camera_prefix_calls=1,camera_iterations=4,**cost);d.write(d.OUT/'progress'/f'{a.arm}_{a.rid}.json',dict(PID=os.getpid(),**cost));print(dict(PID=os.getpid(),arm=a.arm,id=a.rid,prefix=t+4,history_GiB=meter.peak/2**30,seconds=cost['chunk_wall_seconds']),flush=True)
  if STOP or (a.arm=='S' and a.rid==p['catalog'][0]['id'] and t==4 and not (d.OUT/'recovery_validation.json').exists()):
   reason='external_or_requested_yield' if STOP else 'necessary_exact_restore_validation';checkpoint(ctx,meter,model,t+4,a.name,reason)
   if not STOP:d.write(d.OUT/'recovery_validation.json',dict(checkpoint_complete=True,next_frame=8,validated_on_next_restore=True))
   d.write(d.OUT/'resources'/f'{a.name}.json',dict(actual_seconds=time.monotonic()-started,peak_history_bytes=meter.peak,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,stop_reason=reason));raise SystemExit(75)
 # No token/voxel state is retained after this stream's permanent prediction commit.
 meter.detach();plain=dict(ctx);plain['manager']=None;plain['predictions']=dict(ctx['predictions']);plain['predictions']['world_points']=[];plain['predictions']['world_points_conf']=[];d.write(d.OUT/'inventory'/f'{a.arm}_{a.rid}.json',dict(nonspatial_nonmanager=mem.storage(plain),pose_view_dtype=str(ctx['pose_tokens'][0].dtype),pose_view_shape=list(ctx['pose_tokens'][0].shape),pose_view_backing_bytes=ctx['pose_tokens'][0].untyped_storage().nbytes(),manager=mem.storage(ctx['manager']),author_overflow_dropped_count=int(ctx['manager']._vstore._stac_merger.dropped_overflow_count) if a.arm=='S' else None,all_readable_RGB_bytes=p['compressed_readable_all12_RGB_bytes']));del plain;result=dict(arm=a.arm,id=a.rid,prefix_poses=ctx['prefix_poses'],depth=torch.cat(ctx['predictions']['depth'],1)[0,...,0],author_retained_outputs=ctx['predictions'],chunks=ctx['chunks'],configuration_sha256=d.sha(d.OUT/'configuration.json'));extra['permanent_output']=result;meter.record('final_CPU_stack');path=d.DATA/'predictions'/f'{a.arm}_{a.rid}.pt';path.parent.mkdir(exist_ok=True);torch.save(result,path.with_suffix('.tmp'));path.with_suffix('.tmp').replace(path);peak=max(meter.peak,meter.current()['total_history_bytes']+path.stat().st_size)
 if peak>meter.limit:raise RuntimeError('HARD_HISTORY_BYTES_EXCEEDED_AT_FINAL:'+str(peak))
 d.write(d.OUT/'completed'/f'{a.arm}_{a.rid}.json',dict(complete=True,arm=a.arm,id=a.rid,frames=96,chunks=24,prediction_path=str(path),bytes=path.stat().st_size,peak_history_bytes=peak,GT_scored=False))
 d.write(d.OUT/'resources'/f'{a.name}.json',dict(actual_seconds=time.monotonic()-started,peak_history_bytes=peak,history_scope_records=meter.records,peak_GPU_allocated_bytes=max(meter.peak_model_VRAM,torch.cuda.max_memory_allocated()),peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024));print(dict(complete=True,arm=a.arm,id=a.rid),flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as exc:
  if not isinstance(exc,SystemExit) or exc.code not in (0,75):
   d.write(d.OUT/'last_worker_failure.json',dict(PID=os.getpid(),exception_type=type(exc).__name__,exception=str(exc),utc=d.now(),peak_history_bytes=ACTIVE_METER.peak if ACTIVE_METER else None,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024))
  raise
