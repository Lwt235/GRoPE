"""Frozen effective-key C/I/G native streams; only source allocation differs."""
import os,time,signal,argparse,types,resource,shutil
import torch
from experiments import stream_merge_data as d
from experiments import stream_merge_algorithm as alg
from experiments.stream_history_run import cuda,prune
from experiments.stream_retention_run import tensor_storage,samples
STOP=False;CTX=None
def stop(*_):
 global STOP;STOP=True
def sync():torch.cuda.synchronize()
def recent(t,n):return [0]+list(range(max(1,t-n+1),t+1))
def forward(self,x,pos=None,attn_mask=None,past_key_values=None,use_cache=False):
 B,N,C=x.shape;qkv=self.qkv(x).reshape(B,N,3,self.num_heads,self.head_dim).permute(2,0,3,1,4);q,k,v=qkv.unbind(0);q,k=self.q_norm(q),self.k_norm(k)
 if self.rope is not None:q=self.rope(q,pos);k=self.rope(k,pos)
 if CTX['t']==0:
  CTX['state']['scale'][self._merge_layer,0]=k[0,:,5:].float().square().mean((1,2)).sqrt().clamp_min(torch.finfo(torch.float32).tiny);CTX['state']['scale'][self._merge_layer,1]=v[0,:,5:].float().square().mean((1,2)).sqrt().clamp_min(torch.finfo(torch.float32).tiny)
 if past_key_values is not None:k=torch.cat((past_key_values[0].flatten(2,3),k),2);v=torch.cat((past_key_values[1].flatten(2,3),v),2)
 newkv=(k.unsqueeze(2),v.unsqueeze(2));means=CTX['state']['means'];mask=None
 if len(means):
  mk=means[:,self._merge_layer,0].permute(1,0,2).to(k.dtype)[None];mv=means[:,self._merge_layer,1].permute(1,0,2).to(v.dtype)[None];k=torch.cat((k,mk),2);v=torch.cat((v,mv),2);mask=torch.cat((torch.zeros(k.shape[2]-len(means),device='cuda'),CTX['state']['n'].float().log()))[None,None,None]
 y=torch.nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=mask,dropout_p=0.)
 if CTX['t']==0 and self._merge_layer==5 and CTX['profiling']:
  d.save(d.DATA/'validation'/f"{CTX['policy']}_first_attention.pt",dict(q=q[0,:,:6].cpu(),k=k[0].cpu(),v=v[0].cpu(),native_y=y[0,:,:6].cpu(),weight=self.proj.weight.float().cpu(),all16heads=True,source_effective=True))
 return self.proj_drop(self.proj(y.transpose(1,2).reshape(B,N,C))),newkv

def empty():return dict(kv=[None]*24,cam=[None]*4,camids=[],ids=torch.empty(0,dtype=torch.int64,device='cuda'),geo_mu=torch.empty((0,3),dtype=torch.float64,device='cuda'),geo_cov=torch.empty((0,3,3),dtype=torch.float64,device='cuda'),geo_valid=torch.empty(0,dtype=torch.bool,device='cuda'),means=torch.empty((0,24,2,16,64),device='cuda'),n=torch.empty(0,dtype=torch.int64,device='cuda'),m=torch.empty(0,dtype=torch.int64,device='cuda'),mu=torch.empty((0,3),dtype=torch.float64,device='cuda'),cov=torch.empty((0,3,3),dtype=torch.float64,device='cuda'),remote_ids=torch.empty(0,dtype=torch.int64,device='cuda'),scale=torch.ones((24,2,16),device='cuda'),delta=None,arena=torch.zeros(d.ARENA,device='cuda',dtype=torch.uint8))
def restore(policy,rid,name):
 path=d.DATA/'resume'/f'{policy}_{rid}'
 if not (path/'manifest.json').exists():return empty(),dict(depth=[],poses=[],frame_costs=[],merge_records=[]),0
 s=time.monotonic();rec=d.read(path/'manifest.json');assert rec['configuration_sha256']==d.sha(d.OUT/'configuration.json');state=cuda(d.load(path/'state.pt'));pred=d.load(path/'outputs.pt');assert samples(state)==rec['samples'];assert len(pred['depth'])==rec['next_frame'];assert state['remote_ids'].cpu().tolist()==rec['remote_IDs'];assert state['n'].cpu().tolist()==rec['n'];assert state['m'].cpu().tolist()==rec['m'];torch.set_rng_state(torch.tensor(rec['CPU_RNG'],dtype=torch.uint8));torch.cuda.set_rng_state(torch.tensor(rec['CUDA_RNG'],dtype=torch.uint8));shutil.rmtree(path);d.journal(name,kind='exact_restore',policy=policy,id=rid,next_frame=rec['next_frame'],seconds=time.monotonic()-s,completed_queries_repeated=0,consumed_backing_deleted=True);return state,pred,rec['next_frame']
def checkpoint(policy,rid,state,pred,t,name):
 s=time.monotonic();path=d.DATA/'resume'/f'{policy}_{rid}';assert not (path/'state.pt').exists();d.save(path/'state.pt',state);d.save(path/'outputs.pt',pred);rec=dict(policy=policy,id=rid,next_frame=t+1,state_bytes=(path/'state.pt').stat().st_size,outputs_bytes=(path/'outputs.pt').stat().st_size,samples=samples(state),remote_IDs=state['remote_ids'].cpu().tolist(),n=state['n'].cpu().tolist(),m=state['m'].cpu().tolist(),CPU_RNG=torch.get_rng_state().tolist(),CUDA_RNG=torch.cuda.get_rng_state().cpu().tolist(),configuration_sha256=d.sha(d.OUT/'configuration.json'),seconds=time.monotonic()-s);d.write(path/'manifest.json',rec);d.write(d.OUT/'checkpoint_records'/f'{policy}_{rid}_{t:02d}.json',rec);d.journal(name,kind='exact_exit_checkpoint',**rec)

def toy(name):
 x=torch.tensor([[0.,0,0],[1,0,0],[0,0,1]],device='cuda',dtype=torch.float64);cov=torch.diag(torch.tensor([1.,1.,.01],device='cuda',dtype=torch.float64))[None].repeat(3,1,1);m=torch.ones(3,device='cuda',dtype=torch.int64);i=torch.tensor([0,0],device='cuda');j=torch.tensor([1,2],device='cuda');a=alg.gscore('G',x,cov,m,i,j,.001);b=alg.gscore('I',x,cov,m,i,j,.001);assert a[0]<a[1] and abs(float(b[0]-b[1]))<1e-12;d.write(d.OUT/'toy_validation.json',dict(complete=True,shape_direction_G=a.cpu().tolist(),same_trace_I=b.cpu().tolist(),equal_content_equal_distance=True,no_model_calls=True));d.journal(name,kind='necessary_GPU_support_counterfactual',native_model_calls=0)

@torch.inference_mode()
def stream(model,policy,rid,name,profiling):
 global CTX
 state,pred,start=restore(policy,rid,name);meta=next(m for m in d.plan()['catalog'] if m['id']==rid);CTX=dict(t=start,state=state,policy=policy,profiling=profiling);eq=dict(depth_max_abs=0.,pose_max_abs=0.);old=d.load(d.OLDDATA/'predictions'/f'{rid}.pt') if profiling else None
 if policy!='W':
  for layer,block in enumerate(model.aggregator.global_blocks):block.attn._merge_layer=layer;block.attn.forward=types.MethodType(forward,block.attn)
 for t in range(start,96):
  CTX['t']=t;begin=time.monotonic();s=time.monotonic();image=d.rgb(meta['frames'][t])[None,None].cuda();sync();times=dict(RGB_read_transfer_seconds=time.monotonic()-s);s=time.monotonic()
  with torch.autocast('cuda',dtype=torch.bfloat16):tokens,patch,raw=model.aggregator(image,past_key_values=state['kv'],use_cache=True,past_frame_idx=t)
  state['kv']=[None]*24;sync();times['native_aggregator_seconds']=time.monotonic()-s;s=time.monotonic()
  with torch.autocast('cuda',enabled=False):pose,cam=model.camera_head(tokens,past_key_values_camera=state['cam'],use_cache=True);pose=pose[-1][0,0].float();depth,_=model.depth_head(tokens,images=image,patch_start_idx=patch);depth=depth[0,0,...,0].float()
  state['cam']=[None]*4;state['cam'],state['camids']=prune(cam,state['camids']+[t]*4,recent(t,7));sync();times['native_camera_depth_seconds']=time.monotonic()-s;s=time.monotonic();mergeinfo=dict(merges=0,seconds=0.,slots=0,mass=0)
  if policy=='W':
   state['kv'],state['W_ids']=prune(raw,state.get('W_ids',[])+[t],recent(t,7));times['source_geometry_seconds']=0.
  else:
   mu,cov,m,delta=alg.geometry(depth,pose);fullmu=torch.zeros((1041,3),device='cuda',dtype=torch.float64);fullcov=torch.zeros((1041,3,3),device='cuda',dtype=torch.float64);fullvalid=torch.zeros(1041,device='cuda',dtype=torch.bool);fullmu[5:]=mu;fullcov[5:]=cov;fullvalid[5:]=m.bool();state['geo_mu']=torch.cat((state['geo_mu'],fullmu));state['geo_cov']=torch.cat((state['geo_cov'],fullcov));state['geo_valid']=torch.cat((state['geo_valid'],fullvalid));state['ids']=torch.cat((state['ids'],torch.arange(t*1041,(t+1)*1041,device='cuda')))
   if state['delta'] is None and delta is not None:state['delta']=delta
   sync();times['source_geometry_seconds']=time.monotonic()-s;ids=state['ids'];frames=ids//1041;ispatch=ids.remainder(1041)>=5;near=torch.isin(frames,torch.tensor(recent(t,3),device='cuda'));special=~ispatch&torch.isin(frames,torch.tensor(recent(t,7),device='cuda'));evict=(ispatch&~near).nonzero().flatten();keep=((ispatch&near)|special).nonzero().flatten()
   if len(evict):
    incoming=torch.stack([torch.stack([x[0,:,0].index_select(1,evict).permute(1,0,2) for x in pair],1) for pair in raw],1);mergeinfo=alg.merge(state,incoming,(state['geo_mu'][evict],state['geo_cov'][evict],state['geo_valid'][evict].long()),ids[evict],policy,d.plan()['M']);del incoming
   assert int(state['n'].sum())==max(0,t-3)*1036;assert len(state['n'])<=d.plan()['M'];state['kv']=[tuple(x.index_select(3,keep) for x in pair) for pair in raw];state['ids']=ids[keep];state['geo_mu']=state['geo_mu'][keep];state['geo_cov']=state['geo_cov'][keep];state['geo_valid']=state['geo_valid'][keep];pred['merge_records'].append(dict(frame=t,**mergeinfo));del mu,cov,m,fullmu,fullcov,fullvalid,ids,frames,ispatch,near,special,evict,keep
  del raw,cam,tokens,image;sync();times['full_update_seconds']=time.monotonic()-s;dep=depth.cpu();pos=pose.cpu();pred['depth'].append(dep);pred['poses'].append(pos);del depth,pose
  if profiling and (policy=='W' or t<=4):eq['depth_max_abs']=max(eq['depth_max_abs'],float((dep-old['depth_W'][t]).abs().max()));eq['pose_max_abs']=max(eq['pose_max_abs'],float((pos-old['poses']['W'][t]).abs().max()))
  mem=tensor_storage(state);assert policy=='W' or mem['total_bytes']<=d.plan()['state_copy_bound_bytes'];times.update(frame=t,wall_seconds=time.monotonic()-begin,state_storage=mem,remote_slots=len(state['n']),remote_mass=int(state['n'].sum()),merges=mergeinfo['merges'],merge_seconds=mergeinfo['seconds'],invalid_geometry_source_patches=int((~state['geo_valid'][state['ids'].remainder(1041)>=5]).sum()) if policy!='W' else 0);pred['frame_costs'].append(times);d.journal(name,kind='native_frame',policy=policy,id=rid,aggregator_calls=1,camera_head_calls=1,camera_iterations=4,depth_head_calls=1,**times);d.write(d.OUT/'progress'/f'{policy}_{rid}.json',dict(PID=os.getpid(),policy=policy,id=rid,**times));print(dict(policy=policy,id=rid,frame=t,seconds=times['wall_seconds'],merges=times['merges']),flush=True)
  if t==0 and policy!='W':d.save(d.OUT/'initials'/f'{policy}_{rid}.pt',dict(scale=state['scale'].cpu(),delta=state['delta']))
  torch.cuda.empty_cache()
  if STOP:break
 if t==95:
  path=d.DATA/'predictions'/f'{policy}_{rid}.pt';d.save(path,dict(policy=policy,id=rid,depth=torch.stack(pred['depth']),poses=torch.stack(pred['poses']),frame_costs=pred['frame_costs'],merge_records=pred['merge_records'],final_n=state['n'].cpu(),final_m=state['m'].cpu(),final_remote_IDs=state['remote_ids'].cpu(),final_mu=state['mu'].cpu(),final_cov=state['cov'].cpu(),scale=state['scale'].cpu(),delta=state['delta'],configuration_sha256=d.sha(d.OUT/'configuration.json')));d.write(d.OUT/'completed'/f'{policy}_{rid}.json',dict(complete=True,policy=policy,id=rid,frames=96,path=str(path),bytes=path.stat().st_size,sha256=d.sha(path),profiling=profiling,no_merge_or_W_native_replay=eq if profiling else None))
 else:checkpoint(policy,rid,state,pred,t,name)
 return t

def main():
 p=argparse.ArgumentParser();p.add_argument('--policy',required=True);p.add_argument('--rid',required=True);p.add_argument('--name',required=True);p.add_argument('--phase',choices=['profile','formal'],required=True);a=p.parse_args();signal.signal(signal.SIGTERM,stop);start=time.monotonic();torch.set_num_threads(2);torch.manual_seed(20261114);torch.backends.cuda.matmul.allow_tf32=False;assert torch.cuda.is_available();torch.cuda.set_per_process_memory_fraction(39*2**30/torch.cuda.get_device_properties(0).total_memory)
 from experiments.gaussian_memory_native_run import model_load
 model=model_load(d.plan());sync();d.journal(a.name,kind='model_loaded',seconds=time.monotonic()-start,CUDA=True,frozen=not any(p.requires_grad for p in model.parameters()))
 if a.phase=='profile' and a.policy=='C' and not (d.OUT/'toy_validation.json').exists():toy(a.name)
 for key in ('aggregator','camera_head','depth_head'):
  mod=getattr(model,key);mod.register_forward_pre_hook(lambda m,x,k=key:d.journal(a.name,kind='module_begin',module=k));mod.register_forward_hook(lambda m,x,y,k=key:d.journal(a.name,kind='module_complete',module=k))
 last=stream(model,a.policy,a.rid,a.name,a.phase=='profile');io={k:int(v) for k,v in (line.split(':') for line in open('/proc/self/io'))};d.write(d.OUT/'worker_resources'/f'{a.name}.json',dict(PID=os.getpid(),last_frame=last,body_seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),peak_GPU_reserved_bytes=torch.cuda.max_memory_reserved(),process_IO_counters=io));print(dict(complete=last==95),flush=True)
 if last!=95:raise SystemExit(75)
if __name__=='__main__':main()
