"""Real native four-cell counterfactuals, memory-mapped CPU raw-KV staging."""
import os,time,signal,argparse,gc,resource,shutil
import torch
from experiments import stream_factorization_data as d
from experiments.stream_history_run import cpu,cuda,prune,keep_ids
from experiments.stream_retention_run import tensor_storage
STOP=False
def stop(*_):
 global STOP;STOP=True
def sync():torch.cuda.synchronize()
def sig(pairs):
 return [[dict(shape=list(x.shape),dtype=str(x.dtype),sample=x.reshape(-1)[[0,x.numel()//3,x.numel()-1]].float().cpu().tolist()) for x in p] if p else None for p in pairs]
class Staging:
 def __init__(self,model,folder,gpu_banks=False):
  self.folder=folder;folder.mkdir(parents=True,exist_ok=True);self.banks={};self.gpu_banks={};self.use_gpu_banks=gpu_banks;self.handles=[];self.ctx={};self.totals=dict(H2D_bytes=0,D2H_bytes=0,H2D_seconds=0.,D2H_seconds=0.,bank_write_seconds=0.,D2D_bank_write_bytes=0,D2D_bank_write_seconds=0.)
  for i,block in enumerate(model.aggregator.global_blocks):
   self.handles.append(block.attn.register_forward_pre_hook(self.pre,with_kwargs=True))
   self.handles.append(block.attn.register_forward_hook(lambda m,a,k,o,j=i:self.post(j,o),with_kwargs=True))
 def allocate(self,layer,pair):
  if layer in self.banks:return
  shape=list(pair[0].shape);shape[2]=96;n=1
  for x in shape:n*=x
  banks={}
  for source in ('F','W'):
   banks[source]=[]
   for j in range(2):
    path=self.folder/f'{source}_{layer:02d}_{j}.bin';need=n*pair[j].element_size()
    if not path.exists():
     with path.open('wb') as f:f.truncate(need)
    assert path.stat().st_size==need
    banks[source].append(torch.from_file(str(path),shared=True,size=n,dtype=pair[j].dtype).reshape(shape))
  self.banks[layer]=banks
  if self.use_gpu_banks:
   begin=time.monotonic();self.gpu_banks[layer]={source:[torch.empty(shape,device='cuda',dtype=pair[j].dtype) for j in range(2)] for source in ('F','W')};t=self.ctx.get('t',0)
   if t:
    for source in ('F','W'):
     for j in range(2):
      self.gpu_banks[layer][source][j][:,:,:t].copy_(banks[source][j][:,:,:t]);self.totals['H2D_bytes']+=banks[source][j][:,:,:t].numel()*banks[source][j].element_size()
    sync();self.totals['H2D_seconds']+=time.monotonic()-begin
 def pre(self,module,args,kwargs):
  p=kwargs.get('past_key_values')
  if p is not None and any(x.device.type!='cuda' for x in p):
   begin=time.monotonic();kwargs=dict(kwargs);kwargs['past_key_values']=tuple(x.cuda() for x in p);sync();self.totals['H2D_seconds']+=time.monotonic()-begin;self.totals['H2D_bytes']+=sum(x.numel()*x.element_size() for x in p)
  return args,kwargs
 def post(self,i,out):
  tokens,pair=out;cell=self.ctx['cell'];t=self.ctx['t'];self.allocate(i,pair)
  if cell in ('WW','FF'):
   begin=time.monotonic();new=tuple(x[:,:, -1:].cpu() for x in pair);sync();self.totals['D2H_seconds']+=time.monotonic()-begin;self.totals['D2H_bytes']+=sum(x.numel()*x.element_size() for x in new);begin=time.monotonic()
   source='W' if cell=='WW' else 'F'
   for j in range(2):self.banks[i][source][j][:,:,t:t+1].copy_(new[j])
   if t<9 and cell=='WW':
    for j in range(2):self.banks[i]['F'][j][:,:,t:t+1].copy_(new[j])
   self.totals['bank_write_seconds']+=time.monotonic()-begin
   if self.use_gpu_banks:
    begin=time.monotonic();sources=(source,'F') if t<9 and cell=='WW' else (source,)
    for target in sources:
     for j in range(2):
      raw=pair[j][:,:,-1:];self.gpu_banks[i][target][j][:,:,t:t+1].copy_(raw);self.totals['D2D_bank_write_bytes']+=raw.numel()*raw.element_size()
    sync();self.totals['D2D_bank_write_seconds']+=time.monotonic()-begin
   if cell=='FF':return tokens,tuple(x[:,:,:t+1] for x in self.banks[i]['F'])
   begin=time.monotonic();bounded=tuple(x.cpu() for x in pair);sync();self.totals['D2H_seconds']+=time.monotonic()-begin;self.totals['D2H_bytes']+=sum(x.numel()*x.element_size() for x in pair)
   return tokens,bounded
  return tokens,None
 def history(self,source,t,ids=None):
  if t==0:return [None]*24
  if not self.banks:
   prototype=(torch.empty((1,16,1,1041,64),dtype=torch.bfloat16),)*2
   for i in range(24):self.allocate(i,prototype)
  banks=self.gpu_banks if self.use_gpu_banks else self.banks
  return [tuple(x[:,:,:t] if ids is None else x.index_select(2,torch.tensor(ids,device=x.device)) for x in banks[i][source]) for i in range(24)]
 def close(self):
  for h in self.handles:h.remove()
def flush(folder):
 begin=time.monotonic()
 for p in folder.glob('*.bin'):
  fd=os.open(p,os.O_RDWR);os.fsync(fd);os.close(fd)
 return time.monotonic()-begin
@torch.inference_mode()
def main():
 p=argparse.ArgumentParser();p.add_argument('--rid',required=True);p.add_argument('--name',required=True);p.add_argument('--profile',action='store_true');a=p.parse_args();signal.signal(signal.SIGTERM,stop);begin=time.monotonic();torch.set_num_threads(2);torch.manual_seed(20261114);assert torch.cuda.is_available();torch.cuda.set_per_process_memory_fraction(39*2**30/torch.cuda.get_device_properties(0).total_memory)
 from experiments.gaussian_memory_native_run import model_load
 model=model_load(d.plan());sync();d.journal(a.name,kind='model_loaded',seconds=time.monotonic()-begin,CUDA=True,frozen=not any(x.requires_grad for x in model.parameters()))
 mode_path=d.OUT/'engineering_revisions/gpu_bank006.json';use_gpu_banks=mode_path.exists() and a.rid in d.read(mode_path)['remaining_streams'];folder=d.DATA/'resume'/a.rid;stage=Staging(model,folder/'banks',gpu_banks=use_gpu_banks);meta=next(m for m in d.plan()['catalog'] if m['id']==a.rid);pred={k:dict(depth=[],poses=[]) for k in ('WW','WF','FW','FF')};costs=[];window=[None]*24;cam=[None]*4;wid=[];cid=[];first=0
 d.journal(a.name,kind='cache_tier',gpu_source_bank_mirrors=use_gpu_banks,extra_GPU_raw_capacity_bytes=19648217088 if use_gpu_banks else 0,native_numeric_function_unchanged=True)
 if (folder/'manifest.json').exists():
  rec=d.read(folder/'manifest.json');assert rec['configuration_sha256']==d.sha(d.OUT/'configuration.json');v=d.load(folder/'bounded.pt');window=v['window'];cam=cuda(v['cam']);wid=rec['wid'];cid=rec['cid'];pred=d.load(folder/'outputs.pt');costs=rec['costs'];first=rec['next_frame'];assert sig(window)==rec['window_signature'];assert [[{k:v for k,v in x.items() if k!='version'} for x in p] if p else None for p in sig(cam)]==rec['camera_signature'];torch.set_rng_state(v['rng']);torch.cuda.set_rng_state(v['cuda_rng']);(folder/'bounded.pt').unlink();(folder/'outputs.pt').unlink();(folder/'manifest.json').unlink();d.journal(a.name,kind='restore',next_frame=first,completed_frames_repeated=0)
 oldW=d.load(d.OLDDATA/'predictions'/f'{a.rid}.pt') if a.profile else None;replay={'depth_max_abs':0.,'pose_max_abs':0.};t=first-1
 for t in range(first,96):
  start=time.monotonic();x=d.rgb(meta['frames'][t])[None,None].cuda();sync();row=dict(frame=t,RGB_load_seconds=time.monotonic()-start,cells={},transfers_before=stage.totals.copy());prefix=cam;prefix_sig=sig(prefix);newcam=None;nextwindow=None;logical={}
  for cell in (('WW',) if t<9 else ('WW','FF','WF','FW')):
   if cell=='WW':history=list(window)
   elif cell=='FF':history=stage.history('F',t)
   elif cell=='WF':history=stage.history('W',t)
   else:history=stage.history('F',t,wid)
   stage.ctx=dict(cell=cell,t=t);s=time.monotonic()
   with torch.autocast('cuda',dtype=torch.bfloat16):tokens,st,returned=model.aggregator(x,past_key_values=history,use_cache=True,past_frame_idx=t)
   sync();agg=time.monotonic()-s;s=time.monotonic();work=list(prefix);assert sig(work)==prefix_sig
   with torch.autocast('cuda',enabled=False):
    poses,outcam=model.camera_head(tokens,past_key_values_camera=work,use_cache=True);depth,_=model.depth_head(tokens,images=x,patch_start_idx=st)
   pose=poses[-1][0,0].float().cpu();dep=depth[0,0,...,0].float().cpu();sync();head=time.monotonic()-s;assert sig(prefix)==prefix_sig
   pred[cell]['poses'].append(pose);pred[cell]['depth'].append(dep);row['cells'][cell]=dict(aggregator_seconds=agg,camera_depth_seconds=head,camera_prefix_tensor_identical=True,camera_iterations=4,depth_finite=bool(torch.isfinite(dep).all()),pose_finite=bool(torch.isfinite(pose).all()))
   if cell=='WW':newcam=outcam;nextwindow=returned
   del tokens,depth,poses,outcam,returned,history,work
  if t<9:
   for cell in ('FF','WF','FW'):pred[cell]['poses'].append(pose);pred[cell]['depth'].append(dep)
   row['shared_cells']=['WW','WF','FW','FF'];row['equivalence_scope']='All past histories identical before first window eviction; F reference writes exactly shared rawKV, all four cameras same actual prefix.'
  if oldW is not None:
   replay['depth_max_abs']=max(replay['depth_max_abs'],float((pred['WW']['depth'][-1]-oldW['depth_W'][t]).abs().max()));replay['pose_max_abs']=max(replay['pose_max_abs'],float((pred['WW']['poses'][-1]-oldW['poses']['W'][t]).abs().max()))
  kept=keep_ids(t);window,wid=prune(nextwindow,wid+[t],kept);cam,cid=prune(newcam,cid+[t]*4,kept);del prefix,newcam,nextwindow
  if use_gpu_banks or t in (31,63):
   cache_begin=time.monotonic();torch.cuda.empty_cache();row['unused_allocator_cache_release_seconds']=time.monotonic()-cache_begin
  row.update(wid=wid.copy(),camera_ids=cid.copy(),frame_wall_seconds=time.monotonic()-start,transfers_after=stage.totals.copy(),CPU_file_backed_bank_bytes=d.size(folder/'banks'),GPU_source_bank_capacity_bytes=19648217088 if use_gpu_banks else 0,bounded_cpu_bytes=tensor_storage(window)['unique_tensor_bytes'],common_camera_GPU_bytes=tensor_storage(cam)['unique_tensor_bytes']);costs.append(row)
  d.journal(a.name,kind='native_frame',id=a.rid,**row);d.write(d.OUT/'progress'/f'{a.rid}.json',dict(id=a.rid,PID=os.getpid(),frame=t,name=a.name,complete=False));print(dict(id=a.rid,frame=t,seconds=row['frame_wall_seconds']),flush=True)
  if STOP:break
 if use_gpu_banks:
  sync();stage.gpu_banks.clear();torch.cuda.empty_cache();d.journal(a.name,kind='GPU_source_mirrors_released_after_segment',last_frame=t,CPU_source_mirrors_complete=True,before_checkpoint_or_final_archive_IO=True)
 del oldW
 if t==95:
  if a.profile:
   old=d.load(d.OLDDATA/'predictions'/f'{a.rid}.pt');replay=dict(depth_max_abs=float((torch.stack(pred['WW']['depth'])-old['depth_W']).abs().max()),pose_max_abs=float((torch.stack(pred['WW']['poses'])-old['poses']['W']).abs().max()),FF_depth_max_abs_vs_old_F=float((torch.stack(pred['FF']['depth'])-old['depth_full']).abs().max()),all96_including_restored_prefix=True,old_pixel_metrics_recomputed=0);del old
  path=d.DATA/'predictions'/f'{a.rid}.pt';outputs={k:dict(depth=torch.stack(v['depth']),poses=torch.stack(v['poses'])) for k,v in pred.items()};d.save(path,dict(id=a.rid,cells=outputs,frame_costs=costs,configuration_sha256=d.sha(d.OUT/'configuration.json')));d.write(d.OUT/'completed'/f'{a.rid}.json',dict(complete=True,id=a.rid,path=str(path),bytes=path.stat().st_size,sha256=d.sha(path),frames=96,profile=a.profile,WW_replay=replay if a.profile else None,transfers=stage.totals));stage.close();del stage;gc.collect();shutil.rmtree(folder)
 else:
  s=time.monotonic();fs=flush(folder/'banks');d.save(folder/'bounded.pt',dict(window=window,cam=cpu(cam),rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state().cpu()));d.save(folder/'outputs.pt',pred);d.write(folder/'manifest.json',dict(id=a.rid,next_frame=t+1,wid=wid,cid=cid,costs=costs,configuration_sha256=d.sha(d.OUT/'configuration.json'),window_signature=sig(window),camera_signature=[[{k:v for k,v in x.items() if k!='version'} for x in p] if p else None for p in sig(cam)]));d.journal(a.name,kind='complete_exit_checkpoint',frame=t,seconds=time.monotonic()-s,bank_fsync_seconds=fs,bank_bytes=d.size(folder/'banks'));stage.close()
 io={k:int(v) for k,v in (line.split(':') for line in open('/proc/self/io'))}
 d.write(d.OUT/'worker_resources'/f'{a.name}.json',dict(PID=os.getpid(),id=a.rid,complete=t==95,last_frame=t,worker_body_seconds=time.monotonic()-begin,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),peak_GPU_reserved_bytes=torch.cuda.max_memory_reserved(),process_kernel_IO_counters=io,IO_scope='Wholeprocesscounters; mmap dirtypagewriteback attribution is kerneldependent, not logicalbankbytes or proof every mappedpage physicallyflushed.'))
 if t!=95:raise SystemExit(75)
if __name__=='__main__':main()
