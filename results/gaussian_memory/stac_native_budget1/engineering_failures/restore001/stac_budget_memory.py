"""Alias-aware history observation and original C++ state checkpoint adapter."""
import sys,types,time,functools
import torch
from experiments import stac_budget_data as d

def storage(obj):
 seen=set();stor={};headers=0;raw=0
 def visit(x):
  nonlocal headers,raw
  if id(x) in seen:return
  seen.add(id(x));headers+=sys.getsizeof(x)
  if isinstance(x,torch.Tensor):
   a=x.untyped_storage()
   if a.nbytes():stor[(str(x.device),a.data_ptr())]=a.nbytes()
  elif isinstance(x,dict):
   for k,v in x.items():visit(k);visit(v)
  elif isinstance(x,(tuple,list,set)):
   for v in x:visit(v)
  elif hasattr(x,'audit_tensors'):
   visit(x.audit_tensors());raw+=int(x.audit_raw_bytes())
  elif not isinstance(x,(torch.nn.Module,types.ModuleType,types.FunctionType,types.MethodType,type)) and hasattr(x,'__dict__'):visit(vars(x))
 visit(obj)
 return dict(unique_tensor_bytes=sum(stor.values()),Python_object_bytes=headers,raw_CUDA_workspace_bytes=raw,
             CUDA_tensor_bytes=sum(v for (dev,_),v in stor.items() if dev.startswith('cuda')),
             CPU_tensor_bytes=sum(v for (dev,_),v in stor.items() if dev.startswith('cpu')),
             total_bytes=sum(stor.values())+headers+raw,unique_storages=len(stor))

class CudaSnapshot:
 def __init__(self,cpp):self.state=cpp.export_state();self.H=cpp.num_heads;self.D=cpp.head_dim;self.P=cpp.pivot_cap;self.B=cpp.budget_cap;self.dtype=next(x.dtype for k,x in self.state.items() if k=='seg_buf_K_')
 def restore(self):
  from merger_cuda import create_merger_wrapper
  # Minimal constructor allocation; exact retained capacities/free lists replace it.
  cpp=create_merger_wrapper(self.H,self.D,self.P,self.B,1,dtype=self.dtype,device='cuda',seg_mode=True)
  cpp.import_state(self.state);return cpp

class HistoryMeter:
 def __init__(self,objects,name,common_backing):self.objects=objects;self.name=name;self.common=common_backing;self.peak=0;self.records=[];self.depth=0;self.wrapped=[];self.peak_model_VRAM=0
 def current(self):
  x=storage(self.objects());x['readable_original_RGB_backing_bytes']=self.common;x['total_history_bytes']=x['total_bytes']+self.common;return x
 def record(self,kind,peak=None,seconds=None):
  m=self.current();value=max(m['total_history_bytes'],peak or 0);self.peak=max(self.peak,value);self.peak_model_VRAM=max(self.peak_model_VRAM,torch.cuda.max_memory_allocated());row=dict(kind=kind,current=m,history_peak_bytes=value,peak_history_so_far=self.peak,seconds=seconds);self.records.append(row)
  if self.peak>16*2**30:raise RuntimeError('HARD_HISTORY_BYTES_EXCEEDED:'+str(self.peak))
  return row
 def wrap(self,obj,method):
  original=getattr(obj,method)
  @functools.wraps(original)
  def measured(*args,**kwargs):
   if self.depth:return original(*args,**kwargs)
   self.depth+=1;torch.cuda.synchronize();before=self.current();allocated=torch.cuda.memory_allocated();self.peak_model_VRAM=max(self.peak_model_VRAM,torch.cuda.max_memory_allocated());torch.cuda.reset_peak_memory_stats();started=time.monotonic()
   try:result=original(*args,**kwargs)
   finally:
    torch.cuda.synchronize();after=self.current();transient=max(0,torch.cuda.max_memory_allocated()-allocated);raw_growth=max(0,after['raw_CUDA_workspace_bytes']-before['raw_CUDA_workspace_bytes']);peak=max(after['total_history_bytes'],before['total_history_bytes']+transient+raw_growth);self.depth-=1;self.record(method,peak,time.monotonic()-started)
   return result
  setattr(obj,method,measured);self.wrapped.append((obj,method))
 def detach(self):
  for obj,key in self.wrapped:delattr(obj,key)
  self.wrapped=[]

def instrument(meter,manager):
 for method in ('append_kv','decode_sparse_attn','append_positions','retrieve_kv_parallel','prune_kv'):
  if hasattr(manager,method):meter.wrap(manager,method)

def snapshot_cpp(manager):
 if hasattr(manager,'_vstore'):
  cpp=manager._vstore._stac_merger;manager._vstore._stac_merger=CudaSnapshot(cpp);return cpp
 return None
def restore_cpp(manager):
 if hasattr(manager,'_vstore') and isinstance(manager._vstore._stac_merger,CudaSnapshot):
  manager._vstore._stac_merger=manager._vstore._stac_merger.restore()
