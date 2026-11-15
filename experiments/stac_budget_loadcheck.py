"""Necessary actual author load and first-source no-compression correspondence."""
import os,sys,time,resource
from experiments import stac_budget_data as d
sys.path[:0]=[str(d.AUTHOR),str(d.AUTHOR/'attn-cuda'),str(d.AUTHOR/'merger-cuda'),str(d.ROOT/'third_party/native_history_python'),str(d.ROOT/'third_party/streamvggt/src')]
import torch
from experiments.stream_history_data import rgb

def load_author():
 from causalvggt.models.vggt import CausalVGGT
 p=d.plan();model=CausalVGGT(base_model='streamvggt')
 weights=torch.load(p['existing_model_path'],map_location='cpu',weights_only=True)
 wrapped='model' in weights
 if wrapped:weights=weights['model']
 result=model.load_state_dict(weights,strict=False)
 assert not result.missing_keys,result.missing_keys
 assert all(k.startswith('track_head.') for k in result.unexpected_keys),result.unexpected_keys
 rec=dict(checkpoint_path=p['existing_model_path'],weight_identity_inherited=p['model_sha256_inherited'],checkpoint_model_wrapper_unwrapped=wrapped,
          missing_keys=result.missing_keys,unexpected_keys=result.unexpected_keys,loaded_parameter_tensors=len(model.state_dict()),
          parameter_elements=sum(x.numel() for x in model.parameters()),weights_bytes=sum(x.untyped_storage().nbytes() for x in model.parameters()),
          policy_kernels='fixed author CUDA; no Python substitute',frozen=True)
 del weights
 return model.eval().requires_grad_(False),rec

@torch.inference_mode()
def main():
 started=time.monotonic();torch.set_num_threads(2);torch.manual_seed(20261115);torch.backends.cuda.matmul.allow_tf32=False;torch.cuda.set_per_process_memory_fraction(40*2**30/torch.cuda.get_device_properties(0).total_memory)
 print(dict(PID=os.getpid(),stage='actual_author_load'),flush=True)
 model,rec=load_author();model.cuda();torch.cuda.synchronize();rec['load_seconds']=time.monotonic()-started
 import attn_cuda,merger_cuda
 assert attn_cuda.is_available() and merger_cuda.has_merger_wrapper()
 m=d.plan()['catalog'][0];image=rgb(m['frames'][0])[None,None].cuda()
 with torch.autocast('cuda',dtype=torch.bfloat16):out=model(image,mode='full',streaming=True,is_anchor_exist=True)
 native={k:out[k].float().cpu() for k in ('depth','pose_enc','world_points')}
 assert all(torch.isfinite(x).all() for x in native.values())
 print(dict(PID=os.getpid(),stage='author_CUDA_forward_complete',shapes={k:list(v.shape) for k,v in native.items()}),flush=True)
 del out;model.cpu();torch.cuda.empty_cache()
 from streamvggt.models.streamvggt import StreamVGGT
 official=StreamVGGT().eval().requires_grad_(False)
 # Exact shared tensors from the successfully loaded existing checkpoint; unused track head excluded from comparison.
 result=official.load_state_dict(model.state_dict(),strict=False)
 assert all(k.startswith('track_head.') for k in result.missing_keys) and not result.unexpected_keys
 del model
 official.cuda()
 with torch.autocast('cuda',dtype=torch.bfloat16):tokens,patch=official.aggregator(image)
 with torch.autocast('cuda',enabled=False):
  poses=official.camera_head(tokens)[-1];depth,_=official.depth_head(tokens,images=image,patch_start_idx=patch);points,_=official.point_head(tokens,images=image,patch_start_idx=patch)
 checks={k:dict(max_abs=float((native[k]-value.float().cpu()).abs().max()),mean_abs=float((native[k]-value.float().cpu()).abs().mean()),finite=bool(torch.isfinite(value).all())) for k,value in [('depth',depth),('pose_enc',poses),('world_points',points)]}
 rec.update(PID=os.getpid(),complete=True,source_id=m['id'],source_frame=0,author_model_forward_calls=1,official_aggregator_calls=1,official_camera_calls=1,official_point_calls=1,official_depth_calls=1,no_compression_correspondence=checks,GT_opened_or_scored=False,
            actual_seconds=time.monotonic()-started,peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
 d.write(d.OUT/'load_validation.json',rec)
 print(rec,flush=True)
if __name__=='__main__':main()
