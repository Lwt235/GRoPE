"""Exactly200 full warm queries: encoder, original render, required PE, native depth."""
import time
import numpy as np
import torch
from experiments.native_render_parameter_data import *
from experiments.native_receiver_runtime import model_load,forward_read as old_forward
from experiments.native_receiver_data import core_load
from experiments.native_render_parameter_train import modules,verify_lock
from experiments.native_render_parameter_prepare import fields
from models.gaussian_scene_parameter_pe import forward_read
from models.gaussian_scene_memory import rasterize_field,contribution_weights
from models.gaussian_memory_coarse_pe import patch_tokens
from experiments.gaussian_scene_data import query_image


def sync():torch.cuda.synchronize()


@torch.no_grad()
def main():
    torch.set_num_threads(4);p=plan();verify_lock();begin=time.monotonic();model=model_load(p,offload_unused=False);sync();cold=time.monotonic()-begin;fusion,pe=modules(SEEDS[0]);records=[]
    marker=OUT/'cost.json'
    if marker.exists():assert read(marker)['complete'];return
    for kind in ['P','S','G','Render']:
        state=load(parent.checkpoint_path('Render',SEEDS[0]) if kind=='Render' else checkpoint_path(kind,SEEDS[0]));core_load(model,state['native']);fusion.load_state_dict(state['model'] if kind=='Render' else state['fusion'])
        if kind!='Render':pe.load_state_dict(state['pe'])
        del state
        for rid,q,c in [('fit_00',1,'missing'),('fidelity_00',2,'full')]:
            query=load(query_path(rid,q),'cuda');field=load(scene_path(rid),'cuda');K=query['K'].clone();K[:2]/=14;values=[];segments=[];torch.cuda.reset_peak_memory_stats()
            for iteration in range(25):
                budget_charge('nontraining_forwards');sync();start=time.monotonic()
                with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):patches=patch_tokens(model,query_image(query,c))
                sync();encoded=time.monotonic()
                # Original direct content rendering; no cached rendered features or PE.
                colors=field['features'].float()*field['has_content'][:,None]
                rendered,_,_=rasterize_field(field,query['c2w'][None],K[None],32,32,colors);extra=rendered[0].reshape(1024,1024);sync();rendered_at=time.monotonic()
                sample=dict(patches=patches,extra=extra,hw=(448,448))
                if kind!='Render':
                    w,_=contribution_weights(field,query['c2w'][None],K[None],32,32);payload=fields(field,query['c2w'],K,w[0],kind=kind);sample.update(position=position(payload,kind),alpha=payload['a'])
                sync();position_at=time.monotonic()
                depth=old_forward(model,sample,fusion,'Render',checkpoint_blocks=False) if kind=='Render' else forward_read(model,sample,fusion,pe,checkpoint_blocks=False)
                sync();end=time.monotonic();assert torch.isfinite(depth).all() and (depth>0).all()
                if iteration>=5:values.append(end-start);segments.append(dict(encoder_seconds=encoded-start,render_seconds=rendered_at-encoded,PE_fields_seconds=position_at-rendered_at,native_seconds=end-position_at))
                del sample,patches,rendered,extra,depth
            records.append(dict(kind=kind,id=rid,query=q,condition=c,seed=SEEDS[0],warmup=5,measured=20,seconds=values,segments=segments,peak_GiB=torch.cuda.max_memory_allocated()/2**30));print(dict(cost_done=kind,id=rid,median_ms=1000*float(np.median(values))),flush=True)
            del query,field;torch.cuda.empty_cache()
    summary={k:dict(median_ms=1000*float(np.median([v for r in records if r['kind']==k for v in r['seconds']])),peak_GiB=max(r['peak_GiB'] for r in records if r['kind']==k)) for k in ['P','S','G','Render']}
    write(marker,dict(complete=True,records=records,summary=summary,total_native_forwards=200,cold_model_load_seconds=cold,scope='Preprocessed448RGB and saved source field resident with given camera; includes frozen encoder, actual original direct content render, actual required contribution/FP64 PE-field construction, shared PE projection, all24 native frame/global and original DPT. Excludes disk/JPEG preprocessing, camera estimation and source construction. No full feature/PE cache substituted.',required_fields_only='P computes center46 plus zeros; S adds sphere6; G adds full6. All use unchanged full-shape content rendering.',shared_GPU_not_exclusive_speed_claim=True,source_construction_reference=str(parent.OUT/'cost_description.json'),scene_tensor_bytes=4319652,source_not_rebuilt_or_retimed_as_zero=True))
    print('Exactly200 full warm-query calls completed.',flush=True)


if __name__=='__main__':main()
