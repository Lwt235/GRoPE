"""Full warm native queries; construction timing from paired prepared source tensors."""
import time
import os
import subprocess

import numpy as np
import torch

from experiments.source_geometry_data import *
from experiments.source_geometry_build import sampled_inputs,fit_one
from experiments.gaussian_memory_native_run import model_load
from experiments.gaussian_scene_data import query_image
from models.gaussian_scene_memory import SceneReader,field_bytes
from models.gaussian_scene_native import prepare_read,forward_read

def live():
    uuid=os.environ.get('CUDA_VISIBLE_DEVICES','0')
    return dict(gpu=subprocess.check_output(['nvidia-smi','-i',uuid,'--query-gpu=uuid,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip(),pid=os.getpid())

@torch.no_grad()
def query_cost(model,p):
    records=[]
    for mode in MODES:
        for meta in p['training_catalog'][:2]:
            rid=meta['id'];query=load(query_path(rid,2),'cuda');begin=time.monotonic()
            field=load(branch_cache(mode)/rid/'scene_memory.pt','cuda');torch.cuda.synchronize();field_load=time.monotonic()-begin
            for kind in KINDS:
                reader=SceneReader(center_pe=kind=='P').cuda();reader.load_state_dict(load(branch_out(mode)/f'{kind}_{SEEDS[0]}'/'step_2000.pt')['model']);reader.eval()
                for condition in CONDITIONS:
                    image=query_image(query,condition);samples=[];geometry=[];native=[];torch.cuda.reset_peak_memory_stats()
                    for i in range(7):
                        torch.cuda.synchronize();begin=time.monotonic()
                        with torch.autocast('cuda',dtype=torch.bfloat16):sample=prepare_read(model,image,query['c2w'],query['K'],field,kind)
                        torch.cuda.synchronize();middle=time.monotonic()
                        with torch.autocast('cuda',dtype=torch.bfloat16):depth=forward_read(model,sample,reader,kind,checkpoint_blocks=False)
                        torch.cuda.synchronize();end=time.monotonic();assert torch.isfinite(depth).all()
                        if i>=2:samples.append(end-begin);geometry.append(middle-begin);native.append(end-middle)
                        del sample,depth
                    records.append(dict(mode=mode,kind=kind,id=rid,condition=condition,field_load_seconds=field_load,source_state_tensor_bytes=field_bytes(field),total_query_seconds=samples,current_encoder_and_geometry_seconds=geometry,native_and_four_reader_calls_seconds=native,peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30))
                del reader
            del field,query;torch.cuda.empty_cache()
    return records


def construction_cost(model,p):
    # Rebuild cost-only clones, no cached geometry or source features reused in timed region.
    from experiments.source_geometry_prepare import bind
    records=[]
    for meta in p['training_catalog'][:2]:
        rid=meta['id'];bid='cost_'+rid+'_v1';path=source_path(bid)
        if path.exists():raise RuntimeError('Partial cost clone exists; preserve and version explicitly before retiming')
        source=load(source_path(rid));source['id']=bid;path.parent.mkdir(parents=True,exist_ok=True);save_tensor(path,source)
        for mode in MODES:
            torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();begin=time.monotonic()
            source_gpu=load(path,'cuda');depths,valid,indices,candidate=sampled_inputs(source_gpu)
            # Fitting needs gradients even though query timing does not.
            with torch.enable_grad():fit_one(source_gpu,mode,depths,valid,indices,candidate)
            if mode=='sensor':
                # Timed construction of either arm should include its own source encoder work.
                shared=CACHE/'inputs'/bid/'source_features.pt'
                if shared.exists():shared.rename(shared.with_name('source_features_pred_cost.pt'))
            bind(model,dict(meta,id=bid),mode);torch.cuda.synchronize()
            records.append(dict(id=rid,mode=mode,cost_clone=bid,prepared_source_to_saved_field_seconds=time.monotonic()-begin,peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30))
            del source_gpu,depths,valid,indices;torch.cuda.empty_cache()
    return records


def main():
    torch.set_num_threads(4);p=plan();marker=OUT/'cost/complete.json'
    if marker.exists():return
    before=live();begin=time.monotonic();model=model_load(p);torch.cuda.synchronize();cold=time.monotonic()-begin
    construction=construction_cost(model,p);queries=query_cost(model,p)
    result=dict(complete=True,cold_native_load_seconds=cold,construction=construction,queries=queries,live_before=before,live_after=live(),
        construction_scope='Prepared source RGB/predicted depth/sensor depth/common masks/cameras on disk to saved field; loading, selection, fitting, fresh frozen source encoder and lifting included. No raw image decoding/preprocessing, AnySplat prediction generation, sensor acquisition or model cold load in warm region. Both arms use precomputed predictor outputs for fixed masks/scale; not end-to-end RGB cost.',
        query_scope='Prepared GPU current RGB plus resident saved field to fresh encoder, query geometry, all native blocks, four shared reads and DPT. Field and model loading separate; no cached current encoder, no image-file I/O; shared GPU is not exclusive speed claim.')
    write_json(marker,result)

if __name__=='__main__':main()
