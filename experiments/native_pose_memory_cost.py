"""Actual uncached native initialisation + GS raster + reader + original camera head timings."""
import os,time,argparse
import numpy as np
import torch
from experiments.native_pose_memory_data import *
from experiments.native_pose_memory_run import Worker,official,to,alignment,native_step,patchgeometry,camrun
from models.native_pose_memory_reader import NativePoseMemoryReader
from models.gaussian_scene_memory import rasterize_field,field_bytes
from models.camera_only_feature_reader import view_fields
from torch.nn import functional as F

@torch.no_grad()
def main():
    torch.set_num_threads(4);w=Worker('cost_0');records=[]
    write(OUT/'runtime_environment.json',dict(torch=torch.__version__,CUDA_compiled=torch.version.cuda,CUDA_access=True,device=torch.cuda.get_device_name(),visible_device_count=torch.cuda.device_count(),numpy=np.__version__,python=__import__('sys').version,GLIBC_unchanged=True))
    try:
        t=time.monotonic();m=official();torch.cuda.synchronize();cold_model=time.monotonic()-t;w.event('model_load',seconds=cold_model)
        readers={}
        for k in KINDS:
            r=NativePoseMemoryReader().cuda();r.load_state_dict(load(CACHE/'fits'/f'{k}_20261110'/'final.pt')['model']);r.eval();readers[k]=r
        for meta in [plan()['training_catalog'][0],plan()['catalog'][0]]:
            rid=meta['id'];t=time.monotonic();sourcep=load(source(rid));field=to(load(scene(rid)));sc=load(sceneout(rid));torch.cuda.synchronize();cold_state=time.monotonic()-t
            t=time.monotonic();warm=load(sceneout(rid));warm_state=time.monotonic()-t;del warm
            state=dict(kv=[None]*m.aggregator.depth,cam=[None]*4,n=0);torch.cuda.synchronize();t=time.monotonic()
            for rgb in sourcep['rgb']:
                state,*_=native_step(m,rgb,state,False);w.event('source_native')
            torch.cuda.synchronize();prefix_seconds=time.monotonic()-t
            geom=to(sc['geometry']);align=to(sc['alignment']);enc={};build={}
            for k in ('P','S','G'):
                torch.cuda.synchronize();t=time.monotonic();enc[k]=readers[k].encode_source(sc['ordinary'][None].cuda());torch.cuda.synchronize();build[k]=time.monotonic()-t
                # Use actual persistent K/V reload (exported once for eight normal queries).
                t=time.monotonic();saved=load(CACHE/'public_states'/f'{k}_20261110'/f'{rid}.pt');loaded=to(saved['encoded']);torch.cuda.synchronize();reload_seconds=time.monotonic()-t
                assert all(torch.equal(enc[k][n],loaded[n]) for n in enc[k]);enc[k]=loaded
                log('cost_state.jsonl',dict(id=rid,kind=k,KV_build_seconds=build[k],KV_reload_seconds=reload_seconds,KV_tensor_bytes=field_bytes(loaded),cold_state_load_seconds=cold_state,warm_state_load_seconds=warm_state,prefix_seconds=prefix_seconds))
            for q in QUERIES:
                tio=time.monotonic();payload=load(query(rid,q));images=payload['rgb'][None,None].cuda().float()/255;torch.cuda.synchronize();query_io_seconds=time.monotonic()-tio
                torch.cuda.synchronize();t=time.monotonic()
                with torch.autocast('cuda',dtype=torch.bfloat16):tokens,start,_=m.aggregator(images,past_key_values=list(state['kv']),use_cache=True,past_frame_idx=4)
                torch.cuda.synchronize();agg_seconds=time.monotonic()-t;t=time.monotonic()
                poses,_=m.camera_head(tokens,past_key_values_camera=list(state['cam']),use_cache=True);pose=poses[-1][0,0];torch.cuda.synchronize();initial_head_seconds=time.monotonic()-t;t=time.monotonic()
                depth=m.depth_head(tokens,images=images,patch_start_idx=start)[0][0,0,...,0];torch.cuda.synchronize();depth_seconds=time.monotonic()-t;w.event('query_native_initial')
                from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri
                _,K=pose_encoding_to_extri_intri(pose[None,None],image_size_hw=(448,448));K=K[0,0];cam=__import__('experiments.native_pose_memory_data',fromlist=['mappose']).mappose(pose,align)
                torch.cuda.synchronize();t=time.monotonic();lg=patchgeometry(depth*align['scale'],K,cam,sc['unit'].cuda(),field['source_c2w'][0]);v=view_fields(geom,cam,K);lv=view_fields(lg,cam,K);torch.cuda.synchronize();geometry_seconds=time.monotonic()-t
                colors=torch.cat((field['features'].float()*field['has_content'][:,None],field['has_content'][:,None].float()),-1);Kr=K.clone();Kr[:2]/=14
                torch.cuda.synchronize();t=time.monotonic();rr,_,_=rasterize_field(field,cam[None],Kr[None],32,32,colors);torch.cuda.synchronize();raster_seconds=time.monotonic()-t;w.event('gsplat_query')
                r=rr[0].reshape(1024,1025);alpha=r[:,-1:];anchor=torch.where(alpha>0,(r[:,:1024].double()/alpha.double().clamp_min(1e-300)).float(),0.);features=tokens[-1][0,0,start:,:1024].float();qo=torch.cat((F.layer_norm(features,(1024,)),lg['source20']),-1)[None];normK=K.clone();normK[:2]/=448;xy=lv['common25'][:,3:5]/2
                for k in KINDS:
                    ordinary=(qo if k=='L' else sc['ordinary'][None].cuda());vv={n:x[None] for n,x in (lv if k=='L' else v).items()};aa=torch.zeros_like(anchor) if k=='L' else anchor;al=torch.zeros_like(alpha) if k=='L' else alpha
                    torch.cuda.synchronize();t=time.monotonic();res=readers[k](qo,aa[None],al[None],ordinary,vv,normK[None],xy[None],k,enc.get(k));torch.cuda.synchronize();reader_seconds=time.monotonic()-t;w.event('cost_reader_samples')
                    t=time.monotonic();refined=camrun(m.camera_head,tokens[-1][0,0,0].float()[None]+res,list(state['cam']));torch.cuda.synchronize();head_seconds=time.monotonic()-t;w.event('cost_camera_head_samples')
                    total=query_io_seconds+agg_seconds+initial_head_seconds+depth_seconds+geometry_seconds+reader_seconds+head_seconds+(raster_seconds if k!='L' else 0.)
                    records.append(dict(id=rid,query=q,kind=k,query_input_load_transfer_seconds=query_io_seconds,query_aggregator_seconds=agg_seconds,initial_camera_head_seconds=initial_head_seconds,initial_depth_seconds=depth_seconds,query_geometry_seconds=geometry_seconds,gsplat_seconds=raster_seconds if k!='L' else 0.,reader_seconds=reader_seconds,refined_camera_head_seconds=head_seconds,total_query_seconds=total,native_pose_only_seconds=agg_seconds+initial_head_seconds,native_shared_initialisation_with_depth_seconds=agg_seconds+initial_head_seconds+depth_seconds,prefix_build_seconds=prefix_seconds,source_KV_build_seconds=build.get(k,0.),GS_tensor_bytes=field_bytes(field),source_state_load_seconds=cold_state))
            del state,field;torch.cuda.empty_cache()
        write(OUT/'actual_cost.json',dict(complete=True,cold_model_load_seconds=cold_model,records=records,summary={k:{n:dict(median=float(np.median([r[n] for r in records if r['kind']==k])),p90=float(np.quantile([r[n] for r in records if r['kind']==k],.9))) for n in ['total_query_seconds','query_aggregator_seconds','initial_camera_head_seconds','initial_depth_seconds','query_geometry_seconds','gsplat_seconds','reader_seconds','refined_camera_head_seconds']} for k in KINDS},amortization=[dict(kind=k,queries=N,mean_prefix_plus_KV_per_query_seconds=float(np.mean([(r['prefix_build_seconds']+r['source_KV_build_seconds'])/N for r in records if r['kind']==k])),mean_total_including_prefix_KV_seconds=float(np.mean([r['total_query_seconds']+(r['prefix_build_seconds']+r['source_KV_build_seconds'])/N for r in records if r['kind']==k]))) for k in KINDS for N in (1,4,8,32)],known_historical_build_amortization=[dict(kind=k,queries=N,original24_prepared_source_GS_additional_per_query_seconds=[x/N for x in read(OUT/'cost_inherited.json')['original24_warm_sensor_prepared_source_to_field_seconds']],added20_construction_worker_sum_per_scene_per_query_seconds=sum(read(OUT/'cost_inherited.json')['added20_construction'][n] for n in ['source_prediction_worker_seconds','source_fit_worker_seconds','binding_encoding_render_cache_worker_seconds'])/20/N,scope='Inherited stages differ; not a newly measured full RGB construction total; sensor/preprocessing/predictor cold-load unknown') for k in KINDS for N in (1,4,8,32)],shared_source_GS_construction='Inherited original source/GS construction records; not zero or rerun; see report and cost_inherited.json',KV_reload_exact=True,cache_IO_not_substitute_for_encoder=True))
        w.close('complete')
    except BaseException:
        w.close('failed');raise
if __name__=='__main__':main()
