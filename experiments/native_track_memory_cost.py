"""One fixed fresh RGB query: measured common initializer, actual read and tracker.

No training, parameter sweep or substitute cached full-query timing. Shared stages
are executed once and charged to each applicable deployment pipeline explicitly.
"""
import argparse,time,resource,os,sys
import torch
from experiments import native_track_memory_data as d
from experiments.native_track_memory_prepare import model_load,alignment,camera_head,local_view
from experiments.native_pose_memory_data import mappose
from experiments.native_scene_exchange_data import scene_view
from models.native_track_memory import TrackMemory,native_tokens,query_tokens,fmap,track
from models.gaussian_scene_memory import contribution_weights

@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);a=p.parse_args();name=a.name
    torch.set_num_threads(2);torch.backends.cuda.enable_mem_efficient_sdp(False);start=time.monotonic();times={}
    def measured(label,fn):
        d.journal(name,kind='cost_stage_attempt',stage=label);torch.cuda.synchronize();t=time.monotonic();out=fn();torch.cuda.synchronize();times[label]=time.monotonic()-t;d.journal(name,kind='cost_stage',stage=label,seconds=times[label]);return out
    m=measured('cold_model_load',lambda:model_load(name));measured('DINO_weight_restore_to_GPU',lambda:m.aggregator.patch_embed.cuda())
    rid=d.plan()['training_catalog'][0]['id'];anchor=0;q=2
    def input_load():
        return d.load(d.source(rid)),d.load(d.pair(rid)),d.load(d.old.query_path(rid,q)),d.device(d.load(d.old.scene_path(rid))),d.device(d.load(d.fields_path(rid)))
    sp,pp,qp,field,g=measured('input_load_transfer',input_load)
    def patch(rgb):
        image=rgb[None,None].cuda().float()/255 if rgb.dtype==torch.uint8 else rgb[None,None].cuda().float();agg=m.aggregator
        with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):v=agg.patch_embed(((image-agg._resnet_mean.to(image.device))/agg._resnet_std.to(image.device)).reshape(1,3,448,448))
        return v['x_norm_patchtokens'] if isinstance(v,dict) else v
    def native(patches,prefix=None,index=0):
        with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):return native_tokens(m,patches,prefix,index)
    ap=measured('fresh_anchor_DINO',lambda:patch(sp['rgb'][0]));at,prefix=measured('anchor_native_prefix',lambda:native(ap));af=measured('anchor_DPT',lambda:fmap(m,at))
    def tensor_bytes(v):return v.numel()*v.element_size() if torch.is_tensor(v) else sum(tensor_bytes(x) for x in v.values()) if isinstance(v,dict) else sum(tensor_bytes(x) for x in v) if isinstance(v,(list,tuple)) else 0
    def storage_bytes(v):
        seen={}
        def visit(x):
            if torch.is_tensor(x):
                storage=x.untyped_storage();seen[(str(x.device),storage.data_ptr())]=storage.nbytes()
            elif isinstance(x,dict):
                for y in x.values():visit(y)
            elif isinstance(x,(list,tuple)):
                for y in x:visit(y)
        visit(v);return sum(seen.values())
    state_bytes=dict(single_anchor_native_KV=tensor_bytes(prefix),single_anchor_Track_DPT=tensor_bytes(af),original_GS_field=tensor_bytes(field),original_source_GEOMETRY_fields=tensor_bytes(g))
    state_bytes['single_anchor_KV_unique_storage_including_strided_QKV_base']=storage_bytes(prefix)
    def calibrate():
        pose,cp=camera_head(m,at,[None]*m.camera_head.trunk_depth)
        with torch.autocast('cuda',enabled=False):dep=m.depth_head(at,torch.empty(1,1,3,448,448,device='cuda'),5)[0][0,0,...,0]
        align,diag=alignment(pose,sp,pp,0,dep);return cp,align,diag
    cp,align,adiag=measured('anchor_camera_depth_sensor_calibration',calibrate)
    qpatch=measured('fresh_query_DINO',lambda:patch(qp['rgb']));qt,_=measured('initial_native_query',lambda:native(qpatch,prefix,1))
    def initial_camera():
        from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri
        poseq,_=camera_head(m,qt,cp);_,K=pose_encoding_to_extri_intri(poseq[None,None],image_size_hw=(448,448));return mappose(poseq,align),K[0,0]
    cam,K=measured('initial_query_camera',initial_camera)
    reference2=measured('Native2_query_DPT_TrackHead',lambda:track(m,af,qt,d.point_grid()[None].cuda()))
    def gsfields():
        kp=K.clone();kp[:2]/=14;w,alpha=contribution_weights(field,cam[None],kp[None],32,32);mass=w[0].sum((0,1));return scene_view(field,g,dict(c2w=cam,K=K),mass)
    gs=measured('fresh_GS_raster_and_fields',gsfields);lv=measured('fresh_L_unit_ray_fields',lambda:local_view(cam,K));records=[]
    for kind in d.KINDS:
        def adapter_load():
            out=TrackMemory().cuda();out.load_state_dict(d.load(d.CACHE/'fits'/f'{kind}_{d.SEEDS[0]}'/'final.pt')['adapter']);return out.eval()
        adapter=measured(f'{kind}_final_adapter_load',adapter_load);v=lv if kind=='L' else gs
        def inputs():
            content=qpatch[0].repeat_interleave(2,0) if kind=='L' else field['features'].float()
            return dict(ordinary=torch.cat((content,v['ordinary']),-1)[None],position=torch.cat((v['common25'],v['P' if kind=='L' else kind]),-1)[None],scene_pos=v['scene_pos'][None],key_mask=v['key_mask'][None])
        sample=measured(f'{kind}_query_adapter_inputs',inputs)
        def adapted():
            with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):return query_tokens(m,qpatch,prefix,adapter,sample,False)
        token=measured(f'{kind}_second_native_query_with_GS_read',adapted);res=measured(f'{kind}_query_DPT_TrackHead',lambda:track(m,af,token,d.point_grid()[None].cuda()))
        assert all(torch.isfinite(c).all() for c in res[0]);d.journal(name,kind='cost_adapted_query',arm=kind,tracker_iterations=4)
        source=sum(times[k] for k in ['fresh_anchor_DINO','anchor_native_prefix','anchor_DPT','anchor_camera_depth_sensor_calibration'])
        query=sum(times[k] for k in ['fresh_query_DINO','initial_native_query','initial_query_camera',('fresh_L_unit_ray_fields' if kind=='L' else 'fresh_GS_raster_and_fields'),f'{kind}_query_adapter_inputs',f'{kind}_second_native_query_with_GS_read',f'{kind}_query_DPT_TrackHead'])
        records.append(dict(kind=kind,source_dependency_seconds=source,fresh_full_query_seconds=query,adapter_load_seconds=times[f'{kind}_final_adapter_load'],amortized=[dict(queries=n,seconds_per_query=query+source/n,scope='Mixed native anchor prefix plus GS/L; excludes separately shown model/adapter/input cold loads and historical GS formation') for n in (1,4,8)]));del adapter,token,res
    # Native4: three actual additional source RGB encoders and causal prefixes.
    p4=prefix
    for j in (1,2,3):
        patchj=measured(f'Native4_source{j}_fresh_DINO',lambda j=j:patch(sp['rgb'][j]));_,p4=measured(f'Native4_source{j}_native_prefix',lambda:native(patchj,p4,1))
    t4,_=measured('Native4_native_query',lambda:native(qpatch,p4,4));ref4=measured('Native4_query_DPT_TrackHead',lambda:track(m,af,t4,d.point_grid()[None].cuda()))
    assert all(torch.isfinite(c).all() for c in reference2[0]+ref4[0]);d.journal(name,kind='cost_Native2_reference',tracker_iterations=4);d.journal(name,kind='cost_Native4_reference',tracker_iterations=4)
    n2source=sum(times[k] for k in ['fresh_anchor_DINO','anchor_native_prefix','anchor_DPT']);n2query=times['fresh_query_DINO']+times['initial_native_query']+times['Native2_query_DPT_TrackHead']
    n4source=n2source+sum(times[f'Native4_source{j}_{label}'] for j in (1,2,3) for label in ('fresh_DINO','native_prefix'));n4query=times['fresh_query_DINO']+times['Native4_native_query']+times['Native4_query_DPT_TrackHead']
    for kind,src,query in [('Native2',n2source,n2query),('Native4',n4source,n4query)]:records.append(dict(kind=kind,source_dependency_seconds=src,fresh_full_query_seconds=query,amortized=[dict(queries=n,seconds_per_query=query+src/n) for n in (1,4,8)]))
    state_bytes['Native4_source_KV']=tensor_bytes(p4);state_bytes['Native4_source_KV_unique_storage']=storage_bytes(p4);inherited=d.read(d.ROOT/'results/gaussian_memory/native_pose_memory_pe1/cost_inherited.json');d.write(d.OUT/'cost_inherited.json',inherited)
    d.write(d.OUT/'actual_cost.json',dict(complete=True,fixed_case=dict(id=rid,anchor=anchor,query=q,seed=d.SEEDS[0]),one_case_no_distribution_or_benchmark_claim=True,cost_dispatch=d.read(d.OUT/'cost_dispatch.json') if (d.OUT/'cost_dispatch.json').exists() else None,stages_seconds=times,records=records,state_tensor_bytes=state_bytes,calibration=adiag,shared_stages_measured_once_and_charged_to_each_applicable_pipeline=True,patches_reencoded_from_RGB=True,RGB_scope='Existing prepared448 RGB tensors freshly re-encoded by DINO; raw image decode/resize and sensor acquisition are outside this fixed cost case. Tensor file cold load and transfer are measured separately.',no_cached_query_timing_substitution=True,Native2_camera_overhead_excluded_from_reference_total=True,GS_formation_not_repeated_or_free='See cost_inherited.json; original24 prepared-source warm samples and added20 worker totals have different scopes; acquisition/preprocessing/cold predictors unknown',actual_calls=dict(fresh_DINO=5,native_anchor_or_source_frames=4,initial_query=1,Native4_query=1,adapted_query=4,official_track_head_4iteration_calls=6,source_depth=1,source_camera=1,query_camera=1,GS_raster=1),model_and_input_cold_loads_separate=True,host_wall_seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),python=sys.version,torch=torch.__version__,CUDA=torch.version.cuda,GLIBC_unchanged=True))
    print(dict(cost_complete=True,id=rid,records=records),flush=True)
if __name__=='__main__':main()
