"""One fixed fresh RGB query: measured common initializer, actual read and tracker.

No training, parameter sweep or substitute cached full-query timing. Shared stages
are executed once and charged to each applicable deployment pipeline explicitly.
"""
import argparse,time,resource,os,sys
import torch
from experiments import native_source_resolved_data as d
from experiments.native_track_memory_prepare import alignment,camera_head
from experiments.native_source_resolved_run import model_load,make_adapter
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
        return d.load(d.source(rid)),d.load(d.pair(rid)),d.load(d.old.query_path(rid,q)),d.device({k:v for k,v in d.load(d.old.scene_path(rid)).items() if k!='features'}),d.device(d.load(d.fields_path(rid)))
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
    state_bytes.update(source_camera_KV=tensor_bytes(cp),source_camera_KV_unique_storage=storage_bytes(cp),source_metric_alignment=tensor_bytes(align),global_adapter_FP32_parameters=1236864*4)
    qpatch=measured('fresh_query_DINO',lambda:patch(qp['rgb']));qt,_=measured('initial_native_query',lambda:native(qpatch,prefix,1))
    def initial_camera():
        from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri
        poseq,_=camera_head(m,qt,cp);_,K=pose_encoding_to_extri_intri(poseq[None,None],image_size_hw=(448,448));return mappose(poseq,align),K[0,0]
    cam,K=measured('initial_query_camera',initial_camera)
    def gsfields():
        kp=K.clone();kp[:2]/=14;w,alpha=contribution_weights(field,cam[None],kp[None],32,32);mass=w[0].sum((0,1));return scene_view(field,g,dict(c2w=cam,K=K),mass)
    gs=measured('fresh_GS_raster_and_fields',gsfields);records=[]
    states={rep:measured(rep+'_source_state_load_transfer',lambda rep=rep:d.device(d.load(d.state(rid,rep)))) for rep in ('F','M')}
    state_bytes['source_resolved_F_content']=tensor_bytes(states['F']);state_bytes['matched_mean_M_content']=tensor_bytes(states['M']);state_bytes['shared_global_PCA']=tensor_bytes(d.load(d.CACHE/'PCA255.pt'))
    for kind in d.KINDS:
        def adapter_load():
            out=make_adapter();out.load_state_dict(d.load(d.CACHE/'fits'/f'{kind}_{d.SEEDS[0]}'/'final.pt')['adapter']);return out.eval()
        adapter=measured(f'{kind}_final_adapter_load',adapter_load);v=gs
        def inputs():
            content=states[kind[0]]['content'].float()
            return dict(ordinary=torch.cat((content,v['ordinary']),-1)[None],position=torch.cat((v['common25'],v[kind[-1]]),-1)[None],scene_pos=v['scene_pos'][None],key_mask=v['key_mask'][None])
        sample=measured(f'{kind}_query_adapter_inputs',inputs)
        def adapted():
            with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):return query_tokens(m,qpatch,prefix,adapter,sample,False)
        token=measured(f'{kind}_second_native_query_with_GS_read',adapted);res=measured(f'{kind}_query_DPT_TrackHead',lambda:track(m,af,token,d.point_grid()[None].cuda()))
        assert all(torch.isfinite(c).all() for c in res[0]);d.journal(name,kind='cost_adapted_query',arm=kind,tracker_iterations=4)
        source=sum(times[k] for k in ['fresh_anchor_DINO','anchor_native_prefix','anchor_DPT','anchor_camera_depth_sensor_calibration'])
        query=sum(times[k] for k in ['fresh_query_DINO','initial_native_query','initial_query_camera','fresh_GS_raster_and_fields',f'{kind}_query_adapter_inputs',f'{kind}_second_native_query_with_GS_read',f'{kind}_query_DPT_TrackHead'])
        binding=d.read(d.OUT/'binding'/f'{rid}.json');formation=binding['seconds'];state_load=times[kind[0]+'_source_state_load_transfer']
        records.append(dict(kind=kind,once_per_place_source_content_formation_seconds=formation,source_state_load_transfer_seconds=state_load,global_PCA_seconds=d.read(d.OUT/'PCA.json')['seconds'],source_dependency_seconds=source,fresh_full_query_seconds=query,adapter_load_seconds=times[f'{kind}_final_adapter_load'],amortized=[dict(queries=n,seconds_per_query=query+(source+formation+state_load)/n,scope='Includes one current scene source binding and state load plus native anchor; global training PCA, model/adapter/input cold loads and original GS formation separately shown; source binding timer formed both F and M, conservative per-arm charge') for n in (1,4,8)]));del adapter,token,res
    source_feature_path=d.source_patches(rid)[1]
    reconstruction=dict(source_DINO_cache_path=str(source_feature_path),source_DINO_cache_file_bytes=source_feature_path.stat().st_size,source_RGB_input_path=str(d.source(rid)),source_RGB_input_file_bytes=d.source(rid).stat().st_size,source_sensor_pose_pair_path=str(d.pair(rid)),source_sensor_pose_pair_file_bytes=d.pair(rid).stat().st_size,original_field_path=str(d.old.scene_path(rid)),original_field_file_bytes=d.old.scene_path(rid).stat().st_size,original_source_geometry_path=str(d.fields_path(rid)),original_source_geometry_file_bytes=d.fields_path(rid).stat().st_size,source_prefix_and_DPT_persisted_as_new_files=False,scope='Read-only historical reconstruction inputs, not newly copied DATA or a claim that runtime KV/DPT are actually serialized. Cost freshly re-encodes RGB; evaluation reconstructs native anchor from the source-DINO cache. Shared original physical field file retains an unused mean f; only geometry consumed in this new cost.')
    inherited=d.read(d.prior.OUT/'actual_cost.json');d.write(d.OUT/'cost_inherited.json',dict(round026_actual_cost=inherited,round026_original_formation=d.read(d.prior.OUT/'cost_inherited.json'),scope='Inherited observations, no old Native2/4/P/G/L reruns; old one-case fixed-order timings are not paired new throughput measurements; raw-source acquisition/preprocessing remain outside measured scope.'))
    d.write(d.OUT/'actual_cost.json',dict(complete=True,source_binding_scope='Already executed once per44 places on same hardware, includes reading original state/source features, alpha raster, projection, F+M state writes/reload and numerical disclosure; not benchmarked again',shared_global_PCA_seconds=d.read(d.OUT/'PCA.json')['seconds'],binding_all44_seconds=sum(d.read(p)['seconds'] for p in (d.OUT/'binding').glob('*.json')),inherited_reference_costs_are_not_new_measurements=True,persistent_reconstruction_dependencies=reconstruction,fixed_case=dict(id=rid,anchor=anchor,query=q,seed=d.SEEDS[0]),one_case_no_distribution_or_benchmark_claim=True,cost_dispatch=d.read(d.OUT/'cost_dispatch.json') if (d.OUT/'cost_dispatch.json').exists() else None,stages_seconds=times,records=records,state_tensor_bytes=state_bytes,calibration=adiag,shared_stages_measured_once_and_charged_to_each_applicable_pipeline=True,patches_reencoded_from_RGB=True,RGB_scope='Existing prepared448 RGB tensors freshly re-encoded by DINO; raw image decode/resize and sensor acquisition are outside this fixed cost case. Tensor file cold load and transfer are measured separately.',no_cached_query_timing_substitution=True,Native2_camera_overhead_excluded_from_reference_total=True,GS_formation_not_repeated_or_free='See cost_inherited.json for old original physical GS formation; new global PCA and44 source bindings charged in this round; each one-place binding conservatively charged per deployment record, global PCA listed separately. Acquisition/preprocessing unknown',actual_calls=dict(fresh_DINO=2,native_anchor_or_source_frames=1,initial_query=1,Native4_query=0,adapted_query=5,official_track_head_4iteration_calls=5,source_depth=1,source_camera=1,query_camera=1,GS_raster=1),model_and_input_cold_loads_separate=True,host_wall_seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),python=sys.version,torch=torch.__version__,CUDA=torch.version.cuda,GLIBC_unchanged=True))
    print(dict(cost_complete=True,id=rid,records=records),flush=True)
if __name__=='__main__':main()
