"""One finite actual full-query cost pass, without cached-view timing substitutes."""
import subprocess,time
from experiments import native_scene_exchange_data as d
from experiments.native_scene_exchange_data import np,torch
from models.native_scene_exchange import SceneExchange,forward
from models.gaussian_scene_native import forward_read
from models.gaussian_memory_coarse_pe import patch_tokens
from models.gaussian_scene_memory import contribution_weights,field_to_device,field_bytes
from experiments.gaussian_scene_data import query_image
from experiments.native_receiver_runtime import model_load
from experiments.gaussian_scene_preflight import metric_depth

def local_from_camera(query,unit):
    """Same prepared L formula; only explicitly provided scalar unit is read."""
    K=query['K'].double().clone();K[:2]/=14;u=torch.tensor(unit,dtype=torch.float64)
    yy,xx=torch.meshgrid(torch.arange(32,dtype=torch.float64)+.5,torch.arange(32,dtype=torch.float64)+.5,indexing='ij');pixels=torch.stack((xx,yy,torch.ones_like(xx)),-1).reshape(-1,3)
    p=(pixels@torch.linalg.inv(K).T).repeat_interleave(2,0);world=(p*u)@query['c2w'][:3,:3].double().T+query['c2w'][:3,3].double()
    fake=dict(means=world.float(),unit=u.float(),source_c2w=query['c2w'][None],has_content=torch.ones(2048,dtype=torch.bool),rotations=torch.tensor([1.,0.,0.,0.]).repeat(2048,1),scales=u.float().expand(2048,3).clone(),opacities=torch.zeros(2048))
    g,_=d.base.source_geometry(fake);v=d.scene_view(fake,g,query,torch.zeros(2048));v['ordinary'][:,3:10]=0.;v['ordinary'][:,-1]=0.
    from models.gaussian_memory_native_pe import grid
    v['scene_pos']=grid(448,448,'cpu')[0,5:].repeat_interleave(2,0).float();return v

@torch.no_grad()
def main():
    torch.set_num_threads(4);assert not d.resource(True)['stop_reasons'];torch.cuda.set_per_process_memory_fraction(.48)
    start=time.monotonic();model=model_load(d.plan());model.aggregator.patch_embed.cuda();torch.cuda.synchronize();model_load_seconds=time.monotonic()-start
    rows=[];assets=[];loaders=[]
    for method in ['Native',*d.ARMS]:
        if method!='Native':
            start=time.monotonic();state=d.load(d.final(method,d.SEEDS[0]));adapter=SceneExchange().cuda();adapter.load_state_dict(state['adapter']);torch.cuda.synchronize();loaders.append(dict(arm=method,seconds=time.monotonic()-start,recovery_bytes=d.final(method,d.SEEDS[0]).stat().st_size,adapter_bytes=sum(t.numel()*t.element_size() for t in adapter.state_dict().values())));del state
        for mi,meta in enumerate(d.plan()['training_catalog'][:2]+d.plan()['catalog'][:2]):
            rid=meta['id'];start=time.monotonic()
            if method not in ('Native','L'):
                field=d.load(d.old.scene_path(rid));g=d.load(d.gpath(rid));fg=field_to_device(field,'cuda');torch.cuda.synchronize()
            loadseconds=time.monotonic()-start
            provided_unit=d.read(d.base.OUT/'preparation'/f'{rid}.json')['statistics']['unit'] if method=='L' else None
            if method=='R_P':
                start=time.monotonic();_,geo=d.base.source_geometry(field);assets.append(dict(id=rid,persistent_GS_bytes=d.old.scene_path(rid).stat().st_size,derived_source_geometry_bytes=d.gpath(rid).stat().st_size,first_logical_materialization_seconds=loadseconds,source_center_geometry_rebuild_seconds=time.monotonic()-start,source_constructor_cost='Existing source_geometry_control/native_training_coverage reports; inherited construction not repeated or free',source_tensor_bytes=field_bytes(field)))
            for q in (2,1):
                query=d.load(d.old.query_path(rid,q))
                for condition in ('full','missing'):
                    transfer_start=time.monotonic();image=query_image(d.device({k:query[k] for k in ('rgb','image_missing')}),condition);torch.cuda.synchronize();input_transfer_seconds=time.monotonic()-transfer_start
                    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.monotonic()
                    events={};mark=lambda k:events.setdefault(k,torch.cuda.Event(enable_timing=True)).record()
                    mark('start')
                    with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):patches=patch_tokens(model,image)
                    mark('encoded');torch.cuda.synchronize();render_s=None;geometry_s=None;working_input_bytes=0;handles=[]
                    if method!='Native':
                        gs=time.monotonic()
                        if method=='L':
                            v=local_from_camera(query,provided_unit)
                            ordinary=torch.cat((patches[0].float().repeat_interleave(2,0),v['ordinary'].cuda()),-1)
                        else:
                            K=query['K'].cuda().clone();K[:2]/=14;mark('raster_start');w,_=contribution_weights(fg,query['c2w'].cuda()[None],K[None],32,32);mark('raster_end');mass=w[0].sum((0,1)).cpu();render_s=events['raster_start'].elapsed_time(events['raster_end'])/1000
                            v=d.scene_view(field,g,query,mass);ordinary=torch.cat((field['features'].float().cuda(),v['ordinary'].cuda()),-1);del w
                        geometry_s=time.monotonic()-gs-(render_s or 0)
                        s=dict(patches=patches,ordinary=ordinary[None],scene_pos=v['scene_pos'][None].cuda(),key_mask=v['key_mask'][None].cuda(),position=torch.cat((v['common25'],v['P' if method=='L' else method[-1]]),-1)[None].cuda())
                        working_input_bytes=sum(t.numel()*t.element_size() for t in s.values())
                        handles+=[adapter.source.register_forward_pre_hook(lambda *_:mark('map_start')),adapter.source.register_forward_hook(lambda *_:mark('map_end')),adapter.position.register_forward_pre_hook(lambda *_:mark('pe_start')),adapter.position.register_forward_hook(lambda *_:mark('pe_end')),model.aggregator.frame_blocks[2].register_forward_pre_hook(lambda *_:mark('tail_start')),model.depth_head.register_forward_pre_hook(lambda *_:mark('head_start')),model.depth_head.register_forward_hook(lambda *_:mark('head_end'))]
                        mark('native_start')
                        with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):prediction=forward(model,s,adapter,method,False)
                    else:
                        mark('native_start')
                        with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):prediction=forward_read(model,dict(patches=patches,hw=(448,448)),checkpoint_blocks=False)
                    prediction_metric=metric_depth(prediction[0] if prediction.ndim==3 else prediction)
                    mark('end');torch.cuda.synchronize();wall=time.monotonic()-start
                    for h in handles:h.remove()
                    span=lambda a,b:events[a].elapsed_time(events[b])
                    rows.append(dict(method=method,id=rid,query=q,condition=condition,prepared_image_CPU_to_GPU_and_mask_seconds=input_transfer_seconds,full_wall_seconds=wall,full_GPU_event_ms=span('start','end'),encoder_ms=span('start','encoded'),actual_raster_seconds=render_s,query_geometry_CPU_and_transfer_seconds=geometry_s,source_mapping_ms=span('map_start','map_end') if method!='Native' else None,position_ms=span('pe_start','pe_end') if method!='Native' else None,early_native_exchange_ms=span('pe_end','tail_start') if method!='Native' else None,native_total_ms=span('native_start','end'),tail_and_DPT_ms=span('tail_start','end') if method!='Native' else None,DPT_ms=span('head_start','head_end') if method!='Native' else None,working_input_bytes=working_input_bytes,scene_hidden_pair_payload_bytes=2*2048*1024*2 if method!='Native' else 0,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),new_query_encoder=1,new_native=1,new_raster=int(method not in ('Native','L'))))
                    if method!='Native':del s,ordinary,v
                    del prediction,prediction_metric,patches,image
                    d.journal(d.OUT/'cost_calls.jsonl',rows[-1])
                    assert not d.resource()['stop_reasons']
            if method not in ('Native','L'):del fg,field,g
            torch.cuda.empty_cache()
        if method!='Native':del adapter;torch.cuda.empty_cache()
    summaries={}
    for method in ['Native',*d.ARMS]:
        selected=[r for r in rows if r['method']==method];summaries[method]={k:float(np.median([r[k] for r in selected if r[k] is not None])) if any(r[k] is not None for r in selected) else None for k in ['full_wall_seconds','full_GPU_event_ms','encoder_ms','actual_raster_seconds','query_geometry_CPU_and_transfer_seconds','source_mapping_ms','position_ms','early_native_exchange_ms','native_total_ms','tail_and_DPT_ms','DPT_ms','peak_allocated_bytes','peak_reserved_bytes','working_input_bytes','scene_hidden_pair_payload_bytes']}
    for a in assets:
        a['amortization']={str(n):dict(source_materialize_plus_geometry_per_query_seconds=(a['first_logical_materialization_seconds']+a['source_center_geometry_rebuild_seconds'])/n) for n in (1,4,8,32)}
    hist=d.read(d.ROOT/'results/gaussian_memory/source_geometry_control1/cost_description.json');coverage=d.read(d.ROOT/'results/gaussian_memory/native_training_coverage1/cost_description.json')['source_construction']
    inherited=dict(original_sensor_source_construction_seconds=hist['modes']['sensor']['prepared_source_to_field_seconds'],original_scope=hist['construction_scope'],added20_source_prediction_worker_seconds=coverage['source_prediction_worker_seconds'],added20_source_fit_worker_seconds=coverage['source_fit_worker_seconds'],added20_binding_encoding_render_cache_worker_seconds=coverage['binding_encoding_render_cache_worker_seconds'],scope='Historical measurements, no new source formation or encoding. Added20 binding/cache aggregate contains old query caches and is not a pure per-scene construction total.',references=['source_geometry_control1/cost_description.json','native_training_coverage1/cost_description.json'],N_amortized_original_sensor_partial_construction_seconds={str(n):[v/n for v in hist['modes']['sensor']['prepared_source_to_field_seconds']] for n in (1,4,8,32)},unmeasured_acquisition_and_raw_decode='unknown; never zero')
    d.write(d.OUT/'cost.json',dict(complete=True,query_cases=16,methods=8,new_native=128,new_encoder=128,new_raster=96,model_logical_load_seconds=model_load_seconds,adapter_loads=loaders,assets=assets,inherited_source_construction=inherited,rows=rows,summary=summaries,scope='Actual new image encoder / real gsplat per-slot alpha / fresh query geometry / native scene exchange / original tail and DPT. From prepared GPU RGB; image CPU-to-GPU/masking separately measured, raw decoding/acquisition excluded. All stages inside same128 complete calls. Events include dispatch gaps; CPU and transfer separately wall-timed. First logical asset loads do not flush OS caches and are not physically cold disk. Persistent original GS construction inherited, not zero-cost. Temporary scene evolution rebuilt every query, not amortized. N amortizes reusable source preprocessing only.'))

if __name__=='__main__':main()
