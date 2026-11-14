"""Prescribed fresh complete-query costs, never cached-kernel timing as fresh view."""
import time
import torch
import numpy as np
from experiments import native_read_address_data as d
from experiments import native_read_address_run as run
from experiments.native_receiver_runtime import model_load
from experiments.native_receiver_data import core_load
from models.native_read_address import AddressBias,weights,content_view,center_kernel,centered_kernel,sphere_field,forward_read
from models.gaussian_scene_rendered import RenderedFeatureFusion,forward_read as rendered_forward
from models.gaussian_scene_native import forward_read as native_forward
from models.gaussian_memory_coarse_pe import patch_tokens
from models.gaussian_scene_memory import rasterize_field
from experiments.gaussian_scene_data import query_image
from experiments.gaussian_scene_preflight import metric_depth

def span(fn):
    torch.cuda.synchronize();begin=time.monotonic();x=fn();torch.cuda.synchronize();return x,time.monotonic()-begin

@torch.no_grad()
def main():
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.85);assert not (d.OUT/'cost.json').exists() and not (d.OUT/'calls/cost.jsonl').exists(),'Do not repeat any cost invocation'
    model,loadseconds=span(lambda:model_load(d.plan()));_,encoderload=span(lambda:model.aggregator.patch_embed.cuda());model.requires_grad_(False);records=[];sources=[];assets=[];loaders=[];seed=d.SEEDS[0];cases=[(d.plan()['training_catalog'][0]['id'],2,'missing'),(d.plan()['catalog'][0]['id'],1,'full')];mods=None
    def stop_if_requested():
        if run.STOP:
            d.write(d.OUT/'cost_partial.json',dict(complete=False,records=records,source_preparations=sources,assets=assets,loaders=loaders,utc=d.now()));raise SystemExit(75)
    for method in ('Native','Local','Render','B','A','P','S'):
        stop_if_requested();begin=time.monotonic();address=None;fusion=None
        if method in ('Local','Render'):
            st=d.load(d.parent.old.checkpoint_path(method,seed));core_load(model,st['native']);fusion=RenderedFeatureFusion().cuda().eval().requires_grad_(False);fusion.load_state_dict(st['model']);del st
        elif method=='B':mods=run.base(model,seed)
        elif method in d.ARMS:
            address=AddressBias().cuda();st=d.load(d.final(method,seed));address.load_state_dict(st['gamma']);address.requires_grad_(False)
        torch.cuda.synchronize();loaders.append(dict(method=method,logical_checkpoint_load_seconds=time.monotonic()-begin,reuses_frozen_B_loaded_once=method in d.ARMS));stop_if_requested()
        for rid,q,c in cases:
            query=d.device(d.load(d.parent.old.query_path(rid,q)));field=None;C=None;source_seconds=0.
            expected=None
            if method in ('B',*d.ARMS):
                reference_path=d.parent.CACHE/'predictions'/f'C_P_{seed}'/f'{rid}.pt' if method=='B' else d.predpath(method,seed,rid)
                reference_payload=d.load(reference_path);reference_index=[tuple(k) for k in reference_payload['keys']].index((rid,q,c));expected=reference_payload['depth_metric'][reference_index];del reference_payload
            if method not in ('Native','Local'):
                field,source_seconds=span(lambda:d.load(d.parent.fieldpath('C_P',seed,rid),'cuda')) if method in ('B',*d.ARMS) else span(lambda:d.load(d.parent.old.scene_path(rid),'cuda'))
            if method=='P':
                d.charge('cost','source_Cmu',center_kernel=1);(C,cmeta),center_seconds=span(lambda:center_kernel(field));sources.append(dict(method=method,id=rid,center_kernel_seconds=center_seconds,metadata=cmeta))
            for repeat in range(5):
                stop_if_requested();image,transfer=span(lambda:query_image(query,c));d.charge('cost','full_query',native=1,query_encoder=1);torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();begin=time.monotonic()
                def encode():
                    with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):return patch_tokens(model,image)
                patches,encoder=span(encode);s=dict(patches=patches,hw=(448,448));build=0.;raster_seconds=0.;kernel_seconds=0.
                if method in ('B',*d.ARMS):
                    d.charge('cost','fresh_actual_W',raster=1);W,raster_seconds=span(lambda:weights(field,query));view,content_seconds=span(lambda:content_view(field,query,W));s.update(view);del view;build=raster_seconds+content_seconds
                    if method in d.ARMS:
                        WS=None;shadow_seconds=0.
                        if method=='S':d.charge('cost','fresh_shadow_W',raster=1);WS,shadow_seconds=span(lambda:weights(sphere_field(field),query))
                        s['kernel'],kernel_seconds=span(lambda:centered_kernel(W,method,C,WS));build+=shadow_seconds+kernel_seconds
                        pred,native=span(lambda:forward_read(model,s,*mods,address,checkpoint_blocks=False))
                        if WS is not None:del WS
                    else:
                        pred,native=span(lambda:run.original_forward(model,s,*mods,checkpoint_blocks=False))
                    del W
                elif method in ('Local','Render'):
                    if method=='Local':s['extra']=patches.float().mean(1,keepdim=True).expand_as(patches)
                    else:
                        K=query['K'].clone();K[:2]/=14;d.charge('cost','reference_render',raster=1);rendered,raster_seconds=span(lambda:rasterize_field(field,query['c2w'][None],K[None],32,32,field['features'].float()*field['has_content'][:,None]));s['extra']=rendered[0][0].reshape(1024,1024);del rendered;build=raster_seconds
                    def read():
                        with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):return rendered_forward(model,s,fusion,method,checkpoint_blocks=False)
                    pred,native=span(read)
                else:
                    def read_native():
                        with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):return native_forward(model,s,checkpoint_blocks=False)
                    pred,native=span(read_native)
                pred,resize=span(lambda:metric_depth(pred));pred,outputtransfer=span(lambda:pred.float().cpu());assert torch.isfinite(pred).all();torch.cuda.synchronize();wall=time.monotonic()-begin
                replay_error=float((pred-expected).abs().max()) if expected is not None else None
                records.append(dict(method=method,id=rid,query=q,condition=c,repeat=repeat,warmup=repeat==0,full_wall_seconds=wall,input_GPU_mask_seconds=transfer,encoder_seconds=encoder,fresh_actual_W_seconds=raster_seconds,new_view_build_seconds=build,kernel_seconds=kernel_seconds,native_fusion_oldPE_bias_DPT_seconds=native,resize_seconds=resize,output_GPU_to_CPU_seconds=outputtransfer,fresh_vs_saved_cached_metric_depth_max_abs=replay_error,query_writer_calls=0,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved()));del pred,patches,image
                stop_if_requested()
            if method in ('B',*d.ARMS):
                cache={k:s[k] for k in ('extra','position','alpha')};
                if method in d.ARMS:cache['kernel']=s['kernel']
                cpu_cache,toCPU=span(lambda:d.cpu(cache));path=d.CACHE/'cost_temporary'/f'{method}_{rid}.pt';_,saved=span(lambda:d.save(path,cpu_cache));reload,reloaded=span(lambda:d.load(path,'cuda'));assert all(torch.equal(reload[k],cache[k]) for k in cache);n=path.stat().st_size;path.unlink();assets.append(dict(method=method,id=rid,original_field_path=str(d.parent.fieldpath('C_P',seed,rid)),original_field_bytes=d.parent.fieldpath('C_P',seed,rid).stat().st_size,first_logical_field_load_GPU_seconds=source_seconds,new_single_arm_view_cache_bytes=n,cache_CPU_transfer_seconds=toCPU,cache_atomic_save_fsync_seconds=saved,cache_logical_reload_GPU_seconds=reloaded));del cache,cpu_cache,reload
            del s,field,C;torch.cuda.empty_cache()
        del address,fusion;torch.cuda.empty_cache()
    assert len(records)==70 and sum(x['query_writer_calls'] for x in records)==0
    old=d.read(d.parent.OUT/'cost.json');reference=old['summary']['C_P'];summary={}
    for method in ('Native','Local','Render','B','A','P','S'):
        rr=[r for r in records if r['method']==method and not r['warmup']];v={k:float(np.median([r[k] for r in rr])) for k in ('full_wall_seconds','encoder_seconds','fresh_actual_W_seconds','new_view_build_seconds','kernel_seconds','native_fusion_oldPE_bias_DPT_seconds','peak_allocated_bytes')};source=[x['center_kernel_seconds'] for x in sources if x['method']==method];v['source_Cmu_median_seconds']=float(np.median(source)) if source else 0.
        aa=[x for x in assets if x['method']==method];v['field_logical_load_GPU_median_seconds']=float(np.median([x['first_logical_field_load_GPU_seconds'] for x in aa])) if aa else 0.
        v['view_cache_CPU_save_reload_median_seconds']=float(np.median([x['cache_CPU_transfer_seconds']+x['cache_atomic_save_fsync_seconds']+x['cache_logical_reload_GPU_seconds'] for x in aa])) if aa else 0.
        added=reference['source_writer_median_seconds']+reference['serialization_CPU_and_atomic_save_median_seconds']+v['field_logical_load_GPU_median_seconds'] if method in ('B',*d.ARMS) else 0.
        v['amortized_fresh_view_query_with_incremental_formation_reference_seconds']={str(n):v['full_wall_seconds']+(v['source_Cmu_median_seconds']+added)/n for n in (1,4,8,32)}
        v['derived_same_camera_cache_reuse_seconds']={str(n):v['full_wall_seconds']-v['new_view_build_seconds']+(v['source_Cmu_median_seconds']+added+v['new_view_build_seconds']+v['view_cache_CPU_save_reload_median_seconds'])/n for n in (1,4,8,32)} if aa else None
        v['amortized_with_historical_sensor_reference_seconds']={str(n):value+3.1925/int(n) for n,value in v['amortized_fresh_view_query_with_incremental_formation_reference_seconds'].items()} if method in ('B',*d.ARMS) else None;summary[method]=v
    d.write(d.OUT/'cost.json',dict(complete=True,model_cold_load_seconds=loadseconds+encoderload,receiver_cold_load_seconds=loadseconds,encoder_GPU_load_seconds=encoderload,loaders=loaders,records=records,source_preparations=sources,view_cache_assets=assets,summary=summary,new_native=70,new_query_encoder=70,new_actual_raster=60,new_source_writer_calls=0,new_Cmu_builds=2,full_query_scope='Fresh RGB encoder, fresh actualW/commoncontent and new kernel (S additionally shadowW), original frozen native/DPT, resize and outputGPUtoCPU transfer. Input image/mask transfer before begin separately measured. No precomputed query W/kernel used by model in these70full queries; inherited metric-depth loaded only for independent post-timing replay comparison; instrumented synchronized wall includes log/segment overhead. Shared host; no hardware exclusivity or physical-cold-disk claim.',amortization_scope='N distinct fresh views keeps newW/kernel build in every query; P source-only Cmu, referenced prior writer/persistence and current logical fieldGPUload divided byN. View-cache transfer/fsync/reload actual records never free; same-camera cacheN curve is explicitly derived from measured segments without new modelcalls, unlike measured fresh-view headline. No new writer execution or reconstructed field.',formation_dependency=dict(reference_path=str(d.parent.OUT/'cost.json'),reference_sha256=d.sha(d.parent.OUT/'cost.json'),method='C_P',original_reference_summary=reference,historical_sensor_seconds=[3.230,3.155],historical_reference=str(d.ROOT/'results/gaussian_memory/source_geometry_control1/report.md'),exclusions='AnySplat prediction generation, sensor acquisition, JPEG preprocessing and model cold loading; historical conditions not an end-to-end RGB latency')))
    print(dict(cost_complete=True,native=70,raster=60),flush=True)
if __name__=='__main__':main()
