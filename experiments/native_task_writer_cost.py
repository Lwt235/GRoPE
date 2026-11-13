"""Finite full-query measurements, source formation and cache amortization."""
import time
import torch
import numpy as np
from experiments import native_task_writer_data as d
from experiments import native_task_writer_run as r
from experiments.native_receiver_runtime import model_load
from experiments.native_receiver_data import core_load,restore_native
from experiments.gaussian_scene_data import query_image
from models.gaussian_memory_coarse_pe import patch_tokens
from models.gaussian_scene_native import forward_read as native_forward
from models.gaussian_scene_rendered import RenderedFeatureFusion,forward_read as rendered_forward
from models.gaussian_scene_memory import rasterize_field,field_bytes
from models.gaussian_scene_parameter_pe import forward_read
from models.native_task_writer import read_current

def span(fn):
    torch.cuda.synchronize();start=time.monotonic();value=fn();torch.cuda.synchronize();return value,time.monotonic()-start

@torch.no_grad()
def main():
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.80);r.WORKER='cost';assert not d.resource(True)['stop_reasons'];start=time.monotonic();model=model_load(d.plan());model.aggregator.patch_embed.cuda();torch.cuda.synchronize();loadseconds=time.monotonic()-start
    cases=[(d.plan()['training_catalog'][0]['id'],2,'missing'),(d.plan()['catalog'][0]['id'],1,'full')];records=[];formations=[];assets=[];loaders=[]
    for method in ['Native','Local','Render',*d.ARMS]:
        start=time.monotonic();mods=None;fusion=None
        if method=='Native':restore_native(model,d.plan())
        elif method in ('Local','Render'):
            st=d.load(d.old.checkpoint_path(method,d.SEEDS[0]));core_load(model,st['native']);fusion=RenderedFeatureFusion().cuda();fusion.load_state_dict(st['model']);del st
        else:
            st=d.load(d.final(method,d.SEEDS[0]));mods=r.modules(d.SEEDS[0]);r.restore(st,model,mods);del st
        torch.cuda.synchronize();loaders.append(dict(method=method,checkpoint_logical_load_seconds=time.monotonic()-start))
        for rid,q,c in cases:
            query=d.load(d.old.query_path(rid,q));base,base_transfer=span(lambda:d.load(d.old.scene_path(rid),'cuda'))
            field=base
            if method in d.ARMS:
                field,field_load=span(lambda:d.load(d.fieldpath(method,d.SEEDS[0],rid),'cuda'))
                for rep in range(5):
                    formed,sec=span(lambda:mods[2](base,method[0]));d.charge('cost','source_formation_cost',source_formations=1);formations.append(dict(method=method,id=rid,repeat=rep,warmup=rep==0,seconds=sec,field_tensor_bytes=field_bytes(formed)));del formed
                temp=d.CACHE/'cost_serialization'/f'{method}_{rid}.pt';cpu_field,CPU_transfer=span(lambda:d.cpu(field));_,save_seconds=span(lambda:d.save(temp,cpu_field));reloaded,reload_seconds=span(lambda:d.load(temp,'cuda'));assert all(torch.equal(reloaded[k],field[k]) for k in ('features','means','scales','rotations','opacities'));serialization_bytes=temp.stat().st_size;temp.unlink();del reloaded,cpu_field
                fmeta=d.read(d.OUT/'formation'/f'{method}_{d.SEEDS[0]}_{rid}.json');assets.append(dict(method=method,id=rid,original_source_state_bytes=d.old.scene_path(rid).stat().st_size,saved_field_bytes=d.fieldpath(method,d.SEEDS[0],rid).stat().st_size,serialization_CPU_transfer_seconds=CPU_transfer,serialization_atomic_fsync_seconds=save_seconds,serialization_reload_and_GPU_seconds=reload_seconds,serialization_bytes=serialization_bytes,first_logical_source_load_seconds=base_transfer,first_logical_saved_field_load_seconds=field_load,persisted_formation_record=fmeta,upstream_source_formation='Inherited sensor/coarsening/source-binding construction; referenced source_geometry_control1/native_training_coverage1 reports, not repeated or free'))
            for repeat in range(6):
                qq=d.device(query);image,transfer=span(lambda:query_image(qq,c));torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();begin=time.monotonic()
                def encode():
                    with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):return patch_tokens(model,image)
                patches,encoder=span(encode);s=dict(patches=patches,hw=(448,448));render=0.;geometry=0.;fusiontime=0.
                if method in d.ARMS:
                    view,geometry=span(lambda:read_current(field,qq,method[-1]));s.update(view);d.charge('cost','query_geometry',raster=1)
                    pred,native=span(lambda:forward_read(model,s,mods[0],mods[1],checkpoint_blocks=False))
                elif method in ('Local','Render'):
                    if method=='Local':s['extra']=patches.float().mean(1,keepdim=True).expand_as(patches)
                    else:
                        K=qq['K'].clone();K[:2]/=14
                        rendered,render=span(lambda:rasterize_field(field,qq['c2w'][None],K[None],32,32,field['features'].float()*field['has_content'][:,None]))
                        s['extra']=rendered[0][0].reshape(1024,1024);d.charge('cost','query_render',raster=1)
                    def read():
                        with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):return rendered_forward(model,s,fusion,method,checkpoint_blocks=False)
                    pred,native=span(read)
                else:
                    def read_native():
                        with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):return native_forward(model,s,checkpoint_blocks=False)
                    pred,native=span(read_native)
                metric,_=span(lambda:r.metric_depth(pred));torch.cuda.synchronize();whole=time.monotonic()-begin;assert torch.isfinite(metric).all()
                d.charge('cost','full_query',native=1,query_encoder=1)
                records.append(dict(method=method,id=rid,query=q,condition=c,repeat=repeat,warmup=repeat==0,full_wall_seconds=whole,input_GPU_mask_seconds=transfer,encoder_seconds=encoder,actual_raster_seconds=render if method=='Render' else None,current_field_raster_geometry_seconds=geometry if method in d.ARMS else None,native_fusion_PE_DPT_seconds=native,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),query_writer_calls=0))
                del pred,metric,s,patches,image
                if method in d.ARMS:del view
            del field,base;torch.cuda.empty_cache()
        del mods,fusion;torch.cuda.empty_cache()
    summary={}
    for method in ['Native','Local','Render',*d.ARMS]:
        rr=[x for x in records if x['method']==method and not x['warmup']];ss={}
        for key in ('full_wall_seconds','encoder_seconds','actual_raster_seconds','current_field_raster_geometry_seconds','native_fusion_PE_DPT_seconds','peak_allocated_bytes','peak_reserved_bytes'):
            values=[x[key] for x in rr if x[key] is not None];ss[key]=float(np.median(values)) if values else None
        ff=[x['seconds'] for x in formations if x['method']==method and not x['warmup']];ss['source_writer_median_seconds']=float(np.median(ff)) if ff else 0.
        aa=[x for x in assets if x['method']==method];ss['serialization_CPU_and_atomic_save_median_seconds']=float(np.median([x['serialization_CPU_transfer_seconds']+x['serialization_atomic_fsync_seconds'] for x in aa])) if aa else 0.
        ss['field_reload_median_seconds']=float(np.median([x['first_logical_saved_field_load_seconds'] for x in aa])) if aa else 0.
        ss['amortized_new_writer_and_reload_plus_query_seconds']={str(n):ss['full_wall_seconds']+(ss['source_writer_median_seconds']+ss['serialization_CPU_and_atomic_save_median_seconds']+ss['field_reload_median_seconds'])/n for n in (1,4,8,32)}
        ss['amortized_with_historical_sensor_construction_reference_seconds']={str(n):ss['amortized_new_writer_and_reload_plus_query_seconds'][str(n)]+3.1925/n for n in (1,4,8,32)} if method in d.ARMS or method=='Render' else None
        ss['historical_sensor_construction_reference']=dict(seconds=[3.230,3.155],authority='results/gaussian_memory/source_geometry_control1/report.md',exclusions='AnySplat prediction generation, sensor acquisition, JPEG preprocessing and model cold load; different historical timing conditions, not end-to-end RGB cost')
        ss['amortized_scope']='Incremental shared writer, CPU transfer, atomic save and field reload; upstream source construction and model loading separate, never free. Persist save/CPU transfer costs in actual formation records.';summary[method]=ss
    assert len(records)==108 and len(formations)==60
    d.write(d.OUT/'cost.json',dict(complete=True,model_load_seconds=loadseconds,loaders=loaders,records=records,formation_measurements=formations,assets=assets,summary=summary,new_native=108,new_query_encoder=108,new_source_formations=60,new_actual_raster=84,full_call_scope='Prepared RGB/mask and given cameras; real query encoder, actual current live raster/PE, original native/DPT. No cached-view timing. Warmup1 plus5 retained/case, shared GPU, OS cache not flushed; not exclusive timing.',upstream_cost_references=['results/gaussian_memory/source_geometry_control1/report.md','results/gaussian_memory/native_training_coverage1/report.md']))
if __name__=='__main__':main()
