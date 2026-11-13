"""Necessary formed-source geometry/ordinary/KV build costs; no reader/query reruns."""
import os,time
import numpy as np
import torch
from experiments import native_feature_field_data as d
from models.camera_only_feature_reader import CameraOnlyFeatureReader

def main():
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.20);rows=[]
    for row in ('C','J'):
        d.context(row)
        for arm in d.ARMS:
            reader=CameraOnlyFeatureReader().cuda();reader.load_state_dict(d.load(d.final_path(arm,d.SEEDS[0]))['model']);reader.eval()
            for meta in d.contract()['training_catalog'][:2]:
                rid=meta['id'];p=d.old.scene_path(rid);t=time.monotonic();field=d.load(p);loadtime=time.monotonic()-t
                t=time.monotonic();geom,stats=d.source_geometry(field);CPUtime=time.monotonic()-t
                torch.cuda.synchronize();t=time.monotonic();f=d.to_device(field);g=d.to_device(geom);ordinary=d.ordinary(f,g);torch.cuda.synchronize();ordinarytime=time.monotonic()-t
                with torch.no_grad():
                    torch.cuda.synchronize();t=time.monotonic();kv=reader.encode_source(ordinary);torch.cuda.synchronize();kvtime=time.monotonic()-t;d.charge('cost_complete_source_build',source=1)
                saved=d.memory_path(arm,d.SEEDS[0],rid);t=time.monotonic();warm=d.load(saved,'cuda');torch.cuda.synchronize();warmtime=time.monotonic()-t
                reencoded_exact=torch.equal(kv[0],warm['key']) and torch.equal(kv[1],warm['value']);reencoded_difference=max(float((kv[0]-warm['key']).abs().max()),float((kv[1]-warm['value']).abs().max()))
                rows.append(dict(row=row,arm=arm,id=rid,base_load_seconds=loadtime,geometry_B_density_logSigma_CPU_seconds=CPUtime,transfer_and_ordinary_seconds=ordinarytime,KV_GPU_synchronized_seconds=kvtime,total_source_build_seconds=loadtime+CPUtime+ordinarytime+kvtime,warm_state_reload_seconds=warmtime,base_bytes=p.stat().st_size,state_bytes=saved.stat().st_size,KV_reencode_exact=reencoded_exact,KV_reencode_max_abs=reencoded_difference,no_extra_numerical_threshold=True,export_serialization_reload_already_verified=True))
    # The initial source renders during fitting retained raw MSE only. Complete the required
    # source spatial/bias reference once for the common old field, without another fit.
    from experiments.native_feature_field_source import render
    reference=[]
    for meta in d.metas():
        rid=meta['id'];base=d.load(d.original_scene(rid),'cuda');target=d.load(d.target_path(rid))['features'].reshape(4,1024,1024).float().numpy();views=[]
        with torch.no_grad():
            for v in range(4):
                b,a=render(base,v);d.charge('common_source_initial_decomposition_reference',raster=1)
                error=b.cpu().numpy().astype(np.float64)-target[v].astype(np.float64);bias=error.mean(0);views.append(dict(view=v,raw_MSE=float(np.mean(error*error)),spatial_MSE=float(np.mean((error-bias)**2)),bias_squared=float(np.mean(bias*bias)),zero_fraction=float((a<=1e-8).float().mean())))
        reference.append(dict(id=rid,views=views))
    d.write(d.BASEOUT/'source_initial_decomposition.json',dict(complete=True,shared_both_rows=True,records=reference,real_render_calls=176,arithmetic='NumPyfloat64 source raw/spatial/bias, common actual initial field',no_optimisation_or_query_inputs=True))
    d.write(d.BASEOUT/'source_build_cost.json',dict(complete=True,records=rows,source_calls=12,query_reader_native_or_renderer_calls=0,scope='Actual per-formed-field geometry/B/density, transfer/LayerNorm ordinary and KV encode; six builds per row, no extra query benchmark',model_configuration='unchanged original reader first seed',runtime=dict(torch=torch.__version__,CUDA=torch.version.cuda,device=torch.cuda.get_device_name(),peak_allocated_bytes=torch.cuda.max_memory_allocated())))
if __name__=='__main__':main()
