"""Round011 execution: formed immutable fields, unchanged camera-only measurement reader."""
import argparse,os,time,signal
import numpy as np
import torch
from experiments import native_feature_field_data as d
from experiments import camera_only_feature_run as r
from experiments.native_feature_field_source import fit_one
from models.camera_only_feature_reader import view_fields
r.d=d

def prepare_queries():
    for meta in d.metas():
        rid=meta['id'];marker=d.OUT/'preparation'/f'{rid}.json'
        if marker.exists():continue
        field=d.load(d.old.scene_path(rid));g,stats=d.source_geometry(field);d.tensor_save(d.fields_path(rid),g);field=d.to_device(field);g=d.to_device(g);rows=[]
        for q in d.contract()['evaluation_queries']:
            query=d.load(d.old.query_path(rid,q),'cuda');view=view_fields(g,query['c2w'],query['K']);d.tensor_save(d.view_path(rid,q),view);b,a=r.render_anchor(field,query,'formed_query_preparation');d.tensor_save(d.old.feature_path(rid,q),dict(b=b.cpu(),effective_alpha=a.cpu()))
            oldb,olda=d.parent.cached_anchor(rid,q);rows.append(dict(query=q,old_zero_fraction=float((olda<=1e-8).float().mean()),new_zero_fraction=float((a<=1e-8).float().mean()),old_zero_all=bool((olda<=1e-8).all()),new_zero_all=bool((a<=1e-8).all()),new_covered_patches=int((a>1e-8).sum()),alpha_mean=float(a.mean()),valid_keys=int(view['key_mask'].sum())))
        d.write(marker,dict(complete=True,row=d.ROW,id=rid,inputs={str(d.old.scene_path(rid)):dict(sha256_inherited=d.sha(d.old.scene_path(rid)),identity_type='new formed field actual SHA256')},statistics=stats,coverage=rows,formed_field_frozen=True,query_gradients_to_field=False));print(dict(prepared=d.ROW,id=rid),flush=True)

@torch.no_grad()
def baselines(model):
    for meta in d.metas():
        rid=meta['id']
        for q in d.contract()['evaluation_queries']:
            marker=d.OUT/'baselines'/f'{rid}_{q}.json'
            if marker.exists():continue
            original=d.read(d.parent.OUT/'baselines'/f'{rid}_{q}.json');rows={k:dict(original['methods'][k],inherited=True,authority=str(d.parent.OUT/'baselines'/f'{rid}_{q}.json')) for k in ('Native','Native_full')}
            b,_=d.cached_anchor(rid,q);patches=d.load(d.old.patch_path(rid,q,'missing'),'cuda')['patches'];d.charge('formed_RasterFill',native=1);depth=r.native(model,patches,b.cuda());path=d.CACHE/'depths'/'RasterFill'/f'{rid}_{q}.npy';d.numpy_save(path,depth.cpu().numpy());rows['RasterFill']=dict(path=str(path),sha256=d.sha(path),inherited=False)
            d.write(marker,dict(complete=True,id=rid,place=meta['place'],query=q,methods=rows))
        r.heartbeat('baselines',id=rid);assert not d.resources()['stop_reasons']

def main():
    p=argparse.ArgumentParser();p.add_argument('--row',required=True,choices=['C','J']);a=p.parse_args();d.context(a.row);torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.20);r.START=time.monotonic();r.WORKER=a.row;signal.signal(signal.SIGTERM,r.request_stop)
    import experiments.native_feature_field_source as src
    signal.signal(signal.SIGTERM,lambda *args:(r.request_stop(),src.request_stop()))
    assert d.sha(d.SCIENCE)==d.SCIENCE_SHA;assert d.read(d.BASEOUT/'source_targets.json')['complete']
    d.write(d.OUT/'configuration_lock.json',dict(science_contract=str(d.SCIENCE),science_sha256=d.SCIENCE_SHA,row=d.ROW,reader_source_sha256=d.sha(d.ROOT/'models/camera_only_feature_reader.py'),measurement_runtime_sha256=d.sha(d.ROOT/'experiments/camera_only_feature_run.py'),new_runtime_sha256=d.sha(d.ROOT/'experiments/native_feature_field_run.py'),source_fit_sha256=d.sha(d.ROOT/'experiments/native_feature_field_source.py'),rows_share_field_across_PE_seeds=True,query_supervision_does_not_update_field=True))
    for meta in d.metas():fit_one(meta)
    prepare_queries();r.preflight();r.permuted_indices();model=r.native_load();baselines(model);sources,samples=r.stage_training()
    for seed in d.SEEDS:
        for arm in d.ARMS:r.train(arm,seed,sources,samples);r.evaluate(model,arm,seed)
    del sources,samples;torch.cuda.empty_cache();r.costs(model);r.heartbeat('complete',normal_exit=True);print(dict(row=d.ROW,complete=True,seconds=time.monotonic()-r.START),flush=True)
if __name__=='__main__':main()
