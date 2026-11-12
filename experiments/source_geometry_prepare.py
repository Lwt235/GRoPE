"""Source feature lifting, two-fit interface check and prospective configuration lock."""
import argparse
import copy
import time

import numpy as np
import torch

from experiments.source_geometry_data import *
from experiments.source_geometry_build import CONFIG
from experiments.gaussian_memory_native_run import model_load
from models.gaussian_memory_coarse_pe import patch_tokens
from models.gaussian_scene_memory import lift_source_features,field_bytes,contribution_weights,SceneReader,center_ray_descriptor
from models.gaussian_scene_native import forward_read
from experiments.gaussian_scene_preflight import official_current,objective
from experiments.gaussian_scene_data import query_image

@torch.no_grad()
def bind(model,meta,mode):
    rid=meta['id'];folder=branch_cache(mode)/rid;marker=branch_out(mode)/'binding'/f'{rid}.json'
    if marker.exists():
        record=read(marker);assert all(sha256(path)==value for path,value in record['files'].items());return record
    started=time.monotonic();source=load(source_path(rid),'cuda');field=load(folder/'coarse_geometry.pt','cuda')
    shared=CACHE/'inputs'/rid/'source_features.pt'
    if shared.exists():features=load(shared,'cuda')['features']
    else:
        with torch.autocast('cuda',dtype=torch.bfloat16):
            features=torch.cat([patch_tokens(model,image.float()/255) for image in source['rgb']]).reshape(4,32,32,1024)
        save_tensor(shared,dict(features=features))
    K=source['K'].clone();K[:,:2]/=14
    field['features'],field['mass'],field['has_content']=lift_source_features(field,features,source['c2w'],K)
    field['source_ids']=torch.tensor(SOURCES,dtype=torch.int32,device='cuda')
    assert field['features'].shape==(2048,1024) and torch.isfinite(field['features']).all()
    assert bool(field['has_content'].any())
    path=folder/'scene_memory.pt';save_tensor(path,field_to_cpu(field));restored=load(path,'cuda')
    assert set(restored)==set(field) and all(torch.equal(v,restored[k]) for k,v in field.items())
    record=dict(complete=True,id=rid,mode=mode,source_only=True,queries_used=False,scene_bytes=field_bytes(field),
        content_valid=int(field['has_content'].sum()),tensor_save_reload_exact=True,
        source_pair_sha256=sha256(source_path(rid)),geometry_sha256=sha256(folder/'coarse_geometry.pt'),
        source_features_sha256=sha256(shared),files={str(path):sha256(path)},binding_seconds=time.monotonic()-started)
    write_json(marker,record);print(dict(binding_complete=rid,mode=mode,valid=record['content_valid']),flush=True);return record

def field_to_cpu(field):return {k:v.detach().cpu() for k,v in field.items()}

@torch.no_grad()
def cache(meta,mode):
    p=plan(mode);rid=meta['id'];marker=branch_out(mode)/'query_cache'/f'{rid}.json'
    if marker.exists():
        r=read(marker);assert r['plan_sha256']==sha256(branch_out(mode)/'plan.json')
        assert all(sha256(path)==value for path,value in r['files'].items());return
    fieldpath=branch_cache(mode)/rid/'scene_memory.pt';digest=sha256(fieldpath);field=load(fieldpath,'cuda');files={}
    for q in QUERIES:
        query=load(query_path(rid,q),'cuda');K=query['K'].clone();K[:2]/=14
        weights,alpha=contribution_weights(field,query['c2w'][None],K[None],32,32)
        path=geometry_path(mode,rid,q);save_tensor(path,dict(weights=weights[0].reshape(1024,-1).float(),alpha=alpha[0]))
        files[str(path)]=sha256(path)
        for condition in CONDITIONS:
            parent=patch_path(rid,q,condition);assert parent.exists();files[str(parent)]=sha256(parent)
    assert sha256(fieldpath)==digest
    write_json(marker,dict(complete=True,id=rid,mode=mode,plan_sha256=sha256(branch_out(mode)/'plan.json'),
        files=files,saved_scene_sha256=digest,source_state_unchanged=True,four_queries_same_saved_state=True,
        query_cache_contains_history_prefix=False,current_patches_reused_from_parent=True))


def preflight(model):
    records=[];p=parent_plan()
    for meta in p['training_catalog'][:2]:
        rid=meta['id'];query=load(query_path(rid,2),'cuda');source=load(source_path(rid))
        a=load(branch_cache('pred')/rid/'selected_indices.pt')['indices']
        b=load(branch_cache('sensor')/rid/'selected_indices.pt')['indices'];assert torch.equal(a,b)
        image=query_image(query,'missing');reference=official_current(model,image)
        for mode in MODES:
            field=load(branch_cache(mode)/rid/'scene_memory.pt','cuda')
            for kind in KINDS:
                torch.manual_seed(SEEDS[0]);reader=SceneReader(center_pe=kind=='P').cuda()
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                    patches=patch_tokens(model,image);K=query['K'].clone();K[:2]/=14
                    if kind=='P':geometry=center_ray_descriptor(field['means'],query['c2w'],K,32,32,field['unit'])
                    else:
                        w,_=contribution_weights(field,query['c2w'][None],K[None],32,32);geometry=w[0].reshape(1024,-1)
                    sample=dict(patches=patches.detach(),features=field['features'],valid=field['has_content'],geometry=geometry,hw=(448,448))
                    initial=forward_read(model,sample,reader,kind,checkpoint_blocks=False)
                maximum=float((initial-reference).abs().max());assert maximum<1e-4
                optimizer=torch.optim.AdamW(reader.parameters(),lr=1e-4,weight_decay=.01);grads=[]
                for step in range(2):
                    trace={};optimizer.zero_grad(set_to_none=True)
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        prediction=forward_read(model,sample,reader,kind,trace=trace);loss=objective(prediction,query['target'],query['valid'])
                    loss.backward();assert all(trace.get(str(i),0)>0 for i in [0,6,12,18])
                    gradient={n:float(t.grad.norm()) for n,t in reader.named_parameters() if t.grad is not None}
                    assert all(np.isfinite(v) for v in gradient.values()) and gradient['output.weight']>0
                    if step:assert gradient['query.weight']>0 and (kind!='P' or gradient['center_pe.2.weight']>0)
                    torch.nn.utils.clip_grad_norm_(reader.parameters(),5.);optimizer.step();grads.append(dict(loss=float(loss),gradients=gradient,read_layers=trace))
                records.append(dict(id=rid,mode=mode,kind=kind,initial_native_max_abs=maximum,selected_source_pixels_identical=True,steps=grads))
                del reader,optimizer,sample,prediction,loss,initial,geometry
            del field
        del query,reference;torch.cuda.empty_cache()
    result=dict(complete=True,records=records,pilot_weights_discarded=True,no_development_queries_used=True,
                scope='Two old training clips only; interface checks, not downstream utility')
    write_json(OUT/'preflight.json',result);return result


def freeze():
    path=OUT/'plan.json'
    if path.exists():return plan()
    assert read(OUT/'preflight.json')['complete']
    p=copy.deepcopy(parent_plan());p.update(experiment='source_geometry_control1',kinds=list(KINDS),geometries=list(MODES),
        source_fit=CONFIG,expected_adapted_predictions=1152,expected_intervention_predictions=1728,
        source_state_bytes=read(branch_out('pred')/'binding'/f'{p["training_catalog"][0]["id"]}.json')['scene_bytes'],
        primary_hypothesis='Source geometry and induced binding reliability changes the G-versus-P increment',
        main_endpoint='missing/roi/centered_log_mse; interaction (P_sensor-G_sensor)-(P_pred-G_pred)',
        competing_explanations=['center sufficient','fixed projection rule limits','receiver/content mismatch','source quality does not improve'],
        source_quality_input='Predicted or source sensor depth in the same known-source-camera backprojection builder',
        joint_source_validity_privilege=True,development_tuning=False,new_confirmation=False,
        parent_scene_plan_sha256=sha256(PARENT_OUT/'plan.json'),
        temporary_scope=['extra source sensor data/joint source validity for diagnostic','given cameras','static sources','frozen native model and DPT','old data'],
        decisions='Improved scene and PE interaction: quality matters; content-only improvement: no projection PE promotion; improved scene but no output: receiver/interface remains; no scene improvement: manipulation ineffective. No automatic successor.')
    p['interventions']['model']='G in each geometry mode, all3seeds'
    write_json(path,p)
    for mode in MODES:write_json(branch_out(mode)/'plan.json',dict(p,geometry_mode=mode))
    names=['experiments/source_geometry_data.py','experiments/source_geometry_build.py','experiments/source_geometry_prepare.py',
           'experiments/source_geometry_train.py','models/gaussian_scene_memory.py','models/gaussian_scene_native.py',
           'experiments/gaussian_scene_train.py','experiments/gaussian_scene_config.py']
    write_json(OUT/'configuration_lock.json',dict(plan_sha256=sha256(path),branch_plan_sha256={m:sha256(branch_out(m)/'plan.json') for m in MODES},
        source_sha256={n:sha256(ROOT/n) for n in names},preflight_sha256=sha256(OUT/'preflight.json'),
        fixed_before_development_outputs=True,formal_training_started=False))
    print(dict(configuration_locked=True,arms=4,fits=12),flush=True);return p


def main():
    a=argparse.ArgumentParser();a.add_argument('--pilot',action='store_true');args=a.parse_args()
    torch.set_num_threads(4);p=parent_plan();started=time.monotonic();model=model_load(p)
    torch.cuda.synchronize();load_seconds=time.monotonic()-started
    rows=p['training_catalog'][:2] if args.pilot else p['training_catalog']+p['catalog']
    for meta in rows:
        for mode in MODES:bind(model,meta,mode)
    if args.pilot:
        preflight(model);freeze();write_json(OUT/'pilot_runtime.json',dict(complete=True,cold_native_load_seconds=load_seconds,peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30))
    else:
        for meta in rows:
            for mode in MODES:cache(meta,mode)
        write_json(OUT/'all_fields_and_caches.json',dict(complete=True,fields=48,queries_per_field=4,shared_current_patch_files=192))

if __name__=='__main__':main()
