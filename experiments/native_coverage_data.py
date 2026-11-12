"""A single fixed-compute 12-to-32-place coverage contrast, isolated storage."""
import copy
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from experiments.native_receiver_data import ROOT,DATA,OUT as BASE,FROZEN
from experiments.rendered_feature_data import (SOURCE_CACHE,patch_path as base_patch,
    feature_path as base_feature,StagedSamples)
from experiments.native_receiver_view_data import (OUT as VIEW_OUT,CACHE as VIEW_CACHE,
    query_path as view_query,patch_path as view_patch,feature_path as view_feature)
from experiments.source_geometry_data import load,read,sha256,save_tensor,write_json
from experiments.gaussian_scene_data import (query_path as base_query,transformed_rgb,
    transformed_target,transformed_intrinsics)
from experiments.gaussian_memory_incremental_data import depth_source

OUT=ROOT/'results/gaussian_memory/native_training_coverage1'
CACHE=DATA/'native_training_coverage_v1'
RAM=Path('/dev/shm')/f'liwt_native_training_coverage1_{os.getuid()}'
KINDS=('Local','Render');SEEDS=(20261110,20261111,20261112)
QUERIES=(2,5,8,11);HELDOUT=(1,4,7,10);EVAL_QUERIES=QUERIES+HELDOUT
CONDITIONS=('full','missing')


def plan():return read(OUT/'plan.json')
def is_new(rid):return rid in {m['id'] for m in plan()['added_training_catalog']}
def source_rgb_path(rid):return CACHE/'sources'/rid/'source_input.pt'
def source_pair_path(rid):return CACHE/'sources'/rid/'source_pair.pt'
def scene_path(rid):return CACHE/'scenes'/rid/'scene_memory.pt' if is_new(rid) else SOURCE_CACHE/rid/'scene_memory.pt'
def query_path(rid,q):return CACHE/rid/f'query_{q}.pt' if is_new(rid) else (base_query(rid,q) if q in QUERIES else view_query(rid,q))
def patch_path(rid,q,c):return CACHE/rid/f'patch_{q}_{c}.pt' if is_new(rid) else (base_patch(rid,q,c) if q in QUERIES else view_patch(rid,q,c))
def feature_path(rid,q):return CACHE/rid/f'render_{q}.pt' if is_new(rid) else (base_feature(rid,q) if q in QUERIES else view_feature(rid,q))
def checkpoint_path(kind,seed):return CACHE/'finals'/f'{kind}_{seed}'/'step_2000.pt'
def resume_path(kind,seed):return RAM/f'{kind}_{seed}'/'latest.pt'


def initialize():
    if (OUT/'plan.json').exists():return plan()
    p=copy.deepcopy(read(BASE/'plan.json'));inventory=read(OUT/'inventory.json')
    original=p['training_catalog'];added=inventory['selected']
    p.update(experiment='native_training_coverage1',training_catalog=original+added,
        original_training_catalog=original,added_training_catalog=added,kinds=list(KINDS),
        evaluation_queries=list(EVAL_QUERIES),heldout_queries=list(HELDOUT),
        expected_fits=6,steps=2000,effective_batch=2,checkpoint_interval=100,
        expected_development_predictions=1152,expected_training_predictions=3072,
        expected_new_predictions=4224,expected_inherited_predictions=2304,
        primary_hypothesis='Increasing training coverage at the same2000-update budget improves Render relative to matched Local on the same unseen places.',
        competing_explanations=['generic adaptation benefits both equally','source-content reliability and distribution dominate','fixed budget insufficient for larger data','clip-specific benefit persists'],
        primary_endpoint='development/old/missing/roi/centered_log_mse',
        coverage_interaction='(E_Render-E_Local)_32-(E_Render-E_Local)_12; negative favors increased relative memory utility; must also assess32-place net utility and absolute error.',
        training_sampling='paired seeded permutations of32clips x4queries x2conditions; batch2;4000sample presentations, same optimizer updates as12-place baseline',
        unchanged=['source construction and binding recipe','fusion structure and initial state','native pretrained state/trainable prefixes','optimizer/loss/lr/precision','2000steps, batch2','development places and primary original queries'],
        inventory_sha256=sha256(OUT/'inventory.json'),parent_plan_sha256=sha256(BASE/'plan.json'),
        parent_metrics_path=str(VIEW_OUT/'scoring/metrics.json'),parent_metrics_sha256=sha256(VIEW_OUT/'scoring/metrics.json'),
        source_extension='20additional source-only sensor/joint-validity scenes with identical recipe; original24saved scenes reused unchanged',
        source_prediction_compact='Official AnySplat4-source forward; save predicted source depth/cameras and source-only registration scale/unit, no need to save unused dense Gaussian outputs.',
        scope='Joint increase in independent training places and distinct samples at fixed compute, not a pure diversity/sample-count causal decomposition or asymptotic scaling law.',
        choices_changed=['C1_data_coverage','C6_training_coverage'],outline=['O2','O3','O4','O5'],
        resources=dict(max_workers=3,max_foreign_shared_cards=2,memory_cap_GiB=11,admission_MiB=17408,margin_MiB=6144),
        checkpoint_storage='Full exact resumes in owned RAM tmpfs; verified final weights on data disk; no optimizer/state precision changes.',
        new_confirmation=False,automatic_successor=False,O6_tested=False)
    assert len(p['training_catalog'])==32 and len(p['catalog'])==12
    assert not {m['place'] for m in p['training_catalog']}&{m['place'] for m in p['catalog']}
    assert len({m['place'] for m in p['training_catalog']})==32
    write_json(OUT/'plan.json',p)
    return p


def prepare_inputs():
    p=initialize();identities={};files={};raw={}
    hidden_image=torch.zeros(448,448,dtype=torch.bool);hidden_image[112:336,112:336]=True
    hidden=torch.from_numpy(np.asarray(Image.fromarray(hidden_image.numpy()).resize((210,210),Image.Resampling.NEAREST)).copy())
    for meta in p['added_training_catalog']:
        rid=meta['id'];row=meta['row'];folder=CACHE/rid;folder.mkdir(parents=True,exist_ok=True)
        source_rgb_path(rid).parent.mkdir(parents=True,exist_ok=True)
        marker=OUT/'inputs'/f'{rid}.json'
        if marker.exists():
            r=read(marker);assert all(sha256(k)==v for k,v in r['files'].items());files.update(r['files']);raw.update(r['raw_files']);continue
        one={};rr={};images=[];ks=[];poses=[]
        for q in p['sources']:
            f=row['frames'][q];h=sha256(f['rgb_path']);assert h==f['rgb_sha256'];rr[f['rgb_path']]=h
            rgb,transform=transformed_rgb(f['rgb_path']);images.append(rgb)
            ks.append(transformed_intrinsics(row['K'][q],transform));poses.append(torch.tensor(row['c2w_metric'][q],dtype=torch.float32))
        source=dict(rgb=torch.stack(images),K=torch.stack(ks),c2w=torch.stack(poses),source_ids=p['sources'],id=rid,place=meta['place'],split='training')
        save_tensor(source_rgb_path(rid),source);one[str(source_rgb_path(rid))]=sha256(source_rgb_path(rid))
        for q in EVAL_QUERIES:
            f=row['frames'][q];dep=depth_source(f);h=sha256(f['rgb_path']);assert h==f['rgb_sha256'];rr[f['rgb_path']]=h
            h=sha256(dep)
            if 'depth_sha256' in f:assert h==f['depth_sha256']
            rr[str(dep)]=h;rgb,transform=transformed_rgb(f['rgb_path']);target,valid=transformed_target(dict(f,depth_path=str(dep)),transform)
            value=dict(rgb=rgb,K=transformed_intrinsics(row['K'][q],transform),K_metric=transformed_intrinsics(row['K'][q],transform,210),
                c2w=torch.tensor(row['c2w_metric'][q],dtype=torch.float32),target=target,valid=valid,hidden=hidden,image_missing=hidden_image,id=rid,place=meta['place'],query=q,split='train')
            save_tensor(query_path(rid,q),value);one[str(query_path(rid,q))]=sha256(query_path(rid,q))
        write_json(marker,dict(complete=True,files=one,raw_files=rr,queries_used_in_source=False))
        files.update(one);raw.update(rr)
    for meta in p['original_training_catalog']+p['catalog']:
        rid=meta['id'];identities[rid]=dict(path=str(scene_path(rid)),sha256=sha256(scene_path(rid)))
    write_json(OUT/'input_identity.json',dict(complete=True,existing_scenes=identities,new_inputs=files,raw_files=raw,
        old24unchanged=True,new_training_places=20,development_places_not_promoted=True))
    print(dict(inputs_ready=20,training_places=32,development_places=12),flush=True)


def dataset(p,kind,training,heldout=False):
    samples={};metas=p['training_catalog'] if training else p['catalog'];qs=EVAL_QUERIES if heldout or not training else QUERIES
    for meta in metas:
        rid=meta['id']
        for q in qs:
            query=load(query_path(rid,q));extra=load(feature_path(rid,q))['render'] if kind=='Render' else None
            for c in CONDITIONS:
                samples[rid,q,c]=dict(patches=load(patch_path(rid,q,c))['patches'],extra=extra,hw=(448,448),query=query,condition=c,place=meta['place'])
    return StagedSamples(samples)


if __name__=='__main__':prepare_inputs()
