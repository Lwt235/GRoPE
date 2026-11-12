"""Fixed-weight, fixed-scene view transfer: isolated inputs and identities."""
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from experiments.native_receiver_data import ROOT, DATA, OUT as PARENT, FROZEN
from experiments.rendered_feature_data import SOURCE_CACHE, SOURCE_OUT
from experiments.source_geometry_data import load, read, sha256, save_tensor, write_json
from experiments.gaussian_scene_data import (transformed_rgb, transformed_intrinsics,
    transformed_target, query_path as old_query_path)
from experiments.gaussian_memory_incremental_data import depth_source

OUT = ROOT/'results/gaussian_memory/native_receiver_view_transfer1'
CACHE = DATA/'native_receiver_view_transfer_v1'
QUERIES = (1,4,7,10)
OLD_QUERIES = (2,5,8,11)
CONDITIONS = ('full','missing')
KINDS = ('Local','Render')
REGIMES = ('adapted','frozen')
SEEDS = (20261110,20261111,20261112)


def plan(): return read(OUT/'plan.json')
def query_path(rid,q): return CACHE/rid/f'query_{q}.pt'
def feature_path(rid,q): return CACHE/rid/f'render_{q}.pt'
def patch_path(rid,q,c): return CACHE/rid/f'patch_{q}_{c}.pt'
def support_path(rid): return CACHE/rid/'support.pt'


def initialize():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'plan.json').exists(): return plan()
    p=copy.deepcopy(read(PARENT/'plan.json'))
    p.update(experiment='native_receiver_view_transfer1',queries=list(QUERIES),
        old_queries=list(OLD_QUERIES),kinds=list(KINDS),regimes=list(REGIMES),steps=0,
        expected_fits=0,expected_new_predictions=2304,expected_inherited_predictions=2304,
        expected_models=12,primary_hypothesis='Memory advantage transfers to unseen query views within the same source-defined clip, with the remaining deficit at unseen places.',
        competing_explanations=['advantage limited to fitted observations','new query position has different difficulty or source support','source-content quality and interface limitations'],
        main_endpoint='train/new/missing/roi/centered_log_mse/Render-Local in adapted regime',
        transfer='d=E_Render-E_Local; delta_cohort=d_new-d_old; contrast=delta_train-delta_development. Positive is erosion; not unique evidence of memorization.',
        model_and_training_unchanged=True,source_fields_rebuilt=False,source_content_rebound=False,
        choices_changed=[],evaluation_scope_changed=True,outline=['O2','O3','O4','O5'],
        parent_plan_sha256=sha256(PARENT/'plan.json'),
        inherited_metrics_path=str(PARENT/'scoring/metrics.json'),
        inherited_metrics_sha256=sha256(PARENT/'scoring/metrics.json'),
        proposal_sha256=sha256(ROOT/'docs/native_receiver_results_analysis_20261112.md'),
        proposal='docs/native_receiver_results_analysis_20261112.md',
        new_confirmation=False,automatic_successor=False,
        temporary_scope=['previously used clips','sensor-assisted source and joint validity','given cameras and scale','static sources, future references allowed','frozen saved source geometry/content and image encoder','same-clip held-out views are not new independent scenes','no O6 PE increment test'],
        resources=dict(max_workers=3,cap_GiB=8,margin_MiB=6144))
    assert not set(p['sources'])&set(QUERIES)
    assert not set(OLD_QUERIES)&set(QUERIES)
    assert len(p['catalog'])==len(p['training_catalog'])==12
    write_json(OUT/'plan.json',p)
    return p


def model_identity(regime,kind,seed):
    if regime=='adapted':
        record=read(PARENT/'persistence'/f'{kind}_{seed}.json')
        path=Path(record['persistent_path']);digest=record['checkpoint_sha256']
    else:
        path=FROZEN/f'{kind}_{seed}'/'step_2000.pt'
        record=read(PARENT/'evaluation'/f'frozen_train_{kind}_{seed}.json')
        digest=record['checkpoint_sha256']
    assert path.is_file()
    return dict(path=str(path),sha256=digest,bytes=path.stat().st_size)


def prepare_inputs():
    p=initialize();raw={};scenes={};models={};begin=time.monotonic()
    for regime in REGIMES:
        for seed in SEEDS:
            for kind in KINDS:
                models[f'{regime}_{kind}_{seed}']=model_identity(regime,kind,seed)
    for cohort,metas in [('train',p['training_catalog']),('development',p['catalog'])]:
        for meta in metas:
            rid=meta['id'];row=meta['row'];folder=CACHE/rid;folder.mkdir(parents=True,exist_ok=True)
            source=SOURCE_CACHE/rid/'scene_memory.pt';digest=sha256(source)
            assert read(SOURCE_OUT/'binding'/f'{rid}.json')['files'][str(source)]==digest
            scenes[rid]=dict(path=str(source),sha256=digest)
            marker=OUT/'inputs'/f'{rid}.json'
            if marker.exists():
                r=read(marker);assert all(sha256(k)==v for k,v in r['files'].items())
                raw.update(r['raw_files']);continue
            files={};raw_files={}
            image_missing=torch.zeros(448,448,dtype=torch.bool)
            image_missing[112:336,112:336]=True
            hidden=torch.from_numpy(np.asarray(Image.fromarray(image_missing.numpy()).resize((210,210),Image.Resampling.NEAREST)).copy())
            for q in QUERIES:
                f=row['frames'][q];rgbpath=Path(f['rgb_path']);depthpath=depth_source(f)
                for path,key in [(rgbpath,'rgb_sha256'),(depthpath,'depth_sha256')]:
                    h=sha256(path)
                    if key in f: assert h==f[key]
                    raw_files[str(path)]=h
                rgb,transform=transformed_rgb(rgbpath)
                target,valid=transformed_target(dict(f,depth_path=str(depthpath)),transform)
                payload=dict(rgb=rgb,c2w=torch.tensor(row['c2w_metric'][q],dtype=torch.float32),
                    K=transformed_intrinsics(row['K'][q],transform),
                    K_metric=transformed_intrinsics(row['K'][q],transform,210),
                    target=target,valid=valid,hidden=hidden,image_missing=image_missing,
                    id=rid,place=meta['place'],query=q,split=cohort)
                path=query_path(rid,q);save_tensor(path,payload);files[str(path)]=sha256(path)
            write_json(marker,dict(complete=True,id=rid,cohort=cohort,files=files,raw_files=raw_files))
            raw.update(raw_files)
    write_json(OUT/'identities.json',dict(complete=True,models=models,scenes=scenes,
        raw_files=raw,plan_sha256=sha256(OUT/'plan.json'),
        weights_hash_verification='Expected hashes inherited from completed evaluations; each worker verifies actual checkpoint bytes before loading.',
        preparation_seconds=time.monotonic()-begin))
    print(dict(inputs_ready=24,queries=96,models=12,seconds=time.monotonic()-begin),flush=True)


if __name__=='__main__': prepare_inputs()
