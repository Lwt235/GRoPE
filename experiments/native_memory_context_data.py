"""Round003 independent identities, source-only state and finite counters."""
import copy
import os
from pathlib import Path
import numpy as np
import torch
from experiments import native_coverage_data as parent
from experiments.native_coverage_data import ROOT,DATA,read,load,sha256,save_tensor as original_save_tensor,query_path,patch_path,feature_path,scene_path,SEEDS,QUERIES,HELDOUT,EVAL_QUERIES,CONDITIONS,VIEW_CACHE
from experiments.rendered_feature_data import StagedSamples,to_device
from experiments.native_render_factorization import write,now
OUT=ROOT/'results/gaussian_memory/native_memory_context_pe1'
CACHE=DATA/'native_memory_context_v1'
RAM=Path('/dev/shm')/f'liwt_native_memory_context_pe1_{os.getuid()}'
SCIENCE_PLAN=ROOT/'results/gaussian_memory/autonomous_research_loop/round_003/plan.json'
KINDS=('P','S','G')
def plan():return read(OUT/'plan.json')
def graph_path(rid):return CACHE/'graphs'/f'{rid}.pt'
def weights_path(rid,q):return CACHE/'weights'/rid/f'{q}.pt'
def memory_path(kind,seed,rid):return CACHE/'memories'/f'{kind}_{seed}'/f'{rid}.pt'
def checkpoint_path(kind,seed):return CACHE/'finals'/f'{kind}_{seed}'/'step_2000.pt'
def resume_path(kind,seed):return RAM/f'{kind}_{seed}'/'latest.pt'
def initialize():
    if (OUT/'plan.json').exists():return plan()
    assert sha256(SCIENCE_PLAN)=='1500cb73752237fd7d792e0f10fac2f529d80e7198a2f123fd11fa8220a90e59'
    p=copy.deepcopy(parent.plan());p.update(experiment='native_memory_context_pe1',kinds=list(KINDS),steps=2000,effective_batch=2,checkpoint_interval=100,expected_fits=9,expected_new_predictions=6336,expected_inherited_predictions=4224,bootstrap=dict(replicates=5000,seed=20261113),scientific_plan=str(SCIENCE_PLAN),scientific_plan_sha256=sha256(SCIENCE_PLAN),resources=dict(max_workers=2,max_foreign_shared_cards=1,memory_cap_GiB=12,idle_cap_GiB=14,admission_MiB=18432,idle_admission_MiB=20480,margin_MiB=6144),precision='FP32 writer, parameters and Adam; FP16 encoded content in training and saved state; original native BF16 arithmetic.',new_confirmation=False,automatic_successor=False,O6_scope='source-only writer Transformer relationship PE; native query2D RoPE retained without additional PE')
    science=read(SCIENCE_PLAN)
    p.update(primary_hypothesis=science['hypothesis'],competing_explanations=science['competing_explanations'],primary_endpoint=science['evaluation']['primary_endpoint'],primary_pairs=[['G','P'],['G','S'],['S','P']],writer=science['architecture']['writer'],read_layers=[],reader_heads=0,interventions={},outline=['O2','O3','O4','O5','O6'],choices_changed=['C2','C4','C6'],source_content_rebound=False,source_fields_rebuilt=False,decisions=science['decisions'] if 'decisions' in science else 'See fixed round003 scientific plan decision branches; Sol reports empirical flags only.')
    OUT.mkdir(parents=True,exist_ok=True);CACHE.mkdir(parents=True,exist_ok=True);RAM.mkdir(mode=0o700,exist_ok=True);write(OUT/'plan.json',p);return p

def dataset(p,kind,training,heldout=False):
    samples={};metas=p['training_catalog'] if training else p['catalog'];qs=EVAL_QUERIES if heldout or not training else QUERIES
    for meta in metas:
        rid=meta['id'];graph=load(graph_path(rid));features=load(scene_path(rid))['features']
        for q in qs:
            query=load(query_path(rid,q));weights=load(weights_path(rid,q))['weights']
            for c in CONDITIONS:samples[rid,q,c]=dict(patches=load(patch_path(rid,q,c))['patches'],source_features=features,graph=graph,weights=weights,hw=(448,448),query=query,condition=c)
    return StagedSamples(samples)

def budget_charge(category,amount=1):
    import fcntl
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'budget.lock').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX);path=OUT/'budget.json';b=read(path) if path.exists() else dict(nontraining_forwards=0,preflight_updates=0,formal_updates=0,independent_writer_forwards=0,training_writer_forwards=0)
        limits=dict(nontraining_forwards=6600,preflight_updates=6,formal_updates=18000,independent_writer_forwards=600)
        if category in limits:assert b[category]+amount<=limits[category]
        b[category]=b.get(category,0)+amount;write(path,b)


def save_tensor(path,value):
    # Serialize durable checkpoint/atomic replacement writes. RAM snapshots do
    # not consume data disk, but resource exit spills share the same lock.
    import fcntl
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'publication.lock').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX)
        if Path(path).is_relative_to(CACHE):
            from experiments.native_memory_context_resources import check
            space=check()
            # Already-reserved recovery/partial writes must remain possible
            # after a yield. Other publications stop while preserving RAM.
            if space['stop_reasons'] and 'resumes' not in Path(path).parts and 'partial' not in Path(path).name:raise SystemExit(75)
        original_save_tensor(path,value)
