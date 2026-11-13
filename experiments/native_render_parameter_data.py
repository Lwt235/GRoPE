"""Prospective round_002 identities and streamed sample staging."""
import copy
import os
from pathlib import Path
import numpy as np
import torch
from experiments import native_coverage_data as parent
from experiments.native_coverage_data import ROOT,DATA,read,load,sha256,save_tensor,query_path,patch_path,feature_path,scene_path,SEEDS,QUERIES,HELDOUT,EVAL_QUERIES,CONDITIONS,VIEW_CACHE
from experiments.rendered_feature_data import StagedSamples,to_device
from experiments.native_render_factorization import write,now

OUT=ROOT/'results/gaussian_memory/native_render_parameter_pe1'
CACHE=DATA/'native_render_parameter_pe_v1'
RAM=Path('/dev/shm')/f'liwt_native_render_parameter_pe1_{os.getuid()}'
SCIENCE_PLAN=ROOT/'results/gaussian_memory/autonomous_research_loop/round_002/plan.json'
KINDS=('P','S','G')


def plan():return read(OUT/'plan.json')
def geometry_path(rid,q):return CACHE/'geometry'/rid/f'query_{q}.pt'
def checkpoint_path(kind,seed):return CACHE/'finals'/f'{kind}_{seed}'/'step_2000.pt'
def resume_path(kind,seed):return RAM/f'{kind}_{seed}'/'latest.pt'


def initialize():
    if (OUT/'plan.json').exists():return plan()
    assert sha256(SCIENCE_PLAN)=='efc814d88eeb18f253a358c3a8c8b46c1ef43b075bfcf7c278ebde286830b521'
    p=copy.deepcopy(parent.plan())
    p.update(experiment='native_render_parameter_pe1',kinds=list(KINDS),steps=2000,effective_batch=2,checkpoint_interval=100,expected_fits=9,expected_new_predictions=6336,expected_inherited_predictions=4224,
        bootstrap=dict(replicates=5000,seed=20261113),scientific_plan=str(SCIENCE_PLAN),scientific_plan_sha256=sha256(SCIENCE_PLAN),
        resources=dict(max_workers=2,max_foreign_shared_cards=1,memory_cap_GiB=12,idle_cap_GiB=14,admission_MiB=18432,idle_admission_MiB=20480,margin_MiB=6144),
        precision='Original BF16 operations with FP32 parameters/Adam; PE increment cast to native Q/K dtype before addition.',new_confirmation=False,automatic_successor=False)
    OUT.mkdir(parents=True,exist_ok=True);CACHE.mkdir(parents=True,exist_ok=True);RAM.mkdir(mode=0o700,exist_ok=True)
    write(OUT/'plan.json',p);return p


def position(payload,kind):
    extra=torch.zeros_like(payload['full6']) if kind=='P' else payload['sphere6' if kind=='S' else 'full6']
    return torch.cat((payload['center46'],extra),-1).contiguous()


def sample_cpu(kind,rid,q,c,query=None,geometry=None,extra=None):
    if geometry is None:geometry=load(geometry_path(rid,q))
    return dict(patches=load(patch_path(rid,q,c))['patches'],extra=load(feature_path(rid,q))['render'] if extra is None else extra,
        position=position(geometry,kind),alpha=geometry['a'],hw=(448,448),query=load(query_path(rid,q)) if query is None else query,condition=c)


def dataset(p,kind,training,heldout=False):
    samples={};metas=p['training_catalog'] if training else p['catalog'];qs=EVAL_QUERIES if heldout or not training else QUERIES
    for meta in metas:
        rid=meta['id']
        for q in qs:
            query=load(query_path(rid,q));geom=load(geometry_path(rid,q));extra=load(feature_path(rid,q))['render']
            for c in CONDITIONS:samples[rid,q,c]=sample_cpu(kind,rid,q,c,query,geom,extra)
    return StagedSamples(samples)


def budget_charge(category,amount=1):
    import fcntl
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'budget.lock').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX)
        path=OUT/'budget.json';b=read(path) if path.exists() else dict(nontraining_forwards=0,preflight_updates=0,formal_updates=0)
        if category=='nontraining_forwards':assert b[category]+amount<=6600
        elif category=='preflight_updates':assert b[category]+amount<=6
        elif category=='formal_updates':assert b[category]+amount<=18000
        b[category]+=amount;write(path,b)
