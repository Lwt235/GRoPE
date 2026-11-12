"""Fixed 32-place catalog, unchanged sources, isolated input-attention PE outputs."""
import copy
import os
from pathlib import Path
import numpy as np
import torch
from experiments import native_coverage_data as parent
from experiments.native_coverage_data import (ROOT, DATA, read, load, sha256, save_tensor,
    write_json, query_path, patch_path, scene_path, QUERIES, HELDOUT, EVAL_QUERIES,
    CONDITIONS, SEEDS, VIEW_CACHE)
from experiments.rendered_feature_data import StagedSamples, to_device

PARENT_OUT = parent.OUT
PARENT_CACHE = parent.CACHE
OUT = ROOT/'results/gaussian_memory/native_input_pe1'
CACHE = DATA/'native_input_pe_v1'
RAM = Path('/dev/shm')/f'liwt_native_input_pe1_{os.getuid()}'
KINDS = ('L','P','S','G')


def plan(): return read(OUT/'plan.json')
def geometry_path(rid,q): return CACHE/'geometry'/rid/f'query_{q}.pt'
def checkpoint_path(kind,seed): return CACHE/'finals'/f'{kind}_{seed}'/'step_2000.pt'
def resume_path(kind,seed): return RAM/f'{kind}_{seed}'/'latest.pt'


def initialize():
    if (OUT/'plan.json').exists(): return plan()
    old=parent.plan()
    # Explicit inherited fields avoid unused parent experiment counts/decisions.
    names=['training_catalog','original_training_catalog','added_training_catalog','catalog',
        'sources','model_path','model_sha256','lr','native_lr','weight_decay','gradient_clip',
        'bootstrap','native_trainable_prefixes','native_optimizer_foreach','fusion_optimizer_foreach']
    p={k:copy.deepcopy(old[k]) for k in names if k in old}
    p.update(experiment='native_input_pe1',kinds=list(KINDS),steps=2000,effective_batch=2,
        queries=list(QUERIES),heldout_queries=list(HELDOUT),evaluation_queries=list(EVAL_QUERIES),conditions=list(CONDITIONS),
        checkpoint_interval=100,expected_fits=12,expected_new_predictions=8448,
        expected_development_predictions=2304,expected_training_predictions=6144,
        main_endpoint='development/original/missing/roi/centered_log_mse',
        primary_pairs=[['G','P']],secondary_pairs=[['G','S'],['S','P'],['P','L'],['S','L'],['G','L']],
        primary_hypothesis='At fixed32-place full native adaptation, single input attention with Gaussian projection PE improves native spatial prediction beyond same-content learned center PE.',
        competing_explanations=['center and content sufficient','spherical extent sufficient','projection scalar insufficient','saved content or training limits','clip-specific use without cross-place utility'],
        reader=dict(width=256,heads=4,location='once after encoder before special-token concatenation',
            base_parameters=1054464,pe_parameters=208,formal_parameters=1054672,
            P_effective_extra_column_parameters=0,S_G_effective_extra_column_parameters=16,
            L='current patches as K/V, no geometric bias',
            P='same saved features, center7 plus zero eighth input',
            S='same saved features, center7 plus log1p(2048*w_sphere)',
            G='same saved features, center7 plus log1p(2048*w_full)',
            sphere='PE only covariance trace/3 times identity; same centers, opacity, bound content, valid mask',
            geometry_network=[8,16,4],new_column_initialization='zero',output_initialization='zero',
            original_2d_rope='retained',internal_read_layers=[],prior_mlp_fusion_retained=False),
        native_parameters=637383906,native_mode='eval with full frame/global and DPT gradients; encoder/other heads frozen',
        precision='FP32 parameters and Adam states, BF16 autocast, cast cache disabled',
        loss='0.5*(whole valid raw log MSE + centered log MSE)',
        parent_plan_sha256=sha256(PARENT_OUT/'plan.json'),
        source_fields_rebuilt=False,source_content_rebound=False,
        outline=['O2','O3','O4','O5','O6'],choices_changed=['C3_single_input_attention','C4_conditional_retest'],
        temporary_scope=['old32train/12development places','source sensor and joint-validity privilege',
            'given cameras and scale','static4sources, future sources possible for early queries',
            'frozen encoder and saved scene','depth only; no online or RGB-only claim'],
        decisions='G>P with net utility: candidate, then independent confirmation; G=S favors sphere; source utility without PE increment favors simpler reader; negative/uncertain closes this fixed interface without automatic expansion.',
        resources=dict(max_workers=3,max_foreign_shared_cards=2,memory_cap_GiB=14,admission_MiB=20480,margin_MiB=6144),
        new_confirmation=False,automatic_successor=False)
    assert len(p['training_catalog'])==32 and len(p['catalog'])==12
    assert not {m['place'] for m in p['training_catalog']}&{m['place'] for m in p['catalog']}
    OUT.mkdir(parents=True,exist_ok=True);CACHE.mkdir(parents=True,exist_ok=True)
    write_json(OUT/'plan.json',p)
    return p


def geometry_tensor(payload,kind):
    center=payload['center']
    extra=torch.zeros_like(center[...,:1]) if kind=='P' else payload['sphere' if kind=='S' else 'full'][...,None]
    return torch.cat((center,extra),-1).contiguous()


def sample_cpu(kind,rid,q,c,field=None,geometry=None,query=None):
    patches=load(patch_path(rid,q,c))['patches']
    sample=dict(patches=patches,hw=(448,448),query=query if query is not None else load(query_path(rid,q)),condition=c)
    if kind!='L':
        if field is None:field=load(scene_path(rid))
        if geometry is None:geometry=geometry_tensor(load(geometry_path(rid,q)),kind)
        sample.update(features=field['features'],valid=field['has_content'],geometry=geometry)
    return sample


def dataset(p,kind,training,heldout=False):
    samples={};metas=p['training_catalog'] if training else p['catalog']
    qs=EVAL_QUERIES if heldout or not training else QUERIES
    for meta in metas:
        rid=meta['id'];field=load(scene_path(rid)) if kind!='L' else None
        for q in qs:
            query=load(query_path(rid,q));geometry=geometry_tensor(load(geometry_path(rid,q)),kind) if kind!='L' else None
            for c in CONDITIONS:
                sample=sample_cpu(kind,rid,q,c,field,geometry,query);sample['place']=meta['place']
                samples[rid,q,c]=sample
    return StagedSamples(samples)


if __name__=='__main__':initialize()
