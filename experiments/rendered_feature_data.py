"""Fixed source identities and finite direct-render feature-fusion comparison."""
import copy
from collections.abc import Mapping

import numpy as np
import torch

from experiments.source_geometry_data import (ROOT, DATA, read, load, sha256,
    query_path, patch_path, QUERIES, CONDITIONS, SEEDS, save_tensor, write_json)

OUT = ROOT / 'results/gaussian_memory/rendered_feature_native1'
CACHE = DATA / 'rendered_feature_native_v1'
SOURCE_OUT = ROOT / 'results/gaussian_memory/source_geometry_control1/sensor'
SOURCE_CACHE = DATA / 'source_geometry_control_v1/sensor'
PARENT_OUT = ROOT / 'results/gaussian_memory/gaussian_scene_memory1'
PARENT_CACHE = DATA / 'gaussian_scene_memory_v1'
PROJECTION_OUT = ROOT / 'results/gaussian_memory/learned_projection_pe1'
KINDS = ('Local', 'Global', 'Render')


def plan():
    return read(OUT / 'plan.json')


def feature_path(rid, query):
    return CACHE / rid / f'features_{query}.pt'


def initialize_plan():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'plan.json').exists():
        return plan()
    p = copy.deepcopy(read(SOURCE_OUT / 'plan.json'))
    p.update(experiment='rendered_feature_native1', kinds=list(KINDS),
        expected_fits=9, expected_adapted_predictions=864, expected_intervention_predictions=288,
        primary_hypothesis='Spatially aligned source-encoder content fused at the same encoder output level can improve native depth beyond matched local adaptation and scene-global context.',
        competing_explanations=['scene-global context sufficient', 'saved local content inadequate or mixed',
            'current observation sufficient', 'frozen receiver or training coverage limits'],
        main_endpoint='missing/roi/centered_log_mse',
        primary_pairs=[['Render','Local'],['Render','Global']],
        secondary_pairs=[['Global','Local'],['Render','Render_shuffle_binding']],
        fusion=dict(network=[2048,256,1024],activation='GELU',parameters=787712,
            location='once after image encoder, before first native frame/global pair',
            output_initialization='zero weight and bias',
            Local='current patches plus broadcast current patch mean',
            Global='current patches plus effective rendered coverage times mean valid saved feature',
            Render='current patches plus actual composited valid saved features; no alpha normalization',
            cache_dtype='float32 derived features; original cached current encoder dtype retained',
            special_tokens='unchanged; added after patch fusion'),
        source_sensor_plan_sha256=sha256(SOURCE_OUT/'plan.json'),
        source_fields_rebuilt=False, source_content_rebound=False,
        base_parameters=787712, read_layers=[], reader_heads=0,
        interventions=dict(kinds=['shuffle_binding'], model='Render final, all3seeds',
            shuffle_seed=20261111, queries='all12development places x4queries x2conditions',
            rule='Permute contents only among valid saved slots; geometry, valid mask and all compositing weights unchanged.'),
        outline=['O2','O3','O4','O5'], choices_changed=['C3'],
        pe_scope='O6 remains an unresolved core objective; this experiment does not establish incremental PE or Gaussian-specific representation superiority.',
        temporary_scope=['source sensor and joint-validity privilege', 'given cameras',
            'static four-source scenes with future sources for early queries', 'frozen backbone and DPT',
            'old training/development places'],
        decisions='Render beyond Local/Global supports this interface; Global=Render limits spatial claim; dependency without net utility is insufficient; no useful net effect closes this fixed fusion. No automatic successor.',
        resources=dict(max_workers=4,max_foreign_shared_cards=2,memory_cap_GiB=7,
            admission_free_MiB=13312,yield_free_MiB=6144,cache_residency='CPU, transfer exact sample to CUDA'),
        new_confirmation=False, development_tuning=False)
    write_json(OUT/'plan.json', p)
    return p


def to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {k:to_device(v,device) for k,v in value.items()}
    return value


class StagedSamples(Mapping):
    def __init__(self, samples): self.samples=samples
    def __len__(self): return len(self.samples)
    def __iter__(self): return iter(self.samples)
    def __getitem__(self,key): return to_device(self.samples[key],'cuda')


def dataset(p, kind, training):
    samples={}; fields={}
    for meta in p['training_catalog'] if training else p['catalog']:
        rid=meta['id']; record=read(OUT/'features'/f'{rid}.json')
        assert record['complete'] and sha256(record['source_path'])==record['source_sha256']
        for q in QUERIES:
            path=feature_path(rid,q)
            assert sha256(path)==record['files'][str(path)]
            payload=load(path) if kind!='Local' else None
            query=load(query_path(rid,q))
            assert sha256(query_path(rid,q))==record['inputs'][str(query_path(rid,q))]
            for condition in CONDITIONS:
                patch=patch_path(rid,q,condition)
                assert sha256(patch)==record['inputs'][str(patch)]
                extra=None if kind=='Local' else payload['global' if kind=='Global' else 'render']
                samples[rid,q,condition]=dict(patches=load(patch)['patches'],extra=extra,
                    hw=(448,448),query=query,condition=condition,place=meta['place'])
    return StagedSamples(samples),fields
