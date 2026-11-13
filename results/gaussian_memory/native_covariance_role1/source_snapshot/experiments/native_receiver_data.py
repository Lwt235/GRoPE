"""Finite native receiver adaptation; all scene/cache identities reused."""
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments import rendered_feature_data as source
from experiments.rendered_feature_data import (ROOT, DATA, KINDS, SEEDS, CONDITIONS,
    QUERIES, read, load, sha256, save_tensor, write_json, query_path, patch_path,
    feature_path, SOURCE_CACHE, SOURCE_OUT, PARENT_CACHE, StagedSamples)
from experiments.gaussian_scene_config import schedule

FROZEN = source.OUT
OUT = ROOT/'results/gaussian_memory/native_receiver_adaptation1'
CACHE = DATA/'native_receiver_adaptation_v1'
PREFIXES = ('aggregator.frame_blocks.', 'aggregator.global_blocks.', 'depth_head.')


def plan():
    return read(OUT/'plan.json')


def initialize():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/'plan.json').exists(): return plan()
    p=copy.deepcopy(source.plan())
    p.update(experiment='native_receiver_adaptation1', choices_changed=['C6'],
        primary_hypothesis='Task adaptation of the full native receiver can increase the net utility of the same rendered scene content relative to matched Local and Global.',
        competing_explanations=['saved source content inadequate or mixed',
            'limited training places and generalization', 'current image sufficient',
            'generic task adaptation without memory benefit', 'alternative extra-context inputs not nested'],
        native_trainable_prefixes=list(PREFIXES), native_lr=1e-6, lr=1e-4,
        native_optimizer_foreach=False, fusion_optimizer_foreach=None,
        pretrained_mode='eval retained; gradients enabled on specified blocks/head only',
        native_first_view_tokens_frozen=True, image_encoder_frozen=True,
        expected_fits=9, expected_development_predictions=864,
        expected_adapted_training_predictions=864, expected_frozen_training_predictions=864,
        expected_intervention_predictions=288, expected_new_predictions=2880,
        expected_inherited_development_predictions=864,
        frozen_plan_sha256=sha256(FROZEN/'plan.json'),
        frozen_summary_sha256=sha256(FROZEN/'scoring/summary.json'),
        source_feature_identity=str(FROZEN/'features_complete.json'),
        checkpoint_storage=str(CACHE/'checkpoints'), checkpoint_interval=100,
        discard_obsolete_resume_after_verified_final=True,
        interaction='(E_Render-E_control)_adapted - (E_Render-E_control)_frozen; negative favors receiver adaptation increasing scene utility',
        temporary_scope=['old12train/12development places', 'sensor source and joint-validity privilege',
            'given cameras', 'static saved source set', 'frozen image encoder, saved geometry and content',
            'only original depth task; no O6 PE increment claim'],
        resources=dict(pilot_cap_GiB=18, pilot_admission_MiB=24576,
            yield_below_free_MiB=6144,max_workers=4,max_foreign_shared_cards=2),
        new_confirmation=False, automatic_successor=False)
    write_json(OUT/'plan.json', p)
    return p


def dataset(p,kind,training):
    # This function validates all saved source/input/cache digests before staging.
    return source.dataset(p,kind,training)


def checkpoint_path(kind,seed,latest=False):
    folder=CACHE/'checkpoints'/f'{kind}_{seed}'
    folder.mkdir(parents=True,exist_ok=True)
    return folder/('latest.pt' if latest else 'step_2000.pt')


def core_state(model):
    return {n:t for n,t in model.state_dict().items() if n.startswith(PREFIXES)}


def core_load(model,state):
    result=model.load_state_dict(state,strict=False)
    assert not result.unexpected_keys
    assert not any(n.startswith(PREFIXES) for n in result.missing_keys)


def enable_receiver(model):
    model.eval().requires_grad_(False)
    for name,param in model.named_parameters():
        param.requires_grad_(name.startswith(PREFIXES))
    return [param for param in model.parameters() if param.requires_grad]


def restore_native(model,p):
    if p['model_path'].endswith('.safetensors'):
        from safetensors.torch import load_file
        state=load_file(p['model_path'])
    else:
        state=torch.load(p['model_path'],map_location='cpu',weights_only=True)
        if 'model' in state:state=state['model']
    core_load(model,{n:t for n,t in state.items() if n.startswith(PREFIXES)})


def optimizers(reader,core,p):
    return [torch.optim.AdamW(reader.parameters(),lr=p['lr'],weight_decay=p['weight_decay']),
        torch.optim.AdamW(core,lr=p['native_lr'],weight_decay=p['weight_decay'],foreach=False)]


def digest(module):
    h=hashlib.sha256()
    for name,t in module.state_dict().items():
        h.update(name.encode());h.update(t.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def sync_cuda():
    torch.cuda.synchronize()

