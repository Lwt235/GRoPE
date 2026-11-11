"""Explicit immutable inputs for training on causally updated memory states."""
import json
import shutil

import numpy as np
import torch

from experiments.gaussian_memory_setup import ROOT, DATA, sha256
from experiments.gaussian_memory_soft_data import (
    OUT as PARENT_OUT, PARENT_OUT as MATCH_OUT, BASE_CACHE, SEEDS, QUERIES,
    source_identity as parent_source_identity, write_json, save_tensor, device_tree)
from experiments.gaussian_memory_incremental_states import STATE_OUT
from models.gaussian_memory_bound_read import projected_slot_weights, attention_prior
from models.gaussian_memory_soft_read import soft_attention_prior

OUT = ROOT / 'results/gaussian_memory/state_read1'
CACHE = DATA / 'state_read_v1'
KINDS = ('content', 'soft', 'soft_center')
POLICIES = ('static', 'corrective_clean', 'rebuild_clean')


def source_identity():
    result = parent_source_identity()
    for name in ('experiments/gaussian_memory_state_data.py',
                 'experiments/gaussian_memory_state_train.py'):
        result[name] = sha256(ROOT / name)
    return result


def load_plan():
    return json.loads((OUT / 'plan.json').read_text())


def prepare():
    parent = json.loads((PARENT_OUT / 'plan.json').read_text())
    completion = json.loads((PARENT_OUT / 'phase_completion.json').read_text())
    assert completion['experiment_complete'] and not completion['unmet_requirements']
    assert json.loads((PARENT_OUT / 'method_decision.json').read_text())['next_experiment'] == 'state_read1'
    assert all(sha256(PARENT_OUT / 'source_snapshot' / n) == h for n, h in parent_source_identity().items())
    plan = {k: parent[k] for k in ('catalog', 'training_ids', 'primary_places', 'input_sha256',
                                 'backbone_identity', 'center_variance', 'scale_rule')}
    plan.update(experiment='state_read1', schema=1, seeds=list(SEEDS), kinds=['content', 'soft'],
                steps=2000, effective_batch=2, microbatch=1, lr=3e-4, weight_decay=.01, gradient_clip=5.,
                parameters=137472, memory_slots=256, content_dim=32, insert_global_index=11,
                training_policy='corrective_clean', query_to_state={str(q): i for i, q in enumerate(QUERIES, 1)},
                formula=parent['formula'], reference=[0, 1, 2, 3], write_frames=[4, 6, 8, 10],
                query_frames=list(QUERIES), query_frames_never_written=True, memory_frozen=True,
                parent_plan_sha256=sha256(PARENT_OUT / 'plan.json'),
                parent_completion_sha256=sha256(PARENT_OUT / 'phase_completion.json'),
                parent_report_sha256=sha256(PARENT_OUT / 'report.md'),
                document_sha256=sha256(ROOT / 'docs/gaussian_memory_state_read_plan.md'),
                match_parent_results=str(MATCH_OUT), reused_results=str(PARENT_OUT),
                shared_cache=str(BASE_CACHE), state_results=str(STATE_OUT), new_cache=str(CACHE),
                primary_comparison='soft_dynamic_vs_content_dynamic_on_corrective_clean',
                evaluation_policies=list(POLICIES), all_policy_evaluation_unconditional=True,
                reused_training_policy='initial_M0', all_data_previously_used=True, confirmation_claim=False,
                known_camera=True, interventions={'seed': SEEDS[0], 'policy': 'corrective_clean',
                    'names': ['content_permuted', 'uniform_prior'], 'conditions': [0, 1, 2]},
                retention={'seed': SEEDS[0], 'kinds': ['content', 'soft'], 'query': 5, 'condition': 0,
                    'policy': 'corrective_clean', 'state_indices': [0, 1, 4], 'retrospective': True},
                center_trigger='Corrective Soft vs C relative error reduction >=1% AND at least two paired seeds positive',
                center_policies=['static', 'corrective_clean'],
                fixed_figure_ids=['transfer_00', 'confirm_00', 'fresh_00'])
    target = OUT / 'plan.json'
    if target.exists():
        assert json.loads(target.read_text()) == plan, 'Fixed training-state plan changed.'
    else:
        write_json(target, plan)
    for name, digest in source_identity().items():
        dest = OUT / 'source_snapshot' / name
        if dest.exists():
            assert sha256(dest) == digest, f'Frozen scientific source changed: {name}'
        else:
            dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(ROOT / name, dest)
    return plan


def load_tracks(rid):
    path = STATE_OUT / rid / 'trajectories.pt'
    identity = json.loads((path.parent / 'complete.json').read_text())
    digest = sha256(path)
    assert identity['complete'] and identity['all_queries_readonly'] and not identity['labels_used_for_updates']
    assert digest == identity['sha256']['trajectories.pt']
    previous = PARENT_OUT / 'evaluation/integrated' / f'soft_{SEEDS[0]}_corrective_clean' / f'{rid}.json'
    assert digest == json.loads(previous.read_text())['sources']['state_trajectories_sha256']
    return torch.load(path, map_location='cpu', weights_only=False), digest


@torch.no_grad()
def memory_prior(memory, sample, kind, center_variance):
    assert kind in KINDS
    projected = projected_slot_weights(memory, sample['c2w'].to(memory['features'].device),
                                       sample['K'].to(memory['features'].device),
                                       center_variance if kind == 'soft_center' else None)
    return attention_prior(projected, 'content') if kind == 'content' else soft_attention_prior(projected)


@torch.no_grad()
def training_inputs(ids, samples, kind, device='cuda'):
    plan = load_plan(); result = {}; identities = {}
    lookup = {(s['id'], s['query'], s['condition']): s for s in samples}
    for rid in ids:
        tracks, digest = load_tracks(rid); identities[rid] = {'state_trajectories_sha256': digest, 'queries': {}}
        assert len(tracks['corrective_clean']['states']) == 5
        for ti, query in enumerate(QUERIES, 1):
            memory = device_tree(tracks['corrective_clean']['states'][ti], device)
            actual_frames = sorted(set(memory['sources'].nonzero()[:, 1].cpu().tolist()))
            allowed_frames = [0, 1, 2, 3] + [4, 6, 8, 10][:ti]
            assert set(actual_frames).issubset(allowed_frames) and all(f < query for f in actual_frames)
            sample = lookup[rid, query, 0]
            prior = memory_prior(memory, sample, kind, plan['center_variance'])
            content = memory['features'].detach()
            assert content.shape == (256, 32) and torch.isfinite(content).all()
            assert torch.isfinite(prior[..., :-1]).all() and torch.isneginf(prior[..., -1]).all()
            assert not content.requires_grad and not prior.requires_grad
            result[rid, query] = (content, prior.detach())
            identities[rid]['queries'][str(query)] = {'state_index': ti, 'actual_source_frames': actual_frames,
                                                     'allowed_frames': allowed_frames}
    return result, identities


def inputs_for(sample, inputs):
    return inputs[sample['id'], sample['query']]
