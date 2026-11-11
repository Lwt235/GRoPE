"""Explicit parent reuse and source identities for the fixed soft-read study."""
import json
import shutil
import time

import numpy as np
import torch

from experiments.gaussian_memory_setup import ROOT, DATA, sha256
from experiments.gaussian_memory_bound_data import (
    OUT as PARENT_OUT, BASE_CACHE, CACHE as GAUSSIAN_PRIOR_CACHE, SEEDS, QUERIES,
    load_priors as load_parent_priors, write_json, save_tensor, device_tree)
from experiments.gaussian_memory_bound_train import source_identity as parent_source_identity
from models.gaussian_memory_bound_read import projected_slot_weights
from models.gaussian_memory_soft_read import soft_attention_prior

OUT = ROOT / 'results/gaussian_memory/soft_read1'
CACHE = DATA / 'soft_read_v1'
KINDS = ('soft', 'soft_center')


def source_identity():
    result = parent_source_identity()
    for name in ('models/gaussian_memory_soft_read.py', 'experiments/gaussian_memory_soft_data.py',
                 'experiments/gaussian_memory_soft_train.py'):
        result[name] = sha256(ROOT / name)
    return result


def prepare():
    parent = json.loads((PARENT_OUT / 'plan.json').read_text())
    completion = json.loads((PARENT_OUT / 'phase_completion.json').read_text())
    assert completion['experiment_complete'] and not completion['unmet_requirements']
    assert json.loads((PARENT_OUT / 'method_decision.json').read_text())['next_experiment'] == 'soft_read1'
    assert all(sha256(PARENT_OUT / 'source_snapshot' / name) == digest
               for name, digest in parent_source_identity().items())
    plan = {k: parent[k] for k in ('catalog', 'training_ids', 'primary_places', 'input_sha256',
                                 'backbone_identity', 'center_variance', 'scale_rule')}
    plan.update(experiment='soft_read1', schema=1, seeds=list(SEEDS), new_primary_kind='soft',
                conditional_kind='soft_center', reused_kinds=['local', 'content', 'coverage', 'bound', 'native'],
                steps=2000, effective_batch=2, microbatch=1, lr=3e-4, weight_decay=.01, gradient_clip=5.,
                parameters=137472, memory_slots=256, content_dim=32, insert_global_index=11,
                formula='-log(M) + log1p(M*w_ij); null masked',
                parent_plan_sha256=sha256(PARENT_OUT / 'plan.json'),
                parent_completion_sha256=sha256(PARENT_OUT / 'phase_completion.json'),
                parent_report_sha256=sha256(PARENT_OUT / 'report.md'),
                document_sha256=sha256(ROOT / 'docs/gaussian_memory_soft_read_plan.md'),
                parent_results=str(PARENT_OUT), shared_cache=str(BASE_CACHE),
                gaussian_prior_cache=str(GAUSSIAN_PRIOR_CACHE), new_cache=str(CACHE),
                all_data_previously_used=True, confirmation_claim=False, known_camera=True,
                initial_memory_fixed=True, query_frames_never_written=True,
                reference=[0, 1, 2, 3], write_frames=[4, 6, 8, 10], query_frames=list(QUERIES),
                primary_comparison='soft_vs_content_static',
                dynamic_kinds=['content', 'soft'], dynamic_policies=['corrective_clean', 'rebuild_clean'],
                dynamic_evaluation_unconditional=True,
                interventions={'seed': SEEDS[0], 'conditions': [0, 1, 2],
                               'names': ['content_permuted', 'uniform_prior']},
                retention={'seed': SEEDS[0], 'kinds': ['content', 'soft'], 'query': 5,
                           'condition': 0, 'policy': 'corrective_clean', 'state_indices': [0, 1, 4],
                           'main_change': 'state1_to_state4', 'retrospective': True},
                center_trigger='soft mean lower than content and at least two paired seeds positive in static OR corrective mixed log-MSE',
                center_evaluation_policies=['static', 'corrective_clean'])
    target = OUT / 'plan.json'
    if target.exists():
        assert json.loads(target.read_text()) == plan, 'Fixed study plan changed.'
    else:
        write_json(target, plan)
    for name, digest in source_identity().items():
        dest = OUT / 'source_snapshot' / name
        if dest.exists():
            assert sha256(dest) == digest, f'Frozen scientific source changed: {name}'
        else:
            dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(ROOT / name, dest)
    return plan


def load_plan():
    return json.loads((OUT / 'plan.json').read_text())


def center_path(rid, query):
    return CACHE / 'center_priors' / rid / f'q{query}.pt'


@torch.no_grad()
def prepare_center_priors():
    plan = prepare(); start = time.monotonic(); worst = 0.
    for index, meta in enumerate(plan['catalog']):
        rid = meta['id']; base_identity = json.loads((BASE_CACHE / rid / 'identity.json').read_text())
        identity_path = center_path(rid, QUERIES[0]).parent / 'identity.json'
        expected = {'base_identity_sha256': sha256(BASE_CACHE / rid / 'identity.json'),
                    'initial_sha256': base_identity['sha256']['initial.pt'],
                    'center_variance': plan['center_variance'], 'scientific_source_sha256': source_identity()}
        if identity_path.exists():
            old = json.loads(identity_path.read_text())
            assert all(old[k] == v for k, v in expected.items())
            assert all(sha256(identity_path.parent / n) == h for n, h in old['sha256'].items())
            worst = max(worst, old['max_mass_error']); continue
        assert sha256(BASE_CACHE / rid / 'initial.pt') == expected['initial_sha256']
        initial = torch.load(BASE_CACHE / rid / 'initial.pt', map_location='cpu', weights_only=False)
        hashes = {}; error = 0.
        for query in QUERIES:
            name = f'q{query}_c0.pt'; path = BASE_CACHE / rid / name
            assert sha256(path) == base_identity['sha256'][name]
            sample = torch.load(path, map_location='cpu', weights_only=False)
            projected = projected_slot_weights(initial['memory'], sample['c2w'], sample['K'], plan['center_variance'])
            value = float((projected['coverage'] + projected['transmittance'] - 1).abs().max())
            assert value < 1e-5 and torch.isfinite(projected['weights']).all()
            error = max(error, value); target = center_path(rid, query)
            save_tensor(target, projected); hashes[target.name] = sha256(target)
        worst = max(worst, error)
        write_json(identity_path, {**expected, 'sha256': hashes, 'max_mass_error': error})
        if (index + 1) % 10 == 0: print(json.dumps({'center_prior_clips': index + 1}), flush=True)
    write_json(OUT / 'center_priors_complete.json', {'complete': True, 'clips': len(plan['catalog']),
               'max_mass_error': worst, 'seconds': time.monotonic() - start})


def load_priors(ids, center=False, device='cuda'):
    if not center:
        return load_parent_priors(ids, device=device)
    result = {}; identities = source_identity()
    for rid in ids:
        folder = center_path(rid, QUERIES[0]).parent
        identity = json.loads((folder / 'identity.json').read_text())
        assert identity['scientific_source_sha256'] == identities
        for query in QUERIES:
            path = center_path(rid, query); assert sha256(path) == identity['sha256'][path.name]
            result[rid, query] = device_tree(torch.load(path, map_location='cpu', weights_only=False), device)
    return result


def inputs_for(sample, initial, priors):
    content = initial['memory']['features']
    return content, soft_attention_prior(priors[sample['id'], sample['query']])
