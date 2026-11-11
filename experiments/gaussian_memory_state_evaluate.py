"""Fixed evaluations for dynamic-trained readers and explicit parent reuse."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.gaussian_memory_setup import ROOT, sha256
from experiments.gaussian_memory_view_data import load_backbone
from experiments.gaussian_memory_state_data import (
    OUT, CACHE, PARENT_OUT, BASE_CACHE, SEEDS, KINDS, QUERIES, POLICIES, load_plan,
    source_identity, load_tracks, memory_prior, write_json, save_tensor, device_tree)
from experiments.gaussian_memory_bound_evaluate import clip_samples, predict
from experiments.gaussian_memory_soft_data import load_priors, GAUSSIAN_PRIOR_CACHE
from experiments.gaussian_memory_soft_evaluate import content_permutation
from experiments.gaussian_memory_incremental_evaluate import score
from models.gaussian_memory_bound_read import BoundMemoryAdapter, attention_prior
from models.gaussian_memory_soft_read import soft_attention_prior

PREDICTIONS = CACHE / 'predictions'


def evaluation_protocol():
    plan = load_plan()
    value = {'experiment': 'state_read1', 'schema': 1, 'defined_before_formal_predictions': True,
        'plan_sha256': sha256(OUT / 'plan.json'), 'primary': plan['primary_comparison'],
        'seeds': list(SEEDS), 'bootstrap': '5000 paired place and seed draws; seed 20261110',
        'training_state_interaction': '(C_dynamic-S_dynamic)-(C_static-S_static) on identical Corrective states',
        'metrics': ['log_mse', 'centered_log_mse', 'bias_squared', 'absrel', 'rmse_m',
                    'hidden_log_mse', 'visible_log_mse'],
        'evaluation_policies': list(POLICIES), 'interventions': plan['interventions'],
        'permutation': 'reuse pure soft_read1 deterministic per-clip content permutation; same indices, new state content',
        'retention': plan['retention'], 'center_trigger': plan['center_trigger'],
        'fixed_figure_ids': plan['fixed_figure_ids'],
        'posthoc_counterexample': 'first catalog clip of primary place with largest dynamic-trained Soft-minus-C Corrective log-MSE',
        'all_data_previously_used': True, 'scientific_source_sha256': source_identity(),
        'helper_sha256': {name: sha256(ROOT / name) for name in (
            'experiments/gaussian_memory_bound_evaluate.py', 'experiments/gaussian_memory_soft_evaluate.py',
            'experiments/gaussian_memory_incremental_evaluate.py')}}
    target = OUT / 'evaluation_protocol.json'
    if target.exists(): assert json.loads(target.read_text()) == value, 'Evaluation protocol changed.'
    else: write_json(target, value)
    return value


def evaluator_identity():
    return {'evaluation_protocol_sha256': sha256(OUT / 'evaluation_protocol.json'),
            'evaluation_source_sha256': sha256(ROOT / 'experiments/gaussian_memory_state_evaluate.py')}


def load_adapter(kind, seed):
    assert kind in KINDS and seed in SEEDS
    path = OUT / f'{kind}_{seed}/step_2000.pt'
    status = json.loads((path.parent / 'status.json').read_text())
    assert status['complete'] and status['steps'] == 2000 and not status['pilot']
    assert sha256(path) == status['checkpoint_sha256']
    saved = torch.load(path, map_location='cpu', weights_only=False)
    assert saved['source_sha256'] == source_identity() and saved['kind'] == kind and saved['seed'] == seed
    adapter = BoundMemoryAdapter().cuda().eval().requires_grad_(False)
    adapter.load_state_dict(saved['model'])
    return adapter, status['checkpoint_sha256']


def completed(phase, case, rid, checkpoint_hash):
    path = OUT / 'evaluation' / phase / case / f'{rid}.json'
    if not path.exists(): return False
    record = json.loads(path.read_text())
    assert record['complete'] and record['checkpoint_sha256'] == checkpoint_hash
    assert all(record[k] == v for k, v in evaluator_identity().items())
    assert sha256(record['prediction_path']) == record['prediction_sha256']
    return True


def save_record(phase, case, rid, rows, predictions, checkpoint_hash, sources, seconds):
    target = PREDICTIONS / phase / case / f'{rid}.pt'
    target.parent.mkdir(parents=True, exist_ok=True)
    save_tensor(target, {'prediction_native': torch.stack(predictions),
                        'sample_keys': [[r['query'], r['condition']] for r in rows],
                        'state_keys': [r.get('state_key') for r in rows]})
    write_json(OUT / 'evaluation' / phase / case / f'{rid}.json', {
        'complete': True, 'rows': rows, 'prediction_path': str(target),
        'prediction_sha256': sha256(target), 'checkpoint_sha256': checkpoint_hash,
        'sources': sources, 'seconds': seconds, **evaluator_identity()})


def progress(phase, case, index, start):
    value = {'stage': phase, 'case': case, 'clips_complete': index + 1, 'seconds': time.monotonic() - start}
    write_json(OUT / 'evaluation_progress' / f'{phase}_{case}.json', value)
    if (index + 1) % 10 == 0: print(json.dumps(value), flush=True)


def reuse_baselines():
    evaluation_protocol(); plan = load_plan(); manifest = {}
    paths = {f'{k}_{s}_{pol}': ('static' if pol == 'static' else 'integrated',
                 f'{k}_{s}' if pol == 'static' else f'{k}_{s}_{pol}')
             for k in ('content', 'soft') for s in SEEDS for pol in POLICIES}
    paths.update({f'local_{s}': ('static', f'local_{s}') for s in SEEDS})
    paths['native'] = ('static', 'native')
    for case, (phase, parent_case) in paths.items():
        manifest[case] = {}
        for meta in plan['catalog']:
            rid = meta['id']; source = PARENT_OUT / 'evaluation' / phase / parent_case / f'{rid}.json'
            record = json.loads(source.read_text()); assert record['complete'] and len(record['rows']) == 12
            if case == 'native': assert record['cache_identity_sha256'] == sha256(BASE_CACHE / rid / 'identity.json')
            else:
                assert record['sources']['causal_cache_identity_sha256'] == sha256(BASE_CACHE / rid / 'identity.json')
                assert sha256(record['prediction_path']) == record['prediction_sha256']
            digest = sha256(source); manifest[case][rid] = digest
            record.update(state_reused_from=str(source), state_reused_record_sha256=digest,
                          state_plan_sha256=sha256(OUT / 'plan.json'))
            write_json(OUT / 'evaluation/reused' / case / f'{rid}.json', record)
    write_json(OUT / 'baseline_reuse.json', {'complete': True, 'new_predictions': 0,
               'parent_records': manifest, 'parent_independent_checks_sha256': sha256(PARENT_OUT / 'independent_checks.json'),
               'parent_costs_path': str(PARENT_OUT / 'native_costs.json'),
               'parent_costs_sha256': sha256(PARENT_OUT / 'native_costs.json'),
               'cost_scope': 'unchanged operators; descriptive shared-GPU cached-tail costs, not a new benchmark'})


def evaluate(kind='soft', seed=None, policy=None):
    assert kind in KINDS
    evaluation_protocol(); plan = load_plan(); backbone = load_backbone(); start = time.monotonic()
    policies = ('static', 'corrective_clean') if kind == 'soft_center' else POLICIES
    if policy is not None: assert policy in policies; policies = (policy,)
    for s in ([seed] if seed is not None else SEEDS):
        adapter, checkpoint_hash = load_adapter(kind, s)
        for pol in policies:
            case = f'{kind}_{s}_{pol}'
            for index, meta in enumerate(plan['catalog']):
                rid = meta['id']
                if completed('read', case, rid, checkpoint_hash): continue
                begin = time.monotonic(); samples, initial, sources = clip_samples(rid)
                tracks = priors = None
                if pol == 'static' and kind != 'soft_center':
                    priors = load_priors([rid])
                    source = GAUSSIAN_PRIOR_CACHE / rid / 'identity.json'
                    sources.update(prior_identity_path=str(source), prior_identity_sha256=sha256(source))
                if pol != 'static':
                    tracks, digest = load_tracks(rid); sources['state_trajectories_sha256'] = digest
                rows, predictions = [], []
                for ti, q in enumerate(QUERIES, 1):
                    memory = initial['memory'] if pol == 'static' else device_tree(tracks[pol]['states'][ti])
                    if priors is not None:
                        p = priors[rid, q]
                        prior = attention_prior(p, 'content') if kind == 'content' else soft_attention_prior(p)
                    else: prior = memory_prior(memory, samples[q, 0], kind, plan['center_variance'])
                    for c in range(3):
                        sample = samples[q, c]; prediction = predict(backbone, adapter, sample, memory['features'], prior)
                        row = score(prediction, sample); row['state_key'] = [pol, 0 if pol == 'static' else ti]
                        rows.append(row); predictions.append(prediction)
                save_record('read', case, rid, rows, predictions, checkpoint_hash, sources, time.monotonic() - begin)
                progress('read', case, index, start)
            write_json(OUT / f'read_{case}_complete.json', {'complete': True, 'clips': len(plan['catalog'])})
        del adapter


def interventions():
    evaluation_protocol(); plan = load_plan(); backbone = load_backbone(); start = time.monotonic()
    adapter, checkpoint_hash = load_adapter('soft', SEEDS[0]); names = ('content_permuted', 'uniform_prior')
    for index, meta in enumerate(plan['catalog']):
        rid = meta['id']; cases = [f'soft_{SEEDS[0]}_{name}' for name in names]
        if all(completed('intervention', case, rid, checkpoint_hash) for case in cases): continue
        samples, initial, sources = clip_samples(rid); del initial
        tracks, digest = load_tracks(rid); sources['state_trajectories_sha256'] = digest
        permutation = content_permutation(rid, 256, 'cuda'); sources['content_permutation'] = permutation.cpu().tolist()
        for name, case in zip(names, cases):
            if completed('intervention', case, rid, checkpoint_hash): continue
            begin = time.monotonic(); rows, predictions = [], []
            for ti, q in enumerate(QUERIES, 1):
                memory = device_tree(tracks['corrective_clean']['states'][ti]); content = memory['features']
                values = content[permutation] if name == 'content_permuted' else content
                prior = memory_prior(memory, samples[q, 0], 'content' if name == 'uniform_prior' else 'soft', plan['center_variance'])
                for c in range(3):
                    sample = samples[q, c]; prediction = predict(backbone, adapter, sample, values, prior)
                    row = score(prediction, sample); row.update(intervention=name, state_key=['corrective_clean', ti])
                    rows.append(row); predictions.append(prediction)
            save_record('intervention', case, rid, rows, predictions, checkpoint_hash, sources, time.monotonic() - begin)
            progress('intervention', case, index, start)
    write_json(OUT / 'interventions_complete.json', {'complete': True, 'clips': len(plan['catalog']), 'fixed_seed': SEEDS[0]})


def retention(kind='soft'):
    assert kind in ('content', 'soft')
    evaluation_protocol(); plan = load_plan(); backbone = load_backbone(); start = time.monotonic()
    adapter, checkpoint_hash = load_adapter(kind, SEEDS[0]); case = f'{kind}_{SEEDS[0]}'
    for index, meta in enumerate(plan['catalog']):
        rid = meta['id']
        if completed('retention', case, rid, checkpoint_hash): continue
        begin = time.monotonic(); samples, initial, sources = clip_samples(rid, (0,)); del initial
        sample = samples[5, 0]; tracks, digest = load_tracks(rid); sources['state_trajectories_sha256'] = digest
        rows, predictions = [], []
        for ti in (0, 1, 4):
            memory = device_tree(tracks['corrective_clean']['states'][ti])
            prior = memory_prior(memory, sample, kind, plan['center_variance'])
            prediction = predict(backbone, adapter, sample, memory['features'], prior)
            row = score(prediction, sample); row.update(state_key=['corrective_clean', ti], retrospective_retention=True)
            rows.append(row); predictions.append(prediction)
        save_record('retention', case, rid, rows, predictions, checkpoint_hash, sources, time.monotonic() - begin)
        progress('retention', case, index, start)
    write_json(OUT / f'retention_{case}_complete.json', {'complete': True, 'clips': len(plan['catalog'])})


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=('protocol', 'reuse_baselines', 'evaluate', 'interventions', 'retention'))
    p.add_argument('--kind', choices=KINDS, default='soft'); p.add_argument('--seed', type=int)
    p.add_argument('--policy', choices=POLICIES); args = p.parse_args(); torch.set_num_threads(4)
    if args.stage == 'protocol': evaluation_protocol()
    elif args.stage == 'evaluate': evaluate(args.kind, args.seed, args.policy)
    elif args.stage == 'retention': retention(args.kind)
    elif args.stage == 'interventions': interventions()
    else: reuse_baselines()
