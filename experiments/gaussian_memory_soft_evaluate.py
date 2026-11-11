"""Matched static and dynamic reads for the fixed, globally accessible prior."""
import argparse
import hashlib
import json
import time

import numpy as np
import torch

from experiments.gaussian_memory_setup import ROOT, sha256
from experiments.gaussian_memory_view_data import load_backbone
from experiments.gaussian_memory_soft_data import (
    OUT, CACHE, PARENT_OUT, BASE_CACHE, GAUSSIAN_PRIOR_CACHE, SEEDS, KINDS, QUERIES,
    load_plan, load_priors, inputs_for, source_identity, center_path, write_json, save_tensor, device_tree)
from experiments.gaussian_memory_bound_evaluate import (
    clip_samples, predict, load_adapter as load_parent_adapter)
from experiments.gaussian_memory_incremental_evaluate import score
from experiments.gaussian_memory_incremental_states import STATE_OUT
from models.gaussian_memory_bound_read import BoundMemoryAdapter, projected_slot_weights, attention_prior
from models.gaussian_memory_soft_read import soft_attention_prior

PREDICTIONS = CACHE / 'predictions'
PARENT_KINDS = ('local', 'content', 'coverage', 'bound')


def evaluation_protocol():
    plan = load_plan()
    value = {
        'experiment': 'soft_read1', 'schema': 1, 'defined_before_new_formal_evaluation': True,
        'plan_sha256': sha256(OUT / 'plan.json'),
        'primary': 'place-equal mixed native log-MSE on the same 51 nontraining places; Soft vs C',
        'seeds': list(SEEDS), 'secondary': ['full', 'masked', 'hidden', 'visible',
            'centered_log_mse', 'bias_squared', 'absrel', 'rmse_m'],
        'bootstrap': '5000 paired place and seed draws; seed 20261110',
        'dynamic': plan['dynamic_policies'], 'dynamic_readers': plan['dynamic_kinds'],
        'dynamic_not_gated_by_static': True,
        'interventions': plan['interventions'], 'retention': plan['retention'],
        'center_trigger': plan['center_trigger'], 'center_policies': plan['center_evaluation_policies'],
        'permutation': 'NumPy default_rng(int(SHA256(soft_read1:20261111:content_permutation:<clip>)[:16],16)); one permutation per clip',
        'joint_permutation_sample': [plan['training_ids'][0], 5, 0],
        'intervention_scope': 'within-model dependence; matched C/Soft fits test comparative utility',
        'fixed_figure_ids': ['transfer_00', 'confirm_00', 'fresh_00'],
        'posthoc_counterexample': 'first catalog clip from primary place with largest Soft-minus-C mixed log-MSE; descriptive only',
        'all_data_previously_used': True, 'scientific_source_sha256': source_identity(),
        'helper_sha256': {name: sha256(ROOT / name) for name in (
            'experiments/gaussian_memory_bound_evaluate.py',
            'experiments/gaussian_memory_incremental_evaluate.py')}}
    target = OUT / 'evaluation_protocol.json'
    if target.exists(): assert json.loads(target.read_text()) == value, 'Evaluation protocol changed.'
    else: write_json(target, value)
    return value


def evaluator_identity():
    return {'evaluation_protocol_sha256': sha256(OUT / 'evaluation_protocol.json'),
            'evaluation_source_sha256': sha256(ROOT / 'experiments/gaussian_memory_soft_evaluate.py')}


def load_adapter(kind, seed):
    if kind == 'content': return load_parent_adapter(kind, seed)
    assert kind in KINDS and seed in SEEDS
    path = OUT / f'{kind}_{seed}/step_2000.pt'
    status = json.loads((path.parent / 'status.json').read_text())
    assert status['complete'] and status['steps'] == 2000
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


def prior_identity(rid, center=False):
    path = center_path(rid, QUERIES[0]).parent / 'identity.json' if center else GAUSSIAN_PRIOR_CACHE / rid / 'identity.json'
    return {'prior_identity_path': str(path), 'prior_identity_sha256': sha256(path)}


def reuse_baselines():
    evaluation_protocol(); plan = load_plan(); manifest = {}; checkpoints = {}
    cases = ['native'] + [f'{k}_{s}' for k in PARENT_KINDS for s in SEEDS]
    for case in cases:
        if case != 'native':
            path = PARENT_OUT / case / 'step_2000.pt'
            status = json.loads((path.parent / 'status.json').read_text())
            assert status['complete'] and sha256(path) == status['checkpoint_sha256']
            checkpoints[case] = status['checkpoint_sha256']
        manifest[case] = {}
        for meta in plan['catalog']:
            rid = meta['id']; source = PARENT_OUT / 'evaluation/static' / case / f'{rid}.json'
            record = json.loads(source.read_text()); assert record['complete'] and len(record['rows']) == 12
            cache_hash = sha256(BASE_CACHE / rid / 'identity.json')
            if case == 'native': assert record['cache_identity_sha256'] == cache_hash
            else:
                assert record['checkpoint_sha256'] == checkpoints[case]
                assert record['sources']['causal_cache_identity_sha256'] == cache_hash
                assert sha256(record['prediction_path']) == record['prediction_sha256']
            digest = sha256(source); manifest[case][rid] = digest
            record.update(soft_reused_from=str(source), soft_reused_record_sha256=digest,
                          soft_plan_sha256=sha256(OUT / 'plan.json'))
            write_json(OUT / 'evaluation/static' / case / f'{rid}.json', record)
    write_json(OUT / 'baseline_reuse.json', {'complete': True, 'new_predictions': 0,
               'clips': len(plan['catalog']), 'parent_records': manifest, 'checkpoints': checkpoints,
               'parent_independent_checks_sha256': sha256(PARENT_OUT / 'independent_checks.json')})


def static(kind='soft', seed=None):
    assert kind in KINDS
    evaluation_protocol(); plan = load_plan(); backbone = load_backbone(); start = time.monotonic()
    for s in ([seed] if seed is not None else SEEDS):
        adapter, checkpoint_hash = load_adapter(kind, s); case = f'{kind}_{s}'
        for index, meta in enumerate(plan['catalog']):
            rid = meta['id']
            if completed('static', case, rid, checkpoint_hash): continue
            begin = time.monotonic(); samples, initial, sources = clip_samples(rid)
            priors = load_priors([rid], center=(kind == 'soft_center'))
            sources.update(prior_identity(rid, kind == 'soft_center')); rows, predictions = [], []
            for sample in samples.values():
                content, prior = inputs_for(sample, initial, priors)
                prediction = predict(backbone, adapter, sample, content, prior)
                rows.append(score(prediction, sample)); predictions.append(prediction)
            save_record('static', case, rid, rows, predictions, checkpoint_hash, sources, time.monotonic() - begin)
            progress('static', case, index, start)
        del adapter
        write_json(OUT / f'static_{case}_complete.json', {'complete': True, 'clips': len(plan['catalog'])})


def content_permutation(rid, slots, device):
    seed = int(hashlib.sha256(f'soft_read1:20261111:content_permutation:{rid}'.encode()).hexdigest()[:16], 16)
    return torch.from_numpy(np.random.default_rng(seed).permutation(slots)).to(device)


def interventions():
    evaluation_protocol(); plan = load_plan(); backbone = load_backbone(); start = time.monotonic()
    adapter, checkpoint_hash = load_adapter('soft', SEEDS[0])
    for index, meta in enumerate(plan['catalog']):
        rid = meta['id']; names = ('content_permuted', 'uniform_prior')
        cases = [f'soft_{SEEDS[0]}_{name}' for name in names]
        if all(completed('intervention', case, rid, checkpoint_hash) for case in cases): continue
        samples, initial, sources = clip_samples(rid); priors = load_priors([rid])
        sources.update(prior_identity(rid)); content = initial['memory']['features']
        permutation = content_permutation(rid, len(content), content.device)
        sources['content_permutation'] = permutation.cpu().tolist()
        for name, case in zip(names, cases):
            if completed('intervention', case, rid, checkpoint_hash): continue
            begin = time.monotonic(); rows, predictions = [], []
            for sample in samples.values():
                projected = priors[rid, sample['query']]
                values = content[permutation] if name == 'content_permuted' else content
                prior = attention_prior(projected, 'content') if name == 'uniform_prior' else soft_attention_prior(projected)
                prediction = predict(backbone, adapter, sample, values, prior)
                row = score(prediction, sample); row['intervention'] = name
                rows.append(row); predictions.append(prediction)
            save_record('intervention', case, rid, rows, predictions, checkpoint_hash, sources, time.monotonic() - begin)
            progress('intervention', case, index, start)
    rid = plan['training_ids'][0]; samples, initial, sources = clip_samples(rid, (0,))
    sample = samples[5, 0]; priors = load_priors([rid]); content = initial['memory']['features']
    p = content_permutation(rid, len(content), content.device); prior = soft_attention_prior(priors[rid, 5])
    columns = torch.cat((p, p.new_tensor([len(content)])))
    original = predict(backbone, adapter, sample, content, prior)
    joint = predict(backbone, adapter, sample, content[p], prior[..., columns])
    permuted = predict(backbone, adapter, sample, content[p], prior)
    write_json(OUT / 'binding_interface.json', {'fixed_sample': [rid, 5, 0], 'checkpoint_sha256': checkpoint_hash,
        'sources': sources, 'joint_content_and_geometry_prior_max_abs': float((original - joint).abs().max()),
        'joint_content_and_geometry_prior_mean_abs': float((original - joint).abs().mean()),
        'content_only_max_abs': float((original - permuted).abs().max()),
        'content_only_mean_abs': float((original - permuted).abs().mean()), 'prediction_mean': float(original.mean()),
        'dtype': 'BF16 forward, FP32 exported depth',
        'scope': 'bounded trained interface diagnostic; no exact BF16 permutation invariance or comparative utility claim'})
    write_json(OUT / 'interventions_complete.json', {'complete': True, 'clips': len(plan['catalog']),
               'fixed_seed': SEEDS[0], 'seconds': time.monotonic() - start})


def state_prior(memory, sample, kind, center_variance):
    projected = projected_slot_weights(memory, sample['c2w'].cuda(), sample['K'].cuda(),
                                       center_variance if kind == 'soft_center' else None)
    return attention_prior(projected, 'content') if kind == 'content' else soft_attention_prior(projected)


def integrated(kind='soft', seed=None, policy=None):
    assert kind in ('content', 'soft', 'soft_center')
    evaluation_protocol(); plan = load_plan(); backbone = load_backbone(); start = time.monotonic()
    policies = ('corrective_clean',) if kind == 'soft_center' else ('corrective_clean', 'rebuild_clean')
    if policy is not None: assert policy in policies; policies = (policy,)
    for s in ([seed] if seed is not None else SEEDS):
        adapter, checkpoint_hash = load_adapter(kind, s)
        for pol in policies:
            case = f'{kind}_{s}_{pol}'
            for index, meta in enumerate(plan['catalog']):
                rid = meta['id']
                if completed('integrated', case, rid, checkpoint_hash): continue
                begin = time.monotonic(); samples, initial, sources = clip_samples(rid)
                del initial
                state_path = STATE_OUT / rid / 'trajectories.pt'
                tracks = torch.load(state_path, map_location='cpu', weights_only=False)
                sources['state_trajectories_sha256'] = sha256(state_path); rows, predictions = [], []
                for ti, q in enumerate(QUERIES, 1):
                    memory = device_tree(tracks[pol]['states'][ti])
                    prior = state_prior(memory, samples[q, 0], kind, plan['center_variance'])
                    for c in range(3):
                        sample = samples[q, c]; prediction = predict(backbone, adapter, sample, memory['features'], prior)
                        row = score(prediction, sample); row['state_key'] = [pol, ti]
                        rows.append(row); predictions.append(prediction)
                save_record('integrated', case, rid, rows, predictions, checkpoint_hash, sources, time.monotonic() - begin)
                progress('integrated', case, index, start)
            write_json(OUT / f'integrated_{case}_complete.json', {'complete': True, 'clips': len(plan['catalog'])})
        del adapter


def retention(kind=None):
    assert kind in (None, 'content', 'soft')
    evaluation_protocol(); plan = load_plan(); backbone = load_backbone(); start = time.monotonic()
    for k in ((kind,) if kind else ('content', 'soft')):
        adapter, checkpoint_hash = load_adapter(k, SEEDS[0]); case = f'{k}_{SEEDS[0]}'
        for index, meta in enumerate(plan['catalog']):
            rid = meta['id']
            if completed('retention', case, rid, checkpoint_hash): continue
            begin = time.monotonic(); samples, initial, sources = clip_samples(rid, (0,)); del initial
            sample = samples[5, 0]; state_path = STATE_OUT / rid / 'trajectories.pt'
            tracks = torch.load(state_path, map_location='cpu', weights_only=False)
            sources['state_trajectories_sha256'] = sha256(state_path); rows, predictions = [], []
            for ti in (0, 1, 4):
                memory = device_tree(tracks['corrective_clean']['states'][ti])
                prior = state_prior(memory, sample, k, plan['center_variance'])
                prediction = predict(backbone, adapter, sample, memory['features'], prior)
                row = score(prediction, sample); row.update(state_key=['corrective_clean', ti], retrospective_retention=True)
                rows.append(row); predictions.append(prediction)
            save_record('retention', case, rid, rows, predictions, checkpoint_hash, sources, time.monotonic() - begin)
            progress('retention', case, index, start)
        write_json(OUT / f'retention_{case}_complete.json', {'complete': True, 'clips': len(plan['catalog'])})
        del adapter


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=('protocol', 'reuse_baselines', 'static', 'integrated', 'interventions', 'retention'))
    p.add_argument('--kind', choices=('content',) + KINDS); p.add_argument('--seed', type=int)
    p.add_argument('--policy', choices=('corrective_clean', 'rebuild_clean')); args = p.parse_args(); torch.set_num_threads(4)
    if args.stage == 'protocol': evaluation_protocol()
    elif args.stage == 'static': static(args.kind or 'soft', args.seed)
    elif args.stage == 'integrated': integrated(args.kind or 'soft', args.seed, args.policy)
    elif args.stage == 'retention': retention(args.kind)
    else: globals()[args.stage]()
