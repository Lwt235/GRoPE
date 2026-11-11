"""Fixed native-output comparisons for per-slot Gaussian memory addressing."""
import argparse
import hashlib
import json
import time

import numpy as np
import torch

from experiments.gaussian_memory_setup import ROOT, sha256
from experiments.gaussian_memory_view_data import load_backbone
from experiments.gaussian_memory_bound_data import (
    OUT, CACHE, BASE_CACHE, BASE_OUT, SEEDS, KINDS, QUERIES,
    load_plan, load_priors, inputs_for, write_json, save_tensor, device_tree)
from experiments.gaussian_memory_bound_train import source_identity
from experiments.gaussian_memory_incremental_evaluate import score
from experiments.gaussian_memory_incremental_states import STATE_OUT, AMPLITUDES, track_name
from models.gaussian_memory_bound_read import (
    BoundMemoryAdapter, bound_native_tail, projected_slot_weights, attention_prior)

PREDICTIONS = CACHE / 'predictions'


def evaluation_protocol():
    value = {
        'schema': 1, 'experiment': 'bound_read1',
        'defined_before_formal_evaluation': True,
        'plan_sha256': sha256(OUT / 'plan.json'),
        'primary': 'place-equal full/mask0/mask1 mixed log-MSE on 51 nontraining places',
        'primary_comparisons': ['bound vs coverage', 'bound vs local', 'bound vs content', 'coverage vs content'],
        'seeds': list(SEEDS), 'bootstrap': '5000 paired place and seed draws; seed 20261110',
        'cohorts': ['old12', 'used_summary20', 'used_writer19'],
        'secondary': ['full', 'masked', 'hidden', 'visible', 'centered_log_mse', 'bias_squared', 'absrel', 'rmse_m'],
        'binding_interventions': ['content_permuted', 'coverage_prior'],
        'intervention_seed': SEEDS[0],
        'permutation': 'NumPy default_rng(int(SHA256(bound_read1:20261111:content_permutation:<clip>)[:16],16)); same permutation for all queries/conditions of a clip',
        'joint_permutation_sample': {'id': load_plan()['training_ids'][0], 'query': 5, 'condition': 0},
        'intervention_claim': 'within-model dependence diagnostic; distribution shift is a competing explanation, matched A/B training establishes comparative utility',
        'center_trigger': load_plan()['center_trigger'],
        'error_probe_trigger': load_plan()['error_probe_trigger'],
        'clean_integration': 'coverage and bound, all three fixed readers, corrective and prefix states; static reused',
        'error_and_retention': 'conditional bound first seed, full input, all four signed amplitudes and three policies; retrospective query5 clean-state retention',
        'fixed_figure_ids': ['transfer_00', 'confirm_00', 'fresh_00'],
        'posthoc_counterexample': 'first catalog clip of the primary place with largest bound-minus-local mean log-MSE; descriptive only',
        'all_data_previously_used': True,
        'scientific_source_sha256': source_identity(),
        'scoring_source_sha256': sha256(ROOT / 'experiments/gaussian_memory_incremental_evaluate.py')}
    dest = OUT / 'evaluation_protocol.json'
    if dest.exists():
        assert json.loads(dest.read_text()) == value, 'Evaluation protocol changed.'
    else:
        write_json(dest, value)
    return value


def content_permutation(rid, slots, device='cpu'):
    seed = int(hashlib.sha256(f'bound_read1:20261111:content_permutation:{rid}'.encode()).hexdigest()[:16], 16)
    return torch.from_numpy(np.random.default_rng(seed).permutation(slots)).to(device)


def evaluator_identity():
    return {'evaluation_protocol_sha256': sha256(OUT / 'evaluation_protocol.json'),
            'evaluation_source_sha256': sha256(ROOT / 'experiments/gaussian_memory_bound_evaluate.py')}


def completed(phase, case, rid, checkpoint_hash):
    path = OUT / 'evaluation' / phase / case / f'{rid}.json'
    if not path.exists():
        return False
    record = json.loads(path.read_text())
    assert record['complete'] and record['checkpoint_sha256'] == checkpoint_hash
    assert all(record[k] == v for k, v in evaluator_identity().items())
    assert sha256(record['prediction_path']) == record['prediction_sha256']
    return True


def save_record(phase, case, rid, rows, predictions, checkpoint_hash, sources, seconds):
    folder = PREDICTIONS / phase / case
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f'{rid}.pt'
    save_tensor(target, {'prediction_native': torch.stack(predictions),
                        'sample_keys': [[r['query'], r['condition']] for r in rows],
                        'state_keys': [r.get('state_key') for r in rows]})
    write_json(OUT / 'evaluation' / phase / case / f'{rid}.json', {
        'complete': True, 'rows': rows, 'prediction_path': str(target),
        'prediction_sha256': sha256(target), 'checkpoint_sha256': checkpoint_hash,
        'sources': sources, 'seconds': seconds, **evaluator_identity()})


def load_adapter(kind, seed):
    path = OUT / f'{kind}_{seed}/step_2000.pt'
    state = torch.load(path, map_location='cpu', weights_only=False)
    status = json.loads((path.parent / 'status.json').read_text())
    assert status['complete'] and sha256(path) == status['checkpoint_sha256']
    assert state['source_sha256'] == source_identity() and state['step'] == 2000
    adapter = BoundMemoryAdapter().cuda().eval().requires_grad_(False)
    adapter.load_state_dict(state['model'])
    return adapter, status['checkpoint_sha256']


def clip_samples(rid, conditions=(0, 1, 2)):
    identity_path = BASE_CACHE / rid / 'identity.json'
    identity = json.loads(identity_path.read_text())
    samples = {}
    for query in QUERIES:
        for condition in conditions:
            name = f'q{query}_c{condition}.pt'
            assert sha256(BASE_CACHE / rid / name) == identity['sha256'][name]
            samples[query, condition] = torch.load(BASE_CACHE / rid / name, map_location='cpu', weights_only=False)
    assert sha256(BASE_CACHE / rid / 'initial.pt') == identity['sha256']['initial.pt']
    initial = device_tree(torch.load(BASE_CACHE / rid / 'initial.pt', map_location='cpu', weights_only=False))
    sources = {'causal_cache_identity_sha256': sha256(identity_path),
               'initial_sha256': identity['sha256']['initial.pt']}
    return samples, initial, sources


@torch.no_grad()
def predict(backbone, adapter, sample_cpu, content, prior):
    sample = device_tree(sample_cpu)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        depth, _ = bound_native_tail(backbone, sample, adapter, content, prior)
    return depth.float().cpu()


def progress(phase, case, index, start):
    value = {'stage': phase, 'case': case, 'clips_complete': index + 1,
             'seconds': time.monotonic() - start}
    # Each concurrently evaluated case owns its own progress file.
    write_json(OUT / 'evaluation_progress' / f'{phase}_{case}.json', value)
    if (index + 1) % 10 == 0:
        print(json.dumps(value), flush=True)


def baseline():
    evaluation_protocol(); plan = load_plan()
    for meta in plan['catalog']:
        rid = meta['id']; source = BASE_OUT / 'evaluation/static/native' / f'{rid}.json'
        record = json.loads(source.read_text())
        assert record['complete'] and len(record['rows']) == 12
        assert record['cache_identity_sha256'] == sha256(BASE_CACHE / rid / 'identity.json')
        record.update(reused_from=str(source), reused_record_sha256=sha256(source),
                      current_plan_sha256=sha256(OUT / 'plan.json'))
        write_json(OUT / 'evaluation/static/native' / f'{rid}.json', record)
    write_json(OUT / 'baseline_complete.json', {'complete': True, 'clips': len(plan['catalog']), 'new_predictions': 0})


def static(kind=None, seed=None):
    evaluation_protocol(); plan = load_plan(); backbone = load_backbone(); start = time.monotonic()
    for k in ([kind] if kind else KINDS):
        for s in ([seed] if seed is not None else SEEDS):
            adapter, checkpoint_hash = load_adapter(k, s); case = f'{k}_{s}'
            for index, meta in enumerate(plan['catalog']):
                rid = meta['id']
                if completed('static', case, rid, checkpoint_hash):
                    continue
                begin = time.monotonic(); samples, initial, sources = clip_samples(rid)
                priors = load_priors([rid], center=(k == 'center'))
                prior_identity = CACHE / rid / ('center_identity.json' if k == 'center' else 'identity.json')
                sources['prior_identity_sha256'] = sha256(prior_identity)
                rows, predictions = [], []
                for sample in samples.values():
                    content, prior = inputs_for(sample, initial, k, priors)
                    # The local content in a CPU sample needs the same CUDA placement as historical content.
                    prediction = predict(backbone, adapter, sample, content.cuda(), prior)
                    rows.append(score(prediction, sample)); predictions.append(prediction)
                save_record('static', case, rid, rows, predictions, checkpoint_hash, sources, time.monotonic() - begin)
                progress('static', case, index, start)
            del adapter
            write_json(OUT / f'static_{case}_complete.json', {'complete': True, 'clips': len(plan['catalog'])})
    write_json(OUT / f'static_{kind or "all"}_{seed or "all"}_complete.json', {'complete': True, 'seconds': time.monotonic() - start})


def interventions():
    evaluation_protocol(); plan = load_plan(); backbone = load_backbone(); start = time.monotonic()
    adapter, checkpoint_hash = load_adapter('bound', SEEDS[0])
    for index, meta in enumerate(plan['catalog']):
        rid = meta['id']
        cases = [f'bound_{SEEDS[0]}_{name}' for name in ('content_permuted', 'coverage_prior')]
        if all(completed('intervention', case, rid, checkpoint_hash) for case in cases):
            continue
        samples, initial, sources = clip_samples(rid); priors = load_priors([rid])
        sources['prior_identity_sha256'] = sha256(CACHE / rid / 'identity.json')
        content = initial['memory']['features']
        permutation = content_permutation(rid, len(content), content.device)
        sources['content_permutation'] = permutation.cpu().tolist()
        for intervention, case in zip(('content_permuted', 'coverage_prior'), cases):
            if completed('intervention', case, rid, checkpoint_hash):
                continue
            begin = time.monotonic(); rows, predictions = [], []
            for sample in samples.values():
                projected = priors[rid, sample['query']]
                values = content[permutation] if intervention == 'content_permuted' else content
                prior = attention_prior(projected, 'coverage' if intervention == 'coverage_prior' else 'bound')
                prediction = predict(backbone, adapter, sample, values, prior)
                row = score(prediction, sample); row['intervention'] = intervention
                rows.append(row); predictions.append(prediction)
            save_record('intervention', case, rid, rows, predictions, checkpoint_hash, sources, time.monotonic() - begin)
            progress('intervention', case, index, start)
    # Fixed trained-model equivalence check; not a statistical efficacy gate.
    rid = plan['training_ids'][0]; samples, initial, sources = clip_samples(rid, (0,))
    sample = samples[5, 0]; priors = load_priors([rid]); projected = priors[rid, 5]
    content = initial['memory']['features']; p = content_permutation(rid, len(content), content.device)
    prior = attention_prior(projected, 'bound'); columns = torch.cat((p, p.new_tensor([len(content)])))
    original = predict(backbone, adapter, sample, content, prior)
    joint = predict(backbone, adapter, sample, content[p], prior[..., columns])
    permuted = predict(backbone, adapter, sample, content[p], prior)
    write_json(OUT / 'binding_interface.json', {
        'fixed_sample': [rid, 5, 0], 'checkpoint_sha256': checkpoint_hash, 'sources': sources,
        'joint_content_and_geometry_prior_max_abs': float((original - joint).abs().max()),
        'joint_content_and_geometry_prior_mean_abs': float((original - joint).abs().mean()),
        'content_only_max_abs': float((original - permuted).abs().max()),
        'content_only_mean_abs': float((original - permuted).abs().mean()),
        'prediction_mean': float(original.mean()), 'dtype': 'BF16 forward, FP32 exported depth',
        'interpretation': 'fixed trained-model interface diagnostic, not comparative utility'})
    write_json(OUT / 'interventions_complete.json', {'complete': True, 'clips': len(plan['catalog']),
               'seconds': time.monotonic() - start, 'fixed_seed': SEEDS[0]})


def integrated(kind=None, seed=None):
    evaluation_protocol(); plan = load_plan(); backbone = load_backbone(); start = time.monotonic()
    assert kind in (None, 'coverage', 'bound')
    for k in ([kind] if kind else ('coverage', 'bound')):
        for s in ([seed] if seed is not None else SEEDS):
            adapter, checkpoint_hash = load_adapter(k, s)
            for policy in ('corrective_clean', 'rebuild_clean'):
                case = f'{k}_{s}_{policy}'
                for index, meta in enumerate(plan['catalog']):
                    rid = meta['id']
                    if completed('integrated', case, rid, checkpoint_hash):
                        continue
                    begin = time.monotonic(); samples, _, sources = clip_samples(rid)
                    state_path = STATE_OUT / rid / 'trajectories.pt'
                    tracks = torch.load(state_path, map_location='cpu', weights_only=False)
                    sources['state_trajectories_sha256'] = sha256(state_path)
                    rows, predictions = [], []
                    for ti, q in enumerate(QUERIES, 1):
                        memory = device_tree(tracks[policy]['states'][ti]); sample = samples[q, 0]
                        projected = projected_slot_weights(memory, sample['c2w'].cuda(), sample['K'].cuda())
                        prior = attention_prior(projected, k, len(memory['features']))
                        for c in range(3):
                            sample = samples[q, c]
                            prediction = predict(backbone, adapter, sample, memory['features'], prior)
                            row = score(prediction, sample); row['state_key'] = [policy, ti]
                            rows.append(row); predictions.append(prediction)
                    save_record('integrated', case, rid, rows, predictions, checkpoint_hash, sources, time.monotonic() - begin)
                    progress('integrated', case, index, start)
                write_json(OUT / f'integrated_{case}_complete.json', {'complete': True, 'clips': len(plan['catalog'])})
            del adapter
    write_json(OUT / f'integrated_{kind or "all"}_{seed or "all"}_complete.json', {'complete': True, 'seconds': time.monotonic() - start})


def error_probe():
    evaluation_protocol(); plan = load_plan()
    decision = json.loads((OUT / 'static_analysis.json').read_text())
    assert decision['error_probe_triggered'], 'The prespecified native-utility condition is unmet.'
    backbone = load_backbone(); start = time.monotonic()
    adapter, checkpoint_hash = load_adapter('bound', SEEDS[0]); case = f'bound_{SEEDS[0]}'
    policies = ['average_clean'] + [track_name(st, a) for a in AMPLITUDES for st in ('static', 'average', 'corrective')]
    for index, meta in enumerate(plan['catalog']):
        rid = meta['id']
        if all(completed(phase, case, rid, checkpoint_hash) for phase in ('error_probe', 'retention')):
            continue
        samples, _, sources = clip_samples(rid, (0,))
        state_path = STATE_OUT / rid / 'trajectories.pt'; region_path = STATE_OUT / rid / 'selected_region.pt'
        tracks = torch.load(state_path, map_location='cpu', weights_only=False)
        regions = torch.load(region_path, map_location='cpu', weights_only=False)['query_patch_masks']
        sources.update(state_trajectories_sha256=sha256(state_path), selected_region_sha256=sha256(region_path))
        for qi, q in enumerate(QUERIES):
            samples[q, 0]['selected_region'] = regions[qi].reshape(15, 20).repeat_interleave(14, 0).repeat_interleave(14, 1)
        for phase in ('error_probe', 'retention'):
            if completed(phase, case, rid, checkpoint_hash):
                continue
            begin = time.monotonic(); rows, predictions = [], []
            phase_policies = policies if phase == 'error_probe' else ('static_clean', 'average_clean', 'corrective_clean', 'rebuild_clean')
            for policy in phase_policies:
                indices = enumerate(QUERIES, 1) if phase == 'error_probe' else [(t, 5) for t in range(5)]
                for ti, q in indices:
                    memory = device_tree(tracks[policy]['states'][ti]); sample = samples[q, 0]
                    projected = projected_slot_weights(memory, sample['c2w'].cuda(), sample['K'].cuda())
                    prior = attention_prior(projected, 'bound', len(memory['features']))
                    prediction = predict(backbone, adapter, sample, memory['features'], prior)
                    row = score(prediction, sample); row['state_key'] = [policy, ti]
                    if phase == 'retention': row['retrospective_retention'] = True
                    rows.append(row); predictions.append(prediction)
            save_record(phase, case, rid, rows, predictions, checkpoint_hash, sources, time.monotonic() - begin)
            progress(phase, case, index, start)
    write_json(OUT / 'error_probe_complete.json', {'complete': True, 'clips': len(plan['catalog']),
               'seconds': time.monotonic() - start, 'fixed_seed': SEEDS[0], 'condition': 'full'})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=('protocol', 'baseline', 'static', 'interventions', 'integrated', 'error_probe'))
    p.add_argument('--kind', choices=KINDS + ('center',)); p.add_argument('--seed', type=int)
    args = p.parse_args(); torch.set_num_threads(4)
    if args.stage == 'protocol': evaluation_protocol()
    elif args.stage in ('static', 'integrated'): globals()[args.stage](args.kind, args.seed)
    else: globals()[args.stage]()
