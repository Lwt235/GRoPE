"""Frozen-model evaluation; current RGB/prefix/scale stay identical in all cases."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.gaussian_memory_history_data import (
    ROOT, OUT, CACHE, STATE_READ, SOFT_READ, BASE_CACHE, FAMILIES, POLICIES, SEEDS, QUERIES,
    load_plan, load_tracks, common_sample, reader_kind, checkpoint_path,
    sha256, write_json, save_tensor, device_tree)
from experiments.gaussian_memory_view_data import load_backbone
from experiments.gaussian_memory_bound_evaluate import clip_samples, predict
from experiments.gaussian_memory_soft_evaluate import content_permutation, load_adapter as load_static
from experiments.gaussian_memory_state_evaluate import load_adapter as load_dynamic
from experiments.gaussian_memory_state_data import memory_prior
from experiments.gaussian_memory_incremental_evaluate import score


def identity():
    return {'plan_sha256': sha256(OUT / 'plan.json'),
            'evaluator_sha256': sha256(ROOT / 'experiments/gaussian_memory_history_evaluate.py')}


def load_adapter(family, seed):
    adapter, digest = (load_dynamic if family.endswith('_dynamic') else load_static)(reader_kind(family), seed)
    assert digest == load_plan()['checkpoints'][f'{family}_{seed}']['sha256']
    return adapter, digest


def completed(phase, case, rid, checkpoint_hash):
    path = OUT / 'evaluation' / phase / case / f'{rid}.json'
    if not path.exists(): return False
    record = json.loads(path.read_text())
    assert record['complete'] and record['checkpoint_sha256'] == checkpoint_hash
    assert all(record[k] == v for k, v in identity().items())
    assert sha256(record['prediction_path']) == record['prediction_sha256']
    return True


def save_record(phase, case, rid, rows, predictions, checkpoint_hash, sources, seconds, inference_seconds):
    target = CACHE / 'predictions' / phase / case / f'{rid}.pt'; target.parent.mkdir(parents=True, exist_ok=True)
    save_tensor(target, {'prediction_native': torch.stack(predictions),
                        'sample_keys': [[r['query'], r['condition']] for r in rows],
                        'state_keys': [r['state_key'] for r in rows]})
    write_json(OUT / 'evaluation' / phase / case / f'{rid}.json', {
        'complete': True, 'rows': rows, 'prediction_path': str(target),
        'prediction_sha256': sha256(target), 'checkpoint_sha256': checkpoint_hash,
        'sources': sources, 'seconds': seconds, 'inference_seconds': inference_seconds,
        'peak_allocated_GiB': torch.cuda.max_memory_allocated() / 2**30, **identity()})


def reuse():
    plan = load_plan(); manifest = []; total = 0
    for pair in plan['pairs']:
        rid, a = pair['id'], pair['early_id']
        for case in ['native'] + [f'local_{s}' for s in SEEDS]:
            path = SOFT_READ / 'evaluation/static' / case / f'{rid}.json'
            old = json.loads(path.read_text()); assert old['complete'] and len(old['rows']) == 12
            source_identity = old.get('cache_identity_sha256', old.get('sources', {}).get('causal_cache_identity_sha256'))
            assert source_identity == pair['inputs'][rid]['identity_sha256']
            if case != 'native': assert sha256(old['prediction_path']) == old['prediction_sha256']
            record = dict(old); record.update(reused_from=str(path), reused_record_sha256=sha256(path), **identity())
            write_json(OUT / 'evaluation/reused' / case / f'{rid}.json', record)
            manifest.append({'phase': 'reused', 'case': case, 'id': rid, 'source': str(path), 'sha256': sha256(path), 'rows': 12})
            total += 12
        for family in FAMILIES:
            for seed in SEEDS:
                kind = reader_kind(family); case = f'{family}_{seed}'
                folder = STATE_READ if family.endswith('_dynamic') else SOFT_READ
                phase = 'read' if family.endswith('_dynamic') else 'integrated'
                path = folder / 'evaluation' / phase / f'{kind}_{seed}_corrective_clean' / f'{a}.json'
                old = json.loads(path.read_text()); assert old['complete']
                assert old['sources']['state_trajectories_sha256'] == pair['early_trajectories_sha256']
                assert old['checkpoint_sha256'] == plan['checkpoints'][case]['sha256']
                assert sha256(old['prediction_path']) == old['prediction_sha256']
                selected = [i for i, r in enumerate(old['rows']) if r['query'] == 11 and r['condition'] == 0]
                assert len(selected) == 1
                row = old['rows'][selected[0]]
                record = dict(old); record.update(rows=[row], reused_from=str(path), reused_record_sha256=sha256(path),
                    parent_prediction_indices=selected, early_state_reuse='A Corrective final state exact, padded false source columns only', **identity())
                write_json(OUT / 'evaluation/retention_baseline' / case / f'{a}.json', record)
                manifest.append({'phase': 'retention_baseline', 'case': case, 'id': a, 'source': str(path), 'sha256': sha256(path), 'rows': 1})
                total += 1
    assert total == 756
    write_json(OUT / 'baseline_reuse.json', {'complete': True, 'rows': total, 'records': manifest,
        'parent_checks': {str(p): sha256(p) for p in (STATE_READ / 'independent_checks.json', SOFT_READ / 'independent_checks.json')}})


def evaluate(seed, family=None):
    plan = load_plan(); backbone = load_backbone(); start = time.monotonic()
    for fam in ([family] if family else FAMILIES):
        adapter, digest = load_adapter(fam, seed)
        for index, pair in enumerate(plan['pairs']):
            rid = pair['id']; cases = [f'{fam}_{seed}_{p}' for p in POLICIES]
            if all(completed('read', case, rid, digest) for case in cases): continue
            samples, initial, sources = clip_samples(rid); del initial
            samples = {k: common_sample(v, pair) for k, v in samples.items()}
            tracks, track_identity = load_tracks(rid)
            sources.update(history_trajectories_sha256=track_identity['trajectories_sha256'],
                           common_gauge_pair_id=rid)
            for policy, case in zip(POLICIES, cases):
                if completed('read', case, rid, digest): continue
                begin = time.monotonic(); rows = []; predictions = []; inference_seconds = 0.
                for ti, q in enumerate(QUERIES, 1):
                    memory = device_tree(tracks[policy]['states'][ti])
                    prior = memory_prior(memory, samples[q, 0], reader_kind(fam), plan['center_variance'])
                    for c in range(3):
                        sample = samples[q, c]; t = time.monotonic()
                        prediction = predict(backbone, adapter, sample, memory['features'], prior)
                        inference_seconds += time.monotonic() - t
                        row = score(prediction, sample); row['state_key'] = [policy, ti]
                        rows.append(row); predictions.append(prediction)
                save_record('read', case, rid, rows, predictions, digest, sources,
                            time.monotonic() - begin, inference_seconds)
            progress = {'family': fam, 'seed': seed, 'clips_complete': index + 1,
                        'last_clip': rid, 'seconds': time.monotonic() - start}
            write_json(OUT / 'progress' / f'read_{fam}_{seed}.json', progress); print(json.dumps(progress), flush=True)
        del adapter
    write_json(OUT / f'read_{family or "all"}_{seed}_complete.json', {'complete': True, 'seed': seed})


def interventions():
    plan = load_plan(); backbone = load_backbone(); adapter, digest = load_adapter('soft_dynamic', SEEDS[0])
    for pair in plan['pairs']:
        rid = pair['id']; samples, initial, sources = clip_samples(rid); del initial
        samples = {k: common_sample(v, pair) for k, v in samples.items()}
        tracks, meta = load_tracks(rid); sources['history_trajectories_sha256'] = meta['trajectories_sha256']
        permutation = content_permutation(rid, 256, 'cuda'); sources['content_permutation'] = permutation.cpu().tolist()
        for name in ('content_permuted', 'uniform_prior'):
            case = f'soft_dynamic_{SEEDS[0]}_{name}'
            if completed('intervention', case, rid, digest): continue
            begin = time.monotonic(); rows = []; predictions = []; inference_seconds = 0.
            for ti, q in enumerate(QUERIES, 1):
                memory = device_tree(tracks['history_corrective']['states'][ti])
                content = memory['features'][permutation] if name == 'content_permuted' else memory['features']
                prior = memory_prior(memory, samples[q, 0], 'content' if name == 'uniform_prior' else 'soft', plan['center_variance'])
                for c in range(3):
                    sample = samples[q, c]; t = time.monotonic(); prediction = predict(backbone, adapter, sample, content, prior)
                    inference_seconds += time.monotonic() - t
                    row = score(prediction, sample); row.update(state_key=['history_corrective', ti], intervention=name)
                    rows.append(row); predictions.append(prediction)
            save_record('intervention', case, rid, rows, predictions, digest, sources, time.monotonic() - begin, inference_seconds)
        print(json.dumps({'interventions_complete_clip': rid}), flush=True)
    write_json(OUT / 'interventions_complete.json', {'complete': True, 'predictions': 288})


def retention(seed):
    plan = load_plan(); backbone = load_backbone()
    for fam in FAMILIES:
        adapter, digest = load_adapter(fam, seed); case = f'{fam}_{seed}'
        for pair in plan['pairs']:
            a = pair['early_id']
            if completed('retention', case, a, digest): continue
            begin = time.monotonic(); samples, initial, sources = clip_samples(a, (0,)); del initial
            sample = samples[11, 0]; tracks, meta = load_tracks(pair['id'])
            sources.update(history_trajectories_sha256=meta['trajectories_sha256'], late_clip_id=pair['id'])
            memory = device_tree(tracks['history_corrective']['states'][4])
            prior = memory_prior(memory, sample, reader_kind(fam), plan['center_variance'])
            t = time.monotonic(); prediction = predict(backbone, adapter, sample, memory['features'], prior)
            seconds = time.monotonic() - t
            row = score(prediction, sample); row.update(state_key=['history_corrective', 4], retrospective_retention=True)
            save_record('retention', case, a, [row], [prediction], digest, sources, time.monotonic() - begin, seconds)
        del adapter
    write_json(OUT / f'retention_{seed}_complete.json', {'complete': True, 'predictions': 60})


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=('reuse', 'evaluate', 'interventions', 'retention'))
    p.add_argument('--seed', type=int, choices=SEEDS); p.add_argument('--family', choices=FAMILIES)
    args = p.parse_args(); torch.set_num_threads(4)
    if args.stage == 'reuse': reuse()
    elif args.stage == 'evaluate':
        assert args.seed is not None; evaluate(args.seed, args.family)
    elif args.stage == 'retention':
        assert args.seed is not None; retention(args.seed)
    else: interventions()
