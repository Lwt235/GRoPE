"""Label-free recent/history states with identical Corrective and rebuild rules."""
import argparse
import json
import time

import numpy as np
import torch

from experiments.gaussian_memory_history_data import (
    OUT, CACHE, BASE_CACHE, STATE_READ, STATE_OUT, POLICIES, WRITES, ALLOWED, QUERIES,
    load_plan, checked_load, common_sample, write_json, save_tensor, device_tree, sha256)
from experiments.gaussian_memory_history_geometry import (
    observation_to_common, widen_source_columns, build_common_memory)
from models.gaussian_memory_incremental import (
    clone_memory, observation_frames, concatenate_observations, update_memory as old_update)
from models.gaussian_memory_history import update_memory
from models.gaussian_memory_bound_read import projected_slot_weights


def tensor_bytes(value):
    if isinstance(value, torch.Tensor): return value.numel() * value.element_size()
    if isinstance(value, dict): return sum(tensor_bytes(v) for v in value.values())
    if isinstance(value, (list, tuple)): return sum(tensor_bytes(v) for v in value)
    return 0


@torch.no_grad()
def compatibility():
    plan = load_plan(); records = []; start = time.monotonic()
    for pair in plan['pairs']:
        rid = pair['early_id']; initial = checked_load(rid, 'initial.pt', pair)
        updates = checked_load(rid, 'updates.pt', pair)
        old = device_tree(initial['memory']); new = clone_memory(old)
        wide = widen_source_columns(old, 24); recent = []; count = 0; max_error = {}
        for frame in WRITES:
            if updates[frame] is None: continue
            obs = device_tree(updates[frame]); recent = (recent + [obs])[-4:]
            old, old_log = old_update(old, obs, recent)
            new, new_log = update_memory(new, obs, recent, allowed_frames=WRITES)
            wide, wide_log = update_memory(wide, obs, recent, allowed_frames=WRITES)
            assert old_log == new_log == wide_log
            for key, value in old.items():
                for candidate in (new[key], wide[key][:, :12] if key in ('sources', 'conflicts') else wide[key]):
                    if value.is_floating_point():
                        error = float((value - candidate).abs().max())
                        max_error[key] = max(max_error.get(key, 0.), error)
                        assert torch.allclose(value, candidate, rtol=1e-5, atol=2e-6), (rid, frame, key, error)
                    else: assert torch.equal(value, candidate), (rid, frame, key)
            assert not wide['sources'][:, 12:].any() and not wide['conflicts'][:, 12:].any()
            count += 1
        rejected = dict(obs); rejected['source_frame'] = torch.full_like(obs['source_frame'], 5)
        try: update_memory(new, rejected, [], allowed_frames=WRITES)
        except AssertionError: pass
        else: raise AssertionError('Read-only source not rejected.')
        records.append({'id': rid, 'writes_compared': count, 'event_logs_and_discrete_tensors_exact': True,
                        'float_max_absolute_error': max_error, 'source_padding_invariant': True,
                        'query_source_rejected': True})
    write_json(OUT / 'update_compatibility.json', {'complete': True, 'records': records,
        'seconds': time.monotonic() - start, 'plan_sha256': sha256(OUT / 'plan.json'),
        'scope': 'Old and generalized updater on identical GPU inputs; exact discrete/event equality and float rtol=1e-5, atol=2e-6, including 12-to-24 source padding. Old-vs-old GPU atomic accumulation differences diagnosed separately.'})


def validate_memory(memory, allowed):
    assert memory['means'].shape == (256, 3) and memory['features'].shape == (256, 32)
    assert memory['sources'].shape == memory['conflicts'].shape == (256, 24)
    for key in ('means', 'covars', 'features', 'support', 'opacity'):
        assert torch.isfinite(memory[key]).all(), key
    frames = memory['sources'].any(0).nonzero().flatten().tolist()
    conflicts = memory['conflicts'].any(0).nonzero().flatten().tolist()
    assert set(frames + conflicts).issubset(allowed)
    minimum = float(torch.linalg.eigvalsh(memory['covars']).min())
    assert minimum > 0
    early = memory['sources'][:, :12].any(-1)
    late = memory['sources'][:, 12:].any(-1)
    return {'source_frames': frames, 'conflict_frames': conflicts,
            'slots_with_early_sources': int(early.sum()), 'slots_with_late_sources': int(late.sum()),
            'slots_with_both': int((early & late).sum()), 'covariance_min_eigenvalue': minimum,
            'memory_bytes': tensor_bytes(memory),
            'memory_bytes_by_field': {k: tensor_bytes(v) for k, v in memory.items()}}


@torch.no_grad()
def build_pair(pair):
    start = time.monotonic(); rid = pair['id']; a = pair['early_id']
    ai = checked_load(a, 'initial.pt', pair); au = checked_load(a, 'updates.pt', pair)
    bi = checked_load(rid, 'initial.pt', pair); bu = checked_load(rid, 'updates.pt', pair)
    parent_path = STATE_OUT / a / 'trajectories.pt'
    assert sha256(parent_path) == pair['early_trajectories_sha256']
    parent = torch.load(parent_path, map_location='cpu', weights_only=False)
    early_end = device_tree(widen_source_columns(parent['corrective_clean']['states'][4], 24))
    def converted(obs):
        return device_tree(observation_to_common(obs, pair['common_from_late_metric'],
                    pair['late_unit_meters'], pair['common_unit_meters'], 12))
    binit = converted(bi['observation'])
    bframes = observation_frames(binit)
    late_updates = {f: converted(bu[f]) if bu[f] is not None else None for f in WRITES}
    early_observations = observation_frames(device_tree(ai['observation']))
    early_writes = [device_tree(au[f]) for f in WRITES if au[f] is not None]
    early_observations += early_writes
    tracks = {}; diagnostics = {}
    for policy in POLICIES:
        history = policy.startswith('history_'); corrective = policy.endswith('corrective')
        recent = early_writes[-4:] if history else []
        prefix = early_observations + bframes if history else list(bframes)
        init_logs = []; begin = time.monotonic()
        if history and corrective:
            memory = clone_memory(early_end)
            for obs in bframes:
                recent = (recent + [obs])[-4:]
                memory, event = update_memory(memory, obs, recent, allowed_frames=ALLOWED)
                init_logs.append(event)
        else:
            memory = build_common_memory(concatenate_observations(prefix))
        torch.cuda.synchronize(); init_seconds = time.monotonic() - begin
        # Rebuild may retain all observations offline; Corrective only recent.
        states = [clone_memory(memory)]; logs = []; stats = []
        allow = ([0, 1, 2, 3, *WRITES] if history else []) + [12, 13, 14, 15]
        initial_stats = validate_memory(memory, allow)
        initial_stats.update(recent_buffer_bytes=tensor_bytes(recent) if corrective else 0,
                             offline_observation_bytes=tensor_bytes(prefix) if not corrective else 0)
        for ti, frame in enumerate(WRITES, 1):
            begin = time.monotonic(); obs = late_updates[frame]
            if obs is None:
                event = {'frame': frame + 12, 'skipped': 'unavailable independent camera scale'}
            elif corrective:
                recent = (recent + [obs])[-4:]
                memory, event = update_memory(memory, obs, recent, allowed_frames=ALLOWED)
            else:
                prefix.append(obs); memory = build_common_memory(concatenate_observations(prefix))
                event = {'frame': frame + 12, 'offline_observation_count': sum(len(o['means']) for o in prefix)}
            torch.cuda.synchronize(); event['seconds'] = time.monotonic() - begin
            states.append(clone_memory(memory)); logs.append(event)
            allow += [frame + 12]
            stat = validate_memory(memory, allow)
            stat.update(query=QUERIES[ti - 1], state_index=ti,
                        recent_buffer_bytes=tensor_bytes(recent) if corrective else 0,
                        offline_observation_bytes=tensor_bytes(prefix) if not corrective else 0)
            # Geometry-only coverage diagnostic; never reads labels into memory.
            sample_path = BASE_CACHE / rid / f'q{QUERIES[ti - 1]}_c0.pt'
            identity = json.loads((BASE_CACHE / rid / 'identity.json').read_text())
            assert sha256(sample_path) == identity['sha256'][sample_path.name]
            sample = common_sample(torch.load(sample_path, map_location='cpu', weights_only=False), pair)
            projection = projected_slot_weights(memory, sample['c2w'].cuda(), sample['K'].cuda())
            early = memory['sources'][:, :12].any(-1)
            stat.update(query_mean_coverage=float(projection['coverage'][-1].mean()),
                        query_covered_fraction=float((projection['coverage'][-1] > .1).float().mean()),
                        query_mass_on_early_source_slots=float(projection['weights'][-1, :, early].sum(-1).mean()),
                        all_current_views_mean_coverage=float(projection['coverage'].mean()))
            stats.append(stat)
        tracks[policy] = {'states': states, 'initialization_events': init_logs, 'updates': logs,
                          'initialization_seconds': init_seconds,
                          'recent_final': recent if corrective else [],
                          'initial_stats': initial_stats}
        diagnostics[policy] = stats
    # State reuse for retrospective baseline is exact, not recomputed.
    tracks['early_corrective_end'] = early_end
    path = CACHE / 'states' / rid / 'trajectories.pt'; path.parent.mkdir(parents=True, exist_ok=True)
    save_tensor(path, tracks)
    write_json(OUT / 'states' / rid / 'diagnostics.json', diagnostics)
    write_json(OUT / 'states' / rid / 'updates.json', {k: {'initialization_events': tracks[k]['initialization_events'],
                'updates': tracks[k]['updates'], 'initialization_seconds': tracks[k]['initialization_seconds']} for k in POLICIES})
    result = {'complete': True, 'id': rid, 'early_id': a, 'plan_sha256': sha256(OUT / 'plan.json'),
        'trajectories_path': str(path), 'trajectories_sha256': sha256(path),
        'diagnostics_sha256': sha256(OUT / 'states' / rid / 'diagnostics.json'),
        'seconds': time.monotonic() - start, 'query_frames_never_written': True,
        'labels_used_for_updates': False, 'late_initialization_available_at': 15,
        'early_parent_trajectories_sha256': pair['early_trajectories_sha256'],
        'peak_allocated_GiB': torch.cuda.max_memory_allocated() / 2**30}
    write_json(OUT / 'states' / rid / 'complete.json', result)
    print(json.dumps(result), flush=True)


def main():
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=('compatibility', 'build', 'all'))
    args = p.parse_args(); torch.set_num_threads(4)
    if args.stage in ('compatibility', 'all'): compatibility()
    if args.stage in ('build', 'all'):
        assert json.loads((OUT / 'update_compatibility.json').read_text())['complete']
        for pair in load_plan()['pairs']:
            marker = OUT / 'states' / pair['id'] / 'complete.json'
            if marker.exists():
                from experiments.gaussian_memory_history_data import load_tracks
                load_tracks(pair['id']); continue
            build_pair(pair)
        write_json(OUT / 'states_complete.json', {'complete': True, 'pairs': 12, 'snapshots': 240})


if __name__ == '__main__': main()
