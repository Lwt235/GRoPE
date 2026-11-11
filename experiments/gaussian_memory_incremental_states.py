"""Fixed, label-free update trajectories plus separately isolated state probes."""
import argparse
import json
import time

import numpy as np
import torch

from experiments.gaussian_memory_setup import ROOT, sha256
from experiments.gaussian_memory_incremental_data import (
    OUT, CACHE, QUERIES, WRITES, load_plan, select_rows, write_json, save_tensor, sensor_target, build_memory)
from models.gaussian_memory_incremental import (
    clone_memory, concatenate_observations, observation_frames, update_memory, perturb_memory)
from models.gaussian_memory_vggt_adapter import project_memory
from models.gaussian_memory_transport import GeometryTransport
from models.gaussian_memory_compact_writer import CompactGaussianWriter

STATE_OUT = OUT / 'state_experiment'
AMPLITUDES = (-.2, .2, -.05, .05)


def protocol():
    source = ROOT / 'models/gaussian_memory_incremental.py'
    checkpoint = ROOT / 'results/gaussian_memory/joint_write1/content_gaussian_20261110/step_2000.pt'
    value = {'hypothesis': 'Independent later observations can add coverage and correct a bounded Gaussian state.',
             'created_before_state_results': True, 'shape_is_not_uncertainty': True,
             'slots': 256, 'new_observation_weight': 1., 'history_weight_cap': 4.,
             'relative_depth_agreement': .1, 'free_space_depth_ratio': .9,
             'association_mahalanobis_cutoff': 9., 'conflict_distinct_frames': 2,
             'conflict_support_decay': .5, 'minimum_support': 1.,
             'new_candidate_minimum_frames': 2, 'merge_distance_over_trace_limit': .25,
             'amplitudes': AMPLITUDES, 'perturbed_slots': 52,
             'sign_interpretation': 'negative is toward first camera, positive away; report separately and both, without selecting a favorable sign',
             'recent_buffer': 'starts empty after initialization; at most four new writable frames',
             'initial_observation_access': 'incremental updater never sees unperturbed raw initialization; prefix rebuild may access it as an explicit offline reference',
             'plain_average_control': 'same association and novel-region merging; no cap, no conflict decay or local contradiction reset',
             'query_depth_access': 'probes only, never update or candidate selection',
             'external_probe': str(checkpoint), 'external_probe_sha256': sha256(checkpoint),
             'external_probe_scope': 'first historical seed, full S1 input only; auxiliary, not internal-VGGT evidence',
             'internal_integration': 'C/CG all three seeds on clean static/incremental/prefix trajectories and all conditions',
             'internal_error_probe': 'fixed CG seed20261110, full input; four signed amplitudes and three strategies; no retraining',
             'geometry_content_boundary': 'Native projected PE and content attention are separate; permuting content among Gaussian slots alone is invariant in this first adapter. No claim of per-slot geometry-conditioned content routing.',
             'source_sha256': {str(source.relative_to(ROOT)): sha256(source),
                               'experiments/gaussian_memory_incremental_states.py': sha256(__file__)}}
    dest = STATE_OUT / 'protocol.json'
    if dest.exists():
        assert json.loads(dest.read_text()) == json.loads(json.dumps(value))
    else:
        write_json(dest, value)
    return value


def trajectory(initial, updates, strategy, amplitude=0.):
    mem = clone_memory(initial['memory'])
    selected = []
    if amplitude:
        mem, selected = perturb_memory(mem, initial['c2w'], amplitude)
        selected = selected.tolist()
    states = [clone_memory(mem)]; logs = []; recent = []
    prefix = observation_frames(initial['observation']) if strategy == 'rebuild' else None
    for frame in WRITES:
        obs = updates[frame]
        if obs is None:
            logs.append({'frame': frame, 'skipped': 'unavailable independent camera scale'})
        elif strategy == 'static':
            logs.append({'frame': frame, 'write_disabled': True})
        elif strategy == 'rebuild':
            prefix.append(obs)
            mem = build_memory(concatenate_observations(prefix))
            logs.append({'frame': frame, 'source_frames': sorted(set(mem['sources'].nonzero()[:, 1].tolist())),
                         'all_prefix_observations': sum(len(o['means']) for o in prefix)})
        else:
            recent = (recent + [obs])[-4:]
            mem, diagnostic = update_memory(mem, obs, recent, strategy)
            logs.append(diagnostic)
        states.append(clone_memory(mem))
    return {'states': states, 'updates': logs, 'strategy': strategy,
            'amplitude': amplitude, 'perturbed_indices': selected}


def track_name(strategy, amplitude):
    return f'{strategy}_{"clean" if amplitude == 0 else ("plus" if amplitude > 0 else "minus") + str(round(abs(amplitude) * 100)).zfill(2)}'


def coarse_targets(row):
    targets, masks = [], []
    for q in QUERIES:
        d, valid = sensor_target(row['frames'][q])
        values = d.numpy().reshape(15, 14, 20, 14).transpose(0, 2, 1, 3).reshape(300, 196)
        valid = valid.numpy().reshape(15, 14, 20, 14).transpose(0, 2, 1, 3).reshape(300, 196)
        tmp = np.where(valid, values, np.nan)
        tmp[~valid.any(-1), 0] = 1
        targets.append(torch.from_numpy(np.nanmedian(tmp, -1)).float())
        masks.append(torch.from_numpy(valid.mean(-1) >= .5))
    return torch.stack(targets), torch.stack(masks)


def mean_or_none(values):
    return float(values.mean()) if values.numel() else None


@torch.no_grad()
def probes(tracks, initial, row, teacher, reader, writer):
    K = initial['K'][list(QUERIES)]; cameras = initial['c2w'][list(QUERIES)]
    unit = initial['unit_meters']; target, valid = coarse_targets(row)
    clean = tracks['static_clean']['states'][0]
    fixed = project_memory(clean, cameras, K)['coverage'] > .05
    fixed &= valid
    outputs = {}; projections = {}
    for name, track in tracks.items():
        outputs[name] = []; projections[name] = []
        for t, mem in enumerate(track['states']):
            projected = project_memory(mem, cameras, K)
            depth = projected['depth'].clamp_min(1e-6) * unit
            log_error = depth.log() - target.log()
            coverage = projected['coverage'] > .05
            eig = torch.linalg.eigvalsh(mem['covars'])
            assert eig.min() > 0, (name, t, float(eig.min()))
            seeds = {'seed_mean': mem['means'], 'seed_chol': torch.linalg.cholesky(mem['covars']),
                     'seed_teacher': mem['features']}
            written = writer(seeds)
            prediction = reader.read(teacher, written, cameras, K, unit)
            ext_error = prediction - target.log()
            per_query = []
            for qi, q in enumerate(QUERIES):
                mask = fixed[qi]
                residual = ext_error[qi, valid[qi]]
                per_query.append({'query': q,
                    'fixed_support_patches': int(mask.sum()),
                    'geometry_log_mse_on_fixed_support': mean_or_none(log_error[qi, mask].square()),
                    'retained_fixed_support_fraction': mean_or_none(coverage[qi, mask].float()),
                    'valid_patch_coverage': mean_or_none(coverage[qi, valid[qi]].float()),
                    'new_coverage_fraction': mean_or_none(coverage[qi, valid[qi] & ~fixed[qi]].float()),
                    'external_probe_log_mse': mean_or_none(residual.square()),
                    'external_probe_centered_mse': mean_or_none((residual - residual.mean()).square()),
                    'external_probe_bias_squared': float(residual.mean().square())})
            outputs[name].append({'write_index': t, 'latest_write': None if t == 0 else WRITES[t - 1],
                                  'queries': per_query,
                                  'min_shape_eigenvalue': float(eig.min()),
                                  'total_effective_support': float(mem['support'].sum()),
                                  'represented_initial_lineage': int(mem['lineage'].any(0).sum())})
            projections[name].append({'log_depth': depth.log(), 'coverage': coverage})
    for name, track in tracks.items():
        if track['amplitude'] == 0: continue
        paired = track_name(track['strategy'], 0)
        for t in range(5):
            delta = projections[name][t]['log_depth'] - projections[paired][t]['log_depth']
            for qi in range(4):
                outputs[name][t]['queries'][qi]['paired_clean_geometry_gap'] = mean_or_none(delta[qi, fixed[qi]].square())
    return outputs


def run(subset='all'):
    config = protocol(); plan = load_plan(); rows = select_rows(plan, subset)
    old = json.loads((ROOT / 'results/gaussian_memory/content_center1/plan.json').read_text())
    old_meta = {m['id']: m for m in old['catalog']}
    checkpoint = torch.load(config['external_probe'], map_location='cpu', weights_only=False)
    reader = GeometryTransport('latent', 'gaussian').eval().requires_grad_(False)
    writer = CompactGaussianWriter(False).eval().requires_grad_(False)
    reader.load_state_dict(checkpoint['model']); writer.load_state_dict(checkpoint['writer'])
    source_hash = sha256(STATE_OUT / 'protocol.json')
    completed = []; start = time.monotonic()
    for meta in rows:
        rid = meta['id']; dest = STATE_OUT / rid; dest.mkdir(parents=True, exist_ok=True)
        if (dest / 'complete.json').exists():
            status = json.loads((dest / 'complete.json').read_text())
            assert status['protocol_sha256'] == source_hash
            completed.append(rid); continue
        initial = torch.load(CACHE / rid / 'initial.pt', map_location='cpu', weights_only=False)
        updates = torch.load(CACHE / rid / 'updates.pt', map_location='cpu', weights_only=False)
        tracks = {}
        for strategy in ('static', 'average', 'corrective', 'rebuild'):
            tracks[track_name(strategy, 0)] = trajectory(initial, updates, strategy)
        for amplitude in AMPLITUDES:
            for strategy in ('static', 'average', 'corrective'):
                tracks[track_name(strategy, amplitude)] = trajectory(initial, updates, strategy, amplitude)
        save_tensor(dest / 'trajectories.pt', tracks)
        # Auxiliary S1 probe uses only each complete query image, never future frames.
        rec = torch.load(old_meta[rid]['tokens'], map_location='cpu', weights_only=False)
        assert rec['independent_single_view_axis']
        teacher = rec['teacher_tokens'][list(QUERIES)].float()
        scores = probes(tracks, initial, meta['row'], teacher, reader, writer)
        write_json(dest / 'probes.json', scores)
        write_json(dest / 'updates.json', {name: t['updates'] for name, t in tracks.items()})
        write_json(dest / 'complete.json', {'complete': True, 'protocol_sha256': source_hash,
                   'trajectories': len(tracks), 'snapshots': len(tracks) * 5,
                   'sha256': {n: sha256(dest / n) for n in ('trajectories.pt', 'probes.json', 'updates.json')},
                   'all_queries_readonly': True, 'labels_used_for_updates': False})
        completed.append(rid)
        write_json(STATE_OUT / 'progress.json', {'completed': completed, 'subset': subset, 'seconds': time.monotonic() - start})
        print(json.dumps({'state_clip': rid, 'completed': len(completed), 'planned': len(rows),
                          'seconds': round(time.monotonic() - start, 2)}), flush=True)
    write_json(STATE_OUT / f'{subset}_complete.json', {'complete': True, 'clips': completed,
               'seconds': time.monotonic() - start, 'CPU_threads': torch.get_num_threads()})


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--subset', choices=('smoke', 'train', 'all'), default='all')
    args = p.parse_args(); torch.set_num_threads(4)
    run(args.subset)
