"""Revisit/support and resource diagnostics; never modify a saved state."""
import argparse
import json
import time

import numpy as np
import torch

from experiments.gaussian_memory_setup import sha256
from experiments.gaussian_memory_incremental_data import (
    OUT, CACHE, WRITES, QUERIES, load_plan, write_json, save_tensor, build_memory)
from experiments.gaussian_memory_incremental_states import STATE_OUT, coarse_targets, trajectory
from experiments.gaussian_memory_incremental_analysis import cohorts
from models.gaussian_memory_incremental import observation_frames, concatenate_observations
from models.gaussian_memory_vggt_adapter import project_memory


def nbytes(record):
    if isinstance(record, torch.Tensor): return record.numel() * record.element_size()
    if isinstance(record, dict): return sum(nbytes(v) for v in record.values())
    if isinstance(record, (list, tuple)): return sum(nbytes(v) for v in record)
    return 0


@torch.no_grad()
def revisit():
    plan = load_plan(); output = {}; start = time.monotonic()
    protocol = {'selection': 'the same 52 initial slots used by the signed perturbation',
                'query_region': 'clean selected-slot projected footprint > .05; fixed before comparing trajectories',
                'revisit': 'clean initial slot center in write-view grid, positive depth, valid independent observation, and not more than 10% behind observed foreground',
                'revisit_is_not_ground_truth_visibility': True,
                'sensor_depth_used_only_for_accuracy': True,
                'strata': 'at least one selected slot potentially revisited versus none; also report continuous selected-slot fraction',
                'no_hyperparameter_or_state_change': True, 'source_sha256': sha256(__file__)}
    write_json(OUT / 'revisit_protocol.json', protocol)
    for meta in plan['catalog']:
        rid = meta['id']; folder = STATE_OUT / rid
        initial = torch.load(CACHE / rid / 'initial.pt', map_location='cpu', weights_only=False)
        updates = torch.load(CACHE / rid / 'updates.pt', map_location='cpu', weights_only=False)
        tracks = torch.load(folder / 'trajectories.pt', map_location='cpu', weights_only=False)
        selected = tracks['static_minus20']['perturbed_indices']
        memory = initial['memory']; subset = {k: v[selected] for k, v in memory.items()}
        region = project_memory(subset, initial['c2w'][list(QUERIES)], initial['K'][list(QUERIES)])['coverage'] > .05
        seen = torch.zeros(52, dtype=torch.bool); per_write = []
        for frame in WRITES:
            obs = updates[frame]
            if obs is not None:
                p = project_memory(subset, obs['camera'], obs['K'])
                uv = p['projected'][0].floor().long()
                inside = (uv[:, 0] >= 0) & (uv[:, 0] < 20) & (uv[:, 1] >= 0) & (uv[:, 1] < 15)
                flat = uv[:, 1].clamp(0, 14) * 20 + uv[:, 0].clamp(0, 19)
                visible = inside & p['valid'][0] & obs['valid'][flat]
                visible &= p['center'][0, :, 2] <= 1.1 * obs['depth_patch'][flat]
                seen |= visible
            per_write.append({'frame': frame, 'revisited_slots': int(seen.sum()), 'revisited_fraction': float(seen.float().mean())})
        target, valid = coarse_targets(meta['row'])
        support = json.loads((folder / 'support_probes.json').read_text())
        masks = region & valid
        # Coverage is evaluated on the same fixed selected region, including absent support.
        coverage = {}
        for name, track in tracks.items():
            coverage[name] = []
            for ti, mem in enumerate(track['states']):
                p = project_memory(mem, initial['c2w'][list(QUERIES)], initial['K'][list(QUERIES)])
                coverage[name].append([float((p['coverage'][qi, masks[qi]] > .05).float().mean()) if masks[qi].any() else None for qi in range(4)])
        save_tensor(folder / 'selected_region.pt', {'query_patch_masks': region, 'revisit': per_write})
        output[rid] = {'place': meta['place'], 'writes': per_write, 'selected_region_patches': masks.sum(-1).tolist(),
                       'selected_region_coverage': coverage}
    primary = cohorts(plan)['primary51']; summary = []
    for ti in range(4):
        rows = [output[m['id']]['writes'][ti] for m in primary]
        summary.append({'write_index': ti + 1, 'clips': len(rows),
                        'clips_any_revisit': sum(r['revisited_slots'] > 0 for r in rows),
                        'revisited_fraction_quantiles': np.quantile([r['revisited_fraction'] for r in rows], [0, .25, .5, .75, 1]).tolist()})
    write_json(OUT / 'revisit_diagnostics.json', {'complete': True, 'protocol': protocol, 'clips': output,
               'primary_summary': summary, 'seconds': time.monotonic() - start})
    print(json.dumps({'revisit_complete': True, 'primary_summary': summary}), flush=True)


@torch.no_grad()
def costs():
    plan = load_plan(); rows = []
    for rid in ('transfer_00', 'confirm_00', 'fresh_00'):
        initial = torch.load(CACHE / rid / 'initial.pt', map_location='cpu', weights_only=False)
        updates = torch.load(CACHE / rid / 'updates.pt', map_location='cpu', weights_only=False)
        row = {'id': rid, 'state_bytes_with_lineage': nbytes(initial['memory']),
               'state_bytes_without_diagnostic_lineage': nbytes({k: v for k, v in initial['memory'].items() if k != 'lineage'}),
               'recent_buffer_bytes_at_four_writes': nbytes(list(updates.values())),
               'prefix_raw_observation_bytes_at_four_writes': nbytes(initial['observation']) + nbytes(updates)}
        timings = {}
        for strategy in ('average', 'corrective', 'rebuild'):
            start = time.perf_counter(); trajectory(initial, updates, strategy)
            timings[strategy + '_four_writes_s'] = time.perf_counter() - start
        p = project_memory(initial['memory'], initial['c2w'][list(QUERIES)], initial['K'][list(QUERIES)])
        start = time.perf_counter()
        for _ in range(20): project_memory(initial['memory'], initial['c2w'][list(QUERIES)], initial['K'][list(QUERIES)])
        timings['four_view_projection_s'] = (time.perf_counter() - start) / 20
        row.update(timings); rows.append(row)
    write_json(OUT / 'state_costs.json', {'complete': True, 'CPU_threads': torch.get_num_threads(),
               'exclusive_benchmark': False, 'scope': 'descriptive CPU timing, excludes independent VGGT observations',
               'buffer_is_auxiliary_online_memory': True, 'rows': rows})
    print(json.dumps({'state_costs_complete': True, 'rows': rows}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=('revisit', 'costs'))
    args = p.parse_args(); torch.set_num_threads(4); globals()[args.stage]()
