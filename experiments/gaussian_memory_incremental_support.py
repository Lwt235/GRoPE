"""Separate absent geometric support from depth error, retaining original probes."""
import json
import time

import numpy as np
import torch

from experiments.gaussian_memory_setup import sha256
from experiments.gaussian_memory_incremental_data import OUT, CACHE, QUERIES, load_plan, write_json
from experiments.gaussian_memory_incremental_states import STATE_OUT, AMPLITUDES, track_name, coarse_targets
from models.gaussian_memory_vggt_adapter import project_memory


def mean(x):
    return float(x.mean()) if x.numel() else None


@torch.no_grad()
def main():
    torch.set_num_threads(4); plan = load_plan(); start = time.monotonic()
    record = {'reason': 'Projected mean depth is undefined without alpha support; clamping zero to 1e-6 made the original geometric probe an arbitrary missing-support penalty.',
              'original_probes_retained': True, 'state_or_model_changes': False,
              'original_metric_not_used_as_geometric_accuracy': 'geometry_log_mse_on_fixed_support',
              'coverage_threshold_unchanged': .05,
              'new_clean_mask': 'initial clean support and sensor valid and support of all four clean policies at the same time/view',
              'new_error_mask': 'initial clean support and sensor valid and support of three clean plus three same-amplitude perturbed policies at the same time/view',
              'coverage_reported_separately': True, 'scope': 'development diagnostic correction, not fresh confirmation',
              'source_sha256': sha256(__file__)}
    write_json(STATE_OUT / 'support_metric_revision.json', record)
    for meta in plan['catalog']:
        rid = meta['id']; folder = STATE_OUT / rid
        initial = torch.load(CACHE / rid / 'initial.pt', map_location='cpu', weights_only=False)
        tracks = torch.load(folder / 'trajectories.pt', map_location='cpu', weights_only=False)
        target, valid = coarse_targets(meta['row'])
        poses, K = initial['c2w'][list(QUERIES)], initial['K'][list(QUERIES)]
        base = project_memory(initial['memory'], poses, K)['coverage'] > .05
        fixed = base & valid
        projections = {name: [project_memory(mem, poses, K) for mem in t['states']] for name, t in tracks.items()}
        scores = {name: [{'write_index': t, 'queries': [{} for _ in QUERIES]} for t in range(5)] for name in tracks}
        groups = [(['static_clean', 'average_clean', 'corrective_clean', 'rebuild_clean'], None)]
        groups += [([track_name(st, a) for st in ('static', 'average', 'corrective')] +
                    [track_name(st, 0) for st in ('static', 'average', 'corrective')], a) for a in AMPLITUDES]
        for names, amplitude in groups:
            for ti in range(5):
                common = fixed.clone()
                for name in names: common &= projections[name][ti]['coverage'] > .05
                for name in names:
                    # Preserve clean main-mask scores; error groups only add their perturbed entries.
                    if amplitude is not None and name.endswith('_clean'): continue
                    p = projections[name][ti]
                    log_depth = (p['depth'].clamp_min(1e-6) * initial['unit_meters']).log()
                    delta = log_depth - target.log()
                    paired = track_name(tracks[name]['strategy'], 0)
                    paired_log = (projections[paired][ti]['depth'].clamp_min(1e-6) * initial['unit_meters']).log()
                    for qi, q in enumerate(QUERIES):
                        keep = common[qi]
                        scores[name][ti]['queries'][qi] = {'query': q,
                            'geometry_log_mse_common_support': mean(delta[qi, keep].square()),
                            'common_support_patches': int(keep.sum()),
                            'common_support_fraction_of_initial': float(keep.sum() / fixed[qi].sum().clamp_min(1)),
                            'paired_clean_geometry_gap_common_support': mean((log_depth[qi, keep] - paired_log[qi, keep]).square()) if amplitude is not None else None}
        write_json(folder / 'support_probes.json', scores)
    write_json(STATE_OUT / 'support_complete.json', {'complete': True, 'clips': len(plan['catalog']),
               'seconds': time.monotonic() - start, 'source_sha256': sha256(__file__), 'original_scores_preserved': True})
    print(json.dumps({'support_diagnostic_complete': True, 'clips': len(plan['catalog']), 'seconds': time.monotonic() - start}), flush=True)


if __name__ == '__main__': main()
