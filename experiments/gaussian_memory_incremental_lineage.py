"""Describe repairs/merges affecting the predetermined perturbed slots."""
import json

import numpy as np
import torch

from experiments.gaussian_memory_setup import sha256
from experiments.gaussian_memory_incremental_data import OUT, load_plan, write_json
from experiments.gaussian_memory_incremental_states import STATE_OUT
from experiments.gaussian_memory_incremental_analysis import cohorts


def main():
    torch.set_num_threads(4); plan = load_plan(); rows = {}
    for meta in plan['catalog']:
        tracks = torch.load(STATE_OUT / meta['id'] / 'trajectories.pt', map_location='cpu', weights_only=False)
        outputs = {}
        for name, track in tracks.items():
            if not track['amplitude']: continue
            selected = torch.zeros(256, dtype=torch.bool); selected[track['perturbed_indices']] = True
            repaired = torch.zeros_like(selected); merged = torch.zeros_like(selected)
            per_write = []
            for ti, log in enumerate(track['updates'], 1):
                lineage = track['states'][ti - 1]['lineage'].clone()
                for event in log.get('events', []):
                    if event['action'] == 'local_rebuild':
                        ancestry = lineage[event['slot']]
                        repaired |= ancestry & selected
                        assert set(ancestry.nonzero().flatten().tolist()) == set(event['old_lineage'])
                    else:
                        keep, slot = event['merged_slots']
                        merged |= (lineage[keep] | lineage[slot]) & selected
                        lineage[keep] |= lineage[slot]; lineage[slot] = False
                assert torch.equal(lineage, track['states'][ti]['lineage'])
                sources = track['states'][ti]['sources'].nonzero()[:, 1].unique().tolist()
                assert not any(f in (5, 7, 9, 11) for f in sources)
                per_write.append({'write_index': ti,
                    'initial_selected_ids_with_local_rebuild': int(repaired.sum()),
                    'initial_selected_ids_merged': int(merged.sum()),
                    'initial_selected_lineage_still_represented': int((lineage.any(0) & selected).sum()),
                    'source_frames_in_state': sources})
            outputs[name] = per_write
        rows[meta['id']] = outputs
    primary = cohorts(plan)['primary51']; policies = list(next(iter(rows.values())))
    summaries = {}
    for policy in policies:
        summaries[policy] = {k: float(np.mean([rows[m['id']][policy][-1][k] for m in primary])) for k in (
            'initial_selected_ids_with_local_rebuild', 'initial_selected_ids_merged', 'initial_selected_lineage_still_represented')}
    write_json(OUT / 'perturbed_lineage_diagnostics.json', {'complete': True, 'clips': rows,
        'primary_clip_means_after_final_write': summaries, 'source_sha256': sha256(__file__),
        'interpretation': 'Lineage denotes ancestry, not unchanged geometry or successful error correction. Local rebuild intentionally retains ancestry; use fixed-region coverage and sensor errors to assess forgetting/repair.',
        'source_frames_reconstructed_without_readonly_queries': True})
    print(json.dumps({'lineage_diagnostics_complete': True, 'primary_clip_means': summaries}), flush=True)


if __name__ == '__main__': main()
