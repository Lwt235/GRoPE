"""The frozen Corrective update, with an explicit writable source timeline.

The numerical body is copied verbatim from gaussian_memory_incremental.py;
only the source-frame guard is generalized. The old implementation is retained
and compared on its original four writes before creating longer histories.
"""
import torch
from models.gaussian_memory_incremental import (
    clone_memory, association, fuse_assigned, candidate_from_recent,
    mergeable_pair, merge_pair)


@torch.no_grad()
def update_memory(previous, observation, recent, strategy='corrective', *, allowed_frames):
    """A single independent write, with bounded recent records supplied by caller.

    'average' shares matching/novel-region capacity management but disables the
    finite-history weight cap and the explicit contradiction reset.
    """
    memory = clone_memory(previous)
    frame = int(observation['source_frame'][0])
    assert observation['source_frame'].unique().numel() == 1
    assert frame in allowed_frames, 'Read-only or undeclared source frame.'
    assert 0 <= frame < previous['sources'].shape[1]
    assert previous['conflicts'].shape == previous['sources'].shape
    assert all(int(x['source_frame'][0]) <= frame for x in recent)
    assert all(int(x['source_frame'][0]) in allowed_frames for x in recent)
    assert len(recent) <= 4 and len({int(x['source_frame'][0]) for x in recent}) == len(recent)
    cap = 4. if strategy == 'corrective' else float('inf')
    assignment, matched, contradiction, patch_distance = association(memory, observation)
    hit = fuse_assigned(memory, observation, assignment, matched, cap)
    contradiction &= ~hit
    memory['conflicts'][contradiction, frame] = True
    # Contradictions are counted once per source, not per overlapping window.
    if strategy == 'corrective':
        memory['support'][contradiction] = (memory['support'][contradiction] * .5).clamp_min(1.)
    bad = memory['conflicts'].sum(-1) >= 2
    pending = (~matched & observation['valid']).nonzero().flatten()
    pending = pending[observation['confidence'][pending].argsort(descending=True)]
    events = []; accepted = []; dropped = 0; unsupported = 0
    for idx in pending.tolist():
        proposal = candidate_from_recent(idx, observation, recent)
        if proposal is None:
            unsupported += 1; continue
        if accepted:
            delta = torch.stack(accepted) - proposal['mean']
            if (delta.square().sum(-1) < proposal['cov'].trace().clamp_min(1e-6)).any():
                continue
        slot = None; event = None
        if strategy == 'corrective':
            eligible = bad & (patch_distance[idx] <= 9)
            if eligible.any():
                slot = int(patch_distance[idx].masked_fill(~eligible, float('inf')).argmin())
                event = {'action': 'local_rebuild', 'slot': slot,
                         'old_lineage': memory['lineage'][slot].nonzero().flatten().tolist()}
        if slot is None:
            pair = mergeable_pair(memory, observation['camera'])
            if pair is not None:
                keep, slot = pair
                merge_pair(memory, keep, slot, cap)
                event = {'action': 'merge_and_insert', 'merged_slots': [keep, slot], 'slot': slot}
        if slot is None:
            dropped += 1; continue
        old_lineage = memory['lineage'][slot].clone() if event['action'] == 'local_rebuild' else torch.zeros_like(memory['lineage'][slot])
        memory['means'][slot] = proposal['mean']; memory['covars'][slot] = proposal['cov']
        memory['features'][slot] = proposal['feature']
        memory['sources'][slot] = False; memory['sources'][slot, proposal['sources']] = True
        memory['support'][slot] = min(len(proposal['sources']), cap)
        memory['conflicts'][slot] = False; memory['lineage'][slot] = old_lineage
        bad[slot] = False; accepted.append(proposal['mean'])
        event.update(observation_patch=idx, source_frames=proposal['sources'])
        events.append(event)
    assert torch.isfinite(memory['means']).all() and torch.isfinite(memory['covars']).all()
    diagnostics = {'frame': frame, 'strategy': strategy, 'new_valid_patches': int(observation['valid'].sum()),
                   'matched_patches': int(matched.sum()), 'fused_slots': int(hit.sum()),
                   'free_space_conflict_slots': int(contradiction.sum()), 'pending_unmatched': len(pending),
                   'unsupported_candidates': unsupported, 'capacity_dropped': dropped,
                   'local_rebuilds': sum(e['action'] == 'local_rebuild' for e in events),
                   'new_region_insertions': sum(e['action'] == 'merge_and_insert' for e in events),
                   'recent_source_frames': [int(x['source_frame'][0]) for x in recent],
                   'events': events, 'slot_count': len(memory['means'])}
    return memory, diagnostics

