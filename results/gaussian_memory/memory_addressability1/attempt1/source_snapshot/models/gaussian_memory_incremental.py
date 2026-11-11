"""Bounded Gaussian-state fusion driven only by independent direct observations."""
import torch

from models.gaussian_memory_vggt_adapter import project_memory


def clone_memory(memory):
    return {k: v.clone() for k, v in memory.items()}


def observation_frames(obs):
    frames = []
    names = ('means', 'covars', 'features', 'confidence', 'valid', 'source_frame', 'depth_patch')
    for i, frame in enumerate(obs['source_frame'].unique(sorted=True)):
        keep = obs['source_frame'] == frame
        row = {k: obs[k][keep] for k in names}
        row.update(camera=obs['camera'][i:i + 1], K=obs['K'][i:i + 1])
        frames.append(row)
    return frames


def concatenate_observations(records):
    keys = ('means', 'covars', 'features', 'confidence', 'valid', 'source_frame', 'depth_patch')
    return {k: torch.cat([r[k] for r in records], 0) for k in keys}


def blend(mean1, cov1, feat1, weight1, mean2, cov2, feat2, weight2):
    """Moment-preserving fusion; shape includes within- and between-region extent."""
    ratio = weight2 / (weight1 + weight2).clamp_min(1e-8)
    mean = mean1 * (1 - ratio[..., None]) + mean2 * ratio[..., None]
    delta1, delta2 = mean1 - mean, mean2 - mean
    cov = (cov1 + delta1[..., :, None] * delta1[..., None, :]) * (1 - ratio[..., None, None])
    cov += (cov2 + delta2[..., :, None] * delta2[..., None, :]) * ratio[..., None, None]
    feature = feat1 * (1 - ratio[..., None]) + feat2 * ratio[..., None]
    return mean, (cov + cov.transpose(-1, -2)) * .5, feature


def association(memory, observation):
    projection = project_memory(memory, observation['camera'], observation['K'])
    y, x = torch.meshgrid(torch.arange(15, device=memory['means'].device) + .5,
                         torch.arange(20, device=memory['means'].device) + .5, indexing='ij')
    uv = torch.stack((x, y), -1).flatten(0, 1)
    delta = uv[:, None] - projection['projected'][0, None]
    d2 = torch.einsum('pni,nij,pnj->pn', delta, torch.linalg.inv(projection['footprint'][0]), delta)
    old_z = projection['center'][0, :, 2]
    new_z = observation['depth_patch']
    dz = (new_z[:, None] - old_z[None]).abs() / new_z[:, None].clamp_min(1e-4)
    admissible = (d2 <= 9) & (dz <= .1) & projection['valid'][0, None] & observation['valid'][:, None]
    cost = (d2 + (dz / .1).square()).masked_fill(~admissible, float('inf'))
    best, assignment = cost.min(-1)
    matched = torch.isfinite(best)
    # Only confidently observed free space can contradict old foreground.
    pixel = projection['projected'][0].floor().long()
    inside = (pixel[:, 0] >= 0) & (pixel[:, 0] < 20) & (pixel[:, 1] >= 0) & (pixel[:, 1] < 15)
    flat = pixel[:, 1].clamp(0, 14) * 20 + pixel[:, 0].clamp(0, 19)
    contradiction = inside & projection['valid'][0] & observation['valid'][flat]
    contradiction &= old_z < observation['depth_patch'][flat] * .9
    return assignment, matched, contradiction, d2


def fuse_assigned(memory, observation, assignment, matched, cap):
    indices = matched.nonzero().flatten()
    if len(indices) == 0:
        return torch.zeros(256, dtype=torch.bool, device=assignment.device)
    slots = assignment[indices]
    confidence = observation['confidence'][indices]
    # Within-frame weights do not create extra independent view support.
    confidence = (confidence / confidence.median().clamp_min(1e-5)).clamp(.1, 10)
    total = confidence.new_zeros(256).index_add_(0, slots, confidence)
    means = memory['means'].new_zeros(256, 3).index_add_(0, slots, observation['means'][indices] * confidence[:, None])
    means = means / total[:, None].clamp_min(1e-8)
    delta = observation['means'][indices] - means[slots]
    second = observation['covars'][indices] + delta[..., :, None] * delta[..., None, :]
    cov = memory['covars'].new_zeros(256, 3, 3).index_add_(0, slots, second * confidence[:, None, None])
    cov = cov / total[:, None, None].clamp_min(1e-8)
    features = memory['features'].new_zeros(256, 32).index_add_(0, slots, observation['features'][indices] * confidence[:, None])
    features /= total[:, None].clamp_min(1e-8)
    hit = total > 0
    frame = int(observation['source_frame'][0])
    assert not memory['sources'][hit, frame].any(), 'A source frame must never be fused twice.'
    old = memory['support'][hit].clamp(max=cap)
    mu, sigma, feat = blend(memory['means'][hit], memory['covars'][hit], memory['features'][hit], old,
                           means[hit], cov[hit], features[hit], torch.ones_like(old))
    memory['means'][hit] = mu; memory['covars'][hit] = sigma; memory['features'][hit] = feat
    memory['sources'][hit, frame] = True
    memory['support'][hit] = (old + 1).clamp(max=cap)
    memory['conflicts'][hit] = False
    return hit


def candidate_from_recent(index, current, recent):
    anchor = current['means'][index]
    means, covars, features, sources = [], [], [], []
    for obs in recent:
        distance = (obs['means'] - anchor).square().sum(-1).masked_fill(~obs['valid'], float('inf'))
        j = int(distance.argmin())
        radius2 = 9 * (current['covars'][index].trace() + obs['covars'][j].trace())
        if distance[j] > radius2:
            continue
        means.append(obs['means'][j]); covars.append(obs['covars'][j]); features.append(obs['features'][j])
        sources.append(int(obs['source_frame'][j]))
    if len(set(sources)) < 2:
        return None
    pts = torch.stack(means); mean = pts.mean(0); delta = pts - mean
    cov = (torch.stack(covars) + delta[..., :, None] * delta[..., None, :]).mean(0)
    return {'mean': mean, 'cov': cov, 'feature': torch.stack(features).mean(0), 'sources': sorted(set(sources))}


def mergeable_pair(memory, camera):
    mean, cov = memory['means'], memory['covars']
    distance = torch.cdist(mean, mean).square()
    extent = cov.diagonal(dim1=-2, dim2=-1).sum(-1)
    ratio = distance / (extent[:, None] + extent[None]).clamp_min(1e-8)
    R = camera[0, :3, :3].T
    z = ((mean - camera[0, :3, 3]) @ R.T)[:, 2]
    depth_ok = (z[:, None] - z[None]).abs() <= .1 * torch.minimum(z[:, None], z[None]).clamp_min(0)
    upper = torch.triu(torch.ones_like(ratio, dtype=torch.bool), diagonal=1)
    ratio = ratio.masked_fill(~upper | ~depth_ok, float('inf'))
    best = int(ratio.argmin()); i, j = divmod(best, len(mean))
    return (i, j) if ratio[i, j] <= .25 else None


def merge_pair(memory, i, j, cap):
    w1, w2 = memory['support'][i].clamp(max=cap), memory['support'][j].clamp(max=cap)
    mu, cov, feat = blend(memory['means'][i], memory['covars'][i], memory['features'][i], w1,
                         memory['means'][j], memory['covars'][j], memory['features'][j], w2)
    memory['means'][i] = mu; memory['covars'][i] = cov; memory['features'][i] = feat
    memory['sources'][i] |= memory['sources'][j]
    memory['support'][i] = memory['sources'][i].sum().float().clamp(max=cap)
    memory['lineage'][i] |= memory['lineage'][j]
    memory['conflicts'][i] &= memory['conflicts'][j]


@torch.no_grad()
def update_memory(previous, observation, recent, strategy='corrective'):
    """A single independent write, with bounded recent records supplied by caller.

    'average' shares matching/novel-region capacity management but disables the
    finite-history weight cap and the explicit contradiction reset.
    """
    memory = clone_memory(previous)
    frame = int(observation['source_frame'][0])
    assert frame in (4, 6, 8, 10)
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


def perturb_memory(memory, camera, amplitude):
    result = clone_memory(memory)
    # Spatial selection fixed by initial geometry, with no labels or future views.
    centroid = memory['means'].mean(0)
    anchor = (memory['means'] - centroid).square().sum(-1).argmin()
    selected = (memory['means'] - memory['means'][anchor]).square().sum(-1).argsort()[:52]
    direction = camera[0, :3, 2]
    result['means'][selected] += amplitude * direction
    return result, selected
