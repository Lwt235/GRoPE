"""Shared rigid estimator for a fixed regional-PE utility diagnostic.

The estimator accepts local centers and predicted indices only. World geometry
and recovered ground truth live in the separate scoring functions below.
"""
from itertools import combinations

import numpy as np
import torch


def rigid_torch(source, target):
    """Proper rigid least squares: target = source @ rotation.T + translation."""
    a, b = source.mean(-2), target.mean(-2)
    matrix = (source - a.unsqueeze(-2)).transpose(-1, -2) @ (target - b.unsqueeze(-2))
    u, singular, vh = torch.linalg.svd(matrix)
    diagonal = torch.ones_like(singular)
    diagonal[..., -1] = torch.where(torch.linalg.det(vh.transpose(-1, -2) @ u.transpose(-1, -2)) < 0, -1., 1.)
    rotation = (vh.transpose(-1, -2) * diagonal.unsqueeze(-2)) @ u.transpose(-1, -2)
    translation = b - (rotation @ a.unsqueeze(-1)).squeeze(-1)
    valid = singular[..., 1] > 1e-10 * singular[..., 0].clamp_min(1e-30)
    return rotation, translation, valid


def estimate(source, target, predicted_indices, *, radius=.1, chunk_size=2048, max_refits=5):
    """Enumerate three-point hypotheses; there is no label or confidence input."""
    assert source.ndim == target.ndim == 2 and source.shape[1] == target.shape[1] == 3
    assert source.dtype == target.dtype == torch.float64
    assert predicted_indices.shape == (len(source),) and len(source) >= 3
    assert torch.isfinite(source).all() and torch.isfinite(target).all() and radius > 0
    assert predicted_indices.dtype == torch.long
    assert int(predicted_indices.min()) >= 0 and int(predicted_indices.max()) < len(target)
    triples = torch.tensor(list(combinations(range(len(source)), 3)), device=source.device)
    total = len(triples)
    target_indices = predicted_indices[triples]
    distinct = ((target_indices[:, 0] != target_indices[:, 1]) &
                (target_indices[:, 0] != target_indices[:, 2]) &
                (target_indices[:, 1] != target_indices[:, 2]))
    triples = triples[distinct]
    matched = target[predicted_indices]
    best_rotation = torch.eye(3, dtype=source.dtype, device=source.device)
    best_translation = torch.zeros(3, dtype=source.dtype, device=source.device)
    best_count, best_cost, nondegenerate = -1, float('inf'), 0
    threshold2 = radius * radius

    def residuals(rotation, translation):
        transformed = torch.einsum('...ij,nj->...ni', rotation, source) + translation.unsqueeze(-2)
        squared = (transformed - matched).square().sum(-1)
        return squared <= threshold2, squared.clamp(max=threshold2).sum(-1)

    for start in range(0, len(triples), chunk_size):
        index = triples[start:start + chunk_size]
        rotation, translation, valid = rigid_torch(source[index], matched[index])
        nondegenerate += int(valid.sum())
        if not bool(valid.any()):
            continue
        mask, cost = residuals(rotation, translation)
        counts = mask.sum(-1).masked_fill(~valid, -1)
        maximum = int(counts.max())
        chosen = int(cost.masked_fill(counts != maximum, float('inf')).argmin())
        candidate_cost = float(cost[chosen])
        if maximum > best_count or (maximum == best_count and candidate_cost < best_cost):
            best_count, best_cost = maximum, candidate_cost
            best_rotation, best_translation = rotation[chosen].clone(), translation[chosen].clone()
    valid_estimate = best_count >= 3
    refinements = 0
    if valid_estimate:
        mask, _ = residuals(best_rotation, best_translation)
        for _ in range(max_refits):
            rotation, translation, valid = rigid_torch(source[mask], matched[mask])
            if not bool(valid):
                break
            candidate_mask, cost = residuals(rotation, translation)
            count, cost = int(candidate_mask.sum()), float(cost)
            if count < best_count or (count == best_count and cost > best_cost):
                break
            stable = torch.equal(mask, candidate_mask)
            best_rotation, best_translation = rotation, translation
            best_count, best_cost, mask = count, cost, candidate_mask
            refinements += 1
            if stable:
                break
    return dict(rotation=best_rotation.cpu().numpy(), translation=best_translation.cpu().numpy(),
                valid=valid_estimate, consensus_count=max(best_count, 0),
                truncated_squared_error=best_cost if valid_estimate else None,
                hypotheses_total=total, distinct_target_hypotheses=len(triples),
                nondegenerate_hypotheses=nondegenerate, refits=refinements)


def rigid_numpy(source, target):
    """Scoring-only reconstruction of each view's known coordinate transform."""
    a, b = np.asarray(source, np.float64), np.asarray(target, np.float64)
    ca, cb = a.mean(0), b.mean(0)
    u, singular, vh = np.linalg.svd((a - ca).T @ (b - cb))
    diagonal = np.eye(3)
    diagonal[-1, -1] = 1 if np.linalg.det(vh.T @ u.T) >= 0 else -1
    rotation = vh.T @ diagonal @ u.T
    translation = cb - rotation @ ca
    valid = singular[1] > 1e-10 * max(singular[0], 1e-30)
    return rotation, translation, bool(valid)


def recover_truth(local, world):
    """Ground truth for scoring only; inputs come from matching within each view."""
    mappings = [rigid_numpy(local[v], world[v]) for v in range(2)]
    assert all(v[2] for v in mappings), 'Degenerate coordinate reconstruction.'
    (r0, t0, _), (r1, t1, _) = mappings
    error = max(np.max(np.abs(local[v] @ mappings[v][0].T + mappings[v][1] - world[v])) for v in range(2))
    return r1.T @ r0, r1.T @ (t0 - t1), float(error)


def score_transform(source, rotation, translation, truth_rotation, truth_translation, scale, *, valid=True):
    assert np.isfinite(scale) and scale > 0
    predicted = source @ rotation.T + translation
    truth = source @ truth_rotation.T + truth_translation
    rmse = float(np.sqrt(np.square(predicted - truth).sum(1).mean()) * scale)
    cosine = np.clip((np.trace(rotation @ truth_rotation.T) - 1) / 2, -1., 1.)
    rotation_error = float(np.degrees(np.arccos(cosine)))
    translation_error = float(np.linalg.norm(translation - truth_translation) * scale)
    if not valid:
        rmse = rotation_error = translation_error = float('inf')
    return dict(rmse_m=rmse, rotation_error_deg=rotation_error,
                centered_translation_error_m=translation_error, success=bool(valid and rmse < .2))
