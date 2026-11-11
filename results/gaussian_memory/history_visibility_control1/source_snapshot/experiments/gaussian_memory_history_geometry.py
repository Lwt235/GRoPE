"""Common-gauge inputs for two previously cached observation windows.

This module does not select a research protocol, fit a model, or read labels.
The moment conversion reproduces the existing observation construction in a
shared scene unit, including its normalized covariance floor.
"""
import numpy as np
import torch
import torch.nn.functional as F

from experiments.gaussian_memory_data import merge_seeds


def camera_to_common(camera, common_from_clip_metric, source_unit, common_unit):
    """Transform normalized camera-to-world poses into one common gauge."""
    if min(source_unit, common_unit) <= 0:
        raise ValueError('Scene units must be positive.')
    transform = torch.as_tensor(common_from_clip_metric, dtype=camera.dtype, device=camera.device)
    rotation, translation = transform[:3, :3], transform[:3, 3]
    result = camera.clone()
    result[..., :3, :3] = rotation @ camera[..., :3, :3]
    result[..., :3, 3] = (
        (camera[..., :3, 3] * source_unit) @ rotation.T + translation) / common_unit
    return result


def observation_to_common(observation, common_from_clip_metric, source_unit,
                          common_unit, source_offset, normalized_floor=.005):
    """Convert observation moments, with a fresh floor in the common unit.

    This intentionally differs from rigidly transporting an already fused
    physical memory: the old observation-construction floor is removed before
    scaling and added back in the selected common normalized unit. Features,
    confidence, validity and camera intrinsics retain their original identity.
    """
    if min(source_unit, common_unit) <= 0 or source_offset < 0:
        raise ValueError('Invalid units or source offset.')
    result = {key: value.clone() for key, value in observation.items()}
    mean = observation['means']
    transform = torch.as_tensor(common_from_clip_metric, dtype=mean.dtype, device=mean.device)
    rotation, translation = transform[:3, :3], transform[:3, 3]
    ratio = source_unit / common_unit
    result['means'] = ratio * (mean @ rotation.T) + translation / common_unit
    eye = torch.eye(3, dtype=mean.dtype, device=mean.device)
    unfloored = observation['covars'] - eye * normalized_floor**2
    covariance = ratio**2 * (rotation @ unfloored @ rotation.T) + eye * normalized_floor**2
    result['covars'] = (covariance + covariance.transpose(-1, -2)) * .5
    result['camera'] = camera_to_common(observation['camera'], transform, source_unit, common_unit)
    result['depth_patch'] = observation['depth_patch'] * ratio
    result['source_frame'] = observation['source_frame'] + source_offset
    return result


def widen_source_columns(memory, source_columns):
    """Extend source diagnostics without changing any represented Gaussian."""
    current = memory['sources'].shape[1]
    if source_columns < current:
        raise ValueError('Cannot discard existing source provenance.')
    result = {key: value.clone() for key, value in memory.items()}
    for key in ('sources', 'conflicts'):
        if memory[key].shape != (len(memory['means']), current):
            raise ValueError('Source/conflict dimensions disagree.')
        result[key] = F.pad(memory[key], (0, source_columns - current), value=False)
    return result


def build_common_memory(observation, source_columns=24):
    """Same FPS and eight-neighbor aggregation, with explicit timeline width."""
    frame = observation['source_frame']
    if not ((frame >= 0) & (frame < source_columns)).all():
        raise ValueError('Observation source is outside the declared timeline.')
    mean, chol, near, weights = merge_seeds(
        observation['means'], observation['covars'], observation['valid'])
    features = (weights[..., None] * observation['features'][near]).sum(1)
    sources = F.one_hot(frame[near], source_columns).bool().any(1)
    slots = len(mean)
    return {'means': mean, 'covars': chol @ chol.transpose(-1, -2), 'features': features,
            'opacity': mean.new_full((slots,), .7), 'support': sources.sum(-1).float(),
            'sources': sources,
            'conflicts': torch.zeros(slots, source_columns, device=mean.device, dtype=torch.bool),
            'lineage': torch.eye(slots, device=mean.device, dtype=torch.bool)}
