"""Frozen native VGGT with one shared, trainable scene reader at four depths."""
import torch
from torch.utils.checkpoint import checkpoint

from models.gaussian_memory_native_pe import grid, START
from models.gaussian_memory_coarse_pe import patch_tokens
from models.gaussian_scene_memory import center_ray_descriptor, contribution_weights

READ_LAYERS = (0, 6, 12, 18)
DEPTH_LAYERS = (4, 11, 17, 23)


@torch.no_grad()
def prepare_read(model, image, camera, intrinsic, field, kind):
    patches = patch_tokens(model, image)
    height, width = image.shape[-2:]
    geometry = None
    if kind == 'L':
        features = patches[0].detach()
        valid = torch.ones(len(features), device=image.device, dtype=torch.bool)
    else:
        features, valid = field['features'], field['has_content']
        if kind in ('T', 'P'):
            K = intrinsic.clone(); K[:2] /= 14
            geometry = center_ray_descriptor(field['means'], camera, K,
                                            height // 14, width // 14, field['unit'])
        elif kind == 'G':
            K = intrinsic.clone(); K[:2] /= 14
            geometry, _ = contribution_weights(field, camera[None], K[None], height // 14, width // 14)
            geometry = geometry[0].reshape(-1, len(features))
    return dict(patches=patches.detach(), features=features, valid=valid, geometry=geometry,
                hw=(height, width))


def forward_read(model, sample, reader=None, kind='L', checkpoint_blocks=True, trace=None):
    agg = model.aggregator
    # A current query starts a fresh native sequence. All five arms use first-view tokens.
    x = torch.cat((agg.camera_token[:, 0], agg.register_token[:, 0], sample['patches']), 1)
    pos = grid(*sample['hw'], x.device)
    # Reproduce the official single-frame causal mask (all entries zero).
    native_mask = x.new_zeros(x.shape[1], x.shape[1])
    bias = reader.position_bias(kind, sample['geometry']) if reader is not None else None
    outputs = [None] * 24
    for index in range(24):
        def pair(inp, i=index):
            frame = agg.frame_blocks[i](inp, pos=pos)
            result = agg.global_blocks[i](frame, pos=pos, attn_mask=native_mask)
            if reader is not None and i in READ_LAYERS:
                residual = reader(result, sample['features'], bias, sample['valid'])
                if trace is not None and residual.requires_grad:
                    residual.register_hook(lambda grad, j=i: trace.__setitem__(str(j), float(grad.float().norm())))
                result = result + residual
            return (result, torch.cat((frame, result), -1)[:, None].float()) if i in DEPTH_LAYERS else result
        result = checkpoint(pair, x, use_reentrant=False) if checkpoint_blocks and torch.is_grad_enabled() else pair(x)
        if index in DEPTH_LAYERS:
            x, outputs[index] = result
        else:
            x = result
    with torch.autocast('cuda', enabled=False):
        dummy = torch.empty(1, 1, 3, *sample['hw'], device=x.device)
        depth, _ = model.depth_head(outputs, dummy, START)
    return depth[0, 0, ..., 0]
