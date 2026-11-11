"""Use the existing gsplat camera projection consistently when reading memory."""
import torch

from models.gaussian_memory_attention import geometry_inputs


def projected_inputs(memory, c2w, K, height=15, width=20):
    from gsplat.cuda._torch_impl import _persp_proj
    data = geometry_inputs(memory, c2w, K, height, width)
    R = c2w[:, :3, :3].transpose(-1, -2)
    center = torch.einsum('bij,bnj->bni', R, memory['means'][None] - c2w[:, None, :3, 3])
    cov = R[:, None] @ memory['covars'][None] @ R[:, None].transpose(-1, -2)
    projected, footprint = _persp_proj(center, cov, K, width, height)
    footprint = footprint + .1 * torch.eye(2, device=K.device, dtype=K.dtype)
    y, x = torch.meshgrid(torch.arange(height, device=K.device, dtype=K.dtype) + .5,
                          torch.arange(width, device=K.device, dtype=K.dtype) + .5, indexing='ij')
    pixels = torch.stack([x, y], -1).reshape(-1, 2)
    delta = pixels[None, :, None] - projected[:, None]
    quadratic = torch.einsum('bpni,bnij,bpnj->bpn', delta, torch.linalg.inv(footprint), delta).clamp_min(0)
    variance = footprint.diagonal(dim1=-2, dim2=-1).mean(-1)
    # Behind-camera primitives retain finite unused values and the original mask.
    data['mahalanobis'] = torch.where(data['valid'][:, None], quadratic, data['mahalanobis'])
    data['isotropic'] = torch.where(data['valid'][:, None], delta.square().sum(-1) / variance[:, None], data['isotropic'])
    data['projected_variance'] = torch.where(data['valid'], variance, data['projected_variance'])
    return data
