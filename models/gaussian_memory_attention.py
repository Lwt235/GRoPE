"""A single cross-attention read from a frozen, explicitly stored Gaussian state.

No image, teacher feature, scene ID or writer context enters this interface.
All variants see the same complete geometry through the key MLP.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


def geometry_inputs(memory, c2w, K, height=15, width=20):
    """Project geometry once; all tensors have a leading query-camera dimension."""
    R = c2w[:, :3, :3].transpose(-1, -2)
    center = torch.einsum('bij,bnj->bni', R,
                          memory['means'][None] - c2w[:, None, :3, 3])
    cov = R[:, None] @ memory['covars'][None] @ R[:, None].transpose(-1, -2)
    z = center[..., 2].clamp_min(.01)
    homogeneous = torch.einsum('bij,bnj->bni', K, center)
    projected = homogeneous[..., :2] / z[..., None]
    J = (K[:, None, :2] - projected[..., None] * K[:, None, 2:3]) / z[..., None, None]
    footprint = J @ cov @ J.transpose(-1, -2)
    footprint = footprint + .1 * torch.eye(2, device=K.device, dtype=K.dtype)
    y, x = torch.meshgrid(torch.arange(height, device=K.device, dtype=K.dtype) + .5,
                          torch.arange(width, device=K.device, dtype=K.dtype) + .5, indexing='ij')
    pixels = torch.stack([x, y], -1).reshape(-1, 2)
    rays = torch.einsum('bij,pj->bpi', torch.linalg.inv(K),
                       torch.cat([pixels, torch.ones_like(pixels[:, :1])], -1))
    rays = F.normalize(rays, dim=-1)
    normalized_pixels = pixels / pixels.new_tensor([width, height]) * 2 - 1
    query = torch.cat([normalized_pixels[None].expand(len(K), -1, -1), rays], -1)
    std = cov.diagonal(dim1=-2, dim2=-1).clamp_min(1e-12).sqrt()
    corr = cov / (std[..., :, None] * std[..., None, :])
    # Log marginal scales and the three correlations retain the full SPD matrix.
    spatial = torch.cat([center.sign() * center.abs().log1p(), std.log(),
                         corr[..., [1, 2, 2], [0, 0, 1]],
                         memory['opacities'][None, :, None].expand(len(K), -1, -1)], -1)
    features = memory['features'][None].expand(len(K), -1, -1)
    delta = pixels[None, :, None] - projected[:, None]
    inverse = torch.linalg.inv(footprint)
    mahalanobis = torch.einsum('bpni,bnij,bpnj->bpn', delta, inverse, delta).clamp_min(0)
    dist2 = delta.square().sum(-1)
    projected_variance = footprint.diagonal(dim1=-2, dim2=-1).mean(-1)
    valid = ((center[..., 2] > .01) & (center[..., 2] < 100.) &
             (memory['opacities'][None] > 0))
    log_opacity = memory['opacities'].clamp_min(1e-12).log()[None].expand_as(valid)
    log_weight = log_opacity.masked_fill(~valid, -torch.inf)
    return {'query': query, 'key': torch.cat([features, spatial], -1),
            'value': features, 'log_weight': log_weight, 'distance2': dist2,
            'mahalanobis': mahalanobis, 'isotropic': dist2 / projected_variance[:, None],
            'projected_variance': projected_variance, 'valid': valid}


class GaussianAttentionReader(nn.Module):
    def __init__(self, variant, center_variance, width=64, heads=4, feature_dim=32):
        super().__init__()
        if variant not in ('mlp', 'center', 'gaussian'):
            raise ValueError(variant)
        self.variant, self.heads, self.width = variant, heads, width
        self.query = nn.Sequential(nn.Linear(5, width), nn.GELU(), nn.Linear(width, width))
        self.key = nn.Sequential(nn.Linear(feature_dim + 10, width), nn.GELU(), nn.Linear(width, width))
        self.value = nn.Linear(feature_dim, width, bias=False)
        self.output = nn.Linear(width, feature_dim, bias=False)
        # An initially information-preserving value/output pair, common to all groups.
        with torch.no_grad():
            nn.init.orthogonal_(self.value.weight)
            self.output.weight.copy_(self.value.weight.T)
        self.position_scale = nn.Parameter(torch.full((heads,), math.log(math.expm1(1.))),
                                           requires_grad=variant != 'mlp')
        self.register_buffer('center_variance', torch.tensor(float(center_variance)))

    def forward(self, data, condition='normal', return_attention=False):
        B, P = data['query'].shape[:2]
        N = data['key'].shape[1]
        H, D = self.heads, self.width // self.heads
        q = self.query(data['query']).view(B, P, H, D).transpose(1, 2)
        k = self.key(data['key']).view(B, N, H, D).transpose(1, 2)
        v = self.value(data['value']).view(B, N, H, D).transpose(1, 2)
        score = (q @ k.transpose(-1, -2)) / math.sqrt(D)
        if self.variant != 'mlp' and condition != 'no_position':
            distance = (data['distance2'] / self.center_variance if self.variant == 'center'
                        else data['isotropic' if condition == 'isotropic' else 'mahalanobis'])
            score = score - .5 * F.softplus(self.position_scale)[None, :, None, None] * distance[:, None]
        score = score + data['log_weight'][:, None, None]
        # One common zero-valued background slot makes empty memory well-defined.
        score = torch.cat([score, score.new_zeros(B, H, P, 1)], -1)
        attention = score.softmax(-1)
        read = (attention[..., :-1] @ v).transpose(1, 2).reshape(B, P, self.width)
        output = self.output(read)
        if return_attention:
            return output, attention
        return output

    def read_memory(self, memory, c2w, K, height=15, width=20, condition='normal'):
        inputs = geometry_inputs(memory, c2w, K, height, width)
        return self(inputs, condition).reshape(len(K), height, width, -1)
