"""Compact jointly written Gaussian memory, one attention read, one surface head."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from models.gaussian_memory import GaussianWriter
from models.gaussian_memory_projection import projected_inputs


def camera_query(c2w, K, unit):
    y, x = torch.meshgrid(torch.arange(15, device=K.device, dtype=K.dtype) + .5,
                          torch.arange(20, device=K.device, dtype=K.dtype) + .5, indexing='ij')
    pixels = torch.stack([x, y], -1).reshape(-1, 2)
    ray = torch.einsum('bij,pj->bpi', torch.linalg.inv(K),
                       torch.cat([pixels, torch.ones_like(pixels[:, :1])], -1))
    ray = F.normalize(ray, dim=-1)
    xy = (pixels / pixels.new_tensor([20, 15]) * 2 - 1)[None].expand(len(K), -1, -1)
    pose = torch.cat([c2w[:, :3, :3].reshape(-1, 9), c2w[:, :3, 3]], -1)
    pose = pose[:, None].expand(-1, 300, -1)
    scale = K.new_full((len(K), 300, 1), math.log(float(unit)))
    return torch.cat([xy, ray, pose, scale], -1)


class SurfaceMemory(nn.Module):
    def __init__(self, variant, local_width=191, center_variance=.7146816253662109):
        super().__init__()
        assert variant in ('local', 'fixed', 'center', 'gaussian')
        self.variant = variant
        width = local_width if variant == 'local' else 64
        self.norm = nn.LayerNorm(2048)
        self.query = nn.Linear(2048 + 18, width)
        self.local = nn.Linear(width, width)
        self.output = nn.Linear(width, 1)
        nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)
        if variant != 'local':
            self.writer = GaussianWriter(learn_geometry=variant != 'fixed')
            self.key = nn.Linear(42, 64)
            self.value = nn.Linear(33, 64, bias=False)
            self.position_scale = nn.Parameter(torch.full((4,), math.log(math.expm1(1.))))
            self.register_buffer('center_variance', torch.tensor(center_variance))

    def write(self, seeds):
        assert self.variant != 'local'
        return self.writer(seeds)

    def read(self, tokens, memory, c2w, K, unit, condition='normal', memory_c2w=None):
        query = camera_query(c2w, K, unit)
        q = F.gelu(self.query(torch.cat([self.norm(tokens.float()), query], -1)))
        local = self.local(q)
        if self.variant != 'local':
            read_camera = c2w if memory_c2w is None else memory_c2w
            g = projected_inputs(memory, read_camera, K)
            B, P = q.shape[:2]; N = len(memory['means'])
            key = self.key(g['key']).reshape(B, N, 4, 16).transpose(1, 2)
            center = torch.einsum('bij,bnj->bni', read_camera[:, :3, :3].transpose(-1, -2),
                                  memory['means'][None] - read_camera[:, None, :3, 3])
            log_depth = (center[..., 2:3].clamp_min(.01) * unit).log()
            values = self.value(torch.cat([g['value'], log_depth], -1)).reshape(B, N, 4, 16).transpose(1, 2)
            score = q.reshape(B, P, 4, 16).transpose(1, 2) @ key.transpose(-1, -2) / 4
            distance = g['distance2'] / self.center_variance if self.variant == 'center' else g['mahalanobis']
            score = score - .5 * F.softplus(self.position_scale)[None, :, None, None] * distance[:, None]
            score = score + g['log_weight'][:, None, None]
            if condition == 'removed':
                score = torch.full_like(score, -torch.inf)
            weights = torch.cat([score, score.new_zeros(B, 4, P, 1)], -1).softmax(-1)
            read = (weights[..., :-1] @ values).transpose(1, 2).reshape(B, P, 64)
            local = local + read
        return self.output(F.gelu(local)).squeeze(-1)

    def forward(self, tokens, seeds, c2w, K, unit):
        memory = None if self.variant == 'local' else self.write(seeds)
        return self.read(tokens, memory, c2w, K, unit)
