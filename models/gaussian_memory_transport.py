"""One frozen scene state, with matched latent and explicit geometry readouts."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from models.gaussian_memory_projection import projected_inputs
from models.gaussian_memory_surface import camera_query


def frozen_memory(seeds):
    return {'means': seeds['seed_mean'],
            'covars': seeds['seed_chol'] @ seeds['seed_chol'].transpose(-1, -2),
            'opacities': seeds['seed_mean'].new_full((len(seeds['seed_mean']),), .7),
            'features': seeds['seed_teacher']}


def read_inputs(memory, c2w, K, unit, memory_c2w=None):
    camera = c2w if memory_c2w is None else memory_c2w
    geometry = projected_inputs(memory, camera, K)
    center = torch.einsum('bij,bnj->bni', camera[:, :3, :3].transpose(-1, -2),
                          memory['means'][None] - camera[:, None, :3, 3])
    return {**{k: geometry[k] for k in ('key', 'value', 'distance2', 'mahalanobis', 'log_weight')},
            'log_depth': (center[..., 2].clamp_min(.01) * unit).log(),
            'camera_query': camera_query(c2w, K, unit)}


class GeometryTransport(nn.Module):
    def __init__(self, reader, kernel='center', center_variance=.7146816253662109, local_width=67):
        super().__init__()
        assert reader in ('local', 'latent', 'transport', 'prior_residual')
        assert kernel in ('center', 'gaussian')
        self.reader, self.kernel = reader, kernel
        width = local_width if reader in ('local', 'prior_residual') else 64
        self.norm = nn.LayerNorm(2048)
        self.query = nn.Linear(2048 + 18, width)
        self.local = nn.Linear(width, width)
        self.output = nn.Linear(width, 1)
        nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)
        self.register_buffer('center_variance', torch.tensor(float(center_variance)))
        if reader in ('latent', 'transport'):
            self.key = nn.Linear(42, 64)
            self.value = nn.Linear(33, 64, bias=False)
            self.position_scale = nn.Parameter(torch.full((4,), math.log(math.expm1(1.))))
            nn.init.zeros_(self.key.weight); nn.init.zeros_(self.key.bias)

    def forward(self, tokens, data, condition='normal', return_parts=False):
        q = F.gelu(self.query(torch.cat([self.norm(tokens.float()), data['camera_query']], -1)))
        hidden = self.local(q)
        base = hidden.new_zeros(hidden.shape[:2])
        if self.reader != 'local':
            B, P = tokens.shape[:2]; N = data['key'].shape[1]
            distance = data['distance2'] / self.center_variance if self.kernel == 'center' else data['mahalanobis']
            score = -.5 * distance[:, None] + data['log_weight'][:, None, None]
            if self.reader in ('latent', 'transport'):
                if condition != 'routing_frozen':
                    key = self.key(data['key']).reshape(B, N, 4, 16).transpose(1, 2)
                    score = q.reshape(B, P, 4, 16).transpose(1, 2) @ key.transpose(-1, -2) / 4
                    score = score - .5 * F.softplus(self.position_scale)[None, :, None, None] * distance[:, None]
                    score = score + data['log_weight'][:, None, None]
                else:
                    score = score.expand(-1, 4, -1, -1)
            if condition == 'removed':
                score = torch.full_like(score, -torch.inf)
            # The background is available only if there are no finite candidates.
            # Both matched readers use this rule; it preserves the geometric prior.
            background = torch.where(torch.isfinite(score).any(-1, keepdim=True),
                                     score.new_full((), -torch.inf), score.new_zeros(()))
            weights = torch.cat([score, background], -1).softmax(-1)[..., :-1]
            base = (weights * data['log_depth'][:, None, None]).sum(-1).mean(1)
            if self.reader in ('latent', 'transport'):
                value = self.value(torch.cat([data['value'], data['log_depth'][..., None]], -1))
                value = value.reshape(B, N, 4, 16).transpose(1, 2)
                feature = (weights @ value).transpose(1, 2).reshape(B, P, 64)
                hidden = hidden + feature
        residual = self.output(F.gelu(hidden)).squeeze(-1)
        if self.reader in ('latent', 'local') or condition == 'anchor_removed':
            prediction = residual
        elif condition == 'residual_removed':
            prediction = base
        else:
            prediction = base + residual
        if return_parts:
            return prediction, {'geometry': base, 'residual': residual}
        return prediction

    def read(self, tokens, memory, c2w, K, unit, condition='normal', memory_c2w=None):
        data = ({'camera_query': camera_query(c2w, K, unit)} if self.reader == 'local'
                else read_inputs(memory, c2w, K, unit, memory_c2w))
        return self(tokens, data, condition)
