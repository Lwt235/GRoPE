"""A shared residual writer using only the existing compact reference state."""
import math

import torch
from torch import nn


class CompactGaussianWriter(nn.Module):
    def __init__(self, geometry=True, width=64):
        super().__init__()
        self.geometry = geometry
        self.norm = nn.LayerNorm(32)
        self.net = nn.Sequential(nn.Linear(38, width), nn.GELU(),
                                 nn.Linear(width, width), nn.GELU(),
                                 nn.Linear(width, 42))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, seeds):
        mean, chol, content = (seeds[k].float() for k in
                               ('seed_mean', 'seed_chol', 'seed_teacher'))
        spatial = torch.cat([mean, chol.diagonal(dim1=-2, dim2=-1).log()], -1)
        delta = self.net(torch.cat([self.norm(content), spatial], -1))
        if self.geometry:
            center = mean + .25 * delta[:, :3].tanh()
            transform = torch.diag_embed((.7 * delta[:, 3:6].tanh()).exp())
            transform[:, 1, 0] = .5 * delta[:, 6].tanh()
            transform[:, 2, 0] = .5 * delta[:, 7].tanh()
            transform[:, 2, 1] = .5 * delta[:, 8].tanh()
            factor = chol @ transform
            opacity = torch.sigmoid(delta[:, 9] + math.log(.7 / .3))
        else:
            center, factor = mean, chol
            opacity = mean.new_full((len(mean),), .7)
        return {'means': center.contiguous(),
                'covars': (factor @ factor.transpose(-1, -2)).contiguous(),
                'opacities': opacity.contiguous(),
                'features': (content + delta[:, 10:]).contiguous()}
