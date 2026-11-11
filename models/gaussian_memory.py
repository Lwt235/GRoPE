"""One shared writer; saved Gaussian state is the entire query-time scene input."""
import torch
from torch import nn


class GaussianWriter(nn.Module):
    def __init__(self, learn_geometry=True, width=128, feature_dim=32):
        super().__init__()
        self.learn_geometry = learn_geometry
        self.feature_dim = feature_dim
        self.context_norm = nn.LayerNorm(2048)
        self.net = nn.Sequential(nn.Linear(2054, width), nn.GELU(),
                                 nn.Linear(width, width), nn.GELU(), nn.Linear(width, 10 + feature_dim))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, seeds):
        mean, chol = seeds['seed_mean'], seeds['seed_chol']
        spatial = torch.cat([mean, chol.diagonal(dim1=-2, dim2=-1).log()], -1)
        x = self.net(torch.cat([self.context_norm(seeds['seed_context'].float()), spatial], -1))
        if self.learn_geometry:
            center = mean + .25 * x[:, :3].tanh()
            diag = (.7 * x[:, 3:6].tanh()).exp()
            transform = torch.diag_embed(diag)
            transform[:, 1, 0] = .5 * x[:, 6].tanh()
            transform[:, 2, 0] = .5 * x[:, 7].tanh()
            transform[:, 2, 1] = .5 * x[:, 8].tanh()
            factor = chol @ transform
        else:
            center, factor = mean, chol
        return {'means': center.contiguous(),
                'covars': (factor @ factor.transpose(-1, -2)).contiguous(),
                'opacities': (x[:, 9] + 0.8472978604).sigmoid().contiguous(),
                'features': (seeds['seed_teacher'] + x[:, 10:]).contiguous()}


def read_memory(memory, c2w, K, height=15, width=20):
    from gsplat import rasterization
    # No images, reference features, scene identifier or model parameters are
    # accepted here. The exact same saved state supports any query camera.
    colors, alpha, _ = rasterization(
        means=memory['means'], covars=memory['covars'], quats=None, scales=None,
        opacities=memory['opacities'], colors=memory['features'],
        viewmats=torch.linalg.inv(c2w).contiguous(), Ks=K.contiguous(),
        width=width, height=height, near_plane=.01, far_plane=100.,
        eps2d=.1, packed=False, channel_chunk=32,
    )
    return colors, alpha


def seeds_to_device(data, device):
    return {k: data[k].to(device).float() for k in ('seed_mean', 'seed_chol', 'seed_context', 'seed_teacher')}


def detached_memory(memory, device='cpu'):
    return {k: v.detach().to(device).contiguous() for k, v in memory.items()}
