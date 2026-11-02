"""A matched global self-attention classifier for native 3D PE ablations.

This is not a reproduction of the local vector-attention Point Transformer.
All roles receive the same XYZ content. NoPE means no *explicit attention PE*.
The R47 Gaussian parameterization is reused unchanged, shared across layers.
"""
import torch
from torch import nn
import torch.nn.functional as F

from models.geometry import rotary
from models.round5_cross_attention import _CrossAttentionBlock, _temperature
from models.round46_native3d_pe import SpatialMap3D, sphere_frequencies
from models.round47_conditional_gaussian import GaussianPosition

GAUSSIAN = ('conditional', 'diagonal', 'isotropic', 'static', 'fixed', 'free_quadratic')
ROLES = GAUSSIAN + ('rope', 'fourier', 'mlp', 'nope')


def normalize_points(points):
    centered = points-points.mean(dim=1, keepdim=True)
    radius = centered.square().sum(-1).mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
    return centered/radius.unsqueeze(-1)


class PointClassifier(nn.Module):
    def __init__(self, role):
        super().__init__()
        if role not in ROLES:
            raise ValueError(role)
        self.role = role
        self.stem = nn.Sequential(nn.Linear(3, 96), nn.GELU(), nn.Linear(96, 96))
        self.layers = nn.ModuleList([_CrossAttentionBlock() for _ in range(4)])
        self.final_norm = nn.LayerNorm(96)
        self.classifier = nn.Sequential(nn.Linear(192, 128), nn.GELU(),
                                        nn.Dropout(.3), nn.Linear(128, 40))
        self.register_buffer('frequency', sphere_frequencies(24))
        # Construct after all shared parameters to keep backbone initialization paired.
        if role in GAUSSIAN:
            self.position_field = GaussianPosition(role+'3d')
        elif role in ('fourier', 'mlp'):
            self.position_field = SpatialMap3D(role+'3d')
        else:
            self.position_field = nn.Module()

    def scores(self, layer, q, k, features, positions):
        if self.role in ('rope', 'fourier', 'mlp'):
            x = positions.double()
            if self.role != 'rope':
                x = x+self.position_field(positions)
            q = rotary(q.double(), x.unsqueeze(1), self.frequency.double()).to(q.dtype)
            k = rotary(k.double(), x.unsqueeze(1), self.frequency.double()).to(k.dtype)
        score = _temperature(layer.log_temperature)*(q@k.transpose(-1, -2))
        if self.role in GAUSSIAN:
            score = score+self.position_field.bias(features, positions, positions).to(score.dtype)
        return score

    def forward(self, points):
        x = normalize_points(points)
        positions = 12*x
        content = self.stem(x)
        for layer in self.layers:
            q, k, v = layer._project(content)
            q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
            score = self.scores(layer, q, k, layer.attention_norm(content), positions)
            aggregate = (score.softmax(-1)@v).transpose(1, 2).reshape(*content.shape)
            content = content+layer.output(aggregate)
            content = content+layer.ffn(layer.ffn_norm(content))
        content = self.final_norm(content)
        pooled = torch.cat((content.mean(1), content.max(1).values), -1)
        return self.classifier(pooled)


def make_model(role):
    return PointClassifier(role)
