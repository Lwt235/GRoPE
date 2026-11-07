"""Direction-neutral pair relations with matched self terms and initialization."""
import torch
from torch import nn

from models.round69_compact_relation import CompactMatcher
from models.round73_native_relative5 import CompactNativeClassifier

ROLES = ('mean_precision5',)


def mean_precision_descriptors(centers, covariance):
    c, s = centers.double(), covariance.double()
    precision = torch.linalg.inv(s)
    radius = s.diagonal(dim1=-2, dim2=-1).mean(-1)
    mean_precision = precision.diagonal(dim1=-2, dim2=-1).mean(-1)
    delta = c.unsqueeze(-2) - c.unsqueeze(-3)
    distance_squared = delta.square().sum(-1)
    metric_i = mean_precision.unsqueeze(-1) * distance_squared
    metric_j = metric_i.transpose(-1, -2)
    shape_i = 3 * mean_precision.unsqueeze(-1) * radius.unsqueeze(-2)
    # A token is not independently rotated relative to itself. Keep tr(P_i S_i)=3,
    # rather than introducing a new anisotropy-dependent diagonal attention bias.
    diagonal = torch.eye(c.shape[-2], device=c.device, dtype=torch.bool)
    shape_i = shape_i.masked_fill(diagonal, 3.)
    shape_j = shape_i.transpose(-1, -2)
    terms = torch.stack((metric_i, metric_j, shape_i, shape_j), dim=-1)
    return torch.cat(((distance_squared.sqrt() / .2).unsqueeze(-1),
                      terms.clamp_min(0).log1p()), dim=-1).to(centers)


class MeanPrecisionPosition(nn.Module):
    def __init__(self, mlp):
        super().__init__()
        self.mlp = mlp

    def forward(self, centers, covariance):
        return self.mlp(mean_precision_descriptors(centers, covariance))


class MeanPrecisionMatcher(CompactMatcher):
    def __init__(self, role='mean_precision5'):
        if role not in ROLES:
            raise ValueError(role)
        super().__init__('relative5')
        self.role = role
        self.position = MeanPrecisionPosition(self.position.mlp)


class MeanPrecisionNative(CompactNativeClassifier):
    def __init__(self, role='mean_precision5'):
        if role not in ROLES:
            raise ValueError(role)
        super().__init__('relative5')
        self.role = role
        self.position = MeanPrecisionPosition(self.position.mlp)
