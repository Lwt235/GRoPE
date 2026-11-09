"""Remove mean-precision amplification from off-diagonal full5 relations."""
import torch
from torch import nn

from models.round58_support_pe import gaussian_terms
from models.round69_compact_relation import CompactMatcher
from models.round73_native_relative5 import CompactNativeClassifier

ROLES = ('normalized_precision5',)


def normalized_precision_descriptors(centers, covariance):
    d2, mi, mj, ti, tj = gaussian_terms(centers, covariance)
    s = covariance.double()
    radius = s.diagonal(dim1=-2, dim2=-1).mean(-1)
    mean_precision = torch.linalg.inv(s).diagonal(dim1=-2, dim2=-1).mean(-1)
    amplification = radius * mean_precision
    metric_i = mi / amplification.unsqueeze(-1)
    shape_i = ti / amplification.unsqueeze(-1)
    diagonal = torch.eye(centers.shape[-2], dtype=torch.bool, device=centers.device)
    # Preserve the original computed self terms, including their numerical path.
    shape_i = torch.where(diagonal, ti, shape_i)
    terms = torch.stack((metric_i, metric_i.transpose(-1, -2),
                         shape_i, shape_i.transpose(-1, -2)), dim=-1)
    return torch.cat(((d2.sqrt() / .2).unsqueeze(-1),
                      terms.clamp_min(0).log1p()), dim=-1).to(centers)


class NormalizedPrecisionPosition(nn.Module):
    def __init__(self, mlp):
        super().__init__()
        self.mlp = mlp

    def forward(self, centers, covariance):
        return self.mlp(normalized_precision_descriptors(centers, covariance))


class NormalizedPrecisionMatcher(CompactMatcher):
    def __init__(self, role='normalized_precision5'):
        if role not in ROLES:
            raise ValueError(role)
        super().__init__('relative5')
        self.role = role
        self.position = NormalizedPrecisionPosition(self.position.mlp)


class NormalizedPrecisionNative(CompactNativeClassifier):
    def __init__(self, role='normalized_precision5'):
        if role not in ROLES:
            raise ValueError(role)
        super().__init__('relative5')
        self.role = role
        self.position = NormalizedPrecisionPosition(self.position.mlp)
