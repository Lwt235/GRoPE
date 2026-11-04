"""A learned scalar-divergence control, with the candidate's exact MLP state."""
import torch
from torch import nn
from models.round58_support_pe import scalar_geometry
from models.round69_compact_relation import CompactMatcher

ROLES = ('jeffreys_mlp',)


def jeffreys_inputs(centers, covariance):
    d2, metric, compatibility = scalar_geometry(centers, covariance)
    scalar = (.5 * (metric + compatibility)).log1p()
    distance = d2.sqrt() / .2
    # Two independent scalars in five slots preserve the comparison's full
    # initialized state and parameter count, not five-dimensional information.
    return torch.stack((distance, scalar, scalar, scalar, scalar), -1).to(centers)


class JeffreysPosition(nn.Module):
    def __init__(self, mlp):
        super().__init__()
        self.mlp = mlp

    def forward(self, centers, covariance):
        return self.mlp(jeffreys_inputs(centers, covariance))


class JeffreysMatcher(CompactMatcher):
    def __init__(self, role='jeffreys_mlp'):
        if role not in ROLES:
            raise ValueError(role)
        super().__init__('relative5')
        self.role = role
        self.position = JeffreysPosition(self.position.mlp)
