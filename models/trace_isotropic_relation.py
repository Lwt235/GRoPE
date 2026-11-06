"""Keep each support's mean squared scale only in its positional relation."""
import torch
from torch import nn
from models.round69_compact_relation import CompactMatcher, descriptors
from models.round73_native_relative5 import CompactNativeClassifier

ROLES = ('trace_isotropic5',)


def trace_covariance(covariance):
    radius_squared = covariance.diagonal(dim1=-2, dim2=-1).mean(-1)
    return radius_squared[..., None, None] * torch.eye(3, device=covariance.device, dtype=covariance.dtype)


class TraceIsotropicPosition(nn.Module):
    def __init__(self, mlp):
        super().__init__()
        self.mlp = mlp

    def forward(self, centers, covariance):
        return self.mlp(descriptors('relative5', centers, trace_covariance(covariance)))


class TraceIsotropicMatcher(CompactMatcher):
    def __init__(self, role='trace_isotropic5'):
        if role not in ROLES:
            raise ValueError(role)
        super().__init__('relative5')
        self.role = role
        self.position = TraceIsotropicPosition(self.position.mlp)


class TraceIsotropicNative(CompactNativeClassifier):
    def __init__(self, role='trace_isotropic5'):
        if role not in ROLES:
            raise ValueError(role)
        super().__init__('relative5')
        self.role = role
        self.position = TraceIsotropicPosition(self.position.mlp)
