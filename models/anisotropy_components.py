"""Separate anisotropy in displacement metrics and covariance relations."""
import torch
from torch import nn

from models.round69_compact_relation import CompactMatcher, descriptors
from models.round73_native_relative5 import CompactNativeClassifier
from models.trace_isotropic_relation import trace_covariance

ROLES = ('metric_anisotropy5', 'shape_anisotropy5')


def component_descriptors(role, centers, covariance):
    if role not in ROLES:
        raise ValueError(role)
    full = descriptors('relative5', centers, covariance)
    local = descriptors('relative5', centers, trace_covariance(covariance))
    if role == 'metric_anisotropy5':
        return torch.cat((full[..., :3], local[..., 3:]), dim=-1)
    return torch.cat((local[..., :3], full[..., 3:]), dim=-1)


class ComponentPosition(nn.Module):
    def __init__(self, role, mlp):
        super().__init__()
        self.role = role
        self.mlp = mlp

    def forward(self, centers, covariance):
        return self.mlp(component_descriptors(self.role, centers, covariance))


class ComponentMatcher(CompactMatcher):
    def __init__(self, role):
        if role not in ROLES:
            raise ValueError(role)
        super().__init__('relative5')
        self.role = role
        self.position = ComponentPosition(role, self.position.mlp)


class ComponentNative(CompactNativeClassifier):
    def __init__(self, role):
        if role not in ROLES:
            raise ValueError(role)
        super().__init__('relative5')
        self.role = role
        self.position = ComponentPosition(role, self.position.mlp)
