"""Restrict the five-input relation to three symmetric independent quantities.

Repeated input slots retain the paired MLP initialization and parameter budget.
They can be folded into three first-layer columns for inference, but this module
does not claim an independently trained three-column optimizer is equivalent.
"""
import torch
from torch import nn

from models.round69_compact_relation import CompactMatcher, descriptors
from models.round73_native_relative5 import CompactNativeClassifier
from models.region_jeffreys_control import JeffreysPosition

NATIVE_ROLES = ('symmetric3', 'jeffreys_mlp')
REGION_ROLES = ('symmetric3',)


def symmetric_inputs(centers, covariance):
    original = descriptors('relative5', centers, covariance)
    metric = .5 * (original[..., 1] + original[..., 2])
    shape = .5 * (original[..., 3] + original[..., 4])
    return torch.stack((original[..., 0], metric, metric, shape, shape), -1)


class SymmetricPosition(nn.Module):
    def __init__(self, mlp):
        super().__init__()
        self.mlp = mlp

    def forward(self, centers, covariance):
        return self.mlp(symmetric_inputs(centers, covariance))


class SymmetricRegionMatcher(CompactMatcher):
    def __init__(self, role='symmetric3'):
        if role not in REGION_ROLES:
            raise ValueError(role)
        super().__init__('relative5')
        self.role = role
        self.position = SymmetricPosition(self.position.mlp)


class SymmetricNativeClassifier(CompactNativeClassifier):
    def __init__(self, role):
        if role not in NATIVE_ROLES:
            raise ValueError(role)
        super().__init__('relative5')
        self.role = role
        wrapper = SymmetricPosition if role == 'symmetric3' else JeffreysPosition
        self.position = wrapper(self.position.mlp)
