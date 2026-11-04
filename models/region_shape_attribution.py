"""Matched fixed-covariance control for the existing five-input regional PE."""
import math
import torch
from models.round69_compact_relation import CompactMatcher

ROLES = ('fixed_isotropic5',)


def fixed_inputs(descriptor):
    if descriptor.shape[-1] != 5:
        raise ValueError('expected five existing relation inputs')
    distance = descriptor[..., 0]
    radial = distance.square().log1p()
    constant = torch.full_like(distance, math.log(4.))
    return torch.stack((distance, radial, radial, constant, constant), -1)


def fixed_hook(module, args):
    return (fixed_inputs(args[0]),)


class FixedIsotropicMatcher(CompactMatcher):
    def __init__(self, role='fixed_isotropic5'):
        if role not in ROLES:
            raise ValueError(role)
        super().__init__('relative5')
        self.role = role
        self.position.mlp.register_forward_pre_hook(fixed_hook)
