"""Same-capacity C=Q control; geometry remains in the common content branch."""
import math
import torch
from models.round71_target_projection import TargetProjectionNVS
from models.round72_rgb_projection import RGBTargetProjectionNVS

ROLES=('observed_target_fixedshape','rgb_target_fixedshape')


def fixed_shape_inputs(descriptors):
    if descriptors.shape[-1]!=5:raise ValueError('expected the frozen five-input descriptor')
    constant=torch.full_like(descriptors[...,0],math.log(3.))
    return torch.stack((descriptors[...,0],descriptors[...,1],descriptors[...,1],constant,constant),-1)


def fixed_shape_hook(module,args):
    return (fixed_shape_inputs(args[0]),)


class ObservedFixedShape(TargetProjectionNVS):
    def __init__(self):
        super().__init__('observed_target_relative5')
        self.role=ROLES[0]
        self.relative.register_forward_pre_hook(fixed_shape_hook)


class RGBFixedShape(RGBTargetProjectionNVS):
    def __init__(self):
        super().__init__('rgb_target_relative5')
        self.role=ROLES[1]
        self.relative.register_forward_pre_hook(fixed_shape_hook)


def FixedShapeNVS(role):
    if role==ROLES[0]:return ObservedFixedShape()
    if role==ROLES[1]:return RGBFixedShape()
    raise ValueError(role)
