"""Matched PRoPE x local-plane controls; R52 Gaussian references are unchanged."""
import torch
from torch import nn
from models.round52_camera_frame import CameraFrameMatcher, camera_lift, INPUT_FIELDS

ROLES = ('nope','mlp','hrope','camera_only','camera_offset','camera_plane',
         'camera_iso','camera_gs','camera_affine')
NEW_ROLES = ('camera_offset','camera_plane')


def plane_frame(raw, offset_only=False):
    """HRoPE Eq.7 with a nonsingular two-coordinate chart for the unit normal.

    n=(2a,2b,1-a*a-b*b)/(1+a*a+b*b). The corresponding quaternion
    (1,-b,a,0)/sqrt(1+a*a+b*b) is the minimal rotation from e3 to n.
    This chart covers the sphere except its antipode at infinite coordinates.
    z=tanh(raw[-1]) uses R52's same +/-3m translation bound.
    """
    z = raw[...,-1].tanh()
    if offset_only:
        a = b = torch.zeros_like(z)
    else:
        a,b = raw[...,0],raw[...,1]
    d = 1+a*a+b*b
    r = torch.stack((1-2*a*a/d,-2*a*b/d,2*a/d,
                     -2*a*b/d,1-2*b*b/d,2*b/d,
                     -2*a/d,-2*b/d,1-2*(a*a+b*b)/d),-1).reshape(*z.shape,3,3)
    n = r[...,:,2]
    f = torch.zeros(*z.shape,4,4,dtype=raw.dtype,device=raw.device)
    fi = torch.zeros_like(f)
    f[...,:3,:3]=r;f[...,:3,3]=z[...,None]*n;f[...,3,3]=1
    fi[...,:3,:3]=r.transpose(-1,-2);fi[...,2,3]=-z;fi[...,3,3]=1
    return f,fi


class CalibratedPlaneMatcher(CameraFrameMatcher):
    def __init__(self, variant):
        if variant not in NEW_ROLES:raise ValueError(variant)
        # Keep the inherited camera_only encode path: it constructs known P
        # each layer and calls the overridden matrices method below.
        super().__init__('camera_only')
        self.plane_variant=variant
        self.field=nn.Linear(96,1 if variant=='camera_offset' else 3)
        nn.init.zeros_(self.field.weight);nn.init.zeros_(self.field.bias)

    def matrices(self, features, intrinsics, camera=None):
        if camera is None:raise ValueError('known camera transform required')
        f,fi=plane_frame(self.field(features),self.plane_variant=='camera_offset')
        p,pi=camera
        return p[:,None]@f,fi@pi[:,None]


def make_model(role):
    if role in NEW_ROLES:return CalibratedPlaneMatcher(role)
    if role not in ROLES:raise ValueError(role)
    return CameraFrameMatcher(role)
