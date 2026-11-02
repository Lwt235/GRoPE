"""Matched learned-relative-PE controls; R58 references remain the original class."""
import math
import torch
from torch import nn
import torch.nn.functional as F
from models.round58_support_pe import SupportMatcher, ROLES as REFERENCE_ROLES, DIM, HD, triplet_angles

NEW_ROLES = ('distance_mlp', 'descriptor_mlp', 'angle_descriptor_mlp', 'matrix_mlp')
ROLES = tuple(REFERENCE_ROLES)+NEW_ROLES
INPUTS = dict(distance_mlp=1, descriptor_mlp=23, angle_descriptor_mlp=24, matrix_mlp=15)


def symmetric_log_features(covariance):
    eigen, vectors = torch.linalg.eigh(covariance.double())
    matrix = (vectors*eigen.div(.04).log().unsqueeze(-2))@vectors.transpose(-1, -2)
    diagonal = matrix.diagonal(dim1=-2, dim2=-1)
    off = torch.stack((matrix[..., 1, 0], matrix[..., 2, 0], matrix[..., 2, 1]), -1)
    return torch.cat((diagonal, math.sqrt(2)*off), -1)


def pair_inputs(role, features, centers, covariance):
    delta = centers.double().unsqueeze(-2)-centers.double().unsqueeze(-3)
    distance = delta.norm(dim=-1, keepdim=True)/.2
    n = centers.shape[-2]
    if role == 'distance_mlp':
        return distance.to(features.dtype)
    if role == 'matrix_mlp':
        f = symmetric_log_features(covariance)
    else:
        f = features.double()
    a = f.unsqueeze(-2).expand(*f.shape[:-1], n, f.shape[-1])
    b = f.unsqueeze(-3).expand(*f.shape[:-2], n, n, f.shape[-1])
    base = torch.cat((delta/.2 if role == 'matrix_mlp' else distance, a, b), -1)
    if role == 'angle_descriptor_mlp':
        angle = triplet_angles(centers.double())/(math.pi/12)
        base = torch.cat((base.unsqueeze(-2).expand(*base.shape[:-1], 3, base.shape[-1]), angle.unsqueeze(-1)), -1)
    return base.to(features.dtype)


class RelativeMLP(nn.Module):
    def __init__(self, role):
        super().__init__()
        self.role = role
        self.mlp = nn.Sequential(nn.Linear(INPUTS[role], DIM), nn.GELU(), nn.Linear(DIM, DIM, bias=False))

    def forward(self, features, centers, covariance):
        relative = self.mlp(pair_inputs(self.role, features, centers, covariance))
        return relative.max(-2).values if self.role == 'angle_descriptor_mlp' else relative


class ControlMatcher(SupportMatcher):
    def __init__(self, role):
        if role not in NEW_ROLES:
            raise ValueError(role)
        super().__init__('nope')  # common parameters before all role-specific draws
        self.role = role
        self.position = RelativeMLP(role)

    def forward(self, features, centers, covariance):
        x = self.stem(features)
        r = self.position(features, centers, covariance)
        for index, layer in enumerate(self.layers):
            x = layer(x, r if index % 2 == 0 else None, cross=index % 2 == 1)
        descriptor = F.normalize(self.head(self.final_norm(x)), dim=-1)
        temperature = self.log_temperature.clamp(max=math.log(60.)).exp()
        return temperature*(descriptor[:, 0]@descriptor[:, 1].transpose(-1, -2))


def make_model(role):
    if role in REFERENCE_ROLES:
        return SupportMatcher(role)
    return ControlMatcher(role)
