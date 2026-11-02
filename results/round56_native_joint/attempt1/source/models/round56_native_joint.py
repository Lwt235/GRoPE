"""Native point PE: center phase plus a conditional spatial preference.

This does not define a transport between surface neighborhoods. R48 references
retain their original model class and arithmetic. Diagnostics are evaluation-only.
"""
import torch
from models.geometry import rotary
from models.round5_cross_attention import _temperature
from models.round48_point_transformer import PointClassifier, ROLES as REFERENCE_ROLES

NEW_ROLES = ('joint_conditional', 'joint_fixed', 'joint_free_quadratic')
ROLES = NEW_ROLES + REFERENCE_ROLES
INTERVENTIONS = ('isotropic_trace', 'rotate_z90')


def intervene_coefficients(precision, linear, mode):
    if mode not in INTERVENTIONS:
        raise ValueError(mode)
    mean = torch.linalg.solve(precision, linear.unsqueeze(-1)).squeeze(-1)
    if mode == 'isotropic_trace':
        covariance = torch.linalg.inv(precision)
        variance = covariance.diagonal(dim1=-2, dim2=-1).mean(-1)
        eye = torch.eye(3, dtype=precision.dtype, device=precision.device)
        changed = eye/variance[..., None, None]
    else:
        rotation = precision.new_tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        changed = rotation@precision@rotation.T
    return changed, (changed@mean.unsqueeze(-1)).squeeze(-1)


def regional_bias(field, features, positions, mode='none'):
    if mode == 'none':
        return field.bias(features, positions, positions)
    if field.kind != 'conditional':
        raise ValueError('shape diagnostics require the full Gaussian')
    p, b = intervene_coefficients(*field.coefficients(features), mode)
    x = (positions.double()-positions.double().mean(-2, keepdim=True)).unsqueeze(1)
    first = (p@x.unsqueeze(-1)).squeeze(-1)+b
    i, j = torch.triu_indices(3, 3, device=x.device)
    weights = torch.where(i == j, -.5, -1.).to(x.dtype)
    query = torch.cat((first, p[..., i, j]*weights), -1)
    key = torch.cat((x, x[..., i]*x[..., j]), -1)
    return query@key.transpose(-1, -2)


class JointPointClassifier(PointClassifier):
    def __init__(self, role, intervention='none'):
        if role not in NEW_ROLES:
            raise ValueError(role)
        super().__init__(role.removeprefix('joint_'))
        if intervention != 'none' and (role != 'joint_conditional' or intervention not in INTERVENTIONS):
            raise ValueError(intervention)
        self.joint_role, self.intervention = role, intervention

    def scores(self, layer, q, k, features, positions):
        if self.training and self.intervention != 'none':
            raise ValueError('diagnostics cannot be trained')
        angles = positions.double().unsqueeze(1)
        q = rotary(q.double(), angles, self.frequency.double()).to(q.dtype)
        k = rotary(k.double(), angles, self.frequency.double()).to(k.dtype)
        score = _temperature(layer.log_temperature)*(q@k.transpose(-1, -2))
        return score+regional_bias(self.position_field, features, positions,
                                    self.intervention).to(score.dtype)


class RegionOnlyDiagnostic(PointClassifier):
    def __init__(self, intervention):
        super().__init__('conditional')
        if intervention not in INTERVENTIONS:
            raise ValueError(intervention)
        self.intervention = intervention

    def scores(self, layer, q, k, features, positions):
        if self.training:
            raise ValueError('diagnostics cannot be trained')
        score = _temperature(layer.log_temperature)*(q@k.transpose(-1, -2))
        return score+regional_bias(self.position_field, features, positions,
                                    self.intervention).to(score.dtype)


def make_model(role, intervention='none'):
    if role in NEW_ROLES:
        return JointPointClassifier(role, intervention)
    if intervention != 'none':
        if role != 'conditional':
            raise ValueError('only the conditional reference has diagnostics')
        return RegionOnlyDiagnostic(intervention)
    return PointClassifier(role)
