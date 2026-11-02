"""Gaussian-pair positional similarity, alongside unchanged center RoPE.

The geometric score is minus half the Jeffreys divergence. It compares two
latent Gaussians; it is neither a probability-mass integrator nor texture transport.
The distance control removes only covariance compatibility, at equal capacity.
"""
import math
import torch
from torch import nn
from models.geometry import rotary
from models.round5_cross_attention import HEADS, MODEL_DIM, _temperature
from models.round48_point_transformer import PointClassifier
from models.round56_native_joint import ROLES as REFERENCE_ROLES, make_model as reference_model

NEW_ROLES = ('pair_full', 'pair_distance', 'pair_center')
ROLES = NEW_ROLES + REFERENCE_ROLES
INTERVENTIONS = ('shape_off',)
DIAGNOSTIC_ROLES = ('pair_full',)


def svec(matrix):
    i, j = torch.triu_indices(3, 3, device=matrix.device)
    weight = torch.ones(i.shape, dtype=matrix.dtype, device=matrix.device)
    weight[i != j] = math.sqrt(2.)
    return matrix[..., i, j]*weight


def gaussian_vectors(center, precision, covariance=None):
    """19 Q/K channels; the omitted query constant is returned separately."""
    natural = torch.cat(((precision@center.unsqueeze(-1)).squeeze(-1),
                         -.5*svec(precision)), -1)
    second = center.unsqueeze(-1)*center.unsqueeze(-2)
    if covariance is not None:
        second = second+covariance
    moment = torch.cat((center, svec(second)), -1)
    self_product = (natural*moment).sum(-1, keepdim=True)/2
    one = torch.ones_like(self_product)
    query = torch.cat((natural/math.sqrt(2.), moment/math.sqrt(2.), one), -1)
    key = torch.cat((moment/math.sqrt(2.), natural/math.sqrt(2.), -self_product), -1)
    return query, key, self_product


class GaussianPairPosition(nn.Module):
    def __init__(self, role):
        super().__init__()
        if role not in NEW_ROLES:
            raise ValueError(role)
        self.kind, self.width = role, 4.
        self.size = 3 if role == 'pair_center' else 9
        self.predictor = nn.Linear(MODEL_DIM, HEADS*self.size)
        nn.init.zeros_(self.predictor.weight)
        nn.init.zeros_(self.predictor.bias)

    def distribution(self, features, need_covariance=True):
        batch, count, _ = features.shape
        raw = self.predictor(features).reshape(batch, count, HEADS, self.size)
        raw = raw.permute(0, 2, 1, 3).double()
        mean = 4*torch.tanh(raw[..., :3]/4)
        eye = torch.eye(3, dtype=raw.dtype, device=raw.device)
        if self.kind == 'pair_center':
            precision = (eye/self.width**2).expand(*raw.shape[:-1], 3, 3)
            covariance = (eye*self.width**2).expand_as(precision) if need_covariance else None
        else:
            # Same bounded mean/Cholesky parameterization as R47/R48/R56.
            shape = 3*torch.tanh(raw[..., 3:]/3)
            lower = torch.zeros(*raw.shape[:-1], 3, 3, dtype=raw.dtype, device=raw.device)
            i, j = torch.tril_indices(3, 3, offset=-1, device=raw.device)
            lower[..., i, j] = shape[..., 3:]
            factor = torch.diag_embed(shape[..., :3].exp())+lower
            precision = (factor@factor.transpose(-1, -2))/self.width**2
            covariance = torch.cholesky_inverse(factor/self.width) if need_covariance else None
        return precision, covariance, mean

    def vectors(self, features, positions, shape_off=False):
        include_shape = self.kind == 'pair_full' and not shape_off
        precision, covariance, mean = self.distribution(features, include_shape)
        origin = positions.double().mean(-2, keepdim=True)
        center = (positions.double()-origin).unsqueeze(1)+mean
        return gaussian_vectors(center, precision, covariance)

    def bias(self, features, positions, shape_off=False):
        query, key, row_constant = self.vectors(features, positions, shape_off)
        # Removing the query constant leaves the exact symmetric score. It is
        # unnecessary for softmax algebra, but avoids a large irrelevant offset.
        return query@key.transpose(-1, -2)-row_constant


class GaussianPairClassifier(PointClassifier):
    def __init__(self, role, intervention='none'):
        if role not in NEW_ROLES:
            raise ValueError(role)
        super().__init__('rope')
        if intervention != 'none' and (role != 'pair_full' or intervention not in INTERVENTIONS):
            raise ValueError(intervention)
        self.pair_role, self.intervention = role, intervention
        self.position_field = GaussianPairPosition(role)

    def scores(self, layer, q, k, features, positions):
        if self.training and self.intervention != 'none':
            raise ValueError('fixed-checkpoint intervention cannot be trained')
        angles = positions.double().unsqueeze(1)
        q = rotary(q.double(), angles, self.frequency.double()).to(q.dtype)
        k = rotary(k.double(), angles, self.frequency.double()).to(k.dtype)
        score = _temperature(layer.log_temperature)*(q@k.transpose(-1, -2))
        return score+self.position_field.bias(features, positions,
                                              self.intervention == 'shape_off').to(score.dtype)


def make_model(role, intervention='none'):
    if role in NEW_ROLES:
        return GaussianPairClassifier(role, intervention)
    if intervention != 'none':
        raise ValueError('R57 diagnostics only apply to pair_full')
    return reference_model(role)
