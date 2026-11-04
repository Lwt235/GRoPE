"""Algebraically unchanged R66 input layer with a separate new Adam state."""
import torch
from torch import nn
from torch.nn import functional as F
from models.round65_actual_gs import NativeGSClassifier
from models.round66_logshape import shape_distance

ROLES = ('distance_continue', 'shape_after_distance')


class SplitInput(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.weight = original.weight
        self.bias = original.bias
        self.shape_weight = nn.Parameter(torch.zeros_like(original.weight))

    def forward(self, x):
        return F.linear(x[..., :1], self.weight, self.bias)+F.linear(x[..., 1:], self.shape_weight)


class LateShapePosition(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.mlp = original.mlp
        self.mlp[0] = SplitInput(self.mlp[0])

    def forward(self, centers, covariance):
        distance = ((centers.unsqueeze(-2)-centers.unsqueeze(-3))/.2).norm(dim=-1)
        shape = shape_distance(covariance).to(centers)
        return self.mlp(torch.stack((distance, shape), -1))


def attach_shape(model, optimizer=None):
    model.position = LateShapePosition(model.position)
    if optimizer is not None:
        options = {k:v for k,v in optimizer.param_groups[0].items() if k != 'params'}
        optimizer.add_param_group(dict(options, params=[model.position.mlp[0].shape_weight]))
    return model


def make_model(role):
    if role not in ROLES:
        raise ValueError('undeclared training-path role')
    model = NativeGSClassifier('distance_mlp')
    return attach_shape(model) if role == ROLES[1] else model
