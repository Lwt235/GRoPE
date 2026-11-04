"""One relative MLP on center distance and a known log-covariance distance.

The second column starts at zero, exactly embedding the R65 distance model's
initial function. Full and spectrum-only roles have the same active capacity.
Observed covariances are fixed inputs; this is not a new metric theorem.
"""
import torch
from torch import nn
from models.round65_actual_gs import NativeGSClassifier, common_features, shape_features

ROLES = ('log_shape_distance', 'spectrum_distance')


def shape_distance(covariance, spectrum_only=False):
    if spectrum_only:
        feature = .5 * torch.linalg.eigvalsh(covariance.double()).clamp_min(1e-15).div(.04).log()
    else:
        feature = shape_features(covariance)
    squared_norm = feature.square().sum(-1)
    squared_distance = (squared_norm.unsqueeze(-1) + squared_norm.unsqueeze(-2)
                        - 2 * feature @ feature.transpose(-1, -2)).clamp_min(0.)
    return squared_distance.log1p()


class LogShapePosition(nn.Module):
    def __init__(self, role, distance_position):
        super().__init__()
        if role not in ROLES:
            raise ValueError(role)
        self.role = role
        self.mlp = distance_position.mlp
        first = self.mlp[0]
        if first.in_features != 1:
            raise ValueError('extend the one-input distance MLP only')
        first.weight = nn.Parameter(torch.cat((first.weight.detach(),
                                              torch.zeros_like(first.weight)), dim=1))
        first.in_features = 2

    def descriptors(self, centers, covariance):
        delta = (centers.unsqueeze(-2) - centers.unsqueeze(-3)) / .2
        distance = delta.norm(dim=-1)
        shape = shape_distance(covariance, self.role == 'spectrum_distance').to(centers)
        return torch.stack((distance, shape), -1)

    def forward(self, centers, covariance):
        return self.mlp(self.descriptors(centers, covariance))


class LogShapeClassifier(NativeGSClassifier):
    def __init__(self, role):
        if role not in ROLES:
            raise ValueError(role)
        # Preserve all original random draws, including the distance MLP.
        super().__init__('distance_mlp')
        self.role = role
        self.position = LogShapePosition(role, self.position)

    def forward(self, centers, covariance, appearance):
        x = self.stem(common_features(centers, covariance, appearance))
        relative = self.position(centers, covariance)
        for layer in self.layers:
            x = layer(x, relative)
        x = self.final_norm(x)
        return self.classifier(torch.cat((x.mean(1), x.max(1).values), -1))
