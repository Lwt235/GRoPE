"""Remove covariance from token content, retaining each unchanged explicit PE.

This is an information-routing ablation of the native-GS classifier, not a new
PE. The seven retained stem columns inherit their original initialization.
"""
import torch
from torch import nn
from models.round65_actual_gs import NativeGSClassifier

ROLES = ('gaussian_relation', 'direct_mlp', 'distance_mlp')
CONTENT_COLUMNS = (0, 1, 2, 9, 10, 11, 12)


class ContentRouteClassifier(NativeGSClassifier):
    def __init__(self, role):
        if role not in ROLES:
            raise ValueError(role)
        super().__init__(role)
        original = self.stem[0]
        # Preserve every original random draw, including the PE initializer.
        with torch.random.fork_rng(devices=[]):
            reduced = nn.Linear(len(CONTENT_COLUMNS), original.out_features)
        with torch.no_grad():
            reduced.weight.copy_(original.weight[:, CONTENT_COLUMNS])
            reduced.bias.copy_(original.bias)
        self.stem[0] = reduced

    def forward(self, centers, covariance, appearance):
        x = self.stem(torch.cat((centers, appearance.to(centers)), -1))
        relative = self.position(centers, covariance)
        for layer in self.layers:
            x = layer(x, relative, centers, None)
        x = self.final_norm(x)
        return self.classifier(torch.cat((x.mean(1), x.max(1).values), -1))
