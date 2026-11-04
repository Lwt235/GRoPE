"""The unchanged R69 five-input PE inside the unchanged R65 native-GS backbone."""
import torch
from models.round65_actual_gs import NativeGSClassifier,common_features
from models.round69_compact_relation import CompactPosition

ROLES=('relative5',)


class CompactNativeClassifier(NativeGSClassifier):
    def __init__(self,role):
        if role not in ROLES:raise ValueError(role)
        super().__init__('nope')
        self.role=role
        self.position=CompactPosition(role)

    def forward(self,centers,covariance,appearance,disable_relation=False):
        x=self.stem(common_features(centers,covariance,appearance))
        relative=None if disable_relation else self.position(centers,covariance)
        for layer in self.layers:x=layer(x,relative,centers,None)
        x=self.final_norm(x)
        return self.classifier(torch.cat((x.mean(1),x.max(1).values),-1))
