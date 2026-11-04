"""Same complete regional content, five fixed relation encodings."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from models.round58_support_pe import DIM, HD, AttentionBlock, SupportPosition
from models.round65_actual_gs import DirectPosition, shape_features
from models.round61_normal_relations import NormalPosition

ROLES = ('distance_mlp','direct_mlp','normal_scale','gaussian_relation','shuffled_gaussian')


class FixedRegionMatcher(nn.Module):
    def __init__(self,role):
        super().__init__()
        if role not in ROLES:raise ValueError(role)
        self.role=role
        self.stem=nn.Sequential(nn.Linear(20,DIM),nn.GELU(),nn.Linear(DIM,DIM))
        self.layers=nn.ModuleList([AttentionBlock() for _ in range(4)])
        self.final_norm=nn.LayerNorm(DIM);self.head=nn.Linear(DIM,HD)
        self.log_temperature=nn.Parameter(torch.tensor(2.2))
        if role in ROLES[:2]:self.position=DirectPosition(role)
        elif role=='normal_scale':self.position=NormalPosition('normal_ppf_mlp')
        else:self.position=SupportPosition('invariant_mlp')

    def forward(self,features,centers,covariance,permutation):
        content=torch.cat((features,centers,shape_features(covariance).to(features)),-1)
        x=self.stem(content)
        pe_covariance=covariance
        if self.role=='shuffled_gaussian':
            pe_covariance=torch.gather(covariance,-3,
                permutation[...,None,None].expand(*permutation.shape,3,3))
        relative=self.position(centers,pe_covariance)
        for index,layer in enumerate(self.layers):
            x=layer(x,relative if index%2==0 else None,cross=index%2==1)
        descriptor=F.normalize(self.head(self.final_norm(x)),dim=-1)
        temperature=self.log_temperature.clamp(max=math.log(60.)).exp()
        return temperature*(descriptor[:,0]@descriptor[:,1].transpose(-1,-2))
