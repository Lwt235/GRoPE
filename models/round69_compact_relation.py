"""Remove columns of the existing relation descriptor, without new modules."""
import torch
from torch import nn
from models.round58_support_pe import DIM, HD, AttentionBlock, gaussian_terms
from models.round68_fixed_region import FixedRegionMatcher

ROLES=('relative5','metric3','spectrum7')


def descriptors(role,centers,covariance):
    d2,mi,mj,ti,tj=gaussian_terms(centers,covariance)
    distance=(d2.sqrt()/.2).unsqueeze(-1)
    if role=='spectrum7':
        spectrum=.5*torch.linalg.eigvalsh(covariance.double()).clamp_min(1e-15).div(.04).log()
        n=centers.shape[-2]
        a=spectrum.unsqueeze(-2).expand(*spectrum.shape[:-1],n,3)
        b=spectrum.unsqueeze(-3).expand(*spectrum.shape[:-2],n,n,3)
        return torch.cat((distance,a,b),-1).to(centers)
    terms=(mi,mj,ti,tj) if role=='relative5' else (mi,mj)
    if role not in ROLES:raise ValueError(role)
    return torch.cat((distance,torch.stack(terms,-1).clamp_min(0).log1p()),-1).to(centers)


class CompactPosition(nn.Module):
    def __init__(self,role):
        super().__init__();self.role=role
        self.mlp=nn.Sequential(nn.Linear({'relative5':5,'metric3':3,'spectrum7':7}[role],DIM),
                               nn.GELU(),nn.Linear(DIM,DIM,bias=False))

    def forward(self,centers,covariance):
        return self.mlp(descriptors(self.role,centers,covariance))


class CompactMatcher(FixedRegionMatcher):
    def __init__(self,role):
        nn.Module.__init__(self)
        if role not in ROLES:raise ValueError(role)
        self.role=role
        # Same construction order as R68; only the final PE module differs.
        self.stem=nn.Sequential(nn.Linear(20,DIM),nn.GELU(),nn.Linear(DIM,DIM))
        self.layers=nn.ModuleList([AttentionBlock() for _ in range(4)])
        self.final_norm=nn.LayerNorm(DIM);self.head=nn.Linear(DIM,HD)
        self.log_temperature=nn.Parameter(torch.tensor(2.2))
        self.position=CompactPosition(role)
