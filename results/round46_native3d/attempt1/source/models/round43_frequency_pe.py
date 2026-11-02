"""Compact mean/covariance frequency constraints, not a new Fourier kernel."""
import math
import numpy as np
import torch
from torch import nn
from models.geometry import rotary
from models.round5_cross_attention import Round5CrossAttentionMatcher,METHODS

NEW_ROLES=('gaussian_full','gaussian_diag','gaussian_iso','fixed_gaussian',
           'shared_affine','group_affine','free_frequency')
ROLES=NEW_ROLES+METHODS


class FrequencyBank(nn.Module):
    def __init__(self,role,reference,groups=4):
        super().__init__()
        if role not in NEW_ROLES:raise ValueError(role)
        n,d=reference.shape
        if n%groups or n//groups<=d:raise ValueError('template needs more points than dimensions')
        self.role=role;self.groups=groups;self.dimension=d
        base=reference.detach().cpu().double().numpy().reshape(groups,-1,d)
        sigma=np.sqrt(np.mean(np.sum(base**2,-1),-1)/d)
        eps=np.random.default_rng(20261102).normal(size=base.shape);eps-=eps.mean(1,keepdims=True)
        for g in range(groups):
            val,vec=np.linalg.eigh(eps[g].T@eps[g]/len(eps[g]))
            eps[g]=eps[g]@((vec/np.sqrt(val)[None,:])@vec.T)
        self.register_buffer('template',torch.as_tensor(eps,dtype=reference.dtype))
        self.register_buffer('sigma',torch.as_tensor(sigma,dtype=reference.dtype))
        if role.startswith('gaussian_') or role=='group_affine':
            self.mean_offset=nn.Parameter(torch.zeros(groups,d,dtype=reference.dtype))
        if role.startswith('gaussian_'):
            self.log_scale=nn.Parameter(torch.zeros(groups,1 if role=='gaussian_iso' else d,dtype=reference.dtype))
            if role=='gaussian_full':self.lower=nn.Parameter(torch.zeros(groups,d*(d-1)//2,dtype=reference.dtype))
        elif role=='group_affine':self.matrix_offset=nn.Parameter(torch.zeros(groups,d,d,dtype=reference.dtype))
        elif role=='shared_affine':
            self.matrix_offset=nn.Parameter(torch.zeros(d,d,dtype=reference.dtype))
            self.mean_offset=nn.Parameter(torch.zeros(d,dtype=reference.dtype))
        elif role=='free_frequency':self.offset=nn.Parameter(torch.zeros_like(self.template))

    def forward(self):
        e=self.template;s=self.sigma;d=self.dimension;eye=torch.eye(d,dtype=e.dtype,device=e.device)
        base=e*s[:,None,None]
        if self.role=='fixed_gaussian':return base.flatten(0,1)
        if self.role=='free_frequency':return ((e+self.offset)*s[:,None,None]).flatten(0,1)
        if self.role=='shared_affine':
            return (base@(eye+self.matrix_offset).T+self.mean_offset*s.square().mean().sqrt()).flatten(0,1)
        if self.role=='group_affine':L=eye+self.matrix_offset
        else:
            L=torch.diag_embed(self.log_scale.exp().expand(-1,d))
            if self.role=='gaussian_full':
                i,j=torch.tril_indices(d,d,-1,device=e.device);extra=torch.zeros_like(L)
                extra[:,i,j]=self.lower;L=L+extra
        return ((e@L.transpose(-1,-2)+self.mean_offset[:,None,:])*s[:,None,None]).flatten(0,1)


class FrequencyMatcher(Round5CrossAttentionMatcher):
    def __init__(self,role):
        super().__init__('point')
        self.method=role;self.frequency_bank=FrequencyBank(role,self.frequency)

    def _position_encode(self,content,mean,covariance):
        # covariance intentionally unused: learned frequency geometry is not
        # a statement about physical token support or positional uncertainty.
        return rotary(content.double(),mean.unsqueeze(1).double(),
                      self.frequency_bank().double()).to(content.dtype)


def make_model(role):
    return FrequencyMatcher(role) if role in NEW_ROLES else Round5CrossAttentionMatcher(role)
