"""Three-dimensional Gaussian coordinate maps inside the matched Transformer."""
import math
import torch
from torch import nn
import torch.nn.functional as F
from models.geometry import rotary
from models.round5_cross_attention import Round5CrossAttentionMatcher
from models.round45_phase_warp import make_model as make_2d_warp

ROLES=('gaussian3d','iso3d','fixed3d','fixed_centers3d','fourier3d','mlp3d',
       'affine3d','free3d','point3d','center2d','gaussian2d')
TWO_D=('center2d','gaussian2d')
GAUSSIAN=('gaussian3d','iso3d','fixed3d','fixed_centers3d')


def sphere_frequencies(n):
    i=torch.arange(n,dtype=torch.float64)
    z=1-2*(i+.5)/n
    angle=i*math.pi*(3-math.sqrt(5))
    radius=(1-z.square()).sqrt()
    direction=torch.stack((radius*angle.cos(),radius*angle.sin(),z),-1)
    return (direction*torch.logspace(-1,.6,n,dtype=torch.float64)[:,None]).float()


class SpatialMap3D(nn.Module):
    def __init__(self,role):
        super().__init__();self.role=role
        self.projection=nn.Parameter(torch.zeros(64,3))
        if role in GAUSSIAN:
            axis=(torch.arange(4)+.5)/2-1
            grid=torch.meshgrid(axis,axis,axis,indexing='ij')
            self.register_buffer('centers',torch.stack([a.flatten() for a in grid],-1))
            if role in ('gaussian3d','iso3d'):self.mean_offset=nn.Parameter(torch.zeros(64,3))
            if role!='fixed3d':
                self.log_precision=nn.Parameter(torch.zeros(64,1 if role=='iso3d' else 3))
                if role!='iso3d':self.lower=nn.Parameter(torch.zeros(64,3))
        elif role=='fourier3d':
            self.register_buffer('base_frequency',sphere_frequencies(32)*12)
            self.frequency_offset=nn.Parameter(torch.zeros(32,3))
            self.phase_offset=nn.Parameter(torch.zeros(32))
        elif role=='mlp3d':
            self.register_buffer('base_mlp',torch.randn(3,64)/math.sqrt(3))
            self.mlp_offset=nn.Parameter(torch.zeros(3,64))
            self.bias=nn.Parameter(torch.zeros(64))
        else:raise ValueError(role)
        # Uniform, label-free reference cube. No scene or validation statistics.
        axis=(torch.arange(8,dtype=torch.float64)+.5)/4-1
        grid=torch.stack([a.flatten() for a in torch.meshgrid(axis,axis,axis,indexing='ij')],-1)
        with torch.no_grad():rms=self.basis(grid).square().sum(-1).mean().sqrt()
        self.register_buffer('initial_rms',rms)

    def basis(self,z):
        z=z.double()
        if self.role in GAUSSIAN:
            centers=self.centers.double()+getattr(self,'mean_offset',0)
            diag=self.log_precision.double().exp().expand(-1,3) if hasattr(self,'log_precision') else torch.ones_like(centers)
            L=torch.diag_embed(diag)
            if hasattr(self,'lower'):
                ij=torch.tril_indices(3,3,offset=-1,device=L.device)
                extra=torch.zeros_like(L);extra[:,ij[0],ij[1]]=self.lower.double()
                L=L+extra
            delta=z.unsqueeze(-2)-centers
            transformed=torch.einsum('...gi,gij->...gj',delta,2*L)
            return (-.5*transformed.square().sum(-1)).exp()
        if self.role=='fourier3d':
            phase=z@(self.base_frequency.double()+12*self.frequency_offset.double()).T+self.phase_offset.double()
            return torch.cat((phase.cos(),phase.sin()),-1)
        return F.gelu(z@(self.base_mlp.double()+self.mlp_offset.double())+self.bias.double())

    def forward(self,mean):
        return 12*(self.basis(mean.double()/12)/self.initial_rms)@self.projection.double()


class Native3DMatcher(Round5CrossAttentionMatcher):
    def __init__(self,role):
        if role not in ROLES or role=='gaussian2d':raise ValueError(role)
        super().__init__('point');self.method=role
        if role!='center2d':self.frequency=sphere_frequencies(24)
        self.position_field=nn.Module()
        if role in GAUSSIAN or role in ('fourier3d','mlp3d'):
            self.position_field=SpatialMap3D(role)
        elif role=='affine3d':self.position_field.affine_offset=nn.Parameter(torch.zeros(3,3))
        elif role=='free3d':
            self.position_field.frequency_offset=nn.Parameter(torch.zeros(24,3))
            self.position_field.register_buffer('frequency_scale',self.frequency.double().norm(dim=-1,keepdim=True))

    def position_arguments(self,mean):
        x=mean.double();w=self.frequency.double();field=self.position_field
        if isinstance(field,SpatialMap3D):x=x+field(mean)
        elif self.method=='affine3d':x=x+x@field.affine_offset.double().T
        elif self.method=='free3d':w=w+field.frequency_scale*field.frequency_offset.double()
        return x,w

    def _position_encode(self,content,mean,covariance):
        x,w=self.position_arguments(mean)
        return rotary(content.double(),x.unsqueeze(1),w).to(content.dtype)


def make_model(role):
    return make_2d_warp('warp_full') if role=='gaussian2d' else Native3DMatcher(role)
