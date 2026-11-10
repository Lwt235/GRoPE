"""Five scalar relation coefficients in cross attention; no truth-reference input."""
import math
import torch
from torch import nn
from torch.nn import functional as F

from models.round58_support_pe import DIM, HEADS, HD
from models.round65_actual_gs import shape_features
from models.round69_compact_relation import descriptors
from models.region_fixed_hrope import FixedHRegionMatcher
from models.region_hrope_control import homographies, hrope_layer, transport

ROLES=('cross_full5','cross_distance','cross_sphere5')


def relation_energy(source,target,probability):
    """Expected relation difference, excluding each query's own reference row.

    source [B,N,N,C], target [B,M,M,C], probability [B,N,M].
    No validity labels, extrinsics, selection thresholds, or detached references.
    """
    dtype=probability.dtype
    a=source.double().permute(0,3,1,2)
    b=target.double().permute(0,3,1,2)
    p=probability.double()[:,None]
    mass=p[:,0].sum(-1)
    denominator=mass.sum(-1,keepdim=True)-mass
    diagonal=a.diagonal(dim1=-2,dim2=-1)
    a2=(a.square()*mass[:,None,None,:]).sum(-1)-diagonal.square()*mass[:,None,:]
    b2=(b.square()*p[:,0].sum(-2)[:,None,None,:]).sum(-1)
    pb=p@b.transpose(-1,-2)
    cross=(a@p)@b.transpose(-1,-2)-diagonal[...,None]*pb
    numerator=a2[...,None]+b2[:,:,None,:]-p@b.square().transpose(-1,-2)-2*cross
    return (numerator/denominator[:,None,:,None].clamp_min(1e-12)).clamp_min(0).permute(0,2,3,1).to(dtype)


class ConsistencyPosition(nn.Module):
    def __init__(self,role,scale=None):
        super().__init__()
        if role not in ROLES:raise ValueError(role)
        self.role=role
        width=1 if role=='cross_distance' else 5
        self.coefficients=nn.Parameter(torch.zeros(width))
        self.register_buffer('scale',torch.ones(width) if scale is None else torch.as_tensor(scale,dtype=torch.float32))
        assert self.scale.shape==(width,) and bool((self.scale>0).all())

    def forward(self,centers,covariance):
        if self.role=='cross_full5':
            phi=descriptors('relative5',centers,covariance)
        else:
            delta=centers.unsqueeze(-2)-centers.unsqueeze(-3)
            distance=delta.square().sum(-1).sqrt()/.2
            if self.role=='cross_distance':phi=distance[...,None]
            else:
                radial=distance.square().log1p();constant=torch.full_like(distance,math.log(4.))
                phi=torch.stack((distance,radial,radial,constant,constant),-1)
        return phi/self.scale.to(phi)


def consistency_cross_layer(layer,x,matrix,inverse,image_positions,relation,coefficients):
    batch,views,tokens,_=x.shape
    q,k,v=layer.qkv(layer.norm(x)).reshape(batch,views,tokens,3,HEADS,HD).permute(3,0,1,4,2,5).unbind(0)
    q=transport(q,matrix.transpose(-1,-2),image_positions,True)
    k=transport(k,inverse,image_positions,True).flip(1)
    v=transport(v,inverse,image_positions,True).flip(1)
    content=(q@k.transpose(-1,-2))/math.sqrt(HD)
    reference=content.softmax(-1)
    width=relation.shape[-1]
    source=relation.unsqueeze(2).expand(batch,views,HEADS,tokens,tokens,width)
    target=relation.flip(1).unsqueeze(2).expand_as(source)
    e=relation_energy(source.reshape(-1,tokens,tokens,width),target.reshape(-1,tokens,tokens,width),
        reference.reshape(-1,tokens,tokens)).reshape(batch,views,HEADS,tokens,tokens,width)
    probability=(content-(e*coefficients).sum(-1)).softmax(-1)
    aggregate=transport(probability@v,matrix,image_positions,False)
    aggregate=aggregate.transpose(-3,-2).reshape(batch,views,tokens,DIM)
    x=x+layer.output(aggregate)
    return x+layer.ffn(layer.ffn_norm(x))


class CrossConsistencyMatcher(FixedHRegionMatcher):
    def __init__(self,role,scale=None):
        super().__init__('fixed_hrope')
        self.fixed_transform=self.position
        self.position=ConsistencyPosition(role,scale)
        self.role=role

    def forward(self,features,centers,covariance,permutation,intrinsics,image_positions):
        camera=intrinsics[...,:2,:].flatten(-2).unsqueeze(-2).expand(*features.shape[:-1],6)
        content=torch.cat((features,centers,shape_features(covariance).to(features),
            camera.to(features),image_positions.to(features)),-1)
        x=self.stem(content)
        matrix,inverse=homographies(self.fixed_transform(x),intrinsics.unsqueeze(-3))
        relation=self.position(centers,covariance)
        for index,layer in enumerate(self.layers):
            if index%2:
                x=consistency_cross_layer(layer,x,matrix,inverse,image_positions,relation,self.position.coefficients)
            else:x=hrope_layer(layer,x,matrix,inverse,image_positions,False)
        descriptor=F.normalize(self.head(self.final_norm(x)),dim=-1)
        temperature=self.log_temperature.clamp(max=math.log(60.)).exp()
        return temperature*(descriptor[:,0]@descriptor[:,1].transpose(-1,-2))
