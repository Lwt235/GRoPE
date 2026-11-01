"""Actual-GS direct RGB attention. No learned value or residual color path."""
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from models.round40_decoupled_pe import rotary,shape_descriptor,squared_distance,NEW_ROLES,SHAPE_ROLES

ROLES=NEW_ROLES+('point','fixed','full','isotropic')


class GSReadout(nn.Module):
    def __init__(self,role,n_primitives=12000,n_pixels=27648):
        super().__init__()
        if role not in ROLES:raise ValueError(role)
        self.role=role;self.qk=nn.Linear(128,128,bias=False)
        with torch.no_grad():self.qk.weight.copy_(torch.eye(128))
        axes=torch.eye(3)[torch.arange(64)%3]
        self.register_buffer('frequencies',axes*torch.logspace(0.,1.3,64)[:,None])
        if role in SHAPE_ROLES or role=='center_rbf':
            self.strength_raw=nn.Parameter(torch.tensor(math.log(math.expm1(1.))))
        rng=np.random.default_rng(20261101)
        self.register_buffer('q_perm',torch.from_numpy(rng.permutation(n_primitives)))
        self.register_buffer('k_perm',torch.from_numpy(rng.permutation(n_pixels)))

    def encode(self,features,anchor,cov):
        z=F.normalize(self.qk(features),dim=-1)
        if self.role not in NEW_ROLES:return z
        use=self.role in ('expected_rope','normalized_rope')
        z=rotary(z,anchor,cov if use else None,self.frequencies)
        return F.normalize(z,dim=-1) if self.role=='normalized_rope' else z

    def logits(self,q,k,key_encoded=None,zero_shape=False):
        qf=self.encode(q['feature'],q['anchor'],q['cov'])
        kf=self.encode(k['feature'],k['anchor'],k['cov']) if key_encoded is None else key_encoded
        score=10.*(qf@kf.T)
        if self.role in SHAPE_ROLES:
            qc,kc=q['cov'],k['cov']
            if self.role=='shape_shuffle':qc,kc=q['shuffle_cov'],k['cov'][self.k_perm]
            if self.role=='shape_isotropic':
                eye=torch.eye(3,dtype=qc.dtype,device=qc.device)
                qc=qc.diagonal(dim1=-2,dim2=-1).mean(-1)[:,None,None]*eye
                kc=kc.diagonal(dim1=-2,dim2=-1).mean(-1)[:,None,None]*eye
            penalty=squared_distance(shape_descriptor(qc),shape_descriptor(kc))/7.
            return score-(0. if zero_shape else F.softplus(self.strength_raw))*penalty
        if self.role=='center_rbf':
            return score-F.softplus(self.strength_raw)*squared_distance(q['anchor'],k['anchor'])
        if self.role in NEW_ROLES:return score
        cov=q['projected_cov'];eye=torch.eye(2,dtype=cov.dtype,device=cov.device)
        if self.role=='point':cov=torch.zeros_like(cov)
        elif self.role=='fixed':cov=eye.expand_as(cov)
        elif self.role=='isotropic':cov=cov.diagonal(dim1=-2,dim2=-1).mean(-1)[:,None,None]*eye
        precision=torch.linalg.inv(cov+(1/12+.5**2)*eye)
        dx=q['xy'][:,0,None]-k['xy'][None,:,0]
        dy=q['xy'][:,1,None]-k['xy'][None,:,1]
        return score-.5*(precision[:,0,0,None]*dx.square()+2*precision[:,0,1,None]*dx*dy+precision[:,1,1,None]*dy.square())

    def forward(self,q,k):
        return self.logits(q,k).softmax(-1)@k['rgb']
