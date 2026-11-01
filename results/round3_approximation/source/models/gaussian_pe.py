"""Gaussian support as a normalized, per-token Q/K positional feature map.

The continuous map is sqrt(N(x; mean, covariance)). Its inner product is the
classical Bhattacharyya probability product kernel (Jebara et al., JMLR 2004).
This file implements its Fourier quadrature, not a new Gaussian kernel.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F
from models.geometry import rotary, expected_phase


def spectral_rule(count=256, dimension=2, dtype=torch.float32):
    """Fixed Sobol importance quadrature, independent of data and labels."""
    if count < 4 or count & (count - 1):
        raise ValueError('count must be a power of two >= 4')
    u=torch.quasirandom.SobolEngine(dimension, scramble=True, seed=20261029).draw(count).to(dtype)
    z=math.sqrt(2)*torch.erfinv((2*u-1).clamp(-1+1e-7,1-1e-7))
    scales=torch.tensor([1.,4.],dtype=dtype)
    omega=z*scales[torch.arange(count)%2,None]
    logp=-.5*(omega[:,None,:]/scales[None,:,None]).square().sum(-1)
    logp=logp-dimension*scales.log()[None,:]-.5*dimension*math.log(2*math.pi)
    logp=torch.logsumexp(logp,-1)-math.log(2)
    return omega,logp


def density_features(mean,covariance,omega,log_proposal):
    """Real Fourier coefficients of square-root density, normalized in L2.

    Fourier(sqrt(N))(w) is proportional to |S|^.25 exp(-w^T S w)
    exp(i w^T mean). The determinant cancels in finite-map normalization.
    The returned feature has unit norm even when the quadrature is inaccurate.
    """
    variance=torch.einsum('fd,...de,fe->...f',omega,covariance,omega)
    logamp=-variance-.5*log_proposal
    amplitude=torch.softmax(2*logamp,-1).sqrt()
    phase=mean@omega.T
    return torch.stack([amplitude*phase.cos(),amplitude*phase.sin()],-1).flatten(-2)


def lift(content,position):
    return (content[..., :,None]*position[...,None,:]).flatten(-2)


def bhattacharyya(mean_a,cov_a,mean_b,cov_b):
    """Exact continuous reference, used for approximation audit only."""
    avg=(cov_a[..., :,None,:,:]+cov_b[...,None,:,:,:])/2
    delta=mean_a[..., :,None,:]-mean_b[...,None,:,:]
    mahal=(delta[...,None,:]@torch.linalg.solve(avg,delta[...,None]))[...,0,0]/8
    ld=torch.linalg.slogdet(avg)[1]/2
    ld=ld-(torch.linalg.slogdet(cov_a)[1][..., :,None]+torch.linalg.slogdet(cov_b)[1][...,None,:])/4
    return torch.exp(-mahal-ld)


METHODS=['rope3d','projected_rope','expected_fixed','expected_shape','density_fixed','density_shape']


class GaussianPEMatcher(nn.Module):
    def __init__(self,method,count=256):
        super().__init__()
        if method not in METHODS: raise ValueError(method)
        self.method=method
        self.encoder=nn.Sequential(nn.Linear(108,96),nn.GELU(),nn.Linear(96,48))
        self.logtemp=nn.Parameter(torch.tensor(2.2))
        self.dustbin=nn.Parameter(torch.tensor(2.))
        self.dustbin_head=nn.Linear(48,1)
        norms=torch.logspace(-.7,.8,24)
        self.register_buffer('freq3',torch.eye(3)[torch.arange(24)%3]*norms[:,None])
        angle=torch.arange(24)*math.pi*(3-math.sqrt(5))
        freq=torch.stack([angle.cos(),angle.sin()],-1)*torch.logspace(-1,.6,24)[:,None]
        self.register_buffer('freq2',freq)
        omega,lp=spectral_rule(count)
        self.register_buffer('omega',omega)
        self.register_buffer('log_proposal',lp)

    def positional_inputs(self,data):
        mq,mk=data['pixq'],data['k_to_q']
        cq=torch.eye(2,device=mq.device,dtype=mq.dtype).expand(*mq.shape[:-1],2,2)/12
        ck=cq if self.method.endswith('fixed') else data['projected_kcov']
        return mq,cq,mk,ck

    def forward(self,data):
        q=F.normalize(self.encoder(data['q']),dim=-1)
        k=F.normalize(self.encoder(data['k']),dim=-1)
        original_q=q
        name=self.method
        if name=='rope3d':
            q=rotary(q,data['mq'],self.freq3)
            k=rotary(k,data['mk'],self.freq3)
            score=q@k.transpose(-1,-2)
        else:
            mq,cq,mk,ck=self.positional_inputs(data)
            if name.startswith('density'):
                pq=density_features(mq,cq,self.omega,self.log_proposal)
                pk=density_features(mk,ck,self.omega,self.log_proposal)
                # Exactly equals lift(q,pq) @ lift(k,pk).T without allocating
                # the D*2M head. Cost O(N^2(D+2M)), not a claim of free features.
                score=(q@k.transpose(-1,-2))*(pq@pk.transpose(-1,-2))
            else:
                use_cov=name.startswith('expected')
                q=rotary(q,mq,self.freq2,cq if use_cov else None)
                k=rotary(k,mk,self.freq2,ck if use_cov else None)
                score=q@k.transpose(-1,-2)
        score=self.logtemp.exp().clamp(max=60)*score
        unmatched=self.dustbin+self.dustbin_head(original_q).squeeze(-1)
        return torch.cat([score,unmatched[...,None]],-1)
