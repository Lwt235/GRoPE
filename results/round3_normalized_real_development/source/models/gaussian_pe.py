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
    """Fixed stratified Sobol mixture quadrature, independent of labels.

    Each Gaussian component receives a complete Sobol net. Assigning mixture
    components by Sobol index parity would correlate component with location.
    """
    if count < 4 or count & (count - 1):
        raise ValueError('count must be a power of two >= 4')
    u=torch.quasirandom.SobolEngine(dimension, scramble=True, seed=20261029).draw(count//2,dtype=dtype)
    z=math.sqrt(2)*torch.erfinv((2*u-1).clamp(-1+1e-7,1-1e-7))
    scales=torch.tensor([1.,4.],dtype=dtype)
    omega=torch.cat([z*scale for scale in scales],dim=0)
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
    amplitude=torch.exp(logamp-.5*torch.logsumexp(2*logamp,-1,keepdim=True))
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
REGION_METHODS=METHODS+['density_isotropic','density_area','density_shuffled','absolute_geometry',
                       'isometric_fixed','isometric_shape','isometric_isotropic',
                       'isometric_area','isometric_shuffled','normalized_fixed',
                       'normalized_shape','normalized_trace','normalized_area','normalized_shuffled']


def normalized_rotary(features,mean,covariance,frequency):
    """Gaussian frequency filtering followed by per-token Q/K normalization.

    The common log-amplitude offset cancels in normalization and prevents all
    channels underflowing for very large supports. This is nonlinear in content,
    unlike an orthogonal PE. It does not equal the unnormalized expectation.
    """
    variance=torch.einsum('fd,...de,fe->...f',frequency,covariance,frequency)
    logamp=-variance/2
    amplitude=(logamp-logamp.amax(-1,keepdim=True)).exp()
    phase=mean@frequency.T
    x,y=features.reshape(*features.shape[:-1],-1,2).unbind(-1)
    transformed=torch.stack([(x*phase.cos()-y*phase.sin())*amplitude,
                             (x*phase.sin()+y*phase.cos())*amplitude],-1).flatten(-2)
    return F.normalize(transformed,dim=-1)*features.norm(dim=-1,keepdim=True)


class GaussianPEMatcher(nn.Module):
    def __init__(self,method,count=256,resolution=0.):
        super().__init__()
        if method not in REGION_METHODS: raise ValueError(method)
        self.method=method
        self.encoder=nn.Sequential(nn.Linear(108,96),nn.GELU(),nn.Linear(96,48))
        self.logtemp=nn.Parameter(torch.tensor(2.2))
        self.dustbin=nn.Parameter(torch.tensor(2.))
        self.dustbin_head=nn.Linear(48,1)
        # A deliberately stronger learned same-information control. These
        # parameters exist only for that control, not for the proposed PE.
        if method=='absolute_geometry':
            self.geometry_encoder=nn.Sequential(nn.Linear(5,48),nn.GELU(),nn.Linear(48,48))
        self.register_buffer('resolution',torch.tensor(float(resolution)))
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
        identity=torch.eye(2,device=mq.device,dtype=mq.dtype)
        cq=data.get('projected_qcov',identity.expand(*mq.shape[:-1],2,2)/12)
        ck=identity.expand(*data['k_to_q'].shape[:-1],2,2)/12 if self.method.endswith('fixed') else data['projected_kcov']
        if self.method.endswith('_isotropic') or self.method.endswith('_trace'):
            ck=identity*(ck.diagonal(dim1=-2,dim2=-1).sum(-1)/2)[...,None,None]
        if self.method.endswith('_area'):
            ck=identity*torch.linalg.det(ck).clamp_min(1e-12).sqrt()[...,None,None]
        if self.method.endswith('_shuffled'):
            # Fixed cyclic derangement, never labels or correspondences.
            ck=ck.roll(13,dims=-3)
        cq=cq+identity*self.resolution.square()
        ck=ck+identity*self.resolution.square()
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
            if name=='absolute_geometry':
                def descriptor(mean,cov):
                    return torch.cat([mean/4,cov[...,0,0,None],cov[...,0,1,None],cov[...,1,1,None]],-1)
                q=F.normalize(q+self.geometry_encoder(descriptor(mq,cq)),dim=-1)
                k=F.normalize(k+self.geometry_encoder(descriptor(mk,ck)),dim=-1)
                score=q@k.transpose(-1,-2)
            elif name.startswith('density'):
                pq=density_features(mq,cq,self.omega,self.log_proposal)
                pk=density_features(mk,ck,self.omega,self.log_proposal)
                # Exactly equals lift(q,pq) @ lift(k,pk).T without allocating
                # the D*2M head. Cost O(N^2(D+2M)), not a claim of free features.
                score=(q@k.transpose(-1,-2))*(pq@pk.transpose(-1,-2))
            elif name.startswith('normalized'):
                q=normalized_rotary(q,mq,cq,self.freq2)
                k=normalized_rotary(k,mk,ck,self.freq2)
                score=q@k.transpose(-1,-2)
            elif name.startswith('isometric'):
                def encode(content,mean,cov):
                    var=torch.einsum('fd,...de,fe->...f',self.freq2,cov,self.freq2)
                    rotated=rotary(content,mean,self.freq2,cov)
                    complement=(-torch.expm1(-var)).clamp_min(0).sqrt()
                    residual=(content.reshape(*content.shape[:-1],-1,2)*complement[...,None]).flatten(-2)
                    return torch.cat([rotated,residual],-1)
                q=encode(q,mq,cq)
                k=encode(k,mk,ck)
                score=q@k.transpose(-1,-2)
            else:
                use_cov=name.startswith('expected')
                q=rotary(q,mq,self.freq2,cq if use_cov else None)
                k=rotary(k,mk,self.freq2,ck if use_cov else None)
                score=q@k.transpose(-1,-2)
        score=self.logtemp.exp().clamp(max=60)*score
        unmatched=self.dustbin+self.dustbin_head(original_q).squeeze(-1)
        return torch.cat([score,unmatched[...,None]],-1)
