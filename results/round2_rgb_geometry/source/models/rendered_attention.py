"""Attention-logit ablations with a shared RGB patch encoder.

HRoPE reproduces Eq.(1),(4),(17) at the score level with calibrated, supplied
planes. WorldRoPE reproduces Eq.(3)-(10) with a fixed analytic depth mixture.
These are operator checks, not reproductions of learned NVS architectures.
"""
import torch
from torch import nn
import torch.nn.functional as F
from models.geometry import expected_phase,rotary

METHODS=['appearance','position_rbf','rope3d','rope_normal','rope_normal_anisotropy',
         'rope_covariance','rope_normal_uncertainty','hrope_calibrated_qk',
         'worldrope_mixture_qk','rope_overlap','rope_overlap_constant']
GATE_METHODS=['rope_normal_gated','rope_normal_gate_random','position_gate_only']


def homography_features(features,homography,query):
    chunks=features.reshape(*features.shape[:-1],-1,3)
    operator=homography.transpose(-1,-2) if query else torch.linalg.inv(homography)
    return torch.einsum('...ij,...cj->...ci',operator,chunks).flatten(-2)


def mixture_rotary(features,means,covariances,frequencies):
    real,imag=expected_phase(means,covariances,frequencies)
    weights=torch.tensor([1/6,2/3,1/6],dtype=real.dtype,device=real.device)
    real=(real*weights[None,None,:,None]).sum(-2)
    imag=(imag*weights[None,None,:,None]).sum(-2)
    a,b=features.reshape(*features.shape[:-1],-1,2).unbind(-1)
    return torch.stack([a*real-b*imag,a*imag+b*real],-1).flatten(-2)


class RenderedMatcher(nn.Module):
    def __init__(self,method):
        super().__init__()
        self.method=method
        self.encoder=nn.Sequential(nn.Linear(108,96),nn.GELU(),nn.Linear(96,48))
        self.logtemp=nn.Parameter(torch.tensor(2.2))
        self.weights=nn.Parameter(torch.full((4,),-.3))
        self.dustbin=nn.Parameter(torch.tensor(2.))
        self.dustbin_head=nn.Linear(48,1)
        # Same registered capacity for all ablations; gate is only active in
        # explicitly named variants, with a random-bias and position-only control.
        self.gate=nn.Sequential(nn.Linear(4,12),nn.GELU(),nn.Linear(12,1))
        norms=torch.logspace(-.7,.8,24)
        self.register_buffer('freq3',torch.eye(3)[torch.arange(24)%3]*norms[:,None])
        self.register_buffer('freq2',torch.eye(2)[torch.arange(24)%2]*torch.logspace(-1,.6,24)[:,None])
        self.register_buffer('freq_half',torch.eye(2)[torch.arange(12)%2]*torch.logspace(-1,.6,12)[:,None])

    def forward(self,data,return_gate=False):
        q=F.normalize(self.encoder(data['q']),dim=-1)
        k=F.normalize(self.encoder(data['k']),dim=-1)
        original_q=q
        name=self.method
        if name.startswith('rope') or name=='position_gate_only':
            cq=ck=None
            if name=='rope_normal_uncertainty': cq,ck=data['uq'],data['uk']
            q=rotary(q,data['mq'],self.freq3,cq)
            k=rotary(k,data['mk'],self.freq3,ck)
        elif name=='hrope_calibrated_qk':
            q=torch.cat([homography_features(q[...,:24],data['hq'],True),rotary(q[...,24:],data['pixq'],self.freq_half)],-1)
            k=torch.cat([homography_features(k[...,:24],data['hk'],False),rotary(k[...,24:],data['pixk'],self.freq_half)],-1)
        elif name=='worldrope_mixture_qk':
            q=rotary(q,data['pixq'],self.freq2)
            k=mixture_rotary(k,data['world_means'],data['world_covs'],self.freq2)
        score=self.logtemp.exp().clamp(max=60)*(q@k.transpose(-1,-2))
        weight=F.softplus(self.weights)
        gate=None
        if name=='position_rbf': score=score-weight[0]*data['center']
        if name in ['rope_normal','rope_normal_anisotropy','rope_normal_uncertainty']:
            score=score-weight[0]*data['normal']
        if name=='rope_normal_anisotropy': score=score-weight[1]*data['anisotropy'].clamp(max=10)
        if name=='rope_covariance': score=score-weight[0]*data['shape']
        if name=='rope_overlap': score=score+weight[0]*data['overlap']
        if name=='rope_overlap_constant': score=score+weight[0]*data['overlap_constant']
        if name in GATE_METHODS:
            gate=self.gate(data['gate_inputs']).squeeze(-1).sigmoid()
            if name=='position_gate_only': score=score-weight[0]*(1-gate)
            else:
                distance=data['normal'] if name=='rope_normal_gated' else data['random_normal']
                score=score-weight[0]*gate*distance
        unmatched=self.dustbin+self.dustbin_head(original_q).squeeze(-1)
        logits=torch.cat([score,unmatched[...,None]],-1)
        return (logits,gate) if return_gate else logits
