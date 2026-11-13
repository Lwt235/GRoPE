"""Round009 single attention reader; no pose-regression head or query truth API."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from models.region_hrope_control import homographies


def transport(x,m,xy,inverse=False):
    geom=x[...,:24].reshape(*x.shape[:-1],8,3)
    geom=torch.einsum('bnij,bhnkj->bhnki',m,geom).flatten(-2)
    freq=10000.**(-2*torch.arange(1,7,device=x.device,dtype=x.dtype)/24.)
    phase=(8*xy[...,None]*freq).flatten(-2)[:,None]
    if inverse:phase=-phase
    a,b=x[...,24:].reshape(*x.shape[:-1],12,2).unbind(-1)
    return torch.cat((geom,torch.stack((a*phase.cos()-b*phase.sin(),a*phase.sin()+b*phase.cos()),-1).flatten(-2)),-1)

class NativePoseMemoryReader(nn.Module):
    def __init__(self):
        super().__init__()
        self.source=nn.Sequential(nn.Linear(1044,192),nn.GELU(),nn.LayerNorm(192))
        self.query=nn.Sequential(nn.Linear(2069,192),nn.GELU(),nn.LayerNorm(192))
        self.key=nn.Linear(192,192);self.value=nn.Linear(192,192);self.q=nn.Linear(192,192)
        self.position=nn.Sequential(nn.Linear(29,64),nn.GELU(),nn.Linear(64,192,bias=False))
        self.homography=nn.Linear(192,13)
        self.output=nn.Linear(192,192)
        self.residual=nn.Sequential(nn.Linear(384,192),nn.GELU(),nn.Linear(192,2048))
        for layer in (self.position[-1],self.homography,self.residual[-1]):
            nn.init.zeros_(layer.weight)
            if layer.bias is not None:nn.init.zeros_(layer.bias)
        with torch.no_grad():
            self.homography.bias[:6].copy_(torch.tensor([1.,0,0,0,1.,0]));self.homography.bias[11]=1.;self.homography.bias[12]=.25
    def encode_source(self,ordinary):
        h=self.source(ordinary);return dict(key=self.key(h),value=self.value(h),raw=self.homography(h))
    def forward(self,qordinary,anchor,alpha,ordinary,view,K,xy,kind,encoded=None):
        qh=self.query(torch.cat((qordinary,F.layer_norm(anchor,(1024,)),torch.log1p(alpha)),-1))
        enc=self.encode_source(ordinary) if encoded is None else encoded
        k=enc['key']+self.position(torch.cat((view['common25'],view['P' if kind=='L' else kind]),-1))
        qmat,qinv=homographies(self.homography(qh),K[:,None]);kmat,kinv=homographies(enc['raw'],K[:,None])
        kxy=view['common25'][...,3:5]/2
        heads=lambda z:z.reshape(z.shape[0],-1,4,48).transpose(1,2)
        q=transport(heads(self.q(qh)),qmat.transpose(-1,-2),xy,True)
        k=transport(heads(k),kinv,kxy,True);v=transport(heads(enc['value']),kinv,kxy,True)
        k=F.pad(k,(0,0,0,1));v=F.pad(v,(0,0,0,1));mask=F.pad(view['key_mask'],(0,1),value=True)
        a=F.scaled_dot_product_attention(q,k,v,attn_mask=mask[:,None,None,:],dropout_p=0.)
        a=transport(a,qmat,xy,False).transpose(1,2).flatten(-2)
        return self.residual(torch.cat((qh,self.output(a)),-1).mean(1))
