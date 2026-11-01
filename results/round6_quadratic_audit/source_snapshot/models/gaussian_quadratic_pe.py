"""Exact five-channel Gaussian positional features for uniform target cells.

This is the established quadratic lift of a Gaussian attention energy; here
its query precision comes from projected region support. It is not a new
kernel, nor a finite exact map for arbitrary unequal key covariances.
"""
import math

import torch
from torch import nn


def quadratic_features(qmean, qcov, kmean, common_kcov, bandwidth):
    """Return Qg[...,Nq,5], Kg[...,Nk,5]; their dot is log overlap up to row constants.

    common_kcov is [...,2,2], identical for all keys in an attention set.
    Center both coordinate systems at the mean key position for conditioning.
    Bandwidth and positions use the SAME reference units.
    """
    if common_kcov.shape != qcov.shape[:-3]+(2,2):
        raise ValueError('target covariance must be common across keys')
    dtype=torch.float64 if qmean.dtype==torch.float64 else torch.float32
    qm,qc,km,kc=(x.to(dtype) for x in (qmean,qcov,kmean,common_kcov))
    width=torch.as_tensor(bandwidth,dtype=dtype,device=qm.device)
    eye=torch.eye(2,device=qm.device,dtype=dtype)
    precision=torch.linalg.inv(qc+kc[...,None,:,:]+width.square()*eye)
    origin=km.mean(-2,keepdim=True)
    q,k=qm-origin,km-origin
    linear=(precision@q[...,None]).squeeze(-1)
    qg=torch.cat([linear,(-.5*precision[...,0,0])[...,None],
                  (-precision[...,0,1])[...,None],(-.5*precision[...,1,1])[...,None]],-1)
    x,y=k.unbind(-1)
    kg=torch.stack([x,y,x.square(),x*y,y.square()],-1)
    return qg.to(qmean.dtype),kg.to(qmean.dtype)


def controlled_features(data,method,bandwidth,reverse=False):
    source,target=('k','q') if reverse else ('q','k')
    cov=data[source+'_to_'+target+'cov'];target_cov=data[target+'cov']
    if method=='point':
        cov=torch.zeros_like(cov);target_cov=torch.zeros_like(target_cov)
    elif method=='fixed':
        cov=data[source+'cov']
    elif method=='isotropic':
        eye=torch.eye(2,device=cov.device,dtype=cov.dtype)
        cov=cov.diagonal(dim1=-2,dim2=-1).mean(-1)[...,None,None]*eye
        target_cov=target_cov.diagonal(dim1=-2,dim2=-1).mean(-1)[...,None,None]*eye
    elif method!='full':
        raise ValueError(method)
    # Explicit contract; unequal adaptive key footprints require the full kernel.
    if not torch.allclose(target_cov,target_cov[...,:1,:,:].expand_as(target_cov),atol=0,rtol=0):
        raise ValueError('five-channel operator requires uniform target covariance')
    return quadratic_features(data[source+'_to_'+target],cov,data[target+'mean'],
        target_cov[...,0,:,:],bandwidth)


class GaussianQuadraticMatcher(nn.Module):
    """Frozen visual features plus 5 positional channels, zero learned network.

    h and T are two global development-calibrated scalars. They are registered
    as buffers; repeated inference is deterministic, so seed copies are not
    independent experimental replicates. Scale is absorbed in visual Q/K.
    """
    def __init__(self,method='full',bandwidth=.25,temperature=5.):
        super().__init__()
        if method not in ('point','fixed','isotropic','full') or bandwidth<=0 or temperature<0:
            raise ValueError('invalid Gaussian position configuration')
        self.method=method
        self.register_buffer('bandwidth',torch.tensor(float(bandwidth)))
        self.register_buffer('temperature',torch.tensor(float(temperature)))

    def encode(self,data,reverse=False):
        qg,kg=controlled_features(data,self.method,self.bandwidth,reverse)
        q,k=(data['k'],data['q']) if reverse else (data['q'],data['k'])
        scale=self.temperature.sqrt()
        return torch.cat([scale*q,qg],-1),torch.cat([scale*k,kg],-1)

    def forward(self,data):
        q,k=self.encode(data)
        return q@k.transpose(-1,-2)

    def attend(self,data,values,reverse=False):
        """The same PE in an ordinary attention/value path, without new modules."""
        q,k=self.encode(data,reverse)
        return (q@k.transpose(-1,-2)).softmax(-1)@values
