"""Explicit opt-in float32 fast path for the frozen normalized Gaussian PE.

The real-valued map is unchanged: expected RoPE divided by its norm, times the
original content norm. Conservative numerical eligibility selects the simple
float32 evaluation; every other token uses the existing float64 implementation.
This module never replaces or patches models.gaussian_pe.normalized_rotary.
"""
import torch

from models.gaussian_pe import normalized_rotary as normalized_rotary_reference


VARIANCE_ABSOLUTE_SUM_LIMIT=32.
PHASE_ABSOLUTE_SUM_LIMIT=64.
MIN_FILTERED_NORM_RATIO=1/16
CONTENT_SCALE_MIN=2.**-32
CONTENT_SCALE_MAX=2.**32


def _compatible(features,mean,covariance,frequency):
    prefix=features.shape[:-1]
    return (all(x.dtype==torch.float32 for x in (features,mean,covariance,frequency))
            and all(x.device==features.device for x in (mean,covariance,frequency))
            and frequency.ndim==2 and frequency.shape[-1]==2
            and features.shape[-1]==2*len(frequency)
            and mean.shape==prefix+(2,) and covariance.shape==prefix+(2,2))


@torch.no_grad()
def eligibility(features,mean,covariance,frequency):
    """Per-token conservative ordinary-domain predicate (no fitted thresholds).

    Bounds control cancellation in omega^T C omega and phase, constrain content
    scale, and bound the normalization condition number by 16. No input is
    clamped or floored by this predicate. Thresholds are explicit audit constants.
    """
    if not _compatible(features,mean,covariance,frequency):return None
    variance=torch.einsum('fd,...de,fe->...f',frequency,covariance,frequency)
    absolute_sum=torch.einsum('fd,...de,fe->...f',frequency.abs(),covariance.abs(),frequency.abs())
    phase_absolute_sum=mean.abs()@frequency.abs().T
    scale=features.abs().amax(-1,keepdim=True)
    unit=features/torch.where(scale>0,scale,torch.ones_like(scale))
    logamp=-variance/2
    amplitude=(logamp-logamp.amax(-1,keepdim=True)).exp()
    filtered=(unit.reshape(*unit.shape[:-1],-1,2)*amplitude[...,None]).flatten(-2)
    ratio=filtered.norm(dim=-1)/unit.norm(dim=-1).clamp_min(torch.finfo(features.dtype).tiny)
    return (torch.isfinite(features).all(-1)&torch.isfinite(mean).all(-1)
            &torch.isfinite(covariance).all((-2,-1))&torch.isfinite(variance).all(-1)
            &torch.isfinite(phase_absolute_sum).all(-1)&torch.isfinite(absolute_sum).all(-1)
            &(variance.amin(-1)>=0)&(absolute_sum.amax(-1)<=VARIANCE_ABSOLUTE_SUM_LIMIT)
            &(phase_absolute_sum.amax(-1)<=PHASE_ABSOLUTE_SUM_LIMIT)
            &(scale[...,0]>=CONTENT_SCALE_MIN)&(scale[...,0]<=CONTENT_SCALE_MAX)
            &(ratio>=MIN_FILTERED_NORM_RATIO))


def _ordinary_float32(features,mean,covariance,frequency):
    """Same normalized Gaussian map; called only on eligible tokens."""
    scale=features.abs().amax(-1,keepdim=True)
    unit=features/scale
    original_norm=unit.norm(dim=-1,keepdim=True)*scale
    variance=torch.einsum('fd,...de,fe->...f',frequency,covariance,frequency)
    logamp=-variance/2
    amplitude=(logamp-logamp.amax(-1,keepdim=True)).exp()
    filtered=(unit.reshape(*unit.shape[:-1],-1,2)*amplitude[...,None]).flatten(-2)
    filtered=filtered/filtered.norm(dim=-1,keepdim=True)
    phase=mean@frequency.T
    x,y=filtered.reshape(*filtered.shape[:-1],-1,2).unbind(-1)
    rotated=torch.stack([x*phase.cos()-y*phase.sin(),x*phase.sin()+y*phase.cos()],-1).flatten(-2)
    return rotated*original_norm


def normalized_rotary_fast(features,mean,covariance,frequency,*,return_diagnostics=False):
    """Opt-in mixed evaluation; fallback preserves extreme/unsupported inputs.

    Eligibility is evaluated without building an unsafe discarded autograd
    graph. Mixed inputs are gathered into separate paths, then reassembled.
    This dispatch synchronizes CUDA when testing masks; no GPU speed claim is
    made by the CPU audit. Ordinary gradients retain zero-block derivatives.
    """
    mask=eligibility(features,mean,covariance,frequency)
    if mask is None:
        result=normalized_rotary_reference(features,mean,covariance,frequency)
        info=dict(eligible_tokens=0,total_tokens=features.numel()//features.shape[-1],unsupported_shape_or_dtype=True)
    elif bool(mask.all()):
        result=_ordinary_float32(features,mean,covariance,frequency)
        info=dict(eligible_tokens=mask.numel(),total_tokens=mask.numel(),unsupported_shape_or_dtype=False)
    elif not bool(mask.any()):
        result=normalized_rotary_reference(features,mean,covariance,frequency)
        info=dict(eligible_tokens=0,total_tokens=mask.numel(),unsupported_shape_or_dtype=False)
    else:
        f=features.reshape(-1,features.shape[-1]);m=mean.reshape(-1,2);c=covariance.reshape(-1,2,2)
        selected=mask.reshape(-1).nonzero(as_tuple=False).flatten()
        fallback=(~mask).reshape(-1).nonzero(as_tuple=False).flatten()
        result=torch.empty_like(f).index_copy(0,selected,_ordinary_float32(f[selected],m[selected],c[selected],frequency))
        result=result.index_copy(0,fallback,normalized_rotary_reference(f[fallback],m[fallback],c[fallback],frequency))
        result=result.reshape_as(features)
        info=dict(eligible_tokens=int(mask.sum()),total_tokens=mask.numel(),unsupported_shape_or_dtype=False)
    return (result,info) if return_diagnostics else result
