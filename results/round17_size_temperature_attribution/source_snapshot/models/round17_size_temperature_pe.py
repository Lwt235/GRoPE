"""Unrun fixed native-size attribution; observed Gaussian geometry stays intact."""
from __future__ import annotations
import math
import torch


def native_trace(bounds):
    """Continuous rectangle footprint trace in reference-8 squared units.

    Bounds are integer half-open [x0,y0,x1,y1] in original pixels. Rectangles
    use (width²+height²)/768, never the area-based historical fixed baseline.
    """
    if (not isinstance(bounds,torch.Tensor) or bounds.ndim!=2 or bounds.shape[-1]!=4
            or not len(bounds) or bounds.dtype!=torch.int64):
        raise ValueError('native bounds must be a nonempty int64 [Q,4] tensor')
    b=bounds.to(dtype=torch.float64)
    if not bool(torch.isfinite(b).all()):
        raise ValueError('native pixel bounds must be finite integers')
    size=b[:,2:]-b[:,:2]
    if not bool((size>0).all()):
        raise ValueError('native rectangles must have positive width and height')
    trace=size.square().sum(-1)/(64.*12.)
    if not bool(torch.isfinite(trace).all()):
        raise ValueError('native trace overflow')
    return trace


def size_temperature(bounds,bandwidth=.25,temperature=10.):
    """Float64 bounds trace/formula, one rounding to float32; no extra tuning."""
    if (isinstance(bandwidth,bool) or isinstance(temperature,bool)
            or not math.isfinite(bandwidth) or not math.isfinite(temperature)
            or bandwidth<=0 or temperature<0):
        raise ValueError('positive finite h and nonnegative finite T required')
    h2=float(bandwidth)**2
    if not math.isfinite(h2) or h2<=0:
        raise ValueError('invalid squared bandwidth')
    trace=native_trace(bounds)
    tau=(float(temperature)*(h2/(h2+trace))).to(dtype=torch.float32)
    if not bool(torch.isfinite(tau).all()):
        raise ValueError('float32 native temperature overflow')
    return tau


def probabilities(geometry,content,bounds,bandwidth=.25,temperature=10.):
    """Same complete Gaussian log mass and content; only temperature changes."""
    if (not isinstance(geometry,torch.Tensor) or not isinstance(content,torch.Tensor)
            or geometry.dtype!=torch.float64 or content.dtype!=torch.float32
            or not isinstance(bounds,torch.Tensor) or geometry.ndim!=2 or geometry.shape!=content.shape
            or geometry.device!=content.device or bounds.device!=geometry.device
            or len(bounds)!=len(geometry) or not bool(torch.isfinite(geometry).all())
            or not bool(torch.isfinite(content).all())):
        raise ValueError('finite float64 geometry and matching float32 content required')
    tau=size_temperature(bounds,bandwidth,temperature)
    bias=tau[:,None]*content
    if not bool(torch.isfinite(bias).all()):
        raise ValueError('native-size content bias overflow')
    return (geometry+bias.double()).softmax(-1)
