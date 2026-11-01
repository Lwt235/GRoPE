"""Deterministic numerical subdivision of original keys for quartic mass.

Only the integration grid changes: each original key rectangle is divided
into equal rectangles of side <=1 (reference8 units). Child probabilities are
summed back into the original key. No extra attention tokens, kernel shape,
floor, normalization, content or learned parameter is introduced.
The original Cartesian operator and its GL48 audit remain frozen unchanged.
"""
from __future__ import annotations

import torch

from models.round15_quartic_pe import quartic_cell_mass_cartesian


def integration_tiles(cell_bounds):
    """Return differentiable child bounds and original-key indices.

    Integer subdivision counts are determined only by physical key width.
    Bounds derivatives flow through the equal-width affine subdivision.
    The tested domain is positive finite rectangles with sides <=3.
    """
    if cell_bounds.ndim!=2 or cell_bounds.shape[-1]!=4 or not len(cell_bounds):
        raise ValueError('nonempty [K,4] cell bounds required')
    if not bool(torch.isfinite(cell_bounds).all()):raise ValueError('finite bounds required')
    size=cell_bounds[:,2:]-cell_bounds[:,:2]
    if not bool(((size>0)&(size<=3.+1e-12)).all()):raise ValueError('cell sides must be in (0,3]')
    counts=torch.ceil(size.detach()-1e-12).clamp_min(1).to(torch.long)
    if bool((counts==1).all()):
        return cell_bounds,torch.arange(len(cell_bounds),device=cell_bounds.device)
    children=[];owners=[]
    for nx,ny in torch.unique(counts,dim=0).cpu().tolist():
        ids=torch.nonzero((counts[:,0]==nx)&(counts[:,1]==ny),as_tuple=False).flatten()
        rect=cell_bounds[ids];width=(rect[:,2:]-rect[:,:2])/rect.new_tensor([nx,ny])
        yy,xx=torch.meshgrid(torch.arange(ny,device=rect.device,dtype=rect.dtype),
                            torch.arange(nx,device=rect.device,dtype=rect.dtype),indexing='ij')
        offset=torch.stack((xx,yy),-1).reshape(-1,2)
        low=rect[:,None,:2]+offset[None]*width[:,None]
        high=rect[:,None,:2]+(offset[None]+1)*width[:,None]
        children.append(torch.cat((low,high),-1).reshape(-1,4))
        owners.append(ids[:,None].expand(-1,nx*ny).reshape(-1))
    return torch.cat(children,0),torch.cat(owners,0)


def quartic_cell_mass(qmean,qcov,cell_bounds,*,bandwidth=.5,
                      quadrature_order=16,query_chunk_size=16,key_chunk_size=32):
    """Raw float64 mass of each original key via fixed GL on <=1 tiles.

    Caller must add any declared native-key floor exactly once after this
    function. The numerical subdivision is internal and must not appear in
    content scores, softmax denominators or PCK readout as new keys.
    """
    h=torch.as_tensor(bandwidth)
    if h.numel()!=1 or not bool(torch.isfinite(h).all()) or bool((h<.25).any()):
        raise ValueError('bounded tiled integration requires bandwidth >= .25')
    boxes=cell_bounds.to(device=qmean.device,dtype=torch.float64)
    tiles,owner=integration_tiles(boxes)
    values=quartic_cell_mass_cartesian(qmean,qcov,tiles,bandwidth=bandwidth,
        quadrature_order=quadrature_order,query_chunk_size=query_chunk_size,
        key_chunk_size=key_chunk_size)
    if len(tiles)==len(boxes):return values
    shape=values.shape
    result=values.new_zeros((*shape[:-1],len(boxes)))
    return result.index_add(-1,owner,values)
