"""Four existing equal-area strata, retaining their local support covariance.

This changes only the covariance closure of the existing quad4 measure. Node
means, weights and parent content are unchanged. The positive Gaussian-mixture
integral is known mathematics, not a new kernel or a finite exact Q/K map.
"""
from __future__ import annotations

import torch

from models.round10_partition_pe import prepare_support, group_mean
from models.round10_cell_integral_fast import gaussian_cell_mass_fast


def prepare_gaussian_strata(xy,pixel_cov,pixel_owner,bounds,parent):
    """Return the same four nodes as quad4 plus within-stratum position scatter."""
    result=prepare_support(xy,pixel_cov,pixel_owner,bounds,parent,
                           samples=4,include_full_moments=False)
    count=len(bounds)
    width=int(bounds[:,2].max())
    index=torch.arange(len(xy),device=xy.device)
    x,y=index%width,index//width
    cell=bounds[pixel_owner]
    sx=(x-cell[:,0])*2//(cell[:,2]-cell[:,0])
    sy=(y-cell[:,1])*2//(cell[:,3]-cell[:,1])
    strata=pixel_owner*4+sy*2+sx
    centered=xy-result['nodes'].reshape(-1,2)[strata]
    scatter=group_mean(centered[:,:,None]*centered[:,None,:],strata,count*4).reshape(count,4,2,2)
    result['node_cov']=result['node_cov']+scatter
    return result


def gaussian_stratum_logits(query,key,method,bandwidth,*,diagnostics=None):
    """Integrate full or same-trace isotropic strata over actual native cells."""
    covariance=query['node_cov'].double()
    if method=='iso4':
        eye=torch.eye(2,device=covariance.device,dtype=covariance.dtype)
        covariance=covariance.diagonal(dim1=-2,dim2=-1).mean(-1)[...,None,None]*eye
    elif method!='full4':
        raise ValueError('expected full4 or iso4')
    node_mass=gaussian_cell_mass_fast(query['nodes'],covariance,key['cell_bounds'],
        bandwidth=bandwidth,quadrature_order=32,query_chunk_size=256,key_chunk_size=432)
    mass=(node_mass*query['node_weights'].double()[...,None]).sum(1)
    if diagnostics is not None:
        fov=mass.sum(-1)
        diagnostics.update(query_rows=len(mass),key_count=mass.shape[-1],
            raw_zero_cell_count=int((mass==0).sum()),raw_zero_fov_rows=int((fov==0).sum()),
            raw_tiny_fov_rows=int((fov<1.e-20).sum()),raw_fov_min=float(fov.min()),raw_fov_max=float(fov.max()),
            floor_fov_mass=float(1.e-30*(key['mass'].double()/64).sum()))
    return (mass+1.e-30*(key['mass'].double()/64)[None]).log()
