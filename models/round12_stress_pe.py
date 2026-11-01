"""Frozen-configuration scale/stress adapter; no additional learned parameters."""
from __future__ import annotations

import hashlib
import numpy as np
import torch

from experiments import round10_partition_data as D
from models.round10_partition_pe import prepare_support, pool_content
from models.round11_stratum_pe import prepare_gaussian_strata
from models.round10_cell_integral_fast import gaussian_cell_mass_fast

PARTITIONS = ('g8', 'g16', 'mixed', 'g24')
STATES = ('zero', 'depth09', 'depth11', 'ry_plus1', 'ry_minus1', 'tx_plus01', 'tx_minus01')
SCALE_CONDITIONS = (('g8','g24'), ('g24','g8'), ('g16','g24'), ('g24','g16'), ('g24','g24'))
ORIGINAL_CONDITIONS = tuple((q,k) for q in PARTITIONS[:3] for k in PARTITIONS[:3])
ROLES = ('full4', 'iso4', 'full1', 'point', 'point_pck', 'fixed', 'quad4', 'quad16')


def split_cell_bounds(bounds):
    """Split 24px cells into four 12px rectangles; preserve native key identity."""
    bounds = np.asarray(bounds, dtype=np.int64)
    children, owners = [], []
    for owner, (x0,y0,x1,y1) in enumerate(bounds):
        if max(x1-x0,y1-y0) <= 16:
            children.append([x0,y0,x1,y1]); owners.append(owner)
        elif x1-x0 == y1-y0 == 24:
            for y in (y0,y0+12):
                for x in (x0,x0+12):
                    children.append([x,y,x+12,y+12]); owners.append(owner)
        else:
            raise ValueError('only native <=16px cells or square 24px cells supported')
    return (np.asarray(children, np.float64)-.5)/8, np.asarray(owners, np.int64)


def partitions():
    output = {}
    for name in PARTITIONS:
        bounds = (np.array([[x,y,x+24,y+24] for y in range(0,144,24)
                           for x in range(0,192,24)], np.int32)
                  if name == 'g24' else D.partition_bounds(name))
        owner = D.partition_index(bounds)
        mean,cov,mass = D.native_support(bounds)
        parent = (mean[:,1].astype(np.int64)//16)*12+mean[:,0].astype(np.int64)//16
        if name == 'g24': parent[:] = -1  # No exact common16 parent exists.
        children,child_owner = split_cell_bounds(bounds)
        output[name] = dict(bounds=bounds,owner=owner.ravel(),atoms=owner[4::8,4::8].ravel(),
            parent=parent,mean=mean/8,cov=cov/64,mass=mass,
            cell_bounds=children,cell_owner=child_owner)
    return output


def partition_tensors(parts, device):
    integer = {'bounds','owner','atoms','parent','cell_owner'}
    return {name:{key:torch.as_tensor(value, device=device,
                    dtype=torch.long if key in integer else
                    torch.float64 if key == 'cell_bounds' else torch.float32)
                  for key,value in part.items()} for name,part in parts.items()}


def array_sha(array):
    a=np.ascontiguousarray(array)
    return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()


def perturbed_observations(inputs, state):
    """Reproject 3D points and pixel footprints; sensor data never enter here."""
    if state not in STATES: raise ValueError('unknown perturbation')
    factor = .9 if state == 'depth09' else 1.1 if state == 'depth11' else 1.
    points = np.asarray(inputs['q_points3'],np.float64)*factor
    covariance = np.asarray(inputs['q_pixel_cov3'],np.float64)*factor**2
    transform = np.asarray(inputs['q_to_k_transform'],np.float64).copy()
    if state.startswith('ry_'):
        angle=np.deg2rad(1. if state == 'ry_plus1' else -1.)
        c,s=np.cos(angle),np.sin(angle)
        rotation=np.array([[c,0,s],[0,1,0],[-s,0,c]])
        transform[:3,:] = rotation@transform[:3,:]
    if state.startswith('tx_'):
        transform[0,3] += .01 if state == 'tx_plus01' else -.01
    key_points=points@transform[:3,:3].T+transform[:3,3]
    key_cov=transform[:3,:3]@covariance@transform[:3,:3].T
    xy,cov,front=D.project_finite_support(key_points,key_cov,inputs['k_K'])
    effective_depth={side:np.nan_to_num(np.asarray(inputs[side+'_predicted_depth'],np.float64),
                        nan=10.,posinf=20.,neginf=.1).clip(.1,20.)*factor for side in ('q','k')}
    np.testing.assert_allclose(points[:,2],effective_depth['q'].ravel(),atol=1.e-12,rtol=1.e-12)
    diagnostics=dict(state=state,depth_factor=factor,transform=transform.tolist(),
        effective_depth_sha256={k:array_sha(v) for k,v in effective_depth.items()},
        projected_xy_sha256=array_sha(xy),projected_cov_sha256=array_sha(cov),
        predicted_in_front_count=int(front.sum()),
        key_depth_used_by_positional_operator=False,
        depth_policy='scale both clipped RGB depth fields; no post-scaling reclip')
    if state == 'zero':
        diagnostics['zero_xy_max_error']=float(np.max(np.abs(xy-inputs['q_to_k_xy'])))
        diagnostics['zero_cov_max_error']=float(np.max(np.abs(cov-inputs['q_to_k_pixel_cov'])))
        np.testing.assert_allclose(xy,inputs['q_to_k_xy'],atol=1.e-12,rtol=1.e-12)
        np.testing.assert_allclose(cov,inputs['q_to_k_pixel_cov'],atol=1.e-10,rtol=1.e-12)
    return dict(q_to_k_xy=xy,q_to_k_pixel_cov=cov,q_features=inputs['q_features'],
                k_features=inputs['k_features']),diagnostics


def prepare_query(observations, partition, method):
    args=(observations['q_to_k_xy']/8,observations['q_to_k_pixel_cov']/64,
          partition['owner'],partition['bounds'],partition['parent'])
    if method in ('full4','iso4'): query=prepare_gaussian_strata(*args)
    else:
        count=4 if method == 'quad4' else 16 if method == 'quad16' else 1
        query=prepare_support(*args,samples=count,include_full_moments=count==1)
    query['features']=pool_content(observations['q_features'],partition['atoms'],len(partition['mean']))
    return query


def prepare_key(observations, partition):
    return dict(mean=partition['mean'],mass=partition['mass'],cell_bounds=partition['cell_bounds'],
        cell_owner=partition['cell_owner'],
        features=pool_content(observations['k_features'],partition['atoms'],len(partition['mean'])))


def tiled_cell_mass(mean,covariance,key,bandwidth):
    child=gaussian_cell_mass_fast(mean,covariance,key['cell_bounds'],bandwidth=bandwidth,
            quadrature_order=32,query_chunk_size=256,key_chunk_size=432)
    mass=child.new_zeros((*child.shape[:-1],len(key['mass'])))
    mass.index_add_(-1,key['cell_owner'],child)
    return mass


def geometry_logits(query,key,method,bandwidth,*,diagnostics=None):
    if method in ('full4','iso4','quad4','quad16'):
        covariance=query['node_cov'].double()
        if method == 'iso4':
            eye=torch.eye(2,device=covariance.device,dtype=covariance.dtype)
            covariance=covariance.diagonal(dim1=-2,dim2=-1).mean(-1)[...,None,None]*eye
        raw=tiled_cell_mass(query['nodes'],covariance,key,bandwidth)
        mass=(raw*query['node_weights'].double()[...,None]).sum(1)
    else:
        covariance=query['cov'].double()
        if method in ('point','center'): covariance=torch.zeros_like(covariance)
        elif method == 'fixed':
            covariance=query['mass'].double()[:,None,None]/(64*12)*torch.eye(2,device=covariance.device,dtype=covariance.dtype)
        elif method != 'full': raise ValueError('unknown method')
        mass=tiled_cell_mass(query['mean'],covariance,key,bandwidth)
    if diagnostics is not None:
        fov=mass.sum(-1)
        diagnostics.update(query_rows=len(mass),key_count=mass.shape[-1],
            raw_zero_cell_count=int((mass==0).sum()),raw_zero_fov_rows=int((fov==0).sum()),
            raw_tiny_fov_rows=int((fov<1.e-20).sum()),raw_fov_min=float(fov.min()),raw_fov_max=float(fov.max()))
    return (mass+1.e-30*(key['mass'].double()/64)[None]).log()
