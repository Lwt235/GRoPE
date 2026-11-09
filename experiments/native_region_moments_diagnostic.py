"""Fit-only geometry study of native-Gaussian groups; no model evaluation."""
import argparse
import hashlib
import os
from pathlib import Path
import time

import numpy as np
import torch

from experiments.round65_common import ROOT, CONFIG, read, write, sha
from experiments.geometry_scale_diagnostic import rng
from experiments.round58_data import fps_and_neighbors
from models.round69_compact_relation import descriptors

OUT = ROOT/'results/native_region_moments/diagnostic1'
PLAN = ROOT/'docs/native_region_moments_plan.md'
NATIVE = ROOT/'results/round65_actual_gs/attempt1/data'
OLD = ROOT/'results/geometry_scale_diagnostic/attempt1'
QUANTILES = [.05,.25,.5,.75,.95]


def mixture_moments(centers, covariance, weights=None):
    """Exact first two moments of a local normalized Gaussian mixture."""
    assert centers.shape[-1] == 3 and covariance.shape == (*centers.shape,3)
    if weights is None:
        weights = torch.ones_like(centers[...,0])
    assert weights.shape == centers.shape[:-1]
    assert torch.isfinite(weights).all() and (weights >= 0).all() and (weights.sum(-1)>0).all()
    weights = weights/weights.sum(-1,keepdim=True)
    mean = (weights[...,None]*centers).sum(-2)
    delta = centers-mean.unsqueeze(-2)
    within = (weights[...,None,None]*covariance).sum(-3)
    between = torch.einsum('...k,...ki,...kj->...ij',weights,delta,delta)
    return mean,within,between,within+between


def quantiles(values):
    a=np.asarray(values,dtype=np.float64)
    assert a.size and np.isfinite(a).all()
    return dict(count=a.size,mean=float(a.mean()),quantiles=np.quantile(a,QUANTILES).tolist())


def freeze():
    assert not (OUT/'lock.json').exists()
    unit=read(OUT/'unit_checks.json');assert unit['passed']
    paths=[Path(__file__),PLAN,ROOT/'tests/test_native_region_moments_diagnostic.py',
           ROOT/'experiments/round65_common.py',ROOT/'experiments/round58_data.py',
           ROOT/'experiments/round58_common.py',ROOT/'experiments/geometry_scale_diagnostic.py',
           ROOT/'models/round69_compact_relation.py',ROOT/'models/round58_support_pe.py',
           NATIVE/'manifest.json',OLD/'plan.json',OLD/'result.json',OUT/'unit_checks.json']
    for name,digest in unit['source_sha256'].items():assert sha(name)==digest
    lock=dict(created_unix=time.time(),source_sha256={str(p):sha(p) for p in paths},
              objects_per_class=2,primitive_tokens=512,region_tokens=64,neighbors=32,
              epsilon=CONFIG['covariance_floor'],quantile_probabilities=QUANTILES,
              split='fit',new_fits=0,task_scores_read=False,goal_complete=False)
    write(OUT/'lock.json',lock);(OUT/'lock.sha256').write_text(sha(OUT/'lock.json')+'\n')
    print('FROZEN',sha(OUT/'lock.json'),flush=True)


def checked():
    lock=read(OUT/'lock.json')
    assert sha(OUT/'lock.json')==(OUT/'lock.sha256').read_text().strip()
    for p,d in lock['source_sha256'].items():assert sha(p)==d,p
    return lock


@torch.no_grad()
def run():
    lock=checked();torch.set_num_threads(4)
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    torch.cuda.set_per_process_memory_fraction(3*2**30/torch.cuda.get_device_properties(0).total_memory)
    assert not (OUT/'result.json').exists() and not (OUT/'execution_identity.json').exists()
    write(OUT/'execution_identity.json',dict(pid=os.getpid(),started_unix=time.time(),
          start_ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]))
    fit=next(r for r in read(NATIVE/'manifest.json')['partitions'] if r['split']=='fit')
    arrays={};verified={}
    for name in ('centers','covariance','labels'):
        p=NATIVE/'fit'/(name+'.npy');verified[str(p)]=sha(p)
        assert verified[str(p)]==fit['hashes'][p.name]
        arrays[name]=np.load(p,mmap_mode='r',allow_pickle=False)
    selected=[];all_centers=[];all_covariances=[];all_token_ids=[]
    for category in range(60):
        indices=sorted(np.flatnonzero(arrays['labels']==category).tolist(),key=lambda i:fit['ids'][i])[:2]
        assert len(indices)==2
        for i in indices:
            identity=fit['ids'][i]
            token_ids=rng(identity+':tokens').choice(arrays['centers'].shape[1],512,replace=False)
            selected.append(dict(identity=identity,category=category,fit_index=i))
            all_centers.append(arrays['centers'][i,token_ids]);all_covariances.append(arrays['covariance'][i,token_ids])
            all_token_ids.append(token_ids)
    old_ids={r['identity'] for r in read(OLD/'result.json')['contexts'] if r['kind']=='native'}
    assert len(selected)==120 and {r['identity'] for r in selected}==old_ids
    centers=np.stack(all_centers);covariances=np.stack(all_covariances)
    collected={};records=[];epsilon=lock['epsilon'];start=time.time()
    eye=torch.eye(3,device='cuda',dtype=torch.float64)
    mask=~torch.eye(64,device='cuda',dtype=torch.bool)
    for offset in range(0,120,8):
        c=torch.as_tensor(centers[offset:offset+8],device='cuda',dtype=torch.float64)
        s=torch.as_tensor(covariances[offset:offset+8],device='cuda',dtype=torch.float64)
        anchors,groups=fps_and_neighbors(c,64,32)
        rows=torch.arange(len(c),device='cuda')[:,None,None]
        mean,within,between,total=mixture_moments(c[rows,groups],s[rows,groups])
        physical=within-epsilon*eye
        values,vectors=torch.linalg.eigh(total)
        inverse_root=(vectors*values.rsqrt().unsqueeze(-2))@vectors.transpose(-1,-2)
        fractional=torch.linalg.eigvalsh(inverse_root@physical@inverse_root)
        full_phi=descriptors('relative5',mean,total)[:,mask]
        point_phi=descriptors('relative5',mean,between+epsilon*eye)[:,mask]
        assert torch.equal(full_phi[...,0],point_phi[...,0])
        difference=full_phi[...,1:]-point_phi[...,1:]
        trace_total=total.diagonal(dim1=-2,dim2=-1).sum(-1)
        values_out=dict(anchors=anchors,groups=groups,means=mean,within=within,between=between,total=total,
            trace_physical_fraction=physical.diagonal(dim1=-2,dim2=-1).sum(-1)/trace_total,
            trace_floor_fraction=3*epsilon/trace_total,
            directional_fraction_eigenvalues=fractional,
            shape_descriptor_signed_difference=difference,
            shape_descriptor_relative_l2=difference.norm(dim=-1)/point_phi[...,1:].norm(dim=-1).clamp_min(1e-12),
            full_shape_descriptor=full_phi[...,1:],centers_shape_descriptor=point_phi[...,1:])
        values_out={k:v.cpu().numpy() for k,v in values_out.items()}
        for k,v in values_out.items():collected.setdefault(k,[]).append(v)
        for j in range(len(c)):
            raw=dict(trace_physical_fraction=values_out['trace_physical_fraction'][j],
                     trace_floor_fraction=values_out['trace_floor_fraction'][j],
                     directional_fraction_mean=values_out['directional_fraction_eigenvalues'][j].mean(-1),
                     directional_fraction_min=values_out['directional_fraction_eigenvalues'][j,:,0],
                     directional_fraction_max=values_out['directional_fraction_eigenvalues'][j,:,-1],
                     shape_descriptor_relative_l2=values_out['shape_descriptor_relative_l2'][j])
            for k,name in enumerate(('metric_i','metric_j','trace_i','trace_j')):
                raw[name+'_absolute_difference']=np.abs(values_out['shape_descriptor_signed_difference'][j,:,k])
            records.append(dict(**selected[offset+j],metrics={k:quantiles(v) for k,v in raw.items()}))
        print('GROUPED',len(records),'/ 120',flush=True)
    collected={k:np.concatenate(v) for k,v in collected.items()}
    collected.update(input_centers=centers,input_covariances=covariances,input_token_ids=np.stack(all_token_ids))
    path=OUT/'arrays.npz';assert not path.exists();np.savez_compressed(path,**collected)
    context_summary={k:quantiles([r['metrics'][k]['quantiles'][2] for r in records]) for k in records[0]['metrics']}
    result=dict(created_unix=time.time(),lock_sha256=sha(OUT/'lock.json'),verified_inputs_sha256=verified,
        objects=120,regions=120*64,pairs=120*64*63,context_median_summary=context_summary,records=records,
        arrays_sha256=sha(path),epsilon=epsilon,seconds=time.time()-start,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),new_fits=0,task_scores_read=False,
        utility_evidence=False,scientific_review_pending=True,goal_complete=False)
    write(OUT/'result.json',result);checked()
    print('DIAGNOSTIC_COMPLETE',context_summary,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('freeze','run'))
    globals()[parser.parse_args().action]()
