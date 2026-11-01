"""Independent Round10 numeric and saved-prediction/selection verification.

The reference geometry, partition reduction, metrics and selection below do not
import the producer or experiment runner. The numeric stage imports only the
actual model being checked, and opens calibration observations without labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.special import logsumexp, softmax

ROOT = Path(__file__).resolve().parents[1]
PARTITIONS = ('g8','g16','mixed')
METHODS = ('point','center','fixed','isotropic','full','quad4','quad16')
WIDTHS = (.25,.5,1.,2.)
TEMPERATURES = (0.,5.,10.)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def archive(path):
    with np.load(path,allow_pickle=False) as data:
        return {name:data[name] for name in data.files}


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False))


def entries(data,split=None):
    complete=read(data/'completion.json')
    output=[]
    for item in complete['manifests']:
        path=Path(item['path'])
        if sha(path)!=item['sha256']:raise ValueError('changed manifest')
        meta=read(path)
        if split is not None and meta['split']!=split:continue
        if meta['split'] not in ('train','val') or meta['forward_uses_sensor']:
            raise ValueError('nondevelopment or sensor forward input')
        if sha(path.parent/'inputs.npz')!=meta['input_sha256']:
            raise ValueError('changed forward input')
        output.append((path.parent,meta))
    return output


def reference_features(features,bounds):
    image=np.asarray(features,np.float64).reshape(18,24,-1)
    pooled=[]
    for x0,y0,x1,y1 in bounds:
        row=image[y0//8:y1//8,x0//8:x1//8].mean((0,1))
        pooled.append(row/max(np.linalg.norm(row),1e-12))
    return np.asarray(pooled)


def reference_support(inputs,parts,name,samples=1):
    xy=np.asarray(inputs['q_to_k_xy'],np.float64).reshape(144,192,2)/8
    within=np.asarray(inputs['q_to_k_pixel_cov'],np.float64).reshape(144,192,2,2)/64
    bounds=parts[name+'_bounds']
    means,covariances,nodes,node_cov,node_weights=[],[],[],[],[]
    for x0,y0,x1,y1 in bounds:
        points=xy[y0:y1,x0:x1].reshape(-1,2)
        pixel=within[y0:y1,x0:x1].reshape(-1,2,2)
        m=points.mean(0);centered=points-m
        means.append(m);covariances.append(centered.T@centered/len(points)+pixel.mean(0))
        if samples>1:
            side=int(np.sqrt(samples)); sx=(x1-x0)//side;sy=(y1-y0)//side
            nm,nc,nw=[],[],[]
            for by in range(side):
                for bx in range(side):
                    ys=slice(y0+by*sy,y0+(by+1)*sy);xs=slice(x0+bx*sx,x0+(bx+1)*sx)
                    nm.append(xy[ys,xs].mean((0,1)));nc.append(within[ys,xs].mean((0,1)))
                    nw.append(sx*sy/len(points))
            nodes.append(nm);node_cov.append(nc);node_weights.append(nw)
    output=dict(mean=np.asarray(means),cov=np.asarray(covariances),
        mass=np.asarray(parts[name+'_area'],np.float64),parent=parts[name+'_parent_g16'],
        features=reference_features(inputs['q_features'],bounds))
    if samples>1:
        output.update(nodes=np.asarray(nodes),node_cov=np.asarray(node_cov),node_weights=np.asarray(node_weights))
    return output


def reference_key(inputs,parts,name):
    return dict(mean=np.asarray(parts[name+'_centers'],np.float64)/8,
        cov=np.asarray(parts[name+'_covariance'],np.float64)/64,
        mass=np.asarray(parts[name+'_area'],np.float64),
        features=reference_features(inputs['k_features'],parts[name+'_bounds']))


def log_density(delta,covariance):
    factor=np.linalg.cholesky(covariance)
    whitened=np.linalg.solve(factor,delta[...,None])[...,0]
    logdet=2*np.log(np.diagonal(factor,axis1=-2,axis2=-1)).sum(-1)
    return -.5*(np.square(whitened).sum(-1)+logdet)-np.log(2*np.pi)


def reference_logits(query,key,method,width):
    identity=np.eye(2)
    qc,kc=query['cov'].copy(),key['cov'].copy()
    if method.startswith('quad'):
        score=log_density(query['nodes'][:,:,None]-key['mean'][None,None],
            query['node_cov'][:,:,None]+kc[None,None]+width**2*identity)
        return logsumexp(score+np.log(query['node_weights'])[:,:,None],axis=1)+np.log(key['mass'])[None]
    if method=='point':qc[:]=0;kc[:]=0
    elif method=='center':qc[:]=0
    elif method=='fixed':
        qc=query['mass'][:,None,None]*identity/(64*12)
        kc=key['mass'][:,None,None]*identity/(64*12)
    elif method=='isotropic':
        qc=np.trace(qc,axis1=1,axis2=2)[:,None,None]*identity/2
        kc=np.trace(kc,axis1=1,axis2=2)[:,None,None]*identity/2
    elif method=='rotated':
        rot=np.array([[0.,-1.],[1.,0.]])
        qc=rot@qc@rot.T
    elif method!='full':raise ValueError(method)
    return log_density(query['mean'][:,None]-key['mean'][None],qc[:,None]+kc[None]+width**2*identity)+np.log(key['mass'])[None]


def numeric(data,out,device):
    import torch
    from models.round10_partition_pe import prepare_support,geometry_logits,pool_content
    torch.set_num_threads(4)
    if out.exists():raise ValueError('refuse overwrite')
    out.mkdir(parents=True)
    checked_sources={str(Path(__file__).resolve()):sha(__file__),
        str(ROOT/'models/round10_partition_pe.py'):sha(ROOT/'models/round10_partition_pe.py')}
    started=time.time();parts=archive(data/'partitions.npz')
    first={}
    for path,meta in entries(data,'train'):
        first.setdefault(meta['scene'],(path,meta))
    if len(first)!=3:raise ValueError('expected three calibration scans')
    records=[];moment_errors=[]
    for scene,(path,meta) in first.items():
        inputs=archive(path/'inputs.npz')
        xy=torch.as_tensor(inputs['q_to_k_xy'],dtype=torch.float32,device=device)/8
        within=torch.as_tensor(inputs['q_to_k_pixel_cov'],dtype=torch.float32,device=device)/64
        qfeatures=torch.as_tensor(inputs['q_features'],dtype=torch.float32,device=device)
        kfeatures=torch.as_tensor(inputs['k_features'],dtype=torch.float32,device=device)
        for qname in PARTITIONS:
            owner=torch.as_tensor(parts[qname+'_pixel_index'].ravel(),dtype=torch.long,device=device)
            bounds=torch.as_tensor(parts[qname+'_bounds'],dtype=torch.long,device=device)
            parent=torch.as_tensor(parts[qname+'_parent_g16'],dtype=torch.long,device=device)
            qatom=torch.as_tensor(parts[qname+'_pixel_index'][4::8,4::8].ravel(),dtype=torch.long,device=device)
            qf=pool_content(qfeatures,qatom,len(bounds))
            queries={};references={}
            for method,budget in [('full',1),('quad4',4),('quad16',16)]:
                actual=prepare_support(xy,within,owner,bounds,parent,samples=budget,include_full_moments=budget==1)
                reference=reference_support(inputs,parts,qname,budget)
                for field in ('mean','cov','nodes','node_cov','node_weights'):
                    if field in actual:
                        delta=np.abs(actual[field].cpu().numpy()-reference[field])
                        moment_errors.append(dict(scene=scene,partition=qname,method=method,field=field,
                            max_absolute=float(delta.max()),max_relative_scale=float((delta/(1+np.abs(reference[field]))).max())))
                # Fixed even rows plus geometry-defined extremes; no label or
                # model-score selection. Every native key is retained.
                eigen=np.linalg.eigvalsh(reference['cov'])
                indices=np.unique(np.r_[np.linspace(0,len(bounds)-1,min(32,len(bounds))).astype(int),
                    np.argmax(eigen[:,-1]),np.argmax(eigen[:,-1]/np.maximum(eigen[:,0],1e-12))])
                queries[method]={k:v[indices] for k,v in actual.items()}
                references[method]={k:v[indices] for k,v in reference.items()}
                queries[method]['features']=qf[indices]
            for kname in PARTITIONS:
                katom=torch.as_tensor(parts[kname+'_pixel_index'][4::8,4::8].ravel(),dtype=torch.long,device=device)
                key={k:torch.as_tensor(parts[kname+'_'+field],dtype=torch.float32,device=device)/scale
                     for k,field,scale in [('mean','centers',8),('cov','covariance',64),('mass','area',1)]}
                key['features']=pool_content(kfeatures,katom,len(key['mean']))
                key_ref=reference_key(inputs,parts,kname)
                for method in queries:
                    query,ref=queries[method],references[method]
                    c32=query['features']@key['features'].T
                    c64=ref['features']@key_ref['features'].T
                    for width in WIDTHS:
                        g32=geometry_logits(query,key,method,width)
                        g64=reference_logits(ref,key_ref,method,width)
                        for temperature in (0.,10.):
                            p32=(g32+temperature*c32).softmax(-1).cpu().numpy()
                            p64=softmax(g64+temperature*c64,axis=-1)
                            error=np.abs(p32-p64)
                            records.append(dict(scene=scene,pair=meta['pair'],q=qname,k=kname,
                                method=method,h=width,temperature=temperature,query_rows=len(p32),key_rows=p32.shape[1],
                                max_probability_error=float(error.max()),mean_probability_error=float(error.mean()),
                                topkey_disagreements=int((p32.argmax(-1)!=p64.argmax(-1)).sum())))
        print('NUMERIC',scene,'finished',flush=True)
    maximum=max(r['max_probability_error'] for r in records)
    result=dict(passed=maximum<=1e-4,tolerance=1e-4,max_probability_error=maximum,
        topkey_disagreements=sum(r['topkey_disagreements'] for r in records),
        scope='first calibration pair per scan; all three partitions on each side; full/quad4/quad16; all widths; T 0/10; at most34 fixed/extreme query rows, all keys',
        opened_sensor_labels=False,opened_gate_inputs=False,opened_gate_metrics=False,
        moment_errors=moment_errors,records=records,seconds=time.time()-started,
        source_hashes=checked_sources,
        inputs={str(path/'inputs.npz'):sha(path/'inputs.npz') for path,_ in first.values()})
    if any(sha(path)!=digest for path,digest in checked_sources.items()):
        raise ValueError('numeric audit source changed during execution')
    write(out/'numeric_verification.json',result)
    print('MAXIMUM_PROBABILITY_ERROR',maximum,'PASSED',result['passed'],flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['numeric'],required=True)
    parser.add_argument('--data',type=Path,default=ROOT/'results/round10_partition_data')
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--device',default='cpu')
    args=parser.parse_args()
    numeric(args.data,args.out,args.device)


if __name__=='__main__':main()
