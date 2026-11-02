"""Bounded old-VAL input-geometry diagnosis; no fitting or model selection."""
import argparse
import time
from pathlib import Path
import numpy as np
import torch

from experiments import round46_native3d_train as R46
from experiments import round46_native3d_data as D46
from experiments.round43_frequency_train import ROOT, sha, read, write, checked

OUT=ROOT/'results/round47_conditional/geometry_diagnostic1'
CONDITIONS=('projected2d','native3d','native3d_xy')


def freeze():
    if OUT.exists():raise ValueError('retain existing diagnostic')
    checked(R46.OUT)
    OUT.mkdir(parents=True)
    write(OUT/'plan.json',dict(question='Does the distance representation itself alter center correspondence on old VAL?',
        conditions=CONDITIONS,split='val',pairs=64,all_queries=True,all432keys=True,
        methods='untrained nearest Euclidean key; no scale/temperature sweep',
        metrics=['PCK4','PCK8','argmax pixel EPE','paired correctness counts',
                 'label-weighted predicted-coordinate residual mean/covariance/eigenvalues/axis fractions/norm quantiles'],
        source_sha256=sha(__file__),prior_lock_sha256=sha(R46.OUT/'lock.json'),
        manifest_sha256=sha(R46.OUT/'data/val/manifest.json'),
        new_training=False,changes_R47=False,fresh_data=False,unix=time.time(),
        scope='Residuals use predicted anchors and old correspondence labels, not ground-truth3D positions. XY drops both depth information and part of the metric. No automatic causal attribution or candidate selection.'))
    (OUT/'plan.sha256').write_text(sha(OUT/'plan.json')+'\n')


def run():
    plan=read(OUT/'plan.json')
    if sha(OUT/'plan.json')!=(OUT/'plan.sha256').read_text().strip():raise ValueError('plan identity')
    if sha(__file__)!=plan['source_sha256'] or sha(R46.OUT/'lock.json')!=plan['prior_lock_sha256']:
        raise ValueError('source identity')
    if sha(R46.OUT/'data/val/manifest.json')!=plan['manifest_sha256']:raise ValueError('data identity')
    if (OUT/'result.json').exists():raise ValueError('retain result')
    bundle,ids=D46.load('val',R46.OUT,'cpu')
    if len(ids)!=64:raise ValueError('all old VAL pairs required')
    data={k:v.numpy() for k,v in bundle.items()}
    mask=data['gt_valid'];weights=data['key_weights'].astype(float);indices=data['key_indices'].clip(0)
    if not np.allclose(weights.sum(-1)[mask],1.,atol=1e-6):raise ValueError('valid interpolation weights')
    yy,xx=np.mgrid[:18,:24];grid=np.stack((xx.ravel()*8+3.5,yy.ravel()*8+3.5),-1)
    rows=[];summary={};saved={};correct={}
    for condition in CONDITIONS:
        if condition=='projected2d':x,y=data['qmean_2d'],data['k_to_q_2d']
        else:
            x,y=data['qmean'],data['k_to_q']
            if condition=='native3d_xy':x,y=x[...,:2],y[...,:2]
        residual=[];args=[];errors=[]
        for n,identity in enumerate(ids):
            q=x[n].astype(float);k=y[n].astype(float)
            distance=(q*q).sum(-1)[:,None]+(k*k).sum(-1)[None]-2*q@k.T
            arg=distance.argmin(-1);error=np.linalg.norm(grid[arg]-data['pixcontinuous'][n],axis=-1)
            predicted_match=(k[indices[n]]*weights[n,...,None]).sum(-2)
            residual.append((predicted_match-q)[mask[n]])
            args.append(arg);errors.append(error)
            rows.append(dict(identity,condition=condition,n_valid=int(mask[n].sum()),
                pck4=int(((error<=4)&mask[n]).sum()),pck8=int(((error<=8)&mask[n]).sum()),
                epe_sum=float(error[mask[n]].sum())))
        residual=np.concatenate(residual);error=np.stack(errors);args=np.stack(args)
        covariance=np.cov(residual,rowvar=False,bias=True)
        eigenvalues=np.linalg.eigvalsh(covariance)
        selected=[r for r in rows if r['condition']==condition];nv=sum(r['n_valid'] for r in selected)
        summary[condition]=dict(valid_queries=nv,pck4=sum(r['pck4'] for r in selected)/nv,
            pck8=sum(r['pck8'] for r in selected)/nv,epe=sum(r['epe_sum'] for r in selected)/nv,
            residual_mean=residual.mean(0).tolist(),residual_covariance=covariance.tolist(),
            residual_cov_eigenvalues=eigenvalues.tolist(),residual_cov_condition=float(eigenvalues[-1]/eigenvalues[0]),
            residual_axis_variance_fraction=(covariance.diagonal()/covariance.trace()).tolist(),
            residual_norm_quantiles=dict(zip(('p50','p90','p95'),np.quantile(np.linalg.norm(residual,axis=-1),[.5,.9,.95]).tolist())))
        saved[condition+'_argmax']=args;saved[condition+'_residual']=residual
        correct[condition]=(error<=8)&mask
    paired={}
    for left,right in (('projected2d','native3d'),('native3d_xy','native3d')):
        a,b=correct[left],correct[right]
        paired[left+'__'+right]=dict(both_correct=int((a&b).sum()),left_only=int((a&~b).sum()),
            right_only=int((~a&b).sum()),both_wrong=int((~a&~b&mask).sum()))
    np.savez_compressed(OUT/'query_outputs.npz',**saved)
    write(OUT/'frame_rows.json',rows)
    write(OUT/'result.json',dict(summary=summary,paired=paired,plan_sha256=sha(OUT/'plan.json'),
        outputs_sha256=sha(OUT/'query_outputs.npz'),frame_rows_sha256=sha(OUT/'frame_rows.json'),
        fresh_data=False,changes_R47=False,scope=plan['scope']))
    print('Geometry diagnosis complete; input evidence only, no R47 modification.',flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('freeze','run'));args=parser.parse_args()
    torch.set_num_threads(4);torch.set_num_interop_threads(4)
    globals()[args.stage]()


if __name__=='__main__':main()
