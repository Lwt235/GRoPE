"""Independent formulas/metric replay; same author, separate numerical routes.

Uses an independent complex rotary formula and explicit pairwise differences,
not the production rotary/distance helpers. Not an independent human review.
"""
import argparse
import json
import math
from pathlib import Path
import time
import unittest
import numpy as np
import torch
from scipy.special import logsumexp
from models.round40_decoupled_pe import ROLES, NEW_ROLES, SHAPE_ROLES
from models.round7_depth_attention import CONFIGS
from experiments.round7_depth_audit import numpy_rows, numpy_macro, numpy_attention
from experiments.round40_decoupled_data import OUT, ROOT, verify, read, write, sha


def numpy_logits(data,state,role):
    def arr(x):
        return x.detach().cpu().numpy().astype(np.float64) if isinstance(x,torch.Tensor) else np.asarray(x,np.float64)
    q,k=arr(data['q']),arr(data['k']); W=arr(state['qk.weight'])
    q,k=q@W.T,k@W.T
    q/=np.maximum(np.linalg.norm(q,axis=-1,keepdims=True),1e-12)
    k/=np.maximum(np.linalg.norm(k,axis=-1,keepdims=True),1e-12)
    qa,ka,qc,kc=(arr(data[n]) for n in ('q_anchor3','k_anchor3','q_cov3','k_cov3'))
    freq=arr(state['frequencies'])
    def encoded(x,a,c):
        z=(x[...,0::2]+1j*x[...,1::2])*np.exp(1j*(a@freq.T))
        if role in ('expected_rope','normalized_rope'):
            variance=np.sum((c[...,None,:,:]*freq[:,None,:]).sum(-1)*freq,axis=-1)
            z*=np.exp(-.5*np.maximum(variance,0.))
        if role=='normalized_rope':
            z/=np.maximum(np.sqrt(np.sum(abs(z)**2,axis=-1,keepdims=True)),1e-12)
        return z
    score=10.*(encoded(q,qa,qc)@encoded(k,ka,kc).conj().swapaxes(-1,-2)).real
    if role not in SHAPE_ROLES and role!='center_rbf':
        return score
    beta=float(np.logaddexp(0.,arr(state['strength_raw'])))
    if role=='center_rbf':
        return score-beta*np.sum((qa[..., :,None,:]-ka[...,None,:,:])**2,axis=-1)
    if role=='shape_isotropic':
        qc=np.trace(qc,axis1=-2,axis2=-1)[...,None,None]*np.eye(3)/3.
        kc=np.trace(kc,axis1=-2,axis2=-1)[...,None,None]*np.eye(3)/3.
    elif role=='shape_shuffle':
        qc=qc[...,arr(state['q_permutation']).astype(np.int64),:,:]
        kc=kc[...,arr(state['k_permutation']).astype(np.int64),:,:]
    def desc(c):
        t=np.maximum(np.trace(c,axis1=-2,axis2=-1),1e-12); z=3*c/t[...,None,None]
        return np.stack((z[...,0,0],z[...,1,1],z[...,2,2],
                         np.sqrt(2)*z[...,0,1],np.sqrt(2)*z[...,0,2],np.sqrt(2)*z[...,1,2],np.log(t)),-1)
    s,t=desc(qc),desc(kc)
    return score-beta*np.sum((s[..., :,None,:]-t[...,None,:,:])**2,axis=-1)/7.


def toy_data(n=12,device='cpu',dtype=torch.float64):
    gen=torch.Generator().manual_seed(8481)
    def rand(*shape):
        return torch.randn(*shape,generator=gen,dtype=torch.float64).to(device=device,dtype=dtype)
    q,k=rand(1,n,128),rand(1,n,128)
    a,b=rand(1,n,3),rand(1,n,3)
    x,y=rand(1,n,3,3)*.1,rand(1,n,3,3)*.1
    eye=torch.eye(3,device=device,dtype=dtype)
    cov=x@x.transpose(-1,-2)+.001*eye; kcov=y@y.transpose(-1,-2)+.001*eye
    native=torch.eye(2,device=device,dtype=dtype).expand(1,n,2,2)/12
    z=torch.full((1,n),2.,device=device,dtype=dtype)
    return dict(q=q,k=k,q_anchor3=a,k_anchor3=b,q_cov3=cov,k_cov3=kcov,
        qmean=a[...,:2],kmean=b[...,:2],qcov=native,kcov=native,
        q_to_k=a[...,:2],q_to_kcov=native,q_depth=z,q_value_depth=z,k_value_depth=z,
        target=z+.2,valid=torch.ones((1,n),dtype=torch.bool,device=device),
        boundary=torch.zeros((1,n),dtype=torch.bool,device=device))


def preflight(out):
    verify(out)
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_round40_decoupled.py')
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    write(out/'preflight.json',dict(passed=result.wasSuccessful(),tests=result.testsRun,
        errors=[str(x) for x in result.errors],failures=[str(x) for x in result.failures],
        real_arrays_read=False,unix=time.time()))
    if not result.wasSuccessful():
        raise SystemExit(1)


def audit(out):
    from experiments.round40_decoupled_train import load_split
    verify(out)
    dest=out/'audit'
    if dest.exists():
        raise ValueError('preserve audit attempt')
    dest.mkdir()
    arrays,ids=load_split('val','cpu',out)
    raw={k:v.numpy() for k,v in arrays.items()}
    rows=[]; summaries={}; paired={}; metric_error=0.; numeric=[]
    for role in ROLES:
        summaries[role]={}
        for seed in (11,22,33,44,55):
            run=out/'runs'/role/str(seed); meta=read(run/'run.json')
            for name,key in (('best.pt','checkpoint_sha256'),('validation_predictions.npz','predictions_sha256'),
                    ('score_samples.npz','score_samples_sha256'),('validation.json','validation_sha256'),('history.json','history_sha256')):
                if sha(run/name)!=meta[key]:
                    raise ValueError('changed run output')
            identity=(meta['initial_shared_sha256'],meta['batch_sequence_sha256'])
            if seed in paired and paired[seed]!=identity:
                raise ValueError('unpaired initialization/batches')
            paired[seed]=identity
            history=read(run/'history.json')['validation']
            if min(history,key=lambda r:r['validation']['metrics']['AbsRel'])['step']!=meta['best_step']:
                raise ValueError('checkpoint selection mismatch')
            with np.load(run/'validation_predictions.npz',allow_pickle=False) as z:
                for field in z.files:
                    independent=numpy_macro(numpy_rows(z[field],raw['target'],raw['valid'],raw['boundary'],ids))
                    recorded=(read(run/'validation.json')['summary'] if field=='depth'
                              else meta['interventions'][field]['summary'])
                    for scene,s in independent['by_scene'].items():
                        for key,val in s.items():
                            other=recorded['by_scene'][scene][key]
                            if val is None:
                                if other is not None:raise ValueError('null mismatch')
                                continue
                            metric_error=max(metric_error,abs(val-other))
                            if not np.isclose(val,other,rtol=1e-6,atol=1e-8):
                                raise ValueError('metric mismatch '+role+'/'+key)
                            if key.endswith('count') and val!=other:
                                raise ValueError('metric count mismatch')
                    if field=='depth':
                        summaries[role][str(seed)]=independent
            state=torch.load(run/'best.pt',map_location='cpu',weights_only=True)['state_dict']
            with np.load(run/'score_samples.npz',allow_pickle=False) as z:
                subset={k:v[z['indices']] for k,v in raw.items()}
                if role in NEW_ROLES:
                    ref=numpy_logits(subset,state,role)
                    reference_p=np.exp(ref-logsumexp(ref,axis=-1,keepdims=True))
                    logit_error=float(np.max(abs(ref-z['logits'])))
                else:
                    method,h,temp=CONFIGS[role]
                    if role=='singleview':
                        subset.update(k=subset['q'],kmean=subset['qmean'],kcov=subset['qcov'],
                                      q_to_k=subset['qmean'],q_to_kcov=subset['qcov'])
                    reference_p=numpy_attention(subset,state,method,h,temp)
                    logit_error=None
                saved=z['probability']; pe=float(np.max(abs(saved-reference_p)))
                if pe>2e-5 or (logit_error is not None and logit_error>2e-4):
                    raise ValueError('score precision needs impact diagnosis '+role+str(seed)+str((logit_error,pe)))
                winners=saved.argmax(-1); refwinner=reference_p.argmax(-1)
                top=np.partition(reference_p,-2,axis=-1)[...,-2:]
                margins=top[...,-1]-top[...,-2]
                numeric.append(dict(role=role,seed=seed,logit_max_error=logit_error,p_max_error=pe,
                    winner_changes=int(np.sum(winners!=refwinner)),
                    changes_outside_2error=int(np.sum((winners!=refwinner)&(margins>2*pe)))))
            rows.append(meta)
            print('AUDIT',role,seed,flush=True)
    means={role:float(np.mean([s['metrics']['AbsRel'] for s in summary.values()])) for role,summary in summaries.items()}
    old_best=min(('point','point_pck','fixed'),key=lambda role:means[role])
    new_best=min(('center_rope','center_rbf'),key=lambda role:means[role])
    checks=[]
    for control,ratio in ((old_best,.98),(new_best,.98),('singleview',1.)):
        differences=np.array([summaries['decoupled'][str(s)]['metrics']['AbsRel']-summaries[control][str(s)]['metrics']['AbsRel'] for s in (11,22,33,44,55)])
        scene_gains={scene:float(np.mean([summaries['decoupled'][str(s)]['by_scene'][scene]['AbsRel']-summaries[control][str(s)]['by_scene'][scene]['AbsRel'] for s in (11,22,33,44,55)])) for scene in summaries[control]['11']['by_scene']}
        magnitude=means['decoupled']<means[control] if control=='singleview' else means['decoupled']<=ratio*means[control]
        rmse_c=float(np.mean([s['metrics']['RMSE'] for s in summaries[control].values()]))
        rmse_p=float(np.mean([s['metrics']['RMSE'] for s in summaries['decoupled'].values()]))
        rmse_pass=control=='singleview' or rmse_p<=1.02*rmse_c
        checks.append(dict(control=control,relative_absrel_change=means['decoupled']/means[control]-1,
            magnitude_pass=bool(magnitude),positive_seeds=int(np.sum(differences<0)),scene_differences=scene_gains,
            rmse_relative_change=rmse_p/rmse_c-1,rmse_pass=rmse_pass,
            passed=bool(magnitude and np.sum(differences<0)>=4 and all(v<0 for v in scene_gains.values()) and rmse_pass)))
    write(dest/'summary.json',dict(role_absrel_means=means,by_role_seed=summaries,checks=checks,
        preliminary_depth_screen_passed=all(x['passed'] for x in checks),goal_complete=False,
        region_center_verified=False,actual_gs_verified=False,untouched_confirmation=False))
    write(dest/'verification.json',dict(passed=True,runs=len(rows),metric_max_absolute_error=metric_error,
        score_checks=numeric,paired_initialization_and_batches=True,unix=time.time(),
        independent_formulas=True,independently_authored_by_different_agent=False))


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('preflight','audit'))
    p.add_argument('--out',type=Path,default=OUT);a=p.parse_args()
    torch.set_num_threads(4)
    if a.stage=='preflight':preflight(a.out)
    else:audit(a.out)


if __name__=='__main__':
    main()
