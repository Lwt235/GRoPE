"""Leave-query-out relation consistency with truth-assisted soft references."""
import argparse
import os
from pathlib import Path
import time

import numpy as np
import torch
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from experiments.round65_common import ROOT, read, write, sha
from experiments.region_pose_utility import arrays, configure_cuda
from models.round69_compact_relation import descriptors
from models.region_shape_attribution import fixed_inputs

OUT=ROOT/'results/region_relation_consistency/diagnostic1'
PLAN=ROOT/'docs/region_relation_consistency_plan.md'
DATA_LOCK=ROOT/'results/region_hrope_control/preparation/data_lock.json'
ROLES=('distance','full5','sphere5')
MODES=('fixed','calibrated')


def targets(overlap):
    valid=overlap.max(-1).values>=.3
    weights=overlap*(overlap>=.1)
    probability=weights/weights.sum(-1,keepdim=True).clamp_min(1e-12)
    return probability*valid[...,None],valid


def energy(source,target,probability):
    """[B,N,N,C], [B,M,M,C], [B,N,M] -> [B,N,M,C]."""
    a=source.permute(0,3,1,2);b=target.permute(0,3,1,2)
    p=probability[:,None];mass=probability.sum(-1)
    denominator=mass.sum(-1,keepdim=True)-mass
    diag=a.diagonal(dim1=-2,dim2=-1)
    a2=(a.square()*mass[:,None,None,:]).sum(-1)-diag.square()*mass[:,None,:]
    b2=(b.square()*probability.sum(-2)[:,None,None,:]).sum(-1)
    pb=p@b.transpose(-1,-2)
    cross=(a@p)@b.transpose(-1,-2)-diag[...,None]*pb
    numerator=a2[...,None]+b2[:,:,None,:]-(p@b.square().transpose(-1,-2))-2*cross
    result=(numerator/denominator[:,None,:,None].clamp_min(1e-12)).clamp_min(0)
    supported=denominator>1e-8
    result=result.masked_fill(~supported[:,None,:,None],0)
    return result.permute(0,2,3,1),supported


def numpy_phi(centers,covariance):
    precision=np.linalg.inv(covariance)
    delta=centers[..., :,None,:]-centers[...,None,:,:]
    d=np.linalg.norm(delta,axis=-1)/.2
    metric=np.einsum('...ijd,...ide,...ije->...ij',delta,precision,delta)
    trace=np.einsum('...ide,...jed->...ij',precision,covariance)
    return np.stack([d,np.log1p(np.maximum(metric,0)),np.log1p(np.maximum(metric.swapaxes(-1,-2),0)),
        np.log1p(np.maximum(trace,0)),np.log1p(np.maximum(trace.swapaxes(-1,-2),0))],-1)


def numpy_sphere(phi):
    d=phi[...,0];radial=np.log1p(d*d);constant=np.full_like(d,np.log(4.))
    return np.stack((d,radial,radial,constant,constant),-1)


def direct_energy(source,target,p,query):
    weights=p.copy();weights[query]=0
    if weights.sum()<=1e-8:return np.zeros((len(target),source.shape[-1]))
    difference=source[query][None,:,None,:]-target[:,None,:,:]
    return np.sum(difference**2*weights[None,:,:,None],axis=(1,2))/weights.sum()


def loss_gradient(e,t,w):
    logits=-(e*torch.as_tensor(w,device=e.device,dtype=e.dtype)).sum(-1)
    logp=logits.log_softmax(-1)
    loss=-(t*logp).sum(-1).mean()
    gradient=-torch.einsum('qkc,qk->c',e,logp.exp()-t)/len(t)
    return float(loss),gradient.cpu().numpy()


def generated_checks():
    rng=np.random.default_rng(99400)
    c=rng.normal(size=(2,7,3));z=rng.normal(size=(2,7,3,3))
    cov=z@z.swapaxes(-1,-2)+np.eye(3)*.5
    phi=descriptors('relative5',torch.from_numpy(c),torch.from_numpy(cov)).numpy()
    reference=numpy_phi(c,cov)
    formula_error=float(np.max(np.abs(phi-reference)));assert formula_error<1e-10
    p=rng.uniform(size=(7,7));p/=p.sum(-1,keepdims=True);p[5:]=0
    e,supported=energy(torch.from_numpy(phi[None,0]),torch.from_numpy(phi[None,1]),torch.from_numpy(p[None]))
    e=e.numpy()[0]
    explicit=np.stack([direct_energy(phi[0],phi[1],p,i) for i in range(7)])
    energy_error=float(np.max(np.abs(e-explicit)));assert energy_error<1e-9
    changed=p.copy();changed[1]=rng.random(7);changed[1]/=changed[1].sum()
    altered,_=energy(torch.from_numpy(phi[None,0]),torch.from_numpy(phi[None,1]),torch.from_numpy(changed[None]))
    own_error=float(np.max(np.abs(e[1]-altered.numpy()[0,1])));assert own_error<1e-9
    moved=[]
    for v in range(2):
        rotation=Rotation.from_rotvec(rng.normal(size=3)).as_matrix();shift=rng.normal(size=3)
        moved.append(descriptors('relative5',torch.from_numpy(c[v]@rotation.T+shift),
            torch.from_numpy(rotation@cov[v]@rotation.T)).numpy())
    moved_error=float(np.max(np.abs(phi-np.stack(moved))));assert moved_error<1e-9
    pa,pb=rng.permutation(7),rng.permutation(7)
    permuted,_=energy(torch.from_numpy(phi[None,0][:,pa][:,:,pa]),
        torch.from_numpy(phi[None,1][:,pb][:,:,pb]),torch.from_numpy(p[None][:,pa][:,:,pb]))
    permutation_error=float(np.max(np.abs(permuted.numpy()[0]-e[pa][:,pb])));assert permutation_error<1e-9
    toy=torch.from_numpy(rng.random((10,6,5)));t=torch.from_numpy(rng.random((10,6)));t=t/t.sum(-1,keepdim=True)
    w=np.ones(5)/5;_,grad=loss_gradient(toy,t,w);finite=[]
    for i in range(5):
        step=np.zeros(5);step[i]=1e-6
        finite.append((loss_gradient(toy,t,w+step)[0]-loss_gradient(toy,t,w-step)[0])/2e-6)
    gradient_error=float(np.max(np.abs(grad-finite)));assert gradient_error<1e-7
    return dict(passed=True,formula_error=formula_error,explicit_energy_error=energy_error,
        excluded_query_row_error=own_error,independent_view_rigid_error=moved_error,
        permutation_error=permutation_error,gradient_error=gradient_error)


def freeze():
    assert not OUT.exists()
    checks=generated_checks();data=read(DATA_LOCK)
    sources=[Path(__file__),PLAN,DATA_LOCK,ROOT/'models/round69_compact_relation.py',
        ROOT/'models/round58_support_pe.py',ROOT/'models/region_shape_attribution.py',
        ROOT/'experiments/round65_common.py',ROOT/'experiments/region_pose_utility.py',
        ROOT/'results/region_learned_valid_query/diagnostic1/review.json']
    caches={k:data['caches'][k] for k in ('fit0','eval_base')}
    for entry in caches.values():assert sha(entry['path'])==entry['sha256']
    write(OUT/'lock.json',dict(created_unix=time.time(),source_sha256={str(p):sha(p) for p in sources},
        caches=caches,roles=list(ROLES),modes=list(MODES),generated_checks=checks,
        optimizer=dict(method='L-BFGS-B',maxiter=200,maxls=40,ftol=1e-10,gtol=1e-6),
        comparisons=[['full5_calibrated','distance_calibrated'],['full5_calibrated','sphere5_calibrated'],
                     ['full5_fixed','distance_fixed'],['full5_fixed','sphere5_fixed']],
        bootstrap_seeds=[99401,99402,99403,99404],bootstrap_replicates=10000,
        truth_reference_correspondences=True,query_own_reference_row_excluded=True,
        new_transformer_training=0,new_network_inference=0,new_pose_estimation=0,
        scalar_calibration_fits=3,deployable=False,fresh_confirmation=False,goal_complete=False))
    (OUT/'lock.sha256').write_text(sha(OUT/'lock.json')+'\n')
    print('RELATION_CONSISTENCY_FROZEN',sha(OUT/'lock.json'),checks,flush=True)


def check():
    lock=read(OUT/'lock.json');assert sha(OUT/'lock.json')==(OUT/'lock.sha256').read_text().strip()
    for path,digest in lock['source_sha256'].items():assert sha(path)==digest,path
    for entry in lock['caches'].values():assert sha(entry['path'])==entry['sha256']
    return lock


def load(name,lock):
    data=arrays(lock['caches'][name]['path'],('ids','scenes','centers','covariance','overlap'))
    for key in ('centers','covariance','overlap'):data[key]=data[key].astype(np.float64)
    return data


def batch_phi(data,start,end):
    return descriptors('relative5',torch.as_tensor(data['centers'][start:end],device='cuda'),
        torch.as_tensor(data['covariance'][start:end],device='cuda'))


def scales_fit(data):
    moments={r:dict(count=0,mean=np.zeros(5),m2=np.zeros(5),minimum=np.full(5,np.inf),maximum=np.full(5,-np.inf))
        for r in ('full5','sphere5')}
    offdiag=~torch.eye(64,dtype=torch.bool,device='cuda')
    for start in range(0,len(data['ids']),32):
        full=batch_phi(data,start,start+32)
        for role,phi in [('full5',full),('sphere5',fixed_inputs(full))]:
            x=phi[:,:,offdiag].reshape(-1,5)
            count=len(x);mean=x.mean(0).cpu().numpy();m2=(x.var(0,unbiased=False)*count).cpu().numpy()
            minimum=x.min(0).values.cpu().numpy();maximum=x.max(0).values.cpu().numpy();old=moments[role]
            total=old['count']+count;delta=mean-old['mean']
            old['m2']+=m2+delta**2*old['count']*count/total
            old['mean']+=delta*count/total;old['count']=total
            old['minimum']=np.minimum(old['minimum'],minimum);old['maximum']=np.maximum(old['maximum'],maximum)
    result={}
    for role,m in moments.items():
        scale=np.sqrt(m['m2']/m['count']);constant=(m['maximum']-m['minimum'])<1e-12
        scale=np.where(constant,1.,np.maximum(scale,1e-12))
        result[role]=dict(scale=scale.tolist(),constant_columns=constant.tolist(),count=m['count'],
            mean=m['mean'].tolist(),minimum=m['minimum'].tolist(),maximum=m['maximum'].tolist())
    result['distance']={k:(v[:1] if isinstance(v,list) else v) for k,v in result['full5'].items()}
    return result


def construct(data,scales,real_check=False):
    count=len(data['ids']);output={r:np.empty((count,2,64,64,5),np.float64) for r in ('full5','sphere5')}
    supports=np.empty((count,2,64),bool);maximum=0.
    for start in range(0,count,16):
        end=min(start+16,count);full=batch_phi(data,start,end)
        overlap=torch.as_tensor(data['overlap'][start:end],device='cuda')
        for role,phi in [('full5',full),('sphere5',fixed_inputs(full))]:
            scaled=phi/torch.tensor(scales[role]['scale'],device='cuda',dtype=torch.float64)
            for view in range(2):
                ov=overlap if view==0 else overlap.transpose(-1,-2)
                p,_=targets(ov)
                values,supported=energy(scaled[:,view],scaled[:,1-view],p)
                output[role][start:end,view]=values.cpu().numpy();supports[start:end,view]=supported.cpu().numpy()
                if real_check and start==0:
                    for b in range(2):
                        np_phi=numpy_phi(data['centers'][b],data['covariance'][b])
                        if role=='sphere5':np_phi=numpy_sphere(np_phi)
                        np_phi/=np.asarray(scales[role]['scale'])
                        labels=data['overlap'][b] if view==0 else data['overlap'][b].T
                        valid=labels.max(-1)>=.3;weights=labels*(labels>=.1)
                        reference=weights/np.maximum(weights.sum(-1,keepdims=True),1e-12)*valid[:,None]
                        for q in range(4):
                            direct=direct_energy(np_phi[view],np_phi[1-view],reference,q)
                            maximum=max(maximum,float(np.max(np.abs(direct-output[role][b,view,q]))/max(1.,np.max(np.abs(direct)))))
    output['distance']=output['full5'][...,:1]
    if real_check:assert maximum<1e-9
    return output,supports,maximum


def run():
    lock=check();assert not (OUT/'execution_status.json').exists(),'Inspect any previous worker before resuming.'
    configure_cuda();state=dict(state='fit_scales',pid=os.getpid(),started_unix=time.time(),
        CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],lock_sha256=sha(OUT/'lock.json'),completed=[],goal_complete=False)
    def save():state['updated_unix']=time.time();write(OUT/'execution_status.json',state)
    save()
    try:
        fit=load('fit0',lock);scales=scales_fit(fit)
        write(OUT/'scales.json',dict(created_unix=time.time(),scales=scales,fit_cache=lock['caches']['fit0'],eval_accessed=False))
        state['state']='fit_energy';save()
        energies,support,maximum=construct(fit,scales,real_check=True)
        write(OUT/'real_formula_checks.json',dict(passed=True,relative_maximum_error=maximum,fit_pairs=2,queries_each_direction=4,
            both_full_and_sphere=True,independent_numpy_formula=True))
        over=np.stack((fit['overlap'],fit['overlap'].swapaxes(-1,-2)),1)
        t,valid=targets(torch.from_numpy(over));selected=valid.numpy()&support
        target=t.numpy()[selected];calibrations={}
        state['state']='calibrating';save()
        for role in ROLES:
            e=torch.as_tensor(energies[role][selected],device='cuda');target_tensor=torch.as_tensor(target,device='cuda')
            history=[];initial=np.ones(e.shape[-1])/e.shape[-1]
            def objective(w):
                value,gradient=loss_gradient(e,target_tensor,w)
                history.append(dict(loss=value,weights=w.tolist(),gradient=gradient.tolist()))
                return value,gradient
            options={k:lock['optimizer'][k] for k in ('maxiter','maxls','ftol','gtol')}
            result=minimize(objective,initial,method='L-BFGS-B',jac=True,bounds=[(0,None)]*len(initial),options=options)
            value,gradient=loss_gradient(e,target_tensor,result.x)
            projected=np.where((result.x<=1e-10)&(gradient>0),0,gradient)
            assert np.isfinite(result.x).all() and np.isfinite(value) and (result.x>=0).all()
            calibrations[role]=dict(weights=result.x.tolist(),success=bool(result.success),status=int(result.status),
                message=str(result.message),iterations=int(result.nit),evaluations=int(result.nfev),final_ce=value,
                initial_ce=history[0]['loss'],projected_gradient_max=float(np.max(np.abs(projected))),history=history)
            state['completed'].append(role);save();print('CONSISTENCY_SCALAR_CALIBRATED',role,'converged',bool(result.success),flush=True)
            del e,target_tensor
        write(OUT/'calibration.json',dict(completed_unix=time.time(),roles=calibrations,scales_sha256=sha(OUT/'scales.json'),
            fit_valid_queries=int(valid.sum()),fit_supported_valid_queries=int(selected.sum()),
            excluded_from_fit_only_due_to_no_reference=int((valid.numpy()&~support).sum()),eval_accessed=False))
        del energies,fit,t,valid,target
        state['state']='evaluation_energy';save()
        data=load('eval_base',lock);energies,support,_=construct(data,scales)
        over=np.stack((data['overlap'],data['overlap'].swapaxes(-1,-2)),1)
        t,valid=targets(torch.as_tensor(over,device='cuda'));payload={}
        for role in ROLES:
            e=torch.as_tensor(energies[role],device='cuda')
            for mode in MODES:
                weights=np.ones(e.shape[-1])/e.shape[-1] if mode=='fixed' else np.asarray(calibrations[role]['weights'])
                logits=-(e*torch.as_tensor(weights,device='cuda')).sum(-1)
                picked=logits.argmax(-1).cpu().numpy();picked[~support]=-1
                ce=-(t*logits.log_softmax(-1)).sum(-1)
                payload[f'{role}_{mode}']=picked.astype(np.int16)
                payload[f'{role}_{mode}_ce']=ce.cpu().numpy()
            del e,logits,ce
        np.savez(OUT/'predictions.npz',ids=data['ids'],scenes=data['scenes'],valid_queries=valid.cpu().numpy(),
            supported_queries=support,**payload)
        state.update(state='predictions_complete',finished_unix=time.time(),predictions_sha256=sha(OUT/'predictions.npz'),
            calibration_sha256=sha(OUT/'calibration.json'),peak_allocated_bytes=torch.cuda.max_memory_allocated());save()
    except BaseException as error:
        state.update(state='failed',error_type=type(error).__name__,reason=str(error));save();raise


def summarize():
    lock=check();state=read(OUT/'execution_status.json');assert state['state']=='predictions_complete'
    assert sha(OUT/'predictions.npz')==state['predictions_sha256']
    assert sha(OUT/'calibration.json')==state['calibration_sha256']
    assert not (OUT/'result.json').exists()
    data=load('eval_base',lock);keys=[f'{r}_{m}' for r in ROLES for m in MODES]
    saved=arrays(OUT/'predictions.npz',('ids','scenes','valid_queries','supported_queries',
        *keys,*(k+'_ce' for k in keys)))
    np.testing.assert_array_equal(saved['ids'],data['ids']);np.testing.assert_array_equal(saved['scenes'],data['scenes'])
    over=np.stack((data['overlap'],data['overlap'].swapaxes(-1,-2)),1)
    valid=over.max(-1)>=.3;np.testing.assert_array_equal(valid,saved['valid_queries'])
    number=valid.sum(-1,keepdims=True);expected_support=number-valid.astype(int)>0
    np.testing.assert_array_equal(expected_support,saved['supported_queries'])
    scenes=sorted(set(data['scenes'].tolist()));accuracy={};records=[]
    for key in keys:
        index=saved[key];assert ((index>=-1)&(index<64)).all()
        np.testing.assert_array_equal(index>=0,expected_support)
        picked_overlap=np.take_along_axis(over,index.clip(0)[...,None],-1)[...,0]
        correct=(picked_overlap>=.3)&valid&(index>=0)
        # Separate explicit label lookup for every query, independent of gather arithmetic.
        counted=np.zeros_like(correct)
        for b in range(480):
            for view in range(2):
                for q in range(64):
                    k=int(index[b,view,q])
                    if k>=0 and any(float(v)>=.3 for v in over[b,view,q]):
                        counted[b,view,q]=float(over[b,view,q,k])>=.3
        np.testing.assert_array_equal(correct,counted)
        by_scene={s:dict(valid=int(valid[data['scenes']==s].sum()),
            correct=int(correct[data['scenes']==s].sum()),
            region_recall=float(correct[data['scenes']==s].sum()/valid[data['scenes']==s].sum()),
            all_query_correct_fraction=float(correct[data['scenes']==s].mean()),
            ce=float(saved[key+'_ce'][data['scenes']==s][valid[data['scenes']==s]].mean())) for s in scenes}
        accuracy[key]=np.asarray([by_scene[s]['region_recall'] for s in scenes])
        records.append(dict(role=key,macro_region_recall=float(accuracy[key].mean()),
            macro_all_query_correct_fraction=float(np.mean([by_scene[s]['all_query_correct_fraction'] for s in scenes])),
            macro_ce=float(np.mean([by_scene[s]['ce'] for s in scenes])),correct_queries=int(correct.sum()),by_scene=by_scene))
    comparisons=[];maximum=0.
    for (left,right),seed in zip(lock['comparisons'],lock['bootstrap_seeds']):
        delta=100*(accuracy[left]-accuracy[right]);indices=np.random.default_rng(seed).integers(len(scenes),size=(10000,len(scenes)))
        draws=delta[indices].mean(-1);independent=100*(accuracy[left][indices].mean(-1)-accuracy[right][indices].mean(-1))
        maximum=max(maximum,float(np.max(np.abs(draws-independent))))
        comparisons.append(dict(left=left,right=right,mean_gain_pp=float(delta.mean()),by_scene_gain_pp=delta.tolist(),
            scene_interval_95_pp=np.quantile(draws,[.025,.975]).tolist(),bootstrap_seed=seed,bootstrap_replicates=10000))
    assert maximum<1e-10
    calibration=read(OUT/'calibration.json')
    write(OUT/'result.json',dict(completed_unix=time.time(),lock_sha256=sha(OUT/'lock.json'),
        calibration_sha256=sha(OUT/'calibration.json'),records=records,comparisons=comparisons,scenes=scenes,
        valid_queries=int(valid.sum()),supported_valid_queries=int((valid&expected_support).sum()),total_queries=int(valid.size),
        unsupported_valid_queries=int((valid&~expected_support).sum()),
        all_scalar_calibrations_converged=all(x['success'] for x in calibration['roles'].values()),
        scientific_review_pending=True,deployable=False,fresh_confirmation=False,goal_complete=False))
    write(OUT/'independent_checks.json',dict(passed=True,result_sha256=sha(OUT/'result.json'),
        prediction_metric_decisions_changed=0,support_and_own_query_exclusion_checked=True,
        bootstrap_maximum_error_pp=maximum,real_formula_checks_sha256=sha(OUT/'real_formula_checks.json')))
    print('RELATION_CONSISTENCY_SUMMARY_READY',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('freeze','run','summarize'))
    globals()[parser.parse_args().action]()
