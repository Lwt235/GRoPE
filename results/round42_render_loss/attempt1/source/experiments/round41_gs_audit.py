"""Separate NumPy score route and full saved-image metric replay, same author."""
import argparse
from pathlib import Path
import time
import unittest
import numpy as np
import torch
from scipy.special import logsumexp
from models.round41_gs_readout import GSReadout,ROLES,NEW_ROLES,SHAPE_ROLES
from experiments.round41_gs_data import ROOT,OUT,SCENES,SEEDS,checked,load_scene,query,read,write,sha


def arr(x):return x.detach().cpu().numpy().astype(np.float64) if isinstance(x,torch.Tensor) else np.asarray(x,np.float64)


def numpy_logits(q,k,state,role):
    W=arr(state['qk.weight']);freq=arr(state['frequencies'])
    def encoded(d):
        v=arr(d['feature'])@W.T;v/=np.maximum(np.linalg.norm(v,axis=-1,keepdims=True),1e-12)
        if role not in NEW_ROLES:return v
        z=(v[:,::2]+1j*v[:,1::2])*np.exp(1j*(arr(d['anchor'])@freq.T))
        if role in ('expected_rope','normalized_rope'):
            z*=np.exp(-.5*np.maximum(np.einsum('nij,fi,fj->nf',arr(d['cov']),freq,freq),0))
        if role=='normalized_rope':z/=np.maximum(np.sqrt(np.sum(abs(z)**2,-1,keepdims=True)),1e-12)
        return z
    score=10*(encoded(q)@encoded(k).conj().T).real
    if role in SHAPE_ROLES or role=='center_rbf':
        beta=float(np.logaddexp(0,arr(state['strength_raw'])))
        if role=='center_rbf':return score-beta*np.sum((arr(q['anchor'])[:,None]-arr(k['anchor'])[None])**2,-1)
        qc,kc=arr(q['cov']),arr(k['cov'])
        if role=='shape_shuffle':qc,kc=arr(q['shuffle_cov']),kc[arr(state['k_perm']).astype(int)]
        if role=='shape_isotropic':
            qc=np.trace(qc,axis1=-2,axis2=-1)[:,None,None]*np.eye(3)/3
            kc=np.trace(kc,axis1=-2,axis2=-1)[:,None,None]*np.eye(3)/3
        def desc(c):
            t=np.maximum(np.trace(c,axis1=-2,axis2=-1),1e-12);c=3*c/t[:,None,None]
            return np.column_stack((c[:,0,0],c[:,1,1],c[:,2,2],np.sqrt(2)*c[:,0,1],np.sqrt(2)*c[:,0,2],np.sqrt(2)*c[:,1,2],np.log(t)))
        return score-beta*np.sum((desc(qc)[:,None]-desc(kc)[None])**2,-1)/7
    if role in NEW_ROLES:return score
    cov=arr(q['projected_cov'])
    if role=='point':cov=np.zeros_like(cov)
    elif role=='fixed':cov=np.broadcast_to(np.eye(2),cov.shape)
    elif role=='isotropic':cov=np.trace(cov,axis1=-2,axis2=-1)[:,None,None]*np.eye(2)/2
    delta=arr(q['xy'])[:,None]-arr(k['xy'])[None]
    for i in range(len(cov)):
        solution=np.linalg.solve(cov[i]+(1/12+.25)*np.eye(2),delta[i].T).T
        score[i]-=.5*np.sum(solution*delta[i],-1)
    return score


def toy(n=12,m=17,dtype=torch.float64,device='cpu'):
    rng=np.random.default_rng(149)
    def tensor(x):return torch.as_tensor(x,dtype=dtype,device=device)
    def part(count):
        c=rng.normal(size=(count,3,3))*.1;cov=c@c.transpose(0,2,1)+np.eye(3)*.001
        return dict(feature=tensor(rng.normal(size=(count,128))),anchor=tensor(rng.normal(size=(count,3))),
            cov=tensor(cov),xy=tensor(rng.uniform(0,5,(count,2))),rgb=tensor(rng.uniform(0,1,(count,3))))
    q,k=part(n),part(m);q['projected_cov']=tensor(np.broadcast_to(np.eye(2)*.2,(n,2,2)).copy())
    q['shuffle_cov']=q['cov'][torch.from_numpy(np.random.default_rng(20261101).permutation(n)).to(device)]
    return q,k


def preflight(out):
    checked(out)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_round41_gs.py'))
    write(out/'preflight.json',dict(passed=result.wasSuccessful(),tests=result.testsRun,real_arrays_read=False))
    if not result.wasSuccessful():raise SystemExit(1)


def smoke(out):
    checked(out);rows=[];q,k=toy(64,512,torch.float32,'cuda:0')
    for role in ROLES:
        torch.manual_seed(17);model=GSReadout(role,64,512).cuda()
        opt=torch.optim.AdamW(model.parameters(),lr=.001)
        for _ in range(3):
            opt.zero_grad(set_to_none=True);loss=(model(q,k)-q['rgb']).square().mean();loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
        rows.append(dict(role=role,loss=float(loss.detach()),qk_gradient=float(model.qk.weight.grad.norm())))
    if not all(np.isfinite(r['loss']) and r['qk_gradient']>0 for r in rows):raise ValueError('CUDA smoke failed')
    write(out/'smoke.json',dict(passed=True,rows=rows,gpu=torch.cuda.get_device_name(),torch=torch.__version__,real_arrays_read=False))
    print('SMOKE',rows,flush=True)


@torch.no_grad()
def audit(out):
    checked(out);dest=out/'audit'
    if dest.exists():raise ValueError('retain audit attempt')
    dest.mkdir();records=[];pairing={};warnings=[]
    for scene in SCENES:
        data=load_scene(scene,out,device='cpu')
        for role in ROLES:
            for seed in SEEDS:
                path=out/'runs'/scene/role/str(seed);meta=read(path/'run.json')
                for name,key in (('best.pt','checkpoint_sha256'),('history.json','history_sha256'),
                        ('predictions.npz','predictions_sha256'),('colors.npz','colors_sha256'),('score_samples.npz','samples_sha256')):
                    if sha(path/name)!=meta[key]:raise ValueError('changed run '+str(path/name))
                pair=(meta['initial_qk_sha256'],meta['batch_sha256']);pairkey=(scene,seed)
                if pairkey in pairing and pairing[pairkey]!=pair:raise ValueError('unpaired training')
                pairing[pairkey]=pair
                if min(read(path/'history.json')['validation'],key=lambda x:x['mse'])['step']!=meta['best_step']:
                    raise ValueError('wrong selected checkpoint')
                state=torch.load(path/'best.pt',map_location='cpu',weights_only=True)['state_dict']
                model=GSReadout(role);model.load_state_dict(state)
                with np.load(path/'score_samples.npz',allow_pickle=False) as z:
                    k=int(z['view']);ids=torch.from_numpy(z['primitive_ids']);p=z['probability'];savedrgb=z['rgb']
                q=query(data,k,ids,model);key=data['views'][k]
                ref=numpy_logits(q,key,state,role);refp=np.exp(ref-logsumexp(ref,axis=-1,keepdims=True))
                pe=float(abs(refp-p).max());rgb_error=float(abs(refp@arr(key['rgb'])-savedrgb).max())
                numeric=dict(p_max_error=pe,rgb_max_error=rgb_error,winner_changes=int(np.sum(p.argmax(-1)!=refp.argmax(-1))))
                if pe>5e-5 or rgb_error>2e-5:warnings.append(dict(scene=scene,role=role,seed=seed,**numeric))
                with np.load(path/'colors.npz',allow_pickle=False) as z:
                    if not np.array_equal(z['counts'],data['all_count'].numpy()):raise ValueError('visibility differs')
                    if not np.isfinite(z['colors']).all() or z['colors'].min()<0 or z['colors'].max()>1.000001:raise ValueError('invalid colors')
                with np.load(path/'predictions.npz',allow_pickle=False) as z:
                    frames=read(ROOT/'results/round9_context_data'/scene/'target_cameras.json')['frames']
                    if z['target_ids'].tolist()!=[f['id'] for f in frames] or z['target_splits'].tolist()!=[f['split'] for f in frames]:raise ValueError('target identity')
                    if z['predictions'].shape!=(len(meta['variants']),8,144,192,3) or not np.isfinite(z['predictions']).all():raise ValueError('invalid prediction')
                records.append(dict(path=str(path),scene=scene,role=role,seed=seed,run_sha256=sha(path/'run.json'),numeric=numeric))
                print('AUDIT',scene,role,seed,flush=True)
    if len(records)!=110:raise ValueError('incomplete full comparison')
    write(dest/'completion.json',dict(records=records,numeric_warnings=warnings,paired=True,target_rgb_read=False,
        independent_formulas=True,independent_author=False,unix=time.time()))


def metrics(out):
    from experiments.round9_analysis import read_target,frame_metrics
    checked(out);audit=read(out/'audit/completion.json');dest=out/'metrics'
    if dest.exists():raise ValueError('retain metrics attempt')
    if len(audit['records'])!=110:raise ValueError('all forwards must be sealed')
    dest.mkdir();truths={};frames={};hashes={};rows=[];max_replay=0.
    for scene in SCENES:
        frames[scene]=read(ROOT/'results/round9_context_data'/scene/'target_cameras.json')['frames']
        truths[scene]=[]
        for f in frames[scene]:
            truth,digest=read_target(f);truths[scene].append(truth);hashes[f['image_path']]=digest
    def append(scene,role,seed,variant,predictions):
        nonlocal max_replay
        for i,(pred,truth) in enumerate(zip(predictions,truths[scene])):
            m=frame_metrics(pred,truth);delta=(pred.astype(np.float64)-truth).ravel()
            replay=float(np.dot(delta,delta)/delta.size);max_replay=max(max_replay,abs(replay-m['mse']))
            if not np.isclose(replay,m['mse'],rtol=1e-10,atol=1e-12):raise ValueError('metric replay')
            rows.append(dict(scene=scene,role=role,seed=seed,variant=variant,
                frame_id=frames[scene][i]['id'],split=frames[scene][i]['split'],**m))
    for rec in audit['records']:
        path=Path(rec['path']);meta=read(path/'run.json')
        if sha(path/'run.json')!=rec['run_sha256'] or sha(path/'predictions.npz')!=meta['predictions_sha256']:raise ValueError('changed sealed predictions')
        with np.load(path/'predictions.npz',allow_pickle=False) as z:
            for variant,pred in zip(z['variants'].tolist(),z['predictions']):append(rec['scene'],rec['role'],rec['seed'],variant,pred)
    for scene in SCENES:
        base=ROOT/'results/round9_gs_readout'/scene
        if sha(base/'predictions.npz')!=read(base/'run.json')['prediction_sha256']:raise ValueError('changed R9 baseline')
        with np.load(base/'predictions.npz',allow_pickle=False) as z:
            if z['target_ids'].tolist()!=[f['id'] for f in frames[scene]]:raise ValueError('R9 target identity')
            for method,pred in zip(z['methods'].tolist(),z['predictions']):append(scene,'r9_'+method,0,'primary',pred)
    def summarize(role,seed,variant='primary'):
        by_scene={}
        for scene in SCENES:
            selected=[r for r in rows if (r['role'],r['seed'],r['variant'],r['scene'])==(role,seed,variant,scene)]
            if len(selected)!=8:raise ValueError('full target population required')
            by_scene[scene]={split:float(np.mean([r['mse'] for r in selected if split=='all' or r['split']==split])) for split in ('calibration','gate','all')}
        return dict(by_scene=by_scene,macro={s:float(np.mean([x[s] for x in by_scene.values()])) for s in ('calibration','gate','all')})
    summary={role:{str(seed):summarize(role,seed) for seed in SEEDS} for role in ROLES}
    external={role:summarize(role,0) for role in sorted({r['role'] for r in rows if r['role'].startswith('r9_')})}
    means={role:float(np.mean([r['macro']['gate'] for r in values.values()])) for role,values in summary.items()}
    learned=min(('point','fixed','center_rope','center_rbf'),key=lambda r:means[r])
    ext=min([r for r in external if r.startswith(('r9_point_','r9_fixed_')) or r=='r9_bilinear'],key=lambda r:external[r]['macro']['gate'])
    checks=[]
    for control,required in ((learned,.02),(ext,.02),('r9_original_gs',0.)):
        isext=control in external;cm=external[control]['macro']['gate'] if isext else means[control]
        win=sum(summary['decoupled'][str(seed)]['macro']['gate']<(external[control] if isext else summary[control][str(seed)])['macro']['gate'] for seed in SEEDS)
        scene_delta={scene:float(np.mean([summary['decoupled'][str(seed)]['by_scene'][scene]['gate']-(external[control] if isext else summary[control][str(seed)])['by_scene'][scene]['gate'] for seed in SEEDS])) for scene in SCENES}
        relative=means['decoupled']/cm-1
        passed=(relative<=-required if required else relative<0) and win>=4 and all(v<0 for v in scene_delta.values())
        checks.append(dict(control=control,relative_mse_change=relative,winning_seeds=win,scene_differences=scene_delta,passed=bool(passed)))
    interventions={v:{str(seed):summarize('decoupled',seed,v) for seed in SEEDS} for v in ('reverse_v','zero_shape')}
    write(dest/'frame_metrics.json',rows)
    write(dest/'summary.json',dict(by_role_seed=summary,external=external,gate_mse_means=means,checks=checks,
        preliminary_gs_passed=all(c['passed'] for c in checks),interventions=interventions,goal_complete=False,
        fresh_confirmation=False,geometry_replicates=1,training_seeds=5,target_image_sha256=hashes))
    write(dest/'verification.json',dict(frame_rows=len(rows),max_mse_replay_error=max_replay,full_pixels_per_frame=27648,
        audit_sha256=sha(out/'audit/completion.json'),metric_replay_passed=True,numeric_warnings=audit['numeric_warnings']))


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('preflight','smoke','audit','metrics'))
    p.add_argument('--out',type=Path,default=OUT/'attempt1');a=p.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    globals()[a.stage](a.out)


if __name__=='__main__':main()
