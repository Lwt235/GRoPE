"""Frozen old q24->g8 diagnosis of the complete R40 selected checkpoints."""
import argparse
from pathlib import Path
import shutil
import time
import math
import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import logsumexp
from models.round40_decoupled_pe import DecoupledDepthAttention, ROLES, NEW_ROLES
from models.gaussian_quadratic_pe import controlled_features
from experiments.round6_region_data import projected_support
from experiments.round40_decoupled_data import ROOT, OUT as TRAIN, sha, read, write, verify
from experiments.round40_decoupled_audit import numpy_logits
from experiments.round7_depth_audit import numpy_attention
from models.round7_depth_attention import CONFIGS
from experiments import round20_anchor_fixed_metrics as M

OUT=ROOT/'results/round40_region_diagnostic'
ACTIVE=tuple(r for r in ROLES if r!='singleview')


def support(depth,K,size):
    z=np.nan_to_num(np.asarray(depth,np.float64),nan=10.,posinf=20.,neginf=.1).clip(.1,20.)
    inv=np.linalg.inv(K); yy,xx=np.mgrid[:144,:192]
    point=(np.stack((xx,yy,np.ones_like(xx)),-1)@inv.T)*z[...,None]
    n=(144//size)*(192//size)
    patches=point.reshape(144//size,size,192//size,size,3).transpose(0,2,1,3,4).reshape(n,size*size,3)
    zp=z.reshape(144//size,size,192//size,size).transpose(0,2,1,3).reshape(n,size,size)
    mean=patches.mean(1);delta=patches-mean[:,None]
    cov=np.einsum('nsi,nsj->nij',delta,delta)/(size*size)
    cov+=np.mean(zp*zp,(1,2))[:,None,None]*(inv[:,:2]@inv[:,:2].T)/12.
    y,x=np.mgrid[:144//size,:192//size]
    pix=np.column_stack((x.ravel()*size+(size-1)/2,y.ravel()*size+(size-1)/2))
    c=size//2; center=zp[:,c-1:c+1,c-1:c+1].mean((1,2))
    anchor=(np.column_stack((pix,np.ones(n)))@inv.T)*center[:,None]
    return anchor,cov,mean,pix


def pair_input(entry):
    with np.load(entry['input_path'],allow_pickle=False) as z:
        qd,kd,qK,kK,T=(z[n] for n in ('q_predicted_depth','k_predicted_depth','q_K','k_K','q_to_k_transform'))
    with np.load(entry['cache_path'],allow_pickle=False) as z:
        q,k=z['g24__q_features'],z['g8__k_features']
    qa,qc,qm,qp=support(qd,qK,24);ka,kc,km,kp=support(kd,kK,8)
    R=T[:3,:3]
    cross,cc=projected_support(qm,qc,T,np.eye(4),kK)
    return {key:np.asarray(value,np.float32) for key,value in dict(q=q,k=k,
        q_anchor3=qa@R.T+T[:3,3],q_cov3=R@qc@R.T,k_anchor3=ka,k_cov3=kc,
        qmean=qp/8,kmean=kp/8,qcov=np.broadcast_to(.75*np.eye(2),(48,2,2)),
        kcov=np.broadcast_to(np.eye(2)/12,(432,2,2)),q_to_k=cross,q_to_kcov=cc).items()}


def freeze(training,out):
    t=verify(training)
    if out.exists():raise ValueError('retain existing diagnostic')
    inherited=ROOT/'results/round38_rms_anchor_screen/lock.json'
    if sha(inherited)!=(inherited.parent/'lock.sha256').read_text().strip():
        raise ValueError('changed old schedule')
    old=read(inherited);entries=old['entries']
    if len(entries)!=48:raise ValueError('wrong old population')
    for e in entries:
        for field in ('input','cache','label','target'):
            if sha(e[field+'_path'])!=e[field+'_sha256']:raise ValueError('changed old data')
    sources=dict(t['source_hashes'])
    for name in ('ROUND40_REGION_DIAGNOSTIC_PROTOCOL.md','experiments/round40_region_diagnostic.py',
                 'experiments/round20_anchor_fixed_metrics.py'):
        sources[name]=sha(ROOT/name)
    checkpoints=[]
    for role in ACTIVE:
        for seed in (11,22,33,44,55):
            path=training/'runs'/role/str(seed)/'best.pt'
            run=read(path.parent/'run.json')
            if sha(path)!=run['checkpoint_sha256']:raise ValueError('changed checkpoint')
            checkpoints.append(dict(role=role,seed=seed,path=str(path),sha256=sha(path)))
    out.mkdir(parents=True)
    for name in sources:
        p=out/'source'/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,p)
    lock=dict(created_unix=time.time(),sources=sources,training_lock_sha256=sha(training/'lock.json'),
        entries=entries,checkpoints=checkpoints,partitions_path=old['partitions_path'],
        partitions_sha256=old['partitions_sha256'],labels_decoded=False)
    write(out/'lock.json',lock);(out/'lock.sha256').write_text(sha(out/'lock.json')+'\n')


def checked(out):
    lock=read(out/'lock.json')
    if sha(out/'lock.json')!=(out/'lock.sha256').read_text().strip():raise ValueError('changed lock')
    for name,digest in lock['sources'].items():
        if sha(ROOT/name)!=digest:raise ValueError('changed source')
    return lock


@torch.no_grad()
def forward(out):
    lock=checked(out);dest=out/'forward'
    if dest.exists():raise ValueError('retain forward attempt')
    dest.mkdir(); records=[];started=time.time()
    all_data={}
    for e in lock['entries']:
        for field in ('input','cache'):
            if sha(e[field+'_path'])!=e[field+'_sha256']:raise ValueError('changed observations')
        all_data[e['pair_id']]=pair_input(e)
    for record in lock['checkpoints']:
        if sha(record['path'])!=record['sha256']:raise ValueError('changed model')
        model=DecoupledDepthAttention(record['role']).cuda()
        model.load_state_dict(torch.load(record['path'],map_location='cuda:0',weights_only=True)['state_dict'])
        if record['role']=='shape_shuffle':
            model.q_permutation=torch.from_numpy(np.random.default_rng(20261101).permutation(48)).cuda()
        probabilities=[]; max_norm=0.; numeric=None
        for idx,e in enumerate(lock['entries']):
            raw=all_data[e['pair_id']];d={k:torch.from_numpy(v[None]).cuda() for k,v in raw.items()}
            if record['role'] in NEW_ROLES:
                logits=model.attention_logits(d)
            else:
                qg,kg=controlled_features(d,model.method,model.bandwidth)
                q=F.normalize(model.qk(d['q']),dim=-1);k=F.normalize(model.qk(d['k']),dim=-1)
                logits=model.temperature*(q@k.transpose(-1,-2))+qg@kg.transpose(-1,-2)
            p=logits.softmax(-1)[0].cpu().numpy()
            max_norm=max(max_norm,float(abs(p.astype(np.float64).sum(-1)-1).max()))
            if not np.isfinite(p).all() or max_norm>2e-6:raise ValueError('invalid probability')
            if idx==0:
                subset={k:v[None] for k,v in raw.items()}
                if record['role'] in NEW_ROLES:
                    ref=numpy_logits(subset,model.state_dict(),record['role'])[0]
                    refp=np.exp(ref-logsumexp(ref,axis=-1,keepdims=True))
                else:
                    method,h,temp=CONFIGS[record['role']]
                    refp=numpy_attention(subset,model.state_dict(),method,h,temp)[0]
                err=float(abs(refp-p).max()); margin=np.diff(np.sort(refp,axis=-1)[:,-2:],axis=-1)[:,0]
                changed=refp.argmax(-1)!=p.argmax(-1)
                numeric=dict(max_p_error=err,winner_changes=int(changed.sum()),
                    changes_outside_2error=int(np.sum(changed&(margin>2*err))),warning=err>2e-5)
            probabilities.append(p)
        path=dest/f"{record['role']}_{record['seed']}.npz"
        bank=np.stack(probabilities)
        np.savez_compressed(path,p=bank,winner=bank.argmax(-1).astype(np.int32))
        records.append(dict(**record,output=str(path),output_sha256=sha(path),
                            max_normalization_error=max_norm,numeric_check=numeric))
        write(dest/'records.json',records)
        print('REGION FORWARD',record['role'],record['seed'],flush=True)
    write(dest/'completion.json',dict(records=records,pairs=48,queries=2304,roles=12,seeds=5,
        seconds=time.time()-started,labels_decoded=False))


def metrics(out):
    lock=checked(out);complete=read(out/'forward/completion.json');dest=out/'metrics'
    if dest.exists():raise ValueError('retain metrics attempt')
    dest.mkdir()
    if len(complete['records'])!=60:raise ValueError('incomplete forward')
    if sha(lock['partitions_path'])!=lock['partitions_sha256']:raise ValueError('changed partition')
    with np.load(lock['partitions_path'],allow_pickle=False) as z:
        parts={g:{n:z[g+'__'+n] for n in ('bounds','owner')} for g in ('g8','g16','g24')}
    targets=[]
    for e in lock['entries']:
        for field in ('label','target'):
            if sha(e[field+'_path'])!=e[field+'_sha256']:raise ValueError('changed labels')
        with np.load(e['label_path'],allow_pickle=False) as z:
            labels={n:z[n] for n in ('gt_xy','gt_visible')}
        counts,_=M.target_counts(labels,parts,'g24','g8')
        with np.load(e['target_path'],allow_pickle=False) as z:
            xy,valid=z['g24_point_xy'],z['g24_point_visible']
        centers=(parts['g8']['bounds'][:,:2]+parts['g8']['bounds'][:,2:]-1)/2
        allowed=np.sum((xy[:,None,:]-centers[None])**2,axis=-1)<=64
        joint=valid&(counts.sum(-1)>0)
        conflict=joint&(np.max(np.where(allowed,counts,-1),axis=-1)<counts.max(-1))
        targets.append((counts,xy,valid,joint,conflict))
    summary={};max_replay=0.
    for rec in complete['records']:
        if sha(rec['output'])!=rec['output_sha256']:raise ValueError('changed prediction')
        with np.load(rec['output'],allow_pickle=False) as z:
            probabilities=z['p'];wins=z['winner']
        rows=[]
        for i,e in enumerate(lock['entries']):
            p=probabilities[i].astype(np.float64); counts,xy,valid,joint,conflict=targets[i]
            kl,js,mass=M.divergence(p,counts)
            center=M.center_metrics(wins[i],parts['g8']['bounds'],xy,valid)
            # Separate scalar-loop KL/center replay; no call to producer metric.
            replay=0.
            for q in range(48):
                total=int(counts[q].sum())
                if total:
                    replay+=math.fsum(float(c)*(math.log(max(float(c)/total,1e-15))-math.log(max(float(p[q,k]),1e-15))) for k,c in enumerate(counts[q]) if c)
            max_replay=max(max_replay,abs(kl-replay))
            if not math.isclose(kl,replay,rel_tol=1e-6,abs_tol=1e-8):raise ValueError('KL replay')
            selected=(parts['g8']['bounds'][wins[i],:2]+parts['g8']['bounds'][wins[i],2:]-1)/2
            hit=np.sum((selected-xy)**2,axis=-1)<=64
            if int(np.sum(hit&valid))!=center['pck_count']:raise ValueError('PCK replay')
            groups={}
            for group,mask in (('conflict',conflict),('consistent',joint&~conflict),('nonjoint',~joint)):
                gkl,_,gmass=M.divergence(p[mask],counts[mask])
                groups[group]=dict(queries=int(mask.sum()),mass=gmass,kl_sum=gkl,
                    center_count=int(np.sum(mask&valid)),center_hits=int(np.sum(mask&valid&hit)))
            rows.append(dict(scene=e['scene'],pair=e['pair'],kl_sum=kl,js_sum=js,mass=mass,center=center,groups=groups))
        scenes={}
        for scene in M.SCENES:
            chosen=[r for r in rows if r['scene']==scene];mass=sum(r['mass'] for r in chosen)
            n=sum(r['center']['n_point'] for r in chosen);hit=sum(r['center']['pck_count'] for r in chosen)
            groups={}
            for group in ('conflict','consistent','nonjoint'):
                groups[group]={key:sum(r['groups'][group][key] for r in chosen) for key in ('queries','mass','kl_sum','center_count','center_hits')}
            scenes[scene]=dict(kl=sum(r['kl_sum'] for r in chosen)/mass,pck=hit/n,count=n,hits=hit,
                rmse=math.sqrt(sum(r['center']['distance_squared_sum'] for r in chosen)/n),groups=groups)
        if [scenes[s]['count'] for s in M.SCENES]!=[382,402,312]:raise ValueError('center population')
        key=rec['role']+'_'+str(rec['seed'])
        summary[key]=dict(by_scene=scenes,macro={m:float(np.mean([s[m] for s in scenes.values()])) for m in ('kl','pck','rmse')})
        write(dest/(key+'_rows.json'),rows)
    write(dest/'summary.json',summary)
    write(dest/'verification.json',dict(passed=True,max_kl_replay_error=max_replay,models=60,
        queries_per_model=2304,center_labels_per_model=1096,goal_complete=False))


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('freeze','forward','metrics'))
    p.add_argument('--training',type=Path,default=TRAIN/'attempt1');p.add_argument('--out',type=Path,default=OUT)
    a=p.parse_args();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    if a.stage=='freeze':freeze(a.training,a.out)
    elif a.stage=='forward':forward(a.out)
    else:metrics(a.out)


if __name__=='__main__':main()
