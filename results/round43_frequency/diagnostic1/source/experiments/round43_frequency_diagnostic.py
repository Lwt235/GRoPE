"""Fixed post-result interventions on old validation only; no retraining."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
import torch
from scipy.special import logsumexp
from experiments.round43_frequency_train import ROOT,OUT,SEEDS,checked,load,read,write,sha
from experiments import round5_train as R5
from models.round43_frequency_pe import make_model,FrequencyBank

DEST=ROOT/'results/round43_frequency/diagnostic1'
ROLES=('gaussian_full','free_frequency','fixed_gaussian','point')
SOURCES=('ROUND43_FREQUENCY_DIAGNOSTIC.md','experiments/round43_frequency_diagnostic.py')


def freeze():
    checked(OUT)
    if DEST.exists():raise ValueError('retain diagnostic attempt')
    sources={str(ROOT/p):sha(ROOT/p) for p in SOURCES}
    selected={str(OUT/'runs'/r/str(s)/'best.pt'):sha(OUT/'runs'/r/str(s)/'best.pt') for r in ROLES for s in SEEDS}
    DEST.mkdir()
    for name in sources:
        target=DEST/'source'/Path(name).relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(name,target)
    write(DEST/'lock.json',dict(sources=sources,checkpoints=selected,original_lock=sha(OUT/'lock.json'),
        roles=ROLES,seeds=SEEDS,split='val',conditions=50,real_arrays_read=False))
    (DEST/'lock.sha256').write_text(sha(DEST/'lock.json')+'\n')


@torch.no_grad()
def run():
    lock=read(DEST/'lock.json')
    if sha(DEST/'lock.json')!=(DEST/'lock.sha256').read_text().strip():raise ValueError('changed diagnostic lock')
    if sha(OUT/'lock.json')!=lock['original_lock']:raise ValueError('changed R43 lock')
    for name,digest in (lock['sources']|lock['checkpoints']).items():
        if sha(name)!=digest:raise ValueError('changed source/checkpoint')
    if (DEST/'summary.json').exists():raise ValueError('retain diagnostic result')
    data,ids=load('val',OUT,'cuda:0')
    yy,xx=np.mgrid[:18,:24];grid=np.stack((xx.ravel()*8+3.5,yy.ravel()*8+3.5),-1)
    records=[];replay=[];allrows=[]
    for role in ROLES:
        for seed in SEEDS:
            path=OUT/'runs'/role/str(seed)
            state=torch.load(path/'best.pt',weights_only=True,map_location='cpu')['state_dict']
            conditions=('normal','zero_phase','initial_bank') if role in ('gaussian_full','free_frequency') else ('normal','zero_phase')
            for condition in conditions:
                model=make_model(role).cuda();model.load_state_dict(state);model.eval()
                if condition=='zero_phase':model._position_encode=lambda content,mean,covariance:content
                elif condition=='initial_bank':model.frequency_bank=FrequencyBank('fixed_gaussian',model.frequency).cuda()
                rows=[];saved={k:[] for k in ('argmax','log_partition','label_logits')}
                for start in range(0,len(ids),8):
                    d=R5.batch(data,slice(start,start+8))
                    logits=model({k:d[k] for k in R5.FIELDS}).cpu().numpy().astype(np.float64)
                    if not np.isfinite(logits).all():raise ValueError('nonfinite logits')
                    indices=d['key_indices'].clamp_min(0).cpu().numpy()
                    weights=d['key_weights'].cpu().numpy().astype(np.float64)
                    mask=d['gt_valid'].cpu().numpy();target=d['pixcontinuous'].cpu().numpy().astype(np.float64)
                    arg=logits.argmax(-1);lp=logsumexp(logits,axis=-1)
                    gathered=np.take_along_axis(logits,indices,axis=-1)
                    ce=-((gathered-lp[...,None])*weights).sum(-1)
                    err=np.linalg.norm(grid[arg]-target,axis=-1)
                    stats=dict(n_valid=mask.sum(-1),ce_sum=np.where(mask,ce,0).sum(-1),
                        pck8_count=((err<=8)&mask).sum(-1),pck4_count=((err<=4)&mask).sum(-1),
                        epe_sum=np.where(mask,err,0).sum(-1))
                    rows.extend(dict(ids[start+i],role=role,seed=seed,condition=condition,
                        **{k:float(v[i]) for k,v in stats.items()}) for i in range(len(arg)))
                    for name,value in dict(argmax=arg,log_partition=lp,label_logits=gathered).items():saved[name].append(value)
                tag=f'{role}_{seed}_{condition}'
                np.savez_compressed(DEST/(tag+'.npz'),**{k:np.concatenate(v) for k,v in saved.items()})
                n=sum(r['n_valid'] for r in rows)
                record=dict(role=role,seed=seed,condition=condition,n_valid=n,
                    ce=sum(r['ce_sum'] for r in rows)/n,pck8=sum(r['pck8_count'] for r in rows)/n,
                    pck4=sum(r['pck4_count'] for r in rows)/n,epe=sum(r['epe_sum'] for r in rows)/n,
                    statistics_sha256=sha(DEST/(tag+'.npz')))
                if condition=='normal':
                    old=read(path/'validation.json')
                    if [(r['scene'],r['pair_id']) for r in old]!=[(r['scene'],r['pair_id']) for r in rows]:raise ValueError('population')
                    replay.append(dict(role=role,seed=seed,
                        max_ce_frame_sum_difference=max(abs(a['ce_sum']-b['ce_sum']) for a,b in zip(old,rows)),
                        pck8_count_changes=sum(a['pck8_count']!=b['pck8_count'] for a,b in zip(old,rows))))
                records.append(record);allrows.extend(rows);print(tag,record['pck8'],record['ce'],flush=True)
    means={role:{condition:{key:float(np.mean([r[key] for r in records if r['role']==role and r['condition']==condition]))
            for key in ('ce','pck8','pck4','epe')} for condition in sorted({r['condition'] for r in records if r['role']==role})} for role in ROLES}
    write(DEST/'frame_rows.json',allrows)
    write(DEST/'summary.json',dict(records=records,means=means,normal_replay=replay,conditions=len(records),
        split='old_val',pairs=len(ids),fresh_data=False,retrained=False,goal_complete=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('freeze','run'));a=p.parse_args()
    torch.set_num_threads(4);torch.set_num_interop_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    globals()[a.stage]()
