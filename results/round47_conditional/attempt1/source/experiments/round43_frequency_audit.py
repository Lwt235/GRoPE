"""Separate NumPy PE formulas and complete old-evaluation metric replay."""
import argparse
from pathlib import Path
import time
import numpy as np
import torch
from scipy.special import logsumexp
from models.round5_cross_attention import METHODS
from models.round43_frequency_pe import NEW_ROLES,ROLES
from experiments.round43_frequency_train import ROOT,OUT,SEEDS,read,write,sha,checked


def array(x):return x.detach().cpu().numpy().astype(np.float64) if isinstance(x,torch.Tensor) else np.asarray(x,np.float64)


def numpy_frequency(role,state):
    if role not in NEW_ROLES:return array(state['frequency'])
    s={k[len('frequency_bank.'):]:array(v) for k,v in state.items() if k.startswith('frequency_bank.')}
    e=s['template'];sigma=s['sigma'];base=e*sigma[:,None,None];g,m,d=e.shape
    if role=='fixed_gaussian':return base.reshape(-1,d)
    if role=='free_frequency':return (base+s['offset']*sigma[:,None,None]).reshape(-1,d)
    if role=='shared_affine':return (base@(np.eye(d)+s['matrix_offset']).T+s['mean_offset']*np.sqrt(np.mean(sigma**2))).reshape(-1,d)
    values=[]
    for index in range(g):
        if role=='group_affine':L=np.eye(d)+s['matrix_offset'][index]
        else:
            L=np.diag(np.broadcast_to(np.exp(s['log_scale'][index]),(d,)))
            if role=='gaussian_full':L[np.tril_indices(d,-1)]=s['lower'][index]
        values.append(sigma[index]*(e[index]@L.T+s['mean_offset'][index]))
    return np.concatenate(values)


def check_sample(role,state,path):
    with np.load(path,allow_pickle=False) as archive:z={k:archive[k] for k in archive.files}
    freq=numpy_frequency(role,state)
    def encode(content,mean,cov):
        c=array(content);phase=array(mean)@freq.T
        value=(c[...,::2]+1j*c[...,1::2])*np.exp(1j*phase)[None]
        if role in NEW_ROLES or role=='point':return value
        cov=array(cov)
        if role=='fixed':cov=np.broadcast_to(np.eye(2)/12,cov.shape).copy()
        elif role=='isotropic':cov=np.trace(cov,axis1=-2,axis2=-1)[:,None,None]*np.eye(2)/2
        cov=cov+float(array(state['resolution']))**2*np.eye(2)
        variance=np.maximum(np.einsum('fi,nij,fj->nf',freq,cov,freq),0)
        if role=='expected':return value*np.exp(-.5*variance)[None]
        amp=-np.log1p(.5*variance) if role=='resolvent' else -.5*variance
        value*=np.exp(amp-amp.max(-1,keepdims=True))[None]
        return value/np.maximum(np.sqrt(np.sum(abs(value)**2,-1,keepdims=True)),1e-300)
    q=encode(z['q'],z['qmean'],z['qcov']);k=encode(z['k'],z['kmean'],z['kcov'])
    temp=np.exp(min(float(array(state['layers.0.log_temperature'])),np.log(60)))
    score=temp*(q@k.conj().swapaxes(-1,-2)).real
    p=np.exp(score-logsumexp(score,axis=-1,keepdims=True));old=z['probability']
    error=float(abs(p-old).max());winner=int(np.sum(p.argmax(-1)!=old.argmax(-1)))
    return dict(p_max_error=error,winner_changes=winner,warning=bool(error>5e-5),
        scope='first validation pair, first attention layer, eight query rows/all keys/two heads')


def training(out):
    checked(out);dest=out/'audit_training'
    for seed in SEEDS:
        if not read(out/f'seed_{seed}_complete.json')['completed']:raise ValueError('all seeds required')
    if dest.exists():raise ValueError('retain audit')
    dest.mkdir();rows=[];identities={};warnings=[]
    for role in ROLES:
        for seed in SEEDS:
            path=out/'runs'/role/str(seed);record=read(path/'run.json')
            for name,digest in record['hashes'].items():
                if sha(path/name)!=digest:raise ValueError('changed training output')
            identity=(record['initial_shared_sha256'],record['batch_sha256'])
            if seed in identities and identity!=identities[seed]:raise ValueError('unpaired training')
            identities[seed]=identity
            h=read(path/'history.json')['validation']
            if [x['step'] for x in h]!=list(range(0,1001,100)) or min(h,key=lambda x:x['ce'])['step']!=record['best_step']:
                raise ValueError('selection mismatch')
            state=torch.load(path/'best.pt',weights_only=True,map_location='cpu')['state_dict']
            numeric=check_sample(role,state,path/'score_sample.npz')
            if numeric['warning']:warnings.append(dict(role=role,seed=seed,**numeric))
            rows.append(dict(role=role,seed=seed,run_sha256=sha(path/'run.json'),checkpoint_sha256=sha(path/'best.pt'),numeric=numeric))
            print('AUDIT',role,seed,numeric['p_max_error'],flush=True)
    write(dest/'completion.json',dict(records=rows,warnings=warnings,paired=True,
        old_eval_read=False,independent_PE_formulas=True,independently_authored=False,unix=time.time()))


def metrics(out):
    lock=checked(out);seal=read(out/'audit_training/completion.json');dest=out/'metrics'
    if dest.exists():raise ValueError('retain metrics')
    for seed in SEEDS:
        if not read(out/f'export_{seed}_complete.json')['completed']:raise ValueError('all exports required')
    labels={k:[] for k in ('gt_valid','pixcontinuous','key_weights')};ids=[]
    for entry in lock['entries']:
        if entry['split']!='test':continue
        path=Path(entry['folder'])/'labels.npz'
        if sha(path)!=entry['hashes'][str(path)]:raise ValueError('changed old labels')
        with np.load(path,allow_pickle=False) as z:
            for k in labels:labels[k].append(z[k])
        ids.extend(dict(scene=entry['scene'],pair_id=f'{q}:{k}',q_index=q,k_index=k) for q,k in entry['pairs'])
    labels={k:np.concatenate(v) for k,v in labels.items()};mask=labels['gt_valid'].astype(bool)
    yy,xx=np.mgrid[:18,:24];grid=np.stack((xx.ravel()*8+3.5,yy.ravel()*8+3.5),-1)
    allrows=[];max_errors={};count_changes={};sources={}
    for rec in seal['records']:
        role,seed=rec['role'],rec['seed'];path=out/'runs'/role/str(seed)
        saved=read(path/'old_eval.json')
        if saved['checkpoint_sha256']!=rec['checkpoint_sha256'] or sha(path/'old_eval_statistics.npz')!=saved['statistics_sha256']:
            raise ValueError('changed evaluation artifacts')
        sources[str(path/'old_eval.json')]=sha(path/'old_eval.json')
        with np.load(path/'old_eval_statistics.npz',allow_pickle=False) as z:
            for variant in ('primary','zero_v'):
                arg=z[variant+'_argmax'];soft=z[variant+'_soft_xy'].astype(np.float64)
                lp=z[variant+'_label_logits'].astype(np.float64)-z[variant+'_log_partition'].astype(np.float64)[...,None]
                ce=-(lp*labels['key_weights']).sum(-1)
                error=np.linalg.norm(grid[arg]-labels['pixcontinuous'],axis=-1)
                se=np.linalg.norm(soft-labels['pixcontinuous'],axis=-1)
                stats=dict(n_valid=mask.sum(-1),ce_sum=np.where(mask,ce,0).sum(-1),
                    pck8_count=((error<=8)&mask).sum(-1),pck4_count=((error<=4)&mask).sum(-1),
                    epe_argmax_sum=np.where(mask,error,0).sum(-1),epe_soft_sum=np.where(mask,se,0).sum(-1))
                old=[r for r in saved['rows'] if r['variant']==variant]
                if [{k:r[k] for k in ids[0]} for r in old]!=ids:raise ValueError('evaluation population mismatch')
                if arg.shape!=mask.shape or not all(np.isfinite(v).all() for v in stats.values()):raise ValueError('bad full statistics')
                for key,values in stats.items():
                    diff=values-np.asarray([r[key] for r in old]);max_errors[key]=max(max_errors.get(key,0.),float(abs(diff).max()))
                    if key in ('n_valid','pck8_count','pck4_count'):count_changes[key]=count_changes.get(key,0)+int(np.count_nonzero(diff))
                allrows.extend(dict(ids[i],role=role,seed=seed,variant=variant,**{k:float(v[i]) for k,v in stats.items()}) for i in range(len(ids)))
    def summarize(role,seed,variant='primary'):
        scene={}
        for name in sorted({r['scene'] for r in ids}):
            rows=[r for r in allrows if (r['role'],r['seed'],r['variant'],r['scene'])==(role,seed,variant,name)]
            if len(rows)!=96:raise ValueError('all96 old pairs required')
            n=sum(r['n_valid'] for r in rows)
            scene[name]={name_:sum(r[key] for r in rows)/n for name_,key in
                (('pck8','pck8_count'),('pck4','pck4_count'),('ce','ce_sum'),('epe','epe_argmax_sum'),('soft_epe','epe_soft_sum'))}
        return dict(by_scene=scene,macro={k:float(np.mean([v[k] for v in scene.values()])) for k in next(iter(scene.values()))})
    table={r:{str(seed):summarize(r,seed) for seed in SEEDS} for r in ROLES}
    means={r:{k:float(np.mean([v['macro'][k] for v in entries.values()])) for k in next(iter(entries.values()))['macro']} for r,entries in table.items()}
    strongest=max(METHODS,key=lambda r:means[r]['pck8']);checks=[]
    for control in (strongest,'fixed_gaussian','gaussian_iso','group_affine','free_frequency'):
        delta=means['gaussian_full']['pck8']-means[control]['pck8'];ce=means['gaussian_full']['ce']/means[control]['ce']-1
        wins=sum(table['gaussian_full'][str(s)]['macro']['pck8']>table[control][str(s)]['macro']['pck8'] for s in SEEDS)
        scene={name:float(np.mean([table['gaussian_full'][str(s)]['by_scene'][name]['pck8']-table[control][str(s)]['by_scene'][name]['pck8'] for s in SEEDS])) for name in next(iter(table[control].values()))['by_scene']}
        checks.append(dict(control=control,pck8_difference_pp=100*delta,ce_relative_change=ce,winning_seeds=wins,
            scene_differences=scene,passed=bool(delta>=.01 and wins>=4 and sum(v>0 for v in scene.values())>=4 and ce<=.02)))
    dest.mkdir();write(dest/'frame_rows.json',allrows)
    write(dest/'summary.json',dict(means=means,by_role_seed=table,checks=checks,
        zero_v={r:{str(seed):summarize(r,seed,'zero_v') for seed in SEEDS} for r in ROLES},
        promising_development_only=all(c['passed'] for c in checks),fresh_confirmation=False,goal_complete=False))
    write(dest/'verification.json',dict(rows=len(allrows),pairs_per_run=len(ids),valid_queries_per_run=int(mask.sum()),
        max_gpu_numpy_frame_sum_difference=max_errors,count_changed_frames=count_changes,
        sampled_PE_warnings=seal['warnings'],input_hashes=sources,
        scope='all-query metric replay from saved sufficient statistics; sampled PE score audit, not full independent network replay'))


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('training','metrics'));p.add_argument('--out',type=Path,default=OUT);a=p.parse_args()
    torch.set_num_threads(4);globals()[a.stage](a.out)


if __name__=='__main__':main()
