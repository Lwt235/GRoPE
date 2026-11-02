"""Independent direct pair-energy check and all-query metric replay for R47."""
import argparse
from pathlib import Path
import time
import numpy as np
import torch
from scipy.special import logsumexp
from models.round47_conditional_gaussian import ROLES as NEW_ROLES, split_role
from experiments.round47_conditional_train import ROOT,OUT,SEEDS,read,write,sha,checked,CONTROL_ALIAS
from experiments.round43_frequency_audit import array
ROLES=NEW_ROLES+tuple(CONTROL_ALIAS.values())


def numpy_energy(role,state,features,query_mean,key_mean):
    kind,dim=split_role(role)
    nq,nk=len(query_mean),len(key_mean)
    result=np.zeros((2,nq,nk),dtype=np.float64)
    if kind=='nope':return result
    size=dim+(1 if kind=='isotropic' else dim)+(dim*(dim-1)//2 if kind in ('conditional','static','free_quadratic') else 0)
    if kind=='fixed':raw=np.zeros((2,nq,size))
    elif kind=='static':raw=np.broadcast_to(array(state['position_field.raw'])[:,None,:],(2,nq,size))
    else:
        raw=(array(features)@array(state['position_field.predictor.weight']).T+array(state['position_field.predictor.bias'])).reshape(nq,2,size).transpose(1,0,2)
    for head in range(2):
        for i in range(nq):
            center=4*np.tanh(raw[head,i,:dim]/4)
            shape=3*np.tanh(raw[head,i,dim:]/3)
            diagonal=np.broadcast_to(shape[:1],(dim,)) if kind=='isotropic' else shape[:dim]
            lower=np.zeros((dim,dim))
            if kind in ('conditional','static','free_quadratic'):
                lower[np.tril_indices(dim,-1)]=shape[dim:]
            delta=array(key_mean)-array(query_mean)[i]
            if kind=='free_quadratic':
                u=np.diag(diagonal)+lower
                precision=(np.eye(dim)+u+u.T)/16
                result[head,i]=-.5*np.einsum('ni,ij,nj->n',delta,precision,delta)+delta@center/16
            else:
                factor=np.diag(np.exp(diagonal))+lower
                precision=factor@factor.T/16
                shifted=delta-center
                result[head,i]=-.5*np.einsum('ni,ij,nj->n',shifted,precision,shifted)
    return result


def check_sample(role,state,path):
    with np.load(path,allow_pickle=False) as archive:z={k:archive[k] for k in archive.files}
    temperature=np.exp(min(float(array(state['layers.0.log_temperature'])),np.log(60)))
    score=temperature*(array(z['q'])@array(z['k']).swapaxes(-1,-2))
    score+=numpy_energy(role,state,z['features'],z['qmean'],z['kmean'])
    probability=np.exp(score-logsumexp(score,axis=-1,keepdims=True))
    error=float(abs(probability-z['probability']).max())
    return dict(p_max_error=error,winner_changes=int(np.sum(probability.argmax(-1)!=z['probability'].argmax(-1))),
        warning=bool(error>5e-5),scope='first validation pair, first layer, eight query rows/all keys/two heads; direct pair energy')


def training(out):
    lock=checked(out);dest=out/'audit_training'
    for seed in SEEDS:
        if not read(out/f'seed_{seed}_complete.json')['completed']:raise ValueError('all seeds required')
    if dest.exists():raise ValueError('retain audit')
    dest.mkdir();rows=[];warnings=[]
    for role in NEW_ROLES:
        for seed in SEEDS:
            path=out/'runs'/role/str(seed);record=read(path/'run.json')
            for name,digest in record['hashes'].items():
                if sha(path/name)!=digest:raise ValueError('changed training output')
            if [record['initial_shared_sha256'],record['batch_sha256']]!=lock['baseline_identity'][str(seed)]:raise ValueError('unpaired inherited training')
            h=read(path/'history.json')['validation']
            if [x['step'] for x in h]!=list(range(0,1001,100)) or min(h,key=lambda x:x['ce'])['step']!=record['best_step']:raise ValueError('selection')
            state=torch.load(path/'best.pt',weights_only=True,map_location='cpu')['state_dict']
            numeric=check_sample(role,state,path/'score_sample.npz')
            if numeric['warning']:warnings.append(dict(role=role,seed=seed,**numeric))
            rows.append(dict(role=role,seed=seed,path=str(path),reused=False,run_sha256=sha(path/'run.json'),checkpoint_sha256=sha(path/'best.pt'),numeric=numeric))
            print('AUDIT',role,seed,numeric['p_max_error'],flush=True)
    for item in lock['inherited']:
        path=Path(item['path'])
        for name,digest in item['hashes'].items():
            if sha(path/name)!=digest:raise ValueError('changed inherited output')
        rows.append(dict(role=item['role'],seed=item['seed'],path=str(path),reused=True,
            run_sha256=item['hashes']['run.json'],checkpoint_sha256=item['hashes']['best.pt'],numeric=item['numeric']))
        if item['numeric']['warning']:warnings.append(dict(role=item['role'],seed=item['seed'],**item['numeric']))
    write(dest/'completion.json',dict(records=rows,warnings=warnings,paired=True,new_checkpoints=65,reused_checkpoints=30,
        new_old_eval_read=False,independent_PE_formulas=True,independently_authored=False,unix=time.time()))


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
        role,seed=rec['role'],rec['seed'];path=Path(rec['path'])
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
    strongest='rope2d';checks=[]
    for control in (strongest,'warp2d','rope3d','warp3d','mlp3d','fourier3d','free_quadratic3d'):
        delta=means['conditional3d']['pck8']-means[control]['pck8'];ce=means['conditional3d']['ce']/means[control]['ce']-1
        wins=sum(table['conditional3d'][str(s)]['macro']['pck8']>table[control][str(s)]['macro']['pck8'] for s in SEEDS)
        scene={name:float(np.mean([table['conditional3d'][str(s)]['by_scene'][name]['pck8']-table[control][str(s)]['by_scene'][name]['pck8'] for s in SEEDS])) for name in next(iter(table[control].values()))['by_scene']}
        checks.append(dict(control=control,pck8_difference_pp=100*delta,ce_relative_change=ce,winning_seeds=wins,
            scene_differences=scene,passed=bool(delta>=.01 and wins>=4 and sum(v>0 for v in scene.values())>=4 and ce<=.02)))
    components=[]
    for control in ('fixed3d','static3d','isotropic3d','diagonal3d'):
        seed_deltas=[100*(table['conditional3d'][str(s)]['macro']['pck8']-table[control][str(s)]['macro']['pck8']) for s in SEEDS]
        scene_deltas={name:100*float(np.mean([table['conditional3d'][str(s)]['by_scene'][name]['pck8']-table[control][str(s)]['by_scene'][name]['pck8'] for s in SEEDS])) for name in next(iter(table[control].values()))['by_scene']}
        components.append(dict(control=control,pck8_difference_pp=float(np.mean(seed_deltas)),paired_seed_pp=seed_deltas,
            scene_pp=scene_deltas,winning_seeds=sum(v>0 for v in seed_deltas),winning_scenes=sum(v>0 for v in scene_deltas.values())))
    secondary=[]
    for control in ('rope2d','warp2d','free_quadratic2d','fixed2d','static2d','isotropic2d','diagonal2d','nope'):
        seed_deltas=[100*(table['conditional2d'][str(s)]['macro']['pck8']-table[control][str(s)]['macro']['pck8']) for s in SEEDS]
        scene_deltas={name:100*float(np.mean([table['conditional2d'][str(s)]['by_scene'][name]['pck8']-table[control][str(s)]['by_scene'][name]['pck8'] for s in SEEDS])) for name in next(iter(table[control].values()))['by_scene']}
        secondary.append(dict(control=control,pck8_difference_pp=float(np.mean(seed_deltas)),paired_seed_pp=seed_deltas,
            scene_pp=scene_deltas,winning_seeds=sum(v>0 for v in seed_deltas),winning_scenes=sum(v>0 for v in scene_deltas.values()),
            ce_relative_change=means['conditional2d']['ce']/means[control]['ce']-1))
    dest.mkdir();write(dest/'frame_rows.json',allrows)
    write(dest/'summary.json',dict(means=means,by_role_seed=table,checks=checks,components=components,secondary_2d=secondary,new_runs=65,reused_controls=30,
        zero_v={r:{str(seed):summarize(r,seed,'zero_v') for seed in SEEDS} for r in ROLES},
        promising_development_only=all(c['passed'] for c in checks),fresh_confirmation=False,goal_complete=False))
    write(dest/'verification.json',dict(rows=len(allrows),pairs_per_run=len(ids),valid_queries_per_run=int(mask.sum()),
        max_gpu_numpy_frame_sum_difference=max_errors,count_changed_frames=count_changes,
        sampled_PE_warnings=seal['warnings'],input_hashes=sources,
        scope='all-query metric replay from saved sufficient statistics; sampled PE score audit, not full independent network replay'))


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('training','metrics'));p.add_argument('--out',type=Path,default=OUT);a=p.parse_args()
    torch.set_num_threads(4);torch.set_num_interop_threads(4);globals()[a.stage](a.out)


if __name__=='__main__':main()
