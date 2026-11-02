"""Independent complex-phase/direct-energy check and full metric replay for R49."""
import argparse
from pathlib import Path
import time
import numpy as np
import torch
from scipy.special import logsumexp
from models.round49_joint_gaussian import ROLES as NEW_ROLES, base_role
from experiments.round49_joint_train import (ROOT,OUT,SEEDS,read,write,sha,checked,
    INHERITED_ROLES,STRONG_3D,CROSS_REPRESENTATION,STRONG_2D)
from experiments.round43_frequency_audit import array
ROLES=NEW_ROLES+INHERITED_ROLES


from experiments.round47_conditional_audit import numpy_energy


def check_sample(role,state,path):
    with np.load(path,allow_pickle=False) as archive:z={k:archive[k] for k in archive.files}
    def encode(content,mean):
        value=array(content)
        phase=array(mean)@array(state['frequency']).T
        return (value[...,0::2]+1j*value[...,1::2])*np.exp(1j*phase[None])
    q=encode(z['q'],z['qmean']);k=encode(z['k'],z['kmean'])
    temperature=np.exp(min(float(array(state['layers.0.log_temperature'])),np.log(60)))
    score=temperature*(q@k.conj().swapaxes(-1,-2)).real
    score+=numpy_energy(base_role(role),state,z['features'],z['qmean'],z['kmean'])
    probability=np.exp(score-logsumexp(score,axis=-1,keepdims=True))
    error=float(abs(probability-z['probability']).max())
    return dict(p_max_error=error,winner_changes=int(np.sum(probability.argmax(-1)!=z['probability'].argmax(-1))),
        warning=bool(error>5e-5),scope='first validation pair/layer, eight query rows/all keys/two heads; complex phase and direct pair energy')


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
    write(dest/'completion.json',dict(records=rows,warnings=warnings,paired=True,new_checkpoints=60,reused_checkpoints=95,
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
    def comparison(primary,control):
        seed_delta=[100*(table[primary][str(seed)]['macro']['pck8']-table[control][str(seed)]['macro']['pck8']) for seed in SEEDS]
        scene={name:100*float(np.mean([table[primary][str(seed)]['by_scene'][name]['pck8']-table[control][str(seed)]['by_scene'][name]['pck8'] for seed in SEEDS])) for name in next(iter(table[control].values()))['by_scene']}
        delta=float(np.mean(seed_delta));ce=means[primary]['ce']/means[control]['ce']-1
        wins=sum(x>0 for x in seed_delta);rooms=sum(x>0 for x in scene.values())
        return dict(primary=primary,control=control,pck8_difference_pp=delta,
            paired_seed_pp=seed_delta,scene_difference_pp=scene,winning_seeds=wins,
            winning_scenes=rooms,ce_relative_change=ce,
            pck4_difference_pp=100*(means[primary]['pck4']-means[control]['pck4']),
            epe_difference=means[primary]['epe']-means[control]['epe'],
            soft_epe_difference=means[primary]['soft_epe']-means[control]['soft_epe'],
            passed=bool(delta>=1 and wins>=4 and rooms>=4 and ce<=.02))
    primary='joint_conditional3d'
    checks=[comparison(primary,c) for c in STRONG_3D]
    cross=[comparison(primary,c) for c in CROSS_REPRESENTATION]
    components={str(dim):[comparison('joint_conditional'+str(dim)+'d','joint_'+kind+str(dim)+'d')
        for kind in ('diagonal','isotropic','static','fixed')] for dim in (2,3)}
    secondary=[comparison('joint_conditional2d',c) for c in STRONG_2D]
    phase=[comparison(role,base_role(role)) for role in NEW_ROLES]
    dest.mkdir();write(dest/'frame_rows.json',allrows)
    write(dest/'summary.json',dict(means=means,by_role_seed=table,primary=primary,
        same_input_checks=checks,cross_representation_checks=cross,components=components,
        secondary_2d=secondary,phase_effects=phase,new_runs=60,reused_controls=95,
        zero_v={r:{str(seed):summarize(r,seed,'zero_v') for seed in SEEDS} for r in ROLES},
        same_input_development_signal=all(c['passed'] for c in checks),
        old_broader_screen_passed=all(c['passed'] for c in checks+cross),
        secondary_2d_signal=all(c['passed'] for c in secondary),
        fresh_confirmation=False,goal_complete=False,
        interpretation='Same-input progress is not cross-representation or general-task success; all old deficits remain visible.'))
    write(dest/'verification.json',dict(rows=len(allrows),pairs_per_run=len(ids),valid_queries_per_run=int(mask.sum()),
        max_gpu_numpy_frame_sum_difference=max_errors,count_changed_frames=count_changes,
        sampled_PE_warnings=seal['warnings'],input_hashes=sources,
        scope='all-query metric replay from saved sufficient statistics; sampled PE score audit, not full independent network replay'))


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('training','metrics'));p.add_argument('--out',type=Path,default=OUT);a=p.parse_args()
    torch.set_num_threads(4);torch.set_num_interop_threads(4);globals()[a.stage](a.out)


if __name__=='__main__':main()
