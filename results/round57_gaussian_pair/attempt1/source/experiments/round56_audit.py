"""Bounded direct-formula checks and complete saved-prediction metric replay."""
import argparse
from pathlib import Path
import time
import numpy as np
import torch
from scipy.special import logsumexp
from models.round56_native_joint import INTERVENTIONS
from experiments.round48_audit import (array, pair_energy, phase, rotate, object_metrics,
                                       check_sample as reference_check)
from experiments.round48_data import load
from experiments.round56_common import (OUT, ROLES, NEW_ROLES, SEEDS, CONFIG, VARIANTS,
    PRIMARY, checked, sha, read, write, model_path, reference)


def check_sample(role,state,path):
    if role not in NEW_ROLES:return reference_check(role,state,path)
    with np.load(path,allow_pickle=False) as z:values={k:z[k] for k in z.files}
    q=rotate(array(values['q']),phase('rope',state,values['qmean']))
    k=rotate(array(values['k']),phase('rope',state,values['kmean']))
    temperature=np.exp(min(float(array(state['layers.0.log_temperature'])),np.log(60)))
    scores=temperature*(q@k.swapaxes(-1,-2))+pair_energy(role.removeprefix('joint_'),state,
        values['features'],values['qmean'],values['kmean'])
    probability=np.exp(scores-logsumexp(scores,axis=-1,keepdims=True))
    error=float(np.abs(probability-values['probability']).max())
    return dict(p_max_error=error,winner_changes=int(np.sum(probability.argmax(-1)!=values['probability'].argmax(-1))),
        warning=bool(error>5e-5),scope='first layer, first validation cloud,eight queries,all keys,two heads')


def training(out):
    lock=checked(out)
    dest=out/'audit_training'
    if dest.exists():raise ValueError('preserve training seal')
    val=load('val',out)
    rows=[];warnings=[];identities={}
    for role in ROLES:
        for seed in SEEDS:
            path=model_path(role,seed,out);record=read(path/'run.json')
            if (record['role'],record['seed'],record['epochs'],record['steps'])!=(role,seed,120,29760):
                raise ValueError('incomplete or misidentified model')
            for name,digest in record['hashes'].items():
                if sha(path/name)!=digest:raise ValueError('training artifact changed')
            if role not in NEW_ROLES:
                ref=reference(role,seed,lock)
                for name,digest in ref['hashes'].items():
                    if sha(path/name)!=digest:raise ValueError('frozen reference changed')
            identity=[record['initial_shared_sha256'],record['batch_augmentation_sha256']]
            if identities.setdefault(str(seed),identity)!=identity or identity!=lock['paired_reference_identity'][str(seed)]:
                raise ValueError('initialization or actual full sampling stream differs')
            history=read(path/'history.json')
            if [r['epoch'] for r in history['validation']]!=list(range(0,121,5)):
                raise ValueError('selection schedule changed')
            selected=min(history['validation'],key=lambda r:r['ce'])
            if selected['epoch']!=record['best_epoch'] or selected['ce']!=record['best_val_ce'] or len(history['training'])!=120:
                raise ValueError('selection/budget mismatch')
            saved=torch.load(path/'best.pt',map_location='cpu',weights_only=True)
            if saved['role']!=role or saved['seed']!=seed or saved['epoch']!=record['best_epoch']:
                raise ValueError('selected checkpoint metadata mismatch')
            numeric=check_sample(role,saved['state_dict'],path/'score_sample.npz')
            with np.load(path/'validation.npz',allow_pickle=False) as z:
                if not np.array_equal(z['ids'],val['ids']) or not np.array_equal(z['labels'],val['labels']):
                    raise ValueError('validation population identity')
                metric=object_metrics(z['logits'],z['labels'])
            ce_error=abs(metric['ce']-record['best_val_ce'])
            if numeric['warning'] or numeric['winner_changes'] or ce_error>2e-5:
                warnings.append(dict(role=role,seed=seed,numeric=numeric,val_ce_error=ce_error))
            rows.append(dict(role=role,seed=seed,path=str(path),reused=role not in NEW_ROLES,
                checkpoint_sha256=sha(path/'best.pt'),run_sha256=sha(path/'run.json'),
                numeric=numeric,validation_metric=metric,val_ce_error=ce_error))
            print('SEALED',role,seed,numeric['p_max_error'],ce_error,flush=True)
    dest.mkdir()
    write(dest/'completion.json',dict(records=rows,warnings=warnings,paired_identity=identities,
        unix=time.time(),model_dev_eval_performed=False,official_test_access=False,
        independently_authored=False,scope='independent formula and saved validation logits,not full CPU-network reproduction'))


def metrics(out):
    checked(out);seal=read(out/'audit_training/completion.json');data=load('dev_eval',out)
    for seed in SEEDS:
        if not read(out/f'export_{seed}_complete.json')['completed']:raise ValueError('all exports required')
    dest=out/'metrics'
    if dest.exists():raise ValueError('preserve metric report')
    table={};replays=[];permutations=[];sources={}
    for saved in seal['records']:
        role,seed=saved['role'],saved['seed'];path=out/'evaluation'/role
        record=read(path/(str(seed)+'.json'));npz=path/(str(seed)+'.npz')
        if record['checkpoint_sha256']!=saved['checkpoint_sha256'] or sha(npz)!=record['predictions_sha256']:
            raise ValueError('evaluation model/prediction identity')
        if record['export_unix']<seal['unix']:raise ValueError('export precedes selection seal')
        row={};sources[str(npz)]=sha(npz)
        with np.load(npz,allow_pickle=False) as z:
            if not np.array_equal(z['ids'],data['ids']) or not np.array_equal(z['labels'],data['labels']):
                raise ValueError('evaluation population changed')
            for variant in VARIANTS:
                if z[variant].shape!=(len(data['ids']),40) or not np.isfinite(z[variant]).all():
                    raise ValueError('invalid logits')
                row[variant]=object_metrics(z[variant],data['labels']);old=record['metrics'][variant]
                replays.append(dict(role=role,seed=seed,variant=variant,
                    correct_difference=row[variant]['correct']-old['correct'],ce_error=abs(row[variant]['ce']-old['ce'])))
            permutations.append(dict(role=role,seed=seed,max_logit_error=float(abs(z['permuted']-z['base']).max()),
                winner_changes=int(np.sum(z['permuted'].argmax(-1)!=z['base'].argmax(-1)))))
        table.setdefault(role,{})[str(seed)]=row
    means={role:{v:{key:float(np.mean([table[role][str(seed)][v][key] for seed in SEEDS]))
        for key in ('oa','mca','ce')} for v in VARIANTS} for role in ROLES}
    comparisons=[]
    for control in ROLES:
        if control==PRIMARY:continue
        delta={key:[table[PRIMARY][str(s)]['base'][key]-table[control][str(s)]['base'][key] for s in SEEDS]
               for key in ('oa','mca','ce')}
        comparisons.append(dict(control=control,oa_difference_pp=100*float(np.mean(delta['oa'])),
            mca_difference_pp=100*float(np.mean(delta['mca'])),ce_difference=float(np.mean(delta['ce'])),
            paired_seed_oa_pp=[100*d for d in delta['oa']],paired_seed_ce=delta['ce'],
            winning_seeds=sum(d>0 for d in delta['oa']),tied_seeds=sum(d==0 for d in delta['oa'])))
    diagnostics=[]
    for role in ('joint_conditional','conditional'):
        for seed in SEEDS:
            for mode in INTERVENTIONS:
                path=out/'diagnostics'/role;name=str(seed)+'_'+mode
                record=read(path/(name+'.json'));npz=path/(name+'.npz')
                selected=next(r for r in seal['records'] if r['role']==role and r['seed']==seed)
                if sha(npz)!=record['predictions_sha256'] or record['checkpoint_sha256']!=selected['checkpoint_sha256']:
                    raise ValueError('diagnostic model/prediction identity')
                with np.load(npz,allow_pickle=False) as z:
                    if not np.array_equal(z['ids'],data['ids']) or not np.array_equal(z['labels'],data['labels']):
                        raise ValueError('diagnostic population identity')
                    metric=object_metrics(z['base'],data['labels'])
                original=table[role][str(seed)]['base']
                diagnostics.append(dict(role=role,seed=seed,mode=mode,metrics=metric,
                    oa_difference_pp=100*(metric['oa']-original['oa']),mca_difference_pp=100*(metric['mca']-original['mca']),
                    ce_difference=metric['ce']-original['ce'],ce_replay_error=abs(metric['ce']-record['metrics']['ce']),
                    correct_replay_difference=metric['correct']-record['metrics']['correct']))
                sources[str(npz)]=sha(npz)
    dest.mkdir()
    write(dest/'summary.json',dict(means=means,by_role_seed=table,comparisons=comparisons,
        diagnostics=diagnostics,main_role=PRIMARY,total_models=39,new_fits=9,reused_fits=30,
        seeds=SEEDS,official_test_access=False,fresh_confirmation=False,goal_complete=False,
        interpretation='Development-only;inspect paired effects and simple/generic controls,not an automatic success gate.'))
    write(dest/'verification.json',dict(metric_replays=replays,permutation_checks=permutations,
        sampled_PE_warnings=seal['warnings'],diagnostic_replays=diagnostics,input_hashes=sources,
        independent_formula_paths=True,independently_authored=False))
    print('METRICS',len(replays),'groups',len(diagnostics),'diagnostics',comparisons,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('training','metrics'))
    parser.add_argument('--out',type=Path,default=OUT);args=parser.parse_args()
    torch.set_num_threads(4);globals()[args.stage](args.out)
