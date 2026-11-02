"""New fit identities and separate metric arithmetic, with exact old reuse."""
import argparse
import time
import numpy as np
import torch
from experiments.round58_audit import metrics_numpy
from experiments.round58_run import shared_digest,model_inputs
from experiments.round59_run import reframed
from experiments.round61_run import source_model,multiplicities
from models.round61_normal_relations import make_model
from experiments.round61_common import (OUT,DOMAINS,PHASES,CONFIGS,CONDITIONS,ROLES,NEW_ROLES,
    SEEDS,checked,read,write,sha,load_cache,run_path,prediction_path,reference_run,reference_prediction)


def training(domain):
    lock=checked();destination=OUT/domain/'selected.json'
    if destination.exists():raise ValueError('preserve existing selections')
    validation=load_cache(domain,'val');inputs=model_inputs(validation,slice(0,2),device='cpu')
    rows=[];preflight=read(OUT/domain/'preflight.json')
    for role in NEW_ROLES:
        for seed in SEEDS:
            path=run_path(domain,role,seed);run=read(path/'run.json');history=read(path/'history.json')
            expected=reference_run(lock,domain,'invariant_mlp',seed)
            if (run['role'],run['seed'],run['epochs'],run['steps'])!=(role,seed,60,14880 if domain=='synthetic' else 1080):
                raise ValueError('fit identity/budget')
            if run['stream_sha256']!=expected['stream_sha256']:raise ValueError('actual stream mismatch')
            if domain=='synthetic':
                if run['initial_shared_sha256']!=expected['initial_shared_sha256']:raise ValueError('common initialization')
            else:
                if run['source_checkpoint_sha256']!=sha(run_path('synthetic',role,seed)/'best.pt'):
                    raise ValueError('source checkpoint identity')
                if run['initial_shared_sha256']!=shared_digest(source_model(role,seed)):
                    raise ValueError('actual target initial state')
            if run['data_sha256']!=preflight['cache_sha256']:raise ValueError('actual fit inputs')
            for name,digest in run['hashes'].items():
                if sha(path/name)!=digest:raise ValueError('changed fit artifact')
            best=min(history['validation'],key=lambda x:x['selection_ce'])
            if (best['epoch'],best['selection_ce'])!=(run['selected_epoch'],run['best_val_ce']):
                raise ValueError('validation selection rule')
            with np.load(path/'validation.npz',allow_pickle=False) as f:
                if not np.array_equal(f['ids'],validation['ids']):raise ValueError('val IDs')
                metrics,_=metrics_numpy(f['logits'],validation['overlap'],validation['world_centers'])
                saved=f['logits'][:2].copy()
            ce=abs(metrics['selection_ce']-run['best_val_ce'])
            checkpoint=torch.load(path/'best.pt',map_location='cpu',weights_only=True)
            if (checkpoint['role'],checkpoint['seed'],checkpoint['epoch'])!=(role,seed,run['selected_epoch']):
                raise ValueError('checkpoint selection')
            model=make_model(role).eval();model.load_state_dict(checkpoint['state_dict'],strict=True)
            with torch.no_grad():
                base=model(*inputs);rigid=model(*reframed(inputs))
                permuted=model(inputs[0].flip(-2),inputs[1].flip(-2),inputs[2].flip(-3)).flip(-2,-1)
                replay=float(np.abs(base.numpy()-saved).max())
                re=float((base.softmax(-1)-rigid.softmax(-1)).abs().max())
                pe=float((base-permuted).abs().max())
                winner=int((base.argmax(-1)!=rigid.argmax(-1)).sum())
            if ce>1e-6 or max(replay,re,pe)>.002 or not run['shared_parameters_changed']:
                raise ValueError('substantive selection/replay mismatch')
            rows.append(dict(domain=domain,role=role,seed=seed,selected_epoch=run['selected_epoch'],
                checkpoint_sha256=sha(path/'best.pt'),run_sha256=sha(path/'run.json'),
                validation_ce_error=ce,replay_logit_error=replay,rigid_probability_error=re,
                rigid_winner_changes=winner,permutation_logit_error=pe))
    write(destination,dict(runs=rows,sealed_unix=time.time(),lock_sha256=sha(OUT/'lock.json'),fresh_confirmation=False))
    if domain=='real':
        combined=read(OUT/'synthetic/selected.json')['runs']+rows
        if len(combined)!=12 or (OUT/'selected.json').exists():raise ValueError('global selection scope')
        write(OUT/'selected.json',dict(runs=combined,sealed_unix=time.time(),lock_sha256=sha(OUT/'lock.json'),
            reused_source_models=15,reused_target_models=15,fresh_confirmation=False,goal_complete=False))
    print('SEALED',domain,'six selections',flush=True)


def metrics():
    lock=checked();selection=read(OUT/'selected.json');rows=[];errors=[];counts={}
    if len(selection['runs'])!=12:raise ValueError('all selections required')
    def compare(actual,expected):
        for k,v in actual.items():
            if k=='valid_counts':
                if v!=expected[k]:raise ValueError('valid count mismatch')
            else:
                e=abs(v-expected[k]);errors.append(e)
                if e>1e-6:raise ValueError('substantive metric difference '+k)
    for phase in PHASES:
        domain='synthetic' if phase=='synthetic' else 'real'
        for condition in CONDITIONS[domain]:
            cache=load_cache(domain,'eval_'+condition)
            key=domain+'/'+condition
            if key not in counts:counts[key]=multiplicities(cache,'cpu')
            for role in ROLES:
                for seed in SEEDS:
                    path=prediction_path(phase,role,seed,condition);rec=read(path.with_suffix('.json'))
                    if role not in NEW_ROLES:
                        old=reference_prediction(lock,phase,role,seed,condition)
                        if sha(path)!=old['sha256'] or sha(path.with_suffix('.json'))!=old['metadata_sha256']:
                            raise ValueError('reference prediction changed')
                        rows.append(old['metric_row']);continue
                    model_domain='real' if phase=='real_adapted' else 'synthetic'
                    selected=next(x for x in selection['runs'] if (x['domain'],x['role'],x['seed'])==(model_domain,role,seed))
                    if rec['checkpoint_sha256']!=selected['checkpoint_sha256'] or rec['selection_sha256']!=sha(OUT/'selected.json'):
                        raise ValueError('prediction selected identity')
                    if rec['exported_unix']<selection['sealed_unix'] or sha(path)!=rec['predictions_sha256']:
                        raise ValueError('prediction time/identity')
                    if rec['cache_sha256']!=lock['caches'][domain]['eval_'+condition]['sha256']:
                        raise ValueError('prediction input identity')
                    with np.load(path,allow_pickle=False) as f:
                        if not np.array_equal(f['ids'],cache['ids']):raise ValueError('prediction IDs')
                        actual,objects=metrics_numpy(f['logits'],cache['overlap'],cache['world_centers'])
                        by_scene={}
                        if domain=='real':
                            for scene in sorted(set(cache['scenes'])):
                                mask=cache['scenes']==scene
                                by_scene[str(scene)],_=metrics_numpy(f['logits'][mask],cache['overlap'][mask],cache['world_centers'][mask])
                    compare(actual,rec['metrics'])
                    extra={}
                    if domain=='real':
                        for scene,value in by_scene.items():compare(value,rec['by_scene'][scene])
                        usable=[s for s in by_scene.values() if sum(s['valid_counts'])>0]
                        primary={k:float(np.mean([s[k] for s in usable])) for k in ('region_recall','center_pck','soft_epe','ce')} if usable else None
                        if primary is None:raise ValueError('no evaluable scene;inspect instead of inventing score')
                        compare(primary,rec['macro'])
                        extra=dict(by_scene=by_scene,macro=primary,evaluable_scenes=len(usable),total_scenes=len(by_scene))
                    else:primary={k:actual[k] for k in ('region_recall','center_pck','soft_epe','ce')}
                    dest=OUT/'metrics'/phase/role/str(seed);dest.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(dest/(condition+'_pairs.npz'),**objects,ids=cache['ids'])
                    rows.append(dict(domain=domain,phase=phase,role=role,seed=seed,condition=condition,reused=False,
                                     micro=actual,primary_metrics=primary,**extra))
            print('METRICS',phase,condition,len(rows),flush=True)
    if len(rows)!=168 or sum(not r['reused'] for r in rows)!=48:raise ValueError('metric scope')
    write(OUT/'metrics/summary.json',dict(rows=rows,groups=len(rows),new_groups=48,reused_groups=120,
        new_source_fits=6,new_target_fits=6,max_new_metric_error=max(errors),multiplicities=counts,
        fresh_confirmation=False,goal_complete=False))
    print('METRICS COMPLETE168;48new independent groups,120 exact reuse;error',max(errors),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('training','metrics'));parser.add_argument('--domain',choices=DOMAINS)
    args=parser.parse_args();torch.set_num_threads(4)
    if args.stage=='training':training(args.domain)
    else:metrics()
