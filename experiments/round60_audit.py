"""Bounded native-data checks, source/selection pairing and independent metrics."""
import argparse
import hashlib
import math
import time
import numpy as np
import torch
from experiments.round58_audit import metrics_numpy
from experiments.round58_run import shared_digest,model_inputs
from models.round59_relation_controls import make_model
from experiments.round60_run import source_model
from experiments.round60_data import load_cache
from experiments.round60_common import OUT,CONFIG,ROLES,SEEDS,CONDITIONS,PHASES,checked,read,write,sha,source_record


def audit_sample(path):
    with np.load(path,allow_pickle=False) as f:a={k:f[k] for k in f.files}
    errors={k:0. for k in ('local','world','center','covariance','world_center','overlap','features')}
    def compare(key,x,y):errors[key]=max(errors[key],float(np.abs(x-y).max()))
    for b in range(len(a['points'])):
        scale=a['scale'][b]
        expected_local=np.einsum('vni,vji->vnj',(a['sensor'][b]-a['origins'][b,:,None])/scale,
                                a['augment_rotations'][b])+a['augment_translations'][b,:,None]
        expected_world=np.stack([a['sensor'][b,v]@a['poses'][b,v,:3,:3].T+
            a['poses'][b,v,:3,3]-a['poses'][b,0,:3,3] for v in range(2)])/scale
        compare('local',expected_local,a['points'][b]);compare('world',expected_world,a['world'][b])
        patches=[]
        for v in range(2):
            indices=a['token_neighbors'][b,v]
            points=a['points'][b,v].astype(np.float64)[indices]
            center=points.mean(1);delta=points-center[:,None]
            covariance=np.einsum('nki,nkj->nij',delta,delta)/32+np.eye(3)*.02**2
            radial=np.sort(np.linalg.norm(delta,axis=-1),axis=-1)
            features=np.concatenate((.5*np.log(np.linalg.eigvalsh(covariance)/.04),
                np.log1p(radial[:,np.arange(1,9)*4-1]/.2)),axis=-1)
            compare('center',center,a['token_centers'][b,v]);compare('covariance',covariance,a['token_covariance'][b,v])
            compare('world_center',a['world'][b,v].astype(np.float64)[indices].mean(1),a['token_world_centers'][b,v])
            compare('features',features,a['token_features'][b,v])
            patches.append(a['clean'][b,v].astype(np.float64)[indices])
        for i,first in enumerate(patches[0]):
            for j,second in enumerate(patches[1]):
                near=np.square(first[:,None]-second[None]).sum(-1)<.05**2
                value=.5*(near.any(0).mean()+near.any(1).mean())
                errors['overlap']=max(errors['overlap'],abs(value-a['token_overlap'][b,i,j]))
    if max(errors.values())>2e-4:
        raise ValueError('substantive input/support mismatch '+str(errors))
    return dict(path=str(path),errors=errors,pose_in_forward=False)


def expected_stream(seed,count):
    rng=np.random.default_rng(seed+58000);digest=hashlib.sha256()
    for epoch in range(CONFIG['epochs']):
        digest.update(np.array([epoch%2],dtype='<i8').tobytes())
        digest.update(rng.permutation(count).astype('<i8').tobytes())
    return digest.hexdigest()


def training():
    lock=checked()
    if (OUT/'selected.json').exists():raise ValueError('selection already sealed')
    preflight=read(OUT/'preflight.json');validation=load_cache('val')
    count=read(OUT/'data/fit0.json')['count'];steps=CONFIG['epochs']*math.ceil(count/CONFIG['batch_size'])
    for name,digest in preflight['cache_sha256'].items():
        if sha(OUT/'data'/f'{name}.npz')!=digest:raise ValueError('fit input changed')
        if sha(OUT/'data'/f'{name}.json')!=preflight['cache_record_sha256'][name]:raise ValueError('fit metadata changed')
    inputs=model_inputs(validation,slice(0,2),device='cpu');records=[]
    for role in ROLES:
        for seed in SEEDS:
            path=OUT/'runs'/role/str(seed);run=read(path/'run.json');history=read(path/'history.json')
            if (run['role'],run['seed'],run['epochs'],run['steps'])!=(role,seed,60,steps):raise ValueError('fit identity/budget')
            source=source_model(role,seed,lock).eval()
            if run['source_checkpoint']!=source_record(lock,role,seed) or run['initial_shared_sha256']!=shared_digest(source):
                raise ValueError('source starting state')
            if run['stream_sha256']!=expected_stream(seed,count):raise ValueError('target sampling stream')
            if run['data_sha256']!=preflight['cache_sha256']:raise ValueError('unmatched actual caches')
            for name,digest in run['hashes'].items():
                if sha(path/name)!=digest:raise ValueError('run artifact changed')
            best=min(history['validation'],key=lambda x:x['selection_ce'])
            if (best['epoch'],best['selection_ce'])!=(run['selected_epoch'],run['best_val_ce']):raise ValueError('selection rule')
            with np.load(path/'validation.npz',allow_pickle=False) as f:
                if not np.array_equal(f['ids'],validation['ids']):raise ValueError('validation IDs')
                metric,_=metrics_numpy(f['logits'],validation['overlap'],validation['world_centers'])
                saved=f['logits'][:2].copy()
            ce_error=abs(metric['selection_ce']-run['best_val_ce'])
            checkpoint=torch.load(path/'best.pt',map_location='cpu',weights_only=True)
            if (checkpoint['role'],checkpoint['seed'],checkpoint['epoch'])!=(role,seed,run['selected_epoch']):raise ValueError('checkpoint selection')
            model=make_model(role).eval();model.load_state_dict(checkpoint['state_dict'])
            with torch.no_grad():replay=float(np.abs(model(*inputs).numpy()-saved).max())
            if ce_error>1e-6 or replay>.002 or not run['shared_parameters_changed']:
                raise ValueError('substantive fit/replay mismatch')
            records.append(dict(role=role,seed=seed,selected_epoch=run['selected_epoch'],
                checkpoint_sha256=sha(path/'best.pt'),source_checkpoint_sha256=source_record(lock,role,seed)['checkpoint_sha256'],
                run_sha256=sha(path/'run.json'),validation_ce_error=ce_error,replay_logit_error=replay))
    write(OUT/'selected.json',dict(runs=records,sealed_unix=time.time(),lock_sha256=sha(OUT/'lock.json'),
        source_models_reused=30,target_fits=30,initial_backbone_shared_across_roles=False,fresh_confirmation=False))
    print('SEALED30 target selections and paired source identities',flush=True)


def metrics():
    lock=checked();selected=read(OUT/'selected.json');rows=[];errors=[]
    def compare(actual,recorded):
        for key,value in actual.items():
            if key=='valid_counts':
                if value!=recorded[key]:raise ValueError('query count mismatch')
            else:
                error=abs(value-recorded[key]);errors.append(error)
                if error>1e-6:raise ValueError('substantive metric mismatch '+key)
    cache_checks=[]
    for condition in CONDITIONS:
        name='eval_'+condition;cache=load_cache(name)
        cache_checks.append(audit_sample(OUT/'data'/f'{name}_sample.npz'))
        scenes=sorted(set(cache['scenes']))
        for phase in PHASES:
            for role in ROLES:
                for seed in SEEDS:
                    path=OUT/'evaluation'/phase/role/str(seed)/(condition+'.npz');rec=read(path.with_suffix('.json'))
                    choice=next(x for x in selected['runs'] if (x['role'],x['seed'])==(role,seed))
                    digest=choice['checkpoint_sha256'] if phase=='adapted' else source_record(lock,role,seed)['checkpoint_sha256']
                    if rec['checkpoint_sha256']!=digest or rec['selection_sha256']!=sha(OUT/'selected.json'):
                        raise ValueError('prediction selection identity')
                    if rec['exported_unix']<selected['sealed_unix'] or sha(path)!=rec['predictions_sha256']:
                        raise ValueError('prediction identity/timing')
                    if rec['cache_sha256']!=sha(OUT/'data'/f'{name}.npz'):raise ValueError('prediction input identity')
                    with np.load(path,allow_pickle=False) as f:
                        if not np.array_equal(f['ids'],cache['ids']):raise ValueError('prediction IDs')
                        actual,objects=metrics_numpy(f['logits'],cache['overlap'],cache['world_centers'])
                        by_scene={}
                        for scene in scenes:
                            mask=cache['scenes']==scene
                            by_scene[str(scene)],_=metrics_numpy(f['logits'][mask],cache['overlap'][mask],cache['world_centers'][mask])
                    compare(actual,rec['metrics'])
                    for scene in scenes:compare(by_scene[str(scene)],rec['by_scene'][str(scene)])
                    usable=[s for s in by_scene.values() if sum(s['valid_counts'])>0]
                    macro={k:float(np.mean([s[k] for s in usable])) for k in ('region_recall','center_pck','soft_epe','ce')} if usable else None
                    if macro is not None:
                        if rec['macro'] is None:raise ValueError('missing macro')
                        for k,v in macro.items():
                            error=abs(v-rec['macro'][k]);errors.append(error)
                            if error>1e-6:raise ValueError('macro mismatch')
                    elif rec['macro'] is not None:raise ValueError('unexpected macro')
                    destination=OUT/'metrics'/phase/role/str(seed);destination.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(destination/(condition+'_pairs.npz'),**objects,ids=cache['ids'],scenes=cache['scenes'])
                    rows.append(dict(phase=phase,role=role,seed=seed,condition=condition,micro=actual,
                        by_scene=by_scene,macro=macro,evaluable_scenes=len(usable),total_scenes=len(scenes)))
        print('METRICS',condition,len(rows),flush=True)
    write(OUT/'metrics/summary.json',dict(rows=rows,groups=len(rows),source_models_reused=30,target_fits=30,
        max_metric_error=max(errors),cache_checks=cache_checks,fresh_confirmation=False,goal_complete=False))
    print('METRICS COMPLETE',len(rows),'groups;maxerror',max(errors),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('training','metrics'))
    torch.set_num_threads(4);globals()[parser.parse_args().stage]()
