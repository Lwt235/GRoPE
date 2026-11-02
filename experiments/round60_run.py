"""Exact source transfer and fixed adaptation with the unchanged R58 fit loop."""
import argparse
from contextlib import contextmanager
import shutil
import time
import unittest
import numpy as np
import torch
from models.round59_relation_controls import make_model
from experiments import round58_run as original
from experiments.round58_data import loss_function
from experiments.round60_common import (ROOT,OUT,RAW,PAIR_LOCK,PRIOR_OUT,CONFIG,ROLES,SEEDS,
    CONDITIONS,PHASES,SOURCES,checked,read,write,sha,capture_inputs,source_record,run_path)
from experiments.round60_data import load_cache


def source_model(role,seed,lock=None):
    lock=checked() if lock is None else lock
    record=source_record(lock,role,seed);path=record['checkpoint_path']
    if sha(path)!=record['checkpoint_sha256']:
        raise ValueError('source checkpoint changed')
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    if (checkpoint['role'],checkpoint['seed'])!=(role,seed):
        raise ValueError('source role/seed')
    model=make_model(role)
    model.load_state_dict(checkpoint['state_dict'],strict=True)
    return model


@contextmanager
def fitting_adapter(seed,factory=None):
    changes=dict(OUT=OUT,CONFIG=CONFIG,ROLES=ROLES,checked=checked,load_cache=load_cache,
                 SupportMatcher=(lambda role:source_model(role,seed)) if factory is None else factory)
    saved={name:getattr(original,name) for name in changes}
    try:
        for name,value in changes.items():setattr(original,name,value)
        yield
    finally:
        for name,value in saved.items():setattr(original,name,value)


def freeze():
    if OUT.exists():raise ValueError('preserve existing R60 attempt')
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_round60_real_scene.py')
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():raise RuntimeError('pre-freeze implementation tests')
    from experiments.round59_common import checked as previous_check
    previous_check()
    if not read(PRIOR_OUT/'completion.json')['round_complete']:
        raise ValueError('R59 must be complete')
    selected=read(PRIOR_OUT/'selected.json');models=[]
    for record in selected['runs']:
        role,seed=record['role'],record['seed'];path=run_path(role,seed)/'best.pt'
        if sha(path)!=record['checkpoint_sha256']:raise ValueError('source selection identity')
        models.append(dict(role=role,seed=seed,checkpoint_path=str(path),checkpoint_sha256=sha(path)))
    if len(models)!=30:raise ValueError('30 exact source models')
    raw=capture_inputs()
    records=(PAIR_LOCK,RAW/'selection.json',RAW/'download_manifest.json',
             PRIOR_OUT/'lock.json',PRIOR_OUT/'selected.json',PRIOR_OUT/'completion.json')
    sources={name:sha(ROOT/name) for name in SOURCES}
    OUT.mkdir(parents=True)
    for name in sources:
        dest=OUT/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest)
    lock=dict(config=CONFIG,roles=ROLES,seeds=SEEDS,conditions=CONDITIONS,phases=PHASES,
        sources=sources,input_records={str(p):sha(p) for p in records},source_models=models,**raw,
        frozen_unix=time.time(),primary='invariant_mlp',new_target_fits=30,source_models_reused=30,
        confirmation_access=False,goal_complete=False)
    write(OUT/'lock.json',lock);(OUT/'lock.sha256').write_text(sha(OUT/'lock.json')+'\n')
    write(OUT/'units.json',dict(passed=True,tests=result.testsRun,sources=sources))
    print('FROZEN R60',sha(OUT/'lock.json'),'sources',len(sources),'native files',len(raw['raw_hashes']),flush=True)


def preflight():
    lock=checked()
    from experiments.round60_audit import audit_sample
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    caches={n:load_cache(n) for n in ('fit0','fit1','val')}
    if not np.array_equal(caches['fit0']['ids'],caches['fit1']['ids']):
        raise ValueError('fit realizations must share pair IDs')
    for name,cache in caches.items():
        if not (cache['overlap'].max(-1)>=.3).any():raise ValueError('no valid queries '+name)
    samples=[audit_sample(OUT/'data'/f'{n}_sample.npz') for n in caches]
    cache=caches['fit0'];valid=(cache['overlap'].max(-1)>=.3).any(-1)
    indices=np.flatnonzero(valid)[:2]
    data=original.model_inputs(cache,indices);truth=torch.as_tensor(cache['overlap'][indices],device='cuda')
    rows=[]
    for role in ROLES:
        model=source_model(role,11,lock).cuda().train()
        initial=original.shared_digest(model)
        optimizer=torch.optim.AdamW(model.parameters(),lr=CONFIG['lr'],weight_decay=CONFIG['weight_decay'])
        loss=loss_function(model(*data),truth);loss.backward()
        grads={name:None if p.grad is None else float(p.grad.norm()) for name,p in model.position.named_parameters()}
        if any(g is None or not np.isfinite(g) for g in grads.values()) or (grads and not any(g>0 for g in grads.values())):
            raise ValueError('position gradient '+role)
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        if not torch.isfinite(loss) or original.shared_digest(model)==initial:
            raise ValueError('actual source update '+role)
        rows.append(dict(role=role,seed=11,loss=float(loss),gradients=grads,
                         source_checkpoint_sha256=source_record(lock,role,11)['checkpoint_sha256']))
    manifest=read(OUT/'data/fit_manifest.json')
    write(OUT/'preflight.json',dict(passed=True,rows=rows,cache_checks=samples,
        cache_sha256=manifest['caches'],cache_record_sha256=manifest['records'],
        gpu=torch.cuda.get_device_name(),unix=time.time()))
    print('PREFLIGHT10source updates;native input/support checks passed',flush=True)


def train(role,seed):
    lock=checked()
    if role not in ROLES or seed not in SEEDS:raise ValueError('scope')
    with fitting_adapter(seed):original.train(role,seed)
    path=OUT/'runs'/role/str(seed)/'run.json';run=read(path)
    source=source_model(role,seed,lock)
    if run['initial_shared_sha256']!=original.shared_digest(source):
        raise ValueError('actual target initialization is not its source checkpoint')
    run.update(source_checkpoint=source_record(lock,role,seed),
        initial_backbone_shared_across_roles=False,training_function='unchanged experiments.round58_run.train')
    write(path,run)
    print('SOURCE INITIALIZATION VERIFIED',role,seed,flush=True)


@torch.no_grad()
def evaluate(model,cache):
    return original.evaluate(model,cache)


def scene_metrics(scores,cache):
    by_scene={}
    for scene in sorted(set(cache['scenes'])):
        mask=cache['scenes']==scene
        stats=original.metric_torch(scores[mask],cache['overlap'][mask],cache['world_centers'][mask])
        by_scene[str(scene)]=stats
    keys=('region_recall','center_pck','soft_epe','ce')
    usable=[s for s in by_scene.values() if sum(s['valid_counts'])>0]
    macro={k:float(np.mean([s[k] for s in usable])) for k in keys} if usable else None
    return by_scene,macro


def export(seed):
    lock=checked();selected=read(OUT/'selected.json')
    if seed not in SEEDS or len(selected['runs'])!=30:raise ValueError('target selection scope')
    for condition in CONDITIONS:
        cache=load_cache('eval_'+condition)
        for role in ROLES:
            rec=next(x for x in selected['runs'] if (x['role'],x['seed'])==(role,seed))
            for phase in PHASES:
                if phase=='source':
                    model=source_model(role,seed,lock)
                    digest=source_record(lock,role,seed)['checkpoint_sha256']
                else:
                    path=OUT/'runs'/role/str(seed)/'best.pt';digest=sha(path)
                    if digest!=rec['checkpoint_sha256']:raise ValueError('target checkpoint changed')
                    model=make_model(role)
                    model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)['state_dict'])
                path=OUT/'evaluation'/phase/role/str(seed)/(condition+'.npz')
                if path.exists():raise ValueError('preserve exported predictions')
                path.parent.mkdir(parents=True,exist_ok=True)
                scores,stats=evaluate(model.cuda(),cache)
                by_scene,macro=scene_metrics(scores,cache)
                np.savez_compressed(path,logits=scores,ids=cache['ids'])
                write(path.with_suffix('.json'),dict(phase=phase,role=role,seed=seed,condition=condition,
                    metrics=stats,by_scene=by_scene,macro=macro,checkpoint_sha256=digest,
                    predictions_sha256=sha(path),cache_sha256=sha(OUT/'data'/('eval_'+condition+'.npz')),
                    selection_sha256=sha(OUT/'selected.json'),exported_unix=time.time()))
                print('EXPORTED',phase,role,seed,condition,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('freeze','preflight','train','export'))
    parser.add_argument('--role',choices=ROLES);parser.add_argument('--seed',type=int)
    args=parser.parse_args();original.configure()
    if args.stage=='train':train(args.role,args.seed)
    elif args.stage=='export':export(args.seed)
    else:globals()[args.stage]()
