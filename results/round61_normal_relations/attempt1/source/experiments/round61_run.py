"""R61 adapters reuse the original source and target optimization functions."""
import argparse
from contextlib import contextmanager
from pathlib import Path
import shutil
import time
import unittest
import numpy as np
import torch
from experiments import round58_run as original
from experiments.round59_run import reframed
from experiments.round60_run import scene_metrics
from models.round61_normal_relations import make_model, spectral_geometry
from experiments.round61_common import (ROOT,OUT,ATTRIBUTION_OUT,REAL_OUT,DOMAINS,PHASES,
    CONFIGS,CONDITIONS,DATA_ROOTS,ROLES,NEW_ROLES,REFERENCE_ROLES,SEEDS,SOURCES,
    checked,read,write,sha,cache_names,load_cache,run_path,prediction_path,reference_run)


def source_model(role, seed):
    selected = read(OUT/'synthetic/selected.json')
    rec = next(x for x in selected['runs'] if (x['role'],x['seed'])==(role,seed))
    path = run_path('synthetic',role,seed)/'best.pt'
    if sha(path) != rec['checkpoint_sha256']:
        raise ValueError('selected source checkpoint changed')
    model = make_model(role)
    checkpoint = torch.load(path,map_location='cpu',weights_only=True)
    if (checkpoint['role'],checkpoint['seed']) != (role,seed):
        raise ValueError('source checkpoint scope')
    model.load_state_dict(checkpoint['state_dict'],strict=True)
    return model


@contextmanager
def fitting_adapter(domain, seed, factory=None):
    if domain not in DOMAINS:
        raise ValueError('domain')
    if factory is None:
        factory = make_model if domain=='synthetic' else lambda role:source_model(role,seed)
    changes = dict(OUT=OUT/domain,CONFIG=CONFIGS[domain],ROLES=NEW_ROLES,
                   SupportMatcher=factory,checked=checked,load_cache=lambda name:load_cache(domain,name))
    saved = {name:getattr(original,name) for name in changes}
    try:
        for name,value in changes.items():setattr(original,name,value)
        yield
    finally:
        for name,value in saved.items():setattr(original,name,value)


def freeze():
    if OUT.exists():raise ValueError('preserve existing R61 attempt')
    from experiments.round60_common import checked as check60
    check60()
    if read(REAL_OUT/'completion.json')['status']!='complete':
        raise ValueError('finish R60 first')
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_round61_normal_relations.py')
    tests=unittest.TextTestRunner(verbosity=2).run(suite)
    if not tests.wasSuccessful():raise RuntimeError('new formula/adapter tests')
    records={}
    for root in (ATTRIBUTION_OUT,REAL_OUT):
        for name in ('lock.json','completion.json','selected.json','metrics/summary.json'):
            records[str(root/name)]=sha(root/name)
    caches={};runs=[];predictions=[]
    old_synthetic=read(ATTRIBUTION_OUT/'metrics/summary.json')['rows']
    old_real=read(REAL_OUT/'metrics/summary.json')['rows']
    for domain in DOMAINS:
        caches[domain]={}
        for name in cache_names(domain):
            meta=DATA_ROOTS[domain]/'data'/f'{name}.json';data=meta.with_suffix('.npz');info=read(meta)
            if sha(data)!=info['sha256']:raise ValueError('cache changed '+str(data))
            caches[domain][name]=dict(path=str(data),sha256=info['sha256'],count=info['count'])
            records[str(meta)]=sha(meta)
        for role in REFERENCE_ROLES:
            for seed in SEEDS:
                path=run_path(domain,role,seed);run=read(path/'run.json')
                if (run['role'],run['seed'],run['epochs'],run['steps'])!=(role,seed,60,14880 if domain=='synthetic' else 1080):
                    raise ValueError('reference fitting identity')
                for name,digest in run['hashes'].items():
                    if sha(path/name)!=digest:raise ValueError('reference run artifact')
                runs.append(dict(domain=domain,role=role,seed=seed,path=str(path),
                    checkpoint_sha256=run['hashes']['best.pt'],run_sha256=sha(path/'run.json'),
                    initial_shared_sha256=run['initial_shared_sha256'],stream_sha256=run['stream_sha256']))
    for phase in PHASES:
        domain='synthetic' if phase=='synthetic' else 'real'
        for condition in CONDITIONS[domain]:
            for role in REFERENCE_ROLES:
                for seed in SEEDS:
                    path=prediction_path(phase,role,seed,condition);meta=read(path.with_suffix('.json'))
                    digest=next(x['checkpoint_sha256'] for x in runs if (x['domain'],x['role'],x['seed'])==
                                ('real' if phase=='real_adapted' else 'synthetic',role,seed))
                    if sha(path)!=meta['predictions_sha256'] or meta['checkpoint_sha256']!=digest:
                        raise ValueError('reference prediction identity')
                    if meta['cache_sha256']!=caches[domain]['eval_'+condition]['sha256']:
                        raise ValueError('reference prediction input')
                    if domain=='synthetic':
                        old=next(x for x in old_synthetic if (x['role'],x['seed'],x['condition'])==(role,seed,condition))
                        micro={k:old[k] for k in meta['metrics']}
                        normalized=dict(micro=micro,primary_metrics={k:micro[k] for k in ('region_recall','center_pck','soft_epe','ce')})
                    else:
                        old=next(x for x in old_real if (x['phase'],x['role'],x['seed'],x['condition'])==
                                 (phase.removeprefix('real_'),role,seed,condition))
                        normalized={k:old[k] for k in ('micro','by_scene','macro','evaluable_scenes','total_scenes')}
                        normalized['primary_metrics']=old['macro']
                    normalized.update(domain=domain,phase=phase,role=role,seed=seed,condition=condition,reused=True)
                    predictions.append(dict(phase=phase,role=role,seed=seed,condition=condition,path=str(path),
                        sha256=sha(path),metadata_sha256=sha(path.with_suffix('.json')),metric_row=normalized))
    sources={name:sha(ROOT/name) for name in SOURCES}
    OUT.mkdir(parents=True)
    for name in sources:
        dest=OUT/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest)
    for domain in DOMAINS:
        for name in cache_names(domain):
            dest=OUT/domain/'data'/f'{name}.json';dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(DATA_ROOTS[domain]/'data'/f'{name}.json',dest)
    lock=dict(configs=CONFIGS,roles=ROLES,new_roles=NEW_ROLES,seeds=SEEDS,sources=sources,
        input_records=records,caches=caches,reference_runs=runs,reference_predictions=predictions,
        frozen_unix=time.time(),primary='invariant_mlp',new_source_fits=6,new_target_fits=6,
        reused_source_models=15,reused_target_models=15,new_prediction_groups=48,reused_prediction_groups=120,
        fresh_confirmation=False,goal_complete=False)
    write(OUT/'lock.json',lock);(OUT/'lock.sha256').write_text(sha(OUT/'lock.json')+'\n')
    write(OUT/'units.json',dict(passed=True,tests=tests.testsRun,sources=sources))
    print('FROZEN R61',sha(OUT/'lock.json'),'six source+six target fits',flush=True)


@torch.no_grad()
def multiplicities(cache, device):
    counts=np.zeros(3,dtype=np.int64)
    for begin in range(0,len(cache['ids']),128):
        covariance=torch.as_tensor(cache['covariance'][begin:begin+128],device=device)
        _,_,_,k=spectral_geometry(covariance)
        counts+=torch.bincount(k.flatten(),minlength=4)[1:4].cpu().numpy()
    return dict(counts_k1_k2_k3=counts.tolist(),total=int(counts.sum()),
                repeated_fraction=float(counts[1:].sum()/counts.sum()))


def preflight(domain):
    lock=checked()
    if domain not in DOMAINS or not torch.cuda.is_available():raise RuntimeError('domain/CUDA')
    cache=load_cache(domain,'fit0');data=original.model_inputs(cache,slice(0,2))
    truth=torch.as_tensor(cache['overlap'][:2],device='cuda');rows=[]
    for role in NEW_ROLES:
        torch.manual_seed(11)
        model=(make_model(role) if domain=='synthetic' else source_model(role,11)).cuda()
        initial=original.shared_digest(model)
        if domain=='synthetic' and initial!=reference_run(lock,domain,'invariant_mlp',11)['initial_shared_sha256']:
            raise ValueError('common source initialization')
        optimizer=torch.optim.AdamW(model.parameters(),lr=CONFIGS[domain]['lr'])
        loss=original.loss_function(model(*data),truth);loss.backward()
        gradients={n:None if p.grad is None else float(p.grad.norm()) for n,p in model.position.named_parameters()}
        if any(v is None or not np.isfinite(v) or v==0 for v in gradients.values()):raise ValueError('PE gradients')
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        if original.shared_digest(model)==initial:raise ValueError('no actual backbone update')
        model.eval()
        with torch.no_grad():
            base=model(*data);rigid=model(*reframed(data))
            permuted=model(data[0].flip(-2),data[1].flip(-2),data[2].flip(-3)).flip(-2,-1)
            re=float((base.softmax(-1)-rigid.softmax(-1)).abs().max());pe=float((base-permuted).abs().max())
        if not torch.isfinite(loss) or re>.002 or pe>.002:raise ValueError('substantive geometry/learning discrepancy')
        rows.append(dict(role=role,loss=float(loss),gradients=gradients,rigid_probability_error=re,
                         permutation_logit_error=pe,pe_parameters=sum(p.numel() for p in model.position.parameters())))
    counts={'fit0':multiplicities(cache,'cuda')}
    for name in ('fit1','val'):counts[name]=multiplicities(load_cache(domain,name),'cuda')
    write(OUT/domain/'preflight.json',dict(passed=True,rows=rows,multiplicities=counts,
        cache_sha256={n:lock['caches'][domain][n]['sha256'] for n in ('fit0','fit1','val')},
        gpu=torch.cuda.get_device_name(),unix=time.time()))
    print('PREFLIGHT',domain,'two updates and bounded geometry checks',flush=True)


def train(domain,role,seed):
    lock=checked()
    if role not in NEW_ROLES or seed not in SEEDS:raise ValueError('never refit reference')
    with fitting_adapter(domain,seed):original.train(role,seed)
    path=run_path(domain,role,seed)/'run.json';run=read(path)
    expected=reference_run(lock,domain,'invariant_mlp',seed)
    if run['stream_sha256']!=expected['stream_sha256']:raise ValueError('unmatched actual input order')
    if domain=='synthetic':
        if run['initial_shared_sha256']!=expected['initial_shared_sha256']:raise ValueError('unmatched common initialization')
    else:
        if run['initial_shared_sha256']!=original.shared_digest(source_model(role,seed)):
            raise ValueError('wrong source starting state')
        run.update(source_checkpoint_sha256=sha(run_path('synthetic',role,seed)/'best.pt'),
                   initial_backbone_shared_across_roles=False)
        write(path,run)
    print('PAIRED actual training',domain,role,seed,flush=True)


def export(seed):
    lock=checked();selection=read(OUT/'selected.json')
    if seed not in SEEDS or len(selection['runs'])!=12:raise ValueError('all selections required')
    for phase in PHASES:
        domain='synthetic' if phase=='synthetic' else 'real'
        model_domain='real' if phase=='real_adapted' else 'synthetic'
        for condition in CONDITIONS[domain]:
            cache=load_cache(domain,'eval_'+condition)
            for role in NEW_ROLES:
                rec=next(x for x in selection['runs'] if (x['domain'],x['role'],x['seed'])==(model_domain,role,seed))
                path=run_path(model_domain,role,seed)/'best.pt'
                if sha(path)!=rec['checkpoint_sha256']:raise ValueError('selected checkpoint changed')
                model=make_model(role).cuda();model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)['state_dict'])
                scores,stats=original.evaluate(model,cache)
                dest=prediction_path(phase,role,seed,condition)
                if dest.exists():raise ValueError('preserve prediction')
                dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest,logits=scores,ids=cache['ids'])
                extra={}
                if domain=='real':extra['by_scene'],extra['macro']=scene_metrics(scores,cache)
                write(dest.with_suffix('.json'),dict(phase=phase,role=role,seed=seed,condition=condition,metrics=stats,
                    checkpoint_sha256=rec['checkpoint_sha256'],cache_sha256=lock['caches'][domain]['eval_'+condition]['sha256'],
                    predictions_sha256=sha(dest),selection_sha256=sha(OUT/'selected.json'),exported_unix=time.time(),**extra))
                print('EXPORTED',phase,role,seed,condition,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('freeze','preflight','train','export'))
    parser.add_argument('--domain',choices=DOMAINS);parser.add_argument('--role',choices=NEW_ROLES);parser.add_argument('--seed',type=int)
    args=parser.parse_args();original.configure()
    if args.stage=='train':train(args.domain,args.role,args.seed)
    elif args.stage=='preflight':preflight(args.domain)
    elif args.stage=='export':export(args.seed)
    else:freeze()
