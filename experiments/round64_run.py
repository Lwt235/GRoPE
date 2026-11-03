"""Freeze, learn, and export the fixed R64 coupled depth-field comparison."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
import gc
import hashlib
import math
import time
import unittest
import numpy as np
import torch
from experiments.round55_train import (setup,tensor_hash,common,subset,query_ce,objective,
    metric_rows,macro,check as old_check)
from experiments.round55_optimization import learning_rate,sampling,save_resume
from experiments.round64_common import (ROOT,OUT,PARENT_OUT,PREPARATION,CONFIG,SOURCES,ROLES,NEW_ROLES,
    REFERENCE_ROLES,SEEDS,read,write,sha,checked,load,run_path,prediction_path,pixels)
from models.round64_depth_field import make_model,select_inputs


def freeze():
    from experiments.round62_common import checked as parent_checked
    if (OUT/'lock.json').exists():raise ValueError('already frozen')
    parent_checked();old_check()
    prep=read(PREPARATION)
    parent_completion=ROOT/'results/round63_rgb_diagnostic/attempt1/completion.json'
    if sha(parent_completion)!=prep['parent_completion_sha256'] or not read(parent_completion)['completed']:
        raise ValueError('R63 completion identity')
    if sha(ROOT/'docs/round64_depth_field_plan.md')!=prep['plan_sha256']:
        raise ValueError('prospective plan changed')
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_round64_depth_field.py')
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():raise ValueError('new unit checks failed')
    old_manifest=PARENT_OUT/'data/manifest.json';manifest=read(old_manifest)
    if sha(old_manifest)!=prep['data_manifest_sha256']:raise ValueError('prepared data identity')
    for e in manifest['entries']:
        if sha(e['path'])!=e['sha256']:raise ValueError('old RGB data changed')
    target=OUT/'data/manifest.json';target.parent.mkdir(parents=True,exist_ok=True)
    target.write_bytes(old_manifest.read_bytes())
    records={str(p):sha(p) for p in (PREPARATION,old_manifest,PARENT_OUT/'lock.json',
        PARENT_OUT/'selected.json',PARENT_OUT/'completion.json',parent_completion)}
    references=[]
    for entry in prep['reused_models']:
        role,seed=entry['role'],entry['seed'];path=run_path(role,seed);r=read(path/'run.json')
        if sha(path/'run.json')!=entry['run_sha256'] or sha(path/'model.pt')!=entry['sha256']:
            raise ValueError('prepared reference changed')
        if r.get('training_budget',r.get('steps'))!=5000:raise ValueError('reference budget')
        expected=hashlib.sha256(sampling(seed,576).tobytes()).hexdigest()
        if r['sampling_sha256']!=expected:raise ValueError('reference stream')
        for candidate in (role,)+NEW_ROLES:
            torch.manual_seed(seed);model=make_model(candidate)
            if tensor_hash(common(model))!=r['shared_init_sha256']:
                raise ValueError('common initialization '+candidate)
        references.append(dict(role=role,seed=seed,run_sha256=sha(path/'run.json'),
            checkpoint_sha256=sha(path/'model.pt'),shared_init_sha256=r['shared_init_sha256'],
            sampling_sha256=expected))
    predictions=prep['reused_predictions']
    for entry in predictions:
        path=prediction_path(entry['role'],entry['seed'],entry['variant'])
        if str(path)!=entry['path'] or sha(path)!=entry['sha256'] or sha(path.with_suffix('.json'))!=entry['metadata_sha256']:
            raise ValueError('prepared prediction changed')
        meta=read(path.with_suffix('.json'))
        ref=next(r for r in references if (r['role'],r['seed'])==(entry['role'],entry['seed']))
        if meta['checkpoint_sha256']!=ref['checkpoint_sha256']:raise ValueError('reference checkpoint')
    if len(references)!=24 or len(predictions)!=48:raise ValueError('reference scope')
    for name in SOURCES:
        target=OUT/'source'/name;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((ROOT/name).read_bytes())
    write(OUT/'units.json',dict(passed=True,count=result.testsRun,completed_unix=time.time()))
    write(OUT/'lock.json',dict(config=CONFIG,roles=ROLES,seeds=SEEDS,
        sources={n:sha(ROOT/n) for n in SOURCES},input_records=records,
        data_manifest_sha256=sha(OUT/'data/manifest.json'),references=references,
        reference_predictions=predictions,software=dict(torch=torch.__version__,numpy=np.__version__),
        frozen_unix=time.time(),fresh_confirmation=False,goal_complete=False))
    (OUT/'lock.sha256').write_text(sha(OUT/'lock.json')+'\n')
    print('FROZEN',sha(OUT/'lock.json'),len(SOURCES),'sources',flush=True)


@torch.no_grad()
def validation(model,data,role,save_logits=False):
    model.eval();total=0.;count=0;predictions=[]
    for start in range(0,len(data['q']),CONFIG['batch']):
        d=subset(data,slice(start,start+CONFIG['batch']))
        logits=model(select_inputs(d,role));loss=query_ce(logits,d)
        total+=float(loss[d['gt_valid']].sum());count+=int(d['gt_valid'].sum())
        if save_logits:predictions.append(logits.cpu().numpy())
    if not count:raise ValueError('empty validation')
    return total/count,np.concatenate(predictions) if save_logits else None


def preflight():
    lock=checked();device=setup(0)
    fit,_=load('fit',device);d=subset(fit,slice(0,8));rows=[]
    for role in NEW_ROLES:
        torch.manual_seed(11);model=make_model(role).to(device)
        reference=next(r for r in lock['references'] if r['seed']==11)
        if tensor_hash(common(model))!=reference['shared_init_sha256']:raise ValueError('preflight initialization')
        opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
        torch.cuda.reset_peak_memory_stats();times=[];losses=[];gradients={}
        initial={k:v.detach().clone() for k,v in model.named_parameters() if k.startswith('field.')}
        for step in range(3):
            torch.cuda.synchronize();start=time.monotonic();opt.zero_grad(set_to_none=True)
            loss=objective(model(select_inputs(d,role)),d)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True)
            if not torch.isfinite(loss):raise ValueError('preflight nonfinite')
            gradients={n:float(p.grad.abs().max()) for n,p in model.named_parameters() if n.startswith('field.') and p.grad is not None}
            opt.step();torch.cuda.synchronize();times.append(time.monotonic()-start);losses.append(float(loss.detach()))
        changed={n:float((p-initial[n]).abs().max().detach()) for n,p in model.named_parameters() if n in initial}
        if not any(changed.values()) or not all(math.isfinite(v) for v in gradients.values()):raise ValueError('PE not learning')
        rows.append(dict(role=role,losses=losses,seconds=times,gradients=gradients,parameter_changes=changed,
            total_parameters=sum(p.numel() for p in model.parameters()),pe_parameters=sum(p.numel() for n,p in model.named_parameters() if n.startswith('field.')),
            peak_allocated_bytes=torch.cuda.max_memory_allocated()))
        print('PREFLIGHT',role,losses,'memory',rows[-1]['peak_allocated_bytes'],flush=True)
        del model,opt;gc.collect();torch.cuda.empty_cache()
    write(OUT/'preflight.json',dict(rows=rows,device=torch.cuda.get_device_name(),
        lock_sha256=sha(OUT/'lock.json'),passed=True,scope='three fit-only updates per new role;not an outcome selection'))


def train(role,seed):
    lock=checked();device=setup(0)
    if role not in NEW_ROLES or seed not in SEEDS:raise ValueError('fit scope')
    folder=run_path(role,seed)
    if folder.exists():raise ValueError('preserve existing fit;inspect its original process before resuming')
    folder.mkdir(parents=True)
    fit,_=load('fit',device)
    val,_=load('val',device)
    torch.manual_seed(seed);model=make_model(role).to(device)
    shared=tensor_hash(common(model));initial={n:p.detach().cpu().clone() for n,p in model.named_parameters() if n.startswith('field.')}
    reference=next(r for r in lock['references'] if r['seed']==seed)
    ix=sampling(seed,len(fit['q']),CONFIG['steps'],CONFIG['batch'])
    expected_stream=hashlib.sha256(ix.tobytes()).hexdigest();actual_stream=hashlib.sha256()
    if shared!=reference['shared_init_sha256'] or expected_stream!=reference['sampling_sha256']:
        raise ValueError('initialization/stream mismatch')
    indices=torch.from_numpy(ix).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=CONFIG['lr'],weight_decay=CONFIG['weight_decay'])
    best=math.inf;best_state=None;best_step=None;history=[];interval=[]
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.monotonic()
    for step in range(CONFIG['steps']+1):
        if step%CONFIG['validation_every']==0:
            value,_=validation(model,val,role)
            if not math.isfinite(value):raise ValueError('validation nonfinite')
            if value<best:
                best=value;best_step=step;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            history.append(dict(step=step,val_ce=value,best_step=best_step,
                train_interval_ce=float(np.mean(interval)) if interval else None,elapsed=time.monotonic()-start))
            interval=[];write(folder/'history.json',history)
            if step%500==0:
                save_resume(folder/'resume.pt',model,opt,device,dict(role=role,seed=seed,step=step,
                    best=best,best_step=best_step,best_state=best_state,history=history,
                    sampling_sha256=expected_stream,lock_sha256=sha(OUT/'lock.json')))
            print('TRAIN',role,seed,step,'val',round(value,6),'best',best_step,flush=True)
        if step==CONFIG['steps']:break
        model.train();d=subset(fit,indices[step]);opt.zero_grad(set_to_none=True)
        for group in opt.param_groups:group['lr']=learning_rate(step)
        loss=objective(model(select_inputs(d,role)),d)
        if not torch.isfinite(loss):raise ValueError('training nonfinite')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);opt.step()
        actual_stream.update(ix[step].tobytes());interval.append(float(loss.detach()))
    torch.cuda.synchronize();elapsed=time.monotonic()-start
    if actual_stream.hexdigest()!=expected_stream:raise ValueError('actual sampling changed')
    torch.save(best_state,folder/'model.pt');model.load_state_dict(best_state)
    replay,logits=validation(model,val,role,True)
    if abs(replay-best)>1e-6:raise ValueError('selected validation replay')
    np.savez_compressed(folder/'validation.npz',logits=logits)
    write(folder/'run.json',dict(role=role,seed=seed,best_step=best_step,val_ce=best,steps=CONFIG['steps'],
        shared_init_sha256=shared,shared_final_sha256=tensor_hash(common(model)),
        sampling_sha256=actual_stream.hexdigest(),model_sha256=sha(folder/'model.pt'),
        validation_sha256=sha(folder/'validation.npz'),history_sha256=sha(folder/'history.json'),
        lock_sha256=sha(OUT/'lock.json'),data_manifest_sha256=sha(OUT/'data/manifest.json'),
        observed_data_sha256={},training_selection_seconds=elapsed,
        parameters=sum(p.numel() for p in model.parameters()),
        pe_parameters=sum(p.numel() for n,p in model.named_parameters() if n.startswith('field.')),
        field_changes={n:float((p.detach().cpu()-initial[n]).abs().max()) for n,p in model.named_parameters() if n in initial},
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),fresh_confirmation=False))
    print('DONE',role,seed,elapsed,flush=True)


@torch.no_grad()
def export(seed):
    checked();selection=read(OUT/'selected.json');device=setup(0)
    data,ids=load('old_eval',device);grid=torch.from_numpy(pixels()).to(device)
    permutations=[]
    for i in range(len(ids)):
        rng=np.random.default_rng(510000+i)
        permutations.append(np.stack([rng.permutation(432),rng.permutation(432)]))
    permutations=torch.from_numpy(np.stack(permutations)).to(device)
    for role in NEW_ROLES:
        path=run_path(role,seed);run=read(path/'run.json')
        chosen=next(r for r in selection['runs'] if (r['role'],r['seed'])==(role,seed))
        if sha(path/'run.json')!=chosen['run_sha256'] or sha(path/'model.pt')!=chosen['checkpoint_sha256']:
            raise ValueError('selected model identity')
        model=make_model(role).to(device);model.load_state_dict(torch.load(path/'model.pt',map_location=device,weights_only=True));model.eval()
        for variant in ('baseline','rgb_permuted'):
            dest=prediction_path(role,seed,variant)
            if dest.exists():raise ValueError('preserve predictions')
            result={k:[] for k in ('argmax','soft_xy','target_logp')}
            for start in range(0,len(ids),CONFIG['batch']):
                d=subset(data,slice(start,start+CONFIG['batch']));x=select_inputs(d,role)
                if variant=='rgb_permuted':
                    p=permutations[start:start+len(d['q'])]
                    x=dict(x,q=x['q'].gather(1,p[:,0,:,None].expand_as(x['q'])),
                           k=x['k'].gather(1,p[:,1,:,None].expand_as(x['k'])))
                logits=model(x)
                if not torch.isfinite(logits).all():raise ValueError('prediction nonfinite')
                result['argmax'].append(logits.argmax(-1).cpu().numpy().astype(np.int16))
                result['soft_xy'].append((logits.softmax(-1)@grid).cpu().numpy())
                result['target_logp'].append(logits.log_softmax(-1).gather(-1,d['key_indices'].clamp_min(0)).cpu().numpy())
            result={k:np.concatenate(v) for k,v in result.items()};dest.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(dest,**result);rows,tails=metric_rows(result,data,ids)
            write(dest.with_suffix('.json'),dict(role=role,seed=seed,variant=variant,rows=rows,metrics=macro(rows),scene_tails=tails,
                checkpoint_sha256=chosen['checkpoint_sha256'],prediction_sha256=sha(dest),
                selection_sha256=sha(OUT/'selected.json'),exported_unix=time.time(),
                privileged_diagnostic=role.startswith('observed_'),confirmation_access=False))
            print('EXPORT',role,seed,variant,flush=True)
        del model;gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('freeze','preflight','train','export'))
    parser.add_argument('--role',choices=NEW_ROLES);parser.add_argument('--seed',type=int,choices=SEEDS)
    args=parser.parse_args()
    if args.stage=='freeze':freeze()
    elif args.stage=='preflight':preflight()
    elif args.stage=='train':train(args.role,args.seed)
    else:export(args.seed)
