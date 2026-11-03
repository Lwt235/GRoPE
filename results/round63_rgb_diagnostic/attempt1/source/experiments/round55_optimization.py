"""Continuous AdamW extension with explicit original-prefix and resume records."""
import copy
import gc
import hashlib
import math
import os
import time
import numpy as np
import torch


def learning_rate(step):
    return .001*(.2+.8*(1+math.cos(math.pi*step/1000))/2) if step<1000 else .0002


def sampling(seed,size,steps=5000,batch=8):
    return np.random.default_rng(seed+50000).integers(0,size,size=(steps,batch),dtype=np.int64)


def cpu_state(model):
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}


def save_resume(path,model,optimizer,device,metadata):
    state=dict(metadata,model=cpu_state(model),optimizer=optimizer.state_dict(),
        cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(device) if device.type=='cuda' else None,
        model_training=model.training)
    temporary=path.with_suffix('.tmp')
    torch.save(state,temporary);os.replace(temporary,path)


def restore_resume(path,model,optimizer,device):
    state=torch.load(path,map_location=device,weights_only=False)
    model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
    torch.set_rng_state(state['cpu_rng'].cpu())
    if device.type=='cuda':torch.cuda.set_rng_state(state['cuda_rng'].cpu(),device)
    model.train(state['model_training'])
    if state.get('best_state') is not None:
        state['best_state']={k:v.cpu() for k,v in state['best_state'].items()}
    return state


@torch.no_grad()
def prefix_audit(P,model,best_state,best_step,history,entry,val):
    original=P.read(entry['run_path'])
    if P.sha(entry['run_path'])!=entry['run_sha256'] or P.sha(entry['checkpoint_path'])!=entry['checkpoint_sha256']:
        raise ValueError('original prefix model identity changed')
    reference=torch.load(entry['checkpoint_path'],map_location='cpu',weights_only=True)
    weight_delta=max(float((v.cpu()-reference[k]).abs().max()) for k,v in best_state.items())
    old_history={r['step']:r['val_ce'] for r in original['history']}
    deltas=[r['val_ce']-old_history[r['step']] for r in history if r['step']<=1000]
    candidate=copy.deepcopy(model).eval();candidate.load_state_dict(best_state)
    baseline=copy.deepcopy(model).eval();baseline.load_state_dict(reference)
    sample=P.inputs(P.subset(val,slice(0,8)))
    function_delta=float((candidate(sample)-baseline(sample)).abs().max())
    result=dict(origin=entry,best_step=best_step,original_best_step=original['best_step'],
        selected_state_max_difference=weight_delta,selected_function_max_difference=function_delta,
        history_ce_differences=deltas,history_ce_max_difference=max(abs(d) for d in deltas),
        original_history_points=len(old_history),prefix_history_points=len(deltas),
        warning=bool(best_step!=original['best_step'] or function_delta>5e-5 or max(abs(d) for d in deltas)>5e-6),
        scope='original1000-step training process; no old evaluation accessed')
    del candidate,baseline
    return result


def fit_one(P,role,seed,fit,val,device):
    folder=P.OUT/'runs'/role/str(seed);lock_digest=P.sha(P.OUT/'lock.json')
    if (folder/'run.json').exists():
        record=P.read(folder/'run.json')
        if record['lock_sha256']!=lock_digest or P.sha(folder/'model.pt')!=record['model_sha256']:
            raise ValueError('completed run identity mismatch')
        print('RETAIN COMPLETED',role,seed,flush=True);return
    if folder.exists() and not (folder/'resume.pt').exists():
        raise FileExistsError('incomplete folder without a resumable state; preserve it')
    folder.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(seed);model=P.make_model(role).to(device)
    initial=cpu_state(model);init_hash=P.tensor_hash(P.common(model))
    ix=sampling(seed,len(fit['q']),P.CONFIG['steps'],P.CONFIG['batch'])
    sampling_hash=hashlib.sha256(ix.tobytes()).hexdigest()
    prefix_hash=hashlib.sha256(ix[:1000].tobytes()).hexdigest()
    indices=torch.from_numpy(ix).to(device)
    entry=next(e for e in P.read(P.OUT/'prefix_manifest.json')['entries'] if (e['role'],e['seed'])==(role,seed))
    if init_hash!=entry['shared_init_sha256'] or prefix_hash!=entry['prefix_sampling_sha256']:
        raise ValueError('original initialization/batch prefix changed')
    if P.sha(entry['origin_lock_path'])!=entry['origin_lock_sha256']:raise ValueError('original lock changed')
    optimizer=torch.optim.AdamW(model.parameters(),lr=P.CONFIG['lr'],weight_decay=P.CONFIG['weight_decay'])
    best=math.inf;best_state=None;best_step=None;history=[];start_step=0
    elapsed_prior=0.;peak_prior=0;prefix_record=None;interval_losses=[]
    if (folder/'resume.pt').exists():
        state=restore_resume(folder/'resume.pt',model,optimizer,device)
        for key,value in (('lock_sha256',lock_digest),('role',role),('seed',seed),('sampling_sha256',sampling_hash)):
            if state[key]!=value:raise ValueError('resume identity mismatch '+key)
        best=state['best'];best_state=state['best_state'];best_step=state['best_step'];history=state['history']
        start_step=state['step'];elapsed_prior=state['elapsed'];peak_prior=state['peak_allocated_bytes']
        prefix_record=state['prefix_record']
        P.write(folder/'resume_event.json',dict(step=start_step,time=time.time(),lock_sha256=lock_digest,
            restored_optimizer=True,restored_rng=True,retained_history_points=len(history)))
        print('RESTORE',role,seed,'step',start_step,flush=True)
    torch.cuda.reset_peak_memory_stats(device);torch.cuda.synchronize(device);begin=time.monotonic()
    for step in range(start_step,P.CONFIG['steps']+1):
        already_validated=bool(history and history[-1]['step']==step)
        if step%P.CONFIG['validation_every']==0 and not already_validated:
            value=P.validation(model,val)
            if not math.isfinite(value):raise ValueError('validation nonfinite')
            if value<best:best=value;best_step=step;best_state=cpu_state(model)
            history.append(dict(step=step,val_ce=value,best_step=best_step,
                train_interval_ce=float(np.mean(interval_losses)) if interval_losses else None,
                train_interval_updates=len(interval_losses),elapsed=elapsed_prior+time.monotonic()-begin))
            interval_losses=[]
            if step==1000:
                prefix_record=prefix_audit(P,model,best_state,best_step,history,entry,val)
                P.write(folder/'prefix_check.json',prefix_record)
                torch.save(best_state,folder/'prefix_model.pt')
                print('PREFIX',role,seed,'function_error',prefix_record['selected_function_max_difference'],
                    'CE_error',prefix_record['history_ce_max_difference'],'warning',prefix_record['warning'],flush=True)
            P.write(folder/'history.json',history)
            peak=max(peak_prior,torch.cuda.max_memory_allocated(device))
            save_resume(folder/'resume.pt',model,optimizer,device,dict(role=role,seed=seed,step=step,
                best=best,best_state=best_state,best_step=best_step,history=history,
                elapsed=elapsed_prior+time.monotonic()-begin,peak_allocated_bytes=peak,
                sampling_sha256=sampling_hash,lock_sha256=lock_digest,prefix_record=prefix_record))
            print('TRAIN',seed,role,step,'val',round(value,6),'best',best_step,flush=True)
        if step==P.CONFIG['steps']:break
        model.train();d=P.subset(fit,indices[step]);optimizer.zero_grad(set_to_none=True)
        for group in optimizer.param_groups:group['lr']=learning_rate(step)
        loss=P.objective(model(P.inputs(d)),d)
        if not torch.isfinite(loss):raise ValueError('training nonfinite')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);optimizer.step()
        interval_losses.append(float(loss.detach()))
    torch.cuda.synchronize(device);elapsed=elapsed_prior+time.monotonic()-begin
    if prefix_record is None:raise ValueError('missing1000-step comparison')
    torch.save(best_state,folder/'model.pt')
    P.write(folder/'run.json',dict(role=role,seed=seed,best_step=best_step,val_ce=best,
        shared_init_sha256=init_hash,shared_final_sha256=P.tensor_hash({k:v for k,v in best_state.items() if not k.startswith('field.')}),
        sampling_sha256=sampling_hash,prefix_sampling_sha256=prefix_hash,model_sha256=P.sha(folder/'model.pt'),
        lock_sha256=lock_digest,history=history,parameters=sum(p.numel() for p in model.parameters()),
        pe_parameters=sum(p.numel() for n,p in model.named_parameters() if n.startswith('field.')),
        field_changes={k:float((v-initial[k]).abs().max()) for k,v in best_state.items() if k.startswith('field.')},
        training_selection_seconds=elapsed,peak_allocated_bytes=max(peak_prior,torch.cuda.max_memory_allocated(device)),
        prefix_check_sha256=P.sha(folder/'prefix_check.json'),prefix_warning=prefix_record['warning'],
        original_run_path=entry['run_path'],original_run_sha256=entry['run_sha256'],
        training_budget=5000,reused=False,confirmation_access=False))
    print('DONE',role,seed,round(elapsed,2),'best',best_step,flush=True)
    del model,optimizer;gc.collect();torch.cuda.empty_cache()
