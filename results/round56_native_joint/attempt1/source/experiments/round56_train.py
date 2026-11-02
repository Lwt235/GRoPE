"""Nine native joint fits with exact R48 budget, paired stream and resumable state."""
import argparse
import hashlib
import math
from pathlib import Path
import random
import shutil
import time
import numpy as np
import torch
import torch.nn.functional as F
from models.round56_native_joint import make_model, INTERVENTIONS
from experiments.round48_data import load, evaluation_points
from experiments.round48_train import configure, shared_digest, score_sample, evaluate, learning_rate
from experiments.round56_common import (OUT, NEW_ROLES, ROLES, SEEDS, CONFIG, VARIANTS,
    checked, sha, read, write, atomic_json, reference, model_path)
from experiments.round56_optimization import epoch_batches, replay_sampler, atomic_save, snapshot, restore


def smoke(out):
    from experiments.round56_audit import check_sample
    checked(out)
    if not torch.cuda.is_available() or not read(out/'preflight.json')['passed']:
        raise ValueError('CUDA access and CPU preflight required')
    rows=[]
    for role in NEW_ROLES:
        torch.manual_seed(5619)
        x=torch.randn(2,37,3,device='cuda');y=torch.tensor([1,9],device='cuda')
        model=make_model(role).cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
        loss=F.cross_entropy(model(x),y,label_smoothing=.1)
        loss.backward()
        gradients={n:None if p.grad is None else float(p.grad.norm()) for n,p in model.position_field.named_parameters()}
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        path=out/('smoke_'+role+'.npz');score_sample(model,x,path)
        numeric=check_sample(role,model.state_dict(),path)
        if not torch.isfinite(loss) or numeric['warning'] or numeric['winner_changes']:
            raise ValueError('CUDA operator smoke discrepancy: '+str(numeric))
        rows.append(dict(role=role,loss=float(loss),gradients=gradients,numeric=numeric))
    torch.manual_seed(5620)
    model=make_model('joint_conditional').cuda().train()
    x=torch.randn(32,512,3,device='cuda');y=torch.arange(32,device='cuda')%40
    torch.cuda.reset_peak_memory_stats()
    F.cross_entropy(model(x),y,label_smoothing=.1).backward()
    peak=torch.cuda.max_memory_allocated()
    if peak>CONFIG['memory_ceiling_bytes']:raise MemoryError('CUDA smoke memory ceiling')
    write(out/'smoke.json',dict(passed=True,rows=rows,gpu=torch.cuda.get_device_name(),
        peak_allocated=peak,real_data_arrays_read=False,unix=time.time()))
    print('CUDA_SMOKE',torch.cuda.get_device_name(),peak,flush=True)


def fit(role,seed,out):
    lock=checked(out)
    if role not in NEW_ROLES or seed not in SEEDS or not read(out/'smoke.json')['passed']:
        raise ValueError('undeclared fit or missing CUDA smoke')
    dest=out/'runs'/role/str(seed)
    if (dest/'run.json').exists():raise ValueError('preserve completed fit')
    dest.mkdir(parents=True,exist_ok=True)
    data,val=load('fit',out),load('val',out)
    train_points=torch.as_tensor(data['points'],device='cuda')
    train_labels=torch.as_tensor(data['labels'],device='cuda')
    val_points=torch.as_tensor(evaluation_points(val,'base'),device='cuda')
    val_labels=torch.as_tensor(val['labels'],device='cuda')
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    model=make_model(role).cuda()
    initial_shared=shared_digest(model)
    expected=reference('rope',seed,lock)
    if initial_shared!=expected['initial_shared_sha256']:raise ValueError('unpaired initialization')
    initial_position={n:p.detach().cpu().clone() for n,p in model.position_field.named_parameters()}
    optimizer=torch.optim.AdamW(model.parameters(),lr=CONFIG['learning_rate'],weight_decay=CONFIG['weight_decay'])
    rng=np.random.default_rng(seed+48000);stream_hash=hashlib.sha256()
    history=[];training_history=[];best=float('inf');best_epoch=None;best_state=None;best_logits=None
    first_gradients={};step=0;completed=0;elapsed_before=0.;peak_before=0
    count=len(train_points);steps_per_epoch=math.ceil(count/CONFIG['batch_size'])
    resume=dest/'resume.pt';resumed=False
    started=time.monotonic();deadline=started+CONFIG['shard_seconds']
    torch.cuda.reset_peak_memory_stats()
    if resume.exists():
        saved=torch.load(resume,map_location='cpu',weights_only=False)
        if saved['role']!=role or saved['seed']!=seed or saved['lock_sha256']!=sha(out/'lock.json'):
            raise ValueError('resume identity mismatch')
        completed=saved['epoch_completed'];step=saved['step']
        if step!=completed*steps_per_epoch:raise ValueError('resume budget mismatch')
        rng,stream_hash=replay_sampler(seed,completed,count)
        if stream_hash.hexdigest()!=saved['sampling_sha256'] or rng.bit_generator.state!=saved['numpy_generator_state']:
            raise ValueError('resume sampler replay mismatch')
        archive=dest/'interruptions'/str(time.time_ns());archive.mkdir(parents=True)
        for name in ('best.pt','validation.npz','history.json','interrupted.json'):
            if (dest/name).exists():shutil.copyfile(dest/name,archive/name)
        write(archive/'resume_record.json',dict(resume_sha256=sha(resume),epoch=completed,step=step,
            sampling_replay_exact=True,unix=time.time()))
        restore(model,optimizer,saved['optimization'])
        history=saved['validation_history'];training_history=saved['training_history']
        best=saved['best'];best_epoch=saved['best_epoch'];best_state=saved['best_state'];best_logits=saved['best_logits']
        first_gradients=saved['first_gradients'];elapsed_before=saved['seconds'];peak_before=saved['peak_allocated']
        resumed=True
        atomic_save(dest/'best.pt',dict(role=role,seed=seed,epoch=best_epoch,state_dict=best_state))
        np.savez_compressed(dest/'validation.npz',logits=best_logits,labels=val['labels'],ids=val['ids'])
        atomic_json(dest/'history.json',dict(validation=history,training=training_history))
        print('RESUME_EXACT',role,seed,completed,step,flush=True)

    def select(epoch):
        nonlocal best,best_epoch,best_state,best_logits
        logits,stats=evaluate(model,val_points,val_labels)
        history.append(dict(epoch=epoch,step=step,**stats))
        if stats['ce']<best:
            best,best_epoch=stats['ce'],epoch
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            best_logits=logits
            atomic_save(dest/'best.pt',dict(role=role,seed=seed,epoch=epoch,state_dict=best_state))
            np.savez_compressed(dest/'validation.npz',logits=logits,labels=val['labels'],ids=val['ids'])
        atomic_json(dest/'history.json',dict(validation=history,training=training_history))
        print('VAL',role,seed,epoch,stats['ce'],stats['oa'],flush=True)

    def save_state(epoch):
        atomic_save(resume,dict(role=role,seed=seed,lock_sha256=sha(out/'lock.json'),
            epoch_completed=epoch,step=step,optimization=snapshot(model,optimizer),
            numpy_generator_state=rng.bit_generator.state,sampling_sha256=stream_hash.hexdigest(),
            validation_history=history,training_history=training_history,best=best,best_epoch=best_epoch,
            best_state=best_state,best_logits=best_logits,first_gradients=first_gradients,
            seconds=elapsed_before+time.monotonic()-started,
            peak_allocated=max(peak_before,torch.cuda.max_memory_allocated())))

    if not resumed:
        select(0);save_state(0)
    for epoch in range(completed,CONFIG['epochs']):
        epoch_loss=0.
        for index,subset,angle,jitter in epoch_batches(rng,count,stream_hash):
            ii=torch.as_tensor(index,device='cuda');jj=torch.as_tensor(subset,device='cuda')
            x=train_points[ii[:,None],jj];angles=torch.as_tensor(angle,device='cuda')
            c,s=angles.cos()[:,None],angles.sin()[:,None]
            x=torch.stack((c*x[...,0]+s*x[...,2],x[...,1],-s*x[...,0]+c*x[...,2]),-1)
            x=x+torch.as_tensor(jitter,device='cuda');y=train_labels[ii]
            torch.manual_seed(seed*100000000+step)
            model.train();optimizer.zero_grad(set_to_none=True)
            for group in optimizer.param_groups:group['lr']=learning_rate(step,steps_per_epoch)
            loss=F.cross_entropy(model(x),y,label_smoothing=CONFIG['label_smoothing'])
            if not torch.isfinite(loss):raise ValueError('nonfinite loss')
            loss.backward()
            if step==0:
                first_gradients={n:None if p.grad is None else float(p.grad.norm()) for n,p in model.named_parameters()}
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),CONFIG['clip_norm'],error_if_nonfinite=True)
            optimizer.step();epoch_loss+=float(loss)*len(index);step+=1
        completed=epoch+1
        training_history.append(dict(epoch=completed,loss=epoch_loss/count,steps=step,last_gradient_norm=float(norm)))
        if completed%CONFIG['validation_every']==0:
            select(completed);save_state(completed)
        if torch.cuda.max_memory_allocated()>CONFIG['memory_ceiling_bytes']:
            raise MemoryError('declared allocation ceiling')
        if time.monotonic()>deadline:
            save_state(completed)
            write(dest/'interrupted.json',dict(epoch=completed,reason='eight-hour worker limit',completed=False))
            raise TimeoutError('resume existing state; no new budget')
    if stream_hash.hexdigest()!=expected['batch_augmentation_sha256']:
        raise ValueError('actual complete sampling stream differs from R48')
    model.load_state_dict(best_state);score_sample(model,val_points,dest/'score_sample.npz')
    record=dict(role=role,seed=seed,epochs=120,steps=step,best_epoch=best_epoch,best_val_ce=best,
        initial_shared_sha256=initial_shared,batch_augmentation_sha256=stream_hash.hexdigest(),
        exact_original_sampling=True,first_gradients=first_gradients,
        shared_parameters_changed=shared_digest(model)!=initial_shared,
        position_parameter_changes={n:float((p.detach().cpu()-initial_position[n]).norm()) for n,p in model.position_field.named_parameters()},
        parameters=sum(p.numel() for p in model.parameters()),positional_parameters=sum(p.numel() for p in model.position_field.parameters()),
        seconds=elapsed_before+time.monotonic()-started,peak_allocated=max(peak_before,torch.cuda.max_memory_allocated()),
        resumed=resumed,lock_sha256=sha(out/'lock.json'),
        hashes={n:sha(dest/n) for n in ('best.pt','history.json','validation.npz','score_sample.npz')})
    atomic_json(dest/'run.json',record)
    print('DONE',role,seed,record['seconds'],'original_sampling_exact',flush=True)


@torch.no_grad()
def export(seed,out):
    lock=checked(out)
    seal=read(out/'audit_training/completion.json')
    if seed not in SEEDS or len(seal['records'])!=len(ROLES)*len(SEEDS):
        raise ValueError('all39 selected models must be sealed')
    data=load('dev_eval',out);labels=torch.as_tensor(data['labels'],device='cuda')
    for role in ROLES:
        saved=next(r for r in seal['records'] if r['role']==role and r['seed']==seed)
        path=model_path(role,seed,out)
        if sha(path/'best.pt')!=saved['checkpoint_sha256'] or sha(path/'run.json')!=saved['run_sha256']:
            raise ValueError('selected model changed')
        dest=out/'evaluation'/role;dest.mkdir(parents=True,exist_ok=True)
        npz=dest/(str(seed)+'.npz');meta=dest/(str(seed)+'.json')
        if npz.exists() or meta.exists():raise ValueError('preserve existing evaluation')
        if role not in NEW_ROLES:
            ref=reference(role,seed,lock)
            for name in ('dev_eval.npz','dev_eval.json'):
                if sha(path/name)!=ref['hashes'][name]:raise ValueError('reference prediction changed')
            shutil.copyfile(path/'dev_eval.npz',npz)
            record=read(path/'dev_eval.json')
            record.update(reused=True,source=str(path/'dev_eval.npz'),source_record_sha256=sha(path/'dev_eval.json'))
        else:
            model=make_model(role).cuda().eval()
            model.load_state_dict(torch.load(path/'best.pt',map_location='cpu',weights_only=True)['state_dict'])
            arrays=dict(labels=data['labels'],ids=data['ids']);metrics={}
            for variant in VARIANTS:
                points=torch.as_tensor(evaluation_points(data,variant),device='cuda')
                arrays[variant],metrics[variant]=evaluate(model,points,labels)
                del points
            np.savez_compressed(npz,**arrays)
            record=dict(role=role,seed=seed,metrics=metrics,checkpoint_sha256=saved['checkpoint_sha256'],
                predictions_sha256=sha(npz),reused=False,official_test_access=False,fresh_confirmation=False)
        record.update(export_unix=time.time(),selection_seal_sha256=sha(out/'audit_training/completion.json'))
        write(meta,record)
        print('EXPORT',role,seed,record['metrics']['base'],flush=True)
    base_points=torch.as_tensor(evaluation_points(data,'base'),device='cuda')
    for role in ('joint_conditional','conditional'):
        path=model_path(role,seed,out)
        state=torch.load(path/'best.pt',map_location='cpu',weights_only=True)['state_dict']
        for mode in INTERVENTIONS:
            model=make_model(role,mode).cuda().eval();model.load_state_dict(state)
            logits,metric=evaluate(model,base_points,labels)
            dest=out/'diagnostics'/role;dest.mkdir(parents=True,exist_ok=True)
            name=str(seed)+'_'+mode
            if (dest/(name+'.json')).exists():raise ValueError('preserve diagnostics')
            np.savez_compressed(dest/(name+'.npz'),base=logits,labels=data['labels'],ids=data['ids'])
            write(dest/(name+'.json'),dict(role=role,seed=seed,mode=mode,metrics=metric,
                checkpoint_sha256=sha(path/'best.pt'),predictions_sha256=sha(dest/(name+'.npz')),
                source_selection_seal_sha256=sha(out/'audit_training/completion.json'),
                trained=False,fresh_confirmation=False,official_test_access=False,unix=time.time()))
            print('DIAGNOSTIC',role,seed,mode,metric,flush=True)
    write(out/f'export_{seed}_complete.json',dict(completed=True,unix=time.time(),official_test_access=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=('smoke','train','export'))
    parser.add_argument('--out',type=Path,default=OUT)
    parser.add_argument('--role',choices=NEW_ROLES)
    parser.add_argument('--seed',type=int)
    args=parser.parse_args();configure()
    if args.stage=='train':fit(args.role,args.seed,args.out)
    elif args.stage=='export':export(args.seed,args.out)
    else:smoke(args.out)
