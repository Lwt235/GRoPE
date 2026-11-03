"""Paired, resumable native-GS classification training and sealed evaluation."""
import argparse
import hashlib
import math
import os
from pathlib import Path
import random
import time
import numpy as np  # Load NumPy before Torch on this host.
import torch
import torch.nn.functional as F
from models.round65_actual_gs import NativeGSClassifier
from experiments.round65_common import (ROOT,OUT,PREP,CONFIG,ROLES,SEEDS,VARIANTS,
                                       checked,identity_seed,read,sha,write)
from experiments.round65_data import load,quaternion_rotation


def configure():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True)


def shared_digest(model):
    h=hashlib.sha256()
    for name,tensor in model.state_dict().items():
        if not name.startswith('position.'):
            h.update(name.encode());h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def atomic_save(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    torch.save(value,temporary);temporary.replace(path)


def stream_update(previous,*arrays):
    h=hashlib.sha256(bytes.fromhex(previous))
    for array in arrays:
        value=np.ascontiguousarray(array)
        h.update(str(value.dtype).encode());h.update(str(value.shape).encode());h.update(value.tobytes())
    return h.hexdigest()


def epoch_batches(rng,count,batch_size=None,tokens=None):
    batch_size=batch_size or CONFIG['batch_size'];tokens=tokens or CONFIG['tokens']
    order=rng.permutation(count)
    for start in range(0,count,batch_size):
        index=order[start:start+batch_size]
        subset=np.stack([rng.choice(CONFIG['cache_tokens'],tokens,replace=False) for _ in index])
        quaternion=rng.normal(size=(len(index),4))
        jitter=rng.normal(0,CONFIG['jitter_std'],size=(len(index),tokens,3))
        jitter=jitter.clip(-CONFIG['jitter_clip'],CONFIG['jitter_clip']).astype(np.float32)
        yield index,subset,quaternion,jitter


def augment(centers,covariance,quaternion,jitter):
    rotation=torch.as_tensor(quaternion_rotation(quaternion),device=centers.device,dtype=torch.float64)
    centers=centers@rotation.to(centers.dtype).transpose(-1,-2)
    centers=centers+torch.as_tensor(jitter,device=centers.device,dtype=centers.dtype)
    r=rotation[:,None]
    return centers,r@covariance@r.transpose(-1,-2)


def learning_rate(step,steps_per_epoch):
    warm=CONFIG['warmup_epochs']*steps_per_epoch
    total=CONFIG['epochs']*steps_per_epoch
    if step<warm:
        return CONFIG['learning_rate']*(step+1)/warm
    progress=(step-warm)/max(1,total-warm-1)
    return CONFIG['final_learning_rate']+.5*(CONFIG['learning_rate']-CONFIG['final_learning_rate'])*(1+math.cos(math.pi*progress))


def metrics(logits,labels):
    x=torch.as_tensor(logits,dtype=torch.float64);y=torch.as_tensor(np.asarray(labels).copy(),dtype=torch.long)
    if x.shape!=(len(y),60) or not torch.isfinite(x).all():
        raise ValueError('invalid classification output')
    correct=x.argmax(-1)==y
    class_accuracy=[float(correct[y==i].double().mean()) if torch.any(y==i) else None for i in range(60)]
    present=[v for v in class_accuracy if v is not None]
    return dict(ce=float(F.cross_entropy(x,y)),oa=float(correct.double().mean()),
                mca=sum(present)/len(present),class_accuracy=class_accuracy,count=len(y))


def evaluation_batch(bundle,start,end,variant,device):
    tokens=256 if variant=='sparse256' else 1024 if variant=='dense1024' else CONFIG['tokens']
    # Canonical cached sample order is fixed per object before training.
    centers=torch.tensor(np.array(bundle['centers'][start:end,:tokens]),device=device)
    covariance=torch.tensor(np.array(bundle['covariance'][start:end,:tokens]),device=device)
    appearance=torch.tensor(np.array(bundle['appearance'][start:end,:tokens]),device=device)
    if variant=='jitter02':
        noise=[]
        for identity in bundle['ids'][start:end]:
            rng=np.random.default_rng(identity_seed('R65-eval-jitter02-v1',str(identity)))
            noise.append(rng.normal(0,.02,size=(tokens,3)).clip(-.1,.1).astype(np.float32))
        centers+=torch.tensor(np.stack(noise),device=device)
    elif variant=='tilt':
        x,z=np.pi/6,np.pi/4
        rx=np.array([[1,0,0],[0,np.cos(x),-np.sin(x)],[0,np.sin(x),np.cos(x)]])
        rz=np.array([[np.cos(z),-np.sin(z),0],[np.sin(z),np.cos(z),0],[0,0,1]])
        r=torch.tensor(rz@rx,device=device,dtype=torch.float64)
        centers=centers@r.to(centers.dtype).T;covariance=r@covariance@r.T
    elif variant=='permuted':
        centers=centers.flip(1);covariance=covariance.flip(1);appearance=appearance.flip(1)
    elif variant not in ('base','sparse256','dense1024'):
        raise ValueError('undeclared evaluation variant')
    return centers,covariance,appearance


@torch.no_grad()
def evaluate(model,bundle,variant='base',limit=None):
    model.eval();device=next(model.parameters()).device
    stop=len(bundle['labels']) if limit is None else min(limit,len(bundle['labels']))
    logits=[]
    for start in range(0,stop,CONFIG['evaluation_batch_size']):
        args=evaluation_batch(bundle,start,min(stop,start+CONFIG['evaluation_batch_size']),variant,device)
        logits.append(model(*args).cpu().numpy())
    result=np.concatenate(logits)
    return result,metrics(result,bundle['labels'][:stop])


def synthetic_inputs(batch,tokens,device):
    rng=np.random.default_rng(65004)
    c=torch.tensor(rng.normal(size=(batch,tokens,3)),dtype=torch.float32,device=device)
    q=rng.normal(size=(batch,tokens,4));r=quaternion_rotation(q)
    values=np.exp(rng.uniform(-9,-2,size=(batch,tokens,3)))
    cov=(r*values[...,None,:])@r.swapaxes(-1,-2)+1e-8*np.eye(3)
    s=torch.tensor(cov,device=device,dtype=torch.float64)
    a=torch.tensor(rng.normal(size=(batch,tokens,4)),device=device,dtype=torch.float32)
    return c,s,a


def preflight(destination):
    configure()
    if not torch.cuda.is_available():raise RuntimeError('CUDA required for training preflight')
    destination=Path(destination)
    if destination.exists():raise ValueError('preserve prior CUDA preflight')
    destination.mkdir(parents=True)
    rows=[]
    try:
        for role in ROLES:
            torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
            torch.manual_seed(11);model=NativeGSClassifier(role).cuda().train()
            before=shared_digest(model)
            initial={n:p.detach().cpu().clone() for n,p in model.position.named_parameters()}
            c,s,a=synthetic_inputs(CONFIG['batch_size'],CONFIG['tokens'],'cuda')
            optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
            y=torch.arange(len(c),device='cuda')%60
            started=time.monotonic();loss=F.cross_entropy(model(c,s,a),y,label_smoothing=.1)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            gradients={n:None if p.grad is None else float(p.grad.norm()) for n,p in model.position.named_parameters()}
            optimizer.step();torch.cuda.synchronize()
            peak=torch.cuda.max_memory_allocated()
            changes={n:float((p.detach().cpu()-initial[n]).norm()) for n,p in model.position.named_parameters()}
            if not torch.isfinite(loss) or before==shared_digest(model) or any(v<=0 for v in changes.values()):
                raise ValueError('nonfinite or inactive training preflight')
            if peak>CONFIG['memory_ceiling_bytes']:raise MemoryError('declared allocation ceiling exceeded')
            rows.append(dict(role=role,loss=float(loss),seconds=time.monotonic()-started,peak_allocated=peak,
                gradients=gradients,position_changes=changes,parameters=sum(p.numel() for p in model.parameters()),
                position_parameters=sum(p.numel() for p in model.position.parameters()),initial_shared_sha256=before))
            del model,optimizer,loss,c,s,a,y
            print('CUDA_PREFLIGHT',role,rows[-1]['seconds'],peak,flush=True)
        sources=['models/round65_actual_gs.py','models/round58_support_pe.py','models/geometry.py',
                 'experiments/round65_run.py','experiments/round65_common.py','experiments/round65_data.py']
        write(destination/'result.json',dict(passed=True,rows=rows,config=CONFIG,
            gpu=torch.cuda.get_device_name(),torch_version=torch.__version__,
            source_sha256={n:sha(ROOT/n) for n in sources},real_vertex_arrays_decoded=False))
    except BaseException as error:
        write(destination/'failure.json',dict(error_type=type(error).__name__,message=str(error),completed_roles=rows))
        raise


def fit(role,seed,out=OUT):
    configure();lock=checked(out)
    if role not in ROLES or seed not in SEEDS or not torch.cuda.is_available():
        raise ValueError('undeclared fit or no CUDA')
    if not read(out/'cuda_preflight.json')['passed']:raise ValueError('CUDA preflight required')
    dest=out/'runs'/role/str(seed)
    if (dest/'run.json').exists():raise ValueError('preserve completed fit')
    dest.mkdir(parents=True,exist_ok=True)
    data,val=load('fit',out),load('val',out)
    tensors={k:torch.tensor(np.array(data[k]),device='cuda') for k in ('centers','covariance','appearance','labels')}
    count=len(data['labels']);steps_per_epoch=math.ceil(count/CONFIG['batch_size'])
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    model=NativeGSClassifier(role).cuda();initial_shared=shared_digest(model)
    initial_position={n:p.detach().cpu().clone() for n,p in model.position.named_parameters()}
    optimizer=torch.optim.AdamW(model.parameters(),lr=CONFIG['learning_rate'],weight_decay=CONFIG['weight_decay'])
    rng=np.random.default_rng(seed+65000)
    stream=hashlib.sha256(('R65-sampler-v1:'+str(seed)).encode()).hexdigest()
    history=[];training=[];best=float('inf');best_epoch=None;step=0;completed=0
    best_state=None;best_logits=None;first_gradients={};elapsed_before=0.;peak_before=0;resumed=False
    resume=dest/'resume.pt';manifest_sha=sha(out/'data/manifest.json');lock_sha=sha(out/'lock.json')
    started=time.monotonic();deadline=started+CONFIG['shard_seconds'];torch.cuda.reset_peak_memory_stats()
    if resume.exists():
        saved=torch.load(resume,map_location='cpu',weights_only=False)
        if (saved['role'],saved['seed'],saved['lock_sha256'],saved['data_sha256'])!=(role,seed,lock_sha,manifest_sha):
            raise ValueError('resume identity mismatch')
        completed,step=saved['epoch_completed'],saved['step']
        if step!=completed*steps_per_epoch:raise ValueError('resume budget mismatch')
        model.load_state_dict(saved['state_dict']);optimizer.load_state_dict(saved['optimizer'])
        rng.bit_generator.state=saved['sampler_state'];stream=saved['stream_sha256']
        history,training=saved['validation'],saved['training']
        best,best_epoch,best_state,best_logits=saved['best'],saved['best_epoch'],saved['best_state'],saved['best_logits']
        first_gradients=saved['first_gradients'];elapsed_before=saved['seconds'];peak_before=saved['peak_allocated']
        write(dest/f'resume_{time.time_ns()}.json',dict(snapshot_sha256=sha(resume),epoch_completed=completed,
            steps=step,unchanged_budget=True))
        resumed=True
        atomic_save(dest/'best.pt',dict(role=role,seed=seed,epoch=best_epoch,state_dict=best_state))
        np.savez(dest/'validation.npz',logits=best_logits,labels=val['labels'],ids=val['ids'])

    def select(epoch):
        nonlocal best,best_epoch,best_state,best_logits
        logits,stats=evaluate(model,val)
        history.append(dict(epoch=epoch,step=step,**stats))
        if stats['ce']<best:
            best,best_epoch=stats['ce'],epoch;best_logits=logits
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            atomic_save(dest/'best.pt',dict(role=role,seed=seed,epoch=epoch,state_dict=best_state))
            np.savez(dest/'validation.npz',logits=logits,labels=val['labels'],ids=val['ids'])
        write(dest/'history.json',dict(validation=history,training=training))
        print('VAL',role,seed,epoch,stats['ce'],stats['mca'],flush=True)

    def save_state():
        atomic_save(resume,dict(role=role,seed=seed,lock_sha256=lock_sha,data_sha256=manifest_sha,
            epoch_completed=completed,step=step,state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},
            optimizer=optimizer.state_dict(),sampler_state=rng.bit_generator.state,stream_sha256=stream,
            validation=history,training=training,best=best,best_epoch=best_epoch,best_state=best_state,
            best_logits=best_logits,first_gradients=first_gradients,seconds=elapsed_before+time.monotonic()-started,
            peak_allocated=max(peak_before,torch.cuda.max_memory_allocated())))

    if not resumed:select(0);save_state()
    try:
        for epoch in range(completed,CONFIG['epochs']):
            total_loss=0.
            for index,subset,quaternion,jitter in epoch_batches(rng,count):
                stream=stream_update(stream,index,subset,quaternion,jitter)
                ii=torch.as_tensor(index,device='cuda');jj=torch.as_tensor(subset,device='cuda')
                c,s,a=(tensors[k][ii[:,None],jj] for k in ('centers','covariance','appearance'))
                c,s=augment(c,s,quaternion,jitter);labels=tensors['labels'][ii]
                torch.manual_seed(seed*10000000+step);model.train();optimizer.zero_grad(set_to_none=True)
                for group in optimizer.param_groups:group['lr']=learning_rate(step,steps_per_epoch)
                loss=F.cross_entropy(model(c,s,a),labels,label_smoothing=CONFIG['label_smoothing'])
                if not torch.isfinite(loss):raise ValueError('nonfinite training loss')
                loss.backward()
                if step==0:
                    first_gradients={n:None if p.grad is None else float(p.grad.norm()) for n,p in model.named_parameters()}
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),CONFIG['clip_norm'],error_if_nonfinite=True)
                optimizer.step();total_loss+=float(loss)*len(index);step+=1
            completed=epoch+1
            training.append(dict(epoch=completed,step=step,loss=total_loss/count,stream_sha256=stream,last_gradient_norm=float(norm)))
            if completed%CONFIG['validation_every']==0:select(completed)
            save_state()
            if torch.cuda.max_memory_allocated()>CONFIG['memory_ceiling_bytes']:raise MemoryError('declared allocation ceiling')
            if time.monotonic()>deadline and completed<CONFIG['epochs']:
                print('SHARD_COMPLETE',role,seed,completed,flush=True)
                return 75
        model.load_state_dict(best_state)
        changes={n:float((p.detach().cpu()-initial_position[n]).norm()) for n,p in model.position.named_parameters()}
        record=dict(role=role,seed=seed,epochs=completed,steps=step,best_epoch=best_epoch,best_val_ce=best,
            initial_shared_sha256=initial_shared,shared_parameters_changed=shared_digest(model)!=initial_shared,
            first_gradients=first_gradients,position_parameter_changes=changes,sampling_sha256=stream,
            parameters=sum(p.numel() for p in model.parameters()),position_parameters=sum(p.numel() for p in model.position.parameters()),
            seconds=elapsed_before+time.monotonic()-started,peak_allocated=max(peak_before,torch.cuda.max_memory_allocated()),
            resumed=resumed,lock_sha256=lock_sha,data_sha256=manifest_sha,
            hashes={n:sha(dest/n) for n in ('best.pt','validation.npz','history.json','resume.pt')})
        write(dest/'run.json',record);print('FIT_COMPLETE',role,seed,record['seconds'],flush=True)
        return 0
    except BaseException as error:
        write(dest/f'failure_{time.time_ns()}.json',dict(error_type=type(error).__name__,message=str(error),
            step=step,last_complete_epoch=completed,resume_exists=resume.exists()))
        raise


@torch.no_grad()
def export(seed,out=OUT):
    configure();checked(out);seal=read(out/'selected.json')
    if seed not in SEEDS or len(seal['records'])!=len(ROLES)*len(SEEDS):raise ValueError('complete selection seal required')
    bundle=load('dev_eval',out)
    for role in ROLES:
        entry=next(r for r in seal['records'] if r['role']==role and r['seed']==seed)
        run=out/'runs'/role/str(seed)
        if sha(run/'best.pt')!=entry['checkpoint_sha256'] or sha(run/'run.json')!=entry['run_sha256']:
            raise ValueError('selected model identity changed')
        dest=out/'evaluation'/role/str(seed)
        if dest.exists():raise ValueError('preserve existing evaluation')
        dest.mkdir(parents=True)
        model=NativeGSClassifier(role).cuda().eval()
        model.load_state_dict(torch.load(run/'best.pt',map_location='cpu',weights_only=True)['state_dict'])
        rows=[]
        for variant in VARIANTS:
            logits,stats=evaluate(model,bundle,variant)
            path=dest/(variant+'.npz')
            np.savez(path,logits=logits,labels=bundle['labels'],ids=bundle['ids'])
            rows.append(dict(variant=variant,metrics=stats,sha256=sha(path)))
            print('EXPORT',role,seed,variant,stats['mca'],flush=True)
        write(dest/'record.json',dict(role=role,seed=seed,checkpoint_sha256=entry['checkpoint_sha256'],
            selection_sha256=sha(out/'selected.json'),data_sha256=sha(out/'data/manifest.json'),records=rows))
        del model;torch.cuda.empty_cache()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=('preflight','train','export'))
    parser.add_argument('--role',choices=ROLES);parser.add_argument('--seed',type=int)
    parser.add_argument('--out',type=Path,default=OUT);parser.add_argument('--destination',type=Path)
    args=parser.parse_args()
    if args.action=='preflight':preflight(args.destination or PREP/'cuda_preflight1')
    elif args.action=='train':raise SystemExit(fit(args.role,args.seed,args.out))
    else:export(args.seed,args.out)
