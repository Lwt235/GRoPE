"""Fixed actual-GS training, direct color inference, and frozen target renders."""
import argparse
import hashlib
from pathlib import Path
import time
import numpy as np
import torch
from models.round41_gs_readout import GSReadout,ROLES
from experiments.round41_gs_data import (ROOT,OUT,SCENES,SEEDS,TRAIN_VIEWS,VAL_VIEWS,
    checked,load_scene,query,read,write,sha)
from experiments import round9_gs_readout as R9


def schedule(data):
    train=[];validation=[]
    views=data['views']
    for q in TRAIN_VIEWS:
        for k in TRAIN_VIEWS:
            if q==k:continue
            ids=torch.where(views[q]['visible']&views[k]['visible'])[0]
            if len(ids):train.append((q,k,ids))
    for q in VAL_VIEWS:
        for pos in (0,5,10,15):
            k=TRAIN_VIEWS[pos]
            ids=torch.where(views[q]['visible']&views[k]['visible'])[0]
            if len(ids):
                idx=np.linspace(0,len(ids)-1,min(32,len(ids)),dtype=np.int64)
                validation.append((q,k,ids[torch.from_numpy(idx).to(ids.device)]))
    if not train or not validation:raise ValueError('empty context supervision')
    return train,validation


@torch.no_grad()
def evaluate(model,data,validation):
    model.eval();total=0.;elements=0
    for q,k,ids in validation:
        pred=model(query(data,k,ids,model,target=q),data['views'][k])
        error=(pred-data['views'][q]['center_rgb'][ids]).square()
        total+=float(error.double().sum());elements+=error.numel()
    return total/elements


def fit(role,seed,data,train,validation,dest):
    if dest.exists():raise ValueError('retain run '+str(dest))
    dest.mkdir(parents=True);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    model=GSReadout(role).cuda();initial=model.qk.weight.detach().cpu().clone()
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    rng=np.random.default_rng(seed);batch_hash=hashlib.sha256();history=[];losses=[]
    start=time.time();best=float('inf');torch.cuda.reset_peak_memory_stats()
    for step in range(501):
        if step%100==0:
            value=evaluate(model,data,validation);history.append(dict(step=step,mse=value))
            if value<best:
                best,best_step=value,step
                state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            print('VAL',dest.parts[-4],role,seed,step,value,flush=True)
        if step==500:break
        model.train();idx=int(rng.integers(len(train)));q,k,pool=train[idx]
        chosen=rng.integers(len(pool),size=64);ids=pool[torch.from_numpy(chosen).cuda()]
        batch_hash.update(np.asarray([q,k],dtype='<i8').tobytes());batch_hash.update(ids.cpu().numpy().astype('<i8').tobytes())
        opt.zero_grad(set_to_none=True)
        prediction=model(query(data,k,ids,model,target=q),data['views'][k])
        loss=(prediction-data['views'][q]['center_rgb'][ids]).square().mean()
        if not torch.isfinite(loss):raise ValueError('nonfinite training loss')
        loss.backward()
        if step==0:first_grad={n:float(p.grad.norm()) if p.grad is not None else None for n,p in model.named_parameters()}
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
        losses.append(dict(step=step+1,mse=float(loss.detach()),gradient_norm=float(norm)))
        if torch.cuda.max_memory_allocated()>16*(1<<30):raise MemoryError('16GiB allocation ceiling')
    model.load_state_dict(state);model.eval()
    torch.save(dict(state_dict=state,role=role,seed=seed,step=best_step),dest/'best.pt')
    write(dest/'history.json',dict(validation=history,training=losses))
    meta=dict(role=role,seed=seed,best_step=best_step,context_selection_mse=best,
        parameters=sum(p.numel() for p in model.parameters()),batch_sha256=batch_hash.hexdigest(),
        initial_qk_sha256=hashlib.sha256(initial.numpy().tobytes()).hexdigest(),first_gradient=first_grad,
        qk_weight_change=float((state['qk.weight']-initial).norm()),
        shape_strength=float(torch.nn.functional.softplus(state['strength_raw'])) if 'strength_raw' in state else None,
        training_seconds=time.time()-start,training_peak_bytes=torch.cuda.max_memory_allocated(),
        checkpoint_sha256=sha(dest/'best.pt'),history_sha256=sha(dest/'history.json'))
    write(dest/'training.json',meta)
    return model,meta


@torch.no_grad()
def infer(scene,model,data,dest,meta):
    start=time.time();variants=['primary']+(['reverse_v','zero_shape'] if model.role=='decoupled' else [])
    n=len(data['state']['means']);sums=torch.zeros((len(variants),n,3),dtype=torch.float64,device='cuda')
    counts=torch.zeros(n,dtype=torch.int64,device='cuda');sampled=False
    for k,v in enumerate(data['views']):
        ids=torch.where(v['visible'])[0];counts[ids]+=1
        encoded=model.encode(v['feature'],v['anchor'],v['cov'])
        sample_ids=ids[torch.from_numpy(np.linspace(0,len(ids)-1,min(8,len(ids)),dtype=np.int64)).cuda()] if len(ids) else ids
        if k==0 and len(sample_ids):
            q=query(data,k,sample_ids,model)
            p=model.logits(q,v,key_encoded=encoded).softmax(-1)
            np.savez_compressed(dest/'score_samples.npz',view=np.array(k),primitive_ids=sample_ids.cpu().numpy(),
                probability=p.cpu().numpy(),rgb=(p@v['rgb']).cpu().numpy())
            sampled=True
        for start_q in range(0,len(ids),256):
            ix=ids[start_q:start_q+256];q=query(data,k,ix,model)
            p=model.logits(q,v,key_encoded=encoded).softmax(-1)
            sums[0,ix]+=(p@v['rgb']).double()
            if len(variants)>1:
                sums[1,ix]+=(p@v['rgb'].flip(0)).double()
                off=model.logits(q,v,key_encoded=encoded,zero_shape=True).softmax(-1)
                sums[2,ix]+=(off@v['rgb']).double()
        if torch.cuda.max_memory_allocated()>16*(1<<30):raise MemoryError('16GiB allocation ceiling')
    if not sampled:raise ValueError('first context has no sample primitives')
    original=data['state']['color_logits'].sigmoid();valid=counts>0
    colors=torch.where(valid[None,:,None],sums/counts.clamp_min(1)[None,:,None],original[None].double()).float()
    frames=read(R9.DATA/scene/'target_cameras.json')['frames'];predictions=[]
    for col in colors:
        batch=[]
        for frame in frames:
            K,R,t=(torch.tensor(frame[name],dtype=torch.float32,device='cuda') for name in ('K','R','t'))
            pred,_=R9.render(data['state'],K,R,t,colors=col)
            batch.append(pred.cpu().numpy())
        predictions.append(np.stack(batch))
    np.savez_compressed(dest/'colors.npz',colors=colors.cpu().numpy(),counts=counts.cpu().numpy(),variants=np.array(variants))
    np.savez_compressed(dest/'predictions.npz',predictions=np.stack(predictions),variants=np.array(variants),
        target_ids=np.array([f['id'] for f in frames]),target_splits=np.array([f['split'] for f in frames]))
    record=dict(**meta,scene=scene,variants=variants,readable=int(valid.sum()),fallback=int((~valid).sum()),
        inference_seconds=time.time()-start,peak_bytes=torch.cuda.max_memory_allocated(),
        colors_sha256=sha(dest/'colors.npz'),predictions_sha256=sha(dest/'predictions.npz'),
        samples_sha256=sha(dest/'score_samples.npz'),target_rgb_read=False,completed=True)
    write(dest/'run.json',record)
    print('DONE',scene,model.role,meta['seed'],'train',meta['training_seconds'],'inference',record['inference_seconds'],flush=True)


def train_scene(scene,seeds,out):
    checked(out)
    if not read(out/'preflight.json')['passed'] or not read(out/'smoke.json')['passed']:
        raise ValueError('preflight/smoke required')
    started=time.time();data=load_scene(scene,out);train,val=schedule(data)
    shard=scene+'_'+'_'.join(map(str,seeds))
    write(out/(shard+'_started.json'),dict(unix=started,scene=scene,seeds=seeds,gpu=torch.cuda.get_device_name(),
        train_pairs=len(train),validation_elements=sum(len(ids)*3 for _,_,ids in val)))
    try:
        for seed in seeds:
            offset=SEEDS.index(seed);roles=ROLES[offset:]+ROLES[:offset]
            for role in roles:
                dest=out/'runs'/scene/role/str(seed)
                model,meta=fit(role,seed,data,train,val,dest)
                infer(scene,model,data,dest,meta)
                if time.time()-started>3*3600:raise TimeoutError('3hour shard ceiling')
    except BaseException as exc:
        write(out/(shard+'_failure.json'),dict(error=repr(exc),seconds=time.time()-started));raise
    write(out/(shard+'_complete.json'),dict(completed=True,scene=scene,seeds=seeds,seconds=time.time()-started,goal_complete=False))


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=OUT/'attempt1')
    p.add_argument('--scene',choices=SCENES,required=True);p.add_argument('--seeds',nargs='+',type=int,default=list(SEEDS))
    a=p.parse_args();torch.set_num_threads(4);torch.set_num_interop_threads(4)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    if not set(a.seeds)<=set(SEEDS):raise ValueError('unknown seed')
    train_scene(a.scene,a.seeds,a.out)


if __name__=='__main__':main()
