"""Fixed R40 GPU runs. No fresh-data discovery or result-driven role selection."""
import argparse
import hashlib
import math
from pathlib import Path
import random
import time
import numpy as np
import torch
from models.round40_decoupled_pe import DecoupledDepthAttention, ROLES, NEW_ROLES, SHAPE_ROLES, EXTRA_FIELDS
from models.round7_depth_attention import FIELDS, depth_loss
from experiments.round7_depth_train import evaluate, batch
from experiments.round40_decoupled_data import OUT, ROOT, sha, read, write, freeze, verify, prepare


def configure():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def load_split(split, device, out=OUT):
    lock=verify(out)
    prepared={r['scene']:r for r in read(out/'prepared/completion.json')['records']}
    fields=FIELDS+EXTRA_FIELDS+('target','valid','boundary')
    chunks={f:[] for f in fields}; identities=[]
    for e in lock['entries']:
        if e['split']!=split:
            continue
        folder=Path(e['folder']); rec=prepared[e['scene']]
        for path,digest,names in ((folder/'inputs.npz',e['input_sha256'],FIELDS),
                (Path(rec['path']),rec['sha256'],EXTRA_FIELDS),
                (folder/'labels.npz',e['label_sha256'],('target','valid','boundary'))):
            if sha(path)!=digest:
                raise ValueError('changed array input '+str(path))
            with np.load(path,allow_pickle=False) as z:
                for name in names:
                    chunks[name].append(z[name])
        identities.extend(dict(scene=e['scene'],q_index=q,k_index=k,pair_id=f'{q}:{k}') for q,k in e['pairs'])
    if len(identities)!=(576 if split=='train' else 128):
        raise ValueError('full old cohort required')
    arrays={k:torch.from_numpy(np.concatenate(v)).to(device) for k,v in chunks.items()}
    return arrays,identities


def state_digest(state):
    h=hashlib.sha256()
    for name,value in state.items():
        h.update(name.encode()); h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def fit(role,seed,train,train_ids,val,val_ids,out=OUT):
    dest=out/'runs'/role/str(seed)
    if dest.exists():
        raise ValueError('preserve existing run: '+str(dest))
    dest.mkdir(parents=True)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model=DecoupledDepthAttention(role).to(train['q'].device)
    initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    old_keys=('qk.','query.','value.','head.')
    initial_sha=state_digest({k:v for k,v in initial.items() if k.startswith(old_keys)})
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    rng=np.random.default_rng(seed); sequence=hashlib.sha256()
    start=time.perf_counter(); history=[]; losses=[]; best=float('inf'); first_grad=None
    torch.cuda.reset_peak_memory_stats()
    for step in range(1001):
        if step%100==0:
            result,_=evaluate(model,val,val_ids)
            score=result['summary']['metrics']['AbsRel']
            history.append(dict(step=step,validation=result['summary']))
            if score<best:
                best,best_step=score,step
                best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            print('VAL',role,seed,step,round(score,7),'sec',round(time.perf_counter()-start,2),flush=True)
        if step==1000:
            break
        model.train(); indices=rng.integers(len(train_ids),size=8)
        sequence.update(indices.astype('<i8').tobytes()); d=batch(train,indices)
        optimizer.zero_grad(set_to_none=True)
        prediction=model(d); loss=depth_loss(prediction,d['target'],d['valid'])
        if not torch.isfinite(loss):
            raise ValueError('nonfinite loss')
        loss.backward()
        if step==0:
            first_grad={n:float(p.grad.norm()) if p.grad is not None else None for n,p in model.named_parameters()}
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        losses.append(dict(step=step+1,loss=float(loss.detach()),gradient_norm=float(norm)))
        if torch.cuda.max_memory_allocated()>15*(1<<30):
            raise MemoryError('frozen 15GiB allocation ceiling')
    model.load_state_dict(best_state)
    result,pred=evaluate(model,val,val_ids)
    predictions={'depth':pred.cpu().numpy()}; interventions={}
    modes=['zero_v','reverse_v']
    if role in SHAPE_ROLES or role=='center_rbf':
        modes.append('zero_shape')
    for mode in modes:
        changed,values=evaluate(model,val,val_ids,mode)
        predictions[mode]=values.cpu().numpy()
        interventions[mode]=dict(summary=changed['summary'],mean_absolute_output_change_m=float((values-pred).abs().mean()))
    torch.cuda.synchronize()
    samples=batch(val,np.array([0,64]))
    with torch.no_grad():
        _,attention,_=model(samples,return_attention=True)
        score_arrays=dict(indices=np.array([0,64]),probability=attention.cpu().numpy())
        if role in NEW_ROLES:
            score_arrays['logits']=model.attention_logits(samples).cpu().numpy()
    np.savez_compressed(dest/'score_samples.npz',**score_arrays)
    torch.save(dict(role=role,seed=seed,step=best_step,state_dict=best_state),dest/'best.pt')
    np.savez_compressed(dest/'validation_predictions.npz',**predictions)
    write(dest/'validation.json',result)
    write(dest/'history.json',dict(validation=history,training=losses))
    record=dict(role=role,seed=seed,steps=1000,best_step=best_step,
        parameter_count=sum(p.numel() for p in model.parameters()),
        initial_shared_sha256=initial_sha,batch_sequence_sha256=sequence.hexdigest(),
        first_gradient_norms=first_grad,
        qk_weight_change_norm=float((best_state['qk.weight']-initial['qk.weight']).norm()),
        shape_strength=float(torch.nn.functional.softplus(best_state['strength_raw'])) if 'strength_raw' in best_state else None,
        checkpoint_sha256=sha(dest/'best.pt'),predictions_sha256=sha(dest/'validation_predictions.npz'),
        score_samples_sha256=sha(dest/'score_samples.npz'),
        validation_sha256=sha(dest/'validation.json'),history_sha256=sha(dest/'history.json'),
        seconds=time.perf_counter()-start,max_allocated_bytes=torch.cuda.max_memory_allocated(),
        device=torch.cuda.get_device_name(),interventions=interventions,
        lock_sha256=sha(out/'lock.json'),completed=True)
    write(dest/'run.json',record)
    print('DONE',role,seed,'best',best_step,best,'seconds',record['seconds'],flush=True)


def smoke(out=OUT):
    from experiments.round40_decoupled_audit import toy_data
    verify(out)
    torch.manual_seed(771)
    d=toy_data(432,device='cuda:0',dtype=torch.float32)
    rows=[]
    for role in NEW_ROLES:
        model=DecoupledDepthAttention(role).cuda()
        opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
        torch.cuda.synchronize(); start=time.perf_counter()
        for _ in range(5):
            opt.zero_grad(set_to_none=True)
            pred=model(d); loss=depth_loss(pred,d['target'],d['valid']); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True); opt.step()
        torch.cuda.synchronize()
        rows.append(dict(role=role,loss=float(loss),seconds=time.perf_counter()-start,
            qk_gradient_norm=float(model.qk.weight.grad.norm()),value_gradient_norm=float(model.value.weight.grad.norm()),
            parameter_count=sum(p.numel() for p in model.parameters())))
    write(out/'smoke.json',dict(rows=rows,torch=torch.__version__,gpu=torch.cuda.get_device_name(),
        cuda=True,real_arrays_read=False,created_unix=time.time()))
    print('SMOKE',rows,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('freeze','prepare','smoke','train'))
    p.add_argument('--out',type=Path,default=OUT);p.add_argument('--seeds',nargs='+',type=int)
    a=p.parse_args();configure()
    if a.stage=='freeze':
        lock=freeze(a.out);print('FROZEN',len(lock['source_hashes']),'sources',len(lock['entries']),'scenes');return
    verify(a.out)
    if a.stage=='prepare':
        prepare(a.out);return
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no silent CPU fallback')
    if a.stage=='smoke':
        smoke(a.out);return
    if not (a.out/'smoke.json').exists() or not (a.out/'preflight.json').exists():
        raise ValueError('preflight and smoke required')
    if not read(a.out/'preflight.json')['passed']:
        raise ValueError('failed preflight')
    if not a.seeds or not set(a.seeds)<=set((11,22,33,44,55)):
        raise ValueError('frozen seed shard required')
    started=time.time();train,ti=load_split('train','cuda:0',a.out);val,vi=load_split('val','cuda:0',a.out)
    write(a.out/('shard_'+'_'.join(map(str,a.seeds))+'_started.json'),dict(created_unix=started,
          seeds=a.seeds,gpu=torch.cuda.get_device_name(),torch=torch.__version__,train_pairs=len(ti),val_pairs=len(vi)))
    try:
        for seed in a.seeds:
            # Rotate execution order across seeds; selection never uses results.
            offset=(11,22,33,44,55).index(seed)
            for role in ROLES[offset:]+ROLES[:offset]:
                if time.time()-started>7200:
                    raise TimeoutError('frozen shard budget')
                fit(role,seed,train,ti,val,vi,a.out)
    except BaseException as exc:
        write(a.out/('shard_'+'_'.join(map(str,a.seeds))+'_failure.json'),dict(error=repr(exc),unix=time.time()))
        raise
    verify(a.out)
    write(a.out/('shard_'+'_'.join(map(str,a.seeds))+'_complete.json'),dict(seeds=a.seeds,
        seconds=time.time()-started,completed=True,goal_complete=False))


if __name__=='__main__':
    main()
