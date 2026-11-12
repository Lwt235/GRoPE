"""Matched full native adaptation with a single input PE reader."""
import argparse
import hashlib
import json
import shutil
import signal
import time

from experiments.native_input_pe_data import *
from experiments.native_receiver_data import enable_receiver,core_state,core_load,optimizers,digest
from experiments.native_receiver_runtime import model_load
from models.gaussian_scene_input_pe import forward_read
from experiments.gaussian_scene_config import schedule
from experiments.gaussian_scene_preflight import objective,metric_depth
from models.gaussian_scene_input_pe import InputPEReader

STOP=False


def request_stop(*args):
    global STOP
    STOP=True


def verify_lock():
    lock=read(OUT/'configuration_lock.json')
    assert lock['plan_sha256']==sha256(OUT/'plan.json')
    assert all(sha256(ROOT/n)==h for n,h in lock['source_sha256'].items())
    return lock


def fit(model,p,kind,seed):
    folder=OUT/f'{kind}_{seed}';folder.mkdir(parents=True,exist_ok=True)
    core=enable_receiver(model);torch.manual_seed(seed);reader=InputPEReader().cuda()
    order=schedule(p,seed);lock=verify_lock()
    identity=dict(kind=kind,seed=seed,plan_sha256=sha256(OUT/'plan.json'),
        initial_state_sha256=digest(reader),pretrained_sha256=lock['pretrained_sha256'],
        schedule_sha256=hashlib.sha256(json.dumps(order).encode()).hexdigest(),
        reader_parameters=sum(t.numel() for t in reader.parameters()),native_parameters=sum(t.numel() for t in core),
        training_places=32)
    assert identity['schedule_sha256']==lock['schedule_sha256'][str(seed)]
    assert identity['native_parameters']==637383906 and identity['reader_parameters']==1054672
    final=checkpoint_path(kind,seed);final.parent.mkdir(parents=True,exist_ok=True)
    latest=resume_path(kind,seed);latest.parent.mkdir(parents=True,exist_ok=True)
    RAM.mkdir(mode=0o700,exist_ok=True)
    assert RAM.stat().st_uid==os.getuid() and RAM.parent==Path('/dev/shm')
    status_path=folder/'status.json'
    if status_path.exists():
        status=read(status_path);assert status['complete'] and sha256(final)==status['checkpoint_sha256']
        state=load(final);core_load(model,state['native']);reader.load_state_dict(state['model']);return reader
    opts=optimizers(reader,core,p);first=0
    if latest.exists():
        state=load(latest);assert all(state[n]==v for n,v in identity.items())
        core_load(model,state['native']);reader.load_state_dict(state['model'])
        for opt,saved in zip(opts,state['optimizer']):opt.load_state_dict(saved)
        first=state['step'];del state
    else:save_tensor(folder/'initial.pt',dict(identity,model=reader.state_dict()))
    data=dataset(p,kind,True);logpath=folder/'training.jsonl'
    if logpath.exists():
        old=logpath.read_text().splitlines();keep=[line for line in old if json.loads(line)['step']<=first]
        if len(keep)!=len(old):
            (folder/f'interrupted_{time.time_ns()}.jsonl').write_text('\n'.join(old[len(keep):])+'\n')
            logpath.write_text('\n'.join(keep)+('\n' if keep else ''))
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();begin=time.monotonic()
    with logpath.open('a',buffering=1) as log:
        for step in range(first,p['steps']):
            for opt in opts:opt.zero_grad(set_to_none=True)
            total=0.
            for key in order[step*2:step*2+2]:
                sample=data[tuple(key)];query=sample['query']
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    depth=forward_read(model,sample,reader,kind);loss=objective(depth,query['target'],query['valid'])
                if not torch.isfinite(loss):raise FloatingPointError((kind,seed,step,'loss'))
                (loss/2).backward();total+=float(loss.detach())/2;del depth,loss
            norm=torch.nn.utils.clip_grad_norm_(list(reader.parameters())+core,p['gradient_clip'],foreach=False)
            if not torch.isfinite(norm):raise FloatingPointError((kind,seed,step,'gradient'))
            for opt in opts:opt.step()
            log.write(json.dumps(dict(step=step+1,loss=total,gradient_norm=float(norm),samples=order[step*2:step*2+2],elapsed_seconds=time.monotonic()-begin))+'\n')
            if (step+1)%100==0 or step+1==p['steps'] or STOP:
                for opt in opts:opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache();t=time.monotonic()
                assert shutil.disk_usage(RAM).free>9*1024**3
                save_tensor(latest,dict(identity,model=reader.state_dict(),native=core_state(model),optimizer=[opt.state_dict() for opt in opts],step=step+1))
                with (folder/'checkpoints.jsonl').open('a') as f:f.write(json.dumps(dict(step=step+1,bytes=latest.stat().st_size,save_seconds=time.monotonic()-t))+'\n')
                print(dict(kind=kind,seed=seed,step=step+1,loss=total,elapsed_seconds=time.monotonic()-begin,peak_GiB=torch.cuda.max_memory_allocated()/2**30),flush=True)
                if STOP:raise SystemExit(75)
            if (step+1)%10==0:write_json(folder/'progress.json',dict(identity,step=step+1,total_steps=2000,loss=total,elapsed_seconds=time.monotonic()-begin,peak_GiB=torch.cuda.max_memory_allocated()/2**30))
    for opt in opts:opt.zero_grad(set_to_none=True)
    save_tensor(final,dict(identity,model=reader.state_dict(),native=core_state(model),step=2000))
    h=sha256(final);assert final.stat().st_size>2*1024**3
    write_json(status_path,dict(identity,complete=True,steps=2000,checkpoint_path=str(final),checkpoint_sha256=h,peak_GiB=torch.cuda.max_memory_allocated()/2**30))
    if latest.exists():latest.unlink()
    if not list(latest.parent.iterdir()):latest.parent.rmdir()
    del opt,opts;model.zero_grad(set_to_none=True);reader.zero_grad(set_to_none=True);torch.cuda.empty_cache()
    return reader


@torch.no_grad()
def evaluate(model,reader,p,kind,seed,training):
    cohort='train' if training else 'development';marker=OUT/'evaluation'/f'{cohort}_{kind}_{seed}.json'
    if marker.exists():
        old=read(marker);assert old['complete'] and sha256(old['prediction_path'])==old['prediction_sha256'];return
    reader.eval();data=dataset(p,kind,training,heldout=True);keys=[];pred=[];torch.cuda.synchronize();begin=time.monotonic()
    for key in data:
        if STOP:raise SystemExit(75)
        sample=data[key]
        with torch.autocast('cuda',dtype=torch.bfloat16):depth=forward_read(model,sample,reader,kind,checkpoint_blocks=False)
        assert torch.isfinite(depth).all() and (depth>0).all()
        keys.append(key);pred.append(metric_depth(depth).float().cpu())
    path=CACHE/'predictions'/f'{cohort}_{kind}_{seed}.pt';path.parent.mkdir(parents=True,exist_ok=True)
    save_tensor(path,dict(keys=keys,depth_metric=torch.stack(pred)))
    write_json(marker,dict(complete=True,cohort=cohort,kind=kind,seed=seed,predictions=len(keys),
        plan_sha256=sha256(OUT/'plan.json'),prediction_path=str(path),prediction_sha256=sha256(path),
        checkpoint_path=str(checkpoint_path(kind,seed)),checkpoint_sha256=read(OUT/f'{kind}_{seed}'/'status.json')['checkpoint_sha256'],seconds=time.monotonic()-begin))
    print(dict(evaluated=cohort,kind=kind,seed=seed,predictions=len(keys)),flush=True)


def main():
    a=argparse.ArgumentParser();a.add_argument('--kind',choices=KINDS,required=True);a.add_argument('--seed',type=int,choices=SEEDS,required=True);args=a.parse_args()
    signal.signal(signal.SIGTERM,request_stop);torch.set_num_threads(4);p=plan();verify_lock()
    model=model_load(p);reader=fit(model,p,args.kind,args.seed)
    evaluate(model,reader,p,args.kind,args.seed,False);evaluate(model,reader,p,args.kind,args.seed,True)


if __name__=='__main__':main()
