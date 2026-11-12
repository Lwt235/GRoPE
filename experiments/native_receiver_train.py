"""Full native receiver and one fusion trained with matched finite schedules."""
import argparse
import fcntl
import shutil
import signal

from experiments.native_receiver_data import *
from experiments.native_receiver_runtime import model_load,forward_read
from experiments.gaussian_scene_preflight import metric_depth,objective
from models.gaussian_scene_rendered import RenderedFeatureFusion

STOP_REQUESTED=False


def request_stop(signum,frame):
    global STOP_REQUESTED
    STOP_REQUESTED=True


def verify_lock():
    lock=read(OUT/'configuration_lock.json')
    assert lock['plan_sha256']==sha256(OUT/'plan.json')
    assert all(sha256(ROOT/n)==h for n,h in lock['source_sha256'].items())
    return lock


def fit(model,p,kind,seed,folder):
    core=enable_receiver(model);torch.manual_seed(seed);reader=RenderedFeatureFusion().cuda()
    previous=load(FROZEN/f'{kind}_{seed}'/'initial.pt')
    assert all(torch.equal(t.cpu(),previous['model'][n]) for n,t in reader.state_dict().items())
    order=schedule(p,seed);lock=verify_lock()
    identity=dict(kind=kind,seed=seed,plan_sha256=sha256(OUT/'plan.json'),
        initial_state_sha256=digest(reader),pretrained_sha256=lock['pretrained_sha256'],
        schedule_sha256=hashlib.sha256(json.dumps(order).encode()).hexdigest(),
        fusion_parameters=sum(t.numel() for t in reader.parameters()),
        native_parameters=sum(t.numel() for t in core))
    assert identity['schedule_sha256']==previous['schedule_sha256']
    final=checkpoint_path(kind,seed);latest=checkpoint_path(kind,seed,True);marker=folder/'status.json'
    if marker.exists():
        status=read(marker);assert status['complete'] and all(status[n]==v for n,v in identity.items())
        assert sha256(final)==status['checkpoint_sha256']
        state=load(final);core_load(model,state['native']);reader.load_state_dict(state['model'])
        return reader
    opts=optimizers(reader,core,p);first=0
    if latest.exists():
        state=load(latest);assert all(state[n]==v for n,v in identity.items())
        core_load(model,state['native']);reader.load_state_dict(state['model'])
        for opt,opt_state in zip(opts,state['optimizer']):opt.load_state_dict(opt_state)
        first=state['step'];del state
    else:
        save_tensor(folder/'initial.pt',dict(identity,model=reader.state_dict(),
            native_identity='Exact pretrained checkpoint and all specified original parameter names; no random native weights',
            trainable_native_names=[n for n,t in model.named_parameters() if t.requires_grad]))
    data,_=dataset(p,kind,True);logpath=folder/'training.jsonl'
    if logpath.exists():
        lines=logpath.read_text().splitlines();keep=[line for line in lines if json.loads(line)['step']<=first]
        if len(keep)!=len(lines):
            (folder/f'interrupted_after_{first}_{time.time_ns()}.jsonl').write_text('\n'.join(lines[len(keep):])+'\n')
            logpath.write_text('\n'.join(keep)+('\n' if keep else ''))
    torch.cuda.reset_peak_memory_stats();sync_cuda();begin=time.monotonic()
    with logpath.open('a',buffering=1) as log:
        for step in range(first,p['steps']):
            for opt in opts:opt.zero_grad(set_to_none=True)
            loss_total=0.
            for key in order[step*2:step*2+2]:
                sample=data[tuple(key)];query=sample['query']
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    depth=forward_read(model,sample,reader,kind)
                    loss=objective(depth,query['target'],query['valid'])
                if not torch.isfinite(loss):raise FloatingPointError((kind,seed,step,'loss'))
                (loss/2).backward();loss_total+=float(loss.detach())/2
                del depth,loss
            norm=torch.nn.utils.clip_grad_norm_(list(reader.parameters())+core,p['gradient_clip'],foreach=False)
            if not torch.isfinite(norm):raise FloatingPointError((kind,seed,step,'gradient'))
            for opt in opts:opt.step()
            log.write(json.dumps(dict(step=step+1,loss=loss_total,gradient_norm=float(norm),
                samples=order[step*2:step*2+2],elapsed_seconds=time.monotonic()-begin))+'\n')
            if (step+1)%p['checkpoint_interval']==0 or step+1==p['steps'] or STOP_REQUESTED:
                for opt in opts:opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                # Only the current resumable state is retained. Completed finals remain intact.
                expected=read(OUT/'checkpoint_preflight.json')['bytes']
                free=shutil.disk_usage(latest.parent).free
                if free<expected+2*1024**3:
                    write_json(folder/'disk_wait.json',dict(step=step+1,free_bytes=free,required_bytes=expected+2*1024**3))
                    raise SystemExit(75)
                checkpoint_begin=time.monotonic()
                save_tensor(latest,dict(identity,model=reader.state_dict(),native=core_state(model),
                    optimizer=[o.state_dict() for o in opts],step=step+1))
                write_json(folder/'latest.json',dict(step=step+1,path=str(latest),bytes=latest.stat().st_size,
                    save_seconds=time.monotonic()-checkpoint_begin))
                with (folder/'checkpoints.jsonl').open('a') as checkpoint_log:
                    checkpoint_log.write(json.dumps(dict(step=step+1,save_seconds=time.monotonic()-checkpoint_begin,
                        bytes=latest.stat().st_size,resource_stop=STOP_REQUESTED))+'\n')
                print(dict(kind=kind,seed=seed,step=step+1,loss=loss_total,
                    elapsed_seconds=time.monotonic()-begin,peak_GiB=torch.cuda.max_memory_allocated()/2**30),flush=True)
                if STOP_REQUESTED:
                    write_json(folder/'resource_checkpoint.json',dict(step=step+1,path=str(latest),
                        exact_model_and_both_optimizers_saved=True,scientific_configuration_unchanged=True))
                    raise SystemExit(75)
            if (step+1)%10==0:
                write_json(folder/'progress.json',dict(identity,step=step+1,total_steps=p['steps'],loss=loss_total,
                    elapsed_seconds=time.monotonic()-begin,peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30))
    for opt in opts:opt.zero_grad(set_to_none=True)
    save_tensor(final,dict(identity,model=reader.state_dict(),native=core_state(model),step=p['steps']))
    final_sha=sha256(final)
    write_json(marker,dict(identity,complete=True,steps=p['steps'],checkpoint_path=str(final),
        checkpoint_sha256=final_sha,final_fusion_sha256=digest(reader),
        seconds_this_session=time.monotonic()-begin,peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30))
    if latest.exists():
        write_json(folder/'resume_retirement.json',dict(final_path=str(final),verified_final_sha256=final_sha,
            obsolete_owned_resume_path=str(latest),removed=True,
            reason='Completed exact final checkpoint verified; retire only this experiment redundant optimizer resume state.'))
        latest.unlink()
    del opts;model.zero_grad(set_to_none=True);reader.zero_grad(set_to_none=True);torch.cuda.empty_cache()
    return reader


@torch.no_grad()
def evaluate(model,reader,p,kind,seed,training=False,frozen=False,shuffle=False):
    cohort='train' if training else 'development';regime='frozen' if frozen else 'adapted'
    name=f'{regime}_{cohort}_{kind}_{seed}'+('_shuffle' if shuffle else '')
    marker=OUT/'evaluation'/f'{name}.json'
    if marker.exists():
        r=read(marker);assert r['complete'] and r['plan_sha256']==sha256(OUT/'plan.json')
        assert sha256(r['prediction_path'])==r['prediction_sha256'];return
    checkpoint=FROZEN/f'{kind}_{seed}'/'step_2000.pt' if frozen else checkpoint_path(kind,seed)
    if frozen:
        assert training and not shuffle
        restore_native(model,p);model.requires_grad_(False)
        reader.load_state_dict(load(checkpoint)['model'])
    data,_=dataset(p,kind,training);reader.eval();pred=[];keys=[];begin=time.monotonic();last=None;cached=None
    for key in data:
        if STOP_REQUESTED:raise SystemExit(75)
        sample=data[key]
        if shuffle:
            if last!=key[:2]:last=key[:2];cached=load(feature_path(*key[:2]))['shuffle']
            sample['extra']=cached.cuda()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            depth=forward_read(model,sample,reader,kind,checkpoint_blocks=False)
        assert torch.isfinite(depth).all() and (depth>0).all()
        pred.append(metric_depth(depth).float().cpu());keys.append((*key,'shuffle_binding') if shuffle else key)
    path=CACHE/'predictions'/f'{name}.pt';path.parent.mkdir(parents=True,exist_ok=True)
    save_tensor(path,dict(keys=keys,depth_metric=torch.stack(pred)))
    write_json(marker,dict(complete=True,kind=kind,seed=seed,cohort=cohort,regime=regime,shuffle=shuffle,
        predictions=len(keys),checkpoint_path=str(checkpoint),checkpoint_sha256=sha256(checkpoint),
        prediction_path=str(path),prediction_sha256=sha256(path),plan_sha256=sha256(OUT/'plan.json'),
        seconds=time.monotonic()-begin))
    print(dict(evaluation=name,predictions=len(keys)),flush=True)


def main():
    a=argparse.ArgumentParser();a.add_argument('--kind',choices=KINDS,required=True)
    a.add_argument('--seed',type=int,choices=SEEDS,required=True);args=a.parse_args()
    torch.set_num_threads(4);p=plan();verify_lock();assert read(OUT/'preflight.json')['complete']
    signal.signal(signal.SIGTERM,request_stop)
    folder=OUT/f'{args.kind}_{args.seed}';folder.mkdir(parents=True,exist_ok=True)
    with (folder/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        model=model_load(p)
        if STOP_REQUESTED:raise SystemExit(75)
        reader=fit(model,p,args.kind,args.seed,folder)
        evaluate(model,reader,p,args.kind,args.seed)
        evaluate(model,reader,p,args.kind,args.seed,training=True)
        if args.kind=='Render':evaluate(model,reader,p,args.kind,args.seed,shuffle=True)
        evaluate(model,reader,p,args.kind,args.seed,training=True,frozen=True)


if __name__=='__main__':main()
