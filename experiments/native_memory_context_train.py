"""Nine matched finite fits, exact all-state resumes and bounded evaluation."""
import argparse
import hashlib
import json
import random
import shutil
import signal
import time
from experiments.native_memory_context_data import *
from experiments.native_receiver_data import enable_receiver,core_state,core_load,digest
from experiments.native_receiver_runtime import model_load
from experiments.gaussian_scene_config import schedule
from experiments.gaussian_scene_preflight import objective,metric_depth
from models.gaussian_scene_rendered import RenderedFeatureFusion
from models.gaussian_scene_context_writer import MemoryContextWriter,forward_read
from models.gaussian_scene_memory import rasterize_field

STOP=False
def request_stop(*_):
    global STOP
    STOP=True


def modules(seed):
    torch.manual_seed(seed);fusion=RenderedFeatureFusion().cuda()
    previous=load(parent.OUT/f'Render_{seed}'/'initial.pt')['model']
    assert all(torch.equal(t.cpu(),previous[n]) for n,t in fusion.state_dict().items())
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(seed);pe=MemoryContextWriter().cuda()
    assert sum(t.numel() for t in fusion.parameters())==787712
    return fusion.eval(),pe.eval()


def optimizers(fusion,pe,core,p):
    return [torch.optim.AdamW(list(fusion.parameters())+list(pe.parameters()),lr=p['lr'],weight_decay=p['weight_decay']),torch.optim.AdamW(core,lr=p['native_lr'],weight_decay=p['weight_decay'],foreach=False)]


def rng_state():
    n=np.random.get_state()
    return dict(torch_cpu=torch.get_rng_state(),torch_cuda=torch.cuda.get_rng_state_all(),numpy=dict(name=n[0],keys=n[1].tolist(),pos=n[2],has_gauss=n[3],cached_gaussian=n[4]),python=random.getstate())


def restore_rng(r):
    def tuples(v):return tuple(tuples(x) for x in v) if isinstance(v,(list,tuple)) else v
    torch.set_rng_state(r['torch_cpu']);torch.cuda.set_rng_state_all(r['torch_cuda']);n=r['numpy'];np.random.set_state((n['name'],np.asarray(n['keys'],dtype=np.uint32),n['pos'],n['has_gauss'],n['cached_gaussian']));random.setstate(tuples(r['python']))


def verify_lock():
    lock=read(OUT/'configuration_lock.json');assert lock['plan_sha256']==sha256(OUT/'plan.json') and lock['scientific_plan_sha256']==sha256(SCIENCE_PLAN)
    revision=OUT/'engineering_source_revision_v3.json'
    if revision.exists():
        rev=read(revision);assert rev['original_lock_sha256']==sha256(OUT/'configuration_lock.json')
        assert all(sha256(path)==h for path,h in rev['original_source_snapshots'].items())
        assert all(sha256(ROOT/n)==h for n,h in rev['current_source_sha256'].items())
        assert sha256(ROOT/rev['moved_writer_path'])==lock['source_sha256']['models/gaussian_memory_context.py']
    else:assert all(sha256(ROOT/n)==v for n,v in lock['source_sha256'].items())
    return lock


def snapshot(model,fusion,pe,opts,step,identity):
    return dict(identity,native=core_state(model),fusion=fusion.state_dict(),writer=pe.state_dict(),optimizer=[o.state_dict() for o in opts],step=step,rng=rng_state())


def fit(model,p,kind,seed):
    folder=OUT/f'{kind}_{seed}';folder.mkdir(exist_ok=True);lock=verify_lock();core=enable_receiver(model);fusion,pe=modules(seed)
    order=schedule(p,seed);identity=dict(kind=kind,seed=seed,plan_sha256=sha256(OUT/'plan.json'),science_sha256=sha256(SCIENCE_PLAN),configuration_lock_sha256=sha256(OUT/'configuration_lock.json'),initial_fusion_sha256=digest(fusion),initial_writer_sha256=digest(pe),pretrained_sha256=lock['pretrained_sha256'],schedule_sha256=hashlib.sha256(json.dumps(order).encode()).hexdigest(),native_parameters=sum(t.numel() for t in core),fusion_parameters=787712,writer_parameters=sum(t.numel() for t in pe.parameters()))
    assert identity['schedule_sha256']==lock['schedule_sha256'][str(seed)] and identity['native_parameters']==637383906
    final=checkpoint_path(kind,seed);final.parent.mkdir(parents=True,exist_ok=True);status_path=folder/'status.json'
    if status_path.exists():
        status=read(status_path);assert status['complete'] and sha256(final)==status['checkpoint_sha256'];state=load(final);core_load(model,state['native']);fusion.load_state_dict(state['fusion']);pe.load_state_dict(state['writer']);return fusion,pe
    write(folder/'initial_identity.json',identity);opts=optimizers(fusion,pe,core,p);first=0;latest=resume_path(kind,seed);durable=CACHE/'resumes'/f'{kind}_{seed}.pt'
    if not latest.exists() and durable.exists():latest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(durable,latest)
    latest.parent.mkdir(parents=True,exist_ok=True)
    assert RAM.stat().st_uid==os.getuid() and RAM.resolve().parent==Path('/dev/shm')
    if latest.exists():
        state=load(latest);assert all(state[n]==v for n,v in identity.items());core_load(model,state['native']);fusion.load_state_dict(state['fusion']);pe.load_state_dict(state['writer'])
        for opt,saved in zip(opts,state['optimizer']):opt.load_state_dict(saved)
        first=state['step'];restore_rng(state['rng']);del state
    data=dataset(p,kind,True);logpath=folder/'training.jsonl'
    if logpath.exists():
        old=logpath.read_text().splitlines();keep=[x for x in old if json.loads(x)['step']<=first]
        if len(keep)!=len(old):
            (folder/f'interrupted_{time.time_ns()}.jsonl').write_text('\n'.join(old[len(keep):])+'\n');logpath.write_text('\n'.join(keep)+('\n' if keep else ''))
            write(folder/'discarded_work.json',dict(updates=len(old)-len(keep),reason='Unexpected interruption after last complete atomic checkpoint; retained log and budget charge, no scientific extra updates.'))
    begin=time.monotonic();torch.cuda.reset_peak_memory_stats()
    with logpath.open('a',buffering=1) as log:
        for step in range(first,p['steps']):
            for opt in opts:opt.zero_grad(set_to_none=True)
            total=0.
            for key in order[step*2:step*2+2]:
                sample=data[tuple(key)];query=sample['query']
                budget_charge('training_writer_forwards');depth=forward_read(model,sample,fusion,pe,kind);loss=objective(depth,query['target'],query['valid']);assert torch.isfinite(loss)
                (loss/2).backward();total+=float(loss.detach())/2;del depth,loss
            norm=torch.nn.utils.clip_grad_norm_(list(fusion.parameters())+list(pe.parameters())+core,p['gradient_clip'],foreach=False);assert torch.isfinite(norm)
            budget_charge('formal_updates')
            for opt in opts:opt.step()
            log.write(json.dumps(dict(step=step+1,loss=total,gradient_norm=float(norm),samples=order[step*2:step*2+2],elapsed_seconds=time.monotonic()-begin))+'\n')
            if (step+1)%100==0 or step+1==p['steps'] or STOP:
                for opt in opts:opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache();t=time.monotonic();assert shutil.disk_usage(RAM).free>10*2**30
                save_tensor(latest,snapshot(model,fusion,pe,opts,step+1,identity))
                with (folder/'checkpoints.jsonl').open('a') as f:f.write(json.dumps(dict(step=step+1,bytes=latest.stat().st_size,save_seconds=time.monotonic()-t,resource_stop=STOP))+'\n')
                print(dict(kind=kind,seed=seed,step=step+1,loss=total,elapsed_seconds=time.monotonic()-begin,peak_GiB=torch.cuda.max_memory_allocated()/2**30),flush=True)
                if STOP:
                    durable.parent.mkdir(parents=True,exist_ok=True);save_tensor(durable,snapshot(model,fusion,pe,opts,step+1,identity));write(folder/'durable_resume.json',dict(path=str(durable),bytes=durable.stat().st_size,step=step+1,sha256=sha256(durable),complete=True));raise SystemExit(75)
            if (step+1)%10==0:write(folder/'progress.json',dict(identity,step=step+1,total_steps=2000,loss=total,elapsed_seconds=time.monotonic()-begin,peak_GiB=torch.cuda.max_memory_allocated()/2**30))
    for opt in opts:opt.zero_grad(set_to_none=True)
    t=time.monotonic();save_tensor(final,dict(identity,native=core_state(model),fusion=fusion.state_dict(),writer=pe.state_dict(),step=2000));h=sha256(final)
    write(status_path,dict(identity,complete=True,steps=2000,checkpoint_path=str(final),checkpoint_sha256=h,bytes=final.stat().st_size,persistence_seconds=time.monotonic()-t,peak_GiB=torch.cuda.max_memory_allocated()/2**30))
    assert sha256(final)==h
    if durable.exists():durable.unlink()
    latest.unlink();latest.parent.rmdir();write(folder/'resume_retirement.json',dict(final_sha256=h,owned_resume_retired=True,after_verified_final=True))
    del opts,opt;model.zero_grad(set_to_none=True);torch.cuda.empty_cache();return fusion,pe


@torch.no_grad()
def export_states(pe,p,kind,seed):
    folder=OUT/'memories';folder.mkdir(exist_ok=True)
    for meta in p['training_catalog']+p['catalog']:
        rid=meta['id'];marker=folder/f'{kind}_{seed}_{rid}.json';path=memory_path(kind,seed,rid)
        if marker.exists():assert read(marker)['complete'] and path.stat().st_size==read(marker)['bytes'];continue
        field=load(scene_path(rid),'cuda');graph=load(graph_path(rid),'cuda');budget_charge('independent_writer_forwards')
        t=time.monotonic();field['features']=pe(field['features'],graph,kind);torch.cuda.synchronize();elapsed=time.monotonic()-t
        path.parent.mkdir(parents=True,exist_ok=True);t=time.monotonic();save_tensor(path,field);h=sha256(path);saved=time.monotonic()-t
        t=time.monotonic();reloaded=load(path);assert torch.equal(reloaded['features'],field['features'].cpu());reload_s=time.monotonic()-t
        assert all(torch.equal(reloaded[k],field[k].cpu()) for k in field if torch.is_tensor(field[k]))
        write(marker,dict(complete=True,kind=kind,seed=seed,id=rid,path=str(path),sha256=h,bytes=path.stat().st_size,writer_seconds=elapsed,serialization_seconds=saved,reload_seconds=reload_s,source_sha256=sha256(scene_path(rid)),query_independent=True,queries_share_one_identity=True))

@torch.no_grad()
def evaluate(model,fusion,pe,p,kind,seed,training):
    cohort='train' if training else 'development';marker=OUT/'evaluation'/f'{cohort}_{kind}_{seed}.json'
    if marker.exists():
        r=read(marker);assert r['complete'] and sha256(r['prediction_path'])==r['prediction_sha256'];return
    keys=[];values=[];begin=time.monotonic();partial=CACHE/'predictions'/f'{cohort}_{kind}_{seed}_partial.pt';partial.parent.mkdir(parents=True,exist_ok=True)
    expected=[(m['id'],q,c) for m in (p['training_catalog'] if training else p['catalog']) for q in EVAL_QUERIES for c in CONDITIONS]
    if partial.exists():
        saved=load(partial);keys=saved['keys'];values=list(saved['depth_metric']);assert keys==[list(k) for k in expected[:len(keys)]]
    state_ids={}
    for meta in p['training_catalog'] if training else p['catalog']:
        rid=meta['id'];path=memory_path(kind,seed,rid);field=load(path,'cuda');state_ids[rid]=read(OUT/'memories'/f'{kind}_{seed}_{rid}.json')['sha256']
        for q in EVAL_QUERIES:
            query=load(query_path(rid,q),'cuda');K=query['K'].clone();K[:2]/=14
            colors=field['features'].float()*field['has_content'][:,None];rendered,_,_=rasterize_field(field,query['c2w'][None],K[None],32,32,colors);extra=rendered[0].reshape(1024,1024)
            for c in CONDITIONS:
                key=(rid,q,c)
                if len(keys)>expected.index(key):continue
                sample=dict(patches=load(patch_path(rid,q,c),'cuda')['patches'],extra=extra,hw=(448,448))
                budget_charge('nontraining_forwards');pred=metric_depth(forward_read(model,sample,fusion,checkpoint_blocks=False)).float().cpu();assert torch.isfinite(pred).all() and (pred>0).all();keys.append(list(key));values.append(pred)
                if len(keys)%16==0 or STOP:save_tensor(partial,dict(keys=keys,depth_metric=torch.stack(values)))
                if STOP:raise SystemExit(75)
    path=CACHE/'predictions'/f'{cohort}_{kind}_{seed}.pt';save_tensor(path,dict(keys=keys,depth_metric=torch.stack(values)))
    status=read(OUT/f'{kind}_{seed}'/'status.json');write(marker,dict(complete=True,cohort=cohort,kind=kind,seed=seed,predictions=len(keys),plan_sha256=sha256(OUT/'plan.json'),prediction_path=str(path),prediction_sha256=sha256(path),checkpoint_path=str(checkpoint_path(kind,seed)),checkpoint_sha256=status['checkpoint_sha256'],seconds=time.monotonic()-begin,memory_sha256=state_ids,writer_calls_during_evaluation=0,source_RGB_calls_during_evaluation=0))
    if partial.exists():partial.unlink()
    print(dict(evaluated=cohort,kind=kind,seed=seed,predictions=len(keys)),flush=True)


def main():
    a=argparse.ArgumentParser();a.add_argument('--kind',choices=KINDS,required=True);a.add_argument('--seed',type=int,choices=SEEDS,required=True);args=a.parse_args();signal.signal(signal.SIGTERM,request_stop);torch.set_num_threads(4);p=plan();verify_lock()
    model=model_load(p);fusion,pe=fit(model,p,args.kind,args.seed);export_states(pe,p,args.kind,args.seed);evaluate(model,fusion,pe,p,args.kind,args.seed,False);evaluate(model,fusion,pe,p,args.kind,args.seed,True)


if __name__=='__main__':main()
