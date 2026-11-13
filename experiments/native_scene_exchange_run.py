"""Round012 preparation, finite paired fitting, independent-query exports."""
import argparse,fcntl,functools,signal,sys,time
from experiments import native_scene_exchange_data as d
from experiments.native_scene_exchange_data import torch,np,F
from models.native_scene_exchange import SceneExchange,forward,continuous_rope
from experiments.native_receiver_runtime import model_load
from experiments.gaussian_scene_preflight import objective,metric_depth
from models.gaussian_scene_memory import contribution_weights,field_to_device

STOP=False
def stop(*_):
    global STOP
    STOP=True
signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)

def elapsed(start,phase):
    torch.cuda.synchronize();return dict(phase=phase,seconds=time.monotonic()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())

def prepare():
    d.CACHE.mkdir(parents=True,exist_ok=True);assert not d.resource(True)['stop_reasons'];rows=[]
    for meta in d.metas():
        rid=meta['id'];field=d.load(d.old.scene_path(rid));g=d.load(d.gpath(rid));fg=field_to_device(field,'cuda')
        for q in d.old.EVAL_QUERIES:
            path=d.vpath(rid,q)
            if path.exists():continue
            query=d.load(d.old.query_path(rid,q));K=query['K'].cuda().clone();K[:2]/=14
            start=time.monotonic();w,a=contribution_weights(fg,query['c2w'].cuda()[None],K[None],32,32);torch.cuda.synchronize();raster=time.monotonic()-start
            mass=w[0].sum((0,1)).cpu();start=time.monotonic();v=d.scene_view(field,g,query,mass);v['local']=d.local_view(query)
            v['visibility_mass']=mass;v['alpha']=a.cpu();d.save(path,v)
            row=dict(id=rid,query=q,render_seconds=raster,geometry_local_seconds=time.monotonic()-start,visible_mass_sum=float(mass.sum()),content_mass_sum=float(mass[field['has_content']].sum()),visible_slots=int((mass>0).sum()),valid_keys=int(v['key_mask'].sum()),zero_content=bool((mass[field['has_content']].sum()==0)),path=str(path),sha256=d.sha(path))
            d.write(d.OUT/'preparation'/f'{rid}_{q}.json',row);rows.append(row)
        del fg;torch.cuda.empty_cache();assert not d.resource()['stop_reasons'];print(dict(prepared=rid),flush=True)
        if STOP:raise SystemExit(75)
    rows=[d.read(p) for p in sorted((d.OUT/'preparation').glob('*.json'))]
    d.write(d.OUT/'preparation_complete.json',dict(complete=len(rows)==352,views=rows,raster_calls=len(rows),source_encoding=0,source_optimization=0,new_fields=0))

@functools.lru_cache(256)
def cached_sample(rid,q,c,arm):return d.sample(rid,q,c,arm)

def batch(keys,arm):
    samples=[cached_sample(*k,arm) for k in keys]
    obj={k:torch.cat([s[k] for s in samples],0) for k in ('patches','ordinary','scene_pos','key_mask','position')}
    return d.device(obj),samples

def checkpoint(adapter,opt,step,seed,arm,order,folder):
    state=dict(adapter=d.cpu(adapter.state_dict()),optimizer=d.cpu(opt.state_dict()),step=step,seed=seed,arm=arm,schedule_sha256=d.hashlib.sha256(d.json.dumps(order).encode()).hexdigest(),rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state().cpu(),numpy_state='sampler fully materialized; no per-step numpy calls',science_sha256=d.SCIENCE_SHA)
    d.save(folder/('final.pt' if step==2000 else 'latest.pt'),state)
    return state

def training(model,arm,seed):
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);adapter=SceneExchange().cuda();opt=torch.optim.AdamW(adapter.parameters(),lr=1e-4,weight_decay=.01)
    order=d.schedule(seed);folder=d.final(arm,seed).parent;folder.mkdir(parents=True,exist_ok=True);start_step=0
    path=d.final(arm,seed) if d.final(arm,seed).exists() else folder/'latest.pt'
    if path.exists():
        st=d.load(path);assert st['science_sha256']==d.SCIENCE_SHA and st['arm']==arm and st['seed']==seed
        assert st['schedule_sha256']==d.hashlib.sha256(d.json.dumps(order).encode()).hexdigest()
        adapter.load_state_dict(st['adapter']);opt.load_state_dict(st['optimizer']);start_step=st['step'];torch.set_rng_state(st['rng'].cpu());torch.cuda.set_rng_state(st['cuda_rng'].cpu());del st
    else:
        d.write(d.OUT/'fits'/f'{arm}_{seed}_initial.json',dict(parameters=sum(p.numel() for p in adapter.parameters()),state_digest=state_digest(adapter),schedule_sha256=d.hashlib.sha256(d.json.dumps(order).encode()).hexdigest(),steps=2000,batch=2,unused_L_input_columns=list(range(1027,1034))+[1044] if arm=='L' else [],official_frozen=True))
    d.journal(d.OUT/'resume_events.jsonl',dict(utc=d.now(),arm=arm,seed=seed,pid=d.os.getpid(),loaded_step=start_step,checkpoint=str(path) if path.exists() else None,Adam_and_RNG_restored=bool(start_step),schedule_sha256=d.hashlib.sha256(d.json.dumps(order).encode()).hexdigest()))
    print(dict(fit=arm,seed=seed,resume_from_step=start_step,Adam_RNG_sampler_restored=bool(start_step)),flush=True)
    start=time.monotonic()
    for step in range(start_step,2000):
        opt.zero_grad(set_to_none=True);keys=order[2*step:2*step+2];sample,raw=batch(keys,arm)
        d.journal(d.OUT/'calls'/f'{arm}_{seed}.jsonl',dict(kind='training',step=step+1,native=2,scene_initializations=2,samples=keys))
        with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
            pred=forward(model,sample,adapter,arm);loss=sum(objective(pred[i],r['target'].cuda(),r['valid'].cuda()) for i,r in enumerate(raw))/2
        if not torch.isfinite(loss):raise FloatingPointError(('loss',arm,seed,step))
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(adapter.parameters(),5.)
        if not torch.isfinite(norm):raise FloatingPointError(('gradient',arm,seed,step))
        opt.step();d.journal(d.OUT/'fits'/f'{arm}_{seed}.jsonl',dict(step=step+1,loss=float(loss.detach()),gradient_norm=float(norm),samples=keys,elapsed_seconds=time.monotonic()-start))
        del sample,raw,pred,loss
        if (step+1)%100==0 or STOP or step==1999:
            checkpoint(adapter,opt,step+1,seed,arm,order,folder);r=d.resource();print(dict(arm=arm,seed=seed,step=step+1,seconds=time.monotonic()-start),flush=True)
            if STOP or r['stop_reasons']:raise SystemExit(75)
    final=d.final(arm,seed);d.write(d.OUT/'fits'/f'{arm}_{seed}_complete.json',dict(complete=True,step=2000,path=str(final),sha256=d.sha(final),schedule_sha256=d.hashlib.sha256(d.json.dumps(order).encode()).hexdigest(),parameters=sum(x.numel() for x in adapter.parameters()),training_samples=4000))
    # The complete final includes optimizer/RNG; duplicate recoveries retired.
    if (folder/'latest.pt').exists():(folder/'latest.pt').unlink()
    del opt;cached_sample.cache_clear();torch.cuda.empty_cache();return adapter

def state_digest(adapter):
    h=d.hashlib.sha256()
    for n,t in adapter.state_dict().items():h.update(n.encode());h.update(t.detach().cpu().numpy().tobytes())
    return h.hexdigest()

@torch.no_grad()
def evaluate(model,adapter,arm,seed):
    for meta in d.metas():
        rid=meta['id']
        for q in d.old.EVAL_QUERIES:
            for c in d.old.CONDITIONS:
                marker=d.OUT/'completed'/f'{arm}_{seed}_{rid}_{q}_{c}.json'
                if marker.exists():continue
                s=d.device(d.sample(rid,q,c,arm));d.journal(d.OUT/'calls'/f'{arm}_{seed}.jsonl',dict(kind='evaluation',id=rid,query=q,condition=c,native=1,scene_initializations=1))
                with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):pred=metric_depth(forward(model,s,adapter,arm,False)[0])
                assert torch.isfinite(pred).all() and (pred>0).all()
                path=d.CACHE/'depths'/f'{arm}_{seed}'/f'{rid}_{q}_{c}.npy';d.nsave(path,pred.float().cpu().numpy());d.write(marker,dict(complete=True,id=rid,place=meta['place'],query=q,condition=c,arm=arm,seed=seed,path=str(path),sha256=d.sha(path)))
                del pred,s
            if STOP:raise SystemExit(75)
        assert not d.resource()['stop_reasons'];print(dict(evaluated=rid,arm=arm,seed=seed),flush=True)

def preflight(model):
    rid=d.plan()['training_catalog'][0]['id'];keys=[(rid,2,'missing'),(rid,5,'full')];records=[]
    for arm in ('R_P','U_G','L'):
        torch.manual_seed(d.SEEDS[0]);adapter=SceneExchange().cuda();opt=torch.optim.AdamW(adapter.parameters(),lr=1e-4,weight_decay=.01);s,raw=batch(keys,arm);torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.monotonic()
        with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
            pred=forward(model,s,adapter,arm);loss=sum(objective(pred[i],x['target'].cuda(),x['valid'].cuda()) for i,x in enumerate(raw))/2
        loss.backward();grads={n:float(t.grad.float().norm()) for n,t in adapter.named_parameters() if t.grad is not None};assert all(np.isfinite(x) for x in grads.values());assert grads['source.weight']>0 and grads['position.2.weight']>0 and grads['qkv.0.b.weight']>0
        opt.step();records.append(dict(arm=arm,loss=float(loss),gradients=grads,**elapsed(start,'discarded_training_batch2')))
        del opt,adapter,s,raw,pred,loss;cached_sample.cache_clear();torch.cuda.empty_cache()
    # R scene hidden must be independent of current image in first global;
    # U can depend on it. No threshold about downstream scientific benefits.
    adapter=SceneExchange().cuda();s,_=batch([keys[0]],'R_P');changed={**s,'patches':s['patches']+torch.randn_like(s['patches'])*.1};tests={}
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
        for arm in ('R_P','U_P'):
            a={};b={};forward(model,s,adapter,arm,False,a);forward(model,changed,adapter,arm,False,b)
            tests[arm]=float((a['scene_after_global_0']-b['scene_after_global_0']).abs().max())
    assert tests['R_P']==0 and tests['U_P']>0
    att=model.aggregator.frame_blocks[0].attn;t=torch.randn(1,16,10,64,device='cuda');pos=torch.randint(0,32,(1,10,2),device='cuda')
    rope_error=float((att.rope(t,pos)-continuous_rope(t,pos,att.rope)).abs().max());assert rope_error<1e-5
    # Recreating the same sample reproduces all immutable descriptors.
    assert all(not p.requires_grad for p in model.parameters())
    d.write(d.OUT/'preflight.json',dict(complete=True,discarded_updates=3,discarded_samples=6,native_inference=4,records=records,message_mask_tests=tests,integer_rope_extension_max_difference=rope_error,permanent_field_writes=0,official_trainable_parameters=0))
    print(dict(preflight_complete=True,records=[{k:v for k,v in r.items() if k!='gradients'} for r in records]),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['prepare','preflight','fits']);parser.add_argument('--arms',nargs='*',default=[]);a=parser.parse_args()
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.48);d.OUT.mkdir(parents=True,exist_ok=True)
    assert torch.cuda.is_available();assert not d.resource(True)['stop_reasons']
    if a.phase=='prepare':prepare();return
    model=model_load(d.plan());print(dict(model_loaded=True,phase=a.phase),flush=True)
    if a.phase=='preflight':preflight(model);return
    jobs=[(arm,int(seed)) for job in a.arms for arm,seed in [job.split(':')]] if a.arms and ':' in a.arms[0] else [(arm,seed) for seed in d.SEEDS for arm in a.arms]
    for arm,seed in jobs:
        claim=d.OUT/'fit_claims'/f'{arm}_{seed}.lock';claim.parent.mkdir(exist_ok=True)
        with claim.open('a') as ownership:
            fcntl.flock(ownership,fcntl.LOCK_EX|fcntl.LOCK_NB)
            d.write(claim.with_suffix('.json'),dict(arm=arm,seed=seed,pid=d.os.getpid(),claimed_utc=d.now()))
            adapter=training(model,arm,seed);evaluate(model,adapter,arm,seed);del adapter;torch.cuda.empty_cache()

if __name__=='__main__':main()
