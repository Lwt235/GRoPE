"""Round015 fixed native-task physical writer execution and full-state recovery."""
import argparse,fcntl,hashlib,json,os,signal,time
import torch
from experiments import native_task_writer_data as d
from models.native_task_writer import TaskWriter,ShapeFusion,read_current
from models.gaussian_scene_parameter_pe import ParameterPE,forward_read
from experiments.native_receiver_runtime import model_load
from experiments.native_receiver_data import core_state,core_load,enable_receiver,restore_native,digest
from experiments.native_render_parameter_train import rng_state,restore_rng
from experiments.gaussian_scene_config import schedule
from experiments.gaussian_scene_preflight import objective,metric_depth
STOP=False;WORKER='unset'
def stop(*_):
    global STOP
    STOP=True
signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)

def modules(seed):
    torch.manual_seed(seed);fusion=ShapeFusion().cuda()
    with torch.random.fork_rng(devices=[0]):torch.manual_seed(seed);pe=ParameterPE().cuda()
    with torch.random.fork_rng(devices=[0]):torch.manual_seed(seed);writer=TaskWriter().cuda()
    return fusion.eval(),pe.eval(),writer.eval()

def optims(core,fusion,pe,writer):
    return [torch.optim.AdamW(list(fusion.parameters())+list(pe.parameters())+list(writer.parameters()),lr=1e-4,weight_decay=.01),torch.optim.AdamW(core,lr=1e-6,weight_decay=.01,foreach=False)]

def prepared(rid,q,c,field,kind,trace=None):
    sample=d.device(d.sample(rid,q,c));sample.update(read_current(field,sample['query'],kind,trace));return sample

def prediction(model,mods,key,arm,trace=None,field=None,checkpoint=True,phase='training'):
    fusion,pe,writer=mods;rid,q,c=key
    if field is None:
        field=writer(d.device(d.basefield(rid)),arm[0]);d.charge(WORKER,phase,source_formations=1)
    sample=prepared(rid,q,c,field,arm[-1],trace);d.charge(WORKER,phase,native=1,raster=1)
    depth=forward_read(model,sample,fusion,pe,checkpoint_blocks=checkpoint)
    return depth,sample

def state(model,mods,opts,step,arm,seed,order):
    f,p,w=mods
    return dict(native=core_state(model),fusion=f.state_dict(),pe=p.state_dict(),writer=w.state_dict(),optimizer=[o.state_dict() for o in opts],rng=rng_state(),step=step,arm=arm,seed=seed,schedule_sha256=hashlib.sha256(json.dumps(order).encode()).hexdigest(),science_sha256=d.SCIENCE_SHA)

def save_serial(path,value):
    with (d.OUT/'checkpoint_io.lock').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX);assert not d.resource()['stop_reasons'];d.save(path,value);r=d.resource();assert not r['stop_reasons'],r

def restore(st,model,mods,opts=None):
    core_load(model,st['native'])
    for module,key in zip(mods,('fusion','pe','writer')):module.load_state_dict(st[key])
    if opts is not None:
        for opt in opts:opt.state.clear()
        torch.cuda.empty_cache()
        for opt,s in zip(opts,st['optimizer']):opt.load_state_dict(s)
    restore_rng(st['rng'])

def preflight(model):
    out=d.OUT/'preflight.json'
    if out.exists():assert d.read(out)['complete'];return
    core=enable_receiver(model);records=[];first_identity=None
    rid=d.plan()['training_catalog'][0]['id'];key=(rid,2,'missing')
    from experiments.native_render_parameter_data import geometry_path
    for arm in ('J_G','C_P'):
        restore_native(model,d.plan());mods=modules(d.SEEDS[0]);f,p,w=mods;identity=dict(fusion=digest(f),pe=digest(p),writer=digest(w))
        if first_identity is None:first_identity=identity
        else:assert identity==first_identity
        opt=optims(core,*mods);base=d.device(d.basefield(rid));field=w(base,arm[0]);initial={k:float((field[k].float()-base[k].float()).abs().max()) for k in ('features','means','scales','rotations','opacities')}
        assert max(initial.values())<1e-6,initial
        actual=read_current(field,d.device(d.sample(*key)['query']),arm[-1]);d.charge(WORKER,'preflight_fields',source_formations=1,raster=1)
        old=d.load(geometry_path(rid,2));errors=dict(center=float((actual['position'][:,:46].cpu()-old['center46']).abs().max()),alpha=float((actual['alpha'].cpu()-old['a']).abs().max()))
        assert errors['center']<2e-3 and errors['alpha']<2e-5,errors
        if arm=='C_P':assert actual['position'][:,46:].count_nonzero()==0
        steps=[];del base,field,actual
        for step in range(3):
            for o in opt:o.zero_grad(set_to_none=True)
            trace={};depth,s=prediction(model,mods,key,arm,trace=trace,phase='preflight');loss=objective(depth,s['query']['target'],s['query']['valid']);assert torch.isfinite(loss);loss.backward()
            grads=dict(writer_final=float(w.net[-1].weight.grad.norm()),PE_final=float(p.network[-1].weight.grad.norm()),fusion_final=float(f.output.weight.grad.norm()),frame=float(model.aggregator.frame_blocks[0].attn.qkv.weight.grad.norm()),global_last=float(model.aggregator.global_blocks[-1].attn.qkv.weight.grad.norm()))
            assert grads['frame']>0 and grads['global_last']>0 and grads['PE_final']>0
            if step==2:
                assert grads['writer_final']>0 and trace['features']>0 and trace['opacities']>0 and trace['ordinary6']>0
                if arm=='J_G':assert all(trace[k]>0 for k in ('means','scales','rotations','position'))
            norm=torch.nn.utils.clip_grad_norm_(list(f.parameters())+list(p.parameters())+list(w.parameters())+core,5.,foreach=False);assert torch.isfinite(norm)
            d.charge(WORKER,'preflight',discarded_updates=1)
            for o in opt:o.step()
            steps.append(dict(step=step+1,loss=float(loss.detach()),gradients=grads,field_gradients=trace));del depth,s,loss
        for o in opt:o.zero_grad(set_to_none=True)
        restore_result=None
        if arm=='J_G':
            with torch.no_grad():a,_=prediction(model,mods,key,arm,checkpoint=False,phase='restore_reference');a=a.cpu()
            path=d.CACHE/'preflight_full_resume.pt';save_serial(path,state(model,mods,opt,3,arm,d.SEEDS[0],[]));st=d.load(path)
            with torch.no_grad():w.net[-1].weight.add_(.01);model.aggregator.frame_blocks[0].attn.qkv.weight.add_(.01)
            restore(st,model,mods,opt)
            assert all(len(o.state)==len(s['state']) for o,s in zip(opt,st['optimizer']))
            checks=[]
            for o,saved in zip(opt,st['optimizer']):
                for live,sv in zip(o.state.values(),saved['state'].values()):
                    for n in ('step','exp_avg','exp_avg_sq'):
                        assert torch.equal(live[n].cpu(),sv[n]);checks.append(n)
            with torch.no_grad():b,_=prediction(model,mods,key,arm,checkpoint=False,phase='restore_replay');b=b.cpu()
            err=float((a-b).abs().max());assert err==0
            restore_result=dict(prediction_max_abs=err,all_Adam_state_tensors_exact=len(checks),bytes=path.stat().st_size,model_writer_fusion_PE_step_rng_loaded=True);del st,a,b;path.unlink()
        records.append(dict(arm=arm,initial_max_abs=initial,original_geometry_max_abs=errors,steps=steps,restore=restore_result));del mods,opt,f,p,w;model.zero_grad(set_to_none=True);torch.cuda.empty_cache()
    d.write(out,dict(complete=True,records=records,pilot_updates=6,discarded=True,no_extra_formal_fits=True,common_initial_identity=first_identity,peak_bytes=torch.cuda.max_memory_allocated()))

def fit(model,arm,seed):
    folder=d.OUT/'fits'/f'{arm}_{seed}';folder.mkdir(parents=True,exist_ok=True);core=enable_receiver(model);restore_native(model,d.plan());mods=modules(seed);order=schedule(d.plan(),seed)
    ident=dict(arm=arm,seed=seed,source_only=True,science_sha256=d.SCIENCE_SHA,schedule_sha256=hashlib.sha256(json.dumps(order).encode()).hexdigest(),module_sha256={k:digest(v) for k,v in zip(('fusion','pe','writer'),mods)},parameters={k:sum(x.numel() for x in v.parameters()) for k,v in zip(('fusion','pe','writer'),mods)},native_parameters=sum(x.numel() for x in core))
    if d.final(arm,seed).exists():
        st=d.load(d.final(arm,seed));assert st['step']==2000;restore(st,model,mods);del st;return mods
    opts=optims(core,*mods);first=0;path=d.latest(arm,seed)
    if path.exists():
        st=d.load(path);assert st['arm']==arm and st['seed']==seed and st['science_sha256']==d.SCIENCE_SHA and st['schedule_sha256']==ident['schedule_sha256'];restore(st,model,mods,opts);first=st['step'];del st
    d.write(folder/'identity.json',ident);d.journal(folder/'recoveries.jsonl',dict(utc=d.now(),step=first,pid=os.getpid(),optimizer_rng_loaded=bool(first)))
    logfile=folder/'training.jsonl'
    if logfile.exists():
        lines=logfile.read_text().splitlines();keep=[x for x in lines if json.loads(x)['step']<=first]
        if len(keep)<len(lines):
            (folder/f'discarded_{time.time_ns()}.jsonl').write_text('\n'.join(lines[len(keep):])+'\n');logfile.write_text('\n'.join(keep)+'\n')
    begin=time.monotonic();torch.cuda.reset_peak_memory_stats();allparams=core+[x for mod in mods for x in mod.parameters()]
    for step in range(first,2000):
        for opt in opts:opt.zero_grad(set_to_none=True)
        lossval=0.
        for key in order[2*step:2*step+2]:
            depth,s=prediction(model,mods,key,arm);loss=objective(depth,s['query']['target'],s['query']['valid']);assert torch.isfinite(loss),(arm,seed,step)
            (loss/2).backward();lossval+=float(loss.detach())/2;del depth,s,loss
        norm=torch.nn.utils.clip_grad_norm_(allparams,5.,foreach=False);assert torch.isfinite(norm)
        d.charge(WORKER,'formal',attempted_updates=1,arm=arm,seed=seed,step=step+1)
        for opt in opts:opt.step()
        d.journal(logfile,dict(step=step+1,loss=lossval,gradient_norm=float(norm),samples=order[2*step:2*step+2],elapsed_seconds=time.monotonic()-begin))
        if (step+1)%10==0:d.write(folder/'progress.json',dict(step=step+1,loss=lossval,elapsed_seconds=time.monotonic()-begin,peak_bytes=torch.cuda.max_memory_allocated()))
        if (step+1)%100==0 or STOP or step==1999:
            for opt in opts:opt.zero_grad(set_to_none=True)
            if step==1999:
                saved=state(model,mods,opts,step+1,arm,seed,order);del saved['optimizer'];save_serial(d.final(arm,seed),saved);del saved
                d.write(folder/'complete.json',dict(complete=True,steps=2000,path=str(d.final(arm,seed)),sha256=d.sha(d.final(arm,seed)),peak_bytes=torch.cuda.max_memory_allocated(),**ident))
                if path.exists():path.unlink()
            else:save_serial(path,state(model,mods,opts,step+1,arm,seed,order))
            print(dict(arm=arm,seed=seed,step=step+1,loss=lossval,seconds=time.monotonic()-begin),flush=True)
            if STOP:raise SystemExit(75)
    del opts;model.zero_grad(set_to_none=True);d.sample.cache_clear();d.basefield.cache_clear();torch.cuda.empty_cache();return mods

@torch.no_grad()
def evaluate(model,mods,arm,seed):
    from experiments.native_render_parameter_data import geometry_path
    for meta in d.metas():
        rid=meta['id'];marker=d.OUT/'evaluation'/f'{arm}_{seed}_{rid}.json'
        if marker.exists():assert d.read(marker)['complete'];continue
        path=d.fieldpath(arm,seed,rid);fm=d.OUT/'formation'/f'{arm}_{seed}_{rid}.json'
        if not path.exists():
            start=time.monotonic();field=mods[2](d.device(d.basefield(rid)),arm[0]);torch.cuda.synchronize();formation_seconds=time.monotonic()-start
            d.charge(WORKER,'evaluation_formation',source_formations=1);d.save(path,d.cpu(field));del field
            d.write(fm,dict(id=rid,arm=arm,seed=seed,source_input_sha256=d.sha(d.old.scene_path(rid)),path=str(path),sha256=d.sha(path),bytes=path.stat().st_size,formation_seconds=formation_seconds,query_inputs=False,per_scene_trainable=False,one_source_state_for_queries=8,conditions=2))
        field=d.load(path,'cuda');base=d.device(d.basefield(rid));drift={k:float((field[k].float()-base[k].float()).square().mean().sqrt()) for k in ('means','scales','rotations','opacities','features')}
        assert all(torch.equal(field[k],base[k]) for k in ('has_content','source_ids','unit','source_c2w','source_intrinsic'))
        keys=[];preds=[];coverage=[];identities=[];start=time.monotonic()
        for q in d.old.EVAL_QUERIES:
            query=d.device(d.sample(rid,q,'full')['query']);view=read_current(field,query,arm[-1]);d.charge(WORKER,'evaluation_view',raster=1)
            old=d.load(geometry_path(rid,q));coverage.append(dict(query=q,old_zero_all=bool((old['a']==0).all()),new_zero_all=bool((view['alpha']==0).all()),old_zero_patches=int((old['a']==0).sum()),new_zero_patches=int((view['alpha']==0).sum()),new_alpha_mean=float(view['alpha'].mean())))
            identities.append(dict(query=q,query_sha256=d.sha(d.old.query_path(rid,q)),full_patch_sha256=d.sha(d.old.patch_path(rid,q,'full')),missing_patch_sha256=d.sha(d.old.patch_path(rid,q,'missing'))))
            for c in d.old.CONDITIONS:
                s=d.device(d.sample(rid,q,c));s.update(view);d.charge(WORKER,'evaluation_query',native=1)
                pred=metric_depth(forward_read(model,s,mods[0],mods[1],checkpoint_blocks=False)).float().cpu();assert torch.isfinite(pred).all() and (pred>0).all();keys.append((rid,q,c));preds.append(pred);del pred,s
        predpath=d.CACHE/'predictions'/f'{arm}_{seed}'/f'{rid}.pt';d.save(predpath,dict(keys=keys,depth_metric=torch.stack(preds)))
        d.write(marker,dict(complete=True,id=rid,place=meta['place'],arm=arm,seed=seed,predictions=16,path=str(predpath),sha256=d.sha(predpath),field_sha256=d.sha(path),field_built_once=True,field_loaded_once=True,query_writer_calls=0,geometry_drift_RMS=drift,coverage=coverage,input_identities=identities,seconds=time.monotonic()-start));print(dict(evaluated=rid,arm=arm,seed=seed),flush=True)
        del field,base,view,preds;d.sample.cache_clear();d.basefield.cache_clear();torch.cuda.empty_cache()
        if STOP:raise SystemExit(75)

def main():
    global WORKER
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['preflight','fit']);p.add_argument('--arm',choices=d.ARMS);p.add_argument('--seed',type=int,choices=d.SEEDS);p.add_argument('--worker',required=True);a=p.parse_args();WORKER=a.worker
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.80);assert torch.cuda.is_available();assert d.sha(d.SCIENCE)==d.SCIENCE_SHA
    config=d.read(d.OUT/'configuration.json');assert all(d.sha(d.ROOT/n)==v for n,v in config['source_sha256'].items());assert not d.resource(True)['stop_reasons']
    model=model_load(d.plan());print(dict(model_loaded=True,phase=a.phase),flush=True)
    if a.phase=='preflight':preflight(model)
    else:
        assert d.read(d.OUT/'preflight.json')['complete'];claim=d.OUT/'claims'/f'{a.arm}_{a.seed}.lock';claim.parent.mkdir(exist_ok=True)
        with claim.open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);mods=fit(model,a.arm,a.seed);evaluate(model,mods,a.arm,a.seed)
    print(dict(process_complete=True,phase=a.phase,arm=a.arm,seed=a.seed),flush=True)
if __name__=='__main__':main()
