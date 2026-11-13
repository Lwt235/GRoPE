"""Round006 bounded training, persisted K/V inference and exact shared-call costs."""
import argparse
import inspect
import random
import signal
import time
import numpy as np
import torch
from experiments import camera_only_feature_data as d
from models.camera_only_feature_reader import CameraOnlyFeatureReader,view_fields,anchor_from_composite
from models.gaussian_scene_memory import rasterize_field
from models.gaussian_scene_native import forward_read
from experiments.gaussian_scene_preflight import metric_depth

STOP=False
START=0.
WORKER=''
def request_stop(*args):
    global STOP
    STOP=True

def heartbeat(phase,**kw):
    d.write(d.OUT/'workers'/f'{WORKER}.json',dict(pid=__import__('os').getpid(),utc=d.now(),phase=phase,elapsed_seconds=time.monotonic()-START,peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30,**kw))

def rng_state():
    n=np.random.get_state()
    return dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(),numpy=dict(name=n[0],keys=n[1].tolist(),pos=n[2],has_gauss=n[3],cached=n[4]),python=random.getstate())
def restore_rng(r):
    torch.set_rng_state(r['cpu'].cpu());torch.cuda.set_rng_state(r['cuda'].cpu());n=r['numpy'];np.random.set_state((n['name'],np.asarray(n['keys'],dtype=np.uint32),n['pos'],n['has_gauss'],n['cached']));random.setstate(r['python'])
def model_new(seed):
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);return CameraOnlyFeatureReader().cuda()
def digest(model):
    import hashlib
    h=hashlib.sha256()
    for name,t in model.state_dict().items():h.update(name.encode());h.update(t.detach().cpu().numpy().tobytes())
    return h.hexdigest()
def optimizer(reader):return torch.optim.AdamW(reader.parameters(),lr=3e-4,weight_decay=.01)

def source_input(rid,permuted=False):
    field=d.load(d.old.scene_path(rid),'cuda');g=d.load(d.fields_path(rid),'cuda')
    if permuted:
        perm=d.load(d.CACHE/'permutations'/f'{rid}.pt','cuda')['permutation'];field=dict(field,features=field['features'][perm])
    return d.ordinary(field,g),g,field

def read_one(reader,kv,view,b,a,arm):
    return reader.read(b,a,view['camera20'],view['common25'],view[arm],view['key_mask'],kv[0],kv[1])

@torch.no_grad()
def render_anchor(field,query,purpose):
    d.charge(purpose,raster=1);K=query['K'].clone();K[:2]/=14
    colors=torch.cat((field['features'].float()*field['has_content'][:,None],field['has_content'].float()[:,None]),-1)
    r,_,_=rasterize_field(field,query['c2w'][None],K[None],32,32,colors)
    return anchor_from_composite(r[0,:,:,:1024].reshape(1024,1024),r[0,:,:,1024].reshape(-1,1))

@torch.no_grad()
def native(model,patches,fill=None):
    changed=patches.clone()
    if fill is not None:changed.reshape(1,32,32,1024)[:,8:24,8:24]=fill.reshape(32,32,1024)[8:24,8:24].to(changed.dtype)
    with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):depth=forward_read(model,dict(patches=changed,hw=(448,448)),reader=None,checkpoint_blocks=False)
    out=metric_depth(depth).float()
    assert torch.isfinite(out).all() and bool((out>0).all())
    return out

def native_load():
    from experiments.native_receiver_runtime import model_load
    t=time.monotonic();model=model_load(d.old.plan());torch.cuda.synchronize()
    d.write(d.OUT/'workers'/f'{WORKER}_native_load.json',dict(seconds=time.monotonic()-t,path=d.old.plan()['model_path'],frozen=True,checkpoint_rehash=False))
    return model

def snapshot(reader,opt,step,arm,seed,order):
    return dict(model=reader.state_dict(),optimizer=opt.state_dict(),step=step,arm=arm,seed=seed,order=order,rng=rng_state(),budget=d.read(d.OUT/'budget.json'),configuration_sha256=d.sha(d.OUT/'configuration_lock.json'))

def preflight():
    if (d.OUT/'preflight.json').exists():assert d.read(d.OUT/'preflight.json')['complete'];return
    assert not d.resources(admission=True)['stop_reasons'];checks=[];shared=[]
    recovered=(d.CACHE/'preflight/resume.pt').exists() and d.read(d.OUT/'budget.json')['counts']['pilot_updates']>0
    native_model=None if recovered else native_load()
    for meta in d.contract()['training_catalog'][:2]:
        rid=meta['id'];q=2;x,g,f=source_input(rid);v=d.load(d.view_path(rid,q),'cuda');b,a=d.cached_anchor(rid,q);b=b.cuda();a=a.cuda()
        if recovered:
            shared.append((rid,x,g,v,b,a));checks.append(dict(id=rid,initial_assertions_passed_before_recorded_optimizer_device_assertion=True,actual_render_and_native_repeated=False));continue
        query=d.load(d.old.query_path(rid,q),'cuda');actual,aa=render_anchor(f,query,'preflight')
        checks.append(dict(id=rid,anchor_cache_vs_real_max_abs=float((actual-b).abs().max()),alpha_cache_vs_real_max_abs=float((aa-a).abs().max()),same_physical_render_path=True,valid_keys=int(v['key_mask'].sum())))
        shared.append((rid,x,g,v,b,a))
        for arm in d.ARMS:
            reader=model_new(d.SEEDS[0]);d.charge('preflight',source=1,reader=1)
            with torch.no_grad():kv=reader.encode_source(x);pred=read_one(reader,kv,v,b,a,arm)
            assert torch.equal(pred,b)
            if arm=='P':
                patches=d.load(d.old.patch_path(rid,q,'missing'),'cuda')['patches'];changed=patches.clone();changed.reshape(1,32,32,1024)[:,8:24,8:24]=pred.reshape(32,32,1024)[8:24,8:24]
                outside=torch.ones((32,32),device='cuda',dtype=torch.bool);outside[8:24,8:24]=False
                assert torch.equal(changed.reshape(32,32,1024)[outside],patches.reshape(32,32,1024)[outside])
                d.charge('preflight',native=1);depth=native(native_model,patches,pred);assert depth.shape==(210,210)
                p=d.CACHE/'preflight'/f'{rid}_state.pt';d.tensor_save(p,dict(key=kv[0],value=kv[1],geometry=g));reload=d.load(p,'cuda');assert torch.equal(reload['key'],kv[0]) and torch.equal(reload['value'],kv[1])
    parameter_counts={};identities={};gradients={}
    for arm in d.ARMS:
        reader=model_new(d.SEEDS[0]);opt=optimizer(reader);identities[arm]=digest(reader);parameter_counts[arm]=sum(p.numel() for p in reader.parameters())
        pilot_start=0
        if recovered and arm=='P':
            saved=d.load(d.CACHE/'preflight/resume.pt');reader.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);restore_rng(saved['rng']);pilot_start=saved['step']
            assert pilot_start==1
            assert all(torch.equal(saved['optimizer']['state'][i][k].cpu(),opt.state_dict()['state'][i][k].cpu()) for i in saved['optimizer']['state'] for k in ('step','exp_avg','exp_avg_sq'))
        for step in range(pilot_start,2):
            opt.zero_grad(set_to_none=True);loss=0
            for rid,x,g,v,b,a in shared:
                target=d.load(d.old.patch_path(rid,2,'full'),'cuda')['patches'][0]
                kv=reader.encode_source(x);pred=read_one(reader,kv,v,b,a,arm);part=(pred-target).square().mean()/2;part.backward();loss+=float(part.detach())
            gradients[arm]={name:float(p.grad.norm()) for name,p in reader.named_parameters() if p.grad is not None}
            torch.nn.utils.clip_grad_norm_(reader.parameters(),5.);d.charge('pilot',pilot_updates=1,pilot_reader=2,pilot_source=2);opt.step()
            if arm=='P' and step==0:
                p=d.CACHE/'preflight/resume.pt';d.tensor_save(p,snapshot(reader,opt,step+1,arm,d.SEEDS[0],[]));saved=d.load(p)
                other=model_new(d.SEEDS[0]);other.load_state_dict(saved['model']);other_opt=optimizer(other);other_opt.load_state_dict(saved['optimizer']);restore_rng(saved['rng'])
                rid,x,g,v,b,a=shared[0]
                with torch.no_grad():
                    d.charge('preflight_resume',source=2,reader=2)
                    one=read_one(reader,reader.encode_source(x),v,b,a,arm);two=read_one(other,other.encode_source(x),v,b,a,arm)
                assert torch.equal(one,two)
                assert digest(reader)==digest(other)
                assert all(torch.equal(opt.state_dict()['state'][i][k].cpu(),other_opt.state_dict()['state'][i][k].cpu()) for i in opt.state_dict()['state'] for k in ('step','exp_avg','exp_avg_sq'))
                reader,opt=other,other_opt
        assert gradients[arm]['key.weight']>0 and gradients[arm]['position.2.weight']>0 and gradients[arm]['decoder.2.weight']>0
    assert len(set(identities.values()))==1 and len(set(parameter_counts.values()))==1
    d.write(d.OUT/'preflight.json',dict(complete=True,checks=checks,parameter_counts=parameter_counts,paired_initial_sha256=identities,gradient_norms=gradients,pilot_updates=6,pilot_samples=12,pilot_discarded_and_formal_reset=True,exact_full_resume=True,reader_API_parameters=list(inspect.signature(CameraOnlyFeatureReader.read).parameters),no_native_training_forward=True,geometry_finite=True,background_token_always_present=True,zero_residual_exactly_anchor=True,reload_KV_exact=True))
    del reader,opt,native_model,shared;torch.cuda.empty_cache()

def stage_training():
    sources={};samples={}
    for meta in d.contract()['training_catalog']:
        rid=meta['id'];x,_,_=source_input(rid);sources[rid]=x
        for q in d.contract()['fit_queries']:
            b,a=d.cached_anchor(rid,q);samples[rid,q]=dict(view=d.load(d.view_path(rid,q),'cuda'),b=b.cuda(),a=a.cuda(),target=d.load(d.old.patch_path(rid,q,'full'),'cuda')['patches'][0])
    return sources,samples

def schedule(seed):
    cases=[(m['id'],q) for m in d.contract()['training_catalog'] for q in d.contract()['fit_queries']]
    rng=np.random.default_rng(seed);order=[]
    while len(order)<4000:order.extend([list(cases[i]) for i in rng.permutation(len(cases))])
    return order[:4000]

def train(arm,seed,sources,samples):
    folder=d.OUT/'fits'/f'{arm}_{seed}';status=folder/'status.json';final=d.final_path(arm,seed)
    if status.exists():assert d.read(status)['complete'];return
    reader=model_new(seed);opt=optimizer(reader);order=schedule(seed);initial=digest(reader);first=0;resume=d.CACHE/'resumes'/f'{arm}_{seed}.pt'
    if resume.exists():
        state=d.load(resume);assert state['arm']==arm and state['seed']==seed and state['order']==order
        reader.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);restore_rng(state['rng']);first=state['step']
    d.write(folder/'initial_identity.json',dict(seed=seed,arm=arm,initial_sha256=initial,parameter_count=sum(x.numel() for x in reader.parameters()),schedule=order,precision='FP32',fixed_final_step=2000))
    start=time.monotonic();last_resource=0;space=None
    for step in range(first,2000):
        opt.zero_grad(set_to_none=True);keys=[tuple(k) for k in order[step*2:step*2+2]]
        inp=torch.stack([sources[k[0]] for k in keys]);kv=reader.encode_source(inp)
        vals=[samples[k] for k in keys];stack=lambda name:torch.stack([v[name] for v in vals]);vs=lambda name:torch.stack([v['view'][name] for v in vals])
        prediction=reader.read(stack('b'),stack('a'),vs('camera20'),vs('common25'),vs(arm),vs('key_mask'),*kv)
        loss=(prediction-stack('target')).square().mean();assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(reader.parameters(),5.);assert torch.isfinite(norm)
        d.charge('formal_training',formal_updates=1,training_reader=2,training_source=2);opt.step()
        d.journal(folder/'training.jsonl',dict(step=step+1,loss=float(loss.detach()),gradient_norm=float(norm),samples=order[step*2:step*2+2],seconds=time.monotonic()-start))
        resource_stop=False
        if time.monotonic()-last_resource>=30:
            space=d.resources(space);last_resource=time.monotonic();resource_stop=bool(space['stop_reasons']);heartbeat('training',arm=arm,seed=seed,step=step+1)
        if (step+1)%100==0 or STOP or resource_stop:
            d.tensor_save(resume,snapshot(reader,opt,step+1,arm,seed,order));d.write(folder/'progress.json',dict(step=step+1,loss=float(loss.detach()),seconds=time.monotonic()-start))
            print(dict(arm=arm,seed=seed,step=step+1,loss=float(loss.detach()),seconds=time.monotonic()-start),flush=True)
            if STOP or resource_stop:raise SystemExit(75)
        del prediction,loss,inp,kv
    opt.zero_grad(set_to_none=True);d.tensor_save(final,snapshot(reader,opt,2000,arm,seed,order));h=d.sha(final)
    saved=d.load(final);assert saved['step']==2000 and saved['order']==order
    d.write(status,dict(complete=True,arm=arm,seed=seed,steps=2000,path=str(final),sha256=h,bytes=final.stat().st_size,seconds=time.monotonic()-start,initial_sha256=initial,model_sha256=digest(reader),full_optimizer_RNG_sampler_budget_retained=True))
    if resume.exists():resume.unlink()
    del reader,opt;torch.cuda.empty_cache()

def permuted_indices():
    rng=np.random.default_rng(20261113)
    for meta in sorted(d.contract()['development_catalog'],key=lambda m:m['id']):
        rid=meta['id'];p=d.CACHE/'permutations'/f'{rid}.pt'
        if p.exists():
            # Advance deterministically even when resuming without regenerating mappings.
            valid=d.load(d.fields_path(rid))['valid'];rng.permutation(int(valid.sum()));continue
        valid=d.load(d.fields_path(rid))['valid'];ids=torch.nonzero(valid).flatten();perm=torch.arange(len(valid));perm[ids]=ids[torch.from_numpy(rng.permutation(len(ids)))]
        d.tensor_save(p,dict(permutation=perm,seed=20261113,scene_order='sorted development id',valid_slots_only=True))

@torch.no_grad()
def export_state(reader,arm,seed,meta,permuted=False):
    rid=meta['id'];path=d.memory_path(arm,seed,rid,permuted);marker=d.OUT/'states'/f'{arm}_{seed}_{rid}{"_permuted" if permuted else ""}.json'
    if marker.exists():return d.load(path,'cuda'),d.read(marker)
    x,g,_=source_input(rid,permuted);d.charge('export_permuted' if permuted else 'export_normal',source=1)
    torch.cuda.synchronize();t=time.monotonic();key,value=reader.encode_source(x);torch.cuda.synchronize();encoded=time.monotonic()-t
    state=dict(key=key,value=value,geometry=g,base_path=str(d.old.scene_path(rid)),base_sha256=d.read(d.OUT/'preparation'/f'{rid}.json')['inputs'][str(d.old.scene_path(rid))]['sha256_inherited'],arm=arm,seed=seed,permuted=permuted)
    t=time.monotonic();d.tensor_save(path,state);h=d.sha(path);saved=time.monotonic()-t;t=time.monotonic();reloaded=d.load(path,'cuda');torch.cuda.synchronize();loaded=time.monotonic()-t
    assert torch.equal(key,reloaded['key']) and torch.equal(value,reloaded['value'])
    record=dict(complete=True,id=rid,arm=arm,seed=seed,permuted=permuted,path=str(path),sha256=h,bytes=path.stat().st_size,source_encode_seconds=encoded,serialization_seconds=saved,reload_seconds=loaded,base_reference=state['base_path'],base_sha256=state['base_sha256'],source_RGB_or_query_inputs_in_export=False,queries_share_one_reload=True)
    d.write(marker,record);return reloaded,record

@torch.no_grad()
def evaluate(model,arm,seed):
    reader=model_new(seed);reader.load_state_dict(d.load(d.final_path(arm,seed),'cuda')['model']);reader.eval();space=None;last=0
    for meta in d.metas():
        for permuted in ([False,True] if arm=='G' and meta in d.contract()['development_catalog'] else [False]):
            rid=meta['id'];state,record=export_state(reader,arm,seed,meta,permuted);g=state['geometry'];kv=(state['key'],state['value'])
            field=source_input(rid,True)[2] if permuted else None
            for q in (d.contract()['fit_queries'] if permuted else d.contract()['evaluation_queries']):
                key=f'{arm}_{seed}_{rid}_{q}{"_permuted" if permuted else ""}';marker=d.OUT/'completed'/f'{key}.json'
                if marker.exists():assert d.read(marker)['complete'];continue
                query=d.load(d.old.query_path(rid,q),'cuda')
                if permuted:b,a=render_anchor(field,query,'formal_permuted')
                else:b,a=d.cached_anchor(rid,q);b=b.cuda();a=a.cuda()
                view=d.load(d.view_path(rid,q),'cuda');d.charge('formal_permuted' if permuted else 'formal_normal',reader=1,native=1)
                prediction=read_one(reader,kv,view,b,a,arm);assert prediction.dtype==torch.float32 and torch.isfinite(prediction).all()
                patches=d.load(d.old.patch_path(rid,q,'missing'),'cuda')['patches'];depth=native(model,patches,prediction)
                fp=d.CACHE/'features'/f'{arm}_{seed}'/f'{rid}_{q}{"_permuted" if permuted else ""}.npy';dp=d.CACHE/'depths'/f'{arm}_{seed}'/fp.name
                d.numpy_save(fp,prediction.cpu().numpy());d.numpy_save(dp,depth.cpu().numpy())
                d.write(marker,dict(complete=True,id=rid,place=meta['place'],query=q,arm=arm,seed=seed,permuted=permuted,feature_path=str(fp),feature_sha256=d.sha(fp),depth_path=str(dp),depth_sha256=d.sha(dp),state_sha256=record['sha256'],source_encoder_calls_at_read=0,source_RGB_calls_at_read=0))
                if time.monotonic()-last>=30:
                    space=d.resources(space);last=time.monotonic();heartbeat('evaluation',arm=arm,seed=seed,id=rid,query=q)
                    if STOP or space['stop_reasons']:raise SystemExit(75)
            del state,kv,field;torch.cuda.empty_cache()
        print(dict(evaluated=arm,seed=seed,id=meta['id']),flush=True)
    del reader

@torch.no_grad()
def baselines(model):
    for meta in d.metas():
        rid=meta['id']
        for q in d.contract()['evaluation_queries']:
            marker=d.OUT/'baselines'/f'{rid}_{q}.json'
            if marker.exists():continue
            b,_=d.cached_anchor(rid,q);b=b.cuda();rows={}
            for name,condition,fill in [('Native','missing',None),('RasterFill','missing',b),('Native_full','full',None)]:
                d.charge('formal_baseline',native=1);patches=d.load(d.old.patch_path(rid,q,condition),'cuda')['patches'];depth=native(model,patches,fill)
                path=d.CACHE/'depths'/name/f'{rid}_{q}.npy';d.numpy_save(path,depth.cpu().numpy());rows[name]=dict(path=str(path),sha256=d.sha(path))
            d.write(marker,dict(complete=True,id=rid,place=meta['place'],query=q,methods=rows))
        heartbeat('baselines',id=rid);space=d.resources()
        if STOP or space['stop_reasons']:raise SystemExit(75)
        print(dict(baseline_scene=rid),flush=True)

@torch.no_grad()
def costs(model):
    if (d.OUT/'cost_measured.json').exists():return
    from models.gaussian_memory_coarse_pe import patch_tokens
    from experiments.gaussian_scene_data import query_image
    model.aggregator.patch_embed.cuda();records=[];build=[];metadata=[]
    readers={};states={};fields={};queries={};geos={}
    for arm in d.ARMS:
        r=model_new(d.SEEDS[0]);r.load_state_dict(d.load(d.final_path(arm,d.SEEDS[0]),'cuda')['model']);readers[arm]=r.eval()
    for meta in d.contract()['training_catalog'][:2]:
        rid=meta['id'];q=2;queries[rid]=d.load(d.old.query_path(rid,q),'cuda');fields[rid]=d.load(d.old.scene_path(rid),'cuda')
        t=time.monotonic();gc,stats=d.source_geometry(d.load(d.old.scene_path(rid)));metadata.append(dict(id=rid,**stats));geos[rid]=d.to_device(gc)
        for arm in d.ARMS:
            t=time.monotonic();state=d.load(d.memory_path(arm,d.SEEDS[0],rid),'cuda');torch.cuda.synchronize();load_s=time.monotonic()-t;states[arm,rid]=state
            x=d.ordinary(fields[rid],geos[rid]);times=[]
            for i in range(25):
                d.charge('cost_source',source=1);torch.cuda.synchronize();t=time.monotonic();readers[arm].encode_source(x);torch.cuda.synchronize()
                if i>=5:times.append(time.monotonic()-t)
            build.append(dict(arm=arm,id=rid,source_encoding_seconds=times,cold_state_load_seconds=load_s,base_bytes=d.old.scene_path(rid).stat().st_size,derived_bytes=d.memory_path(arm,d.SEEDS[0],rid).stat().st_size))
        names=['Native','RasterFill','P','S','G']
        for i in range(20):
            for offset in range(5):
                name=names[(i+offset)%5];query=queries[rid];field=fields[rid];d.charge('cost_native',native=1)
                torch.cuda.synchronize();wall=time.monotonic();ev=[torch.cuda.Event(enable_timing=True) for _ in range(6)];ev[0].record()
                with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):patches=patch_tokens(model,query_image(query,'missing'))
                ev[1].record();fill=None;reader_ms=None
                if name!='Native':
                    b,a=render_anchor(field,query,'cost_actual_raster');ev[2].record()
                    if name in d.ARMS:
                        v=view_fields(geos[rid],query['c2w'],query['K']);ev[3].record();d.charge('cost_reader_inside_native',reader=1)
                        fill=read_one(readers[name],(states[name,rid]['key'],states[name,rid]['value']),v,b,a,name);ev[4].record()
                    else:fill=b;ev[3].record();ev[4].record()
                else:ev[2].record();ev[3].record();ev[4].record()
                native(model,patches,fill);ev[5].record();torch.cuda.synchronize();end=time.monotonic()
                records.append(dict(method=name,id=rid,query=q,iteration=i,warmup=i<5,wall_seconds=end-wall,total_GPU_ms=ev[0].elapsed_time(ev[5]),encoder_GPU_ms=ev[0].elapsed_time(ev[1]),raster_GPU_ms=ev[1].elapsed_time(ev[2]),position_GPU_ms=ev[2].elapsed_time(ev[3]),reader_GPU_ms=ev[3].elapsed_time(ev[4]),memory_read_GPU_ms=ev[1].elapsed_time(ev[4]),native_GPU_ms=ev[4].elapsed_time(ev[5])))
        heartbeat('cost',id=rid)
    d.write(d.OUT/'cost_measured.json',dict(complete=True,records=records,source_records=build,center_geometry_preparation=metadata,native_calls=200,reader_calls_shared_inside_native=120,extra_reader_benchmark_calls=0,source_calls=150,real_raster_calls=160,scope='Preprocessed missingRGB + saved base source, given cameras; actual image encoder, real GS raster and actual FP64 position plus reader, original native/DPT. Event segments labeledGPU, CPUwall includes accounting overhead. Disk/JPEG/camera estimation separate. No cached render/feature geometry substituted in cost.',shared_GPU_timing_conditions=True))

def main():
    global START,WORKER
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['preflight','worker']);parser.add_argument('--worker',default='preflight');parser.add_argument('--fits',default='');parser.add_argument('--baselines',action='store_true');parser.add_argument('--cost',action='store_true');args=parser.parse_args()
    START=time.monotonic();WORKER=args.worker;torch.set_num_threads(4);signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    free,total=torch.cuda.mem_get_info();assert free>=16*2**30;torch.cuda.set_per_process_memory_fraction(10*2**30/total);assert not d.resources(admission=True)['stop_reasons'];heartbeat('started',initial_free_GiB=free/2**30)
    if args.mode=='preflight':preflight()
    else:
        assert d.read(d.OUT/'preflight.json')['complete'];permuted_indices();model=native_load()
        if args.baselines:baselines(model)
        sources,samples=stage_training()
        for spec in args.fits.split(','):
            arm,seed=spec.split(':');seed=int(seed);train(arm,seed,sources,samples);evaluate(model,arm,seed)
        del sources,samples;torch.cuda.empty_cache()
        if args.cost:costs(model)
    heartbeat('complete',normal_exit=True);print(dict(worker=WORKER,complete=True,seconds=time.monotonic()-START),flush=True)

if __name__=='__main__':main()
