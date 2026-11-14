"""Round019 finite cache preparation, minimal new-interface preflight and nine fits."""
import argparse,copy,fcntl,json,os,signal,time
import torch
from experiments import native_read_address_data as d
from models.native_read_address import AddressBias,weights,center_kernel,centered_kernel,content_view,sphere_field,forward_read,normalize
from experiments.native_receiver_runtime import model_load
from experiments.native_receiver_data import core_load
from models.native_task_writer import ShapeFusion,read_current
from models.gaussian_scene_parameter_pe import ParameterPE,forward_read as original_forward
from experiments.gaussian_scene_preflight import objective,metric_depth
from experiments.native_render_parameter_train import rng_state,restore_rng
STOP=False;WORKER='unset'
def stop(*_):
    global STOP
    STOP=True
signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)

def base(model,seed):
    st=d.load(d.parent.final('C_P',seed));core_load(model,st['native']);fusion=ShapeFusion().cuda();fusion.load_state_dict(st['fusion']);pe=ParameterPE().cuda();pe.load_state_dict(st['pe']);del st
    model.eval().requires_grad_(False);fusion.eval().requires_grad_(False);pe.eval().requires_grad_(False)
    return fusion,pe

def marker(seed,rid,q):return d.OUT/'preparation'/'views'/f'{seed}_{rid}_{q}.json'

@torch.no_grad()
def prepare(seed):
    for meta in d.metas():
        rid=meta['id'];field=d.load(d.parent.fieldpath('C_P',seed,rid),'cuda');fieldmeta=d.read(d.parent.OUT/'formation'/f'C_P_{seed}_{rid}.json');assert fieldmeta['sha256']==d.sha(d.parent.fieldpath('C_P',seed,rid))
        C,cmeta=center_kernel(field);d.charge(WORKER,'source_center_kernel',center_kernel=1)
        for q in d.parent.old.EVAL_QUERIES:
            path=d.viewpath(seed,rid,q);mp=marker(seed,rid,q)
            if mp.exists():assert d.read(mp)['sha256']==d.sha(path);continue
            query=d.device(d.sample(rid,q,'full')['query']);torch.cuda.synchronize();start=time.monotonic();W=weights(field,query);d.charge(WORKER,'prepare_actual_W',raster=1);common=content_view(field,query,W);WS=weights(sphere_field(field),query);d.charge(WORKER,'prepare_shadow_W',raster=1)
            kernels={a:centered_kernel(W,a,C,WS) for a in d.ARMS};assert all(torch.isfinite(k).all() for k in kernels.values());assert all(k[common['alpha']==0].count_nonzero()==0 for a,k in kernels.items() if a in ('A','P'))
            torch.cuda.synchronize();seconds=time.monotonic()-start;payload=dict(**common,kernels=kernels,identity=dict(seed=seed,id=rid,query=q,field_sha256=fieldmeta['sha256'],query_sha256=d.sha(d.parent.old.query_path(rid,q)),science_sha256=d.SCIENCE_SHA));d.save(path,d.cpu(payload))
            d.write(mp,dict(complete=True,path=str(path),sha256=d.sha(path),bytes=path.stat().st_size,seed=seed,id=rid,query=q,field_path=str(d.parent.fieldpath('C_P',seed,rid)),field_sha256=fieldmeta['sha256'],query_sha256=payload['identity']['query_sha256'],seconds_build=seconds,center_kernel=cmeta,zero_actual_rows=int((W.sum(-1)==0).sum()),zero_shadow_rows=int((WS.sum(-1)==0).sum()),row_mean_max_abs={a:float(k.mean(-1).abs().max()) for a,k in kernels.items()},writer_calls=0,current_RGB_or_target_inputs=False))
            del W,WS,common,kernels,query,payload
            if STOP:raise SystemExit(75)
        print(dict(prepared=rid,seed=seed),flush=True);del field,C;d.sample.cache_clear();torch.cuda.empty_cache()
    d.write(d.OUT/'preparation'/f'complete_{seed}.json',dict(complete=True,seed=seed,views=352,fields=44,source_only_states_reused=True))

def state(address,opt,step,arm,seed):return dict(gamma=address.state_dict(),optimizer=opt.state_dict(),rng=rng_state(),step=step,arm=arm,seed=seed,schedule_sha256=d.schedule_sha(seed),science_sha256=d.SCIENCE_SHA,base_checkpoint_sha256=d.read(d.parent.OUT/'fits'/f'C_P_{seed}'/'complete.json')['sha256'])
def restore(st,address,opt=None):
    assert st['science_sha256']==d.SCIENCE_SHA;address.load_state_dict(st['gamma'])
    if opt is not None:opt.load_state_dict(st['optimizer'])
    restore_rng(st['rng'])
def predict(model,mods,address,key,seed,arm,phase,checkpoint=True):
    s=d.prepared(key,seed,arm);d.charge(WORKER,phase,native=1);return forward_read(model,s,*mods,address,checkpoint_blocks=checkpoint),s

def preflight(model):
    seed=d.SEEDS[0];key=(d.plan()['training_catalog'][0]['id'],2,'missing');mods=base(model,seed);address=AddressBias().cuda();opt=torch.optim.Adam(address.parameters(),lr=.01,weight_decay=0)
    frozen=[p for module in (model,*mods) for p in module.parameters()];versions=[p._version for p in frozen];assert all(not p.requires_grad for p in frozen)
    s=d.prepared(key,seed,'A');field=d.load(d.parent.fieldpath('C_P',seed,key[0]),'cuda')
    with torch.no_grad():
        oldview=read_current(field,s['query'],'P');d.charge(WORKER,'preflight_content',raster=1);viewdiff={k:float((oldview[k]-s[k]).abs().max()) for k in ('extra','position','alpha')}
        ref=original_forward(model,s,*mods,checkpoint_blocks=False);d.charge(WORKER,'preflight_reference',native=1);zeros=[]
        saved=d.load(d.parent.CACHE/'predictions'/f'C_P_{seed}'/f'{key[0]}.pt');j=[tuple(k) for k in saved['keys']].index(key);oldpred=saved['depth_metric'][j].cuda();zeros_vs_saved=[]
        for arm in d.ARMS:
            pred,ss=predict(model,mods,address,key,seed,arm,'preflight_zero',False);zeros.append(float((pred-ref).abs().max()));zeros_vs_saved.append(float((metric_depth(pred)-oldpred).abs().max()));del pred,ss
    # Actual A definition independently checked in FP64 at one fixed real view.
    with torch.no_grad():
        W=weights(field,s['query']);d.charge(WORKER,'preflight_definition',raster=1);raw=normalize(W.double()@W.double().T);expected=(raw-raw.mean(-1,keepdim=True)).float();definition_error=float((expected-s['kernel']).abs().max());del W,raw,expected
    pred,ss=predict(model,mods,address,key,seed,'A','preflight_gradient');loss=objective(pred,ss['query']['target'],ss['query']['valid']);loss.backward();gradient=address.gamma.grad.detach().clone();assert torch.isfinite(gradient).all() and gradient.abs().sum()>0
    norm=torch.nn.utils.clip_grad_norm_(address.parameters(),5.);opt.step();d.charge(WORKER,'preflight',discarded_updates=1);opt.zero_grad(set_to_none=True);del pred,ss,loss
    assert versions==[p._version for p in frozen] and all(p.grad is None for p in frozen)
    with torch.no_grad():a,_=predict(model,mods,address,key,seed,'A','restore_reference',False)
    path=d.CACHE/'preflight_resume.pt';d.save(path,d.cpu(state(address,opt,1,'A',seed)));saved=d.load(path);address.gamma.data.add_(.5);restore(saved,address,opt)
    with torch.no_grad():b,_=predict(model,mods,address,key,seed,'A','restore_replay',False)
    restore_error=float((a-b).abs().max());assert restore_error==0
    assert all(torch.equal(opt.state_dict()['state'][0][k].cpu(),saved['optimizer']['state'][0][k]) for k in ('step','exp_avg','exp_avg_sq'));path.unlink()
    # Formula edge cases: positive-neighbor spacing, coincident centers and zeros.
    toy=dict(means=torch.tensor([[0.,0,0],[0,0,0],[2,0,0],[5,0,0]],device='cuda'),has_content=torch.ones(4,dtype=torch.bool,device='cuda'))
    _,spacing=center_kernel(toy);toy['means'].zero_();ones,coincident=center_kernel(toy);assert coincident['all_valid_centers_coincident'] and torch.equal(ones,torch.ones_like(ones));assert spacing['nearest_positive_distance_median']==2
    tw=torch.tensor([[1.,0,0],[.5,.5,0],[0,0,0]],device='cuda');tn=normalize(tw@tw.T);tk=tn-tn.mean(-1,keepdim=True)
    assert tn[2].count_nonzero()==0 and tn[:,2].count_nonzero()==0 and tk[2].count_nonzero()==0 and tk.mean(-1).abs().max()<1e-6
    assert torch.nn.functional.pad(tk,(5,0,5,0))[:5].count_nonzero()==0
    d.write(d.OUT/'preflight.json',dict(complete=True,seed=seed,key=key,zero_gamma_depth_max_abs_vs_original=zeros,zero_gamma_metric_depth_max_abs_vs_saved_B=zeros_vs_saved,common_content_view_max_abs=viewdiff,A_actual_kernel_FP64_max_abs=definition_error,zero_gamma_materiality='Exact old route comparison and savedB metric-depth differences recorded; no score-based tuning.',gamma_gradients=gradient.cpu().tolist(),gradient_norm=float(norm),only_gamma_trainable=24,frozen_parameters_unchanged=True,full_gamma_Adam_RNG_restore_max_abs=restore_error,positive_distance_example=spacing,all_coincident_example=coincident,pilot_updates_discarded=1,native_calls=7,source_state_read_only=True,zero_support_query_rows_preserved=True,zero_support_key_note='N zero columns become negative row-mean offsets after prescribed centering; all patch keys retained. Special token increments remain exactly0.'))
    print(dict(preflight_complete=True,zero_gamma=zeros,restore_max_abs=restore_error),flush=True)

def fit(model,arm,seed):
    mods=base(model,seed);address=AddressBias().cuda();opt=torch.optim.Adam(address.parameters(),lr=.01,weight_decay=0);order=d.schedule(seed);folder=d.OUT/'fits'/f'{arm}_{seed}';folder.mkdir(exist_ok=True);first=0
    if d.final(arm,seed).exists():restore(d.load(d.final(arm,seed)),address);return mods,address
    if d.latest(arm,seed).exists():
        st=d.load(d.latest(arm,seed));assert st['schedule_sha256']==d.schedule_sha(seed) and st['arm']==arm and st['seed']==seed;restore(st,address,opt);first=st['step']
    frozen=[p for module in (model,*mods) for p in module.parameters()];versions=[p._version for p in frozen]
    d.write(folder/'identity.json',dict(arm=arm,seed=seed,science_sha256=d.SCIENCE_SHA,schedule_sha256=d.schedule_sha(seed),base_checkpoint_path=str(d.parent.final('C_P',seed)),base_checkpoint_sha256=d.read(d.parent.OUT/'fits'/f'C_P_{seed}'/'complete.json')['sha256'],learned_parameters=24,frozen_parameters=sum(p.numel() for p in frozen),initial_gamma=[0.]*24,requires_grad_only_gamma=all(not p.requires_grad for p in frozen)))
    d.journal(folder/'recoveries.jsonl',dict(utc=d.now(),step=first,pid=os.getpid(),Adam_RNG_sampler_loaded=bool(first)))
    log=folder/'training.jsonl'
    if log.exists():assert [d.json.loads(x)['step'] for x in log.read_text().splitlines()]==list(range(1,first+1)),'Unexpected updates beyond exact durable state: preserve and hand back, never silently repeat'
    torch.cuda.reset_peak_memory_stats();begin=time.monotonic()
    for step in range(first,1000):
        opt.zero_grad(set_to_none=True);value=0.
        for key in order[2*step:2*step+2]:
            pred,s=predict(model,mods,address,key,seed,arm,'training');loss=objective(pred,s['query']['target'],s['query']['valid']);assert torch.isfinite(loss);(loss/2).backward();value+=float(loss.detach())/2;del pred,s,loss
        norm=torch.nn.utils.clip_grad_norm_(address.parameters(),5.);assert torch.isfinite(norm);d.charge(WORKER,'formal',attempted_updates=1,arm=arm,seed=seed,step=step+1);opt.step()
        d.journal(log,dict(step=step+1,loss=value,gradient_norm=float(norm),samples=order[2*step:2*step+2],elapsed_seconds=time.monotonic()-begin))
        if (step+1)%100==0 or STOP or step==999:
            assert versions==[p._version for p in frozen] and all(p.grad is None for p in frozen);st=d.cpu(state(address,opt,step+1,arm,seed));path=d.final(arm,seed) if step==999 else d.latest(arm,seed);d.save(path,st)
            d.write(folder/'progress.json',dict(step=step+1,loss=value,seconds=time.monotonic()-begin,peak_bytes=torch.cuda.max_memory_allocated(),gamma=address.gamma.detach().cpu().tolist()));print(dict(arm=arm,seed=seed,step=step+1,loss=value,seconds=time.monotonic()-begin),flush=True)
            if step==999:
                d.write(folder/'complete.json',dict(complete=True,steps=1000,path=str(path),sha256=d.sha(path),frozen_parameter_versions_unchanged=True,peak_bytes=torch.cuda.max_memory_allocated()));d.latest(arm,seed).unlink(missing_ok=True)
            if STOP:raise SystemExit(75)
    del opt;d.view.cache_clear();d.sample.cache_clear();torch.cuda.empty_cache();return mods,address

@torch.no_grad()
def evaluate(model,mods,address,arm,seed):
    for meta in d.metas():
        rid=meta['id'];mp=d.OUT/'evaluation'/f'{arm}_{seed}_{rid}.json'
        if mp.exists():assert d.read(mp)['complete'];continue
        keys=[];preds=[];alphas=[];inputs=[]
        for q in d.parent.old.EVAL_QUERIES:
            v=d.view(seed,rid,q);alphas.append(v['alpha']);inputs.append(d.read(marker(seed,rid,q)))
            for c in d.parent.old.CONDITIONS:
                key=(rid,q,c);pred,s=predict(model,mods,address,key,seed,arm,'evaluation',False);pred=metric_depth(pred).float().cpu();assert torch.isfinite(pred).all() and (pred>0).all();keys.append(key);preds.append(pred);del pred,s
        path=d.predpath(arm,seed,rid);d.save(path,dict(keys=keys,depth_metric=torch.stack(preds),query_order=list(d.parent.old.EVAL_QUERIES),alpha_patch=torch.stack(alphas)))
        d.write(mp,dict(complete=True,id=rid,place=meta['place'],arm=arm,seed=seed,predictions=16,path=str(path),sha256=d.sha(path),field_sha256=inputs[0]['field_sha256'],prepared_view_identities=[dict(path=v['path'],sha256=v['sha256'],query=v['query'],query_sha256=v['query_sha256']) for v in inputs],query_writer_calls=0,source_state_read_only=True));print(dict(evaluated=rid,arm=arm,seed=seed),flush=True);d.view.cache_clear();d.sample.cache_clear()
        if STOP:raise SystemExit(75)

def main():
    global WORKER
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','preflight','fit']);p.add_argument('--seed',type=int,choices=d.SEEDS);p.add_argument('--arm',choices=d.ARMS);p.add_argument('--worker',required=True);a=p.parse_args();WORKER=a.worker
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.85);assert torch.cuda.is_available() and d.sha(d.SCIENCE)==d.SCIENCE_SHA and not d.resource(True)['stop_reasons']
    claim=d.OUT/'claims'/f'{a.phase}_{a.arm}_{a.seed}.lock'
    with claim.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if a.phase=='prepare':prepare(a.seed)
        else:
            model=model_load(d.plan());print(dict(model_loaded=True,phase=a.phase),flush=True)
            if a.phase=='preflight':preflight(model)
            else:
                assert d.read(d.OUT/'preflight.json')['complete'];mods,address=fit(model,a.arm,a.seed);evaluate(model,mods,address,a.arm,a.seed)
    print(dict(process_complete=True,phase=a.phase),flush=True)
if __name__=='__main__':main()
