"""Source-only per-scene actual 1024D feature field fits; no query API."""
import argparse,time,os,signal
import numpy as np
import torch
import torch.nn.functional as F
from experiments import native_feature_field_data as d
from models.gaussian_scene_memory import rasterize_field
STOP=False
def request_stop(*a):
    global STOP
    STOP=True

def current(base,f,rot=None,logs=None):
    valid=base['has_content'];features=base['features'].float().clone();features[valid]=f
    out=dict(base,features=features)
    if rot is not None:
        rotations=base['rotations'].clone();scales=base['scales'].clone();rotations[valid]=F.normalize(rot,dim=-1);scales[valid]=logs.exp();out.update(rotations=rotations,scales=scales)
    return out

def render(field,v):
    K=field['source_intrinsic'][v:v+1].clone();K[:,:2]/=14
    colors=torch.cat((field['features'].float()*field['has_content'][:,None],field['has_content'][:,None].float()),-1)
    r,_,_=rasterize_field(field,field['source_c2w'][v:v+1],K,32,32,colors);r=r[0].reshape(1024,1025);a=r[:,-1:]
    b=torch.where(a>1e-8,r[:,:1024]/a.clamp_min(1e-8),0.)
    return b,a

@torch.no_grad()
def prepare_targets():
    d.BASECACHE.mkdir(parents=True,exist_ok=True);assert d.sha(d.SCIENCE)==d.SCIENCE_SHA;assert not d.resources(admission=True)['stop_reasons']
    from experiments.gaussian_memory_native_run import model_load
    from models.gaussian_memory_coarse_pe import patch_tokens
    todo=[m for m in d.metas() if not d.target_path(m['id']).exists()];m=model_load(d.old.plan()) if todo else None;records=[]
    for meta in d.metas():
        rid=meta['id'];p=d.target_path(rid)
        if not p.exists():
            # Only the four original source RGB tensors enter the encoder.
            source=d.load(source_rgb_path(rid),'cuda');features=[]
            for image in source['rgb']:
                with torch.autocast('cuda',dtype=torch.bfloat16):features.append(patch_tokens(m,image.float()/255))
            d.tensor_save(p,dict(features=torch.cat(features).reshape(4,32,32,1024).cpu()))
        x=d.load(p)['features'];assert x.shape==(4,32,32,1024) and torch.isfinite(x).all()
        records.append(dict(id=rid,path=str(p),bytes=p.stat().st_size,sha256=d.sha(p),new_encoder_samples=4 if meta in todo else 0,source_ids=[0,3,6,9],query_inputs=False))
    d.write(d.BASEOUT/'source_targets.json',dict(complete=True,records=records,new_source_encoder_samples=4*len(todo),inherited_targets=44-len(todo)))
def source_rgb_path(rid):
    from experiments.native_pose_memory_data import source
    return source(rid)

def fit_one(meta):
    rid=meta['id'];mark=d.OUT/'formation'/f'{rid}.json';checkpoint=d.CACHE/'source_checkpoints'/f'{rid}.pt'
    if mark.exists():return
    assert not d.resources(admission=True)['stop_reasons'];base=d.load(d.original_scene(rid),'cuda');target=d.load(d.target_path(rid),'cuda')['features'].reshape(4,1024,1024).float();valid=base['has_content'];f=torch.nn.Parameter(base['features'][valid].float().clone());rot=torch.nn.Parameter(base['rotations'][valid].clone()) if d.ROW=='J' else None;logs=torch.nn.Parameter(base['scales'][valid].log()) if d.ROW=='J' else None
    groups=[dict(params=[f],lr=.01)]
    if rot is not None:groups +=[dict(params=[rot],lr=.001),dict(params=[logs],lr=.02)]
    opt=torch.optim.Adam(groups);first=0;history=[];initial=[];finalstats=[];begin=time.monotonic();prior_seconds=0.;firstgrads=None
    if checkpoint.exists():
        s=d.load(checkpoint);prior_seconds=s.get('elapsed_seconds',0.)
        if not prior_seconds:
            import ast
            for logfile in d.BASEOUT.glob(f'row_{d.ROW}_*.log'):
                for line in logfile.read_text().splitlines():
                    try:entry=ast.literal_eval(line)
                    except (ValueError,SyntaxError):continue
                    if isinstance(entry,dict) and entry.get('source')==rid and entry.get('step')==s['step']:prior_seconds=max(prior_seconds,entry.get('seconds',0.))
        f.data.copy_(s['f'].cuda());first=s['step'];history=s['history'];initial=s['initial'];firstgrads=s['first_gradients']
        if rot is not None:rot.data.copy_(s['rotation'].cuda());logs.data.copy_(s['log_scale'].cuda())
        opt.load_state_dict(s['optimizer']);torch.set_rng_state(s['torch_rng'].cpu());torch.cuda.set_rng_state(s['cuda_rng'].cpu())
    else:
        with torch.no_grad():
            for v in range(4):
                b,a=render(base,v);d.charge('source_initial',raster=1);initial.append(dict(view=v,raw_MSE=float((b-target[v]).square().mean()),zero_fraction=float((a<=1e-8).float().mean())))
    def persist(step):
        d.tensor_save(checkpoint,dict(elapsed_seconds=prior_seconds+time.monotonic()-begin,step=step,f=f.detach().cpu(),rotation=rot.detach().cpu() if rot is not None else None,log_scale=logs.detach().cpu() if logs is not None else None,optimizer=opt.state_dict(),history=history,initial=initial,first_gradients=firstgrads,source_target_path=str(d.target_path(rid)),row=d.ROW,id=rid,science_sha256=d.SCIENCE_SHA,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),budget=d.read(d.OUT/'budget.json')))
    for step in range(first,300):
        opt.zero_grad(set_to_none=True);field=current(base,f,rot,logs);v=step%4;b,a=render(field,v);d.charge('source_training',raster=1,source_samples=1);loss=(b-target[v]).square().mean();assert torch.isfinite(loss);loss.backward()
        if firstgrads is None:
            firstgrads=dict(f=float(f.grad.norm()),rotation=float(rot.grad.norm()) if rot is not None else None,log_scale=float(logs.grad.norm()) if logs is not None else None);assert firstgrads['f']>0
            if rot is not None:assert np.isfinite(firstgrads['rotation']) and np.isfinite(firstgrads['log_scale']) and firstgrads['rotation']>0 and firstgrads['log_scale']>0
        opt.step();d.charge('source_training',source_updates=1)
        if logs is not None:
            with torch.no_grad():logs.clamp_(min=float(torch.log(base['unit']*1e-5)),max=float(torch.log(base['unit']*2)))
        history.append(dict(step=step+1,view=v,raw_MSE=float(loss.detach()),zero_fraction=float((a<=1e-8).float().mean())))
        if (step+1)%100==0 or STOP:
            persist(step+1);r=d.resources();print(dict(row=d.ROW,source=rid,step=step+1,loss=history[-1]['raw_MSE'],seconds=prior_seconds+time.monotonic()-begin),flush=True)
            if STOP or r['stop_reasons']:raise SystemExit(75)
    final=current(base,f,rot,logs)
    # Commit the actual FP16 feature field and FP32 geometry, then evaluate that saved representation.
    final={k:v.detach().cpu() for k,v in final.items()};final['features']=final['features'].half();path=d.old.scene_path(rid);d.tensor_save(path,final);saved=d.load(path,'cuda')
    assert all(torch.equal(saved[n],base[n]) for n in ('means','opacities','colors','has_content','source_c2w','source_intrinsic','unit','source_ids','mass'))
    if d.ROW=='C':assert torch.equal(saved['rotations'],base['rotations']) and torch.equal(saved['scales'],base['scales'])
    with torch.no_grad():
        for v in range(4):
            b,a=render(saved,v);d.charge('source_final',raster=1);delta=(b-target[v]).double();bias=delta.mean(0);finalstats.append(dict(view=v,raw_MSE=float(delta.square().mean()),spatial_MSE=float((delta-bias).square().mean()),bias_squared=float(bias.square().mean()),zero_fraction=float((a<=1e-8).float().mean())))
    d.write(mark,dict(complete=True,row=d.ROW,id=rid,steps=300,seconds=prior_seconds+time.monotonic()-begin,field_path=str(path),field_sha256=d.sha(path),field_bytes=path.stat().st_size,checkpoint_path=str(checkpoint),checkpoint_bytes=checkpoint.stat().st_size,source_target_path=str(d.target_path(rid)),source_target_sha256=d.sha(d.target_path(rid)),initial=initial,final=finalstats,first_gradients=firstgrads,fixed_fields_exact=True,query_inputs=False,final_only=True))
    del base,target,f,rot,logs,opt,field,saved;torch.cuda.empty_cache()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['targets','fit']);p.add_argument('--row',default='C');a=p.parse_args();torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.20);signal.signal(signal.SIGTERM,request_stop);d.context(a.row)
    if a.mode=='targets':prepare_targets()
    else:
        for meta in d.metas():fit_one(meta)
