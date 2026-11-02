"""R55: fixed training-budget extension of unchanged existing model classes."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import gc
import hashlib
import math
import time
import numpy as np
import torch
from experiments.round55_data import ROOT,OUT,read,write,sha,load,pixels
from models.round55_budget_diagnostic import make_model,ROLES,GROUPS,INPUT_FIELDS
from experiments import round52_train as P52,round53_train as P53,round54_train as P54
from experiments.round55_optimization import fit_one,learning_rate,sampling

SEEDS=(11,22,33)
CONFIG=dict(roles=ROLES,seeds=SEEDS,steps=5000,prefix_steps=1000,batch=8,lr=.001,
 weight_decay=.0001,validation_every=100,extension_lr=.0002,primary='ray_gaussian',
 primary_control='ray_point',primary_metric='epe_soft',new_fits=24,reused_references=0,
 confirmation_access=False)
SOURCES=tuple(dict.fromkeys(('ROUND55_BUDGET_DIAGNOSTIC_PROTOCOL.md',
 'models/round55_budget_diagnostic.py','experiments/round55_data.py',
 'experiments/round55_train.py','experiments/round55_optimization.py',
 'experiments/round55_queue.py','tests/test_round55_budget_diagnostic.py',
 'experiments/round53_train.py','experiments/round54_train.py')+P53.SOURCES+P54.SOURCES))
METRICS=P54.METRICS
ORIGINS={role:(P53.OUT if role.startswith('ray_') else P54.OUT if role in ('camera_offset','camera_plane') else P52.OUT) for role in ROLES}


def check(out=OUT):
    lock=read(out/'lock.json')
    if sha(out/'lock.json')!=(out/'lock.sha256').read_text().strip():raise ValueError('lock changed')
    for path,digest in lock['sources'].items():
        if sha(ROOT/path)!=digest:raise ValueError('scientific source changed '+path)
    for name,key in (('data/manifest.json','data_manifest_sha256'),('prefix_manifest.json','prefix_manifest_sha256')):
        if sha(out/name)!=lock[key]:raise ValueError('manifest changed '+name)
    return lock


def freeze():
    if (OUT/'lock.json').exists():raise FileExistsError('already frozen')
    for path in SOURCES:
        dest=OUT/'source'/path;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/path).read_bytes())
    write(OUT/'lock.json',dict(config=CONFIG,sources={p:sha(ROOT/p) for p in SOURCES},
        data_manifest_sha256=sha(OUT/'data/manifest.json'),prefix_manifest_sha256=sha(OUT/'prefix_manifest.json'),
        software=dict(torch=torch.__version__,numpy=np.__version__),preflight=read(OUT/'preflight.json'),confirmation_access=False))
    (OUT/'lock.sha256').write_text(sha(OUT/'lock.json')+'\n')
    print('FROZEN',sha(OUT/'lock.json'),flush=True)


def setup(gpu=None):
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if gpu is None:return torch.device('cpu')
    torch.cuda.set_device(gpu)
    return torch.device('cuda',gpu)


def tensor_hash(state):
    h=hashlib.sha256()
    for key,value in sorted(state.items()):
        h.update(key.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def common(model):return {k:v for k,v in model.state_dict().items() if not k.startswith('field.')}


def subset(data,index):return {k:v[index] for k,v in data.items()}


def inputs(data):return {k:data[k] for k in INPUT_FIELDS}


def query_ce(logits,data):
    lp=logits.log_softmax(-1).gather(-1,data['key_indices'].clamp_min(0))
    return -(lp*data['key_weights']).sum(-1)


def objective(logits,data):return query_ce(logits,data)[data['gt_valid']].mean()


@torch.no_grad()
def validation(model,data):
    model.eval();total=0.;count=0
    for start in range(0,len(data['q']),CONFIG['batch']):
        d=subset(data,slice(start,start+CONFIG['batch']));ce=query_ce(model(inputs(d)),d)
        total+=float(ce[d['gt_valid']].sum());count+=int(d['gt_valid'].sum())
    return total/count


def smoke(gpu):
    check();device=setup(gpu);fit,_=load(OUT,'fit',device);d=subset(fit,slice(0,8));records=[]
    for role in ROLES:
        torch.manual_seed(11);model=make_model(role).to(device);opt=torch.optim.AdamW(model.parameters(),lr=.001)
        times=[]
        for i in range(5):
            torch.cuda.synchronize(device);start=time.monotonic();opt.zero_grad(set_to_none=True)
            loss=objective(model(inputs(d)),d);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True)
            if not torch.isfinite(loss):raise ValueError('smoke loss')
            opt.step();torch.cuda.synchronize(device)
            if i>=2:times.append(time.monotonic()-start)
        r=dict(role=role,parameters=sum(p.numel() for p in model.parameters()),
            pe_parameters=sum(p.numel() for n,p in model.named_parameters() if n.startswith('field.')),
            median_step_seconds=float(np.median(times)),last_loss=float(loss.detach()))
        records.append(r);print('SMOKE',r,flush=True);del model,opt;gc.collect();torch.cuda.empty_cache()
    write(OUT/'smoke.json',dict(gpu=gpu,device=torch.cuda.get_device_name(gpu),records=records))


def seal():
    check();records=[]
    for role in ROLES:
        for seed in SEEDS:
            path=OUT/'runs'/role/str(seed);r=read(path/'run.json')
            if sha(path/'model.pt')!=r['model_sha256']:raise ValueError('checkpoint changed')
            if r['shared_init_sha256']==r['shared_final_sha256']:raise ValueError('no shared learning')
            if r.get('reused'):
                origin=r['origin'];original=read(origin['run_path'])
                if sha(origin['run_path'])!=origin['run_sha256'] or sha(origin['checkpoint_path'])!=original['model_sha256']:raise ValueError('reused origin changed')
            records.append(r)
    for seed in SEEDS:
        rows=[r for r in records if r['seed']==seed]
        for key in ('shared_init_sha256','sampling_sha256'):
            if len({r[key] for r in rows})!=1:raise ValueError('matched control identity '+key)
    write(OUT/'selection.json',dict(records=records,lock_sha256=sha(OUT/'lock.json'),time=time.time(),confirmation_access=False))


def metric_rows(pred,data,ids):
    grid=pixels();valid=data['gt_valid'].cpu().numpy();gt=data['pixcontinuous'].cpu().numpy()
    err=np.linalg.norm(grid[pred['argmax']]-gt,axis=-1)
    soft=np.linalg.norm(pred['soft_xy']-gt,axis=-1)
    ce=-(pred['target_logp']*data['key_weights'].cpu().numpy()).sum(-1)
    values=dict(ce=ce,pck4=err<=4,pck8=err<=8,auc8=np.clip(1-err/8,0,1),
        epe_argmax=err,epe_soft=soft,tail16=err>16,soft_tail16=soft>16)
    rows=[]
    for i,identity in enumerate(ids):
        mask=valid[i]
        rows.append(dict(identity,n_valid=int(mask.sum()),frame_gap=identity['k_index']-identity['q_index'],
            **{k:float(v[i][mask].sum(dtype=np.float64)) for k,v in values.items()}))
    tails={}
    for scene in sorted({x['scene'] for x in ids}):
        idx=np.array([i for i,x in enumerate(ids) if x['scene']==scene]);m=valid[idx]
        tails[scene]=dict(epe_p95=float(np.quantile(err[idx][m],.95)),soft_p95=float(np.quantile(soft[idx][m],.95)))
    return rows,tails


def macro(rows):
    scene=[]
    for name in sorted({r['scene'] for r in rows}):
        selected=[r for r in rows if r['scene']==name];n=sum(r['n_valid'] for r in selected)
        if n:scene.append({k:sum(r[k] for r in selected)/n for k in METRICS})
    return {k:float(np.mean([r[k] for r in scene])) for k in METRICS}


@torch.no_grad()
def evaluate(gpu,seed):
    check();selection=read(OUT/'selection.json');device=setup(gpu)
    data,ids=load(OUT,'old_eval',device);grid=torch.from_numpy(pixels()).to(device)
    permutations=[]
    for i in range(len(ids)):
        rng=np.random.default_rng(510000+i)
        permutations.append(np.stack([rng.permutation(432),rng.permutation(432)]))
    permutations=torch.from_numpy(np.stack(permutations)).to(device)
    for role in ROLES:
        folder=OUT/'runs'/role/str(seed);r=read(folder/'run.json')
        selected=next(x for x in selection['records'] if x['role']==role and x['seed']==seed)
        if r!=selected or sha(folder/'model.pt')!=r['model_sha256']:raise ValueError('selection changed')
        model=make_model(role).to(device);model.load_state_dict(torch.load(folder/'model.pt',map_location=device,weights_only=True));model.eval()
        for variant in ('baseline','rgb_permuted'):
            destination=OUT/'evaluation'/role/f'{seed}_{variant}.json'
            if destination.exists():raise FileExistsError('retain evaluation')
            prediction={k:[] for k in ('argmax','soft_xy','target_logp')}
            for start in range(0,len(ids),CONFIG['batch']):
                d=subset(data,slice(start,start+CONFIG['batch']));x=inputs(d)
                if variant=='rgb_permuted':
                    p=permutations[start:start+len(d['q'])]
                    x=dict(x,q=x['q'].gather(1,p[:,0,:,None].expand_as(x['q'])),
                           k=x['k'].gather(1,p[:,1,:,None].expand_as(x['k'])))
                logits=model(x)
                if not torch.isfinite(logits).all():raise ValueError('evaluation nonfinite')
                prediction['argmax'].append(logits.argmax(-1).cpu().numpy().astype('int16'))
                prediction['soft_xy'].append((logits.softmax(-1)@grid).cpu().numpy())
                prediction['target_logp'].append(logits.log_softmax(-1).gather(-1,d['key_indices'].clamp_min(0)).cpu().numpy())
            pred={k:np.concatenate(v) for k,v in prediction.items()}
            destination.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(destination.with_suffix('.npz'),**pred)
            rows,tails=metric_rows(pred,data,ids)
            write(destination,dict(role=role,seed=seed,variant=variant,rows=rows,metrics=macro(rows),scene_tails=tails,
                checkpoint_sha256=r['model_sha256'],prediction_sha256=sha(destination.with_suffix('.npz'))))
            print('EVAL',role,seed,variant,macro(rows),flush=True)
        del model;gc.collect();torch.cuda.empty_cache()
    check()


def report():
    records=[]
    for role in ROLES:
        for seed in SEEDS:
            for variant in ('baseline','rgb_permuted'):
                r=read(OUT/'evaluation'/role/f'{seed}_{variant}.json')
                records.append(r)
    aggregate={}
    for role in ROLES:
        aggregate[role]={}
        for variant in ('baseline','rgb_permuted'):
            group=[r for r in records if r['role']==role and r['variant']==variant]
            aggregate[role][variant]={k:float(np.mean([r['metrics'][k] for r in group])) for k in METRICS}
    pairs={}
    for role in ('ray_gaussian','camera_plane','camera_gs'):
        pairs[role]={}
        for control in ROLES:
            if role==control:continue
            deltas={}
            for metric in METRICS:
                delta=[]
                for seed in SEEDS:
                    a=next(r for r in records if r['role']==role and r['seed']==seed and r['variant']=='baseline')
                    b=next(r for r in records if r['role']==control and r['seed']==seed and r['variant']=='baseline')
                    delta.append(a['metrics'][metric]-b['metrics'][metric])
                lower=metric in ('ce','epe_argmax','epe_soft','tail16','soft_tail16')
                deltas[metric]=dict(mean=float(np.mean(delta)),per_seed=delta,wins=sum(d<0 if lower else d>0 for d in delta))
            pairs[role][control]=deltas
    strata={}
    for role in ROLES:
        group=[r for r in records if r['role']==role and r['variant']=='baseline']
        strata[role]={}
        for gap in (15,60):
            values=[macro([x for x in r['rows'] if x['frame_gap']==gap]) for r in group]
            strata[role][str(gap)]={k:float(np.mean([v[k] for v in values])) for k in METRICS}
    write(OUT/'summary.json',dict(aggregate=aggregate,paired=pairs,frame_gap=strata,
        selection_sha256=sha(OUT/'selection.json'),evaluation_groups=len(records),confirmation_access=False,full_goal_achieved=False))
    print('SUMMARY', {role:aggregate[role]['baseline'] for role in ROLES},flush=True)



def preflight():
    setup();fit,_=load(OUT,'fit',torch.device('cpu'));sample=subset(fit,slice(0,1))
    for previous in (P52,P53,P54):
        previous.check()
        if not read(previous.OUT/'completion.json')['round_complete']:raise ValueError('previous round incomplete')
        if read(previous.OUT/'data/manifest.json')['entries']!=read(OUT/'data/manifest.json')['entries']:raise ValueError('data changed')
    if (OUT/'prefix_manifest.json').exists():raise FileExistsError('retain prefix manifest')
    entries=[];gradients=[]
    for role in ROLES:
        for seed in SEEDS:
            origin=ORIGINS[role];path=origin/'runs'/role/str(seed);r=read(path/'run.json')
            if r!=next(x for x in read(origin/'selection.json')['records'] if (x['role'],x['seed'])==(role,seed)):raise ValueError('origin selection changed')
            if sha(path/'model.pt')!=r['model_sha256']:raise ValueError('origin model changed')
            torch.manual_seed(seed);model=make_model(role)
            if tensor_hash(common(model))!=r['shared_init_sha256']:raise ValueError('initialization changed')
            ix=sampling(seed,len(fit['q']),CONFIG['steps'],CONFIG['batch'])
            if hashlib.sha256(ix[:1000].tobytes()).hexdigest()!=r['sampling_sha256']:raise ValueError('batch prefix changed')
            if seed==11:
                loss=objective(model(inputs(sample)),sample);loss.backward()
                grads=[p.grad for p in model.parameters() if p.grad is not None]
                if not torch.isfinite(loss) or not all(torch.isfinite(g).all() for g in grads):raise ValueError('nonfinite gradient')
                field=[float(p.grad.abs().max()) for n,p in model.named_parameters() if n.startswith('field.') and p.grad is not None]
                gradients.append(dict(role=role,loss=float(loss.detach()),field_gradient_maxima=field))
            state=torch.load(path/'model.pt',map_location='cpu',weights_only=True);model.load_state_dict(state);model.eval()
            previous=P53 if role.startswith('ray_') else P54 if role in ('camera_offset','camera_plane') else P52
            factory=previous.CameraFrameMatcher if previous is P52 else previous.make_model
            old=factory(role);old.load_state_dict(state);old.eval()
            with torch.no_grad():difference=float((model(inputs(sample))-old(inputs(sample))).abs().max())
            if difference!=0:raise ValueError('factory changed selected function')
            entries.append(dict(role=role,seed=seed,run_path=str(path/'run.json'),run_sha256=sha(path/'run.json'),
                checkpoint_path=str(path/'model.pt'),checkpoint_sha256=sha(path/'model.pt'),
                origin_lock_path=str(origin/'lock.json'),origin_lock_sha256=sha(origin/'lock.json'),
                selected_function_max_difference=difference,shared_init_sha256=r['shared_init_sha256'],prefix_sampling_sha256=r['sampling_sha256']))
    write(OUT/'prefix_manifest.json',dict(entries=entries,confirmation_access=False))
    write(OUT/'preflight.json',dict(gradients=gradients,original_functions_checked=len(entries),confirmation_access=False))
    print('PREFLIGHT',len(gradients),'real-input gradients;',len(entries),'prefix identities',flush=True)


def train(gpu,seed,group):
    import sys
    check();device=setup(gpu);fit,_=load(OUT,'fit',device);val,_=load(OUT,'val',device)
    order=np.random.default_rng(seed+81000+group).permutation(len(GROUPS[group]))
    for i in order:fit_one(sys.modules[__name__],GROUPS[group][i],seed,fit,val,device)
    check();print('WORKER COMPLETE',seed,group,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['preflight','freeze','smoke','train','seal','eval','report'])
    p.add_argument('--gpu',type=int);p.add_argument('--seed',type=int);p.add_argument('--group',type=int)
    a=p.parse_args()
    if a.action=='preflight':preflight()
    elif a.action=='freeze':freeze()
    elif a.action=='smoke':smoke(a.gpu)
    elif a.action=='train':train(a.gpu,a.seed,a.group)
    elif a.action=='seal':seal()
    elif a.action=='eval':evaluate(a.gpu,a.seed)
    else:report()
