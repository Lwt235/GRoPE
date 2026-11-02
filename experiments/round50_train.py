"""Fixed-budget local-frame development experiment. Never reads old test sets."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
import argparse
import gc
import hashlib
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from experiments.round50_data import ROOT, OUT, CONDITIONS, centers, sha, write
from models.round50_local_frame import LocalFrameMatcher, ROLES

SEEDS = (11, 22, 33)
REGIMES = ('oracle', 'noisy')
CONFIG = dict(seeds=SEEDS, regimes=REGIMES, roles=ROLES, steps=1000, batch=16,
              lr=.001, weight_decay=.0001, validation_every=100,
              primary='gs_frame', primary_regime='oracle_oracle', scientific_scope='development_only')
SOURCES = ('ROUND50_LOCAL_FRAME_PROTOCOL.md', 'experiments/round50_data.py',
    'experiments/round50_train.py', 'experiments/round50_queue.py',
    'models/round50_local_frame.py', 'tests/test_round50_local_frame.py',
    'models/round5_cross_attention.py', 'models/round47_conditional_gaussian.py',
    'models/geometry.py', 'models/gaussian_pe.py', 'models/gaussian_spectral_pe.py')


def read(path): return json.loads(Path(path).read_text())


def setup(gpu=None):
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if gpu is None: return torch.device('cpu')
    torch.cuda.set_device(gpu)
    return torch.device('cuda', gpu)


def tensor_hash(state):
    h = hashlib.sha256()
    for k, v in sorted(state.items()):
        h.update(k.encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def common_state(model):
    return {k:v for k,v in model.state_dict().items() if not k.startswith('field.')}


def check_lock(out):
    lock = read(out/'lock.json')
    if sha(out/'lock.json') != (out/'lock.sha256').read_text().strip(): raise ValueError('lock changed')
    for p, h in lock['sources'].items():
        if sha(ROOT/p) != h: raise ValueError('scientific source changed: '+p)
    if sha(out/'data/manifest.json') != lock['data_manifest_sha256']: raise ValueError('data manifest changed')
    return lock


def load(out, split, condition, device):
    e = next(e for e in read(out/'data/manifest.json')['entries'] if e['split']==split and e['condition']==condition)
    if sha(e['path']) != e['sha256']: raise ValueError('data changed')
    with np.load(e['path'], allow_pickle=False) as z:
        return {k:torch.from_numpy(z[k]).to(device) for k in z.files}


def subset(data, index): return {k:v[index] for k,v in data.items()}


def rows_loss(logits, data, prefix):
    lp = logits.log_softmax(-1).gather(-1, data[prefix+'indices'])
    return -(lp*data[prefix+'weights']).sum(-1)


def objective(logits, data):
    q, k = rows_loss(logits, data, 'q'), rows_loss(logits.transpose(-1,-2), data, 'k')
    return (q[data['qvalid']].sum()+k[data['kvalid']].sum())/(data['qvalid'].sum()+data['kvalid'].sum())


@torch.no_grad()
def validation(model, data, regime):
    model.eval(); total = count = 0
    for start in range(0, len(data['qrgb']), 16):
        batch=subset(data,slice(start,start+16));logits=model(batch,regime)
        c=int(batch['qvalid'].sum()+batch['kvalid'].sum())
        total += float(objective(logits,batch))*c; count += c
    return total/count


def freeze(out):
    if (out/'lock.json').exists(): raise FileExistsError('already frozen')
    sources={p:sha(ROOT/p) for p in SOURCES}
    for p in SOURCES:
        dst=out/'source'/p;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes((ROOT/p).read_bytes())
    write(out/'lock.json',dict(config=CONFIG,sources=sources,data_manifest_sha256=sha(out/'data/manifest.json'),
        software=dict(torch=torch.__version__,numpy=np.__version__),
        confirmation_access=False,preflight='four bounded unittest cases passed before freeze',
        source_paper='https://arxiv.org/html/2610.28284v1'))
    (out/'lock.sha256').write_text(sha(out/'lock.json')+'\n')
    print('FROZEN',sha(out/'lock.json'),flush=True)


def smoke(out,gpu):
    device=setup(gpu);data=subset(load(out,'fit','base',device),slice(0,16));records=[]
    for role in ROLES:
        torch.manual_seed(11);m=LocalFrameMatcher(role).to(device);opt=torch.optim.AdamW(m.parameters(),lr=.001)
        times=[]
        for i in range(7):
            torch.cuda.synchronize(device);start=time.monotonic();opt.zero_grad(set_to_none=True)
            loss=objective(m(data),data);loss.backward()
            if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in m.parameters()):
                raise ValueError('nonfinite smoke '+role)
            torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt.step();torch.cuda.synchronize(device)
            if i>=2:times.append(time.monotonic()-start)
        rec=dict(role=role,parameters=sum(p.numel() for p in m.parameters()),
            pe_parameters=sum(p.numel() for n,p in m.named_parameters() if n.startswith('field.')),
            median_step_seconds=float(np.median(times)),last_smoke_loss=float(loss))
        records.append(rec);print('SMOKE',rec,flush=True);del m,opt;gc.collect();torch.cuda.empty_cache()
    write(out/'smoke.json',dict(gpu=gpu,device=torch.cuda.get_device_name(gpu),records=records))


def fit_one(out,role,seed,regime,fit,val,indices,device):
    folder=out/'runs'/regime/role/str(seed)
    if folder.exists():raise FileExistsError('retain previous run '+str(folder))
    folder.mkdir(parents=True)
    torch.manual_seed(seed);model=LocalFrameMatcher(role).to(device)
    initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    shared_init=tensor_hash(common_state(model));opt=torch.optim.AdamW(model.parameters(),lr=CONFIG['lr'],weight_decay=CONFIG['weight_decay'])
    best=math.inf;best_state=None;best_step=None;history=[]
    torch.cuda.reset_peak_memory_stats(device);torch.cuda.synchronize(device);begin=time.monotonic()
    for step in range(CONFIG['steps']+1):
        if step%CONFIG['validation_every']==0:
            value=validation(model,val,regime)
            if not math.isfinite(value):raise ValueError('nonfinite validation')
            if value<best:
                best=value;best_step=step;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            record=dict(step=step,val_ce=value,best_step=best_step,elapsed=time.monotonic()-begin)
            history.append(record)
            with open(folder/'history.jsonl','a') as f:f.write(json.dumps(record)+'\n')
            print('TRAIN',seed,regime,role,step,'val',round(value,6),'best',best_step,flush=True)
        if step==CONFIG['steps']:break
        model.train();batch=subset(fit,indices[step])
        lr=CONFIG['lr']*(.2+.8*(1+math.cos(math.pi*step/CONFIG['steps']))/2)
        for group in opt.param_groups:group['lr']=lr
        opt.zero_grad(set_to_none=True);loss=objective(model(batch,regime),batch);loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True)
        if not torch.isfinite(loss):raise ValueError('nonfinite training')
        opt.step()
    torch.cuda.synchronize(device);elapsed=time.monotonic()-begin
    torch.save(best_state,folder/'model.pt')
    changed={k:float((v-initial[k]).abs().max()) for k,v in best_state.items() if k.startswith('field.')}
    record=dict(role=role,seed=seed,regime=regime,steps=CONFIG['steps'],best_step=best_step,val_ce=best,
        shared_init_sha256=shared_init,shared_final_sha256=tensor_hash({k:v for k,v in best_state.items() if not k.startswith('field.')}),
        last_shared_sha256=tensor_hash(common_state(model)),
        sampling_sha256=hashlib.sha256(indices.cpu().numpy().tobytes()).hexdigest(),
        model_sha256=sha(folder/'model.pt'),lock_sha256=sha(out/'lock.json'),field_max_changes=changed,
        parameters=sum(p.numel() for p in model.parameters()),pe_parameters=sum(p.numel() for n,p in model.named_parameters() if n.startswith('field.')),
        training_selection_seconds=elapsed,peak_allocated_bytes=torch.cuda.max_memory_allocated(device),history=history)
    write(folder/'run.json',record)
    del model,opt;gc.collect();torch.cuda.empty_cache()
    return record


def train(out,gpu,seed):
    check_lock(out);device=setup(gpu);fit=load(out,'fit','base',device);val=load(out,'val','base',device)
    rng=np.random.default_rng(seed+50000);ix=rng.integers(0,len(fit['qrgb']),size=(CONFIG['steps'],CONFIG['batch']),dtype=np.int64)
    indices=torch.from_numpy(ix).to(device)
    tasks=[(regime,role) for regime in REGIMES for role in ROLES]
    order=np.random.default_rng(seed+80000).permutation(len(tasks))
    for i in order:
        regime,role=tasks[i];fit_one(out,role,seed,regime,fit,val,indices,device)
    check_lock(out);print('SEED COMPLETE',seed,flush=True)


def seal(out):
    check_lock(out);records=[]
    for seed in SEEDS:
        current=[]
        for regime in REGIMES:
            for role in ROLES:
                p=out/'runs'/regime/role/str(seed);r=read(p/'run.json')
                if r['model_sha256']!=sha(p/'model.pt') or r['steps']!=1000 or r['lock_sha256']!=sha(out/'lock.json'):
                    raise ValueError('run identity')
                if r['shared_init_sha256']==r['last_shared_sha256']:raise ValueError('training did not change common weights')
                if r['best_step']!=min(r['history'],key=lambda x:x['val_ce'])['step']:raise ValueError('selection')
                current.append(r)
        if len({r['shared_init_sha256'] for r in current})!=1 or len({r['sampling_sha256'] for r in current})!=1:
            raise ValueError('paired initialization/sampling')
        records.extend(current)
    write(out/'sealed_models.json',dict(records=records,all_sealed_before_dev=True,confirmation_access=False))
    print('SEALED',len(records),flush=True)


@torch.no_grad()
def evaluate(out,gpu,seed):
    check_lock(out)
    if len(read(out/'sealed_models.json')['records'])!=48:raise ValueError('seal all before development')
    device=setup(gpu)
    for condition in CONDITIONS:
        data=load(out,'dev',condition,device)
        for regime in REGIMES:
            for role in ROLES:
                path=out/'runs'/regime/role/str(seed)
                model=LocalFrameMatcher(role).to(device)
                model.load_state_dict(torch.load(path/'model.pt',map_location=device,weights_only=True));model.eval()
                for input_regime in REGIMES:
                    destination=out/'predictions'/regime/role/str(seed)/f'{condition}_{input_regime}.npz'
                    if destination.exists():raise FileExistsError('retain predictions')
                    rows={k:[] for k in ('argmax','selected_logp','soft_xy')}
                    for start in range(0,len(data['qrgb']),16):
                        batch=subset(data,slice(start,start+16));logits=model(batch,input_regime)
                        lp=logits.log_softmax(-1)
                        xy=torch.as_tensor(centers(),dtype=logits.dtype,device=device)
                        values=dict(argmax=logits.argmax(-1),selected_logp=lp.gather(-1,batch['qindices']),soft_xy=lp.exp()@xy)
                        for k,v in values.items():rows[k].append(v.cpu().numpy())
                    destination.parent.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(destination,**{k:np.concatenate(v) for k,v in rows.items()})
                del model
        del data;torch.cuda.empty_cache();print('EVAL',seed,condition,flush=True)
    print('EVAL COMPLETE',seed,flush=True)


def metrics_from_predictions(pred,data):
    valid=data['qvalid'];error=np.linalg.norm(centers()[pred['argmax']]-data['qtarget'],axis=-1)
    soft_error=np.linalg.norm(pred['soft_xy']-data['qtarget'],axis=-1)
    ce=-(pred['selected_logp'].astype(np.float64)*data['qweights']).sum(-1)
    count=valid.sum(-1)
    if np.any(count==0):raise ValueError('empty eligible scene; report rather than drop')
    rows=dict(auc=np.clip(1-error/8,0,1),pck4=error<=4,pck8=error<=8,epe=error,soft_epe=soft_error,ce=ce)
    per_scene={k:(v*valid).sum(-1)/count for k,v in rows.items()}
    result={k:float(v.mean()) for k,v in per_scene.items()}
    result.update(valid=int(valid.sum()),total=int(valid.size),p90_error=float(np.quantile(error[valid],.9)))
    strata=[]
    for lo,hi in ((0,1.3),(1.3,1.8),(1.8,np.inf)):
        mask=valid&(data['qanisotropy']>=lo)&(data['qanisotropy']<hi)
        strata.append(dict(low=lo,high=None if not np.isfinite(hi) else hi,count=int(mask.sum()),
            auc=float(rows['auc'][mask].mean()) if mask.any() else None,
            epe=float(error[mask].mean()) if mask.any() else None))
    result['strata']=strata
    return result,per_scene


def bootstrap_scene(differences):
    rng=np.random.default_rng(20261150)
    index=rng.integers(0,len(differences),(2000,len(differences)))
    return np.quantile(differences[index].mean(-1),[.025,.975]).tolist()


def report(out):
    check_lock(out);all_rows=[];scene_rows={};hashes={}
    for condition in CONDITIONS:
        p=out/'data'/f'dev_{condition}.npz'
        with np.load(p,allow_pickle=False) as z:data={k:z[k] for k in ('qvalid','qtarget','qweights','qanisotropy')}
        for regime in REGIMES:
            for role in ROLES:
                for seed in SEEDS:
                    for inputs in REGIMES:
                        p=out/'predictions'/regime/role/str(seed)/f'{condition}_{inputs}.npz'
                        with np.load(p,allow_pickle=False) as z:pred={k:z[k] for k in z.files}
                        metrics,scenes=metrics_from_predictions(pred,data)
                        # A second scalar count/formula path checks aggregation.
                        good=data['qvalid'];idx=pred['argmax'];x=3.5+8*(idx%8);y=3.5+8*(idx//8)
                        dist=np.sqrt((x-data['qtarget'][...,0])**2+(y-data['qtarget'][...,1])**2)
                        replay=np.mean([np.maximum(0,1-dist[i][good[i]]/8).mean() for i in range(len(good))])
                        if abs(replay-metrics['auc'])>1e-12:raise ValueError('metric replay')
                        key=(regime,role,seed,condition,inputs);scene_rows[key]=scenes
                        all_rows.append(dict(train_regime=regime,role=role,seed=seed,condition=condition,input_regime=inputs,**metrics))
                        hashes[str(p.relative_to(out))]=sha(p)
    tables=[];comparisons=[]
    for regime in REGIMES:
        for inputs in REGIMES:
            for role in ROLES:
                for condition in CONDITIONS:
                    selected=[r for r in all_rows if (r['train_regime'],r['role'],r['condition'],r['input_regime'])==(regime,role,condition,inputs)]
                    tables.append(dict(train_regime=regime,input_regime=inputs,role=role,condition=condition,
                        **{m:float(np.mean([r[m] for r in selected])) for m in ('auc','pck4','pck8','epe','soft_epe','ce','p90_error')}))
            for control in [r for r in ROLES if r!='gs_frame']:
                for conditions in [('base',),CONDITIONS]:
                    effects=[];scene_d=[];pck_d=[]
                    for seed in SEEDS:
                        ds=np.mean([scene_rows[(regime,'gs_frame',seed,c,inputs)]['auc']-scene_rows[(regime,control,seed,c,inputs)]['auc'] for c in conditions],0)
                        dp=np.mean([scene_rows[(regime,'gs_frame',seed,c,inputs)]['pck8']-scene_rows[(regime,control,seed,c,inputs)]['pck8'] for c in conditions],0)
                        effects.append(float(ds.mean()));scene_d.append(ds);pck_d.append(float(dp.mean()))
                    comparisons.append(dict(train_regime=regime,input_regime=inputs,control=control,
                        conditions=list(conditions),auc_delta=float(np.mean(effects)),seed_deltas=effects,
                        positive_seeds=sum(e>0 for e in effects),pck8_delta=float(np.mean(pck_d)),
                        scene_bootstrap_ci=bootstrap_scene(np.mean(scene_d,0)),
                        development_guide=bool(min(effects)>0 and np.mean(effects)>=.01 and np.mean(pck_d)>=-.01)))
    signals={regime:all(r['development_guide'] for r in comparisons if r['train_regime']==regime and r['input_regime']==regime and r['control'] in ('mlp','affine')) for regime in REGIMES}
    write(out/'metrics/summary.json',dict(tables=tables,comparisons=comparisons,signals=signals,all_rows=all_rows,
        full_objective_complete=False,fresh_confirmation=False,paired_scene_bootstrap_scope='scene sampling only; three optimization seeds'))
    write(out/'metrics/verification.json',dict(prediction_files=len(hashes),prediction_sha256=hashes,scalar_metric_replay=True,source_hashes_unchanged=True))
    lines=['# R50 local-frame development results','',
        'Controlled procedural matching; three paired seeds. No real-task confirmation.','',
        'All methods receive the same RGB, center/shape/camera information.','']
    for regime in REGIMES:
        lines += [f'## Train/evaluate with {regime} geometry','',
            '| role | base AUC % | slant | zoom | wide | mean AUC % | base PCK8 % | base EPE |',
            '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
        for role in ROLES:
            rs=[next(t for t in tables if (t['train_regime'],t['input_regime'],t['role'],t['condition'])==(regime,regime,role,c)) for c in CONDITIONS]
            lines.append('| '+role+' | '+' | '.join(f"{r['auc']*100:.3f}" for r in rs)+f" | {np.mean([r['auc'] for r in rs])*100:.3f} | {rs[0]['pck8']*100:.3f} | {rs[0]['epe']:.3f} |")
        lines += ['',f'Fixed development guide versus MLP and affine: {signals[regime]}.','']
    lines += ['## Full-GS paired AUC differences','',
        '| geometry | control | scope | delta pp | positive seeds | scene-bootstrap interval pp |',
        '| --- | --- | --- | ---: | ---: | --- |']
    for r in comparisons:
        if r['train_regime']!=r['input_regime']:continue
        ci=r['scene_bootstrap_ci'];scope='base' if len(r['conditions'])==1 else 'four conditions'
        lines.append(f"| {r['train_regime']} | {r['control']} | {scope} | {100*r['auc_delta']:.3f} | {r['positive_seeds']}/3 | [{100*ci[0]:.3f}, {100*ci[1]:.3f}] |")
    lines += ['', 'Continuous results and all cross-regime/shape-stratum metrics remain in summary.json.',
        'A failed guide is not a theorem of infeasibility. A synthetic pass is not evidence of general real-task utility.','']
    (out/'metrics/report.md').write_text('\n'.join(lines))
    print('REPORT',signals,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','smoke','train','seal','eval','report'])
    p.add_argument('--out',type=Path,default=OUT);p.add_argument('--gpu',type=int,default=0);p.add_argument('--seed',type=int,default=11)
    a=p.parse_args()
    if a.action=='freeze':freeze(a.out)
    elif a.action=='smoke':smoke(a.out,a.gpu)
    elif a.action=='train':train(a.out,a.gpu,a.seed)
    elif a.action=='seal':seal(a.out)
    elif a.action=='eval':evaluate(a.out,a.gpu,a.seed)
    else:report(a.out)
