"""Fixed-budget old-scene NVS training, selection and prediction export."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np
import torch

from models.round70_nvs import CompactNVS, ROLES
from experiments.round70_nvs_data import OUT, ROOT, NVSData
from experiments.round65_common import read, write, sha
from experiments.round65_run import configure, atomic_save, stream_update
from experiments import round65_queue as scheduler

SEEDS = (11,22,33)
CONDITIONS = ('base','extrapolate','target_zoom')
SETTINGS = dict(epochs=60,batch_size=8,eval_batch_size=8,lr=4e-4,final_lr=4e-6,
    warmup_epochs=3,weight_decay=.01,clip=1.,validation_every=3,loss='unclipped RGB MSE',
    seeds=SEEDS,roles=ROLES,conditions=CONDITIONS,maximum_gpu_jobs=3,
    selection='minimum old-val unclipped MSE, earliest on exact tie',
    precision='bf16 autocast, float32 geometry, float64 inverses/contractions/matrix log')
MODULE = 'experiments.round70_nvs_run'


def digest(model):
    h = hashlib.sha256()
    for name,tensor in model.state_dict().items():
        if name.startswith(('relative.','hfield.','geometry_head.')): continue
        h.update(name.encode());h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def freeze():
    if (OUT/'lock.json').exists(): raise ValueError('preserve the existing lock')
    pre = read(OUT/'cuda_cost_preflight.json')
    assert pre['passed'] and pre['source_sha256']==sha(ROOT/'models/round70_nvs.py')
    unit = read(OUT/'units.json'); assert unit['passed']
    files = [ROOT/'docs/round70_nvs_plan.md',Path(__file__),ROOT/'experiments/round70_nvs_data.py',
        ROOT/'models/round70_nvs.py',ROOT/'models/round62_rgb_relations.py',ROOT/'models/round50_local_frame.py',
        ROOT/'tests/test_round70_nvs.py',ROOT/'experiments/round65_queue.py',
        ROOT/'experiments/round65_run.py',ROOT/'experiments/round65_common.py',
        OUT/'data/manifest.json',OUT/'units.json',OUT/'cuda_cost_preflight.json',OUT/'interface_preflight.json']
    for folder in ('third_party/round70_prope','third_party/round70_lpips','third_party/round70_weights'):
        files.extend(p for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc')
    manifest = read(OUT/'data/manifest.json')
    for path,expected in manifest['cache_sha256'].items(): assert sha(path)==expected,path
    lock = dict(settings=SETTINGS,source_sha256={str(p):sha(p) for p in files},
        cache_sha256=manifest['cache_sha256'],created_unix=time.time(),fresh_confirmation=False,
        goal_complete=False,preflight_updates=0,fit_updates_per_role_seed=2160)
    write(OUT/'lock.json',lock);(OUT/'lock.sha256').write_text(sha(OUT/'lock.json')+'\n')
    print('R70_FROZEN',sha(OUT/'lock.json'),flush=True)


def check():
    lock=read(OUT/'lock.json')
    assert sha(OUT/'lock.json')==(OUT/'lock.sha256').read_text().strip()
    for path,expected in lock['source_sha256'].items():
        assert sha(path)==expected,('changed source',path)
    for path,expected in lock['cache_sha256'].items(): assert sha(path)==expected,('changed cache',path)
    return lock


def infer(model,data,role,condition='base'):
    outputs=[];targets=[]
    model.eval()
    with torch.inference_mode():
        for start in range(0,len(data),8):
            inp,target=data.batch(np.arange(start,min(start+8,len(data))),role,'cuda',condition)
            with torch.autocast('cuda',dtype=torch.bfloat16): pred=model(inp)
            if not torch.isfinite(pred).all(): raise FloatingPointError('nonfinite inference')
            outputs.append(pred.float().cpu().numpy());targets.append(target.cpu().numpy())
    return np.concatenate(outputs),np.concatenate(targets)


def train(role,seed):
    check();configure();torch.manual_seed(seed);np.random.seed(seed)
    assert torch.cuda.is_available()
    folder=OUT/'runs'/role/str(seed)
    folder.mkdir(parents=True,exist_ok=False)
    model=CompactNVS(role).cuda();initial_digest=digest(model)
    opt=torch.optim.AdamW(model.parameters(),lr=SETTINGS['lr'],weight_decay=SETTINGS['weight_decay'])
    fit,val=NVSData('fit'),NVSData('val')
    rng=np.random.default_rng(seed+700000);history={'training':[],'validation':[]}
    step=0;best=float('inf');stream='00'*32
    torch.cuda.reset_peak_memory_stats();began=time.monotonic()
    for epoch in range(1,61):
        model.train();order=rng.permutation(len(fit));losses=[];norms=[]
        epoch_start=time.monotonic()
        for start in range(0,len(fit),8):
            indices=order[start:start+8];swaps=rng.random(len(indices))<.5
            stream=stream_update(stream,indices,swaps)
            inp,target=fit.batch(indices,role,'cuda',swaps=swaps)
            if step<108: lr=SETTINGS['lr']*(step+1)/108
            else:
                fraction=(step-108)/(2160-108-1)
                lr=SETTINGS['final_lr']+(SETTINGS['lr']-SETTINGS['final_lr'])*.5*(1+math.cos(math.pi*fraction))
            for group in opt.param_groups: group['lr']=lr
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16): prediction=model(inp)
            loss=(prediction.float()-target).square().mean()
            if not torch.isfinite(loss): raise FloatingPointError('nonfinite train loss')
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            opt.step();step+=1;losses.append(loss.item());norms.append(norm.item())
        history['training'].append(dict(epoch=epoch,step=step,mse=float(np.mean(losses)),
            gradient_norm_max=float(np.max(norms)),last_lr=lr,input_stream_sha256=stream,
            epoch_seconds=time.monotonic()-epoch_start))
        if epoch%3==0:
            pred,target=infer(model,val,role)
            mse=float(np.square(pred.astype('float64')-target.astype('float64')).mean())
            history['validation'].append(dict(epoch=epoch,step=step,mse=mse))
            if mse<best:
                best=mse
                atomic_save(folder/'best.pt',dict(model=model.state_dict(),epoch=epoch,step=step,
                    validation_mse=mse,role=role,seed=seed,lock_sha256=sha(OUT/'lock.json')))
            print(json.dumps({'role':role,'seed':seed,'epoch':epoch,'fit_mse':float(np.mean(losses)),
                              'val_mse':mse,'best_val_mse':best}),flush=True)
        write(folder/'history.json',history)
    atomic_save(folder/'terminal.pt',dict(model=model.state_dict(),optimizer=opt.state_dict(),
                                        epoch=60,step=step,role=role,seed=seed))
    chosen=torch.load(folder/'best.pt',map_location='cpu',weights_only=False)
    write(folder/'run.json',dict(role=role,seed=seed,steps=step,epochs=60,
        shared_initial_sha256=initial_digest,input_stream_sha256=stream,
        selected_epoch=chosen['epoch'],selected_val_mse=best,selected_sha256=sha(folder/'best.pt'),
        terminal_sha256=sha(folder/'terminal.pt'),parameters=sum(p.numel() for p in model.parameters()),
        pe_parameters=sum(p.numel() for n,p in model.named_parameters() if n.startswith(('relative.','hfield.'))),
        elapsed_seconds=time.monotonic()-began,peak_gib=torch.cuda.max_memory_allocated()/2**30,
        gpu=torch.cuda.get_device_name(),lock_sha256=sha(OUT/'lock.json'),finite_training=True))
    print('TRAIN_COMPLETE',role,seed,step,flush=True)


def selections():
    check();rows=[]
    for seed in SEEDS:
        common=[];streams=[]
        for role in ROLES:
            folder=OUT/'runs'/role/str(seed);run=read(folder/'run.json');history=read(folder/'history.json')
            assert run['steps']==2160 and len(history['training'])==60
            assert run['selected_epoch']==min(history['validation'],key=lambda x:x['mse'])['epoch']
            assert sha(folder/'best.pt')==run['selected_sha256']
            common.append(run['shared_initial_sha256'])
            streams.append([x['input_stream_sha256'] for x in history['training']])
            rows.append(run)
        assert len(set(common))==1 and all(s==streams[0] for s in streams)
    write(OUT/'selections.json',dict(rows=rows,all_30_selected_before_old_eval=True,
         shared_initialization_and_all_epoch_streams_paired=True,unix=time.time(),lock_sha256=sha(OUT/'lock.json')))
    print('R70_SELECTIONS_COMPLETE',len(rows),flush=True)


def perceptual_metric():
    sys.path.insert(0,str(ROOT/'third_party/round70_lpips'))
    import lpips
    torch.hub.set_dir(str(ROOT/'third_party/round70_weights'))
    net=lpips.LPIPS(net='squeeze',version='0.1',verbose=False).cuda().eval()
    for p in net.parameters(): p.requires_grad_(False)
    return net


def metrics(prediction,target,net):
    from skimage.metrics import structural_similarity
    pred=prediction.clip(0,1)
    raw_mse=np.square(prediction.astype('float64')-target.astype('float64')).mean((1,2,3))
    mse=np.square(pred.astype('float64')-target.astype('float64')).mean((1,2,3))
    psnr=-10*np.log10(np.maximum(mse,1e-15))
    ssim=np.array([structural_similarity(a,b,data_range=1.,channel_axis=-1,gaussian_weights=True,
                         sigma=1.5,use_sample_covariance=False) for a,b in zip(pred,target)])
    lp=[]
    with torch.inference_mode():
        for start in range(0,len(pred),8):
            a=torch.from_numpy(pred[start:start+8]).permute(0,3,1,2).cuda()
            b=torch.from_numpy(target[start:start+8]).permute(0,3,1,2).cuda()
            lp.extend(net(a,b,normalize=True).flatten().cpu().tolist())
    return dict(psnr=psnr,ssim=ssim,lpips=np.array(lp),raw_mse=raw_mse,
        clipped_fraction=((prediction<0)|(prediction>1)).mean((1,2,3)))


def export(seed):
    check();configure();sel=read(OUT/'selections.json')
    assert sel['all_30_selected_before_old_eval']
    data=NVSData('old_eval');net=perceptual_metric()
    for role in ROLES:
        folder=OUT/'runs'/role/str(seed)
        selected=torch.load(folder/'best.pt',map_location='cpu',weights_only=False)
        model=CompactNVS(role).cuda();model.load_state_dict(selected['model'])
        dest=OUT/'predictions'/role/str(seed);dest.mkdir(parents=True,exist_ok=False)
        for condition in CONDITIONS:
            before=time.monotonic();pred,target=infer(model,data,role,condition)
            values=metrics(pred,target,net)
            # bf16 model output is exactly representable in float16 here; assert,
            # so saved images can independently reproduce the primary metrics.
            stored=pred.astype('float16')
            if not np.array_equal(stored.astype('float32'),pred):
                stored=pred  # Preserve exact predictions rather than a quantized metric source.
            np.savez_compressed(dest/f'{condition}.npz',prediction=stored,
                target_uint8=np.rint(target*255).astype('uint8'),scene_id=data.arrays['scene_id'],**values)
            write(dest/f'{condition}.json',dict(role=role,seed=seed,condition=condition,
                checkpoint_sha256=sha(folder/'best.pt'),selections_sha256=sha(OUT/'selections.json'),
                prediction_sha256=sha(dest/f'{condition}.npz'),samples=len(pred),
                elapsed_seconds=time.monotonic()-before,metrics={k:float(v.mean()) for k,v in values.items()}))
            print('EXPORTED',role,seed,condition,flush=True)
        del model;torch.cuda.empty_cache()
    print('EXPORT_COMPLETE',seed,flush=True)


def launch(stage,job,gpu):
    tag=stage if job is None else stage+'_'+'_'.join(map(str,job))
    key=tag+'_'+str(time.time_ns())
    command=[sys.executable,'-m',MODULE,stage]
    if stage=='train': command+=['--role',job[0],'--seed',str(job[1])]
    elif stage=='export': command+=['--seed',str(job[0])]
    env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
        CUBLAS_WORKSPACE_CONFIG=':4096:8',CUDA_DEVICE_ORDER='PCI_BUS_ID',
        CUDA_VISIBLE_DEVICES='' if gpu is None else gpu['uuid'])
    stream=(OUT/(key+'.log')).open('x')
    try: process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    except BaseException: stream.close();raise
    meta=dict(tag=key,stage=stage,job=job,pid=process.pid,gpu=gpu,command=command,started_unix=time.time())
    scheduler.ACTIVE[key]=dict(process=process,stream=stream,meta=meta)
    print('LAUNCHED',key,process.pid,None if gpu is None else gpu['index'],flush=True)


def queue():
    check()
    if (OUT/'queue_status.json').exists(): raise ValueError('inspect existing controller first')
    scheduler.OUT=OUT;scheduler.INFO.update(pid=os.getpid(),started_unix=time.time(),
        start_ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],lock_sha256=sha(OUT/'lock.json'))
    def stop(signum,frame): raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    jobs=[(ROLES[(i+j)%len(ROLES)],seed) for j in range(len(ROLES)) for i,seed in enumerate(SEEDS)]
    try:
        for stage,jobs in [('train',jobs),('selections',[None]),('export',[(s,) for s in SEEDS])]:
            pending=list(jobs);previous=set()
            while pending or scheduler.ACTIVE:
                if scheduler.collect(): raise ValueError('unexpected retry')
                if not pending and not scheduler.ACTIVE: break
                if stage=='selections':
                    if pending: launch(stage,pending.pop(0),None)
                    scheduler.record('running',stage=stage,pending=pending)
                else:
                    rows=scheduler.occupancy();available={r['uuid'] for r in rows if r['free']};stable=available&previous
                    owned={x['meta']['gpu']['uuid'] for x in scheduler.ACTIVE.values()}
                    for gpu in rows:
                        if not pending or len(scheduler.ACTIVE)>=3: break
                        if gpu['uuid'] in stable-owned: launch(stage,pending.pop(0),gpu);owned.add(gpu['uuid'])
                    previous=available-owned
                    scheduler.record('running' if scheduler.ACTIVE else 'waiting_for_gpu',stage=stage,pending=pending,occupancy=rows)
                time.sleep(5)
        scheduler.record('complete',task_execution_completed=True,scientific_review_pending=True,goal_complete=False)
    except BaseException as error:
        scheduler.stop_owned();scheduler.record('stopped',error_type=type(error).__name__,error=str(error),goal_complete=False);raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('freeze','train','selections','export','queue'))
    parser.add_argument('--role',choices=ROLES);parser.add_argument('--seed',type=int,choices=SEEDS)
    args=parser.parse_args()
    if args.action=='train': train(args.role,args.seed)
    elif args.action=='export': export(args.seed)
    else: globals()[args.action]()
