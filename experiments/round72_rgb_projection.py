"""Frozen RGB counterpart of R71, with independent multi-GPU fits."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import numpy as np
import torch

from experiments import round71_target_projection as engine
from experiments import round70_nvs_run as base
from experiments.round70_nvs_data import ROOT, NVSData
from experiments.round65_common import read, write, sha
from models.round70_nvs import CompactNVS
from models.round72_rgb_projection import RGBTargetProjectionNVS, ROLES

OUT=ROOT/'results/round72_rgb_projection/attempt1'
OLD=ROOT/'results/round70_nvs/attempt1'
PARENT=ROOT/'results/round71_target_projection/attempt1'
SEEDS=(11,22,33)
CONDITIONS=base.CONDITIONS
MODULE='experiments.round72_rgb_projection'
MAX_PARALLEL=4
MIN_FREE_MIB=12288


def check():
    lock=read(OUT/'lock.json')
    assert sha(OUT/'lock.json')==(OUT/'lock.sha256').read_text().strip()
    for path,digest in lock['source_sha256'].items():assert sha(path)==digest,('source changed',path)
    for path,digest in lock['cache_sha256'].items():assert sha(path)==digest,('cache changed',path)
    return lock


def bind():
    engine.OUT=OUT;engine.OLD=OLD;engine.ROLES=ROLES
    engine.TargetProjectionNVS=RGBTargetProjectionNVS;engine.check=check


def preflight():
    base.configure();assert torch.cuda.is_available()
    assert not (OUT/'preflight.json').exists()
    fit=NVSData('fit');inputs,target=fit.batch(np.arange(8),'rgb_prope','cuda')
    assert set(inputs)=={'ref_rgb','K','c2w'}
    torch.manual_seed(11);reference=CompactNVS('rgb_prope').cuda().eval()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):expected=reference(inputs).clone()
    del reference;torch.cuda.empty_cache();rows=[]
    for role in ROLES:
        torch.manual_seed(11);model=RGBTargetProjectionNVS(role).cuda()
        torch.cuda.reset_peak_memory_stats();started=time.monotonic()
        with torch.autocast('cuda',dtype=torch.bfloat16):prediction=model(inputs)
        loss=(prediction.float()-target).square().mean();loss.backward()
        assert torch.isfinite(prediction).all() and torch.isfinite(loss)
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        grad=model.geometry_head.weight.grad
        depth_grad=float(grad[0].norm());shape_grad=float(grad[1:].norm())
        assert depth_grad>0 and shape_grad>0
        error=None
        if role!='rgb_urope':
            model.eval()
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):disabled=model(inputs,disable_relation=True)
            error=float((disabled-expected).abs().max());assert error==0.,error
        torch.cuda.synchronize()
        rows.append(dict(role=role,parameters=sum(p.numel() for p in model.parameters()),
            pe_parameters=sum(p.numel() for n,p in model.named_parameters() if n.startswith('relative.')),
            peak_gib=torch.cuda.max_memory_allocated()/2**30,seconds=time.monotonic()-started,
            depth_head_gradient=depth_grad,shape_head_gradient=shape_grad,
            finite=True,disabled_relation_vs_R70_max_error=error,optimizer_updates=0))
        print('PREFLIGHT',rows[-1],flush=True)
        del model,prediction,loss;torch.cuda.empty_cache()
    write(OUT/'preflight.json',dict(passed=True,rows=rows,model_sha256=sha(ROOT/'models/round72_rgb_projection.py'),
        gpu=torch.cuda.get_device_name(),used_split='fixed first eight old fit RGB triples',optimizer_updates=0))


def freeze():
    assert not (OUT/'lock.json').exists()
    assert read(PARENT/'review.json')['scientific_review_complete']
    assert read(PARENT/'result.json')['decision']['fixed_utility_criterion_met']
    assert read(OUT/'units.json')['passed'] and read(OUT/'preflight.json')['passed']
    assert read(OUT/'preflight.json')['model_sha256']==sha(ROOT/'models/round72_rgb_projection.py')
    parent=read(PARENT/'lock.json');sources=dict(parent['source_sha256'])
    for f in [ROOT/'docs/round72_rgb_projection_plan.md',Path(__file__),ROOT/'models/round72_rgb_projection.py',
              ROOT/'tests/test_round72_rgb_projection.py',ROOT/'experiments/round72_report.py',
              PARENT/'lock.json',PARENT/'result.json',PARENT/'review.json',PARENT/'audit.json',
              OUT/'units.json',OUT/'preflight.json']:
        sources[str(f)]=sha(f)
    for path,digest in sources.items():assert sha(path)==digest,path
    for path,digest in parent['cache_sha256'].items():assert sha(path)==digest,path
    settings=dict(base.SETTINGS,roles=ROLES,maximum_gpu_jobs=MAX_PARALLEL,
        task='RGB-only target-reference projected support',resource_policy='idle GPUs; GPU1 sharing allowed')
    write(OUT/'lock.json',dict(settings=settings,source_sha256=sources,cache_sha256=parent['cache_sha256'],
        created_unix=time.time(),parent_lock_sha256=sha(PARENT/'lock.json'),fresh_confirmation=False,goal_complete=False))
    (OUT/'lock.sha256').write_text(sha(OUT/'lock.json')+'\n')
    print('R72_FROZEN',sha(OUT/'lock.json'),flush=True)


def selections():
    check();rows=[]
    for seed in SEEDS:
        p=OLD/'runs/rgb_prope'/str(seed);reference=read(p/'run.json');old_history=read(p/'history.json')
        for role in ROLES:
            folder=OUT/'runs'/role/str(seed);run=read(folder/'run.json');history=read(folder/'history.json')
            assert run['steps']==2160 and len(history['training'])==60
            assert run['selected_epoch']==min(history['validation'],key=lambda x:x['mse'])['epoch']
            assert run['selected_sha256']==sha(folder/'best.pt')
            assert run['shared_initial_sha256']==reference['shared_initial_sha256']
            assert [x['input_stream_sha256'] for x in history['training']]==[x['input_stream_sha256'] for x in old_history['training']]
            rows.append(run)
    write(OUT/'selections.json',dict(rows=rows,all_12_selected_before_old_eval=True,
        paired_to_R70_RGB_initialization_and_every_epoch_stream=True,lock_sha256=sha(OUT/'lock.json'),unix=time.time()))


def available_gpus():
    output=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu',
        '--format=csv,noheader,nounits'],text=True,timeout=10)
    rows=[]
    for line in output.strip().splitlines():
        i,uuid,used,free,util=[x.strip() for x in line.split(',')]
        row=dict(index=int(i),uuid=uuid,memory_mib=int(used),free_mib=int(free),utilization=int(util))
        idle=row['memory_mib']<=512 and row['utilization']<=10
        if row['free_mib']>=MIN_FREE_MIB and (idle or row['index']==1):
            row['shared_allowed']=row['index']==1;row['idle_at_observation']=idle;rows.append(row)
    return sorted(rows,key=lambda x:(not x['idle_at_observation'],x['index']))


def queue():
    check();identity=OUT/'execution_identity.json'
    if identity.exists():raise RuntimeError('Inspect existing coordinator; do not duplicate')
    meta=dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],
        source_sha256=sha(Path(__file__)),lock_sha256=sha(OUT/'lock.json'),maximum_owned_gpu_jobs=MAX_PARALLEL,
        started_unix=time.time(),user_authorized_parallel=True,goal_complete=False)
    write(identity,meta);active=[];completed=[];pending=[];stage='initializing'
    def record(state,**extra):
        write(OUT/'queue_status.json',dict(meta,state=state,stage=stage,active=[r[1] for r in active],
            completed=completed,pending=pending,observed_unix=time.time(),**extra))
    def stop(signum,frame):raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def batch(action,jobs,use_gpu=True):
        nonlocal stage,pending
        stage=action;pending=list(jobs)
        while pending or active:
            for process,item,stream in list(active):
                code=process.poll()
                if code is not None:
                    stream.close();active.remove((process,item,stream));completed.append(dict(item,exit_code=code))
                    if code:raise RuntimeError(f"{item['log']}: exit {code}")
            choices=available_gpus() if use_gpu and pending else [None]
            busy={item['gpu']['uuid'] for _,item,_ in active if item['gpu'] is not None}
            for gpu in choices:
                if not pending or len(active)>=(MAX_PARALLEL if use_gpu else 1):break
                if gpu is not None and gpu['uuid'] in busy:continue
                role,seed=pending.pop(0)
                command=[sys.executable,'-m','experiments.round72_report' if action=='summarize' else MODULE,action]
                if role:command+=['--role',role]
                if seed is not None:command+=['--seed',str(seed)]
                env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
                    CUBLAS_WORKSPACE_CONFIG=':4096:8',CUDA_DEVICE_ORDER='PCI_BUS_ID',
                    CUDA_VISIBLE_DEVICES='' if gpu is None else gpu['uuid'])
                log=OUT/(f'{action}_{role}_{seed}_{time.time_ns()}.log');stream=log.open('x')
                process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                item=dict(pid=process.pid,action=action,role=role,seed=seed,gpu=gpu,log=str(log),started_unix=time.time())
                active.append((process,item,stream))
                if gpu is not None:busy.add(gpu['uuid'])
                print('LAUNCHED',item,flush=True)
            record('running' if active else 'waiting_for_gpu');time.sleep(5)
    try:
        jobs=[(ROLES[(i+j)%len(ROLES)],seed) for j in range(len(ROLES)) for i,seed in enumerate(SEEDS)]
        batch('train',jobs);batch('selections',[(None,None)],False)
        batch('export',[(None,s) for s in SEEDS]);batch('summarize',[(None,None)],False)
        batch('audit',[(None,None)])
        stage='complete';record('complete',scientific_review_pending=True,visual_review_pending=True)
    except BaseException as error:
        for process,item,stream in active:
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
            stream.close()
        record('stopped',error=str(error));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['preflight','freeze','train','selections','export','queue','audit'])
    p.add_argument('--role',choices=ROLES);p.add_argument('--seed',type=int,choices=SEEDS);args=p.parse_args()
    if args.action in ('train','export','audit'):
        bind()
        if args.action=='train':engine.bind_training();base.train(args.role,args.seed)
        elif args.action=='export':engine.export(args.seed)
        else:engine.audit()
    else:globals()[args.action]()
