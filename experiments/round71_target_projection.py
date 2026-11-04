"""Fixed R71 experiment; reuse R70 training/metrics without editing its sources."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np
import torch

from models.round71_target_projection import TargetProjectionNVS, ROLES
from experiments import round70_nvs_run as base
from experiments.round70_nvs_data import NVSData, ROOT
from experiments.round65_common import read, write, sha
from experiments.round70_gpu1_shared import gpu_status, GPU_UUID, MIN_FREE_MIB

OUT=ROOT/'results/round71_target_projection/attempt1'
OLD=ROOT/'results/round70_nvs/attempt1'
SEEDS=(11,22,33)
CONDITIONS=base.CONDITIONS
MODULE='experiments.round71_target_projection'


def check():
    lock=read(OUT/'lock.json')
    assert sha(OUT/'lock.json')==(OUT/'lock.sha256').read_text().strip()
    for name,expected in lock['source_sha256'].items():
        assert sha(name)==expected,('source changed',name)
    for name,expected in lock['cache_sha256'].items():
        assert sha(name)==expected,('data changed',name)
    return lock


def bind_training():
    # Function globals explicitly refer to this experiment's output/model/check.
    # Dataset globals remain R70, so no data rebuild or holdout reads occur.
    base.OUT=OUT; base.CompactNVS=TargetProjectionNVS; base.ROLES=ROLES; base.check=check


def preflight():
    base.configure();assert torch.cuda.is_available()
    fit=NVSData('fit');rows=[]
    torch.manual_seed(11);reference=base.CompactNVS('observed_prope').cuda().eval()
    ref_inputs,_=fit.batch(np.arange(8),'observed_prope','cuda')
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        reference_output=reference(ref_inputs).clone()
    del reference;torch.cuda.empty_cache()
    for role in ROLES:
        torch.manual_seed(11);model=TargetProjectionNVS(role).cuda()
        inp,target=fit.batch(np.arange(8),role,'cuda')
        torch.cuda.reset_peak_memory_stats();started=time.monotonic()
        with torch.autocast('cuda',dtype=torch.bfloat16):pred=model(inp)
        loss=(pred.float()-target).square().mean();loss.backward()
        assert torch.isfinite(pred).all() and torch.isfinite(loss)
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        disabled_error=None
        if role!='observed_urope':
            model.eval()
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                disabled=model(inp,disable_relation=True)
            disabled_error=float((disabled-reference_output).abs().max())
            assert disabled_error==0.,disabled_error
        torch.cuda.synchronize()
        rows.append(dict(role=role,parameters=sum(p.numel() for p in model.parameters()),
            pe_parameters=sum(p.numel() for n,p in model.named_parameters() if n.startswith('relative.')),
            peak_gib=torch.cuda.max_memory_allocated()/2**30,forward_backward_seconds=time.monotonic()-started,
            finite=True,optimizer_updates=0,disabled_relation_vs_R70_max_error=disabled_error))
        print('PREFLIGHT',rows[-1],flush=True)
        del model,pred,loss;torch.cuda.empty_cache()
    write(OUT/'preflight.json',dict(passed=True,rows=rows,model_sha256=sha(ROOT/'models/round71_target_projection.py'),
        gpu=torch.cuda.get_device_name(),optimizer_updates=0,used_split='old fit first eight, fixed correctness only'))


def freeze():
    assert not (OUT/'lock.json').exists()
    pre=read(OUT/'preflight.json');assert pre['passed'] and pre['model_sha256']==sha(ROOT/'models/round71_target_projection.py')
    old=read(OLD/'lock.json');assert sha(OLD/'lock.json')==(OLD/'lock.sha256').read_text().strip()
    sources=dict(old['source_sha256'])
    for p in [ROOT/'docs/round71_target_projection_plan.md', Path(__file__),
              ROOT/'models/round71_target_projection.py',ROOT/'tests/test_round71_target_projection.py',ROOT/'experiments/round71_report.py',
              ROOT/'experiments/round70_nvs_report.py',ROOT/'experiments/round70_gpu1_shared.py',
              OLD/'lock.json',OLD/'result.json',OLD/'review.json',OUT/'units.json',OUT/'preflight.json']:
        sources[str(p)]=sha(p)
    for p in (ROOT/'third_party/round71_urope').iterdir():
        if p.is_file():sources[str(p)]=sha(p)
    for name,h in sources.items():assert sha(name)==h,name
    settings=dict(base.SETTINGS,roles=ROLES,maximum_gpu_jobs=1,task='target-reference projection; observed geometry')
    write(OUT/'lock.json',dict(settings=settings,source_sha256=sources,cache_sha256=old['cache_sha256'],
        old_lock_sha256=sha(OLD/'lock.json'),created_unix=time.time(),fresh_confirmation=False,goal_complete=False))
    (OUT/'lock.sha256').write_text(sha(OUT/'lock.json')+'\n')
    print('R71_FROZEN',sha(OUT/'lock.json'),flush=True)


def selections():
    check();rows=[]
    for seed in SEEDS:
        reference=read(OLD/'runs/observed_prope'/str(seed)/'run.json')
        old_history=read(OLD/'runs/observed_prope'/str(seed)/'history.json')
        for role in ROLES:
            p=OUT/'runs'/role/str(seed);run=read(p/'run.json');h=read(p/'history.json')
            assert run['steps']==2160 and len(h['training'])==60
            assert run['selected_epoch']==min(h['validation'],key=lambda x:x['mse'])['epoch']
            assert run['selected_sha256']==sha(p/'best.pt')
            assert run['shared_initial_sha256']==reference['shared_initial_sha256']
            assert [x['input_stream_sha256'] for x in h['training']]==[x['input_stream_sha256'] for x in old_history['training']]
            rows.append(run)
    write(OUT/'selections.json',dict(rows=rows,all_12_selected_before_old_eval=True,
        paired_to_R70_initialization_and_every_epoch_stream=True,lock_sha256=sha(OUT/'lock.json'),unix=time.time()))


def export(seed):
    check();base.configure();assert read(OUT/'selections.json')['all_12_selected_before_old_eval']
    data=NVSData('old_eval');net=base.perceptual_metric()
    for role in ROLES:
        p=OUT/'runs'/role/str(seed);selected=torch.load(p/'best.pt',map_location='cpu',weights_only=False)
        model=TargetProjectionNVS(role).cuda();model.load_state_dict(selected['model'])
        dest=OUT/'predictions'/role/str(seed);dest.mkdir(parents=True,exist_ok=False)
        for condition in CONDITIONS:
            pred,target=base.infer(model,data,role,condition);values=base.metrics(pred,target,net)
            stored=pred.astype('float16')
            if not np.array_equal(stored.astype('float32'),pred):stored=pred
            np.savez_compressed(dest/f'{condition}.npz',prediction=stored,target_uint8=np.rint(target*255).astype('uint8'),
                scene_id=data.arrays['scene_id'],**values)
            write(dest/f'{condition}.json',dict(role=role,seed=seed,condition=condition,
                checkpoint_sha256=sha(p/'best.pt'),selections_sha256=sha(OUT/'selections.json'),
                prediction_sha256=sha(dest/f'{condition}.npz'),samples=len(pred)))
            print('EXPORTED',role,seed,condition,flush=True)
        del model;torch.cuda.empty_cache()


def queue():
    check();identity=OUT/'execution_identity.json'
    if identity.exists():raise RuntimeError('Existing coordinator: inspect it, do not duplicate')
    meta=dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],
        source_sha256=sha(Path(__file__)),gpu_uuid=GPU_UUID,maximum_owned_gpu_jobs=1,
        min_free_mib=MIN_FREE_MIB,started_unix=time.time())
    write(identity,meta);completed=[];child=None;active=None;stage='initializing'
    def record(state,**extra):
        write(OUT/'queue_status.json',dict(meta,state=state,stage=stage,active=active,
              completed=completed,observed_unix=time.time(),goal_complete=False,**extra))
    def stop(signum,frame):raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def run(action,role=None,seed=None,gpu=True):
        nonlocal child,active,stage
        stage=action;occupancy=None
        if gpu:
            while True:
                occupancy=gpu_status()
                if occupancy['free_mib']>=MIN_FREE_MIB:break
                record('waiting_for_gpu1_memory',occupancy=occupancy);time.sleep(5)
        module=MODULE if action!='summarize' else 'experiments.round71_report'
        cmd=[sys.executable,'-m',module,action]
        if role:cmd+=['--role',role]
        if seed is not None:cmd+=['--seed',str(seed)]
        env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
            CUBLAS_WORKSPACE_CONFIG=':4096:8',CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=GPU_UUID if gpu else '')
        log=OUT/(action+'_'+str(role)+'_'+str(seed)+'_'+str(time.time_ns())+'.log')
        with log.open('x') as f:
            child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            active=dict(pid=child.pid,action=action,role=role,seed=seed,log=str(log),occupancy=occupancy)
            print('LAUNCHED',active,flush=True)
            while child.poll() is None:
                record('running');time.sleep(5)
            code=child.returncode
        completed.append(dict(active,exit_code=code));child=None;active=None
        if code:raise RuntimeError(f'{log.name}: exit {code}')
    try:
        for j in range(len(ROLES)):
            for i,seed in enumerate(SEEDS):
                role=ROLES[(i+j)%len(ROLES)]
                run('train',role,seed)
        run('selections',gpu=False)
        for seed in SEEDS:run('export',seed=seed)
        run('summarize',gpu=False)
        run('audit')
        stage='complete';record('complete',scientific_review_pending=True,visual_review_pending=True)
    except BaseException as error:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        record('stopped',error=str(error));raise


def audit():
    check();base.configure();net=base.perceptual_metric();data=NVSData('val');rows=[]
    for role in ROLES:
        for seed in SEEDS:
            folder=OUT/'runs'/role/str(seed);run=read(folder/'run.json')
            model=TargetProjectionNVS(role).cuda()
            model.load_state_dict(torch.load(folder/'best.pt',map_location='cpu',weights_only=False)['model'])
            pred,target=base.infer(model,data,role)
            val=float(np.square(pred.astype('float64')-target.astype('float64')).mean())
            assert abs(val-run['selected_val_mse'])<1e-10
            with np.load(OUT/'predictions'/role/str(seed)/'base.npz') as z:
                ids=[0,48,96,144,192]
                checked=base.metrics(z['prediction'][ids].astype('float32'),z['target_uint8'][ids].astype('float32')/255,net)
                error={k:float(np.max(np.abs(v-z[k][ids]))) for k,v in checked.items()}
            assert max(error.values())<2e-5
            rows.append(dict(role=role,seed=seed,val_mse_replay_error=abs(val-run['selected_val_mse']),sampled_metric_errors=error))
            del model;torch.cuda.empty_cache()
    write(OUT/'audit.json',dict(passed=True,rows=rows,result_sha256=sha(OUT/'result.json'),external_replication=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['preflight','freeze','train','selections','export','queue','audit'])
    p.add_argument('--role',choices=ROLES);p.add_argument('--seed',type=int,choices=SEEDS);args=p.parse_args()
    if args.action=='train':bind_training();base.train(args.role,args.seed)
    elif args.action=='export':export(args.seed)
    else:globals()[args.action]()
