"""Six old-data fits for the prospectively fixed C=Q attribution control."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import numpy as np
import torch

from experiments.round65_common import ROOT,read,write,sha
from experiments.round70_nvs_data import NVSData
from experiments import round70_nvs_run as base
from models.projected_shape_attribution import FixedShapeNVS,ROLES
from models.round70_nvs import CompactNVS

OUT=ROOT/'results/projected_shape_attribution/attempt1'
PLAN=ROOT/'results/projected_shape_attribution/preparation/plan.json'
R71=ROOT/'results/round71_target_projection/attempt1'
R72=ROOT/'results/round72_rgb_projection/attempt1'
R73=ROOT/'results/round73_native_relative5/attempt1'
SEEDS=(11,22,33)
CONDITIONS=base.CONDITIONS
MODULE='experiments.projected_shape_attribution'


def parent(role):
    return (R71 if role.startswith('observed_') else R72),role.replace('fixedshape','relative5')


def freeze():
    assert not (OUT/'lock.json').exists();plan=read(PLAN);units=read(OUT/'units.json')
    assert units['passed']
    for path,digest in plan['source_sha256'].items():assert sha(path)==digest,path
    for path,digest in units['source_sha256'].items():assert sha(path)==digest,path
    old=read(R72/'lock.json');sources=dict(old['source_sha256'])
    files=[PLAN,Path(__file__),ROOT/'experiments/projected_shape_report.py',
        ROOT/'models/projected_shape_attribution.py',ROOT/'tests/test_projected_shape_attribution.py',OUT/'units.json']
    for root in (R71,R72):
        for name in ('lock.json','result.json','review.json','selections.json'):files.append(root/name)
    for role in ROLES:
        root,candidate=parent(role)
        for seed in SEEDS:
            for name in ('run.json','history.json','best.pt'):files.append(root/'runs'/candidate/str(seed)/name)
    for path in files:sources[str(path)]=sha(path)
    for path,digest in sources.items():assert sha(path)==digest,path
    write(OUT/'lock.json',dict(settings=dict(base.SETTINGS,roles=ROLES,maximum_gpu_jobs=1),
        source_sha256=sources,cache_sha256=old['cache_sha256'],plan_sha256=sha(PLAN),created_unix=time.time(),
        new_fits=6,no_confirmation_data_access=True,fresh_confirmation=False,goal_complete=False,
        resource_order='Wait until all R73 seeds have been launched, or that controller is terminal with no owned child; then one idle GPU.'))
    (OUT/'lock.sha256').write_text(sha(OUT/'lock.json')+'\n')
    print('SHAPE_ATTRIBUTION_FROZEN',sha(OUT/'lock.json'),flush=True)


def check():
    lock=read(OUT/'lock.json');assert sha(OUT/'lock.json')==(OUT/'lock.sha256').read_text().strip()
    for path,digest in lock['source_sha256'].items():assert sha(path)==digest,path
    for path,digest in lock['cache_sha256'].items():assert sha(path)==digest,path
    return lock


def preflight():
    check();base.configure();assert torch.cuda.is_available();assert not (OUT/'preflight.json').exists()
    data=NVSData('fit');rows=[]
    for role in ROLES:
        inputs,target=data.batch(np.arange(8),role,'cuda')
        torch.manual_seed(11);reference=CompactNVS('observed_prope' if role.startswith('observed_') else 'rgb_prope').cuda().eval()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):expected=reference(inputs).clone()
        del reference;torch.cuda.empty_cache()
        torch.manual_seed(11);model=FixedShapeNVS(role).cuda()
        assert len(model.relative._forward_pre_hooks)==1
        torch.cuda.reset_peak_memory_stats()
        with torch.autocast('cuda',dtype=torch.bfloat16):prediction=model(inputs)
        loss=(prediction.float()-target).square().mean();loss.backward()
        assert torch.isfinite(prediction).all() and torch.isfinite(loss)
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        gradient=float(model.relative[0].weight.grad.norm());assert gradient>0
        shape_gradient=None
        if role.startswith('rgb_'):
            shape_gradient=float(model.geometry_head.weight.grad[1:].norm());assert shape_gradient>0
        model.eval()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):disabled=model(inputs,disable_relation=True)
        error=float((disabled-expected).abs().max());assert error==0
        rows.append(dict(role=role,finite=True,pe_gradient=gradient,RGB_common_shape_head_gradient=shape_gradient,
            disabled_relation_error=error,parameters=sum(p.numel() for p in model.parameters()),
            pe_parameters=sum(p.numel() for p in model.relative.parameters()),peak_gib=torch.cuda.max_memory_allocated()/2**30))
        del model,prediction,loss;torch.cuda.empty_cache()
    write(OUT/'preflight.json',dict(passed=True,rows=rows,model_sha256=sha(ROOT/'models/projected_shape_attribution.py'),
        optimizer_updates=0,data='fixed first eight old fit triples only'))


def train(role,seed):
    check();pre=read(OUT/'preflight.json')
    assert pre['passed'] and pre['model_sha256']==sha(ROOT/'models/projected_shape_attribution.py')
    base.OUT=OUT;base.CompactNVS=FixedShapeNVS;base.ROLES=ROLES;base.check=check
    base.train(role,seed)


def selections():
    check();rows=[]
    for role in ROLES:
        root,candidate=parent(role)
        for seed in SEEDS:
            folder=OUT/'runs'/role/str(seed);run=read(folder/'run.json');history=read(folder/'history.json')
            old=root/'runs'/candidate/str(seed);oldrun=read(old/'run.json');oldhistory=read(old/'history.json')
            assert run['steps']==2160 and len(history['training'])==60
            assert run['selected_epoch']==min(history['validation'],key=lambda x:x['mse'])['epoch']
            assert run['selected_sha256']==sha(folder/'best.pt')
            assert run['shared_initial_sha256']==oldrun['shared_initial_sha256']
            assert run['parameters']==oldrun['parameters'] and run['pe_parameters']==oldrun['pe_parameters']
            assert [x['input_stream_sha256'] for x in history['training']]==[x['input_stream_sha256'] for x in oldhistory['training']]
            rows.append(run)
    write(OUT/'selections.json',dict(rows=rows,all_six_selected_before_old_eval=True,
        matched_parent_budget_capacity_initialization_and_input_stream=True,lock_sha256=sha(OUT/'lock.json'),unix=time.time()))


def export(seed):
    check();base.configure();assert read(OUT/'selections.json')['all_six_selected_before_old_eval']
    data=NVSData('old_eval');net=base.perceptual_metric()
    for role in ROLES:
        folder=OUT/'runs'/role/str(seed);model=FixedShapeNVS(role).cuda()
        model.load_state_dict(torch.load(folder/'best.pt',map_location='cpu',weights_only=False)['model'])
        dest=OUT/'predictions'/role/str(seed);dest.mkdir(parents=True,exist_ok=False)
        for condition in CONDITIONS:
            pred,target=base.infer(model,data,role,condition);values=base.metrics(pred,target,net)
            stored=pred.astype('float16')
            if not np.array_equal(stored.astype('float32'),pred):stored=pred
            path=dest/f'{condition}.npz'
            np.savez_compressed(path,prediction=stored,target_uint8=np.rint(target*255).astype('uint8'),scene_id=data.arrays['scene_id'],**values)
            write(path.with_suffix('.json'),dict(role=role,seed=seed,condition=condition,checkpoint_sha256=sha(folder/'best.pt'),
                selections_sha256=sha(OUT/'selections.json'),prediction_sha256=sha(path),samples=len(pred)))
            print('EXPORTED',role,seed,condition,flush=True)
        del model;torch.cuda.empty_cache()


def audit():
    check();base.configure();data=NVSData('val');net=base.perceptual_metric();rows=[]
    for role in ROLES:
        for seed in SEEDS:
            folder=OUT/'runs'/role/str(seed);run=read(folder/'run.json');model=FixedShapeNVS(role).cuda()
            model.load_state_dict(torch.load(folder/'best.pt',map_location='cpu',weights_only=False)['model'])
            pred,target=base.infer(model,data,role)
            error=abs(float(np.square(pred.astype('float64')-target.astype('float64')).mean())-run['selected_val_mse'])
            assert error<1e-10
            with np.load(OUT/'predictions'/role/str(seed)/'base.npz') as z:
                ids=[0,48,96,144,192]
                values=base.metrics(z['prediction'][ids].astype('float32'),z['target_uint8'][ids].astype('float32')/255,net)
                metric_error={k:float(np.max(np.abs(v-z[k][ids]))) for k,v in values.items()}
            assert max(metric_error.values())<2e-5
            rows.append(dict(role=role,seed=seed,val_mse_replay_error=error,sampled_metric_errors=metric_error))
            del model;torch.cuda.empty_cache()
    write(OUT/'audit.json',dict(passed=True,rows=rows,result_sha256=sha(OUT/'result.json'),external_replication=False))


def native_priority_released():
    q=read(R73/'queue_status.json')
    launched={r['seed'] for r in q['completed']+q['active'] if r['action']=='train'}
    if set(SEEDS).issubset(launched):return True
    if q['state'] not in ('complete','stopped'):return False
    for row in q['active']:
        cmd=Path(f"/proc/{row['pid']}/cmdline")
        if cmd.exists() and b'experiments.round73_native_relative5' in cmd.read_bytes():return False
    return True


def idle_gpus():
    output=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu',
        '--format=csv,noheader,nounits'],text=True,timeout=10);rows=[]
    for line in output.strip().splitlines():
        index,uuid,used,free,util=[x.strip() for x in line.split(',')]
        if int(used)<=512 and int(free)>=12288 and int(util)<=10:
            rows.append(dict(index=int(index),uuid=uuid,memory_mib=int(used),free_mib=int(free),utilization=int(util)))
    return rows


def queue():
    check();assert not (OUT/'execution_identity.json').exists()
    meta=dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],
        source_sha256=sha(Path(__file__)),lock_sha256=sha(OUT/'lock.json'),maximum_owned_gpu_jobs=1,
        started_unix=time.time(),goal_complete=False)
    write(OUT/'execution_identity.json',meta);completed=[];active=None;child=None;stage='native_priority'
    def record(state,**extra):
        write(OUT/'queue_status.json',dict(meta,state=state,stage=stage,active=active,completed=completed,
            observed_unix=time.time(),**extra))
    def stop(signum,frame):raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def run(action,role=None,seed=None,gpu=True):
        nonlocal stage,active,child
        stage=action;device=None;probe=None
        if gpu:
            while True:
                choices=idle_gpus()
                if not choices:record('waiting_for_idle_gpu');time.sleep(5);continue
                device=choices[0]
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=device['uuid'],CUDA_DEVICE_ORDER='PCI_BUS_ID')
                code='import numpy; import torch; assert torch.cuda.is_available(); x=torch.ones(4,device="cuda"); assert float(x.sum())==4; torch.cuda.synchronize(); print(torch.cuda.get_device_name())'
                probe=subprocess.check_output([sys.executable,'-c',code],env=env,text=True,timeout=60).strip()
                if any(r['uuid']==device['uuid'] for r in idle_gpus()):break
                time.sleep(5)
        cmd=[sys.executable,'-m',MODULE,action]
        if role:cmd+=['--role',role]
        if seed is not None:cmd+=['--seed',str(seed)]
        env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
            CUBLAS_WORKSPACE_CONFIG=':4096:8',CUDA_DEVICE_ORDER='PCI_BUS_ID',
            CUDA_VISIBLE_DEVICES='' if device is None else device['uuid'])
        log=OUT/f'{action}_{role}_{seed}_{time.time_ns()}.log'
        with log.open('x') as stream:
            child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            active=dict(pid=child.pid,action=action,role=role,seed=seed,gpu=device,cuda_probe=probe,log=str(log),started_unix=time.time())
            print('LAUNCHED',active,flush=True)
            while child.poll() is None:record('running');time.sleep(5)
            code=child.returncode
        completed.append(dict(active,exit_code=code));active=None;child=None
        if code:raise RuntimeError(f'{log}: exit {code}')
    try:
        while not native_priority_released():record('waiting_for_R73_seed_launches');time.sleep(10)
        run('preflight')
        for j in range(2):
            for i,seed in enumerate(SEEDS):run('train',ROLES[(i+j)%2],seed)
        run('selections',gpu=False)
        for seed in SEEDS:run('export',seed=seed)
        run('summarize',gpu=False);run('audit')
        stage='complete';record('complete',scientific_review_pending=True,visual_review_pending=True)
    except BaseException as error:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        record('stopped',error=str(error));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','queue','preflight','train','selections','export','summarize','audit'])
    parser.add_argument('--role',choices=ROLES);parser.add_argument('--seed',type=int,choices=SEEDS);args=parser.parse_args()
    if args.action=='train':train(args.role,args.seed)
    elif args.action=='export':export(args.seed)
    elif args.action=='summarize':
        from experiments.projected_shape_report import summarize
        summarize()
    else:globals()[args.action]()
