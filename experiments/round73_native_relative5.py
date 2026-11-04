"""Single fixed native-GS candidate; reuse R65 fits/selection/evaluation semantics."""
import argparse
import ast
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import numpy as np
import torch

from experiments import round65_run as runner,round65_audit as audit,round65_data as data_module
from experiments.round65_common import ROOT,OUT as OLD,PREP,CONFIG,SEEDS,VARIANTS,ROLES as OLD_ROLES,read,write,sha
from models.round73_native_relative5 import CompactNativeClassifier,ROLES

OUT=ROOT/'results/round73_native_relative5/attempt1'
MODULE='experiments.round73_native_relative5'
CONFIRMATION=ROOT/'results/projection_confirmation/attempt1'
ARCHIVE=ROOT/'docs/archive/2026-11-04'


def legacy_sources():
    lock=read(OLD/'lock.json')
    assert sha(OLD/'lock.json')==(OLD/'lock.sha256').read_text().strip()
    assert lock['config']==CONFIG and sha(PREP/'train_catalog.json')==lock['catalog_sha256']
    sources={};aliases={}
    for name,digest in lock['sources'].items():
        path=ROOT/name
        if not path.exists():
            path=ARCHIVE/name;aliases[name]=str(path)
        assert sha(path)==digest,('R65 source identity',name)
        sources[str(path)]=digest
    return lock,sources,aliases


def local_dependencies(paths):
    pending=list(paths);result=set()
    while pending:
        path=Path(pending.pop()).resolve()
        if path in result:continue
        result.add(path)
        if path.suffix!='.py':continue
        for node in ast.walk(ast.parse(path.read_text())):
            modules=[]
            if isinstance(node,ast.ImportFrom) and node.module:modules.append(node.module)
            elif isinstance(node,ast.Import):modules.extend(x.name for x in node.names)
            for module in modules:
                if module.split('.')[0] not in ('models','experiments'):continue
                candidate=ROOT/(module.replace('.','/')+'.py')
                if candidate.is_file():pending.append(candidate)
    return sorted(result)


def freeze():
    assert not (OUT/'lock.json').exists()
    parent,sources,aliases=legacy_sources()
    assert not read(OLD/'review.json')['scientific_review_pending']
    unit=read(OUT/'units.json');assert unit['passed']
    for path,digest in unit['source_sha256'].items():assert sha(path)==digest,path
    files=local_dependencies([Path(__file__),ROOT/'models/round73_native_relative5.py',
        ROOT/'experiments/round73_report.py',ROOT/'tests/test_round73_native_relative5.py'])
    files+=[ROOT/'docs/round73_native_relative5_plan.md',OUT/'units.json',OLD/'lock.json',PREP/'train_catalog.json']
    for path in files:sources[str(path)]=sha(path)
    prior={name:sha(OLD/name) for name in ('selected.json','metrics/summary.json','review.json',
        'checkpoint_policy1/review.json','data/manifest.json')}
    for role in OLD_ROLES:
        for seed in SEEDS:
            for name in ('run.json','history.json','best.pt'):
                path=f'runs/{role}/{seed}/{name}';prior[path]=sha(OLD/path)
    write(OUT/'lock.json',dict(config=CONFIG,roles=ROLES,seeds=SEEDS,variants=VARIANTS,
        source_sha256=sources,archived_source_resolutions=aliases,prior_sha256=prior,
        catalog_sha256=parent['catalog_sha256'],previous_lock_sha256=sha(OLD/'lock.json'),
        fixed_candidate='R69 CompactPosition(relative5), unchanged',
        new_fits=3,previous_baselines_reused=15,fresh_confirmation=False,
        official_test_access=False,created_unix=time.time(),goal_complete=False))
    (OUT/'lock.sha256').write_text(sha(OUT/'lock.json')+'\n')
    (OUT/'data').symlink_to((OLD/'data').resolve(),target_is_directory=True)
    print('R73_FROZEN',sha(OUT/'lock.json'),'three paired native-GS fits',flush=True)


def check(out=OUT):
    if Path(out)==OLD:return legacy_sources()[0]
    assert Path(out)==OUT
    lock=read(OUT/'lock.json')
    assert sha(OUT/'lock.json')==(OUT/'lock.sha256').read_text().strip()
    assert lock['config']==CONFIG and lock['roles']==list(ROLES)
    assert lock['seeds']==list(SEEDS) and lock['variants']==list(VARIANTS)
    for path,digest in lock['source_sha256'].items():assert sha(path)==digest,path
    for name,digest in lock['prior_sha256'].items():assert sha(OLD/name)==digest,name
    assert (OUT/'data').resolve()==(OLD/'data').resolve()
    return lock


def bind():
    check()
    runner.NativeGSClassifier=CompactNativeClassifier;runner.ROLES=ROLES;runner.checked=check
    audit.NativeGSClassifier=CompactNativeClassifier;audit.ROLES=ROLES;audit.checked=check
    data_module.checked=check


def preflight():
    bind();runner.preflight(OUT/'preflight1')
    result=read(OUT/'preflight1/result.json')
    result.update(lock_sha256=sha(OUT/'lock.json'),model_sha256=sha(ROOT/'models/round73_native_relative5.py'),
        formal_fit_updates=0,synthetic_optimizer_updates_per_role=1)
    write(OUT/'cuda_preflight.json',result)


def selections():
    bind();audit.selections(OUT);selected=read(OUT/'selected.json')
    selected.pop('all_fifteen_selected_before_development_evaluation')
    selected['all_three_selected_before_development_evaluation']=True
    pairing=[];operator=[]
    for seed in SEEDS:
        current=OUT/'runs/relative5'/str(seed);previous=OLD/'runs/distance_mlp'/str(seed)
        assert read(current/'run.json')['initial_shared_sha256']==read(previous/'run.json')['initial_shared_sha256']
        streams=lambda p:[r['stream_sha256'] for r in read(p/'history.json')['training']]
        assert streams(current)==streams(previous)
        pairing.append(dict(seed=seed,all120_epoch_streams_match=True))
        model=CompactNativeClassifier('relative5')
        model.load_state_dict(torch.load(current/'best.pt',map_location='cpu',weights_only=True)['state_dict'])
        c,s,_=runner.synthetic_inputs(1,9,'cpu')
        expected=audit.independent_relation(c[0].numpy(),s[0].numpy())[...,[0,7,8,9,10]]
        reference=audit.independent_projection(expected,model.position)
        with torch.inference_mode():actual=model.position(c,s)[0].numpy()
        np.testing.assert_allclose(actual,reference,atol=2e-4,rtol=2e-4)
        operator.append(dict(seed=seed,maximum_error=float(np.max(np.abs(actual-reference)))))
    selected.update(old_baseline_pairing=pairing,independent_five_input_operator=operator,goal_complete=False)
    write(OUT/'selected.json',selected)


def confirmation_released():
    """Scheduling dependency only: success and explicit failure both release GPUs."""
    identity=read(CONFIRMATION/'execution_identity.json');status=read(CONFIRMATION/'queue_status.json')
    if status['state'] not in ('complete','stopped'):return False
    path=Path(f"/proc/{identity['pid']}/stat")
    if path.exists():
        fields=path.read_text().rsplit(')',1)[1].split()
        if fields[19]==str(identity['start_ticks']) and fields[0]!='Z':return False
    for item in status.get('active',[]):
        command=Path(f"/proc/{item['pid']}/cmdline")
        if command.exists() and b'experiments.projection_confirmation' in command.read_bytes():return False
    return True


def available_gpus():
    output=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu',
        '--format=csv,noheader,nounits'],text=True,timeout=10);rows=[]
    for line in output.strip().splitlines():
        index,uuid,used,free,util=[x.strip() for x in line.split(',')]
        if int(used)<=512 and int(free)>=34*1024 and int(util)<=10:
            rows.append(dict(index=int(index),uuid=uuid,memory_mib=int(used),free_mib=int(free),utilization=int(util)))
    return rows


def queue():
    check();assert not (OUT/'execution_identity.json').exists()
    meta=dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],
        source_sha256=sha(Path(__file__)),lock_sha256=sha(OUT/'lock.json'),maximum_owned_gpu_jobs=3,
        started_unix=time.time(),goal_complete=False)
    write(OUT/'execution_identity.json',meta);active=[];completed=[];pending=[];stage='resource_dependency'
    def record(state,**extra):
        write(OUT/'queue_status.json',dict(meta,state=state,stage=stage,active=[r[1] for r in active],
            completed=completed,pending=pending,observed_unix=time.time(),**extra))
    def stop(signum,frame):raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def batch(action,jobs,gpu=True):
        nonlocal stage,pending
        stage=action;pending=list(jobs)
        while pending or active:
            for process,item,stream in list(active):
                code=process.poll()
                if code is not None:
                    stream.close();active.remove((process,item,stream));completed.append(dict(item,exit_code=code))
                    if code==75 and action=='train':pending.append(item['seed'])
                    elif code:raise RuntimeError(f"{item['log']}: exit {code}")
            choices=available_gpus() if gpu and pending else [None]
            busy={item['gpu']['uuid'] for _,item,_ in active if item['gpu'] is not None}
            for device in choices:
                if not pending or len(active)>=(3 if gpu else 1):break
                if device is not None and device['uuid'] in busy:continue
                seed=pending[0];cmd=[sys.executable,'-m',MODULE,action]
                if seed is not None:cmd+=['--seed',str(seed)]
                env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
                    CUBLAS_WORKSPACE_CONFIG=':4096:8',CUDA_DEVICE_ORDER='PCI_BUS_ID',
                    CUDA_VISIBLE_DEVICES='' if device is None else device['uuid'])
                probe=None
                if device is not None:
                    code='import numpy; import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count()==1; x=torch.ones(4,device="cuda"); assert float(x.sum())==4; torch.cuda.synchronize(); print(torch.cuda.get_device_name())'
                    probe=subprocess.check_output([sys.executable,'-c',code],env=env,text=True,timeout=60).strip()
                    if not any(r['uuid']==device['uuid'] for r in available_gpus()):continue
                pending.pop(0);log=OUT/f'{action}_{seed}_{time.time_ns()}.log';stream=log.open('x')
                process=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                item=dict(pid=process.pid,action=action,seed=seed,gpu=device,cuda_probe=probe,log=str(log),started_unix=time.time())
                active.append((process,item,stream))
                if device is not None:busy.add(device['uuid'])
                print('LAUNCHED',item,flush=True)
            record('running' if active else 'waiting_for_idle_gpu');time.sleep(5)
    try:
        while not confirmation_released():
            record('waiting_for_confirmation_resources',dependency_scientific_success_required=False);time.sleep(10)
        batch('preflight',[None]);batch('train',SEEDS);batch('selections',[None])
        batch('export',SEEDS);batch('report',[None],False)
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
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','queue','preflight','train','selections','export','report'])
    parser.add_argument('--seed',type=int,choices=SEEDS);args=parser.parse_args()
    if args.action in ('train','export'):
        bind()
        if args.action=='train':raise SystemExit(runner.fit('relative5',args.seed,OUT))
        runner.export(args.seed,OUT)
    elif args.action=='report':
        from experiments.round73_report import report
        report()
    else:globals()[args.action]()
