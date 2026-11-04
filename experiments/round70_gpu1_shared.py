"""User-authorized GPU 1 sharing; original R70 scientific programs stay frozen.

Only the scheduler changes: one owned GPU child at a time, GPU 1 only, and at
least 12 GiB free before launch. Existing external processes are left intact.
Completed fits are verified and retained. No optimizer, seed, input stream,
checkpoint selection, or evaluation configuration is changed.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np

from experiments.round65_common import read, write, sha
from experiments.round70_nvs_data import OUT, ROOT
from experiments.round70_nvs_run import check, ROLES, SEEDS

REV = OUT/'execution_shared_gpu1'
GPU_UUID = 'GPU-854e2059-9e85-04cc-fdb0-5735913714fe'
MIN_FREE_MIB = 12288


def gpu_status():
    text = subprocess.check_output([
        'nvidia-smi', '--id='+GPU_UUID,
        '--query-gpu=index,uuid,memory.used,memory.free,memory.total,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True, timeout=10).strip()
    index, uuid, used, free, total, utilization = [x.strip() for x in text.split(',')]
    assert int(index)==1 and uuid==GPU_UUID
    return dict(index=1,uuid=uuid,memory_mib=int(used),free_mib=int(free),
                total_mib=int(total),utilization=int(utilization),shared_allowed=True)


def frozen_jobs():
    return [(ROLES[(i+j)%len(ROLES)],seed)
            for j in range(len(ROLES)) for i,seed in enumerate(SEEDS)]


def retained_fits():
    check()
    rows=[];pending=[]
    for role,seed in frozen_jobs():
        folder=OUT/'runs'/role/str(seed)
        if not folder.exists():
            pending.append((role,seed));continue
        if not (folder/'run.json').exists():
            raise RuntimeError(f'Existing unfinished fit must be allowed to finish or inspected: {folder}')
        run=read(folder/'run.json');history=read(folder/'history.json')
        assert run['role']==role and run['seed']==seed
        assert run['steps']==2160 and run['epochs']==60 and run['finite_training']
        assert len(history['training'])==60 and len(history['validation'])==20
        assert run['lock_sha256']==sha(OUT/'lock.json')
        assert run['selected_sha256']==sha(folder/'best.pt')
        assert run['terminal_sha256']==sha(folder/'terminal.pt')
        rows.append(run)
    return rows,pending


def main():
    retained,pending=retained_fits()
    if (REV/'coordinator_identity.json').exists():
        raise RuntimeError('Inspect the existing shared coordinator instead of duplicating it')
    assert read(REV/'handoff_complete.json')['owned_old_controllers_stopped']
    old=read(REV/'original_queue_after_stop.json')
    assert not old['active']
    completed=list(old['completed'])
    existing={tuple(x['job']) for x in completed if x['stage']=='train' and x['exit_code']==0}
    for row in retained:
        pair=(row['role'],row['seed'])
        if pair not in existing:
            completed.append(dict(tag='retained_'+row['role']+'_'+str(row['seed']),
                stage='train',job=list(pair),exit_code=0,retained_verified_fit=True,
                selected_sha256=row['selected_sha256']))
    meta=dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],
        started_unix=time.time(),source_sha256=sha(Path(__file__)),lock_sha256=sha(OUT/'lock.json'),
        execution_policy='user-authorized GPU1 shared serial',maximum_owned_gpu_jobs=1,
        gpu_uuid=GPU_UUID,min_free_mib=MIN_FREE_MIB,goal_complete=False)
    write(REV/'coordinator_identity.json',meta)
    identity=read(OUT/'execution_identity.json')
    for key in list(identity):
        if key.startswith('geometry_diagnostic_') or key=='controller_exec_session':
            del identity[key]
    identity.update(controller_pid=meta['pid'],controller_start_ticks=meta['start_ticks'],
        max_parallel_owned_gpu_jobs=1,execution_policy=meta['execution_policy'],
        execution_revision=str(REV),integrated_postprocessing=True)
    write(OUT/'execution_identity.json',identity)
    child=None;active=[];stage='initializing'

    def record(state,**extra):
        row=dict(meta,state=state,stage=stage,observed_unix=time.time(),
                 completed=completed,active=active,pending=pending,**extra)
        write(OUT/'queue_status.json',row)
        finish=dict(meta,state=state,stage=stage,
            observed_unix=time.time(),integrated_postprocessing=True,
            scientific_review_pending=True,visual_review_pending=True)
        finish.update(extra)
        write(OUT/'finish_status.json',finish)

    def stop(signum,frame):
        raise SystemExit(128+signum)

    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)

    def run(module,action,job=None,use_gpu=True):
        nonlocal child,active,stage
        stage=action
        gpu=None
        if use_gpu:
            while True:
                try:
                    gpu=gpu_status()
                except (subprocess.SubprocessError,ValueError) as error:
                    record('observation_error',error=str(error));time.sleep(5);continue
                if gpu['free_mib']>=MIN_FREE_MIB:
                    break
                record('waiting_for_gpu1_memory',occupancy=[gpu]);time.sleep(5)
        command=[sys.executable,'-m',module]
        if action!='geometry_diagnostic':command.append(action)
        if action=='train':command+=['--role',job[0],'--seed',str(job[1])]
        elif action=='export':command+=['--seed',str(job[0])]
        env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
            CUBLAS_WORKSPACE_CONFIG=':4096:8',CUDA_DEVICE_ORDER='PCI_BUS_ID',
            CUDA_VISIBLE_DEVICES=GPU_UUID if use_gpu else '')
        tag='shared_'+action+('' if job is None else '_'+'_'.join(map(str,job)))+'_'+str(time.time_ns())
        with (OUT/(tag+'.log')).open('x') as stream:
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,
                                   stderr=subprocess.STDOUT,start_new_session=True)
            item=dict(tag=tag,stage=action,job=job,pid=child.pid,gpu=gpu,
                command=command,started_unix=time.time(),resource_sharing=True)
            active=[item];record('running',pending_training=pending)
            print('LAUNCHED',tag,child.pid,'GPU1' if use_gpu else 'CPU',flush=True)
            while child.poll() is None:
                if use_gpu:
                    try:
                        sample=dict(unix=time.time(),child_pid=child.pid,stage=stage,**gpu_status())
                        with (REV/'resource_samples.jsonl').open('a') as out:
                            out.write(json.dumps(sample)+'\n')
                    except (subprocess.SubprocessError,ValueError):
                        pass  # A failed observation does not terminate or duplicate the live child.
                record('running',pending_training=pending);time.sleep(5)
            code=child.returncode
        completed.append(dict(item,exit_code=code,completed_unix=time.time()))
        active=[];child=None
        print('EXIT',tag,code,flush=True)
        if code:raise RuntimeError(f'{tag} failed with {code}; preserve and inspect artifacts')

    try:
        write(REV/'retained_fits.json',dict(rows=retained,pending=pending,
            original_lock_unchanged=True,unix=time.time()))
        while pending:
            job=pending.pop(0)
            run('experiments.round70_nvs_run','train',job)
        run('experiments.round70_nvs_run','selections',use_gpu=False)
        run('experiments.round70_geometry_diagnostic','geometry_diagnostic',use_gpu=False)
        for seed in SEEDS:
            run('experiments.round70_nvs_run','export',(seed,))
        for action in ('baselines','summarize','audit'):
            run('experiments.round70_nvs_report',action,use_gpu=action!='summarize')
        stage='complete'
        record('complete',task_execution_completed=True,scientific_review_pending=True,
               visual_review_pending=True)
    except BaseException as error:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL);child.wait()
        record('stopped',error_type=type(error).__name__,error=str(error))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    if args.check_only:
        rows,pending=retained_fits()
        print(json.dumps(dict(retained=len(rows),pending=pending,lock_unchanged=True)))
    else:
        main()
