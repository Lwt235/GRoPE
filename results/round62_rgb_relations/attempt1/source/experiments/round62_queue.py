"""Owned process controller: observed idle GPUs, one frozen experiment."""
import os
import signal
import subprocess
import sys
import time
from experiments.round62_common import ROOT, OUT, NEW_ROLES, SEEDS, checked, read, write, sha

ACTIVE, DONE, INFO = {}, [], {}


def occupancy():
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu',
                             '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True, timeout=10)
    apps = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'],
                          check=True, capture_output=True, text=True, timeout=10)
    used = {line.split(',')[0].strip() for line in apps.stdout.splitlines() if ',' in line}
    rows = []
    for line in result.stdout.strip().splitlines():
        index, uuid, memory, utilization = [x.strip() for x in line.split(',')]
        rows.append(dict(index=int(index), uuid=uuid, memory_mib=int(memory), utilization=int(utilization),
                         free=uuid not in used and int(memory) < 16 and int(utilization) == 0))
    return rows


def record(state, **extra):
    write(OUT/'queue_status.json', dict(INFO, state=state, observed_unix=time.time(), completed=DONE,
                                     active=[x['meta'] for x in ACTIVE.values()], **extra))


def launch(stage, job, gpu):
    tag = stage if job is None else stage+'_'+'_'.join(map(str, job))
    if stage in ('selections','metrics'):
        command=[sys.executable,'-m','experiments.round62_audit',stage]
    elif stage.startswith('prepare_'):
        command=[sys.executable,'-m','experiments.round62_data',stage.removeprefix('prepare_')]
    else:
        command=[sys.executable,'-m','experiments.round62_run',stage]
        if stage=='train':command+=['--role',job[0],'--seed',str(job[1])]
        elif stage=='export':command+=['--seed',str(job[0])]
    env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
               CUBLAS_WORKSPACE_CONFIG=':4096:8', CUDA_DEVICE_ORDER='PCI_BUS_ID',
               CUDA_VISIBLE_DEVICES='' if gpu is None else gpu['uuid'])
    stream = (OUT/(tag+'.log')).open('x')
    try:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    except BaseException:
        stream.close()
        raise
    meta = dict(tag=tag, stage=stage, job=job, pid=process.pid, process_group=process.pid,
                command=command, gpu=gpu, started_unix=time.time())
    ACTIVE[tag] = dict(process=process, stream=stream, meta=meta)
    print('LAUNCHED', tag, process.pid, None if gpu is None else gpu['index'], flush=True)


def collect():
    failures = []
    for tag, item in list(ACTIVE.items()):
        code = item['process'].poll()
        if code is None:
            continue
        item['stream'].close()
        DONE.append(dict(item['meta'], exit_code=code, completed_unix=time.time()))
        del ACTIVE[tag]
        print('EXIT', tag, code, flush=True)
        if code:
            failures.append((tag, code))
    if failures:
        raise RuntimeError('child failure '+str(failures))


def stop_owned():
    for item in ACTIVE.values():
        if item['process'].poll() is None:
            os.killpg(item['process'].pid, signal.SIGTERM)
    for tag, item in list(ACTIVE.items()):
        try:
            code = item['process'].wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(item['process'].pid, signal.SIGKILL)
            code = item['process'].wait(timeout=5)
        item['stream'].close()
        DONE.append(dict(item['meta'], exit_code=code, stopped_by_driver=True, completed_unix=time.time()))
        del ACTIVE[tag]


def main():
    checked()
    if (OUT/'queue_status.json').exists():
        raise ValueError('inspect original controller before restart')
    if not read(OUT/'units.json')['passed']:
        raise ValueError('unit preflight required')
    INFO.update(pid=os.getpid(), started_unix=time.time(), lock_sha256=sha(OUT/'lock.json'))
    def stop(signum, frame):
        raise SystemExit(128+signum)
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    jobs = [(NEW_ROLES[(j+i) % len(NEW_ROLES)], seed) for j in range(len(NEW_ROLES)) for i, seed in enumerate(SEEDS)]
    phases = (('prepare_fit',[None]),('prepare_val',[None]),('preflight',[None]),
              ('train',jobs),('selections',[None]),('prepare_old_eval',[None]),
              ('export',[(s,) for s in SEEDS]),('metrics',[None]))
    try:
        for stage, pending in phases:
            pending, previous = list(pending), set()
            print('PHASE', stage, flush=True)
            while pending or ACTIVE:
                collect()
                if not pending and not ACTIVE:
                    break
                if stage in ('selections','metrics') or stage.startswith('prepare_'):
                    if pending:
                        launch(stage, pending.pop(0), None)
                    record('running', stage=stage, pending=pending)
                elif not pending or len(ACTIVE) >= 6:
                    previous = set()
                    record('running', stage=stage, pending=pending)
                else:
                    try:
                        rows = occupancy()
                    except (subprocess.SubprocessError, ValueError) as error:
                        previous = set()
                        record('observation_error', stage=stage, error=repr(error))
                        time.sleep(5)
                        continue
                    available = {x['uuid'] for x in rows if x['free']}
                    stable = available & previous
                    owned = {x['meta']['gpu']['uuid'] for x in ACTIVE.values()}
                    for gpu in rows:
                        if not pending or len(ACTIVE) >= 6:
                            break
                        if gpu['uuid'] in stable-owned:
                            launch(stage, pending.pop(0), gpu)
                            owned.add(gpu['uuid'])
                    previous = available-owned
                    record('running' if ACTIVE else 'waiting_for_gpu', stage=stage, pending=pending, occupancy=rows)
                time.sleep(5)
        record('complete', all_children_exit_zero=True, goal_complete=False)
        print('R62 experiment execution complete; report/review still required.', flush=True)
    except BaseException as error:
        stop_owned()
        record('stopped', error=repr(error), goal_complete=False)
        raise


if __name__ == '__main__':
    main()
