"""Bounded R48 process controller; only idle GPUs, only owned children stopped."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/round48_modelnet/attempt1'
ACTIVE, DONE, INFO = {}, [], {}


def record(state, **extra):
    value = dict(INFO, state=state, observed_unix=time.time(), completed=DONE,
                 active=[x['meta'] for x in ACTIVE.values()], **extra)
    temporary = OUT/'queue_status.tmp'
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(OUT/'queue_status.json')


def occupancy():
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu',
                             '--format=csv,noheader,nounits'], check=True, capture_output=True,
                            text=True, timeout=10)
    apps = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                           '--format=csv,noheader,nounits'], check=True, capture_output=True,
                          text=True, timeout=10)
    used = {line.split(',')[0].strip() for line in apps.stdout.splitlines() if ',' in line}
    rows = []
    for line in result.stdout.strip().splitlines():
        index, uuid, memory, utilization = [v.strip() for v in line.split(',')]
        rows.append(dict(index=int(index), uuid=uuid, memory_mib=int(memory),
                         utilization=int(utilization),
                         free=uuid not in used and int(memory) < 16 and int(utilization) == 0))
    return rows


def launch(stage, seed, gpu):
    tag = stage if seed is None else f'{stage}_{seed}'
    if stage == 'data':
        command = [sys.executable, '-m', 'experiments.round48_data']
    elif stage in ('training_audit', 'metrics'):
        command = [sys.executable, '-m', 'experiments.round48_audit',
                   'training' if stage == 'training_audit' else 'metrics']
    else:
        command = [sys.executable, '-m', 'experiments.round48_train', stage]
        if seed is not None:
            command += ['--seeds', str(seed)]
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
               CUBLAS_WORKSPACE_CONFIG=':4096:8')
    if gpu is not None:
        env.update(CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=gpu['uuid'])
    else:
        env['CUDA_VISIBLE_DEVICES'] = ''
    stream = (OUT/(tag+'.log')).open('x')
    try:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    except BaseException:
        stream.close()
        raise
    meta = dict(tag=tag, stage=stage, seed=seed, pid=process.pid, process_group=process.pid,
                command=command, gpu=gpu, started_unix=time.time())
    ACTIVE[tag] = dict(process=process, stream=stream, meta=meta)
    print('LAUNCHED', tag, 'pid', process.pid, 'gpu', None if gpu is None else gpu['index'], flush=True)


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
        DONE.append(dict(item['meta'], exit_code=code, stopped_by_driver=True,
                         completed_unix=time.time()))
        del ACTIVE[tag]


def main():
    if (OUT/'queue_status.json').exists():
        raise ValueError('inspect existing controller handle; never restart blindly')
    if not json.loads((OUT/'preflight.json').read_text())['passed']:
        raise ValueError('CPU checks required')
    INFO.update(pid=os.getpid(), started_unix=time.time(),
                lock_sha256=(OUT/'lock.sha256').read_text().strip())
    def stop(signum, frame):
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    phases = (('smoke', [None]), ('data', [None]), ('train', [11, 22, 33, 44, 55]),
              ('training_audit', [None]), ('export', [11, 22, 33, 44, 55]), ('metrics', [None]))
    try:
        for stage, seeds in phases:
            pending, previous, idle_started = list(seeds), set(), None
            print('PHASE', stage, flush=True)
            while pending or ACTIVE:
                collect()
                if not pending and not ACTIVE:
                    break
                if stage in ('data', 'training_audit', 'metrics'):
                    if pending:
                        launch(stage, pending.pop(0), None)
                    record('running', stage=stage, pending=pending)
                else:
                    try:
                        rows = occupancy()
                    except (subprocess.SubprocessError, ValueError) as error:
                        previous = set()
                        record('observation_error', stage=stage, error=repr(error))
                        time.sleep(5)
                        continue
                    available = {r['uuid'] for r in rows if r['free']}
                    stable = available & previous
                    own = {item['meta']['gpu']['uuid'] for item in ACTIVE.values()}
                    for gpu in rows:
                        if not pending or len(ACTIVE) >= 5:
                            break
                        if gpu['uuid'] in stable-own:
                            launch(stage, pending.pop(0), gpu)
                            own.add(gpu['uuid'])
                    previous = available-own
                    record('running' if ACTIVE else 'waiting_for_gpu', stage=stage,
                           pending=pending, occupancy=rows)
                if pending and not ACTIVE:
                    if idle_started is None:
                        idle_started = time.monotonic()
                    if time.monotonic()-idle_started > 3600:
                        raise TimeoutError('one hour without an available GPU')
                else:
                    idle_started = None
                time.sleep(5)
        record('complete', all_children_exit_zero=True, goal_complete=False)
        print('R48 execution complete; task-specific development evidence only.', flush=True)
    except BaseException as error:
        stop_owned()
        record('stopped', error=repr(error), goal_complete=False)
        raise


if __name__ == '__main__':
    main()
