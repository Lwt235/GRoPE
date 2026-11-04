"""Finish the sealed regional evaluation on GPUs with spare capacity.

Scheduling-only amendment: all child computations remain in the original,
hash-checked implementation. Preserve the stopped coordinator's lineage.
"""
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from experiments.region_geometry_controls import OUT, ROOT, check
from experiments.round65_common import read, write, sha

AMENDMENT = OUT / 'evaluation_resource_update_02'
MODULE = 'experiments.region_geometry_controls'


def available():
    output = subprocess.check_output([
        'nvidia-smi', '--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True, timeout=10)
    choices = []
    for line in output.strip().splitlines():
        index, uuid, used, free, util = (x.strip() for x in line.split(','))
        if int(free) >= 24576:
            choices.append(dict(index=int(index), uuid=uuid, memory_mib=int(used),
                                free_mib=int(free), utilization=int(util)))
    return sorted(choices, key=lambda x: (-x['free_mib'], x['utilization']))


def main():
    check()
    plan = read(AMENDMENT / 'plan.json')
    assert plan['controller_sha256'] == sha(Path(__file__))
    assert plan['selected_sha256'] == sha(OUT / 'selected.json')
    old = read(OUT / 'queue_status.json')
    assert old['stage'] == 'export' and old['state'] == 'stopped'
    assert not old['active']
    assert old['pending'] == [{'seed': 11}, {'seed': 22}, {'seed': 33}]
    path = Path(f"/proc/{old['pid']}/stat")
    if path.exists():
        fields = path.read_text().rsplit(')', 1)[1].split()
        assert fields[19] != str(old['start_ticks']) or fields[0] == 'Z'
    assert not (AMENDMENT / 'execution_identity.json').exists()
    assert not list((OUT / 'evaluation').glob('*/*/record.json'))
    meta = dict(pid=os.getpid(), start_ticks=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19],
        started_unix=time.time(), source_sha256=sha(Path(__file__)),
        lock_sha256=sha(OUT / 'lock.json'), resource_amendment_sha256=sha(AMENDMENT / 'plan.json'),
        previous_execution_identity_sha256=sha(AMENDMENT / 'original_execution_identity.json'),
        maximum_owned_gpu_jobs=2, goal_complete=False)
    write(AMENDMENT / 'execution_identity.json', meta)
    write(OUT / 'execution_identity.json', meta)
    completed, pending, active, stage = list(old['completed']), list(old['pending']), [], 'export'

    def record(state, **extra):
        write(OUT / 'queue_status.json', dict(meta, state=state, stage=stage,
            completed=completed, pending=pending, active=[x[1] for x in active],
            observed_unix=time.time(), **extra))

    def stop(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    previous = set()
    try:
        while pending or active:
            for process, item, stream in list(active):
                code = process.poll()
                if code is not None:
                    stream.close()
                    active.remove((process, item, stream))
                    completed.append(dict(item, exit_code=code))
                    if code:
                        raise RuntimeError(f"{item['log']}: exit {code}")
            choices = available() if pending else []
            current = {x['uuid'] for x in choices}
            stable, previous = current & previous, current
            occupied = {x[1]['gpu']['uuid'] for x in active}
            for device in choices:
                if not pending or len(active) >= 2:
                    break
                if device['uuid'] not in stable or device['uuid'] in occupied:
                    continue
                env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
                    CUBLAS_WORKSPACE_CONFIG=':4096:8', CUDA_DEVICE_ORDER='PCI_BUS_ID',
                    CUDA_VISIBLE_DEVICES=device['uuid'])
                probe_code = 'import numpy; import torch; assert torch.cuda.is_available(); x=torch.ones(4,device="cuda"); assert float(x.sum())==4; torch.cuda.synchronize(); print(torch.cuda.get_device_name())'
                probe = subprocess.check_output([sys.executable, '-c', probe_code], env=env, text=True, timeout=60).strip()
                if not any(x['uuid'] == device['uuid'] for x in available()):
                    continue
                seed = pending.pop(0)['seed']
                # An allocator ceiling is only a resource bound, not a model change.
                code = ('import numpy; import torch; import runpy; import sys; '
                        'torch.cuda.set_per_process_memory_fraction(8*2**30/torch.cuda.get_device_properties(0).total_memory); '
                        f'sys.argv=["{MODULE}","export","--seed","{seed}"]; '
                        f'runpy.run_module("{MODULE}",run_name="__main__")')
                log = OUT / f'export_resource_update_{seed}_{time.time_ns()}.log'
                stream = log.open('x')
                process = subprocess.Popen([sys.executable, '-c', code], cwd=ROOT, env=env,
                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                item = dict(action='export', seed=seed, pid=process.pid, gpu=device, log=str(log),
                    cuda_probe=probe, allocator_ceiling_gib=8, started_unix=time.time())
                active.append((process, item, stream))
                occupied.add(device['uuid'])
                print('LAUNCHED', item, flush=True)
            record('running' if active else 'waiting_for_spare_gpu_capacity')
            if pending or active:
                time.sleep(5)
        stage = 'report'
        record('running')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
        subprocess.run([sys.executable, '-m', MODULE, 'report'], cwd=ROOT, env=env, check=True)
        assert plan['selected_sha256'] == sha(OUT / 'selected.json')
        stage = 'complete'
        record('complete', scientific_review_pending=True, visual_review_pending=True)
    except BaseException as error:
        for process, item, stream in active:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            stream.close()
        record('stopped', error=str(error))
        raise


if __name__ == '__main__':
    main()
