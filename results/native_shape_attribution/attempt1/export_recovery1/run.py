"""Finish only missing exports after the original controller disappeared."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np
from experiments import native_shape_attribution as experiment
from experiments.round65_common import ROOT, VARIANTS, read, sha, write

OUT = Path(__file__).resolve().parent
BASE = experiment.OUT
WORKER = BASE / 'resource_handoff1/run.py'


def main():
    experiment.check()
    assert not (OUT / 'execution_identity.json').exists()
    old = read(BASE / 'queue_status.json')
    # The real host /proc is required here; run in the GPU execution context.
    for pid in {old['pid'], *(r['pid'] for r in old.get('active', []))}:
        assert not Path(f'/proc/{pid}').exists(), f'Previous process {pid} remains live'
    write(OUT / 'interrupted_status.json', old)
    selected = read(BASE / 'selected.json')
    assert selected['all_three_selected_before_development_evaluation']
    assert len(selected['records']) == 3
    for seed in (11, 22, 33):
        run = read(BASE / f'runs/fixed_isotropic5/{seed}/run.json')
        assert (run['epochs'], run['steps']) == (120, 46800)
    done = BASE / 'evaluation/fixed_isotropic5/11'
    record = read(done / 'record.json')
    assert record['selection_sha256'] == sha(BASE / 'selected.json')
    assert [r['variant'] for r in record['records']] == list(VARIANTS)
    for row in record['records']:
        assert sha(done / (row['variant'] + '.npz')) == row['sha256']
    assert all(not (BASE / f'evaluation/fixed_isotropic5/{seed}').exists() for seed in (22, 33))
    assert not (BASE / 'result.json').exists()
    identity = dict(pid=os.getpid(), started_unix=time.time(),
        start_ticks=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19],
        source_sha256=sha(__file__), worker_sha256=sha(WORKER),
        lock_sha256=sha(BASE / 'lock.json'), selection_sha256=sha(BASE / 'selected.json'),
        preserved_seed11_record_sha256=sha(done / 'record.json'),
        recovery_reason='Controller and export process absent; terminal session missing; status stale. Cause unknown.',
        actions=['export22', 'export33', 'report'], new_fits=0,
        science_or_selection_changed=False, fresh_confirmation=False, goal_complete=False)
    write(OUT / 'execution_identity.json', identity)
    completed = []
    active = None

    def status(state, stage, **extra):
        value = dict(identity, state=state, stage=stage, completed=completed,
                     observed_unix=time.time(), **extra)
        write(OUT / 'queue_status.json', value)
        write(BASE / 'queue_status.json', value)

    def interrupted(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        for action, seed in [('export', 22), ('export', 33), ('report', None)]:
            device = None
            env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                       OPENBLAS_NUM_THREADS='4', CUBLAS_WORKSPACE_CONFIG=':4096:8',
                       CUDA_DEVICE_ORDER='PCI_BUS_ID', PYTHONPATH=str(ROOT))
            if action == 'export':
                choices = experiment.parent.available_gpus()
                assert choices, 'No whole-idle GPU; preserve all completed work'
                device = choices[0]
                env['CUDA_VISIBLE_DEVICES'] = device['uuid']
                probe = 'import numpy; import torch; assert torch.cuda.is_available() and torch.cuda.device_count()==1; x=torch.ones(4,device="cuda"); assert float(x.sum())==4; torch.cuda.synchronize(); print(torch.cuda.get_device_name())'
                probe_output = subprocess.check_output([sys.executable, '-c', probe], env=env, text=True, timeout=45).strip()
                assert any(r['uuid'] == device['uuid'] for r in experiment.parent.available_gpus())
            else:
                env['CUDA_VISIBLE_DEVICES'] = ''
                probe_output = None
            command = [sys.executable, '-u', str(WORKER), 'worker', '--action', action]
            if seed is not None:
                command += ['--seed', str(seed)]
            logfile = OUT / f'{action}_{seed}.log'
            with logfile.open('x') as stream:
                active = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
                                          stderr=subprocess.STDOUT, start_new_session=True)
                item = dict(action=action, seed=seed, pid=active.pid, gpu=device,
                            cuda_probe=probe_output, log=str(logfile))
                print('RECOVERY_STARTED', item, flush=True)
                while active.poll() is None:
                    status('running', action, active=[item])
                    try:
                        active.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        pass
                code = active.returncode
                completed.append(dict(item, exit_code=code))
                active = None
                assert code == 0, f'{action} {seed} exited {code}; see {logfile}'
        assert sha(done / 'record.json') == identity['preserved_seed11_record_sha256']
        status('complete', 'complete', active=[], scientific_review_pending=True, visual_review_pending=True)
        print('NATIVE_EXPORT_RECOVERY_COMPLETE', flush=True)
    except BaseException as error:
        if active is not None and active.poll() is None:
            os.killpg(active.pid, signal.SIGTERM)
            active.wait(timeout=15)
        status('stopped', 'recovery_failure', error=repr(error), active=[])
        raise


if __name__ == '__main__':
    main()
