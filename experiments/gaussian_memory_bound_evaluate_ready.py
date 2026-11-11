"""Evaluate finished seed-11 fits while its last fit is still training.

The ordinary primary evaluation queue is temporarily stopped, then always
continued, so two evaluators cannot write the same case concurrently. This
only changes scheduling; the frozen evaluator and scientific cases are reused.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/gaussian_memory/bound_read1'


def read(name):
    return json.loads((OUT / name).read_text())


def process_identity(pid):
    folder = Path('/proc') / str(pid)
    command = (folder / 'cmdline').read_bytes().replace(b'\0', b' ').decode()
    # The process name can contain spaces; fields following its final ')' are stable.
    fields = (folder / 'stat').read_text().rsplit(')', 1)[1].split()
    return command, fields[19], fields[0]


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--gpu', default='6'); args = parser.parse_args()
    assert read('evaluation_seed_20261110_complete.json')['complete']
    assert read('evaluation_seed_20261112_complete.json')['complete']
    assert not read('train_primary_status.json')['complete']
    primary = read('evaluation_queue_primary.json')
    assert primary['stage'] == 'waiting_for_dependencies' and primary['waiting_for'] == ['train_primary_status.json']
    pid = primary['pid']; command, start_id, state = process_identity(pid)
    assert 'experiments.gaussian_memory_bound_queue primary --gpu 1' in command and state != 'T'
    seed = 20261111
    assert all(read(f'{kind}_{seed}/status.json')['complete'] for kind in ('local', 'content'))
    record = {'pid': os.getpid(), 'primary_queue_pid': pid, 'primary_queue_start_id': start_id,
              'gpu': args.gpu, 'seed': seed, 'scientific_change': False, 'stages': [],
              'started_unix': time.time(), 'primary_queue_resumed': False, 'complete': False}

    def save():
        record['updated_unix'] = time.time()
        p = OUT / 'ready_evaluation.json'; temp = p.with_suffix('.tmp')
        temp.write_text(json.dumps(record, indent=2) + '\n'); temp.replace(p)

    def launch(stage, kind):
        assert read(f'{kind}_{seed}/status.json')['complete']
        label = f'{stage}_{kind}_{seed}'; logpath = OUT / f'ready_evaluation_{label}.log'
        cmd = [sys.executable, '-u', '-m', 'experiments.gaussian_memory_launch', '--gpu', args.gpu,
               '--allow-shared', 'experiments.gaussian_memory_bound_evaluate', stage,
               '--kind', kind, '--seed', str(seed)]
        item = {'stage': stage, 'kind': kind, 'command': cmd, 'log': str(logpath), 'started_unix': time.time()}
        record['stages'].append(item); save()
        with logpath.open('a', buffering=1) as log:
            child = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            item['pid'] = child.pid; save(); code = child.wait()
        item.update(returncode=code, finished_unix=time.time()); save()
        print(json.dumps({'ready_evaluation_stage': label, 'returncode': code}), flush=True)
        if code:
            raise RuntimeError(f'{label} exited {code}; ordinary queue will resume and use completed clips.')

    save()
    os.kill(pid, signal.SIGSTOP)
    try:
        for _ in range(30):
            if process_identity(pid)[2] == 'T':
                break
            time.sleep(.1)
        assert process_identity(pid)[2] == 'T'
        record['primary_queue_stopped'] = True; save()
        launch('static', 'local'); launch('static', 'content')
        coverage = OUT / f'coverage_{seed}/status.json'
        if coverage.exists() and json.loads(coverage.read_text())['complete']:
            launch('static', 'coverage'); launch('integrated', 'coverage')
        record['complete'] = True
    except BaseException as error:
        record['error'] = repr(error)
        raise
    finally:
        current_command, current_start, _ = process_identity(pid)
        assert current_command == command and current_start == start_id
        os.kill(pid, signal.SIGCONT)
        record['primary_queue_resumed'] = True; save()


if __name__ == '__main__':
    main()
