"""Wait on the actual training controller, then run the sealed evaluation once."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.native_source_diversity_evaluation import OUT, check_evaluation_lock, selected_models
from experiments.native_source_diversity_queue import resources
from experiments.round65_common import ROOT, read, write, sha


def process_identity(pid):
    try:
        values = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    except FileNotFoundError:
        return None
    return dict(pid=pid, state=values[0], start_ticks=int(values[19]))


def run(queue_pid, queue_start_ticks):
    check_evaluation_lock()
    target = OUT / 'postprocess_status.json'
    assert not target.exists(), 'Inspect the existing postprocessor before any recovery.'
    state = dict(process=process_identity(os.getpid()), state='waiting_for_training', started_unix=time.time(),
        training_controller=dict(pid=queue_pid, start_ticks=queue_start_ticks),
        evaluation_lock_sha256=sha(OUT / 'evaluation_lock.json'), active_child=None,
        scientific_review_complete=False, goal_complete=False)

    def save(**updates):
        state.update(updates, updated_unix=time.time())
        write(target, state)

    def child(command, environment, phase):
        log_path = OUT / f'postprocess_{phase}_{time.time_ns()}.log'
        with log_path.open('x') as log:
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
            save(state=phase, active_child=dict(command=command, process=process_identity(process.pid), log=str(log_path)))
            print('SOURCE_DIVERSITY_POSTPROCESS', phase, process.pid, flush=True)
            while process.poll() is None:
                save()
                time.sleep(15)
            code = process.returncode
        if code:
            raise RuntimeError(f'{phase} exited {code}; inspect {log_path}, preserve completed work.')
        save(active_child=None)

    save()
    print('WAITING_FOR_SOURCE_DIVERSITY_QUEUE', queue_pid, queue_start_ticks, flush=True)
    try:
        while True:
            queue = read(OUT / 'queue_status.json')
            assert queue['pid'] == queue_pid
            if queue['state'] == 'ready_for_evaluation':
                assert queue['all_models_selected'] and not queue['active'] and not queue['pending'] and not queue['failures']
                selected_models()
                break
            if queue['state'] in ('failed', 'controller_failed', 'draining_after_failure'):
                raise RuntimeError('Training reported an execution failure: ' + queue['state'])
            # A timeout is not terminal. Only a missing/replaced/terminal process
            # or an explicit queue failure ends this wait; no job is restarted.
            try:
                actual = process_identity(queue_pid)
            except PermissionError as error:
                save(observation_error=str(error))
                time.sleep(30)
                continue
            if actual is None or actual['start_ticks'] != queue_start_ticks or actual['state'] in ('Z', 'X'):
                # The controller may have committed completion between our two reads.
                queue = read(OUT / 'queue_status.json')
                if queue['state'] == 'ready_for_evaluation':
                    continue
                raise RuntimeError('The recorded training controller is terminal or missing; do not restart automatically.')
            save(last_verified_training_process=actual, training_completed=len(queue['completed']),
                 training_active=len(queue['active']), training_pending=len(queue['pending']))
            time.sleep(30)
        save(state='waiting_for_evaluation_gpu')
        while True:
            try:
                cards, apps = resources()
            except subprocess.CalledProcessError as error:
                save(resource_observation_error=str(error))
                time.sleep(30)
                continue
            occupied = {a['uuid'] for a in apps}
            available = [c for c in cards if c['free_mib'] >= 8192]
            available.sort(key=lambda c: (c['uuid'] not in occupied and c['used_mib'] < 512 and c['utilization'] < 10,
                                           -c['utilization'], c['free_mib']), reverse=True)
            if available:
                card = available[0]
                break
            save(last_resource_check_unix=time.time(), cards=cards)
            time.sleep(30)
        environment = os.environ.copy()
        environment.update(CUDA_VISIBLE_DEVICES=card['uuid'], CUBLAS_WORKSPACE_CONFIG=':4096:8',
                           OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', PYTHONPATH=str(ROOT))
        save(evaluation_gpu=card, evaluation_shared=card['uuid'] in occupied, allocator_cap_bytes=4 * 2**30)
        prefix = [sys.executable, '-u', '-m', 'experiments.native_source_diversity_evaluation']
        child(prefix + ['export_all'], environment, 'exporting')
        child(prefix + ['report'], environment, 'summarizing')
        child([sys.executable, '-u', '-m', 'experiments.native_source_diversity_review'], environment, 'independent_arithmetic')
        save(state='ready_for_scientific_review', finished_unix=time.time(),
             result_sha256=sha(OUT / 'result.json'), independent_checks_sha256=sha(OUT / 'independent_checks.json'))
        print('SOURCE_DIVERSITY_READY_FOR_SCIENTIFIC_REVIEW', flush=True)
        return 0
    except BaseException as error:
        save(state='postprocess_failed', error_type=type(error).__name__, reason=str(error), finished_unix=time.time())
        # Preserve any running child handle; never kill jobs during observer recovery.
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--queue-pid', type=int, required=True)
    parser.add_argument('--queue-start-ticks', type=int, required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.queue_pid, args.queue_start_ticks))
