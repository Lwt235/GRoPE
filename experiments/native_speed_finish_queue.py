"""Finish the approved nine fits concurrently after completed exclusive tests."""
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from experiments import native_source_diversity as native
from experiments.native_source_diversity_queue import resources
from experiments.native_speed_transition import identity, live, checked_run
from experiments.native_speed_followup import OUT
from experiments.round65_common import ROOT, read, write, sha


def terminate_controller(record, sig):
    assert live(record)
    os.kill(record['pid'], sig)
    for _ in range(100):
        if not live(record): return
        time.sleep(.1)
    raise RuntimeError('Owned controller did not terminate.')


def run():
    native.check()
    amendment = read(OUT/'amendment.json')
    for path, digest in amendment['source_sha256'].items(): assert sha(path) == digest, path
    decision = read(OUT/'sphere/decision.json')
    assert sha(OUT/'sphere/result.json') == decision['result_sha256']
    assert read(OUT/'review.json')['engineering_review_complete']
    previous = read(OUT/'status.json')
    assert previous['state'] == 'waiting_for_current_fits' and previous['active_child'] is None
    assert len(previous['stages']) == 5 and all(s['exit_code'] == 0 for s in previous['stages'])
    assert previous['decision'] == decision['implementation']
    assert amendment['pending_at_hold'] == [['fixed_isotropic5', 33]]
    old_controller = previous['process']
    assert live(old_controller)
    assert not (OUT/'parallel_finish_amendment.json').exists()
    assert not (native.OUT/'runs/fixed_isotropic5/33/progress.json').exists()
    assert not (native.OUT/'selected.json').exists()
    # Stop only the idle observer, preserving every existing fit and its parent.
    os.kill(old_controller['pid'], signal.SIGSTOP)
    for _ in range(100):
        if identity(old_controller['pid'])['state'] == 'T': break
        time.sleep(.01)
    held = read(OUT/'status.json')
    if held['active_child'] is not None or held['state'] != 'waiting_for_current_fits':
        os.kill(old_controller['pid'], signal.SIGCONT)
        raise RuntimeError('Observer changed phase; no parallel handoff performed.')
    assert identity(old_controller['pid'])['state'] == 'T'
    shutil.copy2(OUT/'status.json', OUT/'before_parallel_finish_status.json')
    write(OUT/'parallel_finish_amendment.json', dict(created_unix=time.time(),
        reason='Exclusive tests have finished; use the existing authorized four-fit parallelism for the last pending fit while current fits drain.',
        retired_idle_observer=old_controller, training_workers_signalled=[],
        benchmark_decision_sha256=sha(OUT/'sphere/decision.json'),
        original_amendment_sha256=sha(OUT/'amendment.json'),
        original_training_lock_sha256=sha(native.OUT/'lock.json'),
        original_evaluation_lock_sha256=sha(native.OUT/'evaluation_lock.json'),
        source_sha256={str(Path(__file__).resolve()): sha(__file__)},
        scientific_configuration_unchanged=True, maximum_live_fits=4,
        evaluation_requires_all_nine_selected=True))
    terminate_controller(old_controller, signal.SIGKILL)
    state = dict(previous, process=identity(os.getpid()), started_unix=time.time(),
        state='running_remaining_and_draining_current', active_child=None, active_children=[],
        parallel_finish_amendment_sha256=sha(OUT/'parallel_finish_amendment.json'),
        original_current_fits=amendment['original_queue']['active'],
        pending=[['fixed_isotropic5', 33]], child_events=[],
        scientific_review_complete=False, goal_complete=False)
    done = {tuple(x) for x in amendment['original_queue']['completed']}
    for job in done: checked_run(*job)
    original = list(amendment['original_queue']['active'])
    pending = [dict(role='fixed_isotropic5', seed=33, optimized=True)]
    active = []
    sharded = set()

    def save(**updates):
        state.update(updates, updated_unix=time.time(),
            active_children=[item['record'] for item in active],
            pending=[[j['role'], j['seed']] for j in pending],
            verified_completed=[list(x) for x in sorted(done)])
        write(OUT/'status.json', state)

    def launch(job, card):
        module = 'experiments.native_speed_followup' if job['optimized'] else 'experiments.native_source_diversity'
        command = [sys.executable, '-u', '-m', module, 'train', '--role', job['role'], '--seed', str(job['seed'])]
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=card['uuid'], CUBLAS_WORKSPACE_CONFIG=':4096:8',
            PYTHONPATH=str(ROOT), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
        path = OUT/f"parallel_{job['role']}_{job['seed']}_{time.time_ns()}.log"
        stream = path.open('x')
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        record = dict(**job, process=identity(process.pid), gpu=card, log=str(path), command=command,
            started_unix=time.time(), runtime_implementation=decision['implementation'] if job['optimized'] else 'original')
        active.append(dict(job=job, record=record, process=process, stream=stream))
        save()
        print('PARALLEL_REMAINING_FIT_STARTED', job, process.pid, card['index'], flush=True)

    save()
    index = read(native.OUT/'active_execution.json')
    index.update(speed_transition_process=state['process'], speed_transition_entrypoint='experiments.native_speed_finish_queue',
        speed_transition_observed_unix=time.time(), parallel_finish_amendment=str(OUT/'parallel_finish_amendment.json'))
    index.pop('speed_transition_tool_session', None)
    write(native.OUT/'active_execution.json', index)
    try:
        while len(done) < 9:
            observed = []
            for item in original:
                key = (item['role'], item['seed'])
                if key in done or key in sharded: continue
                path = native.OUT/'runs'/key[0]/str(key[1])/'progress.json'
                if path.exists():
                    row = read(path); observed.append(dict(role=key[0], seed=key[1], epoch=row['epoch'], step=row['step']))
                if live(item): continue
                terminal = identity(item['pid'])
                code = None if terminal is None else os.waitstatus_to_exitcode(terminal['exit_status'])
                if code == 75:
                    sharded.add(key); pending.append(dict(role=key[0], seed=key[1], optimized=False))
                    continue
                if code not in (None, 0): raise RuntimeError(f'Original fit failed: {key}, code={code}')
                checked_run(*key); done.add(key)
                print('ORIGINAL_FIT_VERIFIED_COMPLETE', *key, flush=True)
            for item in active[:]:
                code = item['process'].poll()
                job = item['job']; key = (job['role'], job['seed'])
                path = native.OUT/'runs'/key[0]/str(key[1])/'progress.json'
                if path.exists():
                    row = read(path); observed.append(dict(role=key[0], seed=key[1], epoch=row['epoch'], step=row['step']))
                if code is None: continue
                item['stream'].close(); active.remove(item)
                state['child_events'].append(dict(item['record'], exit_code=code, finished_unix=time.time()))
                if code == 75: pending.append(job)
                elif code == 0:
                    checked_run(*key); done.add(key)
                    print('REMAINING_FIT_VERIFIED_COMPLETE', *key, flush=True)
                else: raise RuntimeError(f'Remaining fit failed: {key}, code={code}; preserve checkpoints.')
            if pending:
                cards, apps = resources()
                occupied = {a['uuid'] for a in apps}
                choices = [c for c in cards if c['uuid'] not in occupied and c['used_mib'] < 512
                           and c['free_mib'] >= 24576 and c['utilization'] < 10]
                live_original = sum(live(x) for x in original if (x['role'], x['seed']) not in done | sharded)
                if choices and live_original + len(active) < 4:
                    choices.sort(key=lambda c: (c['index'] == 3, c['free_mib']), reverse=True)
                    launch(pending.pop(0), choices[0])
            save(observed_progress=observed)
            if len(done) < 9: time.sleep(15)
        assert not active and not pending
        native.select_all()
        save(state='all_nine_selected')
        # Training workers are terminal before replacing their old parent and observer.
        post_state = read(native.OUT/'postprocess_status.json')
        assert post_state['state'] == 'waiting_for_training' and post_state['active_child'] is None
        for item in original: assert not live(item)
        for name in ('queue_status.json', 'postprocess_status.json', 'active_execution.json'):
            shutil.copy2(native.OUT/name, OUT/('before_parallel_handoff_'+name))
        terminate_controller(amendment['original_postprocess'], signal.SIGTERM)
        terminate_controller(amendment['original_controller'], signal.SIGKILL)
        write(OUT/'parallel_controller_handoff.json', dict(time_unix=time.time(),
            old_queue=amendment['original_controller'], old_postprocess=amendment['original_postprocess'],
            new_queue=state['process'], workers_signalled=[]))
        (native.OUT/'postprocess_status.json').unlink()
        queue = dict(amendment['original_queue'], pid=os.getpid(), state='ready_for_evaluation',
            started_unix=state['started_unix'], updated_unix=time.time(), active=[], pending=[],
            completed=[list(x) for x in sorted(done)], failures=[], all_models_selected=True,
            queue_source_sha256=sha(__file__), execution_amendment_sha256=sha(OUT/'parallel_finish_amendment.json'))
        write(native.OUT/'queue_status.json', queue)
        path = OUT/'parallel_resumed_postprocess.log'
        with path.open('x') as stream:
            post = subprocess.Popen([sys.executable, '-u', '-m', 'experiments.native_source_diversity_postprocess',
                '--queue-pid', str(os.getpid()), '--queue-start-ticks', str(state['process']['start_ticks'])],
                cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        save(state='waiting_for_evaluation', resumed_postprocess=identity(post.pid))
        write(native.OUT/'active_execution.json', dict(observed_unix=time.time(),
            processes=[dict(name='parallel_finish_queue', **state['process']), dict(name='postprocess', **identity(post.pid))],
            queue_status_file=str(native.OUT/'queue_status.json'), speed_status_file=str(OUT/'status.json'),
            previous_index=str(OUT/'before_parallel_handoff_active_execution.json'), goal_complete=False))
        while post.poll() is None: save(); time.sleep(30)
        assert post.returncode == 0 and read(native.OUT/'postprocess_status.json')['state'] == 'ready_for_scientific_review'
        save(state='ready_for_scientific_review', finished_unix=time.time())
        print('SOURCE_DIVERSITY_COMPLETE_SCIENTIFIC_REVIEW_PENDING', flush=True)
    except BaseException as error:
        save(state='transition_failed', error_type=type(error).__name__, reason=str(error))
        raise


if __name__ == '__main__':
    run()
