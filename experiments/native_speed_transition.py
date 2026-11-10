"""Drain existing fits, benchmark on an idle GPU, then resume the held queue."""
import argparse
import ast
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from experiments.round65_common import ROOT, read, write, sha
from experiments import native_source_diversity as native
from experiments.native_source_diversity_queue import resources
from experiments.native_speed_followup import OUT, paired_summary


def identity(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    except FileNotFoundError:
        return None
    return dict(pid=pid, state=fields[0], start_ticks=int(fields[19]),
                exit_status=int(fields[49]) if len(fields) > 49 else None)


def live(record):
    actual = identity(record['pid'])
    if actual is None: return False
    if actual['start_ticks'] != record['start_ticks']:
        raise RuntimeError(f"PID identity changed: {record['pid']}")
    return actual['state'] not in ('Z', 'X')


def checked_run(role, seed):
    folder = native.OUT/'runs'/role/str(seed)
    row = read(folder/'run.json')
    assert (row['epochs'], row['steps'], row['optimizer_updates']) == (120, 46800, 46800)
    assert row['lock_sha256'] == sha(native.OUT/'lock.json')
    for name, digest in row['hashes'].items(): assert sha(folder/name) == digest, name
    return row


def preparation():
    native.check()
    assert not (OUT/'amendment.json').exists()
    queue = read(native.OUT/'queue_status.json')
    controller = identity(queue['pid'])
    assert controller and controller['pid'] == 65847 and controller['start_ticks'] == 1447823218
    assert controller['state'] == 'T'
    post = read(native.OUT/'postprocess_status.json')
    assert post['state'] == 'waiting_for_training' and post['active_child'] is None
    post_identity = identity(post['process']['pid'])
    assert post_identity['start_ticks'] == post['process']['start_ticks']
    for record, module in ((controller, 'experiments.native_source_diversity_queue'),
                           (post_identity, 'experiments.native_source_diversity_postprocess')):
        assert module.encode() in Path(f"/proc/{record['pid']}/cmdline").read_bytes()
    assert queue['pending'] == [['fixed_isotropic5', 33]] and not queue['failures']
    paths = [Path(__file__), ROOT/'experiments/native_speed_followup.py']
    found = set()
    while paths:
        path = paths.pop().resolve()
        if path in found: continue
        found.add(path)
        for node in ast.walk(ast.parse(path.read_text())):
            names = ([n.name for n in node.names] if isinstance(node, ast.Import) else
                     [node.module]+[node.module+'.'+n.name for n in node.names]
                     if isinstance(node, ast.ImportFrom) and node.module else [])
            for name in names:
                source = ROOT/(name.replace('.', '/')+'.py')
                if name.split('.')[0] in ('models', 'experiments') and source.is_file(): paths.append(source)
    found.update((ROOT/'experiments/efficiency_sidecar').glob('*.py'))
    found.add(ROOT/'docs/native_speed_adoption_plan.md')
    found.add(ROOT/'tests/test_native_speed_transition.py')
    assert read(OUT/'cpu_checks.json')['passed']
    found.add(OUT/'cpu_checks.json')
    write(OUT/'amendment.json', dict(created_unix=time.time(),
        authorization='User requested exclusive speed testing and applying effective training optimization to pending work.',
        source_thread='01a1bea6-819a-7430-9025-5b043d3a3c05',
        original_queue=queue, original_controller=controller, original_postprocess=post_identity,
        pending_at_hold=queue['pending'], original_training_lock_sha256=sha(native.OUT/'lock.json'),
        original_evaluation_lock_sha256=sha(native.OUT/'evaluation_lock.json'),
        source_sha256={str(p): sha(p) for p in sorted(found)},
        already_started_distance33_preserved=True, scientific_plan_unchanged=True,
        floating_point_training_trajectory_may_differ=True))
    print('SPEED_TRANSITION_PREPARED', sha(OUT/'amendment.json'), flush=True)


def run():
    amendment = read(OUT/'amendment.json')
    for p, digest in amendment['source_sha256'].items(): assert sha(p) == digest, p
    native.check()
    assert not (OUT/'status.json').exists(), 'Inspect the recorded process before any recovery.'
    state = dict(process=identity(os.getpid()), started_unix=time.time(), state='waiting_for_current_fits',
        amendment_sha256=sha(OUT/'amendment.json'), active_child=None, stages=[],
        current_fits=amendment['original_queue']['active'], pending=amendment['pending_at_hold'],
        scientific_review_complete=False, goal_complete=False)

    def save(**updates):
        state.update(updates, updated_unix=time.time())
        write(OUT/'status.json', state)

    def gpu(exclusive, phase):
        while True:
            cards, apps = resources()
            used = {a['uuid'] for a in apps}
            choices = [c for c in cards if c['free_mib'] >= (24576 if exclusive else 16384)
                       and (not exclusive or (c['uuid'] not in used and c['used_mib'] < 512 and c['utilization'] < 10))]
            choices.sort(key=lambda c: (c['uuid'] not in used, -c['utilization'], c['free_mib']), reverse=True)
            if choices: return choices[0]
            save(state=phase, last_gpu_check=cards)
            time.sleep(30)

    def child(command, card, phase, bounded=False, threads=1, on_start=None):
        environment = os.environ.copy()
        environment.update(CUDA_VISIBLE_DEVICES=card['uuid'], CUBLAS_WORKSPACE_CONFIG=':4096:8',
            PYTHONPATH=str(ROOT), OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads),
            OPENBLAS_NUM_THREADS=str(threads), TORCHINDUCTOR_CACHE_DIR='/tmp/grope_exclusive_inductor',
            TRITON_CACHE_DIR='/tmp/grope_exclusive_triton', TORCHINDUCTOR_COMPILE_THREADS='1')
        path = OUT/(phase+'_'+str(time.time_ns())+'.log')
        with path.open('x') as stream:
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stream,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            record = dict(process=identity(process.pid), command=command, gpu=card, log=str(path), started_unix=time.time())
            save(state=phase, active_child=record)
            if on_start is not None: on_start(record)
            print('SPEED_TRANSITION_CHILD', phase, process.pid, flush=True)
            while process.poll() is None:
                if bounded and time.time()-record['started_unix'] > 600:
                    os.killpg(process.pid, signal.SIGTERM)
                    try: process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL); process.wait()
                    raise TimeoutError('Bounded own benchmark stopped: '+phase)
                save(); time.sleep(15)
            record.update(exit_code=process.returncode, finished_unix=time.time())
        state['stages'].append(record)
        save(active_child=None)
        return process.returncode

    try:
        save()
        print('WAITING_EXISTING_FITS_QUEUE_HELD', [(x['role'], x['seed']) for x in state['current_fits']], flush=True)
        # The stopped old parent retains exit status; its four workers continue normally.
        done = set()
        while len(done) < len(state['current_fits']):
            observed = []
            for item in state['current_fits']:
                key = (item['role'], item['seed'])
                if key in done: continue
                progress = native.OUT/'runs'/item['role']/str(item['seed'])/'progress.json'
                if progress.exists():
                    row = read(progress)
                    observed.append(dict(role=item['role'], seed=item['seed'], epoch=row['epoch'], step=row['step']))
                if live(item): continue
                terminal = identity(item['pid'])
                status = None if terminal is None else terminal['exit_status']
                code = None if status is None else os.waitstatus_to_exitcode(status)
                if code == 75:
                    # Normal time shards retain the original implementation and exact resume state.
                    while code == 75:
                        card = gpu(True, 'waiting_original_shard_gpu')
                        code = child([sys.executable, '-u', '-m', 'experiments.native_source_diversity',
                                      'train', '--role', item['role'], '--seed', str(item['seed'])],
                                     card, 'resume_original_'+item['role']+'_'+str(item['seed']), threads=4)
                if code not in (None, 0): raise RuntimeError(f'Original fit failed: {key}, exit {code}')
                checked_run(*key)
                done.add(key)
                print('ORIGINAL_FIT_VERIFIED_COMPLETE', *key, flush=True)
            save(state='waiting_for_current_fits', observed_progress=observed, verified_current_completed=[list(x) for x in sorted(done)])
            if len(done) < len(state['current_fits']): time.sleep(30)
        for job in amendment['original_queue']['completed']: checked_run(*job)

        stages = [
            ('fusion_correctness', ['experiments.efficiency_sidecar.fullbatch_gpu', 'correctness', '--mode', 'exclusive'], 1),
            ('fusion_timing', ['experiments.efficiency_sidecar.fullbatch_gpu', 'timing', '--mode', 'exclusive'], 1),
            ('compile_correctness', ['experiments.native_speed_followup', 'compile_correctness'], 1),
            ('compile_timing', ['experiments.native_speed_followup', 'compile_timing'], 1),
            ('sphere_adoption', ['experiments.native_speed_followup', 'sphere'], 4)]
        for phase, arguments, threads in stages:
            card = gpu(True, 'waiting_exclusive_'+phase)
            code = child(['nice', '-n', '15', sys.executable, '-u', '-m']+arguments,
                         card, phase, bounded=True, threads=threads)
            if code: raise RuntimeError(f'{phase} failed with exit {code}; preserve partial output.')

        fusion = read(ROOT/'results/efficiency_sidecar/fullbatch1/exclusive/timing.json')
        compile_result = read(OUT/'compile_exclusive/b16/timing.json')
        summaries = []
        for group, result in (('fusion', fusion), ('compile', compile_result)):
            for role in ('relative5', 'distance_mlp'):
                for mode in (('training',) if group == 'fusion' else ('inference', 'training')):
                    rows = [x for x in result['measurements'] if x['role'] == role and x.get('mode', 'training') == mode]
                    summaries.append(dict(group=group, role=role, mode=mode, **paired_summary(rows, 'baseline', 'candidate')))
        decision = read(OUT/'sphere/decision.json')
        write(OUT/'benchmark_summary.json', dict(results=summaries, sphere_decision=decision,
            exclusive=True, source_sha256={str(OUT/'amendment.json'): sha(OUT/'amendment.json')},
            full_training_accuracy_equivalence_established=False))
        lines = ['# 独占复测与剩余训练接续', '', '| 比较 | 方法 | 模式 | 配对耗时下降 | 正向块 |', '| --- | --- | --- | ---: | ---: |']
        lines += [f"| {r['group']} | {r['role']} | {r['mode']} | {r['median_gain_percent']:.2f}% | {r['positive_blocks']}/8 |" for r in summaries]
        lines += ['', '固定球形剩余任务采用：`'+decision['implementation']+'`。',
                  '选择依据见 sphere/decision.json；原版、重排、融合三臂的数值、显存和短程记录全部保留。',
                  '固定热池速度不等于完整训练节时，短程诊断不证明终点精度无损。',
                  'Distance/33 已提前启动并按原实现完成，只有固定球形/33采用本次决定。',
                  '编译与融合分开测试，未相加百分比，也未用于固定球形训练。']
        (OUT/'report.md').write_text('\n'.join(lines)+'\n')
        save(state='benchmarks_complete', decision=decision['implementation'])

        # Replace only the two owned controllers, after every original worker is terminal.
        original_post = amendment['original_postprocess']
        post_state = read(native.OUT/'postprocess_status.json')
        assert post_state['state'] == 'waiting_for_training' and post_state['active_child'] is None
        for item in amendment['original_queue']['active']: assert not live(item)
        for name in ('queue_status.json', 'postprocess_status.json', 'active_execution.json'):
            shutil.copy2(native.OUT/name, OUT/('before_handoff_'+name))
        for record, sig in ((original_post, signal.SIGTERM), (amendment['original_controller'], signal.SIGKILL)):
            assert live(record)
            os.kill(record['pid'], sig)
            for _ in range(100):
                if not live(record): break
                time.sleep(.1)
            assert not live(record)
        write(OUT/'controller_handoff.json', dict(time_unix=time.time(), old_queue=amendment['original_controller'],
            old_postprocess=original_post, new_queue=state['process'], workers_signalled=[],
            reason='Authorized queue replacement after current fits and exclusive benchmarks finished.'))
        (native.OUT/'postprocess_status.json').unlink()
        queue = dict(amendment['original_queue'], pid=os.getpid(), started_unix=state['started_unix'],
            state='running', active=[], completed=amendment['original_queue']['completed']+[list(x) for x in sorted(done)],
            pending=list(amendment['pending_at_hold']), failures=[],
            queue_source_sha256=sha(__file__), execution_amendment_sha256=sha(OUT/'amendment.json'))

        def save_queue():
            queue['updated_unix'] = time.time(); write(native.OUT/'queue_status.json', queue)

        save_queue()
        path = OUT/'resumed_postprocess.log'
        with path.open('x') as stream:
            post = subprocess.Popen([sys.executable, '-u', '-m', 'experiments.native_source_diversity_postprocess',
                '--queue-pid', str(os.getpid()), '--queue-start-ticks', str(state['process']['start_ticks'])],
                cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        save(resumed_postprocess=identity(post.pid))
        write(native.OUT/'active_execution.json', dict(observed_unix=time.time(),
            processes=[dict(name='speed_transition_queue', **state['process']), dict(name='postprocess', **identity(post.pid))],
            queue_status_file=str(native.OUT/'queue_status.json'), speed_status_file=str(OUT/'status.json'),
            previous_index=str(OUT/'before_handoff_active_execution.json'), goal_complete=False))
        while queue['pending']:
            role, seed = queue['pending'][0]
            card = gpu(False, 'waiting_remaining_training_gpu')

            def record_launch(record):
                queue['active'] = [dict(role=role, seed=seed, **record['process'], gpu=card,
                    log=record['log'], command=record['command'], started_unix=record['started_unix'],
                    allocator_cap_bytes=native.CAP_BYTES, runtime_implementation=decision['implementation'])]
                queue['pending'] = queue['pending'][1:]; save_queue()

            code = child([sys.executable, '-u', '-m', 'experiments.native_speed_followup',
                          'train', '--role', role, '--seed', str(seed)], card, 'remaining_training', threads=4,
                         on_start=record_launch)
            queue['active'] = []
            if code == 75:
                queue['pending'].insert(0, [role, seed])
            elif code == 0:
                checked_run(role, seed); queue['completed'].append([role, seed])
            else:
                queue['failures'].append(dict(role=role, seed=seed, exit_code=code)); queue['state'] = 'failed'
                save_queue(); raise RuntimeError('Remaining training failed; no automatic restart.')
            save_queue()
        native.select_all()
        queue.update(state='ready_for_evaluation', all_models_selected=True, finished_unix=time.time()); save_queue()
        save(state='waiting_for_evaluation')
        while post.poll() is None:
            save(); time.sleep(30)
        assert post.returncode == 0 and read(native.OUT/'postprocess_status.json')['state'] == 'ready_for_scientific_review'
        save(state='ready_for_scientific_review', finished_unix=time.time())
        print('SOURCE_DIVERSITY_SPEED_TRANSITION_COMPLETE_REVIEW_PENDING', flush=True)
    except BaseException as error:
        save(state='transition_failed', error_type=type(error).__name__, reason=str(error), finished_unix=time.time())
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=('prepare', 'run'))
    args = parser.parse_args()
    preparation() if args.action == 'prepare' else run()
