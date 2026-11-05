"""Resource-only continuation of the same ten frozen single-GPU fits."""
import argparse
import difflib
import inspect
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch
from experiments import native_seed_replication as original
from experiments import native_seed_replication_runtime as runtime
from experiments.round65_common import read, sha, write

OUT = Path(__file__).resolve().parent
BASE = original.OUT
SOURCE = OUT / 'queue_source.py'


def process(pid):
    path = Path(f'/proc/{pid}')
    try:
        parts = (path / 'stat').read_text().rsplit(')', 1)[1].split()
        return dict(pid=pid, state=parts[0], start_ticks=parts[19],
                    command=(path / 'cmdline').read_bytes().replace(b'\0', b' ').decode())
    except FileNotFoundError:
        return None


def alive(row):
    current = process(row['pid'])
    return current is not None and current['state'] != 'Z' and current['start_ticks'] == row['start_ticks']


def freeze():
    runtime.check()
    assert not (OUT / 'lock.json').exists()
    identity = read(BASE / 'execution_identity.json')
    controller = process(identity['pid'])
    assert controller and controller['start_ticks'] == identity['start_ticks']
    assert runtime.MODULE + ' queue' in controller['command']
    state = read(BASE / 'queue_status.json')
    assert state['pid'] == controller['pid'] and state['stage'] == 'train' and state['state'] == 'running'
    children = []
    for row in state['active']:
        current = process(row['pid'])
        assert current and runtime.MODULE + ' train' in current['command']
        children.append(dict(current, role=row['role'], seed=row['seed']))
    assert len(children) == 4
    idle = original.parent.available_gpus()
    assert len(idle) >= 2, 'No material resource improvement; leave current workers running'
    plan = dict(created_unix=time.time(), resource_change_only=True,
        reason='Additional whole-idle GPUs exist while other users have entered two currently owned cards; continue owned checkpoints on idle cards and dispatch more of the same pending fits.',
        previous_maximum_owned_gpu_jobs=4, maximum_owned_gpu_jobs=8,
        policy='Use only stable whole-idle devices under original occupancy and CUDA checks; actual concurrency follows availability. Preserve all other jobs.',
        old_controller=controller, old_children=children, initially_idle=idle,
        unchanged_roles=list(original.ROLES), unchanged_seeds=list(original.SEEDS),
        unchanged_fits=10, unchanged_epochs=120, unchanged_updates=46800,
        stopped_partial_epochs_may_be_reexecuted=True,
        original_scientific_lock_sha256=sha(BASE / 'lock.json'),
        original_runtime_lock_sha256=sha(runtime.OUT / 'lock.json'),
        scientific_or_model_changes=False, official_test_access=False, goal_complete=False)
    write(OUT / 'plan.json', plan)
    source = inspect.getsource(original.queue)
    changed = source
    substitutions = [
        ("OUT / 'execution_identity.json'", "HANDOFF / 'execution_identity.json'", 2),
        ('maximum_owned_gpu_jobs=4,', "maximum_owned_gpu_jobs=8, controller_identity_path=str(HANDOFF / 'execution_identity.json'),", 1),
        ('(4 if gpu else 1)', '(8 if gpu else 1)', 1),
        ("batch('preflight', [{}])", "assert read(OUT / 'cuda_preflight.json')['passed']  # Preserve completed preflight.", 1),
        ("        batch('train', jobs)", "        jobs = [job for job in jobs if not (OUT / 'runs' / job['role'] / str(job['seed']) / 'run.json').exists()]\n        batch('train', jobs)", 1),
    ]
    for before, after, expected in substitutions:
        assert changed.count(before) == expected, before
        changed = changed.replace(before, after)
    compile(changed, str(SOURCE), 'exec')
    SOURCE.write_text(changed)
    (OUT / 'queue_changes.diff').write_text(''.join(difflib.unified_diff(
        source.splitlines(True), changed.splitlines(True), fromfile='original_frozen_queue', tofile='resource_continuation')))
    paths = [Path(__file__), SOURCE, OUT / 'plan.json', OUT / 'queue_changes.diff',
             BASE / 'cuda_preflight.json', BASE / 'execution_identity.json', runtime.OUT / 'lock.json']
    write(OUT / 'lock.json', dict(created_unix=time.time(), source_sha256={str(p):sha(p) for p in paths},
        original_scientific_lock_sha256=sha(BASE / 'lock.json'), scientific_scope_unchanged=True))
    (OUT / 'lock.sha256').write_text(sha(OUT / 'lock.json') + '\n')
    print('PARALLEL_CONTINUATION_FROZEN', sha(OUT / 'lock.json'), flush=True)


def check():
    runtime.check()
    lock = read(OUT / 'lock.json')
    assert sha(OUT / 'lock.json') == (OUT / 'lock.sha256').read_text().strip()
    assert sha(BASE / 'lock.json') == lock['original_scientific_lock_sha256']
    for name, digest in lock['source_sha256'].items():
        assert sha(name) == digest, name


def snapshots(prefix):
    rows = []
    histories = {}
    for child in read(OUT / 'plan.json')['old_children']:
        role, seed = child['role'], child['seed']
        folder = BASE / 'runs' / role / str(seed)
        assert not (folder / 'run.json').exists()
        target = OUT / f'{prefix}_{role}_{seed}.pt'
        with (folder / 'resume.pt').open('rb') as source, target.open('xb') as output:
            output.write(source.read())
        saved = torch.load(target, map_location='cpu', weights_only=False)
        assert (saved['role'], saved['seed']) == (role, seed)
        assert saved['lock_sha256'] == sha(BASE / 'lock.json')
        assert saved['data_sha256'] == sha(BASE / 'data/manifest.json')
        assert saved['step'] == 390 * saved['epoch_completed']
        assert len(saved['training']) == saved['epoch_completed']
        histories[role, seed] = [r['stream_sha256'] for r in saved['training']]
        rows.append(dict(role=role, seed=seed, path=str(target), sha256=sha(target),
                         epoch_completed=saved['epoch_completed'], step=saved['step'],
                         optimizer_entries=len(saved['optimizer']['state'])))
    for seed in {r['seed'] for r in rows}:
        pair = [histories[role, seed] for role in original.ROLES]
        n = min(map(len, pair))
        assert pair[0][:n] == pair[1][:n]
    write(OUT / f'{prefix}.json', dict(records=rows, common_completed_stream_prefixes_match=True))
    return rows


def stop_old():
    check()
    assert not (OUT / 'stop_record.json').exists()
    plan = read(OUT / 'plan.json')
    assert alive(plan['old_controller']) and all(alive(r) for r in plan['old_children'])
    state = read(BASE / 'queue_status.json')
    assert {r['pid'] for r in state['active']} == {r['pid'] for r in plan['old_children']}
    assert len(original.parent.available_gpus()) >= 2
    snapshots('before_stop')
    write(OUT / 'old_queue_before_stop.json', state)
    os.kill(plan['old_controller']['pid'], signal.SIGTERM)
    deadline = time.monotonic() + 45
    while alive(plan['old_controller']) or any(alive(r) for r in plan['old_children']):
        if time.monotonic() > deadline:
            raise RuntimeError('Old writers remain live; do not start duplicates')
        time.sleep(.5)
    state = read(BASE / 'queue_status.json')
    assert state['state'] == 'stopped'
    write(OUT / 'old_queue_stopped.json', state)
    final = snapshots('restart')
    for row in final:
        assert sha(BASE / 'runs' / row['role'] / str(row['seed']) / 'resume.pt') == row['sha256']
    write(OUT / 'stop_record.json', dict(observed_unix=time.time(), intentional=True,
        all_old_processes_terminal=True, signaled_controller_pid=plan['old_controller']['pid'],
        restart_snapshots=final, other_processes_signaled=False, formal_budget_unchanged=True))
    print('OWNED_QUEUE_STOPPED_AT_SAVED_CHECKPOINTS', final, flush=True)


def queue():
    check()
    plan, stop = read(OUT / 'plan.json'), read(OUT / 'stop_record.json')
    assert stop['all_old_processes_terminal']
    assert not alive(plan['old_controller']) and all(not alive(r) for r in plan['old_children'])
    assert not (OUT / 'execution_identity.json').exists()
    for row in stop['restart_snapshots']:
        assert sha(BASE / 'runs' / row['role'] / str(row['seed']) / 'resume.pt') == row['sha256']
    scope = dict(vars(original), __file__=str(SOURCE), HANDOFF=OUT,
                 MODULE=runtime.MODULE, check=check)
    exec(compile(SOURCE.read_text(), str(SOURCE), 'exec'), scope)
    scope['queue']()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('freeze', 'stop_old', 'queue'))
    args = parser.parse_args()
    globals()[args.action]()
