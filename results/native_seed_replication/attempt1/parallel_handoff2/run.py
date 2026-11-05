"""Run the final paired fits on two GPUs with user-authorized memory sharing."""
import argparse
import difflib
import fcntl
import inspect
import os
from pathlib import Path
import signal
import subprocess
import time

import numpy as np
import torch

from experiments import native_seed_replication as original
from experiments import native_seed_replication_runtime as runtime
from experiments import native_seed_replication_shared as shared
from experiments.round65_common import read, sha, write
from results.native_seed_replication.attempt1.parallel_handoff1 import run as handoff

OUT = Path(__file__).resolve().parent
BASE = original.OUT
SOURCE = OUT / 'queue_source.py'
HANDOFF_LIBRARY = Path(handoff.__file__).resolve()
MODULE = 'results.native_seed_replication.attempt1.parallel_handoff2.run'
GPU_UUIDS = (shared.GPU_UUID, 'GPU-6874d00d-b517-a763-5f6a-f1680030c4f8')


def gpu_rows():
    output = subprocess.check_output(['nvidia-smi',
        '--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True, timeout=10)
    rows = []
    for line in output.strip().splitlines():
        index, uuid, used, free, utilization = [v.strip() for v in line.split(',')]
        if uuid in GPU_UUIDS:
            rows.append(dict(index=int(index), uuid=uuid, memory_mib=int(used),
                free_mib=int(free), utilization=int(utilization), shared_allowed=True))
    assert {r['uuid'] for r in rows} == set(GPU_UUIDS)
    return rows


def available_gpus():
    return [r for r in gpu_rows() if r['free_mib'] >= shared.MIN_FREE_MIB]


def prepare():
    shared.check()
    assert not (OUT / 'execution_identity.json').exists()
    assert not (OUT / 'stop_record.json').exists()
    assert read(OUT / 'idle_only_prelaunch_v1/superseded.json')['executed'] is False
    state = read(BASE / 'queue_status.json')
    identity = read(shared.REV / 'execution_identity.json')
    controller = handoff.process(identity['pid'])
    assert controller and controller['state'] != 'Z'
    assert controller['start_ticks'] == identity['start_ticks']
    assert shared.MODULE + ' queue' in controller['command']
    assert state['pid'] == controller['pid'] and state['state'] == 'running' and state['stage'] == 'train'
    assert [(r['role'], r['seed']) for r in state['active']] == [('relative5', 88)]
    assert state['pending'] == [dict(seed=88, role='distance_mlp')]
    children = []
    for row in state['active']:
        child = handoff.process(row['pid'])
        assert child and child['state'] != 'Z' and shared.MODULE + ' train' in child['command']
        children.append(dict(child, role=row['role'], seed=row['seed']))
    devices = gpu_rows()
    assert any(r['uuid'] == GPU_UUIDS[1] and r['free_mib'] >= shared.MIN_FREE_MIB for r in devices), 'Keep current execution unless GPU 7 has sufficient headroom.'
    assert not (BASE / 'runs/distance_mlp/88').exists()
    assert not (BASE / 'runs/relative5/88/run.json').exists()
    retained = read(shared.REV / 'retained_fits.json')
    assert retained['count'] == 8
    for row in retained['records']:
        assert sha(BASE / 'runs' / row['role'] / str(row['seed']) / 'run.json') == row['run_sha256']
    write(OUT / 'retained_fits.json', retained)
    write(OUT / 'plan.json', dict(created_unix=time.time(), resource_change_only=True,
        authorization='User reported GPUs newly free; prior explicit authorization permits spare-memory sharing. New foreign jobs entered idle cards before stopping, so use the existing GPU 0 and available headroom on GPU 7.',
        scientific_plan=str(original.PLAN), scientific_plan_sha256=sha(original.PLAN),
        hypothesis_competing_explanation_comparison_and_decision_unchanged=True,
        old_controller=controller, old_children=children,
        previous_maximum_owned_gpu_jobs=1, maximum_owned_gpu_jobs=2,
        actual_remaining_training_jobs=2, retained_complete_fits=8,
        policy='Use only GPU 0 and GPU 7 with two consecutive free-memory observations, at least 20 GiB free and a CUDA probe before each launch. At most one owned worker per card and two overall. Resume original checkpoints; original selection then two concurrent original export jobs. Preserve all foreign jobs.',
        gpu_uuids=GPU_UUIDS, minimum_launch_free_mib=shared.MIN_FREE_MIB,
        allocator_cap_bytes=shared.ALLOCATOR_CAP_BYTES,
        gpu_observations=devices, original_scientific_lock_sha256=sha(BASE / 'lock.json'),
        original_runtime_lock_sha256=sha(runtime.OUT / 'lock.json'),
        unchanged_epochs=120, unchanged_updates=46800, unchanged_batch_size=16,
        stopped_partial_epoch_may_be_reexecuted=True, optimizer_and_sampler_state_restored=True,
        scientific_or_model_changes=False, official_test_access=False, goal_complete=False))
    source = inspect.getsource(original.queue)
    changed = source
    edits = [
        ("OUT / 'execution_identity.json'", "HANDOFF / 'execution_identity.json'", 2),
        ('maximum_owned_gpu_jobs=4,',
         "maximum_owned_gpu_jobs=2, controller_identity_path=str(HANDOFF / 'execution_identity.json'), execution_policy='GPU0 and GPU7 shared parallel final pair',", 1),
        ('(4 if gpu else 1)', '(2 if gpu else 1)', 1),
        ('parent.available_gpus()', 'available_gpus()', 2),
        ("'waiting_for_idle_gpu'", "'waiting_for_shared_headroom'", 1),
        ("active, completed, pending, stage = [], [], [], 'resource_dependency'",
         "active, completed, pending, stage = [], read(HANDOFF / 'retained_fits.json')['records'], [], 'resource_dependency'", 1),
        ("batch('preflight', [{}])", "assert read(OUT / 'cuda_preflight.json')['passed']  # Preserve original full-configuration preflight.", 1),
        ("        batch('train', jobs)",
         "        jobs = [job for job in jobs if not (OUT / 'runs' / job['role'] / str(job['seed']) / 'run.json').exists()]\n        assert jobs == [dict(seed=88, role=role) for role in ROLES]\n        batch('train', jobs)", 1),
    ]
    for before, after, count in edits:
        assert changed.count(before) == count, before
        changed = changed.replace(before, after)
    compile(changed, str(SOURCE), 'exec')
    SOURCE.write_text(changed)
    (OUT / 'queue_changes.diff').write_text(''.join(difflib.unified_diff(
        source.splitlines(True), changed.splitlines(True),
        fromfile='frozen_original_queue', tofile='final_pair_shared_parallel_continuation')))
    paths = [Path(__file__).resolve(), HANDOFF_LIBRARY, SOURCE, OUT / 'plan.json',
             OUT / 'retained_fits.json', OUT / 'queue_changes.diff',
             shared.REV / 'lock.json', shared.REV / 'execution_identity.json',
             BASE / 'cuda_preflight.json', runtime.OUT / 'lock.json']
    write(OUT / 'lock.json', dict(created_unix=time.time(),
        source_sha256={str(p): sha(p) for p in paths},
        original_scientific_lock_sha256=sha(BASE / 'lock.json'), scientific_scope_unchanged=True))
    (OUT / 'lock.sha256').write_text(sha(OUT / 'lock.json') + '\n')
    print('FINAL_PAIR_PARALLEL_CONTINUATION_PREPARED', sha(OUT / 'lock.json'), flush=True)


def snapshots(prefix):
    rows = []
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
        assert len(saved['optimizer']['state']) > 0 and saved['sampler_state']
        rows.append(dict(role=role, seed=seed, path=str(target), sha256=sha(target),
            epoch_completed=saved['epoch_completed'], step=saved['step'],
            stream_sha256=saved['stream_sha256'], optimizer_entries=len(saved['optimizer']['state'])))
    write(OUT / f'{prefix}.json', dict(records=rows,
        paired_prefix_check_pending_until_distance_runs=True))
    return rows


def configure_handoff():
    handoff.OUT = OUT
    handoff.SOURCE = SOURCE
    handoff.snapshots = snapshots


def stop_old():
    handoff.check()
    assert not (OUT / 'stop_record.json').exists()
    plan = read(OUT / 'plan.json')
    assert handoff.alive(plan['old_controller']) and all(handoff.alive(r) for r in plan['old_children'])
    state = read(BASE / 'queue_status.json')
    assert {r['pid'] for r in state['active']} == {r['pid'] for r in plan['old_children']}
    assert any(r['uuid'] == GPU_UUIDS[1] for r in available_gpus())
    snapshots('before_stop')
    write(OUT / 'old_queue_before_stop.json', state)
    os.kill(plan['old_controller']['pid'], signal.SIGTERM)
    deadline = time.monotonic() + 45
    while handoff.alive(plan['old_controller']) or any(handoff.alive(r) for r in plan['old_children']):
        if time.monotonic() > deadline:
            raise RuntimeError('Old writers remain live; do not start duplicates.')
        time.sleep(.5)
    stopped = read(BASE / 'queue_status.json')
    assert stopped['state'] == 'stopped'
    write(OUT / 'old_queue_stopped.json', stopped)
    final = snapshots('restart')
    for row in final:
        assert sha(BASE / 'runs' / row['role'] / str(row['seed']) / 'resume.pt') == row['sha256']
    write(OUT / 'stop_record.json', dict(observed_unix=time.time(), intentional=True,
        all_old_processes_terminal=True, signaled_controller_pid=plan['old_controller']['pid'],
        restart_snapshots=final, other_processes_signaled=False, formal_budget_unchanged=True))
    print('OWNED_QUEUE_STOPPED_AT_SAVED_CHECKPOINT', final, flush=True)


def queue():
    handoff.check()
    plan, stop = read(OUT / 'plan.json'), read(OUT / 'stop_record.json')
    assert stop['all_old_processes_terminal']
    assert not handoff.alive(plan['old_controller']) and all(not handoff.alive(r) for r in plan['old_children'])
    assert not (OUT / 'execution_identity.json').exists()
    for row in stop['restart_snapshots']:
        assert sha(BASE / 'runs' / row['role'] / str(row['seed']) / 'resume.pt') == row['sha256']
    scope = dict(vars(original), __file__=str(SOURCE), HANDOFF=OUT,
        MODULE=MODULE, check=handoff.check, available_gpus=available_gpus)
    exec(compile(SOURCE.read_text(), str(SOURCE), 'exec'), scope)
    scope['queue']()


def worker(action, role=None, seed=None):
    handoff.check()
    if action != 'report':
        assert os.environ.get('CUDA_VISIBLE_DEVICES') in GPU_UUIDS
        assert torch.cuda.is_available()
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(shared.ALLOCATOR_CAP_BYTES / total, 0)
        print('SHARED_CUDA_ALLOCATOR_CAP', shared.ALLOCATOR_CAP_BYTES, flush=True)
    runtime.run(action, role, seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'check', 'stop_old', 'queue', 'train', 'selections', 'export', 'report'))
    parser.add_argument('--role', choices=original.ROLES)
    parser.add_argument('--seed', type=int, choices=original.SEEDS)
    args = parser.parse_args()
    configure_handoff()
    if args.action == 'prepare':
        prepare()
    elif args.action == 'queue':
        with (OUT / 'controller.flock').open('a') as guard:
            fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
            queue()
    elif args.action == 'stop_old':
        stop_old()
    elif args.action in ('train', 'selections', 'export', 'report'):
        worker(args.action, args.role, args.seed)
    else:
        getattr(handoff, args.action)()
