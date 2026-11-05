"""Resource-only recovery of the frozen native replication on one shared GPU.

Original models, fits, sampling, selection and reports are used unchanged.
The user authorized spare-memory sharing after eight of ten fits completed.
"""
import argparse
import difflib
import fcntl
import inspect
import os
from pathlib import Path
import subprocess
import time

import numpy as np  # Keep this host's established NumPy-before-Torch order.
import torch

from experiments import native_seed_replication as experiment
from experiments import native_seed_replication_runtime as runtime
from experiments.round65_common import read, sha, write

REV = experiment.OUT / 'shared_recovery1'
MODULE = 'experiments.native_seed_replication_shared'
GPU_UUID = 'GPU-0c6d9570-5d7a-7c71-9d3c-949f8799f62d'
MIN_FREE_MIB = 20 * 1024
ALLOCATOR_CAP_BYTES = 16 * 2**30


def gpu_status():
    output = subprocess.check_output([
        'nvidia-smi', '--id=' + GPU_UUID,
        '--query-gpu=index,uuid,memory.used,memory.free,memory.total,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True, timeout=10).strip()
    index, uuid, used, free, total, utilization = [x.strip() for x in output.split(',')]
    assert int(index) == 0 and uuid == GPU_UUID
    return dict(index=0, uuid=uuid, memory_mib=int(used), free_mib=int(free),
                total_mib=int(total), utilization=int(utilization), shared_allowed=True)


def shared_gpus():
    row = gpu_status()
    return [row] if row['free_mib'] >= MIN_FREE_MIB else []


def process_identity(pid):
    try:
        path = Path('/proc') / str(pid)
        fields = (path / 'stat').read_text().rsplit(')', 1)[1].split()
        return dict(pid=pid, state=fields[0], start_ticks=fields[19],
                    argv=(path / 'cmdline').read_bytes().split(b'\0'))
    except FileNotFoundError:
        return None


def prepare():
    runtime.check()
    assert not (REV / 'lock.json').exists(), 'Preserve the existing recovery identity.'
    before = read(experiment.OUT / 'queue_status.json')
    assert before['state'] == 'stopped' and not before['active']
    old = process_identity(before['pid'])
    assert old is None or old['state'] == 'Z' or old['start_ticks'] != before['start_ticks']
    retained = []
    for row in before['completed']:
        assert row['seed'] in (44, 55, 66, 77) and row['role'] in experiment.ROLES
        old_worker = process_identity(row['pid'])
        assert old_worker is None or old_worker['state'] == 'Z' or not any(
            b'native_seed_replication' in arg for arg in old_worker['argv'])
        folder = experiment.OUT / 'runs' / row['role'] / str(row['seed'])
        run = read(folder / 'run.json')
        assert sha(folder / 'run.json') == row['run_sha256']
        assert run['epochs'] == 120 and run['steps'] == 46800
        assert run['lock_sha256'] == sha(experiment.OUT / 'lock.json')
        for name, digest in run['hashes'].items():
            assert sha(folder / name) == digest, str(folder / name)
        retained.append(row)
    assert {(r['role'], r['seed']) for r in retained} == {
        (role, seed) for role in experiment.ROLES for seed in (44, 55, 66, 77)}
    assert len(retained) == 8
    for role in experiment.ROLES:
        assert not (experiment.OUT / 'runs' / role / '88').exists()
    assert not (experiment.OUT / 'selected.json').exists()
    assert not (experiment.OUT / 'evaluation').exists()
    assert not (experiment.OUT / 'result.json').exists()
    write(REV / 'queue_before_recovery.json', before)
    write(REV / 'retained_fits.json', dict(records=retained, count=8,
        original_exit_codes_not_inferred=True, hashes_verified=True))
    write(REV / 'plan.json', dict(created_unix=time.time(), resource_change_only=True,
        authorization='User requested execution of the existing plan and spare-memory GPU sharing.',
        original_scientific_plan=str(experiment.PLAN),
        original_scientific_plan_sha256=sha(experiment.PLAN),
        hypothesis_competing_explanation_and_decision_unchanged=True,
        pending_fits=[dict(role=role, seed=88) for role in experiment.ROLES],
        retained_completed_fits=8, original_total_fits=10,
        gpu_uuid=GPU_UUID, gpu_index=0, maximum_owned_gpu_jobs=1,
        minimum_launch_free_mib=MIN_FREE_MIB, allocator_cap_bytes=ALLOCATOR_CAP_BYTES,
        observed_training_peak_gib=12.166, gpu_before_recovery=gpu_status(),
        scheduling='GPU 0 only; two consecutive free-memory observations and CUDA probe before each launch. One owned GPU worker at a time; preserve all other processes.',
        completion='Original ten-model selection, five-seed six-condition export, original report. Actual scientific and plot review still required.',
        model_batch_optimizer_stream_selection_and_budget_unchanged=True,
        shared_gpu_timings_not_for_method_speed_comparisons=True,
        factored_relation_projection_enabled=False,
        official_test_access=False, goal_complete=False))

    original = inspect.getsource(experiment.queue)
    changed = original
    edits = [
        ("OUT / 'execution_identity.json'", "HANDOFF / 'execution_identity.json'", 2),
        ('maximum_owned_gpu_jobs=4,',
         "maximum_owned_gpu_jobs=1, controller_identity_path=str(HANDOFF / 'execution_identity.json'), execution_policy='user-authorized GPU0 shared serial', minimum_free_mib=MIN_FREE_MIB, allocator_cap_bytes=ALLOCATOR_CAP_BYTES,", 1),
        ("active, completed, pending, stage = [], [], [], 'resource_dependency'",
         "active, completed, pending, stage = [], read(HANDOFF / 'retained_fits.json')['records'], [], 'resource_dependency'", 1),
        ('(4 if gpu else 1)', '(1 if gpu else 1)', 1),
        ('parent.available_gpus()', 'shared_gpus()', 2),
        ("'waiting_for_idle_gpu'", "'waiting_for_shared_headroom'", 1),
        ("batch('preflight', [{}])", "assert read(OUT / 'cuda_preflight.json')['passed']  # Reuse completed full-config preflight.", 1),
        ("        batch('train', jobs)",
         "        jobs = [job for job in jobs if not (OUT / 'runs' / job['role'] / str(job['seed']) / 'run.json').exists()]\n        assert jobs == [dict(seed=88, role=role) for role in ROLES]\n        batch('train', jobs)", 1),
    ]
    for before_text, after_text, count in edits:
        assert changed.count(before_text) == count, before_text
        changed = changed.replace(before_text, after_text)
    compile(changed, str(REV / 'queue_source.py'), 'exec')
    (REV / 'queue_source.py').write_text(changed)
    (REV / 'queue_changes.diff').write_text(''.join(difflib.unified_diff(
        original.splitlines(True), changed.splitlines(True),
        fromfile='frozen_original_queue', tofile='shared_resource_recovery')))
    paths = [Path(__file__).resolve(), REV / 'queue_source.py', REV / 'queue_changes.diff',
             REV / 'plan.json', REV / 'retained_fits.json', REV / 'queue_before_recovery.json',
             experiment.OUT / 'cuda_preflight.json', runtime.OUT / 'lock.json']
    write(REV / 'lock.json', dict(created_unix=time.time(),
        source_sha256={str(p): sha(p) for p in paths},
        original_scientific_lock_sha256=sha(experiment.OUT / 'lock.json'),
        scientific_scope_unchanged=True))
    (REV / 'lock.sha256').write_text(sha(REV / 'lock.json') + '\n')
    print('SHARED_RECOVERY_PREPARED', sha(REV / 'lock.json'), flush=True)


def check():
    runtime.check()
    lock = read(REV / 'lock.json')
    assert sha(REV / 'lock.json') == (REV / 'lock.sha256').read_text().strip()
    assert sha(experiment.OUT / 'lock.json') == lock['original_scientific_lock_sha256']
    for name, digest in lock['source_sha256'].items():
        assert sha(name) == digest, name
    for row in read(REV / 'retained_fits.json')['records']:
        assert sha(experiment.OUT / 'runs' / row['role'] / str(row['seed']) / 'run.json') == row['run_sha256']


def queue():
    check()
    with (REV / 'controller.flock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        namespace = dict(vars(experiment), check=check, HANDOFF=REV,
            MODULE=MODULE, shared_gpus=shared_gpus, MIN_FREE_MIB=MIN_FREE_MIB,
            ALLOCATOR_CAP_BYTES=ALLOCATOR_CAP_BYTES, __file__=__file__)
        exec(compile((REV / 'queue_source.py').read_text(), str(REV / 'queue_source.py'), 'exec'), namespace)
        namespace['queue']()


def worker(action, role=None, seed=None):
    check()
    if action != 'report':
        assert os.environ.get('CUDA_VISIBLE_DEVICES') == GPU_UUID
        assert torch.cuda.is_available()
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(ALLOCATOR_CAP_BYTES / total, 0)
        print('SHARED_CUDA_ALLOCATOR_CAP', ALLOCATOR_CAP_BYTES, flush=True)
    runtime.run(action, role, seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'check', 'queue', 'train', 'selections', 'export', 'report'))
    parser.add_argument('--role', choices=experiment.ROLES)
    parser.add_argument('--seed', type=int, choices=experiment.SEEDS)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare()
    elif args.action == 'check':
        check()
        print('SHARED_RECOVERY_CHECKED', flush=True)
    elif args.action == 'queue':
        queue()
    else:
        worker(args.action, args.role, args.seed)
