"""Resource-only inference queue: idle GPUs plus at most two shared-headroom jobs.

Preparing this adapter never interrupts acquisition or starts official inference.
The original queue lock must be available before `queue` can run.
"""
import argparse
import difflib
import inspect
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from experiments import native_confirmation_workflow as workflow
from experiments.native_stream_cache import atomic_json
from experiments.round65_common import ROOT, sha


REV = workflow.OUT / 'gpu_resources1'
MODULE = 'experiments.native_confirmation_gpu_resources'
MIN_FREE_MIB = 20 * 1024
ALLOCATOR_CAP_BYTES = 16 * 2**30
MAX_SHARED_JOBS = 2


def eligible_devices(rows):
    eligible = []
    for row in rows:
        idle = (row['memory_mib'] <= 512 and row['free_mib'] >= 34 * 1024
                and row['utilization'] <= 10)
        if idle or row['free_mib'] >= MIN_FREE_MIB:
            eligible.append(dict(row, shared=not idle))
    return sorted(eligible, key=lambda row: (row['shared'], row['utilization'],
                                            -row['free_mib'], row['index']))


def available_gpus():
    output = subprocess.check_output(['nvidia-smi',
        '--query-gpu=index,uuid,memory.used,memory.free,memory.total,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True, timeout=10)
    rows = []
    for line in output.strip().splitlines():
        index, uuid, used, free, total, utilization = [item.strip() for item in line.split(',')]
        rows.append(dict(index=int(index), uuid=uuid, memory_mib=int(used),
                         free_mib=int(free), total_mib=int(total), utilization=int(utilization)))
    return eligible_devices(rows)


def shared_slot_available(device, active):
    return not device['shared'] or sum(bool(item['gpu']['shared']) for _, item, _ in active) < MAX_SHARED_JOBS


def apply_allocator_cap():
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('exactly one selected CUDA device required')
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(ALLOCATOR_CAP_BYTES / total, 0)
    print('CONFIRMATION_CUDA_ALLOCATOR_CAP', ALLOCATOR_CAP_BYTES, flush=True)


def prepare():
    workflow.checked_seal()
    assert not REV.exists(), 'preserve prior resource preparation'
    REV.mkdir()
    original = inspect.getsource(workflow.queue)
    changed = original
    edits = [
        ('    from experiments.round73_native_relative5 import available_gpus\n', '', 1),
        ("started_unix=time.time(), evaluation_lock_sha256=sha(OUT / 'evaluation_lock.json'))",
         "started_unix=time.time(), evaluation_lock_sha256=sha(OUT / 'evaluation_lock.json'),\n"
         "                        resource_record_sha256=sha(RESOURCE_REV / 'runtime_record.json'),\n"
         "                        maximum_shared_gpu_jobs=MAX_SHARED_JOBS, allocator_cap_bytes=ALLOCATOR_CAP_BYTES)", 1),
        ("                    env = dict(os.environ, OMP_NUM_THREADS='4'",
         "                    if not shared_slot_available(device, active):\n"
         "                        continue\n"
         "                    env = dict(os.environ, OMP_NUM_THREADS='4'", 1),
        ("                    if device['uuid'] not in {d['uuid'] for d in available_gpus()}:\n"
         "                        continue\n",
         "                    fresh = {d['uuid']: d for d in available_gpus()}\n"
         "                    if device['uuid'] not in fresh:\n"
         "                        continue\n"
         "                    device = fresh[device['uuid']]\n"
         "                    if not shared_slot_available(device, active):\n"
         "                        continue\n", 1),
        ("'waiting_for_idle_gpu'", "'waiting_for_gpu_headroom'", 1),
    ]
    for before, after, count in edits:
        assert changed.count(before) == count, before
        changed = changed.replace(before, after)
    compile(changed, str(REV / 'queue_source.py'), 'exec')
    (REV / 'queue_source.py').write_text(changed)
    (REV / 'queue_changes.diff').write_text(''.join(difflib.unified_diff(
        original.splitlines(True), changed.splitlines(True),
        fromfile='frozen_original_queue', tofile='authorized_shared_headroom_queue')))
    atomic_json(REV / 'plan.json', dict(created_unix=time.time(), resource_change_only=True,
        authorization='User authorized idle GPU parallelism and a small number of spare-memory shared GPUs.',
        scientific_settings_changed=False, evaluation_lock_sha256=sha(workflow.OUT / 'evaluation_lock.json'),
        maximum_owned_gpu_jobs=8, maximum_jobs_launched_on_busy_gpus=MAX_SHARED_JOBS,
        minimum_shared_launch_free_mib=MIN_FREE_MIB, allocator_cap_bytes=ALLOCATOR_CAP_BYTES,
        selection='Idle GPUs first, then lower-utilization shared cards; stable availability, CUDA probe, fresh occupancy and shared-slot checks before launch.',
        inference='Original export_model/export_one, original four-object batch and all six conditions; original models, source population and analysis.',
        validation='Pure allocation-policy checks; original six old fit objects, saved relative5/Distance outputs, all seven roles at dense1024 under the cap.',
        activation='Preparation only. Keep live acquisition untouched. Use only after cache is complete and the original controller is confirmed stopped; original queue process lock remains mandatory.',
        shared_gpu_timings_used_for_method_speed_comparisons=False, new_fits=0, goal_complete=False))
    print('GPU_RESOURCE_ADAPTER_PREPARED', flush=True)


def check():
    workflow.checked_seal()
    record = json.loads((REV / 'runtime_record.json').read_text())
    assert record['evaluation_lock_sha256'] == sha(workflow.OUT / 'evaluation_lock.json')
    assert record['scientific_settings_changed'] is False
    assert (record['maximum_shared_jobs'], record['allocator_cap_bytes']) == (MAX_SHARED_JOBS, ALLOCATOR_CAP_BYTES)
    for field in ('source_sha256', 'validation_sha256'):
        for path, digest in record[field].items():
            assert sha(path) == digest, path
    validation = json.loads((REV / 'validation.json').read_text())
    assert validation['passed'] and validation['official_test_inputs_read'] is False
    assert json.loads((REV / 'units.json').read_text())['returncode'] == 0
    return record


def queue():
    check()
    namespace = dict(vars(workflow), MODULE=MODULE, available_gpus=available_gpus,
                     RESOURCE_REV=REV, MAX_SHARED_JOBS=MAX_SHARED_JOBS,
                     ALLOCATOR_CAP_BYTES=ALLOCATOR_CAP_BYTES,
                     shared_slot_available=shared_slot_available)
    exec(compile((REV / 'queue_source.py').read_text(), str(REV / 'queue_source.py'), 'exec'), namespace)
    namespace['queue']()


def export(role, seed):
    check()
    apply_allocator_cap()
    workflow.export_model(role, seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'check', 'queue', 'export'))
    parser.add_argument('--role')
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare()
    elif args.action == 'check':
        check()
        print('GPU_RESOURCE_ADAPTER_CHECKED', flush=True)
    elif args.action == 'queue':
        queue()
    else:
        if args.role is None or args.seed is None:
            parser.error('export needs role and seed')
        export(args.role, args.seed)
