"""Move the final paired fits from shared serial execution to idle GPUs."""
import argparse
import difflib
import fcntl
import inspect
from pathlib import Path
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


def prepare():
    shared.check()
    assert not (OUT / 'lock.json').exists()
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
    idle = original.parent.available_gpus()
    assert len(idle) >= 2, 'Keep the existing execution unless two idle GPUs are available.'
    assert not (BASE / 'runs/distance_mlp/88').exists()
    assert not (BASE / 'runs/relative5/88/run.json').exists()
    retained = read(shared.REV / 'retained_fits.json')
    assert retained['count'] == 8
    for row in retained['records']:
        assert sha(BASE / 'runs' / row['role'] / str(row['seed']) / 'run.json') == row['run_sha256']
    write(OUT / 'retained_fits.json', retained)
    write(OUT / 'plan.json', dict(created_unix=time.time(), resource_change_only=True,
        authorization='User reported GPUs newly free during the already authorized parallel research execution.',
        scientific_plan=str(original.PLAN), scientific_plan_sha256=sha(original.PLAN),
        hypothesis_competing_explanation_comparison_and_decision_unchanged=True,
        old_controller=controller, old_children=children,
        previous_maximum_owned_gpu_jobs=1, maximum_owned_gpu_jobs=4,
        actual_remaining_training_jobs=2, retained_complete_fits=8,
        policy='Resume relative5/88 from the last complete epoch and start Distance/88 on two stable whole-idle GPUs; original selection then up to four concurrent original export jobs. Preserve all foreign jobs.',
        initially_idle=idle, original_scientific_lock_sha256=sha(BASE / 'lock.json'),
        original_runtime_lock_sha256=sha(runtime.OUT / 'lock.json'),
        unchanged_epochs=120, unchanged_updates=46800, unchanged_batch_size=16,
        stopped_partial_epoch_may_be_reexecuted=True, optimizer_and_sampler_state_restored=True,
        scientific_or_model_changes=False, official_test_access=False, goal_complete=False))
    source = inspect.getsource(original.queue)
    changed = source
    edits = [
        ("OUT / 'execution_identity.json'", "HANDOFF / 'execution_identity.json'", 2),
        ('maximum_owned_gpu_jobs=4,',
         "maximum_owned_gpu_jobs=4, controller_identity_path=str(HANDOFF / 'execution_identity.json'), execution_policy='idle GPU continuation of final two fits',", 1),
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
        fromfile='frozen_original_queue', tofile='final_pair_idle_continuation')))
    paths = [Path(__file__).resolve(), HANDOFF_LIBRARY, SOURCE, OUT / 'plan.json',
             OUT / 'retained_fits.json', OUT / 'queue_changes.diff',
             shared.REV / 'lock.json', shared.REV / 'execution_identity.json',
             BASE / 'cuda_preflight.json', runtime.OUT / 'lock.json']
    write(OUT / 'lock.json', dict(created_unix=time.time(),
        source_sha256={str(p): sha(p) for p in paths},
        original_scientific_lock_sha256=sha(BASE / 'lock.json'), scientific_scope_unchanged=True))
    (OUT / 'lock.sha256').write_text(sha(OUT / 'lock.json') + '\n')
    print('FINAL_PAIR_IDLE_CONTINUATION_PREPARED', sha(OUT / 'lock.json'), flush=True)


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'check', 'stop_old', 'queue'))
    args = parser.parse_args()
    configure_handoff()
    if args.action == 'prepare':
        prepare()
    elif args.action == 'queue':
        with (OUT / 'controller.flock').open('a') as guard:
            fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
            handoff.queue()
    else:
        getattr(handoff, args.action)()
