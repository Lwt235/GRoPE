"""Three unchanged-budget controls separating covariance from radial features."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np
import torch

from experiments import region_geometry_controls as legacy
from experiments.round65_common import ROOT, SEEDS, read, write, sha
from experiments.round58_run import shared_digest
from experiments.round58_data import loss_function
from models.region_shape_attribution import FixedIsotropicMatcher, ROLES
from models.round69_compact_relation import CompactMatcher

OUT = ROOT / 'results/region_shape_attribution/attempt1'
PLAN = ROOT / 'results/region_shape_attribution/preparation/plan.json'
CANDIDATE, PARENT = legacy.CANDIDATE, legacy.PARENT
engine, data_module = legacy.engine, legacy.data_module
MODULE = 'experiments.region_shape_attribution'
CONDITIONS = ('base', 'sparse256')


def freeze():
    legacy.check()
    assert not (OUT / 'lock.json').exists()
    assert read(OUT / 'units.json')['passed']
    original = read(legacy.OUT / 'lock.json')
    sources = dict(original['source_sha256'])
    files = legacy.dependencies([Path(__file__), ROOT / 'experiments/region_shape_report.py',
                                 ROOT / 'tests/test_region_shape_attribution.py'])
    for path in files + [PLAN, OUT / 'units.json']:
        sources[str(path)] = sha(path)
    prior = dict(original['prior_sha256'])
    for path in (legacy.OUT / 'result.json', legacy.OUT / 'review.json',
                 ROOT / 'results/projected_shape_attribution/attempt1/review.json'):
        prior[str(path)] = sha(path)
    for path, digest in read(OUT / 'units.json')['source_sha256'].items():
        assert sha(path) == digest
    write(OUT / 'lock.json', dict(created_unix=time.time(), source_sha256=sources,
        prior_sha256=prior, cache_sha256=original['cache_sha256'],
        settings=dict(engine.SETTINGS, roles=ROLES), seeds=SEEDS, roles=ROLES,
        conditions=CONDITIONS, plan_sha256=sha(PLAN), new_fits=3,
        candidate_changed=False, fresh_confirmation=False, goal_complete=False))
    (OUT / 'lock.sha256').write_text(sha(OUT / 'lock.json') + '\n')
    print('REGION_SHAPE_FROZEN', sha(OUT / 'lock.json'), flush=True)


def check():
    lock = read(OUT / 'lock.json')
    assert sha(OUT / 'lock.json') == (OUT / 'lock.sha256').read_text().strip()
    for group in ('source_sha256', 'prior_sha256', 'cache_sha256'):
        for path, digest in lock[group].items():
            assert sha(path) == digest, path
    return lock


def bind():
    engine.OUT, engine.ROLES = OUT, ROLES
    engine.FixedRegionMatcher, engine.check, engine.MODULE = FixedIsotropicMatcher, check, MODULE
    engine.SETTINGS = dict(engine.SETTINGS, roles=ROLES)
    data_module.check = check


def preflight():
    check()
    engine.configure()
    assert torch.cuda.is_available() and not (OUT / 'preflight.json').exists()
    cache = data_module.load_cache('fit0')
    index = np.flatnonzero((cache['overlap'].max(-1) >= .3).any(-1))[:4]
    torch.manual_seed(11)
    reference = CompactMatcher('relative5')
    torch.manual_seed(11)
    model = FixedIsotropicMatcher().cuda().train()
    assert shared_digest(model) == shared_digest(reference)
    assert all(torch.equal(v.cpu(), reference.state_dict()[k]) for k, v in model.state_dict().items())
    torch.cuda.reset_peak_memory_stats()
    loss = loss_function(model(*engine.inputs(cache, index)),
        torch.as_tensor(cache['overlap'][index], device='cuda'))
    loss.backward()
    grads = {name: float(p.grad.norm()) if p.grad is not None else None
             for name, p in model.position.named_parameters()}
    assert torch.isfinite(loss) and all(x is not None and np.isfinite(x) for x in grads.values())
    assert any(x > 0 for x in grads.values())
    write(OUT / 'preflight.json', dict(passed=True, loss=float(loss.detach()), gradients=grads,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(), full_initial_state_matches=True,
        parameters=sum(p.numel() for p in model.parameters()),
        pe_parameters=sum(p.numel() for p in model.position.parameters()),
        optimizer_updates=0, gpu=torch.cuda.get_device_name(), lock_sha256=sha(OUT / 'lock.json')))
    print('REGION_SHAPE_PREFLIGHT_COMPLETE', flush=True)


def selections():
    check()
    assert not (OUT / 'selected.json').exists()
    val, rows = data_module.load_cache('val'), []
    for seed in SEEDS:
        run_dir = OUT / 'runs' / ROLES[0] / str(seed)
        run, history = read(run_dir / 'run.json'), read(run_dir / 'history.json')
        old_dir = CANDIDATE / 'runs/relative5' / str(seed)
        old, old_history = read(old_dir / 'run.json'), read(old_dir / 'history.json')
        assert run['epochs'] == 60 and run['steps'] == 1080 and len(history['training']) == 60
        assert run['initial_shared_sha256'] == old['initial_shared_sha256']
        assert [r['stream_sha256'] for r in history['training']] == [r['stream_sha256'] for r in old_history['training']]
        for name, digest in run['hashes'].items():
            assert sha(run_dir / name) == digest
        best = min(history['validation'], key=lambda r: r['selection_ce'])
        assert best['epoch'] == run['selected_epoch']
        with np.load(run_dir / 'validation.npz') as z:
            assert np.array_equal(z['ids'], val['ids'])
            values = engine.query_values(z['logits'], val)
        ce = np.mean([values['ce'][:, d][values['valid'][:, d]].mean() for d in (0, 1)])
        error = float(abs(ce - run['best_val_ce']))
        assert error < 1e-10
        rows.append(dict(role=ROLES[0], seed=seed, selected_epoch=best['epoch'],
            checkpoint_sha256=sha(run_dir / 'best.pt'), run_sha256=sha(run_dir / 'run.json'),
            validation_ce_error=error))
    write(OUT / 'selected.json', dict(records=rows, all_three_selected_before_eval=True,
        all_common_initializations_and_60_streams_match_relative5=True, lock_sha256=sha(OUT / 'lock.json')))
    print('REGION_SHAPE_SELECTED', flush=True)


def export(seed):
    check()
    engine.configure()
    selected = read(OUT / 'selected.json')
    assert selected['all_three_selected_before_eval'] and len(selected['records']) == 3
    choice = next(r for r in selected['records'] if r['seed'] == seed)
    run = OUT / 'runs' / ROLES[0] / str(seed)
    dest = OUT / 'evaluation' / ROLES[0] / str(seed)
    assert not dest.exists() and sha(run / 'best.pt') == choice['checkpoint_sha256']
    model = FixedIsotropicMatcher().cuda()
    model.load_state_dict(torch.load(run / 'best.pt', map_location='cpu', weights_only=True)['state_dict'])
    logits, _ = engine.evaluate(model, data_module.load_cache('val'))
    with np.load(run / 'validation.npz') as z:
        error = float(abs(logits - z['logits']).max())
        changes = int((logits.argmax(-1) != z['logits'].argmax(-1)).sum())
    assert error < 2e-4 and changes == 0
    dest.mkdir(parents=True)
    rows = []
    for condition in CONDITIONS:
        cache = data_module.load_cache('eval_' + condition)
        logits, independent = engine.evaluate(model, cache)
        metrics = engine.grouped_metrics(logits, cache)
        delta = max(abs(independent[k] - metrics['micro'][k])
                    for k in ('region_recall', 'center_pck', 'soft_epe', 'ce'))
        assert delta < 1e-10
        path = dest / (condition + '.npz')
        np.savez_compressed(path, logits=logits, ids=cache['ids'])
        rows.append(dict(role=ROLES[0], seed=seed, condition=condition, metrics=metrics,
                         prediction_sha256=sha(path), independent_metric_error=delta))
    write(dest / 'record.json', dict(records=rows, checkpoint_sha256=choice['checkpoint_sha256'],
        selection_sha256=sha(OUT / 'selected.json'), manifest_sha256=sha(PARENT / 'data/manifest.json'),
        validation_replay=dict(maximum_logit_error=error, prediction_changes=changes)))
    print('REGION_SHAPE_EXPORTED', seed, flush=True)


def available():
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu',
                                      '--format=csv,noheader,nounits'], text=True, timeout=10)
    choices = []
    for line in output.strip().splitlines():
        index, uuid, used, free, util = (x.strip() for x in line.split(','))
        if int(free) >= 12288:
            choices.append(dict(index=int(index), uuid=uuid, memory_mib=int(used),
                                free_mib=int(free), utilization=int(util)))
    return sorted(choices, key=lambda x: (-x['free_mib'], x['utilization']))


def queue():
    check()
    assert read(legacy.OUT / 'queue_status.json')['state'] == 'complete'
    assert not (OUT / 'execution_identity.json').exists()
    meta = dict(pid=os.getpid(), start_ticks=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19],
        source_sha256=sha(Path(__file__)), lock_sha256=sha(OUT / 'lock.json'),
        started_unix=time.time(), maximum_owned_gpu_jobs=3, allocator_ceiling_gib=2, goal_complete=False)
    write(OUT / 'execution_identity.json', meta)
    active, completed, pending, stage = [], [], [], 'start'

    def record(state, **extra):
        write(OUT / 'queue_status.json', dict(meta, state=state, stage=stage,
            active=[x[1] for x in active], completed=completed, pending=pending,
            observed_unix=time.time(), **extra))

    def stop(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    def batch(action, jobs, gpu=True):
        nonlocal pending, stage
        pending, stage, previous = list(jobs), action, set()
        while pending or active:
            for process, item, stream in list(active):
                code = process.poll()
                if code is not None:
                    stream.close()
                    active.remove((process, item, stream))
                    completed.append(dict(item, exit_code=code))
                    if code:
                        raise RuntimeError(f"{item['log']}: exit {code}")
            if not pending and not active:
                break
            choices = available() if gpu and pending else [None]
            current = {r['uuid'] for r in choices if r}
            stable, previous = current & previous, current
            owned = {x[1]['gpu']['uuid'] for x in active if x[1]['gpu']}
            for device in choices:
                if not pending or len(active) >= (3 if gpu else 1):
                    break
                if device and (device['uuid'] not in stable or device['uuid'] in owned):
                    continue
                env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
                    CUBLAS_WORKSPACE_CONFIG=':4096:8', CUDA_DEVICE_ORDER='PCI_BUS_ID',
                    CUDA_VISIBLE_DEVICES=device['uuid'] if device else '')
                probe = None
                if device:
                    code = 'import numpy; import torch; assert torch.cuda.is_available(); x=torch.ones(4,device="cuda"); assert float(x.sum())==4; torch.cuda.synchronize(); print(torch.cuda.get_device_name())'
                    probe = subprocess.check_output([sys.executable, '-c', code], env=env, text=True, timeout=60).strip()
                    if not any(r['uuid'] == device['uuid'] for r in available()):
                        continue
                job = pending.pop(0)
                args = [MODULE, action]
                for key, value in job.items():
                    args += ['--' + key, str(value)]
                if gpu:
                    code = ('import numpy; import torch; import runpy; import sys; '
                            'torch.cuda.set_per_process_memory_fraction(2*2**30/torch.cuda.get_device_properties(0).total_memory); '
                            f'sys.argv={args!r}; runpy.run_module({MODULE!r},run_name="__main__")')
                    command = [sys.executable, '-c', code]
                else:
                    command = [sys.executable, '-m'] + args
                log = OUT / f'{action}_{job.get("seed")}_{time.time_ns()}.log'
                stream = log.open('x')
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
                    stderr=subprocess.STDOUT, start_new_session=True)
                item = dict(job, action=action, pid=process.pid, gpu=device, log=str(log),
                            cuda_probe=probe, started_unix=time.time())
                active.append((process, item, stream))
                if device:
                    owned.add(device['uuid'])
                print('LAUNCHED', item, flush=True)
            record('running' if active else 'waiting_for_spare_gpu_capacity')
            time.sleep(5)

    try:
        batch('preflight', [{}])
        batch('train', [dict(role=ROLES[0], seed=s) for s in SEEDS])
        batch('selections', [{}], False)
        batch('export', [dict(seed=s) for s in SEEDS])
        batch('report', [{}], False)
        stage = 'complete'
        record('complete', scientific_review_pending=True, visual_review_pending=True)
    except BaseException as error:
        for process, item, stream in active:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            stream.close()
        record('stopped', error=str(error))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('freeze', 'queue', 'preflight', 'train', 'selections', 'export', 'report'))
    parser.add_argument('--role', choices=ROLES)
    parser.add_argument('--seed', type=int, choices=SEEDS)
    args = parser.parse_args()
    if args.action != 'freeze':
        bind()
    if args.action == 'train':
        engine.train(args.role, args.seed)
    elif args.action == 'export':
        export(args.seed)
    elif args.action == 'report':
        from experiments.region_shape_report import report
        report()
    else:
        globals()[args.action]()
