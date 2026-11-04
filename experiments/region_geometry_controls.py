"""Six fixed-budget controls to close the R68/R69 geometric-encoding gap."""
import argparse
import ast
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np
import torch

from experiments import round68_fixed_region_run as engine
from experiments import round68_fixed_region_data as data_module
from experiments.round65_common import ROOT, SEEDS, read, write, sha
from experiments.round58_data import loss_function
from experiments.round58_run import shared_digest
from models.region_geometry_controls import GeometryControlMatcher, ROLES
from models.round69_compact_relation import CompactMatcher

PARENT = ROOT/'results/round68_fixed_region/attempt1'
CANDIDATE = ROOT/'results/round69_compact_relation/attempt1'
OUT = ROOT/'results/region_geometry_controls/attempt1'
PLAN = ROOT/'results/region_geometry_controls/preparation/plan.json'
ARCHIVE = ROOT/'docs/archive/2026-11-04'
MODULE = 'experiments.region_geometry_controls'
CONDITIONS = ('base', 'sparse256')


def dependencies(paths):
    pending = list(paths)
    found = set()
    while pending:
        path = Path(pending.pop()).resolve()
        if path in found:
            continue
        found.add(path)
        if path.suffix != '.py':
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            modules = ([node.module] if isinstance(node, ast.ImportFrom) and node.module else
                       [n.name for n in node.names] if isinstance(node, ast.Import) else [])
            for name in modules:
                if name.split('.')[0] not in ('experiments', 'models', 'third_party'):
                    continue
                candidate = ROOT/(name.replace('.', '/')+'.py')
                if candidate.is_file():
                    pending.append(candidate)
    return sorted(found)


def freeze():
    assert not (OUT/'lock.json').exists()
    assert read(PARENT/'review.json')['scientific_review_complete']
    assert read(CANDIDATE/'review.json')['scientific_review_complete']
    assert read(OUT/'units.json')['passed']
    sources, aliases = {}, {}
    for record in (read(PARENT/'preparation_plan.json'), read(CANDIDATE/'lock.json')):
        for name, digest in record['sources'].items():
            path = ROOT/name
            if not path.exists():
                path = ARCHIVE/name
                aliases[name] = str(path)
            assert sha(path) == digest, name
            sources[str(path)] = digest
    files = dependencies([Path(__file__), ROOT/'experiments/region_geometry_report.py',
                          ROOT/'tests/test_region_geometry_controls.py'])
    files += list((ROOT/'third_party/region_geotransformer_reference').glob('*'))
    files += [PLAN, OUT/'units.json', ROOT/'tests/test_round68_metrics.py']
    for path in files:
        if path.is_file():
            sources[str(path)] = sha(path)
    for name, digest in read(OUT/'units.json')['source_sha256'].items():
        assert sha(name) == digest, name
    prior = {}
    for root in (PARENT, CANDIDATE):
        assert sha(root/'lock.json') == (root/'lock.sha256').read_text().strip()
        for name in ('lock.json', 'result.json', 'review.json', 'selected.json'):
            prior[str(root/name)] = sha(root/name)
    for name in ('data/manifest.json', 'data_review.json', 'preparation_plan.json'):
        prior[str(PARENT/name)] = sha(PARENT/name)
    for role in ('relative5',):
        for seed in SEEDS:
            for name in ('run.json', 'history.json', 'best.pt'):
                path = CANDIDATE/'runs'/role/str(seed)/name
                prior[str(path)] = sha(path)
    manifest = read(PARENT/'data/manifest.json')
    caches = {}
    for entry in manifest['caches'].values():
        assert sha(entry['path']) == entry['sha256']
        caches[entry['path']] = entry['sha256']
    write(OUT/'lock.json', dict(created_unix=time.time(), source_sha256=sources,
        archived_source_resolutions=aliases, prior_sha256=prior, cache_sha256=caches,
        settings=dict(engine.SETTINGS, roles=ROLES), seeds=SEEDS, roles=ROLES,
        conditions=CONDITIONS, plan_sha256=sha(PLAN), new_fits=6,
        reused_evaluation_groups=48, new_evaluation_groups=12,
        candidate_changed=False, fresh_confirmation=False, goal_complete=False))
    (OUT/'lock.sha256').write_text(sha(OUT/'lock.json')+'\n')
    print('REGION_GEOMETRY_FROZEN', sha(OUT/'lock.json'), flush=True)


def check():
    lock = read(OUT/'lock.json')
    assert sha(OUT/'lock.json') == (OUT/'lock.sha256').read_text().strip()
    for group in ('source_sha256', 'prior_sha256', 'cache_sha256'):
        for path, digest in lock[group].items():
            assert sha(path) == digest, path
    return lock


def bind():
    engine.OUT = OUT
    engine.ROLES = ROLES
    engine.FixedRegionMatcher = GeometryControlMatcher
    engine.check = check
    engine.MODULE = MODULE
    engine.SETTINGS = dict(engine.SETTINGS, roles=ROLES)
    # The unchanged loader still reads the R68 caches; source paths are resolved
    # from the archive by our lock instead of changing any historical source.
    data_module.check = check


def preflight():
    check()
    engine.configure()
    assert torch.cuda.is_available() and not (OUT/'preflight.json').exists()
    cache = data_module.load_cache('fit0')
    index = np.flatnonzero((cache['overlap'].max(-1) >= .3).any(-1))[:4]
    torch.manual_seed(11)
    reference = CompactMatcher('relative5')
    common = shared_digest(reference)
    del reference
    rows = []
    for role in ROLES:
        torch.cuda.reset_peak_memory_stats()
        torch.manual_seed(11)
        model = GeometryControlMatcher(role).cuda().train()
        assert shared_digest(model) == common
        loss = loss_function(model(*engine.inputs(cache, index)),
                             torch.as_tensor(cache['overlap'][index], device='cuda'))
        loss.backward()
        grads = {name:float(p.grad.norm()) if p.grad is not None else None
                 for name, p in model.position.named_parameters()}
        assert torch.isfinite(loss)
        assert all(x is not None and np.isfinite(x) for x in grads.values())
        assert any(x > 0 for x in grads.values())
        rows.append(dict(role=role, loss=float(loss.detach()), pe_gradients=grads,
            parameters=sum(p.numel() for p in model.parameters()),
            pe_parameters=sum(p.numel() for p in model.position.parameters()),
            peak_gib=torch.cuda.max_memory_allocated()/2**30))
        del model, loss
        torch.cuda.empty_cache()
    write(OUT/'preflight.json', dict(passed=True, rows=rows, optimizer_updates=0,
        data='first four eligible old fit0 pairs', shared_initialization_matches=True,
        lock_sha256=sha(OUT/'lock.json')))
    print('REGION_GEOMETRY_PREFLIGHT_COMPLETE', flush=True)


def selections():
    check()
    assert not (OUT/'selected.json').exists()
    val = data_module.load_cache('val')
    rows = []
    for seed in SEEDS:
        old = CANDIDATE/'runs/relative5'/str(seed)
        ref, history_ref = read(old/'run.json'), read(old/'history.json')
        for role in ROLES:
            path = OUT/'runs'/role/str(seed)
            run, history = read(path/'run.json'), read(path/'history.json')
            assert run['epochs'] == 60 and run['steps'] == 1080 and len(history['training']) == 60
            assert run['initial_shared_sha256'] == ref['initial_shared_sha256']
            assert ([r['stream_sha256'] for r in history['training']] ==
                    [r['stream_sha256'] for r in history_ref['training']])
            for name, digest in run['hashes'].items():
                assert sha(path/name) == digest
            best = min(history['validation'], key=lambda r:r['selection_ce'])
            assert best['epoch'] == run['selected_epoch']
            with np.load(path/'validation.npz') as z:
                assert np.array_equal(z['ids'], val['ids'])
                values = engine.query_values(z['logits'], val)
            ce = [float(values['ce'][:, d][values['valid'][:, d]].mean()) for d in (0, 1)]
            error = abs(float(np.mean(ce))-run['best_val_ce'])
            assert error < 1e-10
            rows.append(dict(role=role, seed=seed, selected_epoch=best['epoch'],
                checkpoint_sha256=sha(path/'best.pt'), run_sha256=sha(path/'run.json'),
                validation_ce_error=error))
    write(OUT/'selected.json', dict(records=rows, all_six_selected_before_eval=True,
        all_common_initializations_and_60_streams_match_relative5=True,
        lock_sha256=sha(OUT/'lock.json')))
    print('REGION_GEOMETRY_SELECTIONS_COMPLETE', flush=True)


def export(seed):
    check()
    engine.configure()
    selected = read(OUT/'selected.json')
    assert selected['all_six_selected_before_eval'] and len(selected['records']) == 6
    val = data_module.load_cache('val')
    caches = {name:data_module.load_cache('eval_'+name) for name in CONDITIONS}
    for role in ROLES:
        choice = next(x for x in selected['records'] if x['role'] == role and x['seed'] == seed)
        run, dest = OUT/'runs'/role/str(seed), OUT/'evaluation'/role/str(seed)
        assert not dest.exists() and sha(run/'best.pt') == choice['checkpoint_sha256']
        model = GeometryControlMatcher(role).cuda()
        model.load_state_dict(torch.load(run/'best.pt', map_location='cpu', weights_only=True)['state_dict'])
        logits, _ = engine.evaluate(model, val)
        with np.load(run/'validation.npz') as z:
            error = float(abs(logits-z['logits']).max())
            changes = int((logits.argmax(-1) != z['logits'].argmax(-1)).sum())
        assert error < 2e-4 and changes == 0
        dest.mkdir(parents=True)
        rows = []
        for condition, cache in caches.items():
            logits, independent = engine.evaluate(model, cache)
            metrics = engine.grouped_metrics(logits, cache)
            delta = max(abs(independent[k]-metrics['micro'][k])
                        for k in ('region_recall', 'center_pck', 'soft_epe', 'ce'))
            assert delta < 1e-10
            path = dest/(condition+'.npz')
            np.savez_compressed(path, logits=logits, ids=cache['ids'])
            rows.append(dict(role=role, seed=seed, condition=condition, metrics=metrics,
                prediction_sha256=sha(path), independent_metric_error=delta))
        write(dest/'record.json', dict(records=rows, checkpoint_sha256=choice['checkpoint_sha256'],
            selection_sha256=sha(OUT/'selected.json'),
            manifest_sha256=sha(PARENT/'data/manifest.json'),
            validation_replay=dict(maximum_logit_error=error, prediction_changes=changes)))
        del model
        torch.cuda.empty_cache()
        print('REGION_GEOMETRY_EXPORTED', role, seed, flush=True)


def idle_gpus():
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu',
                                      '--format=csv,noheader,nounits'], text=True, timeout=10)
    rows = []
    for line in output.strip().splitlines():
        index, uuid, used, free, util = [x.strip() for x in line.split(',')]
        if int(used) <= 512 and int(free) >= 12288 and int(util) <= 10:
            rows.append(dict(index=int(index), uuid=uuid, memory_mib=int(used), free_mib=int(free), utilization=int(util)))
    return rows


def peer_active_count():
    count = 0
    for name in ('round73_native_relative5', 'projected_shape_attribution'):
        q = read(ROOT/'results'/name/'attempt1/queue_status.json')
        items = q.get('active') or []
        if isinstance(items, dict):
            items = [items]
        for row in items:
            path = Path(f"/proc/{row['pid']}/cmdline")
            if row.get('gpu') and path.exists() and f'experiments.{name}'.encode() in path.read_bytes():
                count += 1
    return count


def queue():
    check()
    assert not (OUT/'execution_identity.json').exists()
    meta = dict(pid=os.getpid(), start_ticks=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19],
        started_unix=time.time(), lock_sha256=sha(OUT/'lock.json'), source_sha256=sha(Path(__file__)),
        maximum_owned_gpu_jobs=2, launch_limit_with_observed_peer_jobs=4, goal_complete=False)
    write(OUT/'execution_identity.json', meta)
    active, completed, pending, stage = [], [], [], 'start'
    def record(state, **extra):
        write(OUT/'queue_status.json', dict(meta, state=state, stage=stage,
            active=[x[1] for x in active], completed=completed, pending=pending,
            observed_unix=time.time(), **extra))
    def stop(signum, frame):
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    def batch(action, jobs, gpu=True):
        nonlocal pending, stage
        stage, pending = action, list(jobs)
        previous = set()
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
            rows = idle_gpus() if gpu and pending else [None]
            available = {r['uuid'] for r in rows if r}
            stable = previous & available
            previous = available
            owned = {r['gpu']['uuid'] for _, r, _ in active if r['gpu']}
            limit = min(2, max(0, 4-peer_active_count())) if gpu else 1
            for device in rows:
                if not pending or len(active) >= limit:
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
                    if not any(r['uuid'] == device['uuid'] for r in idle_gpus()):
                        continue
                job = pending.pop(0)
                cmd = [sys.executable, '-m', MODULE, action]
                for key, value in job.items():
                    cmd += ['--'+key, str(value)]
                log = OUT/f'{action}_{job.get("role")}_{job.get("seed")}_{time.time_ns()}.log'
                stream = log.open('x')
                process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                item = dict(job, action=action, pid=process.pid, gpu=device, log=str(log), cuda_probe=probe, started_unix=time.time())
                active.append((process, item, stream))
                if device:
                    owned.add(device['uuid'])
                print('LAUNCHED', item, flush=True)
            record('running' if active else 'waiting_for_idle_gpu')
            time.sleep(5)
    try:
        batch('preflight', [{}])
        batch('train', [dict(role=ROLES[(i+j)%2], seed=seed) for j in range(2) for i, seed in enumerate(SEEDS)])
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
    bind()
    if args.action == 'train':
        engine.train(args.role, args.seed)
    elif args.action == 'export':
        export(args.seed)
    elif args.action == 'report':
        from experiments.region_geometry_report import report
        report()
    else:
        globals()[args.action]()
