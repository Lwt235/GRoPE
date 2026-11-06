"""Twelve paired fits separating two anisotropic relation components."""
import argparse
import ast
import fcntl
from functools import partial
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from experiments import round65_run as native, round65_audit as native_audit, round65_data as native_data
from experiments import region_shape_attribution as region
from experiments.native_memmap_runtime import install
from experiments.round65_common import ROOT, CONFIG, SEEDS, VARIANTS, read, write, sha
from models.round73_native_relative5 import CompactNativeClassifier
from models.anisotropy_components import ComponentMatcher, ComponentNative, ROLES
REGION_ROLES = NATIVE_ROLES = ROLES

OUT = ROOT / 'results/anisotropy_components/attempt1'
PLAN = ROOT / 'docs/anisotropy_components_plan.md'
UNITS = ROOT / 'results/anisotropy_components/preparation/cpu_checks.json'
PROVENANCE = ROOT / 'results/anisotropy_components/preparation/rationale.json'
MODULE = 'experiments.anisotropy_components'
CASES = ('region', 'native512')
CAP_BYTES = 16 * 2**30
ORIGINAL_EVALUATE = native.evaluate
ORIGINAL_BATCH = native.evaluation_batch
REGION_SETTINGS = dict(region.engine.SETTINGS, roles=REGION_ROLES)
REGION_PRIOR = dict(relative5=ROOT / 'results/round69_compact_relation/attempt1',
    distance_mlp=ROOT / 'results/round68_fixed_region/attempt1',
    fixed_isotropic5=ROOT / 'results/region_shape_attribution/attempt1',
    trace_isotropic5=ROOT / 'results/trace_isotropic/attempt1/region')
NATIVE_PRIOR = {
    'native512': dict(relative5=ROOT / 'results/round73_native_relative5/attempt1',
        distance_mlp=ROOT / 'results/round65_actual_gs/attempt1',
        fixed_isotropic5=ROOT / 'results/native_shape_attribution/attempt1',
        trace_isotropic5=ROOT / 'results/trace_isotropic/attempt1/native512')}


def case_roles(case):
    if case not in CASES:
        raise ValueError(case)
    return REGION_ROLES if case == 'region' else NATIVE_ROLES


def native_model(case, role):
    assert case == 'native512'
    return ComponentNative(role)


def native_evaluate(case, model, bundle, variant=None, limit=None):
    assert case == 'native512'
    return ORIGINAL_EVALUATE(model, bundle, 'base' if variant is None else variant, limit)


def native_selection_batch(case, bundle, start, end, variant, device):
    assert case == 'native512'
    return ORIGINAL_BATCH(bundle, start, end, variant, device)


def available_gpus():
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu',
                                     '--format=csv,noheader,nounits'], text=True, timeout=15)
    result = []
    for line in output.splitlines():
        index, uuid, used, free, utilization = [v.strip() for v in line.split(',')]
        used, free, utilization = int(used), int(free), int(utilization)
        idle = used <= 512 and free >= 34*1024 and utilization <= 10
        if idle or free >= 20*1024:
            result.append(dict(index=int(index), uuid=uuid, used_mib=used, free_mib=free,
                               utilization=utilization, shared=not idle))
    return sorted(result, key=lambda d: (d['shared'], -d['free_mib'], d['index']))


def source_files():
    """Include both 'from package.module import x' and 'from package import module'."""
    pending = [Path(__file__), ROOT / 'experiments/anisotropy_components_report.py',
               ROOT / 'tests/test_anisotropy_components.py', ROOT / 'tests/test_anisotropy_components_report.py']
    found = set()
    while pending:
        path = pending.pop().resolve()
        if path in found:
            continue
        found.add(path)
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module] + [node.module + '.' + alias.name for alias in node.names]
            else:
                continue
            for name in modules:
                candidate = ROOT / (name.replace('.', '/') + '.py')
                if name.split('.')[0] in ('experiments', 'models') and candidate.is_file():
                    pending.append(candidate)
    return sorted(found)


def freeze():
    assert not (OUT / 'lock.json').exists()
    units = read(UNITS)
    assert units['passed'] and sha(PLAN) == units['plan_sha256']
    for name, digest in units['source_sha256'].items():
        assert sha(name) == digest, name
    paths = source_files() + [PLAN, UNITS, PROVENANCE]
    prior = {}
    for case in CASES:
        mapping = REGION_PRIOR if case == 'region' else NATIVE_PRIOR[case]
        conditions = region.CONDITIONS if case == 'region' else VARIANTS
        for role, root in mapping.items():
            review_root = root.parent if role == 'trace_isotropic5' else root
            review = read(review_root / 'review.json')
            assert review.get('scientific_review_complete') or review.get('scientific_review_pending') is False
            evidence = 'result.json' if (review_root / 'result.json').exists() else 'metrics/summary.json'
            for path in (root / 'lock.json', root / 'selected.json', review_root / evidence, review_root / 'review.json'):
                prior[str(path)] = sha(path)
            for seed in SEEDS:
                for name in ('run.json', 'history.json', 'best.pt'):
                    path = root / 'runs' / role / str(seed) / name
                    prior[str(path)] = sha(path)
                for name in ('record.json',) + tuple(c + '.npz' for c in conditions):
                    path = root / 'evaluation' / role / str(seed) / name
                    prior[str(path)] = sha(path)
    for root in (REGION_PRIOR['distance_mlp'], NATIVE_PRIOR['native512']['distance_mlp']):
        path = root / 'data/manifest.json'
        prior[str(path)] = sha(path)
    caches = read(REGION_PRIOR['fixed_isotropic5'] / 'lock.json')['cache_sha256']
    for path, digest in caches.items():
        assert sha(path) == digest
    settings = {c: dict(roles=case_roles(c), seeds=SEEDS,
        settings=REGION_SETTINGS if c == 'region' else CONFIG,
        model_train_tokens=64 if c == 'region' else int(c[6:]),
        validation_variant='base') for c in CASES}
    write(OUT / 'lock.json', dict(created_unix=time.time(), source_sha256={str(p): sha(p) for p in paths},
        prior_sha256=prior, cache_sha256=caches, cases=settings, new_fits=12,
        new_prediction_groups=48, reused_prediction_groups=96,
        bootstrap_replicates=10000, bootstrap_base_seed=82001,
        maximum_gpu_jobs=6, maximum_shared_gpu_jobs=2, shared_minimum_free_mib=20480, allocator_cap_bytes=CAP_BYTES,
        fresh_confirmation=False, official_test_access=False, goal_complete=False))
    digest = sha(OUT / 'lock.json')
    (OUT / 'lock.sha256').write_text(digest + '\n')
    for case in CASES:
        path = OUT / case
        write(path / 'lock.json', dict(root_lock_sha256=digest, case=case, **settings[case]))
        (path / 'lock.sha256').write_text(sha(path / 'lock.json') + '\n')
        if case != 'region':
            (path / 'data').symlink_to((NATIVE_PRIOR['native512']['distance_mlp'] / 'data').resolve(), target_is_directory=True)
    print('ANISOTROPY_COMPONENTS_RELATION_FROZEN', digest, flush=True)


def check(case=None):
    lock = read(OUT / 'lock.json')
    assert sha(OUT / 'lock.json') == (OUT / 'lock.sha256').read_text().strip()
    for group in ('source_sha256', 'prior_sha256', 'cache_sha256'):
        for path, digest in lock[group].items():
            assert sha(path) == digest, path
    if case is None:
        return lock
    assert case in CASES
    path = OUT / case
    child = read(path / 'lock.json')
    assert sha(path / 'lock.json') == (path / 'lock.sha256').read_text().strip()
    assert child['root_lock_sha256'] == sha(OUT / 'lock.json') and child['case'] == case
    assert child['roles'] == list(case_roles(case)) and child['seeds'] == list(SEEDS)
    assert child['validation_variant'] == 'base'
    assert child['model_train_tokens'] == (64 if case == 'region' else int(case[6:]))
    if case == 'region':
        assert child['settings'] == dict(REGION_SETTINGS, roles=list(REGION_ROLES), seeds=list(SEEDS))
    else:
        assert child['settings'] == CONFIG
        assert (path / 'data').resolve() == (NATIVE_PRIOR['native512']['distance_mlp'] / 'data').resolve()
    return child


def checked_native(case, output):
    assert Path(output) == OUT / case
    return check(case)


def bind(case):
    check(case)
    if case == 'region':
        region.OUT, region.ROLES, region.MODULE = OUT / case, REGION_ROLES, MODULE
        region.FixedIsotropicMatcher = ComponentMatcher
        region.check = partial(check, case)
        region.bind()
    else:
        install(native)
        for module in (native, native_audit):
            module.NativeGSClassifier = partial(native_model, case)
            module.ROLES, module.SEEDS = NATIVE_ROLES, SEEDS
            module.checked = partial(checked_native, case)
            module.evaluate = partial(native_evaluate, case)
        native_audit.evaluation_batch = partial(native_selection_batch, case)
        native_data.checked = partial(checked_native, case)


def cuda_setup():
    native.configure()
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    torch.cuda.set_per_process_memory_fraction(CAP_BYTES / torch.cuda.get_device_properties(0).total_memory)


def region_preflight():
    output = OUT / 'region'
    assert not (output / 'preflight.json').exists()
    engine = region.engine
    engine.configure()
    cache = region.data_module.load_cache('fit0')
    index = np.flatnonzero((cache['overlap'].max(-1) >= .3).any(-1))[:4]
    rows = []
    for role in REGION_ROLES:
        torch.manual_seed(11)
        reference = region.CompactMatcher('relative5')
        torch.manual_seed(11)
        model = ComponentMatcher(role).cuda().train()
        assert all(torch.equal(value.cpu(), reference.state_dict()[key]) for key, value in model.state_dict().items())
        torch.cuda.reset_peak_memory_stats()
        loss = region.loss_function(model(*engine.inputs(cache, index)),
                                    torch.as_tensor(cache['overlap'][index], device='cuda'))
        loss.backward()
        gradients = {name: float(param.grad.norm()) for name, param in model.position.named_parameters()}
        assert torch.isfinite(loss) and all(np.isfinite(value) and value > 0 for value in gradients.values())
        peak = torch.cuda.max_memory_allocated()
        assert peak < CAP_BYTES
        rows.append(dict(role=role, full_initial_state_matches=True, real_fit_batch=True, batch_size=len(index),
            tokens=64, loss=float(loss.detach()), gradients=gradients, peak_allocated=peak,
            parameters=sum(p.numel() for p in model.parameters()),
            pe_parameters=sum(p.numel() for p in model.position.parameters())))
        del reference, model, loss
        torch.cuda.empty_cache()
    write(output / 'preflight.json', dict(passed=True, rows=rows, optimizer_updates=0,
        gpu=torch.cuda.get_device_name(), lock_sha256=sha(output / 'lock.json')))
    print('ANISOTROPY_COMPONENTS_PREFLIGHT_COMPLETE', 'region', flush=True)


def region_selections():
    output = OUT / 'region'
    assert not (output / 'selected.json').exists()
    val = region.data_module.load_cache('val')
    rows, pairs = [], []
    for role in REGION_ROLES:
        for seed in SEEDS:
            directory = output / 'runs' / role / str(seed)
            old = REGION_PRIOR['relative5'] / 'runs/relative5' / str(seed)
            run, history = read(directory / 'run.json'), read(directory / 'history.json')
            assert (run['role'], run['seed'], run['epochs'], run['steps']) == (role, seed, 60, 1080)
            assert len(history['training']) == 60
            assert run['lock_sha256'] == sha(output / 'lock.json')
            assert run['initial_shared_sha256'] == read(old / 'run.json')['initial_shared_sha256']
            streams = lambda path: [r['stream_sha256'] for r in read(path / 'history.json')['training']]
            assert streams(directory) == streams(old)
            assert all(np.isfinite(r['loss']) for r in history['training'])
            for name, digest in run['hashes'].items():
                assert sha(directory / name) == digest
            best = min(history['validation'], key=lambda r: r['selection_ce'])
            assert best['epoch'] == run['selected_epoch'] and best['selection_ce'] == run['best_val_ce']
            with np.load(directory / 'validation.npz', allow_pickle=False) as z:
                assert np.array_equal(z['ids'], val['ids'])
                values = region.engine.query_values(z['logits'], val)
            ce = np.mean([values['ce'][:, d][values['valid'][:, d]].mean() for d in (0, 1)])
            error = float(abs(ce - run['best_val_ce']))
            assert error < 1e-10
            rows.append(dict(role=role, seed=seed, selected_epoch=best['epoch'],
                checkpoint_sha256=sha(directory / 'best.pt'), run_sha256=sha(directory / 'run.json'),
                validation_ce_error=error))
            pairs.append(dict(role=role, seed=seed, all_common_initializations_and_60_streams_match_relative5=True))
    assert len(rows) == 6
    write(output / 'selected.json', dict(records=rows, old_candidate_pairing=pairs,
        all_six_selected_before_development_evaluation=True, lock_sha256=sha(output / 'lock.json')))
    print('ANISOTROPY_COMPONENTS_REGION_SELECTED', flush=True)


def region_export(seed):
    output, engine = OUT / 'region', region.engine
    selected = read(output / 'selected.json')
    assert selected['all_six_selected_before_development_evaluation'] and len(selected['records']) == 6
    assert seed in SEEDS
    for role in REGION_ROLES:
        choice = next(r for r in selected['records'] if (r['role'], r['seed']) == (role, seed))
        run, dest = output / 'runs' / role / str(seed), output / 'evaluation' / role / str(seed)
        assert not dest.exists() and sha(run / 'best.pt') == choice['checkpoint_sha256']
        assert sha(run / 'run.json') == choice['run_sha256']
        model = ComponentMatcher(role).cuda()
        model.load_state_dict(torch.load(run / 'best.pt', map_location='cpu', weights_only=True)['state_dict'])
        logits, _ = engine.evaluate(model, region.data_module.load_cache('val'))
        with np.load(run / 'validation.npz', allow_pickle=False) as z:
            error = float(abs(logits - z['logits']).max())
            changes = int((logits.argmax(-1) != z['logits'].argmax(-1)).sum())
        assert error < 2e-4 and changes == 0
        dest.mkdir(parents=True)
        rows = []
        for condition in region.CONDITIONS:
            cache = region.data_module.load_cache('eval_' + condition)
            logits, independent = engine.evaluate(model, cache)
            metrics = engine.grouped_metrics(logits, cache)
            delta = max(abs(independent[k] - metrics['micro'][k])
                        for k in ('region_recall', 'center_pck', 'soft_epe', 'ce'))
            assert delta < 1e-10
            path = dest / (condition + '.npz')
            np.savez_compressed(path, logits=logits, ids=cache['ids'])
            rows.append(dict(role=role, seed=seed, condition=condition, metrics=metrics,
                prediction_sha256=sha(path), independent_metric_error=delta))
        write(dest / 'record.json', dict(records=rows, checkpoint_sha256=choice['checkpoint_sha256'],
            selection_sha256=sha(output / 'selected.json'),
            manifest_sha256=sha(REGION_PRIOR['distance_mlp'] / 'data/manifest.json'),
            validation_replay=dict(maximum_logit_error=error, prediction_changes=changes)))
        del model
        torch.cuda.empty_cache()
    print('ANISOTROPY_COMPONENTS_REGION_EXPORTED', seed, flush=True)


def preflight(case):
    bind(case)
    cuda_setup()
    if case == 'region':
        region_preflight()
        return
    output = OUT / case
    assert not (output / 'cuda_preflight.json').exists()
    fit, val = native_data.load('fit', output), native_data.load('val', output)
    index, subset, quaternion, jitter = next(native.epoch_batches(np.random.default_rng(65011), len(fit['labels'])))
    c, s, a = (torch.tensor(np.asarray(fit[k][index[:, None], subset]), device='cuda')
               for k in ('centers', 'covariance', 'appearance'))
    c, s = native.augment(c, s, quaternion, jitter)
    labels = torch.tensor(np.asarray(fit['labels'][index]), device='cuda')
    rows = []
    for role in NATIVE_ROLES:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.manual_seed(11)
        reference = CompactNativeClassifier('relative5')
        torch.manual_seed(11)
        model = native_model(case, role).cuda().train()
        assert all(torch.equal(v.cpu(), reference.state_dict()[k]) for k, v in model.state_dict().items())
        initial = native.shared_digest(model)
        assert initial == read(NATIVE_PRIOR[case]['relative5'] / 'runs/relative5/11/run.json')['initial_shared_sha256']
        observed = []
        handle = model.register_forward_pre_hook(lambda module, args: observed.append(tuple(x.shape[1] for x in args)))
        loss = F.cross_entropy(model(c, s, a), labels, label_smoothing=CONFIG['label_smoothing'])
        loss.backward()
        assert observed == [(int(case[6:]),) * 3]
        grads = {n: float(p.grad.norm()) for n, p in model.position.named_parameters()}
        assert torch.isfinite(loss) and all(np.isfinite(x) and x > 0 for x in grads.values())
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['clip_norm'], error_if_nonfinite=True)
        opt = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
        opt.step()
        assert native.shared_digest(model) != initial
        handle.remove()
        predicted, _ = native_evaluate(case, model, val, limit=4)
        expected, _ = ORIGINAL_EVALUATE(model, val, 'base', limit=4)
        assert np.array_equal(predicted, expected)
        full, _ = native_evaluate(case, model, val, 'base', limit=4)
        assert full.shape == predicted.shape == (4, 60)
        peak = torch.cuda.max_memory_allocated()
        assert peak < CAP_BYTES
        rows.append(dict(role=role, full_initial_state_matches=True, initial_shared_sha256=initial,
            real_fit_batch=True, actual_training_tokens=int(case[6:]), loss=float(loss.detach()),
            position_gradients=grads, validation_default_verified=True, peak_allocated=peak,
            parameters=sum(p.numel() for p in model.parameters())))
        del reference, model, opt, loss
    write(output / 'cuda_preflight.json', dict(passed=True, rows=rows, formal_fit_updates=0,
        private_optimizer_updates_per_role=1, official_test_access=False,
        gpu=torch.cuda.get_device_name(), lock_sha256=sha(output / 'lock.json')))
    print('ANISOTROPY_COMPONENTS_PREFLIGHT_COMPLETE', case, flush=True)


def selections(case):
    bind(case)
    cuda_setup()
    if case == 'region':
        region_selections()
        return
    output = OUT / case
    native_audit.selections(output)
    selected = read(output / 'selected.json')
    selected.pop('all_fifteen_selected_before_development_evaluation')
    pairs = []
    for role in NATIVE_ROLES:
        for seed in SEEDS:
            new = output / 'runs' / role / str(seed)
            old = NATIVE_PRIOR[case]['relative5'] / 'runs/relative5' / str(seed)
            assert read(new / 'run.json')['initial_shared_sha256'] == read(old / 'run.json')['initial_shared_sha256']
            streams = lambda path: [r['stream_sha256'] for r in read(path / 'history.json')['training']]
            assert streams(new) == streams(old)
            pairs.append(dict(role=role, seed=seed, full_sampling_stream_matches_old_candidate=True))
    selected.update(all_six_selected_before_development_evaluation=True, old_candidate_pairing=pairs, case=case)
    write(output / 'selected.json', selected)


def seal():
    check()
    assert not (OUT / 'selected.json').exists()
    rows, identities = [], {}
    for case in CASES:
        child = OUT / case / 'selected.json'
        identities[case] = sha(child)
        selected = read(child)
        assert len(selected['records']) == len(case_roles(case)) * len(SEEDS)
        assert not (OUT / case / 'evaluation').exists()
        for row in selected['records']:
            path = OUT / case / 'runs' / row['role'] / str(row['seed']) / 'best.pt'
            assert sha(path) == row['checkpoint_sha256']
            rows.append(dict(case=case, **row))
    assert len(rows) == 12
    write(OUT / 'selected.json', dict(records=rows, case_selection_sha256=identities,
        all_twelve_selected_before_development_evaluation=True, created_unix=time.time(),
        root_lock_sha256=sha(OUT / 'lock.json'), fresh_confirmation=False))


def selected_globally():
    selected = read(OUT / 'selected.json')
    assert selected['all_twelve_selected_before_development_evaluation']
    assert selected['root_lock_sha256'] == sha(OUT / 'lock.json')
    for case, digest in selected['case_selection_sha256'].items():
        assert sha(OUT / case / 'selected.json') == digest
    return selected


def queue():
    check()
    guard = (OUT / 'queue.guard').open('a')
    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (OUT / 'execution_identity.json').exists()
    meta = dict(pid=os.getpid(), start_ticks=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19],
        started_unix=time.time(), root_lock_sha256=sha(OUT / 'lock.json'),
        maximum_owned_gpu_jobs=6, maximum_shared_gpu_jobs=2, shared_gpus_enabled=True, goal_complete=False)
    write(OUT / 'execution_identity.json', meta)
    active, completed, pending, failures, stage = [], [], [], [], 'start'

    def record(state, **extra):
        write(OUT / 'queue_status.json', dict(meta, state=state, stage=stage, active=[r[1] for r in active],
            completed=completed, pending=pending, observed_unix=time.time(), **extra))

    def stop(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    def batch(action, jobs, gpu=True):
        nonlocal stage, pending
        stage, pending, previous = action, list(jobs), set()
        while pending or active:
            for process, item, stream in list(active):
                status = process.poll()
                if status is None:
                    continue
                stream.close()
                active.remove((process, item, stream))
                completed.append(dict(item, exit_code=status))
                if status == 75 and action == 'train' and item['case'] != 'region':
                    pending.append({key: item[key] for key in ('case', 'role', 'seed')})
                elif status:
                    failures.append(dict(item, exit_code=status))
            if failures:
                if active:
                    record('draining_after_failure', failures=failures)
                    time.sleep(5)
                    continue
                raise RuntimeError('child failure; healthy jobs preserved, see queue and logs')
            choices = available_gpus() if gpu and pending else [None]
            current = {d['uuid'] for d in choices if d}
            stable, previous = current & previous, current
            busy = {r[1]['gpu']['uuid'] for r in active if r[1]['gpu']}
            for device in choices:
                if not pending or len(active) >= (6 if gpu else 1):
                    break
                if device and (device['uuid'] not in stable or device['uuid'] in busy):
                    continue
                if device and device['shared'] and sum(bool(r[1]['gpu'] and r[1]['gpu']['shared']) for r in active) >= 2:
                    continue
                env = dict(os.environ, PYTHONPATH=str(ROOT), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
                    CUBLAS_WORKSPACE_CONFIG=':4096:8', CUDA_DEVICE_ORDER='PCI_BUS_ID',
                    CUDA_VISIBLE_DEVICES=device['uuid'] if device else '')
                probe = None
                if device:
                    code = 'import numpy; import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count()==1; x=torch.ones(4,device="cuda"); assert float(x.sum())==4; torch.cuda.synchronize(); print(torch.cuda.get_device_name())'
                    probe = subprocess.check_output([sys.executable, '-c', code], env=env, text=True, timeout=45).strip()
                    fresh = next((d for d in available_gpus() if d['uuid'] == device['uuid']), None)
                    if fresh is None:
                        continue
                    device = fresh
                    if device['shared'] and sum(bool(r[1]['gpu'] and r[1]['gpu']['shared']) for r in active) >= 2:
                        continue
                job = pending.pop(0)
                command = [sys.executable, '-m', MODULE, action]
                for key, value in job.items():
                    command += ['--' + key, str(value)]
                logfile = OUT / f'{action}_{job.get("case")}_{job.get("role")}_{job.get("seed")}_{time.time_ns()}.log'
                stream = logfile.open('x')
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
                    stderr=subprocess.STDOUT, start_new_session=True)
                item = dict(job, action=action, pid=process.pid, gpu=device, cuda_probe=probe,
                    log=str(logfile), started_unix=time.time())
                active.append((process, item, stream))
                if device:
                    busy.add(device['uuid'])
                print('LAUNCHED', item, flush=True)
            record('running' if active else 'waiting_for_eligible_gpu')
            if active or pending:
                time.sleep(5)

    try:
        batch('preflight', [dict(case=c) for c in CASES])
        jobs = [dict(case=c, role=r, seed=s) for c in CASES for r in case_roles(c) for s in SEEDS]
        batch('train', jobs)
        batch('selections', [dict(case=c) for c in CASES])
        batch('seal', [{}], False)
        batch('export', [dict(case=c, seed=s) for c in CASES for s in SEEDS])
        batch('report', [{}], False)
        stage = 'complete'
        record('complete', scientific_review_pending=True, visual_review_pending=True)
    except BaseException as error:
        record('controller_stopped_workers_preserved', error_type=type(error).__name__,
            message=str(error), failures=failures)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('freeze', 'queue', 'preflight', 'train', 'selections', 'seal', 'export', 'report'))
    parser.add_argument('--case', choices=CASES)
    parser.add_argument('--role', choices=NATIVE_ROLES)
    parser.add_argument('--seed', type=int, choices=SEEDS)
    args = parser.parse_args()
    if args.action in ('freeze', 'queue', 'seal'):
        globals()[args.action]()
    elif args.action == 'report':
        from experiments.anisotropy_components_report import report
        report()
    elif args.action in ('preflight', 'selections'):
        globals()[args.action](args.case)
    else:
        bind(args.case)
        cuda_setup()
        if args.action == 'train':
            if args.case == 'region':
                region.engine.train(args.role, args.seed)
            else:
                raise SystemExit(native.fit(args.role, args.seed, OUT / args.case))
        else:
            selected_globally()
            if args.case == 'region':
                region_export(args.seed)
            else:
                native.export(args.seed, OUT / args.case)
