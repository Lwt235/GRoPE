"""Six fixed-formula fits testing matched 256/512 count augmentation."""
import argparse
import ast
import fcntl
import hashlib
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from experiments import round65_run as runner, round65_audit as audit, round65_data as data_module
from experiments.native_memmap_runtime import install
from experiments.round65_common import ROOT, CONFIG, SEEDS, read, write, sha
from models.round65_actual_gs import NativeGSClassifier
from models.round73_native_relative5 import CompactNativeClassifier

OUT = ROOT / 'results/native_count_augmentation/attempt1'
PLAN = ROOT / 'docs/native_count_augmentation_plan.md'
UNITS = ROOT / 'results/native_count_augmentation/preparation/cpu_checks.json'
MODULE = 'experiments.native_count_augmentation'
ROLES = ('relative5', 'distance_mlp')
VARIANTS = ('base', 'sparse256', 'dense1024')
OLD = ROOT / 'results/round65_actual_gs/attempt1'
PRIOR = dict(relative5=ROOT / 'results/round73_native_relative5/attempt1', distance_mlp=OLD)
CAP_BYTES = 16 * 2**30
ORIGINAL_EPOCH_BATCHES = runner.epoch_batches
SCHEDULE = None


class CountSchedule:
    """Record the consumed prefix without consuming the original sampling RNG."""
    def __init__(self, seed, directory=None, completed=0):
        self.seed = seed
        self.directory = None if directory is None else Path(directory)
        self.completed = completed
        self.current_tokens = None
        self.calls = 0

    def set_tokens(self, tokens):
        assert tokens in (256, 512)
        self.current_tokens, self.calls = tokens, 0

    def prefix(self, model, args):
        if not model.training:
            return None
        assert self.current_tokens in (256, 512), 'no scheduled training batch'
        assert len(args) == 3 and all(x.shape[1] == 512 for x in args)
        self.calls += 1
        return tuple(x[:, :self.current_tokens] for x in args)

    def batches(self, rng, count, batch_size=None, tokens=None):
        assert tokens in (None, 512)
        epoch = self.completed + 1
        digest = hashlib.sha256(f'native-count-256512-v1:{self.seed}:{epoch}'.encode()).hexdigest()
        updates, objects, exposures = {256: 0, 512: 0}, 0, 0
        for batch, arrays in enumerate(ORIGINAL_EPOCH_BATCHES(rng, count, batch_size, tokens)):
            index, subset, quaternion, jitter = arrays
            n = 256 if batch % 2 == 0 else 512
            self.set_tokens(n)
            yield arrays
            assert self.calls == 1, 'each sampled batch must reach the model exactly once'
            digest = runner.stream_update(digest, np.asarray([n], dtype=np.int64),
                index, subset[:, :n], quaternion, jitter[:, :n])
            updates[n] += 1
            objects += len(index)
            exposures += len(index) * n
        assert objects == count
        record = dict(epoch=epoch, seed=self.seed, batches=sum(updates.values()),
            updates_by_tokens={str(k): v for k, v in updates.items()}, objects=objects,
            token_exposures=exposures, effective_input_sha256=digest)
        if self.directory is not None:
            path = self.directory / f'epoch_{epoch:04d}.json'
            if path.exists():
                assert read(path) == record, 'replayed epoch changed after resume'
            else:
                write(path, record)
        self.completed = epoch
        self.current_tokens = None


def training_prefix(model, args):
    if not model.training:
        return None
    assert SCHEDULE is not None, 'training requires the fixed count schedule'
    return SCHEDULE.prefix(model, args)


def model_for(role):
    if role == 'relative5':
        model = CompactNativeClassifier(role)
    elif role == 'distance_mlp':
        model = NativeGSClassifier(role)
    else:
        raise ValueError(role)
    model.register_forward_pre_hook(training_prefix)
    return model


def source_files():
    pending = [Path(__file__), ROOT / 'experiments/native_count_augmentation_report.py',
               ROOT / 'tests/test_native_count_augmentation.py']
    found = set()
    while pending:
        path = pending.pop().resolve()
        if path in found:
            continue
        found.add(path)
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                modules = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module] + [node.module + '.' + a.name for a in node.names]
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
    assert units['passed'] and units['plan_sha256'] == sha(PLAN)
    for path, digest in units['source_sha256'].items():
        assert sha(path) == digest, path
    prior = {}
    for role, root in PRIOR.items():
        review = read(root / 'review.json')
        assert review.get('scientific_review_complete') or review.get('scientific_review_pending') is False
        evidence = 'result.json' if (root / 'result.json').exists() else 'metrics/summary.json'
        for name in ('lock.json', 'selected.json', 'review.json', evidence, 'data/manifest.json'):
            prior[str(root / name)] = sha(root / name)
        for seed in SEEDS:
            for name in ('run.json', 'history.json', 'best.pt'):
                path = root / 'runs' / role / str(seed) / name
                prior[str(path)] = sha(path)
            for name in ('record.json',) + tuple(v + '.npz' for v in VARIANTS):
                path = root / 'evaluation' / role / str(seed) / name
                prior[str(path)] = sha(path)
    paths = source_files() + [PLAN, UNITS]
    write(OUT / 'lock.json', dict(created_unix=time.time(), config=CONFIG, roles=ROLES,
        seeds=SEEDS, variants=VARIANTS, train_schedule='each_epoch_alternating_256_512',
        validation_variant='base', original_full_512_stream_preserved=True,
        source_sha256={str(p): sha(p) for p in paths}, prior_sha256=prior,
        new_fits=6, new_prediction_groups=18, reused_prediction_groups=18,
        bootstrap_replicates=10000, bootstrap_base_seed=84001, planned_contrasts=18,
        maximum_gpu_jobs=6, maximum_shared_gpu_jobs=2, shared_minimum_free_mib=20480,
        allocator_cap_bytes=CAP_BYTES, fresh_confirmation=False, official_test_access=False,
        goal_complete=False))
    (OUT / 'lock.sha256').write_text(sha(OUT / 'lock.json') + '\n')
    (OUT / 'data').symlink_to((OLD / 'data').resolve(), target_is_directory=True)
    print('COUNT_AUGMENTATION_FROZEN', sha(OUT / 'lock.json'), flush=True)


def check(out=OUT):
    assert Path(out) == OUT
    lock = read(OUT / 'lock.json')
    assert sha(OUT / 'lock.json') == (OUT / 'lock.sha256').read_text().strip()
    assert lock['config'] == CONFIG and lock['roles'] == list(ROLES)
    assert lock['seeds'] == list(SEEDS) and lock['variants'] == list(VARIANTS)
    for group in ('source_sha256', 'prior_sha256'):
        for path, digest in lock[group].items():
            assert sha(path) == digest, path
    assert (OUT / 'data').resolve() == (OLD / 'data').resolve()
    return lock


def bind():
    check()
    install(runner)
    for module in (runner, audit):
        module.NativeGSClassifier, module.ROLES, module.SEEDS = model_for, ROLES, SEEDS
        module.checked, module.VARIANTS = check, VARIANTS
    data_module.checked = check


def cuda_setup():
    runner.configure()
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    torch.cuda.set_per_process_memory_fraction(CAP_BYTES / torch.cuda.get_device_properties(0).total_memory)


def preflight():
    global SCHEDULE
    bind()
    cuda_setup()
    assert not (OUT / 'cuda_preflight.json').exists()
    data = data_module.load('fit', OUT)
    index, subset, quaternion, jitter = next(ORIGINAL_EPOCH_BATCHES(np.random.default_rng(65011), len(data['labels'])))
    c, s, a = (torch.tensor(np.asarray(data[k][index[:, None], subset]), device='cuda')
               for k in ('centers', 'covariance', 'appearance'))
    c, s = runner.augment(c, s, quaternion, jitter)
    labels = torch.tensor(np.asarray(data['labels'][index]), device='cuda')
    rows = []
    for role in ROLES:
        for n in (256, 512):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            SCHEDULE = CountSchedule(11)
            SCHEDULE.set_tokens(n)
            torch.manual_seed(11)
            model = model_for(role).cuda().train()
            initial = runner.shared_digest(model)
            assert initial == read(PRIOR[role] / 'runs' / role / '11/run.json')['initial_shared_sha256']
            observed = []
            hook = model.register_forward_pre_hook(lambda module, args: observed.append(tuple(x.shape[1] for x in args)))
            loss = F.cross_entropy(model(c, s, a), labels, label_smoothing=CONFIG['label_smoothing'])
            loss.backward()
            assert observed == [(n,) * 3] and SCHEDULE.calls == 1
            gradients = {name: float(p.grad.norm()) for name, p in model.position.named_parameters()}
            assert torch.isfinite(loss) and all(np.isfinite(v) and v > 0 for v in gradients.values())
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['clip_norm'], error_if_nonfinite=True)
            optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
            optimizer.step()
            assert runner.shared_digest(model) != initial
            hook.remove()
            peak = torch.cuda.max_memory_allocated()
            assert peak < CAP_BYTES
            rows.append(dict(role=role, model_input_tokens=n, real_fit_batch=True, batch_size=len(index),
                loss=float(loss.detach()), position_gradients=gradients, peak_allocated=peak,
                initial_shared_sha256=initial, parameters=sum(p.numel() for p in model.parameters())))
            del model, optimizer, loss
            print('COUNT_PREFLIGHT', role, n, peak, flush=True)
    SCHEDULE = None
    write(OUT / 'cuda_preflight.json', dict(passed=True, rows=rows, formal_fit_updates=0,
        private_optimizer_updates=4, official_test_access=False, gpu=torch.cuda.get_device_name(),
        lock_sha256=sha(OUT / 'lock.json')))


def train(role, seed):
    global SCHEDULE
    bind()
    cuda_setup()
    dest = OUT / 'runs' / role / str(seed)
    resume = dest / 'resume.pt'
    completed = 0 if not resume.exists() else torch.load(resume, map_location='cpu', weights_only=False)['epoch_completed']
    SCHEDULE = CountSchedule(seed, dest / 'sampling', completed)
    runner.epoch_batches = SCHEDULE.batches
    status = runner.fit(role, seed, OUT)
    if status == 0:
        rows = [read(dest / 'sampling' / f'epoch_{e:04d}.json') for e in range(1, CONFIG['epochs'] + 1)]
        assert SCHEDULE.completed == CONFIG['epochs']
        summary = dict(records=rows, train_schedule='each_epoch_alternating_256_512',
            total_updates=sum(r['batches'] for r in rows),
            updates_by_tokens={str(n): sum(r['updates_by_tokens'][str(n)] for r in rows) for n in (256, 512)},
            total_token_exposures=sum(r['token_exposures'] for r in rows))
        write(dest / 'sampling_summary.json', summary)
        run = read(dest / 'run.json')
        run['hashes']['sampling_summary.json'] = sha(dest / 'sampling_summary.json')
        run['training_schedule'] = summary['train_schedule']
        write(dest / 'run.json', run)
    return status


def selections():
    bind()
    cuda_setup()
    audit.selections(OUT)
    seal = read(OUT / 'selected.json')
    seal.pop('all_fifteen_selected_before_development_evaluation')
    pairs, effective = [], {}
    for role in ROLES:
        for seed in SEEDS:
            new, old = OUT / 'runs' / role / str(seed), PRIOR[role] / 'runs' / role / str(seed)
            nr, oldr = read(new / 'run.json'), read(old / 'run.json')
            assert nr['initial_shared_sha256'] == oldr['initial_shared_sha256']
            assert (nr['parameters'], nr['position_parameters']) == (oldr['parameters'], oldr['position_parameters'])
            streams = lambda p: [r['stream_sha256'] for r in read(p / 'history.json')['training']]
            assert streams(new) == streams(old)
            sampling = read(new / 'sampling_summary.json')
            assert sampling['total_updates'] == 46800 and sampling['updates_by_tokens'] == {'256': 23400, '512': 23400}
            rows = sampling['records']
            assert [r['epoch'] for r in rows] == list(range(1, 121))
            for row in rows:
                assert row == read(new / 'sampling' / f"epoch_{row['epoch']:04d}.json")
                assert row['objects'] == 6227 and row['batches'] == 390
                assert row['updates_by_tokens'] == {'256': 195, '512': 195}
            signatures = [r['effective_input_sha256'] for r in rows]
            assert effective.setdefault(seed, signatures) == signatures
            pairs.append(dict(role=role, seed=seed, full_512_stream_matches_original=True,
                effective_count_stream_paired=True, sampling_sha256=sha(new / 'sampling_summary.json')))
    seal.update(all_six_selected_before_development_evaluation=True, pairing=pairs,
        validation_tokens=512, created_unix=time.time(), fresh_confirmation=False)
    write(OUT / 'selected.json', seal)


def available_gpus():
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu',
                                     '--format=csv,noheader,nounits'], text=True, timeout=15)
    result = []
    for line in output.splitlines():
        index, uuid, used, free, utilization = [v.strip() for v in line.split(',')]
        used, free, utilization = int(used), int(free), int(utilization)
        idle = used <= 512 and free >= 34 * 1024 and utilization <= 10
        if idle or free >= 20 * 1024:
            result.append(dict(index=int(index), uuid=uuid, used_mib=used, free_mib=free,
                utilization=utilization, shared=not idle))
    return sorted(result, key=lambda d: (d['shared'], d['utilization'], -d['free_mib'], d['index']))


def queue():
    check()
    guard = (OUT / 'queue.guard').open('a')
    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (OUT / 'execution_identity.json').exists()
    meta = dict(pid=os.getpid(), start_ticks=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19],
        started_unix=time.time(), lock_sha256=sha(OUT / 'lock.json'), maximum_owned_gpu_jobs=6,
        maximum_shared_gpu_jobs=2, shared_gpus_enabled=True, goal_complete=False)
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
                code = process.poll()
                if code is None:
                    continue
                stream.close()
                active.remove((process, item, stream))
                completed.append(dict(item, exit_code=code))
                if code == 75 and action == 'train':
                    pending.append(dict(role=item['role'], seed=item['seed']))
                elif code:
                    failures.append(dict(item, exit_code=code))
            if failures:
                if active:
                    record('draining_after_failure', failures=failures)
                    time.sleep(5)
                    continue
                raise RuntimeError('child failed; healthy workers preserved, see logs')
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
                env = dict(os.environ, PYTHONPATH=str(ROOT), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                    OPENBLAS_NUM_THREADS='4', CUBLAS_WORKSPACE_CONFIG=':4096:8', CUDA_DEVICE_ORDER='PCI_BUS_ID',
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
                cmd = [sys.executable, '-m', MODULE, action]
                for key, value in job.items():
                    cmd += ['--' + key, str(value)]
                logfile = OUT / f'{action}_{job.get("role")}_{job.get("seed")}_{time.time_ns()}.log'
                stream = logfile.open('x')
                process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=stream,
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
        batch('preflight', [{}])
        jobs = [dict(seed=s, role=ROLES[(i + j) % 2]) for i, s in enumerate(SEEDS) for j in range(2)]
        batch('train', jobs)
        batch('selections', [{}])
        batch('export', [dict(seed=s) for s in SEEDS])
        batch('report', [{}], False)
        stage = 'complete'
        record('complete', scientific_review_pending=True, visual_review_pending=True)
    except BaseException as error:
        record('controller_stopped_workers_preserved', error_type=type(error).__name__, message=str(error), failures=failures)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('freeze', 'queue', 'preflight', 'train', 'selections', 'export', 'report'))
    parser.add_argument('--role', choices=ROLES)
    parser.add_argument('--seed', type=int, choices=SEEDS)
    args = parser.parse_args()
    if args.action == 'train':
        raise SystemExit(train(args.role, args.seed))
    elif args.action == 'export':
        bind()
        cuda_setup()
        assert read(OUT / 'selected.json')['all_six_selected_before_development_evaluation']
        runner.export(args.seed, OUT)
    elif args.action == 'report':
        from experiments.native_count_augmentation_report import report
        report()
    else:
        globals()[args.action]()
