"""Matched 256-token development runs using the unchanged R65 training loop."""
import argparse
import fcntl
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
from experiments.round65_common import ROOT, CONFIG, SEEDS, VARIANTS, read, write, sha
from experiments.round73_native_relative5 import local_dependencies, available_gpus
from models.round65_actual_gs import NativeGSClassifier
from models.round73_native_relative5 import CompactNativeClassifier
from models.native_shape_attribution import NativeFixedIsotropic

OUT = ROOT / 'results/native_matched256/attempt1'
PLAN = ROOT / 'docs/native_matched256_plan.md'
MODULE = 'experiments.native_matched256'
ROLES = ('relative5', 'distance_mlp', 'fixed_isotropic5')
OLD = ROOT / 'results/round65_actual_gs/attempt1'
PRIOR = dict(relative5=ROOT / 'results/round73_native_relative5/attempt1',
             distance_mlp=OLD, fixed_isotropic5=ROOT / 'results/native_shape_attribution/attempt1')
TRAIN_TOKENS = 256
CAP_BYTES = 16 * 2**30
ORIGINAL_EVALUATE = runner.evaluate
ORIGINAL_EVALUATION_BATCH = runner.evaluation_batch


def training_prefix(module, args):
    if not module.training:
        return None
    if len(args) != 3 or any(x.shape[1] != CONFIG['tokens'] for x in args):
        raise ValueError('training must supply the unchanged 512-token parent batch')
    return tuple(x[:, :TRAIN_TOKENS] for x in args)


def model_for(role):
    if role == 'relative5':
        model = CompactNativeClassifier(role)
    elif role == 'distance_mlp':
        model = NativeGSClassifier(role)
    elif role == 'fixed_isotropic5':
        model = NativeFixedIsotropic(role)
    else:
        raise ValueError(role)
    model.register_forward_pre_hook(training_prefix)
    return model


def evaluate(model, bundle, variant=None, limit=None):
    # Omitted condition means checkpoint selection. Explicit base remains 512.
    return ORIGINAL_EVALUATE(model, bundle, 'sparse256' if variant is None else variant, limit)


def selection_batch(bundle, start, end, variant, device):
    return ORIGINAL_EVALUATION_BATCH(bundle, start, end, 'sparse256' if variant == 'base' else variant, device)


def freeze():
    if (OUT / 'lock.json').exists():
        raise ValueError('preserve existing lock')
    paths = local_dependencies([Path(__file__), ROOT / 'experiments/native_matched256_report.py'])
    paths += [PLAN, ROOT / 'tests/test_native_matched256.py']
    prior = {}
    for role, root in PRIOR.items():
        for name in ('lock.json', 'selected.json', 'data/manifest.json'):
            prior[str(root / name)] = sha(root / name)
        for seed in SEEDS:
            for name in ('run.json', 'history.json', 'best.pt'):
                path = root / 'runs' / role / str(seed) / name
                prior[str(path)] = sha(path)
            for name in ('record.json',) + tuple(v + '.npz' for v in VARIANTS):
                path = root / 'evaluation' / role / str(seed) / name
                prior[str(path)] = sha(path)
    write(OUT / 'lock.json', dict(created_unix=time.time(), config=CONFIG, train_tokens=TRAIN_TOKENS,
        validation_variant='sparse256', roles=ROLES, seeds=SEEDS, variants=VARIANTS,
        new_fits=9, maximum_gpu_jobs=6, allocator_cap_bytes=CAP_BYTES,
        source_sha256={str(p): sha(p) for p in paths}, prior_sha256=prior,
        original_full_stream_preserved=True, new_model_formula=False, fresh_confirmation=False,
        official_test_access=False, goal_complete=False))
    (OUT / 'lock.sha256').write_text(sha(OUT / 'lock.json') + '\n')
    (OUT / 'data').symlink_to((OLD / 'data').resolve(), target_is_directory=True)
    print('MATCHED256_FROZEN', sha(OUT / 'lock.json'), flush=True)


def check(out=OUT):
    assert Path(out) == OUT
    lock = read(OUT / 'lock.json')
    assert sha(OUT / 'lock.json') == (OUT / 'lock.sha256').read_text().strip()
    assert lock['config'] == CONFIG and lock['train_tokens'] == TRAIN_TOKENS
    assert lock['roles'] == list(ROLES) and lock['seeds'] == list(SEEDS)
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
        module.checked, module.evaluate = check, evaluate
    audit.evaluation_batch = selection_batch
    data_module.checked = check


def cuda_setup():
    runner.configure()
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    torch.cuda.set_per_process_memory_fraction(CAP_BYTES / torch.cuda.get_device_properties(0).total_memory)


def preflight():
    bind()
    cuda_setup()
    assert not (OUT / 'cuda_preflight.json').exists()
    data, val = data_module.load('fit', OUT), data_module.load('val', OUT)
    index, subset, quaternion, jitter = next(runner.epoch_batches(np.random.default_rng(65011), len(data['labels'])))
    c, s, a = (torch.tensor(np.asarray(data[k][index[:, None], subset]), device='cuda')
               for k in ('centers', 'covariance', 'appearance'))
    c, s = runner.augment(c, s, quaternion, jitter)
    labels = torch.tensor(np.asarray(data['labels'][index]), device='cuda')
    rows = []
    for role in ROLES:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.manual_seed(11)
        model = model_for(role).cuda().train()
        initial = runner.shared_digest(model)
        assert initial == read(PRIOR[role] / 'runs' / role / '11/run.json')['initial_shared_sha256']
        observed = []
        hook = model.register_forward_pre_hook(lambda module, args: observed.append(tuple(x.shape[1] for x in args)))
        loss = F.cross_entropy(model(c, s, a), labels, label_smoothing=CONFIG['label_smoothing'])
        loss.backward()
        assert observed == [(TRAIN_TOKENS,) * 3]
        gradients = {n: float(p.grad.norm()) for n, p in model.position.named_parameters()}
        assert all(np.isfinite(v) and v > 0 for v in gradients.values())
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['clip_norm'], error_if_nonfinite=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
        optimizer.step()
        assert runner.shared_digest(model) != initial and torch.isfinite(loss)
        hook.remove()
        selected, _ = evaluate(model, val, limit=4)
        expected, _ = ORIGINAL_EVALUATE(model, val, 'sparse256', limit=4)
        assert np.array_equal(selected, expected)
        full, _ = evaluate(model, val, 'base', limit=4)
        assert full.shape == selected.shape == (4, 60)
        peak = torch.cuda.max_memory_allocated()
        assert peak < CAP_BYTES
        rows.append(dict(role=role, real_fit_batch=True, model_input_tokens=256,
            initial_shared_sha256=initial, loss=float(loss), position_gradients=gradients,
            default_validation_equals_explicit_sparse=True, peak_allocated=peak,
            parameters=sum(p.numel() for p in model.parameters())))
        del model, optimizer, loss
        print('MATCHED256_PREFLIGHT', role, peak, flush=True)
    write(OUT / 'cuda_preflight.json', dict(passed=True, rows=rows, lock_sha256=sha(OUT / 'lock.json'),
        gpu=torch.cuda.get_device_name(), torch_version=torch.__version__,
        development_fit_batch_only=True, private_optimizer_updates_per_role=1,
        formal_fit_updates=0, official_test_access=False))


def selections():
    bind()
    cuda_setup()
    audit.selections(OUT)
    seal = read(OUT / 'selected.json')
    seal.pop('all_fifteen_selected_before_development_evaluation')
    pairing = []
    for role in ROLES:
        for seed in SEEDS:
            new, old = OUT / 'runs' / role / str(seed), PRIOR[role] / 'runs' / role / str(seed)
            assert read(new / 'run.json')['initial_shared_sha256'] == read(old / 'run.json')['initial_shared_sha256']
            streams = lambda p: [r['stream_sha256'] for r in read(p / 'history.json')['training']]
            assert streams(new) == streams(old)
            pairing.append(dict(role=role, seed=seed, full_512_stream_matches_old=True,
                                actual_model_input='first256 of unchanged augmented512'))
    seal.update(all_nine_selected_before_development_evaluation=True, pairing=pairing,
        validation_tokens=256, formula_changed=False, fresh_confirmation=False, goal_complete=False)
    write(OUT / 'selected.json', seal)


def queue():
    check()
    guard = (OUT / 'queue.guard').open('a')
    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (OUT / 'execution_identity.json').exists()
    meta = dict(pid=os.getpid(), start_ticks=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19],
        lock_sha256=sha(OUT / 'lock.json'), maximum_owned_gpu_jobs=6, shared_gpus_enabled=False,
        started_unix=time.time(), goal_complete=False)
    write(OUT / 'execution_identity.json', meta)
    active, completed, pending, stage = [], [], [], 'preflight'

    def record(state, **extra):
        write(OUT / 'queue_status.json', dict(meta, state=state, stage=stage,
            active=[item for _, item, _ in active], completed=completed, pending=pending,
            observed_unix=time.time(), **extra))

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
                if code is not None:
                    stream.close()
                    active.remove((process, item, stream))
                    completed.append(dict(item, exit_code=code))
                    if code == 75 and action == 'train':
                        pending.append(dict(role=item['role'], seed=item['seed']))
                    elif code:
                        # Let other authorized fits finish; do not kill healthy jobs.
                        record('child_failed_other_workers_preserved', failed=item, exit_code=code)
                        failures.append(dict(item, exit_code=code))
            if failures:
                if active:
                    record('draining_after_failure', failures=failures)
                    time.sleep(5)
                    continue
                raise RuntimeError('child failures; see queue status and preserved logs')
            choices = available_gpus() if gpu and pending else [None]
            available = {r['uuid'] for r in choices if r}
            stable, previous = available & previous, available
            busy = {item['gpu']['uuid'] for _, item, _ in active if item['gpu']}
            for device in choices:
                if not pending or len(active) >= (6 if gpu else 1):
                    break
                if device and (device['uuid'] not in stable or device['uuid'] in busy):
                    continue
                env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
                    CUBLAS_WORKSPACE_CONFIG=':4096:8', CUDA_DEVICE_ORDER='PCI_BUS_ID',
                    CUDA_VISIBLE_DEVICES=device['uuid'] if device else '')
                probe = None
                if device:
                    code = 'import numpy; import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count()==1; x=torch.ones(4,device="cuda"); assert float(x.sum())==4; torch.cuda.synchronize(); print(torch.cuda.get_device_name())'
                    probe = subprocess.check_output([sys.executable, '-c', code], env=env, text=True, timeout=45).strip()
                    if not any(r['uuid'] == device['uuid'] for r in available_gpus()):
                        continue
                job = pending.pop(0)
                cmd = [sys.executable, '-m', MODULE, action]
                for name, value in job.items():
                    cmd += ['--' + name, str(value)]
                log = OUT / f'{action}_{job.get("role")}_{job.get("seed")}_{time.time_ns()}.log'
                stream = log.open('x')
                process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=stream,
                    stderr=subprocess.STDOUT, start_new_session=True)
                item = dict(job, action=action, pid=process.pid, gpu=device, cuda_probe=probe,
                    log=str(log), started_unix=time.time())
                active.append((process, item, stream))
                if device:
                    busy.add(device['uuid'])
                print('LAUNCHED', item, flush=True)
            record('running' if active else 'waiting_for_idle_gpu')
            if pending or active:
                time.sleep(5)

    failures = []
    try:
        batch('preflight', [{}])
        jobs = [dict(seed=seed, role=ROLES[(i + j) % len(ROLES)]) for i, seed in enumerate(SEEDS) for j in range(len(ROLES))]
        batch('train', jobs)
        batch('selections', [{}])
        batch('export', [dict(seed=seed) for seed in SEEDS])
        batch('report', [{}], False)
        stage = 'complete'
        record('complete', scientific_review_pending=True, visual_review_pending=True)
    except BaseException as error:
        record('controller_stopped_workers_preserved', error_type=type(error).__name__, message=str(error), failures=failures)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('freeze', 'preflight', 'queue', 'train', 'selections', 'export', 'report'))
    parser.add_argument('--role', choices=ROLES)
    parser.add_argument('--seed', type=int, choices=SEEDS)
    args = parser.parse_args()
    if args.action in ('train', 'export'):
        bind()
        cuda_setup()
        if args.action == 'train':
            raise SystemExit(runner.fit(args.role, args.seed, OUT))
        runner.export(args.seed, OUT)
    elif args.action == 'report':
        from experiments.native_matched256_report import report
        report()
    else:
        globals()[args.action]()
