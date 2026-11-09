"""Finite, paired native classification experiment on common Gaussian regions."""
import argparse
import ast
import fcntl
import hashlib
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from experiments import round65_run as runner, round65_data as data_module
from experiments.round65_audit import independent_metrics
from experiments.native_memmap_runtime import install
from experiments.round65_common import ROOT, CONFIG, SEEDS, VARIANTS, read, write, sha
from models.native_region_pe import NativeRegionClassifier, ROLES, pool_regions

OUT = ROOT / 'results/native_region_pe/attempt1'
PLAN = ROOT / 'docs/native_region_pe_plan.md'
PREP = ROOT / 'results/native_region_pe/preparation'
OLD = ROOT / 'results/round65_actual_gs/attempt1'
MODULE = 'experiments.native_region_pe'
CAP_BYTES = 4 * 2**30


def source_files():
    pending = [Path(__file__), ROOT / 'experiments/native_region_pe_report.py',
               ROOT / 'tests/test_native_region_pe.py']
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
                if name.split('.')[0] in ('experiments', 'models', 'third_party') and candidate.is_file():
                    pending.append(candidate)
    return sorted(found) + [ROOT / 'third_party/region_geotransformer_reference/SOURCE.json',
                            ROOT / 'third_party/region_geotransformer_reference/LICENSE']


def freeze():
    assert not (OUT / 'lock.json').exists()
    units = read(PREP / 'cpu_checks.json')
    assert units['passed'] and units['plan_sha256'] == sha(PLAN)
    for path, digest in units['source_sha256'].items():
        assert sha(path) == digest, path
    diagnostic = ROOT / 'results/native_region_moments/diagnostic1'
    assert read(diagnostic / 'review.json')['scientific_review_complete']
    prior = [OLD / 'data/manifest.json', OLD / 'data/manifest.sha256',
             diagnostic / 'lock.json', diagnostic / 'result.json', diagnostic / 'review.json']
    write(OUT / 'lock.json', dict(created_unix=time.time(), config=CONFIG,
        roles=ROLES, seeds=SEEDS, variants=VARIANTS, groups=64, neighbors=32,
        new_fits=12, new_prediction_groups=72, bootstrap_replicates=10000,
        bootstrap_base_seed=87001, planned_contrasts=30,
        source_sha256={str(p): sha(p) for p in source_files() + [PLAN, PREP / 'cpu_checks.json']},
        prior_sha256={str(p): sha(p) for p in prior},
        maximum_gpu_jobs=4, maximum_shared_gpu_jobs=2, minimum_free_mib=13 * 1024,
        allocator_cap_bytes=CAP_BYTES, fresh_confirmation=False,
        official_test_access=False, goal_complete=False))
    (OUT / 'lock.sha256').write_text(sha(OUT / 'lock.json') + '\n')
    (OUT / 'data').symlink_to((OLD / 'data').resolve(), target_is_directory=True)
    print('NATIVE_REGION_FROZEN', sha(OUT / 'lock.json'), flush=True)


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
    runner.NativeGSClassifier, runner.ROLES, runner.SEEDS = NativeRegionClassifier, ROLES, SEEDS
    runner.checked, runner.VARIANTS = check, VARIANTS
    data_module.checked = check


def cuda_setup():
    runner.configure()
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    torch.cuda.set_per_process_memory_fraction(CAP_BYTES / torch.cuda.get_device_properties(0).total_memory)


def tensor_digest(tensors):
    digest = hashlib.sha256()
    for name, value in sorted(tensors.items()):
        a = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(a.dtype).encode())
        digest.update(str(a.shape).encode())
        digest.update(a.tobytes())
    return digest.hexdigest()


def preflight():
    bind()
    cuda_setup()
    assert not (OUT / 'cuda_preflight.json').exists()
    data = data_module.load('fit', OUT)
    # Mirror the unchanged trainer's full fit-cache residency, not just activations.
    tensors = {k: torch.tensor(np.asarray(data[k]), device='cuda')
               for k in ('centers', 'covariance', 'appearance', 'labels')}
    index, subset, quaternion, jitter = next(runner.epoch_batches(np.random.default_rng(65011), len(data['labels'])))
    ii, jj = torch.as_tensor(index, device='cuda'), torch.as_tensor(subset, device='cuda')
    c, s, a = (tensors[k][ii[:, None], jj] for k in ('centers', 'covariance', 'appearance'))
    c, s = runner.augment(c, s, quaternion, jitter)
    labels = tensors['labels'][ii]
    region = pool_regions(c, s, a)
    reference_content = reference_shared = reference_f = None
    rows = []
    for role in ROLES:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.manual_seed(11)
        model = NativeRegionClassifier(role).cuda().train()
        shared = runner.shared_digest(model)
        if reference_shared is None:
            reference_shared = shared
        assert shared == reference_shared
        initial = tensor_digest(model.position.state_dict())
        if role == 'relative5':
            reference_f = initial
        if role == 'centers_only':
            assert initial == reference_f
        observed = []
        hook = model.stem.register_forward_pre_hook(lambda module, args: observed.append(args[0].detach()))
        started = time.monotonic()
        loss = F.cross_entropy(model(c, s, a), labels, label_smoothing=CONFIG['label_smoothing'])
        assert len(observed) == 1 and observed[0].shape == (len(index), 64, 13)
        content = tensor_digest({'content': observed[0]})
        if reference_content is None:
            reference_content = content
        assert content == reference_content
        loss.backward()
        gradients = {name: None if p.grad is None else float(p.grad.norm())
                     for name, p in model.position.named_parameters()}
        assert torch.isfinite(loss) and all(v is not None and np.isfinite(v) for v in gradients.values())
        assert all(v > 0 for name, v in gradients.items() if name.endswith('weight'))
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['clip_norm'], error_if_nonfinite=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
        optimizer.step()
        torch.cuda.synchronize()
        assert runner.shared_digest(model) != shared
        peak = torch.cuda.max_memory_allocated()
        assert peak < CAP_BYTES
        rows.append(dict(role=role, batch_size=len(index), raw_tokens=512, model_tokens=64,
            real_fit_batch=True, full_fit_cache_resident=True, loss=float(loss.detach()),
            position_gradients=gradients, peak_allocated=peak,
            peak_reserved=torch.cuda.max_memory_reserved(), private_update_seconds=time.monotonic() - started,
            common_content_sha256=content, initial_shared_sha256=shared,
            initial_position_sha256=initial, parameters=sum(p.numel() for p in model.parameters()),
            position_parameters=sum(p.numel() for p in model.position.parameters())))
        hook.remove()
        del model, optimizer, loss, observed
        print('NATIVE_REGION_PREFLIGHT', role, peak, flush=True)
    write(OUT / 'cuda_preflight.json', dict(passed=True, rows=rows, formal_fit_updates=0,
        private_optimizer_updates=4, pooled_inputs_sha256=tensor_digest(region),
        official_test_access=False, gpu=torch.cuda.get_device_name(), torch_version=torch.__version__,
        lock_sha256=sha(OUT / 'lock.json')))


def train(role, seed):
    bind()
    cuda_setup()
    return runner.fit(role, seed, OUT)


@torch.no_grad()
def selections():
    bind()
    cuda_setup()
    assert not (OUT / 'selected.json').exists()
    val = data_module.load('val', OUT)
    records, shared_by_seed, streams_by_seed = [], {}, {}
    for seed in SEEDS:
        for role in ROLES:
            path = OUT / 'runs' / role / str(seed)
            run, history = read(path / 'run.json'), read(path / 'history.json')
            assert (run['role'], run['seed'], run['epochs'], run['steps']) == (role, seed, 120, 46800)
            assert run['lock_sha256'] == sha(OUT / 'lock.json')
            assert run['data_sha256'] == sha(OUT / 'data/manifest.json')
            for name, digest in run['hashes'].items():
                assert sha(path / name) == digest, name
            assert [r['epoch'] for r in history['validation']] == list(range(0, 121, 5))
            assert [r['epoch'] for r in history['training']] == list(range(1, 121))
            assert [r['step'] for r in history['training']] == [390 * e for e in range(1, 121)]
            best = min(history['validation'], key=lambda r: r['ce'])
            assert (best['epoch'], best['ce']) == (run['best_epoch'], run['best_val_ce'])
            streams = [r['stream_sha256'] for r in history['training']]
            assert run['sampling_sha256'] == streams[-1]
            assert streams_by_seed.setdefault(seed, streams) == streams
            assert shared_by_seed.setdefault(seed, run['initial_shared_sha256']) == run['initial_shared_sha256']
            assert run['shared_parameters_changed'] and run['peak_allocated'] < CAP_BYTES
            assert all(v > 0 for name, v in run['position_parameter_changes'].items() if name.endswith('weight'))
            torch.manual_seed(seed)
            model = NativeRegionClassifier(role).cuda().eval()
            assert runner.shared_digest(model) == run['initial_shared_sha256']
            checkpoint = torch.load(path / 'best.pt', map_location='cpu', weights_only=True)
            assert (checkpoint['role'], checkpoint['seed'], checkpoint['epoch']) == (role, seed, best['epoch'])
            model.load_state_dict(checkpoint['state_dict'])
            with np.load(path / 'validation.npz', allow_pickle=False) as z:
                saved = {k: z[k] for k in z.files}
            assert np.array_equal(saved['ids'], val['ids']) and np.array_equal(saved['labels'], val['labels'])
            stats = independent_metrics(saved['logits'], saved['labels'])
            assert abs(stats['ce'] - best['ce']) < 1e-8
            replay, _ = runner.evaluate(model, val)
            assert np.allclose(replay, saved['logits'], atol=2e-4, rtol=2e-4)
            args = runner.evaluation_batch(val, 0, 2, 'base', 'cuda')
            regions = pool_regions(*args)
            ordered = model.forward_pooled(regions)
            reversed_regions = {k: v.flip(1) for k, v in regions.items()}
            permuted = model.forward_pooled(reversed_regions)
            assert torch.allclose(ordered, permuted, atol=2e-4, rtol=2e-4)
            model.cpu()
            cpu_logits = model(*(x.cpu() for x in args)).numpy()
            # Cross-device FPS tie/rounding changes are recorded, not a universal gate.
            records.append(dict(role=role, seed=seed, best_epoch=best['epoch'], validation_metrics=stats,
                independent_ce_error=abs(stats['ce'] - best['ce']),
                gpu_logit_error=float(np.max(np.abs(replay - saved['logits']))),
                postpool_permutation_error=float((ordered - permuted).abs().max()),
                cpu_logit_error=float(np.max(np.abs(cpu_logits - saved['logits'][:2]))),
                cpu_prediction_disagreements=int(np.sum(cpu_logits.argmax(1) != saved['logits'][:2].argmax(1))),
                checkpoint_sha256=sha(path / 'best.pt'), run_sha256=sha(path / 'run.json')))
            print('SELECTION_CHECKED', role, seed, records[-1]['gpu_logit_error'], flush=True)
            del model, ordered, permuted
            torch.cuda.empty_cache()
    write(OUT / 'selected.json', dict(records=records, all_twelve_selected_before_development_evaluation=True,
        lock_sha256=sha(OUT / 'lock.json'), data_sha256=sha(OUT / 'data/manifest.json'),
        initialization_and_actual_epoch_streams_paired=True, confirmation_access=False))


def available_gpus():
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu',
                                     '--format=csv,noheader,nounits'], text=True, timeout=15)
    result = []
    for line in output.splitlines():
        index, uuid, used, free, utilization = [v.strip() for v in line.split(',')]
        used, free, utilization = int(used), int(free), int(utilization)
        if free >= 13 * 1024:
            result.append(dict(index=int(index), uuid=uuid, used_mib=used, free_mib=free,
                utilization=utilization, shared=not (used <= 512 and utilization <= 10)))
    return sorted(result, key=lambda d: (d['shared'], d['utilization'], -d['free_mib'], d['index']))


def queue():
    check()
    guard = (OUT / 'queue.guard').open('a')
    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (OUT / 'execution_identity.json').exists(), 'inspect existing controller and workers before any recovery'
    meta = dict(pid=os.getpid(), start_ticks=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19],
        started_unix=time.time(), lock_sha256=sha(OUT / 'lock.json'), maximum_owned_gpu_jobs=4,
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
                if not pending or len(active) >= (4 if gpu else 1):
                    break
                if device and (device['uuid'] not in stable or device['uuid'] in busy):
                    continue
                shared_count = sum(bool(r[1]['gpu'] and r[1]['gpu']['shared']) for r in active)
                if device and device['shared'] and shared_count >= 2:
                    continue
                env = dict(os.environ, PYTHONPATH=str(ROOT), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                    OPENBLAS_NUM_THREADS='4', CUBLAS_WORKSPACE_CONFIG=':4096:8', CUDA_DEVICE_ORDER='PCI_BUS_ID',
                    MPLCONFIGDIR='/tmp/grope_native_region_mpl', CUDA_VISIBLE_DEVICES=device['uuid'] if device else '')
                probe = None
                if device:
                    code = 'import numpy; import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count()==1; x=torch.ones(4,device="cuda"); assert float(x.sum())==4; torch.cuda.synchronize(); print(torch.cuda.get_device_name())'
                    probe = subprocess.check_output([sys.executable, '-c', code], env=env, text=True, timeout=45).strip()
                    fresh = next((d for d in available_gpus() if d['uuid'] == device['uuid']), None)
                    if fresh is None or (fresh['shared'] and shared_count >= 2):
                        continue
                    device = fresh
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
        jobs = [dict(seed=s, role=ROLES[(i + j) % len(ROLES)])
                for i, s in enumerate(SEEDS) for j in range(len(ROLES))]
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
        assert read(OUT / 'selected.json')['all_twelve_selected_before_development_evaluation']
        runner.export(args.seed, OUT)
    elif args.action == 'report':
        from experiments.native_region_pe_report import report
        report()
    else:
        globals()[args.action]()
