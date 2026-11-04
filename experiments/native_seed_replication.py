"""Fixed five-pair native replication; reuse the two existing models unchanged."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np
import torch

from experiments import round65_run as runner, round65_audit as audit, round65_data as data_module
from experiments import round73_native_relative5 as parent
from experiments.round65_common import ROOT, CONFIG, VARIANTS, read, write, sha
from models.round65_actual_gs import NativeGSClassifier
from models.round73_native_relative5 import CompactNativeClassifier

OUT = ROOT / 'results/native_seed_replication/attempt1'
PLAN = ROOT / 'results/native_seed_replication/preparation/plan.json'
REGION = ROOT / 'results/region_geometry_controls/attempt1'
OLD = parent.OLD
SEEDS = (44, 55, 66, 77, 88)
OLD_SEEDS = (11, 22, 33)
ROLES = ('relative5', 'distance_mlp')
MODULE = 'experiments.native_seed_replication'


def model_for(role):
    if role == 'relative5':
        return CompactNativeClassifier(role)
    if role == 'distance_mlp':
        return NativeGSClassifier(role)
    raise ValueError(role)


def freeze():
    assert not (OUT / 'lock.json').exists()
    parent.check()
    assert read(parent.OUT / 'review.json')['scientific_review_complete']
    assert read(OUT / 'units.json')['passed']
    plan = read(PLAN)
    assert plan['fixed_comparison']['new_seeds'] == list(SEEDS)
    assert plan['fixed_comparison']['roles'] == list(ROLES)
    sources = dict(read(parent.OUT / 'lock.json')['source_sha256'])
    files = parent.local_dependencies([Path(__file__), ROOT / 'experiments/native_seed_replication_report.py',
                                      ROOT / 'tests/test_native_seed_replication.py'])
    files += [PLAN, OUT / 'units.json']
    for path in files:
        sources[str(path)] = sha(path)
    for path, digest in read(OUT / 'units.json')['source_sha256'].items():
        assert sha(path) == digest, path
    prior = {}
    for root in (parent.OUT, OLD):
        for name in ('lock.json', 'selected.json', 'review.json', 'data/manifest.json'):
            prior[str(root / name)] = sha(root / name)
    for name, digest in plan['original_results_sha256'].items():
        assert sha(ROOT / name) == digest, name
        prior[str(ROOT / name)] = digest
    for root, role in ((parent.OUT, 'relative5'), (OLD, 'distance_mlp')):
        for seed in OLD_SEEDS:
            for name in ('run.json', 'history.json', 'best.pt'):
                path = root / 'runs' / role / str(seed) / name
                prior[str(path)] = sha(path)
        assert all(not (root / 'runs' / role / str(seed)).exists() for seed in SEEDS)
    write(OUT / 'lock.json', dict(created_unix=time.time(), config=CONFIG, roles=ROLES,
        seeds=SEEDS, old_seeds=OLD_SEEDS, variants=VARIANTS, source_sha256=sources,
        prior_sha256=prior, plan_sha256=sha(PLAN), new_fits=10, new_groups=60, reused_groups=36,
        candidate_changed=False, fresh_confirmation=False, official_test_access=False, goal_complete=False))
    (OUT / 'lock.sha256').write_text(sha(OUT / 'lock.json') + '\n')
    (OUT / 'data').symlink_to((OLD / 'data').resolve(), target_is_directory=True)
    print('NATIVE_REPLICATION_FROZEN', sha(OUT / 'lock.json'), flush=True)


def check(out=OUT):
    assert Path(out) == OUT
    lock = read(OUT / 'lock.json')
    assert sha(OUT / 'lock.json') == (OUT / 'lock.sha256').read_text().strip()
    assert lock['config'] == CONFIG and lock['roles'] == list(ROLES) and lock['seeds'] == list(SEEDS)
    for group in ('source_sha256', 'prior_sha256'):
        for path, digest in lock[group].items():
            assert sha(path) == digest, path
    assert (OUT / 'data').resolve() == (OLD / 'data').resolve()
    return lock


def bind():
    check()
    for module in (runner, audit):
        module.NativeGSClassifier = model_for
        module.ROLES = ROLES
        module.SEEDS = SEEDS
        module.checked = check
    data_module.checked = check


def preflight():
    bind()
    runner.preflight(OUT / 'preflight1')
    result = read(OUT / 'preflight1/result.json')
    result.update(lock_sha256=sha(OUT / 'lock.json'), model_factory_sha256=sha(Path(__file__)),
                  formal_fit_updates=0, synthetic_optimizer_updates_per_role=1)
    write(OUT / 'cuda_preflight.json', result)


def selections():
    bind()
    audit.selections(OUT)
    selected = read(OUT / 'selected.json')
    selected.pop('all_fifteen_selected_before_development_evaluation')
    selected['all_ten_selected_before_development_evaluation'] = True
    operators = []
    for seed in SEEDS:
        path = OUT / 'runs/relative5' / str(seed) / 'best.pt'
        model = model_for('relative5')
        model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True)['state_dict'])
        c, s, _ = runner.synthetic_inputs(1, 9, 'cpu')
        inputs = audit.independent_relation(c[0].numpy(), s[0].numpy())[..., [0, 7, 8, 9, 10]]
        expected = audit.independent_projection(inputs, model.position)
        with torch.no_grad():
            actual = model.position(c, s)[0].numpy()
        np.testing.assert_allclose(actual, expected, atol=2e-4, rtol=2e-4)
        operators.append(dict(seed=seed, maximum_error=float(abs(actual - expected).max())))
    selected.update(independent_five_input_operator=operators, original_R73_criterion_unchanged=True,
                    fresh_confirmation=False, goal_complete=False)
    write(OUT / 'selected.json', selected)


def region_released():
    status = read(REGION / 'queue_status.json')
    identity = read(REGION / 'execution_identity.json')
    if status['state'] not in ('complete', 'stopped'):
        return False
    path = Path(f"/proc/{identity['pid']}/stat")
    if path.exists():
        fields = path.read_text().rsplit(')', 1)[1].split()
        if fields[19] == str(identity['start_ticks']) and fields[0] != 'Z':
            return False
    for item in status.get('active', []):
        command = Path(f"/proc/{item['pid']}/cmdline")
        if command.exists() and b'experiments.region_geometry_controls' in command.read_bytes():
            return False
    return True


def queue():
    check()
    assert not (OUT / 'execution_identity.json').exists()
    meta = dict(pid=os.getpid(), start_ticks=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19],
        source_sha256=sha(Path(__file__)), lock_sha256=sha(OUT / 'lock.json'), maximum_owned_gpu_jobs=4,
        started_unix=time.time(), resource_dependency=str(REGION), dependency_scientific_success_required=False,
        goal_complete=False)
    write(OUT / 'execution_identity.json', meta)
    active, completed, pending, stage = [], [], [], 'resource_dependency'

    def record(state, **extra):
        write(OUT / 'queue_status.json', dict(meta, state=state, stage=stage,
            active=[item for _, item, _ in active], completed=completed, pending=pending,
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
                    if code == 75 and action == 'train':
                        pending.append(dict(role=item['role'], seed=item['seed']))
                    elif code:
                        raise RuntimeError(f"{item['log']}: exit {code}")
            if not pending and not active:
                break
            choices = parent.available_gpus() if gpu and pending else [None]
            available = {r['uuid'] for r in choices if r}
            stable, previous = available & previous, available
            busy = {item['gpu']['uuid'] for _, item, _ in active if item['gpu']}
            for device in choices:
                if not pending or len(active) >= (4 if gpu else 1):
                    break
                if device and (device['uuid'] not in stable or device['uuid'] in busy):
                    continue
                env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
                    CUBLAS_WORKSPACE_CONFIG=':4096:8', CUDA_DEVICE_ORDER='PCI_BUS_ID',
                    CUDA_VISIBLE_DEVICES=device['uuid'] if device else '')
                probe = None
                if device:
                    code = 'import numpy; import torch; assert torch.cuda.is_available(); x=torch.ones(4,device="cuda"); assert float(x.sum())==4; torch.cuda.synchronize(); print(torch.cuda.get_device_name())'
                    probe = subprocess.check_output([sys.executable, '-c', code], env=env, text=True, timeout=60).strip()
                    if not any(r['uuid'] == device['uuid'] for r in parent.available_gpus()):
                        continue
                job = pending.pop(0)
                command = [sys.executable, '-m', MODULE, action]
                for key, value in job.items():
                    command += ['--' + key, str(value)]
                log = OUT / f'{action}_{job.get("role")}_{job.get("seed")}_{time.time_ns()}.log'
                stream = log.open('x')
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
                    stderr=subprocess.STDOUT, start_new_session=True)
                item = dict(job, action=action, pid=process.pid, gpu=device, cuda_probe=probe,
                            log=str(log), started_unix=time.time())
                active.append((process, item, stream))
                if device:
                    busy.add(device['uuid'])
                print('LAUNCHED', item, flush=True)
            record('running' if active else 'waiting_for_idle_gpu')
            time.sleep(5)

    try:
        while not region_released():
            record('waiting_for_region_resources')
            time.sleep(10)
        batch('preflight', [{}])
        jobs = [dict(seed=seed, role=ROLES[(i+j) % 2]) for i, seed in enumerate(SEEDS) for j in range(2)]
        batch('train', jobs)
        batch('selections', [{}])
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
    parser.add_argument('action', choices=['freeze', 'queue', 'preflight', 'train', 'selections', 'export', 'report'])
    parser.add_argument('--role', choices=ROLES)
    parser.add_argument('--seed', type=int, choices=SEEDS)
    args = parser.parse_args()
    if args.action in ('train', 'export'):
        bind()
        if args.action == 'train':
            raise SystemExit(runner.fit(args.role, args.seed, OUT))
        runner.export(args.seed, OUT)
    elif args.action == 'report':
        from experiments.native_seed_replication_report import report
        report()
    else:
        globals()[args.action]()
