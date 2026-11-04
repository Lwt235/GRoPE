"""Frozen regional confirmation: 24 existing models, no fitting or reselection."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np
import torch

from experiments import round68_fixed_region_run as engine
from experiments.region_geometry_controls import dependencies
from experiments.region_confirmation_common import (
    ROOT, OUT, PREP, PLAN, RAW, INVENTORY, PARENT, CANDIDATE, GEOMETRY, FIXED,
    ROLES, SEEDS, CONDITIONS, read, write, sha, preserved, run_root, check_lock, check)
from models.round68_fixed_region import FixedRegionMatcher
from models.round69_compact_relation import CompactMatcher
from models.region_geometry_controls import GeometryControlMatcher
from models.region_shape_attribution import FixedIsotropicMatcher

MODULE = 'experiments.region_confirmation'


def make_model(role):
    assert role in ROLES
    if role == 'relative5':
        return CompactMatcher(role)
    if role == 'fixed_isotropic5':
        return FixedIsotropicMatcher()
    if role in ('geotransformer_rpe', 'angle_descriptor_mlp'):
        return GeometryControlMatcher(role)
    return FixedRegionMatcher(role)


def freeze():
    from experiments.region_shape_attribution import check as check_old
    check_old()
    assert not (OUT / 'evaluation_lock.json').exists()
    cohort = check_lock(PREP / 'cohort_lock.json')
    meta = read(PREP / 'metadata_accounted.json')
    assert meta['planned_pairs'] == 960 and len(meta['planned_scenes']) == 10
    assert meta['pairs'] == 96 * len(meta['scenes'])
    assert set(meta['scenes']) | {r['scene'] for r in meta['unavailable']} == set(meta['planned_scenes'])
    assert not set(meta['scenes']) & {r['scene'] for r in meta['unavailable']}
    assert sha(RAW / 'selection.json') == meta['selection_sha256']
    assert sha(RAW / 'metadata_manifest.json') == meta['manifest_sha256']
    units = read(OUT / 'units.json')
    assert units['passed']
    for path, digest in units['source_sha256'].items():
        assert sha(path) == digest, path
    assert not any(RAW.rglob('*.png')) and not any(RAW.rglob('*.jpg'))
    sources = dict(read(FIXED / 'lock.json')['source_sha256'])
    sources.update(cohort['source_sha256'])
    files = dependencies([Path(__file__), ROOT / 'experiments/region_confirmation_data.py',
        ROOT / 'experiments/region_confirmation_report.py', ROOT / 'tests/test_region_confirmation.py'])
    files.extend([PLAN, INVENTORY, OUT / 'units.json', PREP / 'cohort_lock.json', PREP / 'scene_selection.json',
        PREP / 'metadata_accounted.json', PREP / 'availability_resolution.json',
        ROOT / 'experiments/region_confirmation_availability.py', RAW / 'selection.json', RAW / 'metadata_manifest.json',
        PARENT / 'data/manifest.json', PARENT / 'data/strata_thresholds.json'])
    files.extend(RAW / r['path'] for r in read(RAW / 'metadata_manifest.json'))
    files.extend(Path(p) for p in read(PREP / 'scene_selection.json')['listing_sha256'])
    sources.update({str(path): sha(path) for path in files})
    sources.update(read(PREP / 'availability_resolution.json')['source_sha256'])
    checkpoints = {}
    for root in (PARENT, CANDIDATE, GEOMETRY, FIXED):
        assert read(root / 'review.json')['scientific_review_complete']
        for name in ('lock.json', 'selected.json', 'result.json', 'review.json'):
            sources[str(root / name)] = sha(root / name)
    for role in ROLES:
        root = run_root(role)
        selection = read(root / 'selected.json')
        for seed in SEEDS:
            choice = next(r for r in selection['records'] if (r['role'], r['seed']) == (role, seed))
            folder = root / 'runs' / role / str(seed)
            assert sha(folder / 'best.pt') == choice['checkpoint_sha256']
            assert sha(folder / 'run.json') == choice['run_sha256']
            for name in ('run.json', 'validation.npz'):
                sources[str(folder / name)] = sha(folder / name)
            checkpoints[f'{role}/{seed}'] = dict(path=str(folder / 'best.pt'), sha256=choice['checkpoint_sha256'],
                role=role, seed=seed, selected_epoch=choice['selected_epoch'],
                selected_on='old validation only', training_lock_sha256=sha(root / 'lock.json'))
    val = read(PARENT / 'data/manifest.json')['caches']['val']
    assert sha(val['path']) == val['sha256']
    sources[val['path']] = val['sha256']
    for path, digest in sources.items():
        assert sha(path) == digest, path
    preserved(OUT / 'evaluation_lock.json', dict(roles=ROLES, seeds=SEEDS, conditions=CONDITIONS,
        source_sha256=sources, checkpoints=checkpoints, scenes=meta['scenes'], pairs=meta['pairs'],
        planned_scenes=meta['planned_scenes'], planned_pairs=960, unavailable=meta['unavailable'],
        complete_cohort=meta['complete_cohort'],
        cohort_lock_sha256=sha(PREP / 'cohort_lock.json'), created_unix=time.time(),
        new_fits=0, pixels_access_permitted=True, old_validation=val,
        hypothesis='Frozen relative5 retains regional utility and additional covariance dependence on unused places.',
        competing_explanation='Repeated old-scene development inflated utility or radial/established geometry controls suffice.',
        decisions='Use the unchanged prospective plan; utility, covariance attribution and compact tolerance reported separately.',
        inference='Original float32 batch16, complete20D content, no fitting, new normalization or model selection.',
        goal_complete=False))
    (OUT / 'evaluation_lock.sha256').write_text(sha(OUT / 'evaluation_lock.json') + '\n')
    print('REGION_EVALUATION_FROZEN', sha(OUT / 'evaluation_lock.json'), '24 old models', flush=True)


def configure():
    engine.configure()
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(2 * 2**30 / total)


@torch.inference_mode()
def infer(model, cache, device='cuda'):
    model.eval()
    size = engine.SETTINGS['eval_batch_size']
    logits = np.concatenate([model(*engine.inputs(cache, slice(i, i + size), device=device)).cpu().numpy()
                             for i in range(0, len(cache['ids']), size)])
    assert np.isfinite(logits).all()
    return logits


def old_validation(lock):
    with np.load(lock['old_validation']['path'], allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def preflight():
    lock = check()
    configure()
    assert not (OUT / 'preflight.json').exists()
    cache = old_validation(lock)
    records = []
    for role in ROLES:
        for seed in SEEDS:
            identity = lock['checkpoints'][f'{role}/{seed}']
            payload = torch.load(identity['path'], map_location='cpu', weights_only=True)
            assert (payload['role'], payload['seed'], payload['epoch']) == (role, seed, identity['selected_epoch'])
            model = make_model(role).cuda()
            model.load_state_dict(payload['state_dict'])
            torch.cuda.reset_peak_memory_stats()
            logits = infer(model, cache)
            with np.load(Path(identity['path']).parent / 'validation.npz', allow_pickle=False) as z:
                assert np.array_equal(z['ids'], cache['ids'])
                error = float(abs(logits - z['logits']).max())
                changes = int((logits.argmax(-1) != z['logits'].argmax(-1)).sum())
            assert error < 2e-4 and changes == 0, (role, seed, error, changes)
            records.append(dict(role=role, seed=seed, maximum_logit_error=error, prediction_changes=changes,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(), checkpoint_sha256=identity['sha256']))
            print('REGION_REPLAY', role, seed, error, changes, flush=True)
            del model
            torch.cuda.empty_cache()
    preserved(OUT / 'preflight.json', dict(passed=True, records=records, gpu=torch.cuda.get_device_name(),
        evaluation_lock_sha256=sha(OUT / 'evaluation_lock.json'), optimizer_updates=0,
        new_pixel_access=False, completed_unix=time.time()))


def export(role, seed):
    from experiments.region_confirmation_data import load
    lock = check()
    configure()
    assert role in ROLES and seed in SEEDS and read(OUT / 'preflight.json')['passed']
    identity = lock['checkpoints'][f'{role}/{seed}']
    dest = OUT / 'predictions' / role / str(seed)
    dest.mkdir(parents=True, exist_ok=False)
    model = make_model(role).cuda()
    model.load_state_dict(torch.load(identity['path'], map_location='cpu', weights_only=True)['state_dict'])
    records = []
    torch.cuda.reset_peak_memory_stats()
    for condition in CONDITIONS:
        cache = load(condition)
        logits = infer(model, cache)
        path = dest / (condition + '.npz')
        np.savez_compressed(path, logits=logits, ids=cache['ids'])
        records.append(dict(role=role, seed=seed, condition=condition, prediction_sha256=sha(path)))
    preserved(dest / 'identity.json', dict(records=records, checkpoint_sha256=identity['sha256'],
        evaluation_lock_sha256=sha(OUT / 'evaluation_lock.json'), data_manifest_sha256=sha(OUT / 'data_manifest.json'),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(), gpu=torch.cuda.get_device_name()))
    print('REGION_EXPORTED', role, seed, flush=True)


def available():
    owned = set()
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,process_name',
                                   '--format=csv,noheader,nounits'], text=True, timeout=10)
    for line in apps.strip().splitlines():
        uuid, process = [x.strip() for x in line.split(',', 1)]
        if '/home/liwt/' in process:
            owned.add(uuid)
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu',
                                     '--format=csv,noheader,nounits'], text=True, timeout=10)
    choices = []
    for line in output.strip().splitlines():
        index, uuid, used, free, utilization = [x.strip() for x in line.split(',')]
        if int(free) >= 12288 and uuid not in owned:
            choices.append(dict(index=int(index), uuid=uuid, used_mib=int(used), free_mib=int(free), utilization=int(utilization)))
    return sorted(choices, key=lambda r: (-r['free_mib'], r['index']))


def queue():
    check()
    assert not (OUT / 'execution_identity.json').exists()
    identity = dict(pid=os.getpid(), start_ticks=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19],
        source_sha256=sha(Path(__file__)), evaluation_lock_sha256=sha(OUT / 'evaluation_lock.json'),
        started_unix=time.time(), maximum_gpu_jobs=2)
    preserved(OUT / 'execution_identity.json', identity)
    active, completed, pending = [], [], []
    stage = 'preflight'

    def record(state, **extra):
        write(OUT / 'queue_status.json', dict(identity, state=state, stage=stage,
            active=[item for _, item, _ in active], completed=completed, pending=pending,
            observed_unix=time.time(), goal_complete=False, **extra))

    def stop(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    def batch(action, jobs, gpu):
        nonlocal stage, pending
        stage, pending = action, list(jobs)
        while pending or active:
            for process, item, stream in list(active):
                code = process.poll()
                if code is not None:
                    stream.close()
                    active.remove((process, item, stream))
                    completed.append(dict(item, exit_code=code))
                    assert code == 0, (item['log'], code)
            choices = available() if gpu and pending else [None]
            for device in choices:
                if not pending or len(active) >= (2 if gpu else 1):
                    break
                if device is not None and any(i['gpu'] and i['gpu']['uuid'] == device['uuid'] for _, i, _ in active):
                    continue
                role, seed = pending[0]
                env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
                    CUBLAS_WORKSPACE_CONFIG=':4096:8', CUDA_DEVICE_ORDER='PCI_BUS_ID',
                    CUDA_VISIBLE_DEVICES='' if device is None else device['uuid'])
                probe = None
                if device is not None:
                    code = 'import numpy; import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count()==1; x=torch.ones(4,device="cuda"); assert float(x.sum())==4; torch.cuda.synchronize(); print(torch.cuda.get_device_name())'
                    probe = subprocess.check_output([sys.executable, '-c', code], env=env, text=True, timeout=60).strip()
                    if not any(d['uuid'] == device['uuid'] for d in available()):
                        continue
                command = [sys.executable, '-m', MODULE, action]
                if role is not None:
                    command += ['--role', role, '--seed', str(seed)]
                pending.pop(0)
                log = OUT / f'{action}_{role}_{seed}_{time.time_ns()}.log'
                stream = log.open('x')
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                item = dict(pid=process.pid, action=action, role=role, seed=seed, gpu=device,
                            cuda_probe=probe, log=str(log), started_unix=time.time())
                active.append((process, item, stream))
                print('REGION_LAUNCHED', item, flush=True)
            record('running' if active else 'waiting_for_gpu_memory')
            time.sleep(5)

    try:
        batch('preflight', [(None, None)], True)
        batch('download', [(None, None)], False)
        batch('prepare', [(None, None)], False)
        batch('export', [(role, seed) for role in ROLES for seed in SEEDS], True)
        batch('report', [(None, None)], False)
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
    parser.add_argument('action', choices=('freeze', 'queue', 'preflight', 'download', 'prepare', 'export', 'report'))
    parser.add_argument('--role', choices=ROLES)
    parser.add_argument('--seed', type=int, choices=SEEDS)
    args = parser.parse_args()
    if args.action in ('download', 'prepare'):
        from experiments import region_confirmation_data as data
        engine.configure()
        getattr(data, args.action)()
    elif args.action == 'export':
        export(args.role, args.seed)
    elif args.action == 'report':
        from experiments.region_confirmation_report import report
        engine.configure()
        report()
    else:
        globals()[args.action]()
