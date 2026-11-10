"""Matched old-data training with unchanged regional engine and camera inputs."""
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
from experiments import round65_queue as scheduler
from experiments.region_hrope_data import BASE, PREP, CAMERA, PLAN, check as check_data, load_cache
from experiments.region_geometry_controls import dependencies
from experiments.round65_common import ROOT, SEEDS, read, write, sha
from experiments.round65_run import configure as configure_original
from experiments.round58_data import loss_function
from experiments.round58_run import shared_digest
from models.region_hrope_control import CameraRegionMatcher, ROLES

OUT = BASE/'attempt1'
MODULE = 'experiments.region_hrope_control'
INPUT_FIELDS = ['features', 'centers', 'covariance', 'permutation', 'intrinsics', 'image_positions']


def freeze():
    check_data()
    assert not OUT.exists()
    units = read(PREP/'cpu_checks.json')
    assert units['passed'] and units['tests_passed'] == 4
    for path, digest in units['source_sha256'].items():
        assert sha(path) == digest
    data = read(CAMERA/'manifest.json')
    assert set(data['caches']) == {'fit0', 'fit1', 'val', 'eval_base'}
    sources = {str(p): sha(p) for p in dependencies([Path(__file__), ROOT/'tests/test_region_hrope_control.py'])}
    for p in (PLAN, PREP/'data_lock.json', PREP/'cpu_checks.json', CAMERA/'manifest.json'):
        sources[str(p)] = sha(p)
    write(OUT/'lock.json', dict(created_unix=time.time(), source_sha256=sources,
        settings=dict(engine.SETTINGS, roles=ROLES, input_fields=INPUT_FIELDS, conditions=['base']),
        models=list(ROLES), seeds=list(SEEDS), new_fits=15,
        camera_data_manifest_sha256=sha(CAMERA/'manifest.json'),
        hypothesis='Full Gaussian relations improve alignment beyond learned homography-structured PE under matched camera inputs.',
        frozen_comparisons=[['relative5','predicted_hrope'],['relative5','distance_mlp'],
            ['relative5','geotransformer_rpe'],['relative5','fixed_isotropic5'],
            ['predicted_hrope','distance_mlp'],['predicted_hrope','geotransformer_rpe']],
        pose_bootstrap_seeds=list(range(95001,95007)), bootstrap_replicates=10000,
        fresh_confirmation=False, new_scene_pixels_accessed=False, goal_complete=False))
    (OUT/'lock.sha256').write_text(sha(OUT/'lock.json')+'\n')
    print('HROPE_COMPARISON_FROZEN', sha(OUT/'lock.json'), flush=True)


def check():
    check_data()
    lock = read(OUT/'lock.json')
    assert sha(OUT/'lock.json') == (OUT/'lock.sha256').read_text().strip()
    for path, digest in lock['source_sha256'].items():
        assert sha(path) == digest, path
    return lock


def configure():
    configure_original()
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    torch.cuda.set_per_process_memory_fraction(8*2**30/torch.cuda.get_device_properties(0).total_memory)


def bind():
    engine.OUT = OUT
    engine.ROLES = ROLES
    engine.FixedRegionMatcher = CameraRegionMatcher
    engine.check = check
    engine.configure = configure
    engine.load_cache = load_cache
    engine.SETTINGS = dict(engine.SETTINGS, roles=ROLES, input_fields=INPUT_FIELDS, conditions=['base'])


def preflight():
    check()
    configure()
    assert not (OUT/'preflight.json').exists()
    cache = load_cache('fit0')
    rows, common = [], None
    index = np.arange(32)
    for role in ROLES:
        torch.cuda.reset_peak_memory_stats()
        torch.manual_seed(11)
        model = CameraRegionMatcher(role).cuda().train()
        digest = shared_digest(model)
        if common is None:
            common = digest
        assert digest == common
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
        losses = []
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            logits = model(*engine.inputs(cache, index))
            loss = loss_function(logits, torch.as_tensor(cache['overlap'][index], device='cuda'))
            assert torch.isfinite(loss)
            loss.backward()
            grads = {n: float(p.grad.norm()) if p.grad is not None else None for n, p in model.position.named_parameters()}
            assert all(x is not None and np.isfinite(x) for x in grads.values()) and any(x > 0 for x in grads.values())
            camera_grad = float(model.stem[0].weight.grad[:,20:].norm())
            assert np.isfinite(camera_grad) and camera_grad > 0
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
        rows.append(dict(role=role, losses=losses, final_pe_gradients=grads, camera_gradient_norm=camera_grad,
            parameters=sum(p.numel() for p in model.parameters()),
            pe_parameters=sum(p.numel() for p in model.position.parameters()),
            peak_allocated_bytes=torch.cuda.max_memory_allocated()))
        print('HROPE_PREFLIGHT', role, losses, rows[-1]['peak_allocated_bytes'], flush=True)
        del optimizer, model, loss, logits
        torch.cuda.empty_cache()
    write(OUT/'preflight.json', dict(passed=True, rows=rows, batch_size=32, updates_per_discarded_model=2,
        common_initialization_sha256=common, data='First 32 old fit0 pairs; preflight weights discarded.',
        device=torch.cuda.get_device_name(), CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],
        completed_unix=time.time(), lock_sha256=sha(OUT/'lock.json')))


def export(seed):
    check()
    configure()
    selection = read(OUT/'selected.json')
    assert len(selection['records']) == 15 and selection['all15_selected_before_eval']
    val, cache = load_cache('val'), load_cache('eval_base')
    for role in ROLES:
        choice = next(x for x in selection['records'] if (x['role'], x['seed']) == (role, seed))
        folder = OUT/'runs'/role/str(seed)
        assert sha(folder/'best.pt') == choice['checkpoint_sha256']
        model = CameraRegionMatcher(role).cuda()
        model.load_state_dict(torch.load(folder/'best.pt', map_location='cpu', weights_only=True)['state_dict'])
        val_logits, _ = engine.evaluate(model, val)
        with np.load(folder/'validation.npz', allow_pickle=False) as z:
            assert np.array_equal(val['ids'], z['ids'])
            replay_error = float(np.abs(val_logits-z['logits']).max())
            changes = int((val_logits.argmax(-1)!=z['logits'].argmax(-1)).sum())
            assert replay_error < 2e-4 and changes == 0
        logits, independent = engine.evaluate(model, cache)
        metrics = engine.grouped_metrics(logits, cache)
        metric_error = max(abs(independent[k]-metrics['micro'][k]) for k in ('region_recall','center_pck','soft_epe','ce'))
        assert metric_error < 1e-10
        path = OUT/'evaluation'/role/str(seed)/'base.npz'
        assert not path.exists()
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, logits=logits, ids=cache['ids'])
        write(path.with_suffix('.json'), dict(role=role, seed=seed, path=str(path), sha256=sha(path),
            checkpoint_sha256=choice['checkpoint_sha256'], metrics=metrics,
            validation_replay=dict(maximum_logit_error=replay_error, prediction_changes=changes),
            independent_metric_error=metric_error, lock_sha256=sha(OUT/'lock.json')))
        print('HROPE_EXPORTED', role, seed, flush=True)
        del model
        torch.cuda.empty_cache()


def queue():
    check()
    assert not (OUT/'queue_status.json').exists(), 'Inspect existing controller before restarting.'
    active, completed, failures = [], [], []
    identity = dict(pid=os.getpid(), started_unix=time.time(), lock_sha256=sha(OUT/'lock.json'))
    stage = 'starting'
    pending = []

    def record(state, **extra):
        write(OUT/'queue_status.json', dict(identity, stage=stage, state=state, pending=pending,
            active=[x[1] for x in active], completed=completed, failures=failures,
            observed_unix=time.time(), goal_complete=False, **extra))

    def batch(action, jobs, gpu=True):
        nonlocal stage, pending
        stage, pending = action, list(jobs)
        previous = set()
        while pending or active:
            for process, item, stream in list(active):
                code = process.poll()
                if code is None:
                    continue
                stream.close()
                active.remove((process, item, stream))
                done = dict(item, exit_code=code, finished_unix=time.time())
                completed.append(done)
                if code:
                    failures.append(done)
                print('HROPE_CHILD_EXIT', action, item.get('role'), item.get('seed'), code, flush=True)
            if failures:
                record('finishing_existing_jobs_after_failure')
                if not active:
                    raise RuntimeError('Child failure preserved; remaining jobs not launched.')
                time.sleep(3)
                continue
            if not pending and not active:
                break
            rows = scheduler.occupancy() if gpu else []
            free = {r['uuid'] for r in rows if r['free']}
            stable = free & previous
            owned = {item['gpu']['uuid'] for _, item, _ in active if item['gpu']}
            devices = [r for r in rows if r['uuid'] in stable-owned] if gpu else [None]
            for device in devices:
                if not pending or len(active) >= (4 if gpu else 1):
                    break
                env = dict(os.environ, OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4',
                    CUBLAS_WORKSPACE_CONFIG=':4096:8', CUDA_DEVICE_ORDER='PCI_BUS_ID',
                    CUDA_VISIBLE_DEVICES=device['uuid'] if device else '')
                probe = None
                if device:
                    test = subprocess.run([sys.executable, '-c',
                        'import numpy;import torch;assert torch.cuda.is_available();assert torch.cuda.device_count()==1;print((torch.ones(1,device="cuda")+1).item())'],
                        env=env, capture_output=True, text=True, timeout=45)
                    probe = dict(exit_code=test.returncode, stdout=test.stdout.strip(), stderr=test.stderr[-1000:])
                    if test.returncode:
                        record('cuda_probe_failed', probe=probe)
                        continue
                    if not any(r['uuid']==device['uuid'] and r['free'] for r in scheduler.occupancy()):
                        continue
                job = pending.pop(0)
                command = [sys.executable, '-u', '-m', MODULE, action]
                for key, value in job.items():
                    command += ['--'+key, str(value)]
                log = OUT/f'{action}_{job.get("role")}_{job.get("seed")}_{time.time_ns()}.log'
                stream = log.open('x')
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                item = dict(job, action=action, pid=process.pid, gpu=device, cuda_probe=probe,
                    log=str(log), started_unix=time.time())
                active.append((process, item, stream))
                print('HROPE_LAUNCHED', item, flush=True)
            previous = free
            record('running' if active else 'waiting_for_idle_gpu')
            time.sleep(3)
    try:
        batch('preflight', [{}])
        # Rotate role order across seeds, retaining the fixed paired streams.
        batch('train', [dict(role=ROLES[(j+i)%len(ROLES)], seed=seed)
                        for j in range(len(ROLES)) for i, seed in enumerate(SEEDS)])
        batch('selections', [{}], gpu=False)
        batch('export', [dict(seed=s) for s in SEEDS])
        stage = 'exports_complete'
        record('exports_complete', pose_scoring_complete=False, scientific_review_pending=True)
    except BaseException as error:
        # Only explicit controller interruption stops its own children.
        for process, item, stream in active:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
            stream.close()
        record('stopped', error_type=type(error).__name__, reason=str(error))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('freeze','preflight','train','selections','export','queue'))
    parser.add_argument('--role', choices=ROLES)
    parser.add_argument('--seed', type=int, choices=SEEDS)
    args = parser.parse_args()
    bind()
    if args.action == 'train':
        engine.train(args.role, args.seed)
    elif args.action == 'selections':
        engine.selections()
    elif args.action == 'export':
        export(args.seed)
    else:
        globals()[args.action]()
