"""Controlled checkpoint migration; unchanged scientific functions and budget."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from experiments import native_shape_attribution as original
from experiments.native_memmap_runtime import install, MemmapReadView
from experiments.region_geometry_controls import dependencies
from experiments.round65_common import ROOT, CONFIG, read, write, sha

OUT = Path(__file__).resolve().parent
BASE = original.OUT
RUNNER = original.runner
SCRIPT = Path(__file__).resolve()


def process(pid):
    path = Path(f'/proc/{pid}')
    try:
        fields = (path / 'stat').read_text().rsplit(')', 1)[1].split()
        return dict(pid=pid, state=fields[0], start_ticks=fields[19],
                    command=(path / 'cmdline').read_bytes().replace(b'\0', b' ').decode())
    except FileNotFoundError:
        return None


def alive(pid, ticks=None):
    row = process(pid)
    return row is not None and row['state'] != 'Z' and (ticks is None or row['start_ticks'] == str(ticks))


def freeze():
    original.check()
    assert not (OUT / 'lock.json').exists()
    paths = dependencies([SCRIPT, ROOT / 'experiments/native_memmap_runtime.py'])
    sources = {str(path): sha(path) for path in paths}
    sources[str(OUT / 'plan.json')] = sha(OUT / 'plan.json')
    for path in (BASE / 'lock.json', BASE / 'data/manifest.json', BASE / 'cuda_preflight.json'):
        sources[str(path)] = sha(path)
    write(OUT / 'lock.json', dict(source_sha256=sources, created_unix=time.time(),
        original_lock_sha256=sha(BASE / 'lock.json'), scientific_scope_changed=False))
    (OUT / 'lock.sha256').write_text(sha(OUT / 'lock.json') + '\n')
    print('NATIVE_HANDOFF_FROZEN', sha(OUT / 'lock.json'), flush=True)


def check():
    original.check()
    assert sha(OUT / 'lock.json') == (OUT / 'lock.sha256').read_text().strip()
    for name, digest in read(OUT / 'lock.json')['source_sha256'].items():
        assert sha(name) == digest, name
    return read(OUT / 'plan.json')


def snapshots(prefix):
    records = []
    for seed in (22, 33):
        folder = BASE / 'runs/fixed_isotropic5' / str(seed)
        if (folder / 'run.json').exists():
            records.append(dict(seed=seed, completed=True, run_sha256=sha(folder / 'run.json')))
            continue
        path = OUT / f'{prefix}_seed{seed}.pt'
        with (folder / 'resume.pt').open('rb') as source, path.open('xb') as target:
            target.write(source.read())
        saved = torch.load(path, map_location='cpu', weights_only=False)
        assert saved['role'] == 'fixed_isotropic5' and saved['seed'] == seed
        assert saved['lock_sha256'] == sha(BASE / 'lock.json')
        assert saved['data_sha256'] == sha(BASE / 'data/manifest.json')
        assert saved['step'] == 390 * saved['epoch_completed']
        assert len(saved['training']) == saved['epoch_completed']
        old = read(original.NATIVE / f'runs/relative5/{seed}/history.json')['training']
        assert [r['stream_sha256'] for r in saved['training']] == [r['stream_sha256'] for r in old[:saved['epoch_completed']]]
        records.append(dict(seed=seed, path=str(path), sha256=sha(path),
            epoch_completed=saved['epoch_completed'], step=saved['step'],
            all_completed_streams_match=True, optimizer_entries=len(saved['optimizer']['state']),
            sampler_type=saved['sampler_state']['bit_generator']))
    write(OUT / f'{prefix}.json', dict(records=records, observed_unix=time.time()))
    return records


def validate():
    check()
    assert not (OUT / 'validation.json').exists()
    original.bind()
    RUNNER.configure()
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    started = time.monotonic()
    records = snapshots('validation_snapshots')
    chosen = next(r for r in records if not r.get('completed'))
    saved = torch.load(chosen['path'], map_location='cpu', weights_only=False)
    data = RUNNER.load('fit', BASE)
    tensors, transfers = {}, []
    proxy = MemmapReadView()
    for key in ('centers', 'covariance', 'appearance', 'labels'):
        begin = time.monotonic()
        view = proxy.array(data[key])
        assert np.shares_memory(view, data[key]) and view.dtype == data[key].dtype
        tensors[key] = torch.tensor(view, device='cuda')
        torch.cuda.synchronize()
        transfer_seconds = time.monotonic() - begin
        source_hash, target_hash = hashlib.sha256(), hashlib.sha256()
        for i in range(0, len(view), 32):
            source_hash.update(view[i:i + 32].tobytes(order='C'))
            target_hash.update(tensors[key][i:i + 32].cpu().numpy().tobytes(order='C'))
        assert source_hash.hexdigest() == target_hash.hexdigest(), key
        transfers.append(dict(field=key, shape=list(view.shape), dtype=str(view.dtype),
            bytes=view.nbytes, transfer_seconds=transfer_seconds, sha256=source_hash.hexdigest(),
            GPU_round_trip_exact=True))
        print('TRANSFER_VERIFIED', key, transfer_seconds, flush=True)
    rng = np.random.default_rng()
    rng.bit_generator.state = copy.deepcopy(saved['sampler_state'])
    index, subset, quaternion, jitter = next(RUNNER.epoch_batches(rng, len(data['labels'])))
    reference = {k: torch.tensor(np.array(data[k][index[:, None], subset]), device='cuda')
                 for k in ('centers', 'covariance', 'appearance')}
    reference['labels'] = torch.tensor(np.array(data['labels'][index]), device='cuda')
    ii, jj = torch.as_tensor(index, device='cuda'), torch.as_tensor(subset, device='cuda')
    changed = {k: tensors[k][ii[:, None], jj] for k in ('centers', 'covariance', 'appearance')}
    changed['labels'] = tensors['labels'][ii]
    for key in reference:
        assert torch.equal(reference[key], changed[key])
        assert reference[key].stride() == changed[key].stride()
    results = []
    for batch in (reference, changed):
        model = RUNNER.NativeGSClassifier('fixed_isotropic5').cuda()
        model.load_state_dict(saved['state_dict'])
        optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
        optimizer.load_state_dict(copy.deepcopy(saved['optimizer']))
        for group in optimizer.param_groups:
            group['lr'] = RUNNER.learning_rate(saved['step'], 390)
        centers, covariance = RUNNER.augment(batch['centers'], batch['covariance'], quaternion, jitter)
        torch.manual_seed(chosen['seed'] * 10000000 + saved['step'])
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(centers, covariance, batch['appearance']), batch['labels'],
                               label_smoothing=CONFIG['label_smoothing'])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['clip_norm'], error_if_nonfinite=True)
        optimizer.step()
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        opt = {i: {k: v.detach().cpu().clone() if torch.is_tensor(v) else v for k, v in row.items()}
               for i, row in optimizer.state_dict()['state'].items()}
        results.append(dict(loss=float(loss.detach()), state=state, optimizer=opt))
        del model, optimizer, loss
    a, b = results
    assert a['loss'] == b['loss']
    assert all(torch.equal(v, b['state'][k]) for k, v in a['state'].items())
    for index, row in a['optimizer'].items():
        for key, value in row.items():
            other = b['optimizer'][index][key]
            assert torch.equal(value, other) if torch.is_tensor(value) else value == other
    del tensors, results, reference, changed
    torch.cuda.empty_cache()
    # A completed old-validation model is available; no task-evaluation input.
    folder = BASE / 'runs/fixed_isotropic5/11'
    checkpoint = torch.load(folder / 'best.pt', map_location='cpu', weights_only=True)
    model = RUNNER.NativeGSClassifier('fixed_isotropic5').cuda()
    model.load_state_dict(checkpoint['state_dict'])
    val = RUNNER.load('val', BASE)
    install(RUNNER)
    logits, _ = RUNNER.evaluate(model, val)
    with np.load(folder / 'validation.npz') as archive:
        assert np.array_equal(archive['ids'], val['ids'])
        error = float(np.max(np.abs(logits - archive['logits'])))
        changes = int((logits.argmax(1) != archive['logits'].argmax(1)).sum())
    assert error < 2e-4 and changes == 0
    write(OUT / 'validation.json', dict(passed=True, transfers=transfers,
        next_update=dict(seed=chosen['seed'], epoch=chosen['epoch_completed'], step=saved['step'],
                         inputs_exact=True, loss_exact=True, all_parameters_exact=True, all_optimizer_states_exact=True),
        old_validation_replay=dict(seed=11, maximum_logit_error=error, prediction_changes=changes),
        seconds=time.monotonic() - started, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        formal_updates=0, private_discarded_updates=2, GPU=torch.cuda.get_device_name(),
        lock_sha256=sha(OUT / 'lock.json'), no_dev_eval_or_test_access=True))
    print('HANDOFF_RUNTIME_VALIDATED', time.monotonic() - started, flush=True)


def stop_old():
    plan = check()
    assert read(OUT / 'validation.json')['passed'] and not (OUT / 'stop_record.json').exists()
    identity = plan['old_controller']
    row = process(identity['pid'])
    assert row and row['start_ticks'] == identity['start_ticks'] and identity['module'] in row['command']
    state = read(BASE / 'queue_status.json')
    assert state['stage'] == 'train' and state['state'] == 'running'
    expected = {r['pid']: r['seed'] for r in plan['old_children']}
    observed = {}
    for item in state['active']:
        child = process(item['pid'])
        if child and child['state'] != 'Z':
            assert item['pid'] in expected and item['seed'] == expected[item['pid']]
            assert 'experiments.native_shape_attribution train' in child['command']
            observed[item['pid']] = child
    assert observed
    idle = original.parent.available_gpus()
    assert len(idle) >= len(observed), 'Leave current fits running when no idle destinations exist.'
    snapshots('before_stop')
    write(OUT / 'old_queue_before_stop.json', state)
    write(OUT / 'old_execution_identity.json', read(BASE / 'execution_identity.json'))
    # The old controller owns and explicitly terminates only its own child
    # process groups. Every child already has an atomic complete-epoch snapshot.
    os.kill(identity['pid'], signal.SIGTERM)
    deadline = time.monotonic() + 45
    while alive(identity['pid'], identity['start_ticks']) or any(alive(pid, child['start_ticks']) for pid, child in observed.items()):
        if time.monotonic() > deadline:
            raise RuntimeError('Old controller/workers are still live; do not start duplicate writers.')
        time.sleep(.5)
    final = read(BASE / 'queue_status.json')
    assert final['state'] == 'stopped'
    write(OUT / 'old_queue_stopped.json', final)
    restored = snapshots('restart_snapshots')
    for record in restored:
        if not record.get('completed'):
            path = BASE / f'runs/fixed_isotropic5/{record["seed"]}/resume.pt'
            assert sha(path) == record['sha256']
    write(OUT / 'stop_record.json', dict(observed_unix=time.time(), intentional=True,
        controller=row, children=list(observed.values()), idle_destinations_observed=idle,
        all_old_processes_terminal=True, restart_snapshots=restored,
        preserved_completed_seed11_sha256=sha(BASE / 'runs/fixed_isotropic5/11/run.json'),
        unfinished_epoch_may_be_reexecuted=True, scientific_budget_unchanged=True))
    print('OWNED_NATIVE_QUEUE_STOPPED_FOR_MIGRATION', flush=True)


def worker(action, seed):
    check()
    original.bind()
    install(RUNNER)
    if action == 'train':
        raise SystemExit(RUNNER.fit('fixed_isotropic5', seed, BASE))
    if action == 'selections':
        original.selections()
    elif action == 'export':
        RUNNER.export(seed, BASE)
    else:
        from experiments.native_shape_report import report
        report()


def queue():
    plan = check()
    stop = read(OUT / 'stop_record.json')
    assert stop['all_old_processes_terminal'] and not alive(plan['old_controller']['pid'], plan['old_controller']['start_ticks'])
    assert all(not alive(r['pid'], r['start_ticks']) for r in stop['children'])
    assert not (OUT / 'execution_identity.json').exists()
    meta = dict(process(os.getpid()), lock_sha256=sha(OUT / 'lock.json'),
        original_scientific_lock_sha256=sha(BASE / 'lock.json'), started_unix=time.time(),
        source_sha256=sha(SCRIPT), controller_identity_path=str(OUT / 'execution_identity.json'),
        intentional_handoff_record=str(OUT / 'stop_record.json'), goal_complete=False)
    write(OUT / 'execution_identity.json', meta)
    active, completed, pending, stage = [], [], [], 'start'

    def status(state, **extra):
        value = dict(meta, state=state, stage=stage, active=[r[1] for r in active],
                     completed=completed, pending=pending, observed_unix=time.time(), **extra)
        write(OUT / 'queue_status.json', value)
        write(BASE / 'queue_status.json', value)

    def stop_signal(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop_signal)
    signal.signal(signal.SIGINT, stop_signal)

    def batch(action, jobs, gpu=True):
        nonlocal stage, pending
        stage, pending, previous = action, list(jobs), set()
        while pending or active:
            for proc, item, stream in list(active):
                code = proc.poll()
                if code is not None:
                    stream.close()
                    active.remove((proc, item, stream))
                    completed.append(dict(item, exit_code=code))
                    if code == 75 and action == 'train':
                        pending.append(item['seed'])
                    elif code:
                        raise RuntimeError(f'{action} seed{item["seed"]} exited {code}: {item["log"]}')
            if not pending and not active:
                break
            choices = original.parent.available_gpus() if gpu and pending else [None]
            current = {r['uuid'] for r in choices if r}
            stable, previous = current & previous, current
            owned = {r[1]['gpu']['uuid'] for r in active if r[1]['gpu']}
            for device in choices:
                if not pending or len(active) >= (3 if gpu else 1):
                    break
                if device and (device['uuid'] not in stable or device['uuid'] in owned):
                    continue
                env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
                    CUBLAS_WORKSPACE_CONFIG=':4096:8', CUDA_DEVICE_ORDER='PCI_BUS_ID',
                    CUDA_VISIBLE_DEVICES=device['uuid'] if device else '', PYTHONPATH=str(ROOT))
                probe = None
                if device:
                    code = 'import numpy; import torch; assert torch.cuda.is_available() and torch.cuda.device_count()==1; x=torch.ones(4,device="cuda"); assert float(x.sum())==4; torch.cuda.synchronize(); print(torch.cuda.get_device_name())'
                    probe = subprocess.check_output([sys.executable, '-c', code], env=env, text=True, timeout=60).strip()
                    if not any(r['uuid'] == device['uuid'] for r in original.parent.available_gpus()):
                        continue
                seed = pending.pop(0)
                command = [sys.executable, '-u', str(SCRIPT), 'worker', '--action', action]
                if seed is not None:
                    command += ['--seed', str(seed)]
                path = OUT / f'{action}_{seed}_{time.time_ns()}.log'
                stream = path.open('x')
                proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                item = dict(action=action, seed=seed, pid=proc.pid, gpu=device, cuda_probe=probe,
                            log=str(path), started_unix=time.time())
                active.append((proc, item, stream))
                if device:
                    owned.add(device['uuid'])
                print('LAUNCHED_NATIVE_RESUME', item, flush=True)
            status('running' if active else 'waiting_for_idle_gpu')
            time.sleep(5)

    try:
        seeds = [s for s in (22, 33) if not (BASE / f'runs/fixed_isotropic5/{s}/run.json').exists()]
        batch('train', seeds)
        batch('selections', [None])
        batch('export', [11, 22, 33])
        batch('report', [None], gpu=False)
        stage = 'complete'
        status('complete', scientific_review_pending=True, visual_review_pending=True)
    except BaseException as error:
        for proc, item, stream in active:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
            stream.close()
        status('stopped', error=str(error))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('freeze', 'validate', 'stop_old', 'queue', 'worker'))
    parser.add_argument('--action', choices=('train', 'selections', 'export', 'report'))
    parser.add_argument('--seed', type=int, choices=(11, 22, 33))
    args = parser.parse_args()
    if args.mode == 'worker':
        worker(args.action, args.seed)
    else:
        globals()[args.mode]()
