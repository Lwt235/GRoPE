"""Conditional native confirmation: seal, bounded acquisition, inference, report.

No command trains or changes a model. `status` and an unmet `seal` only inspect
metadata; official PLY access starts solely after the reviewed replication entry.
"""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import numpy as np

from experiments.round65_common import ROOT, DATA_ROOT, CONFIG, VARIANTS, read, sha
from experiments.native_stream_cache import atomic_json, build_cache, MirrorProvider, schema
from experiments.native_confirmation_runtime import load_cached_population, population_groups, export_one
from experiments.native_confirmation_report import report

PROTOCOL = ROOT / 'results/native_confirmation_protocol/attempt1'
REPLICATION = ROOT / 'results/native_seed_replication/attempt1'
OUT = ROOT / 'results/native_confirmation/attempt1'
CACHE = DATA_ROOT / 'GRoPE/native_confirmation/attempt1/cache'
MODULE = 'experiments.native_confirmation_workflow'
RESERVE_BYTES = 2 * 2**30


def checked_protocol():
    lock = read(PROTOCOL / 'lock.json')
    if sha(PROTOCOL / 'lock.json') != (PROTOCOL / 'lock.sha256').read_text().strip():
        raise ValueError('protocol lock changed')
    for path, digest in lock['source_sha256'].items():
        if sha(path) != digest:
            raise ValueError('protocol source changed: ' + path)
    return read(PROTOCOL / 'plan.json'), lock


def entry_status(replication=REPLICATION, expected_lock=None):
    """Require the actual completed/reviewed result, never a progress metric."""
    replication = Path(replication)
    required = ('queue_status.json', 'result.json', 'review.json', 'selected.json', 'lock.json')
    missing = [name for name in required if not (replication / name).is_file()]
    if missing:
        return dict(ready=False, reason='replication_incomplete', missing=missing)
    status, result, review, selected, _ = [read(replication / name) for name in required]
    if expected_lock is not None and sha(replication / 'lock.json') != expected_lock:
        raise ValueError('replication lock differs from prospective protocol')
    if status['state'] != 'complete':
        return dict(ready=False, reason='replication_incomplete', stage=status.get('stage'))
    if not review.get('scientific_review_complete') or not review.get('visual_review_complete'):
        return dict(ready=False, reason='replication_review_pending')
    for key, filename in (('result_sha256', 'result.json'), ('selection_sha256', 'selected.json'),
                          ('lock_sha256', 'lock.json')):
        if review.get(key) != sha(replication / filename):
            raise ValueError('replication review does not cover current ' + filename)
    if result['lock_sha256'] != sha(replication / 'lock.json') or result['selection_sha256'] != sha(replication / 'selected.json'):
        raise ValueError('replication result identity differs')
    expected = {(role, seed) for role in ('relative5', 'distance_mlp') for seed in (44, 55, 66, 77, 88)}
    keys = [(r['role'], r['seed']) for r in selected['records']]
    if not selected.get('all_ten_selected_before_development_evaluation') or len(keys) != 10 or set(keys) != expected:
        raise ValueError('replication selection is incomplete')
    primary = next(r for r in result['cohorts']['new5'] if r['variant'] == 'base')
    if primary['seeds'] != [44, 55, 66, 77, 88]:
        raise ValueError('replication primary seeds differ')
    signal_passed = bool(primary['mean_gain_pp'] >= .5 and primary['paired_seed_bootstrap_95_pp'][0] > 0)
    if result['new_replication_signal'] != signal_passed:
        raise ValueError('replication decision differs from its original rule')
    return dict(ready=signal_passed, reason='ready' if signal_passed else 'fixed_replication_signal_absent',
                mean_gain_pp=primary['mean_gain_pp'], primary_seeds=primary['seeds'])


def resolve_models(slots):
    models, seen = [], set()
    for slot in slots:
        key = slot['role'], slot['seed']
        if key in seen:
            raise ValueError('duplicated selected model')
        seen.add(key)
        selection_path = Path(slot['selection_path'])
        selection_sha = sha(selection_path)
        candidates = [r for r in read(selection_path)['records'] if (r['role'], r['seed']) == key]
        if len(candidates) != 1:
            raise ValueError('missing or duplicated original selection')
        selected = candidates[0]
        checkpoint_sha = sha(slot['checkpoint_path'])
        if checkpoint_sha != selected['checkpoint_sha256']:
            raise ValueError('selected checkpoint changed')
        if slot['cohort'] == 'old3' and (slot['selection_sha256'] != selection_sha
                or slot['checkpoint_sha256'] != checkpoint_sha or slot['selected_epoch'] != selected['best_epoch']):
            raise ValueError('old selected model differs from prospective slot')
        models.append(dict(role=slot['role'], seed=slot['seed'], cohort=slot['cohort'],
            path=slot['checkpoint_path'], sha256=checkpoint_sha, selected_epoch=selected['best_epoch'],
            selection_path=str(selection_path), selection_sha256=selection_sha))
    return models


def population(plan):
    source = plan['population']
    for field in ('catalog', 'membership'):
        if sha(source[field + '_path']) != source[field + '_sha256']:
            raise ValueError('population metadata changed')
    catalog = read(source['catalog_path'])
    if (catalog['repository'], catalog['revision']) != (source['repository'], source['revision']):
        raise ValueError('data revision changed')
    rows = catalog['rows']
    groups = population_groups(rows, read(source['membership_path'])['rows'])
    labels = np.asarray([r['taxonomy_id'] for r in rows])
    for name, mask in groups.items():
        expected = source['groups'][name]
        if (int(mask.sum()) != expected['count'] or np.unique(labels[mask]).tolist() != expected['classes']
                or hashlib.sha256(mask.tobytes()).hexdigest() != expected['ordered_bool_mask_sha256']):
            raise ValueError('population group changed: ' + name)
    return rows, groups


def seal():
    plan, protocol_lock = checked_protocol()
    status = entry_status(expected_lock=protocol_lock['scientific_replication_lock_sha256'])
    if not status['ready']:
        return status
    if OUT.exists():
        raise ValueError('preserve existing confirmation output; use the recorded seal')
    if CACHE.exists():
        raise ValueError('unexpected cache before final seal; inspect existing identity')
    rows, _ = population(plan)
    models = resolve_models(read(plan['models']['slots_path'])['slots'])
    if len(models) != plan['models']['total_models'] or tuple(plan['models']['conditions']) != VARIANTS:
        raise ValueError('model or condition grid differs')
    from experiments.region_geometry_controls import dependencies
    from experiments.round65_acquire import REPO, REVISION
    if (REPO, REVISION) != (plan['population']['repository'], plan['population']['revision']):
        raise ValueError('transport source differs')
    sources = dict(protocol_lock['source_sha256'])
    runtime_lock = ROOT / 'results/native_confirmation_runtime_validation/attempt1/lock.json'
    for path, digest in read(runtime_lock)['source_sha256'].items():
        if sha(path) != digest:
            raise ValueError('validated runtime changed: ' + path)
        sources[path] = digest
    files = dependencies([Path(__file__), ROOT / 'experiments/round65_mirror_acquire.py',
                          ROOT / 'experiments/round65_acquire.py'])
    sources.update({str(path): sha(path) for path in files})
    prior = {str(REPLICATION / name): sha(REPLICATION / name)
             for name in ('lock.json', 'result.json', 'review.json', 'selected.json')}
    cache_bytes = sum(np.dtype(dtype).itemsize * int(np.prod(shape))
                      for dtype, shape in schema(len(rows), CONFIG['cache_tokens']).values())
    output_bytes = len(rows) * 60 * 4 * plan['models']['prediction_groups']
    final = dict(created_unix=time.time(), protocol_lock_sha256=sha(PROTOCOL / 'lock.json'),
        source_sha256=sources, prior_sha256=prior, models=models, config=CONFIG,
        conditions=plan['models']['conditions'], entry=status, cache_path=str(CACHE),
        minimum_free_bytes=RESERVE_BYTES, estimated_cache_bytes=cache_bytes,
        maximum_staging_bytes=max(r['size'] for r in rows), estimated_prediction_bytes=output_bytes,
        maximum_owned_gpu_jobs=8, new_fits=0, optimizer_updates=0, goal_complete=False)
    OUT.mkdir(parents=True)
    atomic_json(OUT / 'evaluation_lock.json', final)
    (OUT / 'evaluation_lock.sha256').write_text(sha(OUT / 'evaluation_lock.json') + '\n')
    return dict(ready=True, evaluation_lock_sha256=sha(OUT / 'evaluation_lock.json'))


def checked_seal():
    plan, _ = checked_protocol()
    lock = read(OUT / 'evaluation_lock.json')
    if (sha(OUT / 'evaluation_lock.json') != (OUT / 'evaluation_lock.sha256').read_text().strip()
            or lock['protocol_lock_sha256'] != sha(PROTOCOL / 'lock.json') or lock['config'] != CONFIG):
        raise ValueError('final evaluation lock changed')
    for mapping in ('source_sha256', 'prior_sha256'):
        for path, digest in lock[mapping].items():
            if sha(path) != digest:
                raise ValueError('sealed source changed: ' + path)
    for model in lock['models']:
        if sha(model['path']) != model['sha256'] or sha(model['selection_path']) != model['selection_sha256']:
            raise ValueError('sealed model changed')
    return lock, plan


@contextmanager
def owned_stage(name):
    """Prevent two callers from writing one cache or queue concurrently."""
    with (OUT / (name + '.process_lock')).open('a+') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def acquire():
    lock, plan = checked_seal()  # Must happen before any cache/network side effect.
    rows, _ = population(plan)
    with owned_stage('acquisition'):
        if (OUT / 'acquisition.json').exists():
            return checked_acquisition(lock)
        if shutil.disk_usage(OUT).free < lock['estimated_prediction_bytes'] + lock['minimum_free_bytes']:
            raise OSError('insufficient output space plus reserve')
        started = OUT / 'data_access_started.json'
        registry = ROOT / 'results/research_data_usage/native_confirmation_access_20261105.json'
        existing = started if started.exists() else registry
        if existing.exists():
            record = read(existing)
            if (record['evaluation_lock_sha256'] != sha(OUT / 'evaluation_lock.json')
                    or record['catalog_sha256'] != plan['population']['catalog_sha256']
                    or record['cache_path'] != lock['cache_path']):
                raise ValueError('existing access registry belongs to another evaluation')
        else:
            record = dict(started_unix=time.time(), evaluation_lock_sha256=sha(OUT / 'evaluation_lock.json'),
                catalog_sha256=plan['population']['catalog_sha256'], declared_objects=len(rows),
                cache_path=lock['cache_path'], scope='Official TEST access begins; committed rows are in cache/progress.json',
                training=False, goal_complete=False)
        for path in (started, registry):
            if path.exists():
                if read(path) != record:
                    raise ValueError('access records disagree')
            else:
                atomic_json(path, record)
        provider = MirrorProvider(rows, OUT / ('transport_' + str(time.time_ns())))
        manifest = build_cache(rows, Path(lock['cache_path']), provider, tokens=CONFIG['cache_tokens'],
                               minimum_free_bytes=lock['minimum_free_bytes'])
        result = dict(completed_unix=time.time(), cache_path=lock['cache_path'],
            cache_manifest_sha256=sha(Path(lock['cache_path']) / 'manifest.json'),
            cache_identity_sha256=manifest['identity_sha256'], count=manifest['count'],
            evaluation_lock_sha256=sha(OUT / 'evaluation_lock.json'))
        atomic_json(OUT / 'acquisition.json', result)
        return result


def checked_acquisition(lock):
    record = read(OUT / 'acquisition.json')
    if record['evaluation_lock_sha256'] != sha(OUT / 'evaluation_lock.json') or record['cache_path'] != lock['cache_path']:
        raise ValueError('acquisition belongs to another evaluation')
    manifest = Path(lock['cache_path']) / 'manifest.json'
    if sha(manifest) != record['cache_manifest_sha256'] or read(manifest)['identity_sha256'] != record['cache_identity_sha256']:
        raise ValueError('acquisition cache changed')
    return record


def export_model(role, seed):
    lock, plan = checked_seal()
    rows, groups = population(plan)
    acquisition = checked_acquisition(lock)
    model = next(m for m in lock['models'] if (m['role'], m['seed']) == (role, seed))
    bundle = load_cached_population(lock['cache_path'], rows, acquisition['cache_manifest_sha256'])
    for variant in lock['conditions']:
        destination = OUT / 'exports' / role / str(seed) / (variant + '.npz')
        export_one(bundle, model, variant, groups, destination)
        print('EXPORTED', role, seed, variant, flush=True)


def queue():
    lock, plan = checked_seal()
    checked_acquisition(lock)
    from experiments.round73_native_relative5 import available_gpus
    with owned_stage('queue'):
        if (OUT / 'result.json').exists():
            raise ValueError('result exists; preserve it for actual review')
        identity = dict(pid=os.getpid(), start_ticks=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19],
                        started_unix=time.time(), evaluation_lock_sha256=sha(OUT / 'evaluation_lock.json'))
        atomic_json(OUT / ('execution_identity_' + str(time.time_ns()) + '.json'), identity)
        pending = [dict(role=m['role'], seed=m['seed']) for m in lock['models']]
        active, completed, previous = [], [], set()

        def status(state, **extra):
            atomic_json(OUT / 'queue_status.json', dict(identity, state=state, pending=pending,
                active=[item for _, item, _ in active], completed=completed, observed_unix=time.time(),
                goal_complete=False, **extra))

        def stop(signum, frame):
            raise SystemExit(128 + signum)

        signal.signal(signal.SIGINT, stop)
        signal.signal(signal.SIGTERM, stop)
        try:
            while pending or active:
                for process, item, stream in list(active):
                    code = process.poll()
                    if code is not None:
                        stream.close()
                        active.remove((process, item, stream))
                        completed.append(dict(item, exit_code=code))
                        if code:
                            raise RuntimeError(f"{item['log']}: exit {code}")
                devices = available_gpus() if pending else []
                available = {d['uuid'] for d in devices}
                stable, previous = available & previous, available
                busy = {item['gpu']['uuid'] for _, item, _ in active}
                for device in devices:
                    if not pending or len(active) >= lock['maximum_owned_gpu_jobs']:
                        break
                    if device['uuid'] not in stable or device['uuid'] in busy:
                        continue
                    env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
                        CUBLAS_WORKSPACE_CONFIG=':4096:8', CUDA_DEVICE_ORDER='PCI_BUS_ID',
                        CUDA_VISIBLE_DEVICES=device['uuid'], PYTHONPATH=str(ROOT))
                    code = 'import numpy; import torch; assert torch.cuda.is_available(); x=torch.ones(4,device="cuda"); assert float(x.sum())==4; torch.cuda.synchronize(); print(torch.cuda.get_device_name())'
                    probe = subprocess.check_output([sys.executable, '-c', code], env=env, text=True, timeout=60).strip()
                    if device['uuid'] not in {d['uuid'] for d in available_gpus()}:
                        continue
                    job = pending.pop(0)
                    log = OUT / f"export_{job['role']}_{job['seed']}_{time.time_ns()}.log"
                    stream = log.open('x')
                    process = subprocess.Popen([sys.executable, '-m', MODULE, 'export', '--role', job['role'],
                        '--seed', str(job['seed'])], cwd=ROOT, env=env, stdout=stream,
                        stderr=subprocess.STDOUT, start_new_session=True)
                    item = dict(job, pid=process.pid, gpu=device, cuda_probe=probe, log=str(log))
                    active.append((process, item, stream))
                    busy.add(device['uuid'])
                    print('LAUNCHED', item, flush=True)
                status('running' if active else 'waiting_for_idle_gpu')
                if pending or active:
                    time.sleep(5)
            status('reporting')
            final_report()
            status('complete', scientific_review_pending=True, visual_review_pending=True)
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
            status('stopped', error=str(error))
            raise


def final_report():
    lock, plan = checked_seal()
    if (OUT / 'result.json').exists():
        raise ValueError('preserve existing result')
    rows, groups = population(plan)
    return report(OUT, lock, plan, rows, groups, checked_acquisition(lock))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('status', 'seal', 'acquire', 'queue', 'export', 'report', 'run'))
    parser.add_argument('--role')
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    if args.action == 'status':
        _, lock = checked_protocol()
        print(json.dumps(entry_status(expected_lock=lock['scientific_replication_lock_sha256'])))
    elif args.action == 'seal':
        print(json.dumps(seal()))
    elif args.action == 'acquire':
        print(json.dumps(acquire()))
    elif args.action == 'queue':
        queue()
    elif args.action == 'export':
        if args.role is None or args.seed is None:
            parser.error('export requires --role and --seed')
        export_model(args.role, args.seed)
    elif args.action == 'report':
        final_report()
    else:
        acquire()
        queue()


if __name__ == '__main__':
    main()
