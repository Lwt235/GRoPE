"""Frozen depth-only acquisition and unchanged R68 fixed-query construction."""
import concurrent.futures
from pathlib import Path
import time
import urllib.request

import numpy as np
from PIL import Image

from experiments.round68_fixed_region_data import collect, strata
from experiments.region_confirmation_common import (
    OUT, PREP, RAW, CACHE, PARENT, read, write, sha, preserved, check)


def download():
    check()
    assert read(OUT / 'preflight.json')['passed']
    assert not (OUT / 'download_complete.json').exists()
    if not (OUT / 'data_access_started.json').exists():
        preserved(OUT / 'data_access_started.json', dict(unix=time.time(),
            evaluation_lock_sha256=sha(OUT / 'evaluation_lock.json'),
            scope='First frozen regional confirmation depth acquisition; these places are now used.'))
    jobs = {}
    for scene in read(RAW / 'selection.json')['scenes']:
        for frame in scene['frames']:
            path = RAW / scene['scene'] / 'depth' / frame['depth']
            jobs[str(path)] = (scene['base_url'] + 'depth/' + frame['depth'], path)

    def fetch(job):
        url, path = job
        assert url.startswith('https://sun3d.cs.princeton.edu/data/') and path.suffix == '.png'
        path.parent.mkdir(parents=True, exist_ok=True)
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        for attempt in range(3):
            try:
                if not path.exists():
                    partial = path.with_suffix('.png.part')
                    with opener.open(url, timeout=35) as response, partial.open('wb') as stream:
                        while chunk := response.read(2**20):
                            stream.write(chunk)
                    with Image.open(partial) as im:
                        im.verify()
                    partial.replace(path)
                with Image.open(path) as im:
                    im.verify()
                return dict(path=str(path), url=url, bytes=path.stat().st_size, sha256=sha(path))
            except Exception as error:
                print('DEPTH_RETRY', url, attempt + 1, str(error), flush=True)
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)

    records, failures = [], []
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        tasks = {pool.submit(fetch, job): job[0] for job in jobs.values()}
        for task in concurrent.futures.as_completed(tasks):
            try:
                records.append(task.result())
            except Exception as error:
                failures.append(dict(url=tasks[task], error=str(error)))
            if (len(records) + len(failures)) % 40 == 0:
                progress = dict(completed=len(records), total=len(jobs), failures=failures, unix=time.time())
                write(OUT / 'download_progress.json', progress)
                print('REGION_DEPTHS', len(records), '/', len(jobs), flush=True)
    if failures:
        write(OUT / f'download_failure_{time.time_ns()}.json', dict(failures=failures, completed=len(records)))
        raise RuntimeError('Fixed downloads failed; preserve and retry the same sources.')
    records.sort(key=lambda r: r['path'])
    preserved(OUT / 'download_manifest.json', records)
    preserved(OUT / 'download_complete.json', dict(files=len(records), bytes=sum(r['bytes'] for r in records),
        manifest_sha256=sha(OUT / 'download_manifest.json'), evaluation_lock_sha256=sha(OUT / 'evaluation_lock.json'),
        unix=time.time(), proxy_policy='disabled including redirects', rgb_downloaded=False))
    print('REGION_DOWNLOAD_COMPLETE', len(records), flush=True)


def raw_lock(selection, downloads):
    lookup = {r['path']: r for r in downloads}
    entries, hashes = [], {}
    for row in selection['scenes']:
        k, poses = RAW / row['scene'] / 'intrinsics.txt', RAW / row['pose_file']
        assert sha(k) == row['K_sha256'] and sha(poses) == row['pose_sha256']
        hashes.update({str(k): row['K_sha256'], str(poses): row['pose_sha256']})
        frames = []
        for frame in row['frames']:
            path = RAW / row['scene'] / 'depth' / frame['depth']
            record = lookup[str(path)]
            assert sha(path) == record['sha256']
            frames.append(dict(index=frame['index'], path=str(path), sha256=record['sha256']))
        entries.append(dict(scene=row['scene'], split='old_eval', pairs=row['pairs'],
                            intrinsics=str(k), poses=str(poses), frames=frames))
    # "old_eval" is only the legacy builder's switch to materialize sparse256.
    return dict(entries=entries, raw_hashes=hashes,
                split_alias_note='old_eval invokes unchanged R68 construction; this cohort is fresh before first evaluation.')


def build_arrays(lock, thresholds):
    base, sparse, sample, meta = collect(lock, 'old_eval', 0, device='cpu')
    for cache in (base, sparse):
        cache['strata'] = strata(cache['geometry_scores'], np.asarray(thresholds))
        for key, value in cache.items():
            if key not in ('ids', 'scenes'):
                assert np.isfinite(value).all(), key
    for key in ('ids', 'scenes', 'overlap', 'world_centers', 'strata', 'permutation'):
        assert np.array_equal(base[key], sparse[key]), key
    return base, sparse, sample, meta


def prepare():
    lock = check()
    assert not (OUT / 'data_manifest.json').exists()
    done = read(OUT / 'download_complete.json')
    assert sha(OUT / 'download_manifest.json') == done['manifest_sha256']
    raw = raw_lock(read(RAW / 'selection.json'), read(OUT / 'download_manifest.json'))
    preserved(OUT / 'raw_data_lock.json', raw)
    thresholds = read(PARENT / 'data/strata_thresholds.json')['thresholds']
    base, sparse, sample, meta = build_arrays(raw, thresholds)
    CACHE.mkdir(parents=True, exist_ok=True)
    records = {}
    for condition, cache in (('base', base), ('sparse256', sparse)):
        path = CACHE / (condition + '.npz')
        assert not path.exists(), path
        np.savez_compressed(path, **cache)
        overlap = cache['overlap']
        valid = np.stack((overlap.max(-1) >= .3, overlap.max(-2) >= .3), 1)
        counts = {scene: dict(pairs=int((cache['scenes'] == scene).sum()),
                    valid_queries=int(valid[cache['scenes'] == scene].sum()),
                    total_queries=int(valid[cache['scenes'] == scene].size)) for scene in lock['planned_scenes']}
        records[condition] = dict(path=str(path), sha256=sha(path), pairs=len(cache['ids']),
            valid_queries=int(valid.sum()), total_queries=int(valid.size), by_scene=counts)
    sample_path = CACHE / 'sample.npz'
    assert not sample_path.exists()
    np.savez_compressed(sample_path, **sample)
    # Reuse the established independent scalar NumPy moment/label check.
    from experiments.round60_audit import audit_sample
    adapted = {('token_' + k[len('reference_'):]) if k.startswith('reference_') else k: v
               for k, v in sample.items() if not k.startswith('sparse_')}
    audit_path = CACHE / 'audit_sample.npz'
    np.savez_compressed(audit_path, **adapted)
    audit = audit_sample(audit_path)
    preserved(OUT / 'data_manifest.json', dict(caches=records, scenes=lock['scenes'], planned_pairs=960,
        planned_scenes=lock['planned_scenes'], scheduled_pairs=lock['pairs'], unavailable=lock['unavailable'],
        planned_queries=960 * 128, excluded=meta['excluded'], construction_seconds=meta['seconds'],
        old_thresholds=thresholds, thresholds_sha256=sha(PARENT / 'data/strata_thresholds.json'),
        fixed_sparse_reference_verified=True, fresh_cohort=True, independent_sample_check=audit,
        sample_path=str(sample_path), sample_sha256=sha(sample_path),
        evaluation_lock_sha256=sha(OUT / 'evaluation_lock.json'),
        raw_data_lock_sha256=sha(OUT / 'raw_data_lock.json'), unix=time.time()))
    print('REGION_DATA_COMPLETE', len(base['ids']), 'pairs', records['base']['valid_queries'], 'valid queries', flush=True)


def load(condition):
    assert condition in ('base', 'sparse256')
    manifest = read(OUT / 'data_manifest.json')
    assert manifest['evaluation_lock_sha256'] == sha(OUT / 'evaluation_lock.json')
    record = manifest['caches'][condition]
    assert sha(record['path']) == record['sha256']
    with np.load(record['path'], allow_pickle=False) as z:
        return {k: z[k] for k in z.files}
