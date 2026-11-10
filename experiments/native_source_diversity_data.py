"""Prepare historical TRAIN objects for a fixed within-class source comparison."""
import argparse
from collections import Counter, defaultdict
import contextlib
import hashlib
import json
import os
from pathlib import Path
import time
import zipfile

import numpy as np

from experiments.native_stream_cache import build_cache, atomic_json
from experiments.round65_common import ROOT, DATA_ROOT, CONFIG, read, write, sha

OUT = ROOT / 'results/native_source_diversity/attempt1'
PLAN = ROOT / 'docs/native_source_diversity_plan.md'
DATA = DATA_ROOT / 'GRoPE/native_source_diversity/attempt1'
ARCHIVES = DATA_ROOT / 'GRoPE/modelnet_cross_source/attempt1/archives'
PRIOR = ROOT / 'results/modelnet_cross_source/attempt1'
HISTORY = ROOT / 'results/native_confirmation_metadata/attempt1/historical_overlap1'
OLD_DATA = ROOT / 'results/round65_actual_gs/attempt1/data'
TRAIN = ROOT / 'results/round65_actual_gs/preparation/train_catalog.json'
TEST = ROOT / 'results/native_confirmation_metadata/attempt1/test_catalog.json'
CATEGORIES = dict(airplane=20, bottle=1, bowl=15, car=13, chair=12, guitar=11, table=4)


def metadata():
    assert not (OUT / 'preparation_lock.json').exists()
    original = read(PRIOR / 'lock.json')
    assert sha(PRIOR / 'lock.json') == (PRIOR / 'lock.sha256').read_text().strip()
    assert read(PRIOR / 'review.json')['scientific_review_complete']
    history_plan, history_result = read(HISTORY / 'plan.json'), read(HISTORY / 'result.json')
    history_archive = ROOT / history_plan['archive']
    historical_manifest = ROOT / 'results/round48_modelnet/attempt1/data/manifest.json'
    manifest = read(historical_manifest)
    assert manifest['archive_sha256'] == history_plan['archive_expected_sha256']
    source_rows, source_hashes = {}, {}
    expected = {r['member']: r['sha256'] for r in history_result['metadata_members_read']}
    with zipfile.ZipFile(history_archive) as archive:
        for i in range(5):
            member = f'modelnet40_ply_hdf5_2048/ply_data_train_{i}_id2file.json'
            payload = archive.read(member)
            source_hashes[member] = hashlib.sha256(payload).hexdigest()
            assert source_hashes[member] == expected[member]
            for j, path in enumerate(json.loads(payload)):
                source_rows[f'modelnet40_ply_hdf5_2048/ply_data_train{i}.h5:{j}'] = Path(path).stem
    split_by_id = {source_rows[identity]: part['split'] for part in manifest['partitions'] for identity in part['ids']}
    assert len(split_by_id) == 9840
    upstream = ROOT / 'results/research_data_usage/native_candidates_20261106/split_check1/upstream/modelnet40_train.txt'
    official = set(upstream.read_text().splitlines())
    rows, omitted, archive_records = [], [], []
    for category, label in CATEGORIES.items():
        declaration = next(r for r in original['candidate']['archive_files'] if r['rfilename'] == category + '.zip')
        path = ARCHIVES / declaration['rfilename']
        verified = read(path.with_suffix('.verified.json'))
        assert verified['sha256'] == declaration['lfs']['sha256'] and path.stat().st_size == declaration['size']
        archive_records.append(declaration)
        with zipfile.ZipFile(path) as archive:
            members = set(archive.namelist())
            for identity in sorted(n for n in official if n.rsplit('_', 1)[0] == category):
                member = f'train/{identity}/point_cloud.ply'
                assert member in members
                if identity not in split_by_id:
                    omitted.append(dict(model_id=identity, reason='No R48 historical development partition; keep undecoded'))
                    continue
                rows.append(dict(model_id=identity, category=category, taxonomy_id=label,
                    file_path=f'{label}/{identity}.ply', source='ModelNet', original_split='train',
                    split=split_by_id[identity], archive=category + '.zip', member=member,
                    size=archive.getinfo(member).file_size))
    rows.sort(key=lambda r: (r['taxonomy_id'], r['model_id']))
    assert len(rows) == 2657 and omitted == [dict(model_id='airplane_0050', reason='No R48 historical development partition; keep undecoded')]
    assert Counter(r['split'] for r in rows) == dict(fit=2133, val=262, dev_eval=262)
    prior_ids = {r['model_id'] for r in read(TRAIN)['rows'] + read(TEST)['rows']}
    prior_ids |= {r['model_id'] for r in original['candidate']['rows']}
    assert not prior_ids.intersection(r['model_id'] for r in rows)
    source_files = [Path(__file__), PLAN, ROOT / 'experiments/native_stream_cache.py',
        ROOT / 'experiments/round65_data.py', ROOT / 'experiments/round65_common.py',
        historical_manifest, HISTORY / 'plan.json', HISTORY / 'result.json', upstream,
        TRAIN, TEST, OLD_DATA / 'manifest.json', PRIOR / 'lock.json', PRIOR / 'review.json']
    record = dict(created_unix=time.time(), rows=rows, omitted=omitted,
        repository=original['candidate']['repository'], revision=original['candidate']['revision'],
        archive_files=archive_records, historical_id_member_sha256=source_hashes,
        source_sha256={str(p): sha(p) for p in source_files}, config=CONFIG,
        counts_by_partition=dict(Counter(r['split'] for r in rows)),
        by_class={name: dict(Counter(r['split'] for r in rows if r['category'] == name)) for name in CATEGORIES},
        old_native_partition_counts={r['split']: r['count'] for r in read(OLD_DATA / 'manifest.json')['partitions']},
        native_arrays_read=False, model_scores_read=False, confirmation_objects_used=False,
        all_included_objects_historically_used=True, new_training_authorized_by_fixed_plan=True,
        overall_goal_complete=False)
    write(OUT / 'preparation_lock.json', record)
    (OUT / 'preparation_lock.sha256').write_text(sha(OUT / 'preparation_lock.json') + '\n')
    print('SOURCE_DIVERSITY_DATA_SEALED', len(rows), record['counts_by_partition'], flush=True)


def check():
    lock = read(OUT / 'preparation_lock.json')
    assert sha(OUT / 'preparation_lock.json') == (OUT / 'preparation_lock.sha256').read_text().strip()
    for path, expected in lock['source_sha256'].items():
        assert sha(path) == expected, path
    assert lock['config'] == CONFIG
    return lock


def prepare():
    lock = check()
    state_path = OUT / 'data_runtime.json'
    assert not state_path.exists(), 'Inspect a prior execution before explicit recovery.'
    state = dict(pid=os.getpid(), started_unix=time.time(), state='running', stage='archive_verification',
        preparation_lock_sha256=sha(OUT / 'preparation_lock.json'), decoded_rows=0)
    def update(**values):
        state.update(values, updated_unix=time.time())
        atomic_json(state_path, state)
    update()
    archives = {}
    try:
        for row in lock['archive_files']:
            path = ARCHIVES / row['rfilename']
            assert path.stat().st_size == row['size'] and sha(path) == row['lfs']['sha256']
            archives[row['rfilename']] = zipfile.ZipFile(path)
        access = dict(started_unix=time.time(), objects=2657, prior_historical_use=True,
            includes_original_test=False, includes_previous_620_confirmation=False,
            preparation_lock_sha256=sha(OUT / 'preparation_lock.json'), outcome_path=str(OUT))
        write(OUT / 'data_access_started.json', access)
        write(ROOT / 'results/research_data_usage/native_source_diversity_access_20261109.json', access)
        update(stage='raw_member_hashes')
        old_manifest = read(OLD_DATA / 'manifest.json')
        old_splits = {identity: part['split'] for part in old_manifest['partitions'] for identity in part['ids']}
        used = defaultdict(list)
        for row in read(TRAIN)['rows']:
            used[row['sha256']].append(dict(file_path=row['file_path'], split=old_splits[row['file_path']]))
        for row in read(TEST)['rows']:
            used[row['sha256']].append(dict(file_path=row['file_path'], split='official_test_excluded'))
        rows, reused = [], []
        for r in lock['rows']:
            digest = hashlib.sha256()
            with archives[r['archive']].open(r['member']) as stream:
                for block in iter(lambda: stream.read(2**20), b''):
                    digest.update(block)
            row = dict(r, sha256=digest.hexdigest())
            if row['sha256'] in used:
                reused.append(dict(file_path=row['file_path'], split=row['split'], matches=used[row['sha256']]))
            used[row['sha256']].append(dict(file_path=row['file_path'], split=row['split']))
            rows.append(row)
        write(OUT / 'raw_identity_checks.json', dict(objects=len(rows), exact_byte_reuse=reused,
            all_frozen_source_objects_present=True, confirmation_population_accessed=False))
        if reused:
            raise ValueError('Exact raw reuse requires a recorded resolution before fitting; no silent exclusion.')
        catalog = dict(rows=rows, repository=lock['repository'], revision=lock['revision'],
                       preparation_lock_sha256=sha(OUT / 'preparation_lock.json'))
        write(OUT / 'catalog.json', catalog)
        update(stage='cache')
        def after_row(index, progress):
            if (index + 1) % 25 == 0 or index + 1 == len(rows):
                update(decoded_rows=index + 1)
        provider = lambda row: archives[row['archive']].open(row['member'])
        with (OUT / 'prepare.log').open('x') as log, contextlib.redirect_stdout(log):
            manifest = build_cache(rows, DATA / 'cache', provider, tokens=CONFIG['cache_tokens'],
                minimum_free_bytes=2*2**30, after_row=after_row)
        write(OUT / 'cache_ready.json', dict(objects=manifest['count'],
            counts_by_partition=lock['counts_by_partition'], manifest_sha256=sha(DATA / 'cache/manifest.json'),
            catalog_sha256=sha(OUT / 'catalog.json'), preparation_lock_sha256=sha(OUT / 'preparation_lock.json'),
            fresh_confirmation=False, model_scores_read=False))
        update(state='complete', stage='cache_ready', decoded_rows=len(rows), finished_unix=time.time())
        print('SOURCE_DIVERSITY_CACHE_READY', len(rows), flush=True)
    except Exception as error:
        update(state='failed', error_type=type(error).__name__, reason=str(error), finished_unix=time.time())
        raise
    finally:
        for archive in archives.values():
            archive.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('metadata', 'prepare'))
    args = parser.parse_args()
    globals()[args.action]()
