"""Finish an already committed cache, then resume the unchanged evaluation.

The original cache writer expected row['split'] only when writing its final
manifest; this ModelNet catalog uses the already-frozen 'original_split'.
No arrays, catalog rows, models, source locks, or comparison rules are changed.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from experiments import modelnet_cross_source as experiment
from experiments import native_stream_cache as cache

OUT = experiment.OUT
HERE = Path(__file__).resolve().parent
STATE = OUT / 'pipeline_cache_recovery_state.json'


def finish_manifest():
    lock = experiment.ready()
    previous = experiment.read(OUT / 'pipeline_aria2_state.json')
    assert previous['state'] == 'failed' and previous['stage'] == 'prepare'
    assert previous['stages'][-1]['exit_code'] == 1
    log = OUT / 'prepare_pipeline.log'
    assert "KeyError: 'split'" in log.read_text()
    directory = experiment.DATA / 'cache'
    assert not (directory / 'manifest.json').exists()
    assert not (OUT / 'cache_ready.json').exists()
    catalog = experiment.read(OUT / 'catalog.json')
    rows = catalog['rows']
    marker = experiment.read(directory / 'identity.json')
    identity = {k: v for k, v in marker.items() if k != 'sha256'}
    assert marker['sha256'] == hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    assert marker['rows'] == rows and len(rows) == 620
    assert marker['tokens'] == experiment.CONFIG['cache_tokens']
    assert marker['covariance_floor'] == experiment.CONFIG['covariance_floor']
    assert marker['sample_salt'] == experiment.CONFIG['sample_salt']
    assert marker['implementation_sha256'] == experiment.sha(cache.__file__)
    assert marker['converter_sha256'] == experiment.sha(Path(cache.__file__).with_name('round65_data.py'))
    assert catalog['lock_sha256'] == experiment.sha(OUT / 'lock.json')
    for row, frozen in zip(rows, lock['candidate']['rows']):
        assert all(row[key] == value for key, value in frozen.items())
        assert row['original_split'] == 'test' and 'split' not in row
    progress = experiment.read(directory / 'progress.json')
    assert progress['identity_sha256'] == marker['sha256']
    assert len(progress['completed']) == len(rows)
    shapes = cache.schema(len(rows), marker['tokens'])
    arrays = {name: np.load(directory / (name + '.npy'), mmap_mode='r', allow_pickle=False)
              for name in shapes}
    for name, (dtype, shape) in shapes.items():
        assert arrays[name].dtype == np.dtype(dtype) and arrays[name].shape == shape
    for index, (row, committed) in enumerate(zip(rows, progress['completed'])):
        assert committed['file_path'] == row['file_path']
        assert committed['source_sha256'] == row['sha256']
        assert committed['array_row_sha256'] == cache.row_hashes(arrays, index)
    assert np.array_equal(arrays['labels'], [r['taxonomy_id'] for r in rows])
    array_hashes = {name + '.npy': experiment.sha(directory / (name + '.npy')) for name in shapes}
    failure = dict(recorded_unix=time.time(), error_type='KeyError', missing_key='split',
        declared_key='original_split', original_split='test', completed_rows=len(rows),
        original_pipeline_sha256=experiment.sha(OUT / 'pipeline_aria2_state.json'),
        original_log_sha256=experiment.sha(log), catalog_sha256=experiment.sha(OUT / 'catalog.json'),
        original_cache_identity_sha256=marker['sha256'], committed_rows_verified=True,
        array_sha256_before=array_hashes, original_writer_sha256=experiment.sha(cache.__file__),
        recovery_source_sha256=experiment.sha(__file__), lock_sha256=experiment.sha(OUT / 'lock.json'),
        arrays_modified=False, population_changed=False, models_changed=False, new_scores_read=False,
        reason='The failure occurred after all row commits, while reading split for the final manifest flag.')
    assert not (OUT / 'cache_manifest_failure1.json').exists()
    experiment.write(OUT / 'cache_manifest_failure1.json', failure)
    manifest = dict(complete=True, identity_sha256=marker['sha256'], count=len(rows),
        ids=[r['file_path'] for r in rows], array_sha256=array_hashes,
        schema={name: dict(dtype=np.dtype(dtype).str, shape=shape) for name, (dtype, shape) in shapes.items()},
        raw_originals_modified=False, maximum_raw_staging_bytes=max(r['size'] for r in rows),
        cache_array_bytes=sum((directory / (name + '.npy')).stat().st_size for name in shapes),
        cache_tokens=marker['tokens'], formal_test_population=all(r['original_split'] == 'test' for r in rows))
    cache.atomic_json(directory / 'manifest.json', manifest)
    experiment.write(OUT / 'cache_ready.json', dict(manifest_sha256=experiment.sha(directory / 'manifest.json'),
        catalog_sha256=experiment.sha(OUT / 'catalog.json'), objects=manifest['count'],
        lock_sha256=experiment.sha(OUT / 'lock.json')))
    # The original inference loader independently verifies all finalized files.
    experiment.runtime.load_cached_population(directory, rows, experiment.sha(directory / 'manifest.json'))
    experiment.write(OUT / 'cache_manifest_recovery1.json', dict(completed_unix=time.time(), passed=True,
        objects=len(rows), primary_objects=catalog['primary_count'], known_byte_reuse=catalog['known_byte_reuse'],
        manifest_sha256=experiment.sha(directory / 'manifest.json'), array_sha256=array_hashes,
        unchanged_original_loader_passed=True, raw_staging_empty=not (directory / 'raw_stage/raw.verified.ply').exists(),
        source_sha256=experiment.sha(__file__), failure_sha256=experiment.sha(OUT / 'cache_manifest_failure1.json'),
        lock_sha256=experiment.sha(OUT / 'lock.json'), model_or_scientific_changes=False))
    print('CACHE_MANIFEST_RECOVERED', len(rows), catalog['primary_count'], flush=True)


def state_write(**updates):
    state.update(updates, updated_unix=time.time())
    cache.atomic_json(STATE, state)


if __name__ == '__main__':
    assert not STATE.exists(), 'Preserve this execution; inspect a failure before recovery.'
    state = dict(pid=os.getpid(), started_unix=time.time(), stages=[], training_updates=0,
        lock_sha256=experiment.sha(OUT / 'lock.json'), source_sha256=experiment.sha(__file__),
        scientific_review_complete=False, overall_goal_complete=False)
    state_write(stage='cache_manifest', state='running')
    try:
        finish_manifest()
        environment = os.environ.copy()
        environment.update(PYTHONPATH=str(experiment.ROOT), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                           OPENBLAS_NUM_THREADS='4', CUBLAS_WORKSPACE_CONFIG=':4096:8')
        commands = [('export', [sys.executable, '-u', str(HERE / 'gpu_driver.py'), 'export']),
                    ('report', [sys.executable, '-u', '-m', 'experiments.modelnet_cross_source', 'report'])]
        for name, command in commands:
            log = OUT / (name + '_pipeline.log')
            with log.open('x') as stream:
                child = subprocess.Popen(command, cwd=experiment.ROOT, env=environment,
                                         stdout=stream, stderr=subprocess.STDOUT)
                stage = dict(name=name, pid=child.pid, started_unix=time.time(), command=command, log=str(log))
                state['stages'].append(stage)
                state_write(stage=name, state='running')
                print('STAGE_STARTED', name, child.pid, flush=True)
                code = child.wait()
                stage.update(exit_code=code, finished_unix=time.time())
                state_write()
                if code != 0:
                    raise RuntimeError(name + ' failed; inspect terminal status and original log')
        state_write(stage='review_pending', state='complete', finished_unix=time.time())
        print('FIXED_EVALUATION_COMPLETE_REVIEW_PENDING', flush=True)
    except Exception as error:
        state_write(state='failed', finished_unix=time.time(), error_type=type(error).__name__, reason=str(error))
        raise
