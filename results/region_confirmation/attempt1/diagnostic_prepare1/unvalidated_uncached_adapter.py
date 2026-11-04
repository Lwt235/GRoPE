"""Disable raw-cloud LRU retention, preserving all frozen numerical operations."""
import argparse
import copy
from pathlib import Path
import time

import numpy as np
import torch

from experiments import round68_fixed_region_data as regional
from experiments import round68_fixed_region_run as engine
from experiments import region_confirmation_data as data
from experiments.region_confirmation_common import ROOT, OUT, PARENT, read, write, sha, preserved, check

RUNTIME = OUT / 'prepare_runtime_update_01'
ORIGINAL_READER = regional.scene_reader


def without_retention(entry, lock):
    reader, poses = ORIGINAL_READER(entry, lock)

    def load(index):
        return reader.__wrapped__(index)

    load.cache_clear = reader.cache_clear
    return load, poses


def bind():
    regional.scene_reader = without_retention


def validate():
    check()
    engine.configure()
    bind()
    assert not (RUNTIME / 'old_validation_check.json').exists()
    old = read(ROOT / 'results/round60_real_scene/attempt1/lock.json')
    entries = [copy.deepcopy(e) for e in old['entries'] if e['split'] == 'val']
    for entry in entries:
        entry['split'] = 'old_eval'
    started = time.time()
    arrays, sparse, sample, meta = data.build_arrays(dict(old, entries=entries),
        read(PARENT / 'data/strata_thresholds.json')['thresholds'])
    errors = {}
    with np.load(PARENT / 'data/val.npz', allow_pickle=False) as saved:
        assert len(arrays['ids']) == len(saved['ids']) == 64
        for key in saved.files:
            if key in ('ids', 'scenes', 'overlap', 'strata', 'permutation', 'frame_gaps'):
                np.testing.assert_array_equal(arrays[key], saved[key])
                errors[key] = 0.
            else:
                np.testing.assert_allclose(arrays[key], saved[key], atol=2e-5, rtol=2e-5)
                errors[key] = float(abs(arrays[key] - saved[key]).max())
    assert not meta['excluded']
    preserved(RUNTIME / 'old_validation_check.json', dict(passed=True, pairs=64,
        maximum_error_by_field=errors, seconds=time.time() - started, source_sha256=sha(__file__),
        original_source_sha256=sha(ROOT / 'experiments/round60_data.py'),
        baseline_sha256=sha(PARENT / 'data/val.npz'), new_model_scores=False,
        scope='Only raw-cloud retention is removed; underlying __wrapped__ reader, SHA checks and every numerical operation are unchanged.'))
    print('UNCACHED_VALIDATED', 64, 'pairs', errors, flush=True)


def prepare():
    check()
    validation = read(RUNTIME / 'old_validation_check.json')
    assert validation['passed'] and validation['source_sha256'] == sha(__file__)
    engine.configure()
    bind()
    data.prepare()
    preserved(RUNTIME / 'data_runtime_identity.json', dict(source_sha256=sha(__file__),
        old_validation_check_sha256=sha(RUNTIME / 'old_validation_check.json'),
        data_manifest_sha256=sha(OUT / 'data_manifest.json'),
        evaluation_lock_sha256=sha(OUT / 'evaluation_lock.json'),
        change='Do not retain backprojected full clouds in the legacy LRU; identical decoder, backprojection, normalization, sampling, tokens, thinning, labels and metrics.',
        completed_unix=time.time()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('validate', 'prepare'))
    globals()[parser.parse_args().action]()
