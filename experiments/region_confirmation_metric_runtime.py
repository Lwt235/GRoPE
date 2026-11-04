"""Equivalent bounded-array metric execution; scientific formulas stay original."""
import argparse
from pathlib import Path
import time
import zipfile

import numpy as np
import torch

from experiments import round68_fixed_region_run as engine
from experiments.region_confirmation_common import ROOT, OUT, PARENT, CANDIDATE, read, write, sha
from experiments.region_confirmation_resume import data_checked

RUNTIME = OUT / 'metric_runtime_update_01'
ORIGINAL_QUERY = engine.query_values


class BufferedNpz:
    """Read each checked NPY payload as a view, without NumPy's chunk-copy loop."""
    def __init__(self, path):
        self.archive = zipfile.ZipFile(path)
        self.files = [name[:-4] for name in self.archive.namelist() if name.endswith('.npy')]
        assert len(self.files) == len(self.archive.namelist()) == len(set(self.files))
        self.cache = {}

    def __getitem__(self, key):
        if key not in self.cache:
            with self.archive.open(key + '.npy') as stream:
                version = np.lib.format.read_magic(stream)
                assert version in ((1, 0), (2, 0)), version
                reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
                shape, fortran, dtype = reader(stream)
                assert not dtype.hasobject
                payload = bytearray(stream.read())
                count = 1
                for dim in shape:
                    count *= dim
                assert len(payload) == count * dtype.itemsize
                self.cache[key] = np.frombuffer(payload, dtype=dtype).reshape(shape, order='F' if fortran else 'C')
        return self.cache[key]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.archive.close()


class NumpyReadProxy:
    def __getattr__(self, name):
        return getattr(np, name)

    def load(self, path, *, allow_pickle=False):
        assert not allow_pickle and Path(path).suffix == '.npz'
        return BufferedNpz(path)


def batched_query(logits, cache):
    parts = []
    for start in range(0, len(logits), 16):
        selection = slice(start, start + 16)
        # Only the dtype conversion uses Torch. The NumPy metric equations are
        # unchanged; each sample's reductions are independent of other samples.
        score = torch.as_tensor(logits[selection]).to(torch.float64).numpy()
        small = {k: torch.as_tensor(cache[k][selection]).to(torch.float64).numpy()
                 for k in ('overlap', 'world_centers')}
        parts.append(ORIGINAL_QUERY(score, small))
    return {k: torch.cat([torch.from_numpy(p[k]) for p in parts]).numpy() for k in parts[0]}


def compare_tree(actual, expected, errors):
    if isinstance(expected, dict):
        assert set(actual) == set(expected)
        for key in expected:
            compare_tree(actual[key], expected[key], errors)
    elif isinstance(expected, (float, int)):
        errors.append(abs(actual - expected))
        assert abs(actual - expected) < 1e-10, (actual, expected)
    else:
        assert actual == expected, (actual, expected)


def validate():
    data_checked()
    assert not (RUNTIME / 'old_validation.json').exists()
    engine.configure()
    manifest = read(PARENT / 'data/manifest.json')
    records, identities = [], {}
    engine.query_values = batched_query
    for condition in ('base', 'sparse256'):
        meta = manifest['caches']['eval_' + condition]
        assert sha(meta['path']) == meta['sha256']
        identities[meta['path']] = meta['sha256']
        with BufferedNpz(meta['path']) as z:
            cache = {k: z[k] for k in z.files}
        print('METRIC_RUNTIME_LOADED', condition, len(cache['ids']), flush=True)
        for root, role in ((PARENT, 'distance_mlp'), (CANDIDATE, 'relative5')):
            prior_path = root / 'result.json'
            prior = next(r for r in read(prior_path)['records'] if
                         (r['role'], r['seed'], r['condition']) == (role, 11, condition))
            prediction = root / 'evaluation' / role / '11' / (condition + '.npz')
            assert sha(prediction) == prior['prediction_sha256']
            identities[str(prediction)] = sha(prediction)
            identities[str(prior_path)] = sha(prior_path)
            with BufferedNpz(prediction) as z:
                assert np.array_equal(z['ids'], cache['ids'])
                metrics = engine.grouped_metrics(z['logits'], cache)
            errors = []
            compare_tree(metrics, prior['metrics'], errors)
            records.append(dict(role=role, condition=condition, checked_numeric_cells=len(errors),
                maximum_error=max(errors), old_pairs=len(cache['ids'])))
            print('METRIC_RUNTIME_VERIFIED', role, condition, max(errors), len(errors), flush=True)
    write(RUNTIME / 'old_validation.json', dict(passed=True, records=records, source_sha256=sha(__file__),
        old_identities=identities, new_scores_inspected=False, completed_unix=time.time()))


def report():
    data_checked()
    plan = read(RUNTIME / 'plan.json')
    assert sha(RUNTIME / 'plan.json') == (RUNTIME / 'plan.sha256').read_text().strip()
    for path, digest in plan['source_sha256'].items():
        assert sha(path) == digest, path
    assert read(RUNTIME / 'old_validation.json')['passed']
    assert not (OUT / 'result.json').exists()
    from experiments import region_confirmation_data as data
    from experiments import region_confirmation_report as original
    data.np = NumpyReadProxy()
    original.np = NumpyReadProxy()
    engine.query_values = batched_query
    engine.configure()
    original.report()
    write(RUNTIME / 'completed.json', dict(plan_sha256=sha(RUNTIME / 'plan.json'),
        result_sha256=sha(OUT / 'result.json'), report_sha256=sha(OUT / 'report.md'),
        original_metric_formulas_unchanged=True, completed_unix=time.time()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('validate', 'report'))
    args = parser.parse_args()
    globals()[args.action]()
