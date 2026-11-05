"""Execution checks using old cached objects/validation; no test PLY access."""
import argparse
from pathlib import Path
import time
import unittest
import numpy as np
import torch
from experiments.native_confirmation_runtime import load_cached_population, population_groups, export_one, make_model
from experiments import round65_run as runner
from experiments.native_memmap_runtime import install
from experiments.region_geometry_controls import dependencies
from experiments.round65_common import ROOT, VARIANTS, read, write, sha

OUT = ROOT / 'results/native_confirmation_runtime_validation/attempt1'
OLD = ROOT / 'results/round65_actual_gs/attempt1'


def freeze():
    assert not (OUT / 'lock.json').exists()
    suite = unittest.defaultTestLoader.discover(str(ROOT / 'tests'), pattern='test_native_confirmation_runtime.py')
    tests = unittest.TextTestRunner(verbosity=2).run(suite)
    assert tests.wasSuccessful()
    files = dependencies([Path(__file__), ROOT / 'experiments/round65_run.py', ROOT / 'tests/test_native_confirmation_runtime.py'])
    plan = read(OUT / 'plan.json')
    sources = dict(plan['source_sha256'])
    sources.update({str(p): sha(p) for p in files})
    sources[str(OUT / 'plan.json')] = sha(OUT / 'plan.json')
    for name, digest in sources.items():
        assert sha(name) == digest, name
    write(OUT / 'lock.json', dict(created_unix=time.time(), source_sha256=sources,
        tests_passed=True, tests_run=tests.testsRun, official_test_inputs_read=False))
    (OUT / 'lock.sha256').write_text(sha(OUT / 'lock.json') + '\n')
    print('NATIVE_CONFIRMATION_RUNTIME_FROZEN', sha(OUT / 'lock.json'), flush=True)


def check():
    assert sha(OUT / 'lock.json') == (OUT / 'lock.sha256').read_text().strip()
    for name, digest in read(OUT / 'lock.json')['source_sha256'].items():
        assert sha(name) == digest, name
    return read(OUT / 'plan.json')


def legacy(split):
    partition = next(p for p in read(OLD / 'data/manifest.json')['partitions'] if p['split'] == split)
    folder = OLD / 'data' / split
    arrays = {}
    for name in ('centers', 'covariance', 'appearance', 'indices', 'labels'):
        path = folder / (name + '.npy')
        assert sha(path) == partition['hashes'][path.name]
        arrays[name] = np.load(path, mmap_mode='r')
    arrays['ids'] = np.asarray(partition['ids'])
    return arrays


@torch.inference_mode()
def validate():
    plan = check()
    assert not (OUT / 'result.json').exists()
    runner.configure()
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    started = time.monotonic()
    cache = load_cached_population(Path(plan['cache']), plan['old_rows'], plan['cache_manifest_sha256'])
    original = legacy('fit')
    lookup = {v: i for i, v in enumerate(original['ids'].tolist())}
    take = [lookup[p] for p in cache['ids'].tolist()]
    reference = {k: v[take] for k, v in original.items()}
    del original
    for name in ('centers', 'covariance', 'appearance', 'indices', 'labels', 'ids'):
        assert np.array_equal(cache[name], reference[name]), name
    groups = population_groups(plan['old_rows'])
    # Only pre-existing TEST metadata are read here, never a new input array.
    test_rows = read(Path(plan['test_catalog']))['rows']
    membership = read(Path(plan['historical_membership']))['rows']
    masks = population_groups(test_rows, membership)
    counts = {k: int(v.sum()) for k, v in masks.items()}
    assert counts == dict(full_population=6090, MACGS=2124, ModelNet=3966,
                         historical_source_reused=2245, no_known_historical_source_overlap=3845)
    labels = np.asarray([r['taxonomy_id'] for r in test_rows])
    assert len(set(labels[masks['no_known_historical_source_overlap']].tolist())) == 60
    records = []
    for checkpoint in plan['checkpoints']:
        model = make_model(checkpoint['role']).cuda().eval()
        model.load_state_dict(torch.load(checkpoint['path'], map_location='cpu', weights_only=True)['state_dict'])
        for variant in VARIANTS:
            expected, _ = runner.evaluate(model, reference, variant)
            path = OUT / 'exports' / checkpoint['role'] / (variant + '.npz')
            result = export_one(cache, checkpoint, variant, groups, path)
            with np.load(path) as archive:
                assert np.array_equal(archive['ids'], cache['ids'])
                assert np.array_equal(archive['labels'], cache['labels'])
                difference = float(np.max(np.abs(archive['logits'] - expected)))
                changed = int((archive['logits'].argmax(1) != expected.argmax(1)).sum())
            assert difference == 0 and changed == 0
            # Repeating an already complete export validates identity and reuses
            # the artifact; it does not silently overwrite it or run the model.
            assert export_one(cache, checkpoint, variant, groups, path) == result
            records.append(dict(role=checkpoint['role'], variant=variant,
                maximum_logit_difference=difference, prediction_changes=changed,
                independent_metric_error=result['independent_metric_difference'],
                artifact_sha256=sha(path), completed_export_reuse_verified=True))
        del model
    val = legacy('val')
    replays = []
    install(runner)
    for checkpoint in plan['checkpoints']:
        model = make_model(checkpoint['role']).cuda().eval()
        model.load_state_dict(torch.load(checkpoint['path'], map_location='cpu', weights_only=True)['state_dict'])
        logits, _ = runner.evaluate(model, val)
        with np.load(Path(checkpoint['path']).parent / 'validation.npz') as archive:
            assert np.array_equal(archive['ids'], val['ids'])
            difference = float(np.max(np.abs(logits - archive['logits'])))
            changed = int((logits.argmax(1) != archive['logits'].argmax(1)).sum())
        assert difference < 2e-4 and changed == 0
        replays.append(dict(role=checkpoint['role'], objects=len(val['ids']),
                            maximum_logit_difference=difference, prediction_changes=changed))
        del model
    write(OUT / 'result.json', dict(passed=True, records=records, old_validation_replays=replays,
        official_metadata_group_counts=counts, official_nonoverlap_classes=60,
        old_cache_objects=len(cache['ids']), old_training_cache_arrays_exact=True,
        seconds=time.monotonic() - started, lock_sha256=sha(OUT / 'lock.json'),
        official_test_inputs_read=False, official_test_predictions_generated=False,
        final_confirmation_models_selected=False, formal_fits=0, goal_complete=False))
    print('NATIVE_CONFIRMATION_RUNTIME_VALIDATED', time.monotonic() - started, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('freeze', 'validate'))
    globals()[parser.parse_args().action]()
