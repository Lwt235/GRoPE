"""Exercise workflow joins with old exports only; never open official TEST."""
from pathlib import Path
import tempfile
import time
import unittest

import numpy as np

from experiments.round65_common import ROOT, VARIANTS, read, sha, write
from experiments.native_confirmation_runtime import population_groups, summarize_logits
from experiments.native_confirmation_report import collect_exports, aggregate_comparisons, comparison_specs
from experiments import native_confirmation_workflow as workflow

OUT = ROOT / 'results/native_confirmation_workflow_validation/attempt1'


def validate():
    if (OUT / 'result.json').exists():
        raise ValueError('preserve completed implementation validation')
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    plan, _ = workflow.checked_protocol()
    rows, groups = workflow.population(plan)  # Existing ID metadata only.
    slots = read(plan['models']['slots_path'])['slots']
    old_models = workflow.resolve_models([m for m in slots if m['cohort'] == 'old3'])
    assert len(old_models) == 21
    entry = workflow.seal()
    assert not entry['ready'] and not workflow.OUT.exists()
    suite = unittest.defaultTestLoader.discover(str(ROOT / 'tests'), pattern='test_native_confirmation_workflow.py')
    units = unittest.TextTestRunner(verbosity=2).run(suite)
    if not units.wasSuccessful():
        raise RuntimeError('workflow tests failed')

    # Read the twelve already validated six-object exports through the new
    # final-report collector, without rerunning inference or copying inputs.
    validation = ROOT / 'results/native_confirmation_runtime_validation/attempt1'
    runtime_plan = read(validation / 'plan.json')
    cache = Path(runtime_plan['cache'])
    manifest = read(cache / 'manifest.json')
    labels = np.load(cache / 'labels.npy', allow_pickle=False)
    old_groups = population_groups([dict(r, source='MACGS' if r['taxonomy_id'] < 30 else 'ModelNet')
                                   for r in runtime_plan['old_rows']])
    with tempfile.TemporaryDirectory(prefix='old_export_layout_', dir=OUT) as tmp:
        layout = Path(tmp)
        for model in runtime_plan['checkpoints']:
            (layout / model['role']).mkdir()
            (layout / model['role'] / str(model['seed'])).symlink_to(validation / 'exports' / model['role'])
        collected, _ = collect_exports(layout, runtime_plan['checkpoints'], VARIANTS,
            manifest['ids'], labels, old_groups, manifest['identity_sha256'], sha(cache / 'manifest.json'))
    assert len(collected) == 12

    # Exercise the full old-three, seven-role, six-condition composition on the
    # original 765 used objects. No old score is labeled fresh or used to select.
    records, predictions, sources, reference_mca = [], {}, {}, {}
    ids = labels = None
    for model in old_models:
        base = Path(model['selection_path']).parent
        directory = base / 'evaluation' / model['role'] / str(model['seed'])
        metadata = read(directory / 'record.json')
        assert metadata['checkpoint_sha256'] == model['sha256']
        assert metadata['selection_sha256'] == model['selection_sha256']
        for variant in VARIANTS:
            path = directory / (variant + '.npz')
            prior = next(r for r in metadata['records'] if r['variant'] == variant)
            assert sha(path) == prior['sha256']
            with np.load(path, allow_pickle=False) as archive:
                if ids is None:
                    ids, labels = archive['ids'], archive['labels']
                assert np.array_equal(ids, archive['ids']) and np.array_equal(labels, archive['labels'])
                logits = archive['logits']
            masks = dict(full_population=np.ones(len(labels), dtype=bool), MACGS=labels < 30, ModelNet=labels >= 30)
            metrics = summarize_logits(logits, labels, masks)
            key = model['role'], model['seed'], variant
            predictions[key] = logits.argmax(1)
            reference_mca[key] = prior['metrics']['mca']
            records.append(dict(role=model['role'], seed=model['seed'], variant=variant, groups=metrics))
            sources[str(path)] = prior['sha256']
    comparisons = aggregate_comparisons(records, predictions, labels, masks,
        dict(replicates=200, base_seed=76001, chunk_size=128),
        specs=[r for r in comparison_specs() if r['cohort'] == 'old3'])
    errors = []
    for row in comparisons:
        if row['group'] == 'full_population':
            expected = [100*(reference_mca[row['left'], s, row['variant']]
                            - reference_mca[row['right'], s, row['variant']]) for s in row['seeds']]
            errors.append(float(np.max(np.abs(np.asarray(row['paired_seed_gains_pp']) - expected))))
    assert len(records) == 126 and len(comparisons) == 108 and max(errors) < 1e-10
    result = dict(passed=True, completed_unix=time.time(), seconds=time.monotonic()-started,
        tests_passed=units.testsRun, current_entry=entry, final_seal_created=False,
        official_metadata_group_counts={k: int(v.sum()) for k, v in groups.items()},
        old_models_resolved=len(old_models), old_six_object_exports_collected=len(collected),
        old_export_collection_metric_max_error=max(r['independent_metric_difference'] for r in collected),
        old_development_objects=len(labels), old_development_exports=len(records),
        old_development_comparisons=len(comparisons), original_paired_metric_max_error=max(errors),
        validation_bootstrap_replicates=200, source_predictions=sources,
        source_sha256={str(ROOT / name): sha(ROOT / name) for name in (
            'experiments/native_confirmation_workflow.py', 'experiments/native_confirmation_report.py',
            'experiments/native_confirmation_workflow_validation.py', 'tests/test_native_confirmation_workflow.py')},
        new_fits=0, new_inference=0, official_test_PLY_read=False, test_scores_read=False,
        scope='Old-data execution composition only; fixture and validation intervals are not scientific confirmation.',
        goal_complete=False)
    write(OUT / 'result.json', result)
    print('WORKFLOW_VALIDATED', max(errors), 'old objects', len(labels), flush=True)


if __name__ == '__main__':
    validate()
