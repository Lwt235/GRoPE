"""Resume statistics after a duplicate metadata key; preserve sealed sources."""
import json
from pathlib import Path
import shutil
import time

import numpy as np

from experiments import native_source_diversity_analysis as analysis
from experiments import native_source_diversity_evaluation as evaluation
from experiments.round65_common import read, write, sha


def compare_repaired(predictions, labels, sources, original_native, spec, *, replicates=10000):
    ix = analysis.group_masks(labels, sources, original_native)[spec['group']]
    a = np.stack([predictions['relative5', seed][ix] for seed in analysis.SEEDS])
    b = np.stack([predictions[spec['right'], seed][ix] for seed in analysis.SEEDS])
    domain = sources[ix] if spec['group'] == 'primary_seven' else np.full(ix.sum(), 'pooled')
    result = analysis.paired_source_comparison(a, b, labels[ix], domain,
        seed=spec['bootstrap_seed'], replicates=replicates)
    # Both dictionaries record the same bootstrap seed. Keep it once, and
    # reject disagreement rather than silently replacing protocol metadata.
    for key in spec.keys() & result.keys():
        assert spec[key] == result[key], key
    return dict({**spec, **result}, count=int(ix.sum()),
                interpretation='development' if spec['family'] == 'trained' else 'used-data background')


def run():
    evaluation.check_evaluation_lock()
    root = evaluation.OUT
    destination = root / 'report_recovery1'
    assert not (root / 'result.json').exists()
    failed = read(root / 'postprocess_status.json')
    assert failed['state'] == 'postprocess_failed'
    log = Path(failed['active_child']['log'])
    assert "TypeError: dict() got multiple values for keyword argument 'bootstrap_seed'" in log.read_text()
    exports = read(root / 'evaluation_status.json')
    assert exports['state'] == 'exports_complete' and len(exports['completed_exports']) == 18
    assert read(root / 'evaluation_validation_replay.json')['passed']
    destination.mkdir(exist_ok=False)
    for path in (root / 'postprocess_status.json', root / 'speed_transition/status.json',
                 Path(analysis.__file__), log, root / 'active_execution.json'):
        name = 'failed_speed_transition_status.json' if path.parent.name == 'speed_transition' else path.name
        shutil.copy2(path, destination / name)
    write(destination / 'repair.json', dict(created_unix=time.time(),
        failure="Duplicate identical bootstrap_seed metadata passed through dict(**spec, **result).",
        change='Merge dictionaries after checking shared keys agree; unchanged numerical function and arguments.',
        failed_log_sha256=sha(log), frozen_analysis_sha256=sha(analysis.__file__),
        recovery_implementation_sha256=sha(__file__), evaluation_lock_sha256=sha(root / 'evaluation_lock.json'),
        selection_sha256=sha(root / 'selected.json'), exports_status_sha256=sha(root / 'evaluation_status.json'),
        training_or_inference_repeated=False, comparison_or_decision_rules_changed=False,
        cpu_regression_checks_sha256=sha(root / 'report_recovery_cpu_checks.txt')))
    state = dict(state='summarizing', started_unix=time.time(), goal_complete=False)
    write(destination / 'status.json', state)
    original_compare = analysis.compare
    try:
        analysis.compare = compare_repaired
        evaluation.report()
        analysis.compare = original_compare
        state.update(state='independent_arithmetic', result_sha256=sha(root / 'result.json'))
        write(destination / 'status.json', state)
        from experiments.native_source_diversity_review import review
        review()
        state.update(state='ready_for_scientific_review', completed_unix=time.time(),
                     independent_checks_sha256=sha(root / 'independent_checks.json'))
        write(destination / 'status.json', state)
        print('SOURCE_DIVERSITY_REPORT_RECOVERED_REVIEW_PENDING', flush=True)
    except BaseException as error:
        state.update(state='recovery_failed', error_type=type(error).__name__, reason=str(error))
        write(destination / 'status.json', state)
        raise
    finally:
        analysis.compare = original_compare


if __name__ == '__main__':
    run()
