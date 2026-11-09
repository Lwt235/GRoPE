"""Post-run arithmetic review; reads only this completed development experiment."""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parent


def read(name):
    return json.loads((ROOT / name).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    result, selection, queue = read('result.json'), read('selected.json'), read('queue_status.json')
    assert queue['stage'] == queue['state'] == 'complete' and not queue['active']
    assert all(row['exit_code'] == 0 for row in queue['completed'])
    terminal = [x for x in queue['completed'] if x['action'] == 'train']
    assert len(terminal) == 12 and len({(x['role'], x['seed']) for x in terminal}) == 12
    assert selection['all_twelve_selected_before_development_evaluation']
    assert len(selection['records']) == 12
    assert result['selected_sha256'] == sha(ROOT / 'selected.json')
    assert result['lock_sha256'] == sha(ROOT / 'lock.json')
    assert len(result['records']) == 72 and len(result['contrasts']) == 30
    assert len(result['error_exchanges']) == 90
    max_error = 0.0

    def close(actual, saved):
        nonlocal max_error
        error = float(np.max(np.abs(np.asarray(actual) - np.asarray(saved))))
        assert error < 1e-10, (error, actual, saved)
        max_error = max(max_error, error)

    scores, predictions, labels_by_key = {}, {}, {}
    reference_ids = reference_labels = None
    for row in result['records']:
        path = Path(row['prediction_path'])
        assert sha(path) == row['prediction_sha256']
        with np.load(path, allow_pickle=False) as z:
            logits, labels, ids = z['logits'].astype(np.float64), z['labels'], z['ids']
        assert logits.shape == (765, 60) and np.isfinite(logits).all()
        if reference_ids is None:
            reference_ids, reference_labels = ids, labels
        assert np.array_equal(ids, reference_ids) and np.array_equal(labels, reference_labels)
        counts = np.bincount(labels, minlength=60)
        assert (counts > 0).all()
        correct = logits.argmax(1) == labels
        class_accuracy = np.bincount(labels[correct], minlength=60) / counts
        metric = dict(mca=float(class_accuracy.mean()), oa=float(correct.mean()),
            ce=float(np.mean(logsumexp(logits, axis=1) - logits[np.arange(len(labels)), labels])),
            class_accuracy=class_accuracy)
        for name, value in metric.items():
            close(value, row['metrics'][name])
        key = row['role'], row['seed'], row['variant']
        assert key not in scores
        scores[key], predictions[key], labels_by_key[key] = metric, correct, labels

    base_details = []
    for row in result['contrasts']:
        left = [scores[row['left'], seed, row['variant']] for seed in (11, 22, 33)]
        right = [scores[row['right'], seed, row['variant']] for seed in (11, 22, 33)]
        delta = np.stack([100 * (a['class_accuracy'] - b['class_accuracy']) for a, b in zip(left, right)])
        gains = delta.mean(1)
        close(gains, row['paired_seed_gains_pp'])
        close(gains.mean(), row['mean_gain_pp'])
        close(delta.mean(0), row['class_mean_gain_pp'])
        close([100 * (a['oa'] - b['oa']) for a, b in zip(left, right)], row['paired_oa_gain_pp'])
        close([a['ce'] - b['ce'] for a, b in zip(left, right)], row['paired_ce_difference'])
        assert int((gains > 0).sum()) == row['positive_seed_pairs']
        close(delta[:, :30].mean(1), row['source_seed_gains_pp']['MACGS'])
        close(delta[:, 30:].mean(1), row['source_seed_gains_pp']['ModelNet'])
        rng = np.random.default_rng(row['bootstrap_seed'])
        seed_draw = rng.integers(0, 3, (10000, 3))
        class_draw = rng.integers(0, 60, (10000, 60))
        crossed = np.zeros((10000, 60), dtype=np.float64)
        for j in range(3):
            crossed += delta[seed_draw[:, j, None], class_draw] / 3
        close(np.quantile(crossed.mean(1), [.025, .975]), row['crossed_seed_class_bootstrap_95_pp'])
        close(np.quantile(gains[seed_draw].mean(1), [.025, .975]), row['paired_seed_bootstrap_95_pp'])
        close(np.quantile(delta.mean(0)[class_draw].mean(1), [.025, .975]), row['paired_class_bootstrap_95_pp'])
        if row['variant'] == 'base':
            base_details.append(dict(left=row['left'], right=row['right'],
                mean_gain_pp=float(gains.mean()), paired_gains_pp=gains.tolist(),
                oa_gains_pp=row['paired_oa_gain_pp'], ce_differences=row['paired_ce_difference'],
                source_mean_gains_pp={k: float(np.mean(v)) for k, v in row['source_seed_gains_pp'].items()}))

    exchanges = []
    for row in result['error_exchanges']:
        left = row['left'], row['seed'], row['variant']
        right = row['right'], row['seed'], row['variant']
        a, b, labels = predictions[left], predictions[right], labels_by_key[left]
        counts = np.bincount(labels, minlength=60)
        corrected = np.bincount(labels[a & ~b], minlength=60) / counts
        regressed = np.bincount(labels[~a & b], minlength=60) / counts
        close(100 * corrected.mean(), row['corrected_percent'])
        close(100 * regressed.mean(), row['regressed_percent'])
        close(100 * (corrected - regressed).mean(), row['net_gain_pp'])
        assert int((a & ~b).sum()) == row['corrected_objects']
        assert int((~a & b).sum()) == row['regressed_objects']
        if row['variant'] == 'base':
            exchanges.append(row)

    streams, initializations, training = {}, {}, []
    for row in selection['records']:
        folder = ROOT / 'runs' / row['role'] / str(row['seed'])
        run = json.loads((folder / 'run.json').read_text())
        history = json.loads((folder / 'history.json').read_text())
        assert sha(folder / 'run.json') == row['run_sha256']
        assert sha(folder / 'best.pt') == row['checkpoint_sha256']
        assert (run['epochs'], run['steps']) == (120, 46800)
        seq = [x['stream_sha256'] for x in history['training']]
        assert len(seq) == 120 and streams.setdefault(row['seed'], seq) == seq
        assert initializations.setdefault(row['seed'], run['initial_shared_sha256']) == run['initial_shared_sha256']
        best = min(history['validation'], key=lambda x: x['ce'])
        assert (best['epoch'], best['ce']) == (run['best_epoch'], run['best_val_ce'])
        assert all(np.isfinite(x['loss']) and np.isfinite(x['last_gradient_norm']) for x in history['training'])
        training.append(dict(role=row['role'], seed=row['seed'], best_epoch=run['best_epoch'],
            minutes=run['seconds'] / 60, peak_GiB=run['peak_allocated'] / 2**30,
            parameters=run['parameters'], position_parameters=run['position_parameters']))

    shifts = []
    for role in ('relative5', 'centers_only', 'distance_mlp', 'geotransformer_rpe'):
        means = {v: 100 * np.mean([scores[role, seed, v]['mca'] for seed in (11, 22, 33)])
                 for v in ('base', 'sparse256', 'dense1024', 'jitter02', 'tilt', 'permuted')}
        shifts.append(dict(role=role, mean_mca_percent=means,
            sparse_drop_pp=means['base'] - means['sparse256'], dense_drop_pp=means['base'] - means['dense1024']))
    output = dict(created_unix=time.time(), result_sha256=sha(ROOT / 'result.json'),
        selected_sha256=sha(ROOT / 'selected.json'), source_sha256=sha(__file__),
        all_stage_exit_codes_zero=True, twelve_complete_fits=True,
        all_three_seeds_actual_training_streams_and_shared_initializations_paired=True,
        predictions_independently_recomputed=72, paired_contrasts_recomputed=30,
        paired_error_exchanges_recomputed=90, maximum_arithmetic_error=max_error,
        base_details=base_details, base_error_exchanges=exchanges, input_count_shifts=shifts,
        training=training, fresh_confirmation=False, official_test_access=False,
        metric_arithmetic_passed=True, scientific_review_complete=False, goal_complete=False)
    path = ROOT / 'execution_checks.json'
    assert not path.exists()
    path.write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: output[k] for k in ('maximum_arithmetic_error', 'base_details', 'input_count_shifts', 'training')}, indent=2))


if __name__ == '__main__':
    main()
