"""Independent SciPy probabilities and event counts for completed selection."""
import hashlib
import io
import json
from pathlib import Path
import time
import zipfile

import numpy as np
from scipy.special import softmax

OUT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
read = lambda path: json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load(path, keys):
    with zipfile.ZipFile(path) as archive:
        return {key: np.load(io.BytesIO(archive.read(key + '.npy')), allow_pickle=False) for key in keys}


def independent_queries(logits, overlap):
    score = np.stack([logits, logits.transpose(0, 2, 1)], axis=1).astype(np.float64)
    label = np.stack([overlap, overlap.transpose(0, 2, 1)], axis=1)
    pick = np.argmax(score, axis=3)
    # Preserve each direction's reduction layout at frozen threshold boundaries.
    confidence = np.stack([softmax(np.asarray(s, dtype=np.float64), axis=2).max(axis=2)
                           for s in (logits, logits.transpose(0, 2, 1))], axis=1)
    good = np.take_along_axis(label, pick[..., None], axis=3)[..., 0] >= .3
    eligible = np.any(label >= .3, axis=3)
    # Literal scalar check of normalization for six actual queries per direction.
    for b in range(min(2, len(score))):
        for direction in range(2):
            for i in range(3):
                values = score[b, direction, i].tolist()
                top = max(values)
                scalar = 1 / sum(np.exp(v - top) for v in values)
                assert abs(scalar - confidence[b, direction, i]) < 1e-14
    return confidence, good, eligible


def counts(good, eligible, weight):
    weight = np.asarray(weight, dtype=np.float64)
    n = good.size
    accepted = float(np.sum(weight))
    correct = float(np.dot(weight.ravel(), good.ravel()))
    unmatched = float(np.dot(weight.ravel(), (~eligible).ravel()))
    wrong_valid = float(np.dot(weight.ravel(), (eligible & ~good).ravel()))
    wrong = accepted - correct
    valid = int(np.count_nonzero(eligible))
    assert abs(wrong - unmatched - wrong_valid) < 1e-8
    return dict(queries=n, valid_queries=valid, unmatched_queries=n-valid,
        accepted=accepted, correct_accepted=correct, wrong_accepted=wrong,
        unmatched_accepted=unmatched, wrong_valid_accepted=wrong_valid,
        correct_yield=correct/n, wrong_yield=wrong/n, coverage=accepted/n,
        precision=correct/accepted if accepted else None,
        valid_recall=correct/valid if valid else None,
        unmatched_acceptance=unmatched/(n-valid) if n-valid else None,
        wrong_valid_yield=wrong_valid/n)


max_error, checked_values = 0., 0


def compare_stats(actual, saved):
    global max_error, checked_values
    assert set(actual) == set(saved)
    for key, value in actual.items():
        if value is None:
            assert saved[key] is None
        else:
            error = abs(value - saved[key])
            assert error < 1e-8, (key, value, saved[key])
            max_error = max(max_error, error)
            checked_values += 1


def check_group(good, eligible, weight, scenes, saved):
    compare_stats(counts(good, eligible, weight), saved['micro'])
    per_scene = {}
    for scene in sorted(set(scenes.tolist())):
        mask = scenes == scene
        actual = counts(good[mask], eligible[mask], weight[mask])
        compare_stats(actual, saved['by_scene'][scene])
        per_scene[scene] = actual
    for key, value in saved['macro'].items():
        values = [x[key] for x in per_scene.values() if x[key] is not None]
        assert len(values) == saved['macro_defined_scene_counts'][key]
        if not values:
            assert value is None
        else:
            assert abs(sum(values)/len(values) - value) < 1e-12
    return per_scene


assert counts(np.array([True, False, False]), np.array([True, True, False]),
              np.array([True, True, True]))['wrong_accepted'] == 2.
lock, calibration, result = [read(OUT / name) for name in ('lock.json', 'calibration.json', 'result.json')]
for key in ('source_sha256', 'artifact_sha256'):
    for name, digest in lock[key].items():
        assert sha(name) == digest, name
assert lock['created_unix'] < calibration['created_unix'] < result['evaluation_began_unix']
assert result['calibration_sha256'] == sha(OUT / 'calibration.json')
assert len(calibration['records']) == 45 and len(result['records']) == 60
caches = {k: load(v['path'], ('ids', 'scenes', 'overlap')) for k, v in lock['caches'].items()}
threshold_errors, primary = [], {}
for pred in lock['predictions']:
    cohort, condition, role, seed = (pred[k] for k in ('cohort', 'condition', 'role', 'seed'))
    cache = caches['val' if cohort == 'val' else cohort + '/' + condition]
    z = load(pred['path'], ('ids', 'logits'))
    assert np.array_equal(z['ids'], cache['ids'])
    confidence, good, eligible = independent_queries(z['logits'], cache['overlap'])
    if cohort == 'val':
        values, inverse = np.unique(confidence, return_inverse=True)
        grouped_wrong = np.bincount(inverse.ravel(), weights=(~good).ravel(), minlength=len(values))
        wrong_at_threshold = np.cumsum(grouped_wrong[::-1])[::-1]
        for row in (r for r in calibration['records'] if (r['role'], r['seed']) == (role, seed)):
            feasible = np.flatnonzero(wrong_at_threshold <= row['budget']*good.size + 1e-12)
            expected = (0. if feasible[0] == 0 else float(values[feasible[0]])) if len(feasible) else float(np.nextafter(values[-1], np.inf))
            threshold_errors.append(abs(expected-row['threshold']))
            assert expected == row['threshold']
            check_group(good, eligible, confidence >= expected, cache['scenes'], row['metrics'])
    else:
        row = next(r for r in result['records'] if (r['cohort'], r['condition'], r['role'], r['seed']) == (cohort, condition, role, seed))
        check_group(good, eligible, np.ones_like(confidence), cache['scenes'], row['all_queries'])
        for op in row['operating_points']:
            identity = next(r for r in calibration['records'] if (r['role'], r['seed'], r['budget']) == (role, seed, op['budget']))
            assert op['threshold'] == identity['threshold']
            per_scene = check_group(good, eligible, confidence >= op['threshold'], cache['scenes'], op['metrics'])
            if op['budget'] == .05:
                primary[cohort, condition, role, seed] = per_scene
        for curve in row['coverage_curve']:
            weights = np.zeros_like(confidence)
            q = curve['requested_coverage']
            for scene in sorted(set(cache['scenes'].tolist())):
                mask = cache['scenes'] == scene
                score = confidence[mask]
                if q == 1:
                    weights[mask] = 1.
                    continue
                k = int(np.ceil(q*score.size))
                boundary = np.partition(score.ravel(), score.size-k)[score.size-k]
                selected = score > boundary
                fraction = (q*score.size - selected.sum()) / np.count_nonzero(score == boundary)
                weights[mask] = selected.astype(float) + (score == boundary)*fraction
            check_group(good, eligible, weights, cache['scenes'], curve['metrics'])
    print('INDEPENDENT_CHECKED', cohort, condition, role, seed, flush=True)

contrast_error = 0.
for i, row in enumerate(result['comparisons']):
    deltas = []
    for seed in (11, 22, 33):
        f, d = [primary[row['cohort'], row['condition'], role, seed] for role in ('relative5', row['control'])]
        deltas.append([[100*(f[s][k]-d[s][k]) for k in ('correct_yield', 'wrong_yield')] for s in row['scenes']])
    delta = np.asarray(deltas)
    contrast_error = max(contrast_error, float(np.max(np.abs(delta-row['paired_seed_scene_differences_pp']))))
    assert np.allclose(delta.mean((0,1)), row['mean_differences_pp'], atol=1e-12, rtol=0)
    assert row['bootstrap_seed'] == 86001+i
assert contrast_error < 1e-10
record = dict(passed=True, completed_unix=time.time(), raw_prediction_groups=75,
    independent_probability_path='SciPy softmax plus 12 literal scalar query checks per prediction group',
    threshold_max_error=max(threshold_errors), numeric_values_checked=checked_values,
    maximum_count_or_rate_error=max_error, primary_contrasts=16, contrast_max_error_pp=contrast_error,
    calibration_before_evaluation=True, original_sources_and_artifacts_verified=True,
    retained_bootstrap='Original frozen paired-seed/scene draws; input vectors checked independently; no interval selection or rerun.',
    new_training_or_inference=False, fresh_confirmation=False,
    result_sha256=sha(OUT/'result.json'), script_sha256=sha(__file__))
(HERE/'checks.json').write_text(json.dumps(record,indent=2)+'\n')
print('INDEPENDENT_REVIEW_PASSED', record, flush=True)
