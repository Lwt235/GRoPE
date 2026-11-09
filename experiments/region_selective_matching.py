"""Fixed-score selective matching on previously used regional predictions."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import time
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/region_selective_matching/attempt1'
PLAN = ROOT / 'docs/region_selective_matching_plan.md'
ROLES = ('relative5', 'distance_mlp', 'geotransformer_rpe', 'jeffreys_mlp', 'fixed_isotropic5')
SEEDS = (11, 22, 33)
CONDITIONS = ('base', 'sparse256')
COHORTS = ('old5', 'used9')
BUDGETS = (.01, .05, .10)
COVERAGES = (.01, .02, .05, .10, .20, .40, .60, .80, 1.)
TRAIN = {role: ROOT / 'results' / name / 'attempt1' for role, name in zip(ROLES, (
    'round69_compact_relation', 'round68_fixed_region', 'region_geometry_controls',
    'region_jeffreys_control', 'region_shape_attribution'))}
CONFIRM = ROOT / 'results/region_confirmation/attempt1'
JEFFREYS_EXTENSION = TRAIN['jeffreys_mlp'] / 'used_scene_extension1'
RATES = ('correct_yield', 'wrong_yield', 'coverage', 'precision', 'valid_recall',
         'unmatched_acceptance', 'wrong_valid_yield')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    assert not path.exists(), f'preserve existing artifact: {path}'
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def arrays(path, fields):
    # Read only required payloads in memory; avoids slow small remote-file reads.
    with zipfile.ZipFile(path) as z:
        return {k: np.load(io.BytesIO(z.read(k + '.npy')), allow_pickle=False) for k in fields}


def queries(logits, overlap):
    assert logits.shape == overlap.shape and logits.ndim == 3
    assert logits.shape[1] == logits.shape[2] and np.isfinite(logits).all()
    result = {k: [] for k in ('confidence', 'correct', 'valid', 'picked')}
    for direction in range(2):
        x = np.asarray(logits if direction == 0 else logits.swapaxes(1, 2), dtype=np.float64)
        truth = overlap if direction == 0 else overlap.swapaxes(1, 2)
        picked = x.argmax(-1)
        confidence = 1. / np.exp(x - x.max(-1, keepdims=True)).sum(-1)
        correct = np.take_along_axis(truth, picked[..., None], axis=-1)[..., 0] >= .3
        valid = truth.max(-1) >= .3
        assert not (correct & ~valid).any()
        for key, value in zip(result, (confidence, correct, valid, picked)):
            result[key].append(value)
    return {k: np.stack(v, axis=1) for k, v in result.items()}


def threshold_for_budget(confidence, correct, budget):
    score = np.asarray(confidence).ravel()
    correct = np.asarray(correct, dtype=bool).ravel()
    assert score.shape == correct.shape and len(score) and 0 <= budget <= 1
    order = np.argsort(-score, kind='stable')
    sorted_score = score[order]
    ends = np.r_[np.flatnonzero(sorted_score[:-1] != sorted_score[1:]), len(score) - 1]
    wrong = np.cumsum(~correct[order])[ends]
    feasible = np.flatnonzero(wrong <= budget * len(score) + 1e-12)
    if not len(feasible):
        return float(np.nextafter(sorted_score[0], np.inf))
    last = ends[feasible[-1]]
    return 0. if last == len(score) - 1 else float(sorted_score[last])


def fixed_coverage_weights(confidence, fraction):
    score = np.asarray(confidence)
    assert score.size and 0 <= fraction <= 1
    if fraction == 0:
        return np.zeros_like(score, dtype=np.float64)
    if fraction == 1:
        return np.ones_like(score, dtype=np.float64)
    total = fraction * score.size
    boundary = np.sort(score.ravel())[::-1][int(np.ceil(total)) - 1]
    above, tied = score > boundary, score == boundary
    weights = above.astype(np.float64)
    weights[tied] = (total - above.sum()) / tied.sum()
    assert abs(weights.sum() - total) < 1e-8
    return weights


def statistics(q, weights):
    valid, correct = q['valid'], q['correct']
    weights = np.asarray(weights, dtype=np.float64)
    assert weights.shape == valid.shape and np.all((weights >= 0) & (weights <= 1))
    n = valid.size
    nvalid, nunmatched = int(valid.sum()), int((~valid).sum())
    accepted = float(weights.sum())
    good = float(weights[correct].sum())
    wrong_valid = float(weights[valid & ~correct].sum())
    unmatched = float(weights[~valid].sum())
    bad = wrong_valid + unmatched
    assert abs(accepted - good - bad) < 1e-8
    return dict(queries=n, valid_queries=nvalid, unmatched_queries=nunmatched,
        accepted=accepted, correct_accepted=good, wrong_accepted=bad,
        wrong_valid_accepted=wrong_valid, unmatched_accepted=unmatched,
        correct_yield=good / n, wrong_yield=bad / n, coverage=accepted / n,
        precision=good / accepted if accepted else None,
        valid_recall=good / nvalid if nvalid else None,
        unmatched_acceptance=unmatched / nunmatched if nunmatched else None,
        wrong_valid_yield=wrong_valid / n)


def group_statistics(q, scenes, weights):
    by_scene = {}
    for scene in sorted(set(scenes.tolist())):
        mask = scenes == scene
        by_scene[scene] = statistics({k: v[mask] for k, v in q.items()}, weights[mask])
    macro, defined = {}, {}
    for k in RATES:
        values = [r[k] for r in by_scene.values() if r[k] is not None]
        macro[k] = float(np.mean(values)) if values else None
        defined[k] = len(values)
    return dict(micro=statistics(q, weights), by_scene=by_scene, macro=macro,
                macro_defined_scene_counts=defined)


def freeze():
    assert not (OUT / 'lock.json').exists()
    units = read(OUT.parent / 'preparation/units.json')
    sources = [Path(__file__), ROOT / 'tests/test_region_selective_matching.py', PLAN]
    assert units['passed'] and units['source_sha256'] == {str(p): sha(p) for p in sources}
    sources.append(OUT.parent / 'preparation/units.json')
    artifacts, predictions, caches = {}, [], {}

    def bind(path, expected=None):
        path = Path(path)
        digest = sha(path)
        if expected is not None:
            assert digest == expected, str(path)
        artifacts[str(path)] = digest
        return digest

    parent_manifest = TRAIN['distance_mlp'] / 'data/manifest.json'
    new_manifest = CONFIRM / 'data_manifest.json'
    for p in (parent_manifest, new_manifest, CONFIRM / 'result.json', CONFIRM / 'review.json',
              JEFFREYS_EXTENSION / 'result.json', JEFFREYS_EXTENSION / 'review.json'):
        bind(p)
    original, extension = read(parent_manifest)['caches'], read(new_manifest)['caches']
    caches['val'] = original['val']
    for condition in CONDITIONS:
        caches[f'old5/{condition}'] = original['eval_' + condition]
        caches[f'used9/{condition}'] = extension[condition]
    for meta in caches.values():
        bind(meta['path'], meta['sha256'])
    val_ids = set(arrays(caches['val']['path'], ('ids',))['ids'].tolist())
    for name, meta in caches.items():
        if name != 'val':
            assert not val_ids & set(arrays(meta['path'], ('ids',))['ids'].tolist()), name

    for role in ROLES:
        root = TRAIN[role]
        for name in ('selected.json', 'result.json', 'review.json'):
            bind(root / name)
        review = read(root / 'review.json')
        assert review.get('scientific_review_complete') or review.get('scientific_review_pending') is False
        for seed in SEEDS:
            choice = next(r for r in read(root / 'selected.json')['records']
                          if (r['role'], r['seed']) == (role, seed))
            folder = root / 'runs' / role / str(seed)
            bind(folder / 'best.pt', choice['checkpoint_sha256'])
            bind(folder / 'run.json', choice['run_sha256'])
            val = folder / 'validation.npz'
            digest = bind(val, read(folder / 'run.json')['hashes']['validation.npz'])
            predictions.append(dict(cohort='val', role=role, seed=seed, condition='base',
                path=str(val), sha256=digest, checkpoint_sha256=choice['checkpoint_sha256']))
            for cohort in COHORTS:
                prior = root if cohort == 'old5' else (JEFFREYS_EXTENSION if role == 'jeffreys_mlp' else CONFIRM)
                records = read(prior / 'result.json')['records']
                for condition in CONDITIONS:
                    row = next(r for r in records if (r['role'], r['seed'], r['condition']) == (role, seed, condition))
                    if cohort == 'old5':
                        path = prior / 'evaluation' / role / str(seed) / (condition + '.npz')
                    elif role == 'jeffreys_mlp':
                        path = prior / 'predictions' / str(seed) / (condition + '.npz')
                    else:
                        path = prior / 'predictions' / role / str(seed) / (condition + '.npz')
                    digest = bind(path, row['prediction_sha256'])
                    predictions.append(dict(cohort=cohort, role=role, seed=seed, condition=condition,
                        path=str(path), sha256=digest, checkpoint_sha256=choice['checkpoint_sha256'],
                        prior_region_recall=row['metrics']['macro']['region_recall']))
    assert len(predictions) == 75
    write(OUT / 'lock.json', dict(created_unix=time.time(), source_sha256={str(p): sha(p) for p in sources},
        artifact_sha256=artifacts, predictions=predictions, caches=caches,
        roles=ROLES, seeds=SEEDS, conditions=CONDITIONS, cohorts=COHORTS,
        budgets=BUDGETS, primary_budget=.05, coverages=COVERAGES,
        score='max directional softmax probability; no temperature changes',
        calibration_evaluation_pair_ids_disjoint=True,
        fresh_confirmation=False, new_fits=0, new_inference=0, goal_complete=False))
    (OUT / 'lock.sha256').write_text(sha(OUT / 'lock.json') + '\n')
    print('SELECTIVE_FROZEN', sha(OUT / 'lock.json'), len(artifacts), '75 predictions', flush=True)


def check():
    assert sha(OUT / 'lock.json') == (OUT / 'lock.sha256').read_text().strip()
    lock = read(OUT / 'lock.json')
    for key in ('source_sha256', 'artifact_sha256'):
        for name, digest in lock[key].items():
            assert sha(name) == digest, name
    return lock


def load_queries(prediction, cache):
    z = arrays(prediction['path'], ('ids', 'logits'))
    assert np.array_equal(z['ids'], cache['ids'])
    assert z['logits'].shape == (len(cache['ids']), 64, 64)
    return queries(z['logits'], cache['overlap'])


def calibrate():
    lock = check()
    assert not (OUT / 'calibration.json').exists()
    cache = arrays(lock['caches']['val']['path'], ('ids', 'scenes', 'overlap'))
    assert len(cache['ids']) == 64
    rows = []
    for prediction in (r for r in lock['predictions'] if r['cohort'] == 'val'):
        q = load_queries(prediction, cache)
        for budget in BUDGETS:
            threshold = threshold_for_budget(q['confidence'], q['correct'], budget)
            stats = group_statistics(q, cache['scenes'], q['confidence'] >= threshold)
            assert stats['micro']['wrong_yield'] <= budget + 1e-12
            rows.append(dict(role=prediction['role'], seed=prediction['seed'], budget=budget,
                threshold=threshold, metrics=stats))
    write(OUT / 'calibration.json', dict(records=rows, created_unix=time.time(),
        lock_sha256=sha(OUT / 'lock.json'), calibration_data='old val only',
        evaluation_selective_scores_inspected=False, fresh_confirmation=False))
    print('SELECTIVE_CALIBRATED', len(rows), flush=True)


def compare(records):
    lookup = {(r['cohort'], r['condition'], r['role'], r['seed']): r for r in records}
    result = []
    for cohort in COHORTS:
        for condition in CONDITIONS:
            for control in ROLES[1:]:
                difference = []
                scenes = sorted(lookup[cohort, condition, 'relative5', SEEDS[0]]['operating_points'][1]['metrics']['by_scene'])
                for seed in SEEDS:
                    pair = [lookup[cohort, condition, role, seed]['operating_points'][1]['metrics']['by_scene']
                            for role in ('relative5', control)]
                    assert all(sorted(x) == scenes for x in pair)
                    difference.append([[100 * (pair[0][s][k] - pair[1][s][k])
                                        for k in ('correct_yield', 'wrong_yield')] for s in scenes])
                delta = np.asarray(difference)
                number = 86001 + len(result)
                rng = np.random.default_rng(number)
                si = rng.integers(0, 3, (10000, 3))
                ci = rng.integers(0, len(scenes), (10000, len(scenes)))
                draws = delta[si[:, :, None], ci[:, None, :]].mean((1, 2))
                result.append(dict(cohort=cohort, condition=condition, candidate='relative5', control=control,
                    budget=.05, metrics=['correct_yield', 'wrong_yield'], scenes=scenes,
                    paired_seed_scene_differences_pp=delta.tolist(), paired_seed_differences_pp=delta.mean(1).tolist(),
                    mean_differences_pp=delta.mean((0, 1)).tolist(), bootstrap_seed=number,
                    crossed_seed_scene_95_pp=np.quantile(draws, [.025, .975], axis=0).T.tolist()))
    assert len(result) == 16
    return result


def evaluate():
    lock = check()
    calibration = read(OUT / 'calibration.json')
    assert calibration['lock_sha256'] == sha(OUT / 'lock.json')
    assert not (OUT / 'result.json').exists()
    began = time.time()
    rows, max_recall_error = [], 0.
    for cohort in COHORTS:
        base_identity = None
        for condition in CONDITIONS:
            cache = arrays(lock['caches'][f'{cohort}/{condition}']['path'], ('ids', 'scenes', 'overlap'))
            if base_identity is None:
                base_identity = cache
            else:
                assert all(np.array_equal(cache[k], base_identity[k]) for k in cache)
            assert len(set(cache['ids'].tolist())) == len(cache['ids'])
            for prediction in (r for r in lock['predictions'] if (r['cohort'], r['condition']) == (cohort, condition)):
                role, seed = prediction['role'], prediction['seed']
                q = load_queries(prediction, cache)
                all_output = group_statistics(q, cache['scenes'], np.ones_like(q['confidence']))
                error = abs(all_output['macro']['valid_recall'] - prediction['prior_region_recall'])
                assert error < 1e-10, (cohort, condition, role, seed, error)
                max_recall_error = max(max_recall_error, error)
                operating, curves = [], []
                for budget in BUDGETS:
                    c = next(r for r in calibration['records'] if (r['role'], r['seed'], r['budget']) == (role, seed, budget))
                    weights = q['confidence'] >= c['threshold']
                    operating.append(dict(budget=budget, threshold=c['threshold'],
                        metrics=group_statistics(q, cache['scenes'], weights)))
                for fraction in COVERAGES:
                    weights = np.zeros_like(q['confidence'])
                    for scene in sorted(set(cache['scenes'].tolist())):
                        mask = cache['scenes'] == scene
                        weights[mask] = fixed_coverage_weights(q['confidence'][mask], fraction)
                    curves.append(dict(requested_coverage=fraction,
                        metrics=group_statistics(q, cache['scenes'], weights)))
                rows.append(dict(cohort=cohort, condition=condition, role=role, seed=seed,
                    prediction_sha256=prediction['sha256'], all_queries=all_output,
                    operating_points=operating, coverage_curve=curves))
                print('SELECTIVE_EVALUATED', cohort, condition, role, seed, flush=True)
    assert len(rows) == 60
    write(OUT / 'result.json', dict(records=rows, comparisons=compare(rows),
        original_recall_max_error=max_recall_error, evaluation_began_unix=began, completed_unix=time.time(),
        calibration_sha256=sha(OUT / 'calibration.json'), lock_sha256=sha(OUT / 'lock.json'),
        new_fits=0, new_inference=0, reused_prediction_groups=75, fresh_confirmation=False,
        scientific_review_pending=True, visual_review_pending=True, goal_complete=False))
    print('SELECTIVE_COMPLETE', len(rows), max_recall_error, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('freeze', 'calibrate', 'evaluate', 'check'))
    globals()[parser.parse_args().action]()
