"""Review completed runs and recompute predictions without the metric/report code."""
import hashlib
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np

ROOT = Path('/home/liwt/GRoPE')
OUT = ROOT / 'results/native_count_augmentation/attempt1'
HERE = Path(__file__).resolve().parent
SEEDS = (11, 22, 33)
ROLES = ('relative5', 'distance_mlp')
VARIANTS = ('base', 'sparse256', 'dense1024')
REGIMES = ('fixed512', 'mixed256512')
PRIOR = {'relative5': ROOT / 'results/round73_native_relative5/attempt1',
         'distance_mlp': ROOT / 'results/round65_actual_gs/attempt1'}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as f:
        h = hashlib.sha256()
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def metrics(logits, labels):
    x = np.asarray(logits, dtype=np.float64)
    assert x.shape == (765, 60) and np.isfinite(x).all()
    correct = x.argmax(1) == labels
    counts = np.bincount(labels, minlength=60)
    by_class = np.bincount(labels, weights=correct, minlength=60) / counts
    shifted = x - x.max(1, keepdims=True)
    ce = (np.log(np.exp(shifted).sum(1)) - shifted[np.arange(len(x)), labels]).mean()
    return dict(mca=float(by_class.mean()), oa=float(correct.mean()), ce=float(ce),
                class_accuracy=by_class.tolist()), correct


lock, queue, selected, result = [read(OUT / name) for name in
    ('lock.json', 'queue_status.json', 'selected.json', 'result.json')]
assert sha(OUT / 'lock.json') == (OUT / 'lock.sha256').read_text().strip()
for group in ('source_sha256', 'prior_sha256'):
    for path, digest in lock[group].items():
        assert sha(path) == digest, path
assert queue['state'] == queue['stage'] == 'complete'
assert not queue['active'] and not queue['pending']
assert len(queue['completed']) == 12 and all(x['exit_code'] == 0 for x in queue['completed'])
assert len(selected['records']) == 6 and selected['validation_tokens'] == 512
assert selected['all_six_selected_before_development_evaluation']
assert selected['created_unix'] < min(x['started_unix'] for x in queue['completed'] if x['action'] == 'export')
run_checks = []
for choice in selected['records']:
    role, seed = choice['role'], choice['seed']
    path = OUT / 'runs' / role / str(seed)
    run, history, sampling = [read(path / n) for n in ('run.json', 'history.json', 'sampling_summary.json')]
    old = read(PRIOR[role] / 'runs' / role / str(seed) / 'history.json')
    assert sha(path / 'run.json') == choice['run_sha256']
    for name, digest in run['hashes'].items():
        assert sha(path / name) == digest, str(path / name)
    assert run['epochs'] == 120 and run['steps'] == 46800
    assert len(history['training']) == len(sampling['records']) == 120
    assert [r['epoch'] for r in history['validation']] == list(range(0, 121, 5))
    assert all(np.isfinite(r['loss']) and np.isfinite(r['last_gradient_norm']) for r in history['training'])
    assert all(np.isfinite(r['ce']) for r in history['validation'])
    best = min(history['validation'], key=lambda r: r['ce'])
    assert best['epoch'] == choice['best_epoch'] == run['best_epoch']
    assert best['ce'] == run['best_val_ce']
    assert abs(best['ce'] - choice['validation_metrics']['ce']) < 1e-12
    assert [r['stream_sha256'] for r in history['training']] == [r['stream_sha256'] for r in old['training']]
    assert sampling['total_updates'] == 46800
    assert sampling['updates_by_tokens'] == {'256': 23400, '512': 23400}
    for i, rec in enumerate(sampling['records'], 1):
        assert rec['epoch'] == i and rec['seed'] == seed
        assert rec['batches'] == 390 and rec['objects'] == 6227
        assert rec['updates_by_tokens'] == {'256': 195, '512': 195}
        assert rec == read(path / 'sampling' / f'epoch_{i:04d}.json')
    other = OUT / 'runs' / ROLES[1 - ROLES.index(role)] / str(seed) / 'sampling_summary.json'
    assert sampling == read(other)
    assert choice['gpu_logit_error'] == 0 and choice['cpu_prediction_disagreements'] == 0
    assert run['shared_parameters_changed'] and all(v > 0 for v in run['position_parameter_changes'].values())
    run_checks.append(dict(role=role, seed=seed, best_epoch=best['epoch'], epochs=run['epochs'],
        steps=run['steps'], peak_allocated_gib=run['peak_allocated'] / 2**30, resumed=run['resumed'],
        parent_stream_matches_original=True, effective_stream_matches_paired_method=True,
        existing_cpu_replay_max_error=choice['cpu_logit_error'], existing_gpu_replay_max_error=choice['gpu_logit_error']))

labels = np.load(OUT / 'data/dev_eval/labels.npy', allow_pickle=False)
manifest = read(OUT / 'data/manifest.json')
ids = np.asarray(next(r['ids'] for r in manifest['partitions'] if r['split'] == 'dev_eval'))
lookup, correctness, max_metric_error = {}, {}, 0.
assert len(result['records']) == 36
for row in result['records']:
    key = tuple(row[k] for k in ('regime', 'role', 'seed', 'variant'))
    assert key not in lookup
    path = Path(row['prediction_path'])
    assert sha(path) == row['prediction_sha256']
    with np.load(path, allow_pickle=False) as z:
        assert np.array_equal(z['ids'], ids) and np.array_equal(z['labels'], labels)
        computed, correct = metrics(z['logits'], labels)
    for name in computed:
        error = float(np.max(np.abs(np.asarray(computed[name]) - row['metrics'][name])))
        assert error < 1e-10
        max_metric_error = max(max_metric_error, error)
    lookup[key], correctness[key] = computed, correct


def array(regime, role, variant, metric='class_accuracy'):
    return np.asarray([lookup[regime, role, s, variant][metric] for s in SEEDS])


def gap(regime, variant, metric='class_accuracy'):
    return array(regime, ROLES[0], variant, metric) - array(regime, ROLES[1], variant, metric)


def contrast(row, metric):
    kind = row['kind']
    if kind == 'method_gap':
        return gap(row['regime'], row['variant'], metric)
    if kind == 'gap_change':
        return gap(REGIMES[1], row['variant'], metric) - gap(REGIMES[0], row['variant'], metric)
    if kind == 'training_effect':
        return array(REGIMES[1], row['role'], row['variant'], metric) - array(REGIMES[0], row['role'], row['variant'], metric)
    if kind == 'sparse_extra_gap':
        return gap(row['regime'], 'sparse256', metric) - gap(row['regime'], 'base', metric)
    assert kind == 'sparse_extra_gap_change'
    return ((gap(REGIMES[1], 'sparse256', metric) - gap(REGIMES[1], 'base', metric)) -
            (gap(REGIMES[0], 'sparse256', metric) - gap(REGIMES[0], 'base', metric)))


assert len(result['contrasts']) == 18
max_contrast_error = 0.
for i, row in enumerate(result['contrasts']):
    a = 100 * contrast(row, 'class_accuracy')
    checks = ((a.mean(1), row['paired_seed_gains_pp']), (a.mean(), row['mean_gain_pp']),
              (a.mean(0), row['class_mean_gain_pp']),
              (100 * contrast(row, 'oa'), row['paired_oa_gain_pp']),
              (contrast(row, 'ce'), row['paired_ce_difference']))
    for computed, saved in checks:
        error = float(np.max(np.abs(np.asarray(computed) - saved)))
        assert error < 1e-10
        max_contrast_error = max(max_contrast_error, error)
    assert row['bootstrap_seed'] == 84001 + i
    assert np.isfinite(row['crossed_seed_class_bootstrap_95_pp']).all()
for row in result['error_exchanges']:
    regime, variant, seed = (row[k] for k in ('regime', 'variant', 'seed'))
    f, d = [correctness[regime, role, seed, variant] for role in ROLES]
    counts = np.bincount(labels, minlength=60)
    corrected, regressed = [float(100 * (np.bincount(labels, weights=mask, minlength=60) / counts).mean())
                           for mask in (f & ~d, ~f & d)]
    assert abs(corrected - row['corrected_percent']) < 1e-10
    assert abs(regressed - row['regressed_percent']) < 1e-10

summary = []
for regime in REGIMES:
    for variant in VARIANTS:
        delta = gap(regime, variant)
        summary.append(dict(regime=regime, variant=variant,
            f_mca_percent=float(100 * array(regime, ROLES[0], variant).mean()),
            distance_mca_percent=float(100 * array(regime, ROLES[1], variant).mean()),
            gain_pp=float(100 * delta.mean()), paired_seed_gains_pp=(100 * delta.mean(1)).tolist(),
            oa_gain_pp=float(100 * gap(regime, variant, 'oa').mean()),
            ce_difference=float(gap(regime, variant, 'ce').mean()),
            ce_better_seed_pairs=int((gap(regime, variant, 'ce') < 0).sum()),
            source_gain_pp=dict(MACGS=float(100 * delta[:, :30].mean()), ModelNet=float(100 * delta[:, 30:].mean()))))

review = dict(reviewed_at=datetime.now(timezone(timedelta(hours=8))).isoformat(), passed=True,
    source_files_verified=len(lock['source_sha256']), prior_artifacts_verified=len(lock['prior_sha256']),
    completed_processes=len(queue['completed']), all_exit_codes_zero=True, run_checks=run_checks,
    all_selected_before_export=True, raw_prediction_groups_recomputed=36,
    max_metric_error=max_metric_error, contrasts_recomputed=18, max_contrast_error_pp=max_contrast_error,
    bootstrap_intervals='Frozen original intervals retained; signed input vectors independently checked; no new bootstrap run.',
    fresh_confirmation=False, official_test_access=False, new_training_or_gpu_replay=False,
    summary=summary, result_sha256=sha(OUT / 'result.json'), selected_sha256=sha(OUT / 'selected.json'),
    script_sha256=sha(__file__))
(HERE / 'checks.json').write_text(json.dumps(review, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({k: v for k, v in review.items() if k not in ('run_checks',)}, ensure_ascii=False, indent=2))
