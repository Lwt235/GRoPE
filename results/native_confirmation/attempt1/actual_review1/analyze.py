"""Read-only review of completed confirmation; no refits or new decision rule."""
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import time

import numpy as np


ROOT = Path('/home/liwt/GRoPE')
OUT = ROOT / 'results/native_confirmation/attempt1'
HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def counter_metrics(prediction, labels, mask):
    """Explicit class counts, independent of both existing metric reducers."""
    truth = labels[mask].tolist()
    guessed = prediction[mask].tolist()
    counts = Counter(truth)
    correct = Counter(y for p, y in zip(guessed, truth) if p == y)
    return dict(mca=sum(correct[c] / counts[c] for c in counts) / len(counts),
                oa=sum(correct.values()) / len(truth), count=len(truth),
                class_count=len(counts))


def main():
    result = read(OUT / 'result.json')
    queue = read(OUT / 'queue_status.json')
    seal = read(OUT / 'evaluation_lock.json')
    acquisition = read(OUT / 'acquisition.json')
    assert queue['state'] == 'complete' and not queue['active'] and not queue['pending']
    assert len(queue['completed']) == 31 and all(x['exit_code'] == 0 for x in queue['completed'])
    assert len(result['records']) == 186 and len(result['comparisons']) == 240
    assert result['evaluation_lock_sha256'] == sha(OUT / 'evaluation_lock.json')
    assert result['acquisition_sha256'] == sha(OUT / 'acquisition.json')
    assert result['new_fits'] == result['optimizer_updates'] == 0
    assert not list((OUT / 'exports').rglob('*.pending'))
    marker = read(Path(acquisition['cache_path']) / 'identity.json')
    rows = marker['rows']
    ids = np.array([x['file_path'] for x in rows])
    labels = np.array([x['taxonomy_id'] for x in rows])
    sources = np.array([x['source'] for x in rows])
    historical = read(ROOT / 'results/native_confirmation_metadata/attempt1/historical_overlap1/object_membership.json')['rows']
    historical = {x['file_path']: x for x in historical}
    assert len(ids) == 6090 and len(set(ids.tolist())) == 6090 and set(ids) == set(historical)
    reused = np.array([historical[x]['historical_status'] == 'used_original_train' for x in ids])
    groups = dict(full_population=np.ones(len(ids), bool), MACGS=sources == 'MACGS',
                  ModelNet=sources == 'ModelNet', historical_source_reused=reused,
                  no_known_historical_source_overlap=~reused)
    assert int(reused.sum()) == 2245 and int((~reused).sum()) == 3845
    expected = {(x['role'], x['seed'], v) for x in seal['models'] for v in seal['conditions']}
    assert {(x['role'], x['seed'], x['variant']) for x in result['records']} == expected
    predictions, metrics, records = {}, {}, {}
    metric_error = 0.
    for row in result['records']:
        key = row['role'], row['seed'], row['variant']
        data = Path(row['prediction_path']).read_bytes()
        assert hashlib.sha256(data).hexdigest() == row['prediction_sha256']
        with np.load(io.BytesIO(data), allow_pickle=False) as archive:
            assert np.array_equal(archive['ids'], ids) and np.array_equal(archive['labels'], labels)
            logits = archive['logits']
            assert logits.shape == (6090, 60) and np.isfinite(logits).all()
            prediction = logits.argmax(1)
        predictions[key], records[key] = prediction, row
        for name, mask in groups.items():
            independent = counter_metrics(prediction, labels, mask)
            original = row['groups'][name]
            assert independent['count'] == original['count']
            assert independent['class_count'] == original['class_count']
            metric_error = max(metric_error, *(abs(independent[k] - original['metrics'][k]) for k in ('mca', 'oa')))
            metrics[key + (name,)] = independent
    paired_error, table = 0., []
    for row in result['comparisons']:
        left = [metrics[row['left'], s, row['variant'], row['group']] for s in row['seeds']]
        right = [metrics[row['right'], s, row['variant'], row['group']] for s in row['seeds']]
        pairs = [100 * (a['mca'] - b['mca']) for a, b in zip(left, right)]
        paired_error = max(paired_error, abs(float(np.mean(pairs)) - row['mean_gain_pp']),
                           float(np.max(np.abs(np.array(pairs) - row['paired_seed_gains_pp']))))
        table.append(dict(cohort=row['cohort'], right=row['right'], variant=row['variant'],
            group=row['group'], count=row['count'], class_count=len(row['classes']), seeds=row['seeds'],
            left_mca_pct=float(np.mean(row['left_seed_mca_pct'])),
            right_mca_pct=float(np.mean(row['right_seed_mca_pct'])), gain_pp=row['mean_gain_pp'],
            paired_gains_pp=row['paired_seed_gains_pp'], positive_pairs=row['positive_seed_pairs'],
            joint_interval_pp=row['crossed_seed_object_bootstrap_95_pp'],
            left_oa_pct=float(np.mean(row['left_seed_oa_pct'])), right_oa_pct=float(np.mean(row['right_seed_oa_pct'])),
            paired_oa_gain_pp=row['paired_oa_gain_pp'],
            left_ce=float(np.mean(row['left_seed_ce'])), right_ce=float(np.mean(row['right_seed_ce'])),
            paired_ce_difference=row['paired_ce_difference']))
    assert metric_error < 1e-8 and paired_error < 1e-6
    primary = result['primary']
    assert (primary['cohort'], primary['variant'], primary['group'], primary['seeds']) == (
        'new5', 'base', 'no_known_historical_source_overlap', [44, 55, 66, 77, 88])
    support = primary['mean_gain_pp'] >= .5 and primary['crossed_seed_object_bootstrap_95_pp'][0] > 0
    assert support == result['decision']['primary_support']
    source_descriptive = []
    for source in ('MACGS', 'ModelNet'):
        mask = (~reused) & (sources == source)
        for variant in seal['conditions']:
            left = [counter_metrics(predictions['relative5', s, variant], labels, mask) for s in primary['seeds']]
            right = [counter_metrics(predictions['distance_mlp', s, variant], labels, mask) for s in primary['seeds']]
            pairs = [100 * (a['mca'] - b['mca']) for a, b in zip(left, right)]
            source_descriptive.append(dict(source=source, variant=variant, count=int(mask.sum()),
                class_count=left[0]['class_count'], paired_gains_pp=pairs, mean_gain_pp=float(np.mean(pairs)),
                descriptive_only=True, new_interval_or_decision=False))
    lookup = {(x['cohort'], x['right'], x['variant'], x['group']): x for x in result['comparisons']}
    sparse = lookup['new5', 'distance_mlp', 'sparse256', 'no_known_historical_source_overlap']
    density = dict(scope='Descriptive contrast of already fixed base and sparse conditions; no new test or primary endpoint.',
        relative5_base_to_sparse_drop_pp=(np.array(primary['left_seed_mca_pct']) - sparse['left_seed_mca_pct']).tolist(),
        distance_base_to_sparse_drop_pp=(np.array(primary['right_seed_mca_pct']) - sparse['right_seed_mca_pct']).tolist(),
        gain_increase_sparse_minus_base_pp=(np.array(sparse['paired_seed_gains_pp']) - primary['paired_seed_gains_pp']).tolist())
    permutation = []
    for model in seal['models']:
        key = model['role'], model['seed']
        with np.load(records[key + ('base',)]['prediction_path'], allow_pickle=False) as a, np.load(
                records[key + ('permuted',)]['prediction_path'], allow_pickle=False) as b:
            maximum = float(np.abs(a['logits'] - b['logits']).max())
        permutation.append(dict(role=key[0], seed=key[1], maximum_logit_difference=maximum,
            prediction_changes=int(np.count_nonzero(predictions[key + ('base',)] != predictions[key + ('permuted',)]))))
    delta = np.array(primary['class_mean_gain_pp'])
    result_review = dict(observed_unix=time.time(), result_sha256=sha(OUT / 'result.json'),
        evaluation_lock_sha256=sha(OUT / 'evaluation_lock.json'), queue_sha256=sha(OUT / 'queue_status.json'),
        source_sha256=sha(Path(__file__)), completed_models=31, predictions_checked=186,
        group_metric_pairs_checked=930, paired_comparisons_checked=240,
        independent_counter_mca_oa_max_error=metric_error, independent_paired_mca_max_error_pp=paired_error,
        bootstrap_intervals_regenerated=False, bootstrap_scope='Original sealed, prevalidated analysis retained.',
        primary=primary, original_decision=result['decision'], comparison_table=table,
        primary_class_signs=dict(positive=int((delta > 1e-9).sum()), negative=int((delta < -1e-9).sum()),
            tied=int((np.abs(delta) <= 1e-9).sum()), descriptive_numerical_zero_pp=1e-9),
        primary_seed_spread_descriptive=dict(relative5_std_pp=float(np.std(primary['left_seed_mca_pct'],ddof=1)),
            distance_std_pp=float(np.std(primary['right_seed_mca_pct'],ddof=1)),
            paired_gain_std_pp=float(np.std(primary['paired_seed_gains_pp'],ddof=1)),
            variance_test_performed=False),
        primary_source_descriptive=source_descriptive, density_descriptive=density,
        permutation=permutation, scientific_review_complete=False, external_replication=False, goal_complete=False)
    with (HERE / 'analysis.json').open('x') as stream:
        json.dump(result_review, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(dict(predictions=186, metric_error=metric_error, paired_error_pp=paired_error,
        primary_class_signs=result_review['primary_class_signs'], seed_spread=result_review['primary_seed_spread_descriptive'],
        sources=[x for x in source_descriptive if x['variant']=='base'], density=density,
        permutation_changes=sum(x['prediction_changes'] for x in permutation),
        permutation_logit_max=max(x['maximum_logit_difference'] for x in permutation)), ensure_ascii=False))


if __name__ == '__main__':
    main()
