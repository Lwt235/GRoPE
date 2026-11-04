"""Read-only fitting-input analysis of the fixed isotropic covariance floor.

This does not change training, select a model, access task predictions, or add
an acceptance gate. Fractions describe regularized input variances, not task
performance or the cause of a performance difference.
"""
import argparse
from collections import Counter
import os
from pathlib import Path
import time

import numpy as np

from experiments.round65_common import CONFIG, OUT, PREP, checked, read, sha, write


DEST = OUT / 'axis_floor_diagnostic1'
QUANTILES = [0., .5, .9, 1.]


def describe(covariance, floor):
    eigenvalues = np.linalg.eigvalsh(covariance)
    if not np.isfinite(eigenvalues).all() or np.any(eigenvalues <= 0):
        raise ValueError('input covariance is not finite positive definite')
    fraction = floor / eigenvalues
    trace_fraction = 3 * floor / eigenvalues.sum(-1)
    condition = eigenvalues[:, 2] / eigenvalues[:, 0]
    return dict(
        tokens=len(eigenvalues),
        axis_floor_fraction_quantiles=np.quantile(fraction, QUANTILES, axis=0).T.tolist(),
        axis_counts_gt_half=(fraction > .5).sum(0).tolist(),
        axis_counts_gt_nine_tenths=(fraction > .9).sum(0).tolist(),
        thin_gt_half_and_trace_lt_one_percent=int(((fraction[:, 0] > .5) & (trace_fraction < .01)).sum()),
        thin_gt_nine_tenths_and_trace_lt_one_percent=int(((fraction[:, 0] > .9) & (trace_fraction < .01)).sum()),
        trace_floor_fraction_quantiles=np.quantile(trace_fraction, QUANTILES).tolist(),
        regularized_condition_quantiles=np.quantile(condition, QUANTILES).tolist(),
    )


def aggregate(rows):
    tokens = sum(row['tokens'] for row in rows)
    medians = np.asarray([row['axis_floor_fraction_quantiles'] for row in rows])[:, :, 1]
    result = dict(objects=len(rows), tokens=tokens,
        object_median_axis_floor_fraction_quantiles=np.quantile(medians, QUANTILES, axis=0).T.tolist(),
        maximum_observed_axis_floor_fraction=np.asarray(
            [row['axis_floor_fraction_quantiles'] for row in rows])[:, :, -1].max(0).tolist(),
        object_median_trace_floor_fraction_quantiles=np.quantile(
            [row['trace_floor_fraction_quantiles'][1] for row in rows], QUANTILES).tolist(),
        object_median_regularized_condition_quantiles=np.quantile(
            [row['regularized_condition_quantiles'][1] for row in rows], QUANTILES).tolist())
    for name in ('axis_counts_gt_half', 'axis_counts_gt_nine_tenths'):
        counts = np.asarray([row[name] for row in rows], dtype=np.int64).sum(0)
        result[name] = counts.tolist()
        result[name.replace('counts', 'fractions')] = (counts / tokens).tolist()
    for name in ('thin_gt_half_and_trace_lt_one_percent',
                 'thin_gt_nine_tenths_and_trace_lt_one_percent'):
        count = sum(row[name] for row in rows)
        result[name] = dict(count=count, fraction=count / tokens)
    return result


def fixture():
    # Known physical variances plus epsilon I: fractions are .8, .1, .001.
    floor = 1e-8
    first = np.diag(np.array([1.25, 10., 1000.]) * floor)
    rotation = np.array([[1., -1., 0.], [1., 1., 0.], [0., 0., np.sqrt(2.)]]) / np.sqrt(2.)
    a = describe(np.stack([first, rotation @ first @ rotation.T]), floor)
    b = describe(np.eye(3)[None] * floor, floor)
    np.testing.assert_allclose(np.asarray(a['axis_floor_fraction_quantiles'])[:, 1], [.8, .1, .001])
    np.testing.assert_allclose(a['trace_floor_fraction_quantiles'], [3 / 1011.25] * 4)
    assert a['thin_gt_half_and_trace_lt_one_percent'] == 2
    assert a['axis_counts_gt_nine_tenths'] == [0, 0, 0]
    combined = aggregate([a, b])
    np.testing.assert_allclose(combined['axis_fractions_gt_half'], [1., 1/3, 1/3])
    assert combined['tokens'] == 3
    return dict(passed=True, known_thin_fraction=.8,
                known_trace_fraction=3/1011.25, unequal_token_weighting_checked=True)


def input_manifest():
    checked()
    path = OUT / 'data/manifest.json'
    if sha(path) != (OUT / 'data/manifest.sha256').read_text().strip():
        raise ValueError('input manifest identity changed')
    manifest = read(path)
    fit = next(row for row in manifest['partitions'] if row['split'] == 'fit')
    catalog = {row['file_path']: row for row in read(PREP / 'train_catalog.json')['rows']}
    sources = ['MACGS' if catalog[name]['taxonomy_id'] < 30 else 'ModelNet' for name in fit['ids']]
    if any(catalog[name]['split'] != 'train' for name in fit['ids']):
        raise ValueError('expected official TRAIN fitting identities')
    return fit, sources


def plan():
    fit, sources = input_manifest()
    if DEST.exists():
        raise ValueError('preserve existing diagnostic; inspect before retry')
    record = dict(created_unix=time.time(),
        hypothesis='A small trace contribution can coexist with dominant regularization of the thinnest covariance axis; precision-based relations may then encode regularized rather than original thin-axis variance.',
        competing_explanation='The floor contributes little on individual axes as well as in the trace, leaving this particular representation concern weak.',
        prior_observation='Trace-floor preparation metadata were already inspected; this is a new axiswise fitting-input diagnostic, not fresh confirmation.',
        fixed_comparison='For every existing fitting-cache primitive compare epsilon/lambda_k(Sigma+epsilon I), ordered by ascending eigenvalue, with 3epsilon/trace(Sigma+epsilon I).',
        decisions=dict(dominant='Limit native-thickness interpretations and consider whether orientation/tangential information explains later effects; do not infer task harm or repair the ongoing run.',
                       small='This input diagnostic weakens the thin-axis-floor explanation, without establishing PE efficacy.',
                       unchanged='Complete the original full validation and keep its primary comparisons and historical negatives.'),
        floor=CONFIG['covariance_floor'], split='fit', objects=fit['count'],
        tokens_per_object=CONFIG['cache_tokens'], source_counts=dict(Counter(sources)),
        population='All cached fitting primitives, uniformly sampled during the original preparation; no val/dev_eval/TEST arrays or predictions.',
        statistics=dict(axis_order='smallest, middle, largest regularized eigenvalue',
                        fraction_thresholds=[.5, .9], trace_threshold=.01,
                        quantiles=QUANTILES, aggregation='Exact pooled token fractions and quantiles of per-object medians; no class-balanced accuracy inference.'),
        source_sha256=sha(Path(__file__)), lock_sha256=sha(OUT / 'lock.json'),
        manifest_sha256=sha(OUT / 'data/manifest.json'), covariance_sha256=fit['hashes']['covariance.npy'],
        queue_stage_at_plan=read(OUT / 'queue_status.json').get('stage'),
        selected_checkpoint_file_exists=(OUT / 'selected.json').exists(),
        resources='One CPU thread, read-only mmap, one object of eigensystem work at a time; no GPU.',
        fixture=fixture(), training_changed=False, additional_gate=False,
        task_effect_claim=False, fresh_confirmation=False)
    write(DEST / 'plan.json', record)
    print('PLAN', DEST / 'plan.json', sha(DEST / 'plan.json'), flush=True)


def run():
    plan_record = read(DEST / 'plan.json')
    if (DEST / 'summary.json').exists() or (DEST / 'failure.json').exists():
        raise ValueError('preserve existing result/failure; inspect before retry')
    fit, sources = input_manifest()
    if any(sha(path) != plan_record[key] for path, key in (
        (Path(__file__), 'source_sha256'), (OUT / 'lock.json', 'lock_sha256'),
        (OUT / 'data/manifest.json', 'manifest_sha256'))):
        raise ValueError('planned source/input identity changed')
    covariance_path = OUT / 'data/fit/covariance.npy'
    if sha(covariance_path) != plan_record['covariance_sha256']:
        raise ValueError('fitting covariance identity changed')
    started = time.time()
    try:
        covariance = np.load(covariance_path, mmap_mode='r', allow_pickle=False)
        if covariance.dtype != np.float64 or covariance.shape != (fit['count'], CONFIG['cache_tokens'], 3, 3):
            raise ValueError('unexpected fitting cache shape/dtype')
        rows = []
        for i, (identity, source) in enumerate(zip(fit['ids'], sources)):
            rows.append(dict(identity=identity, source=source,
                             **describe(covariance[i], plan_record['floor'])))
            if (i + 1) % 500 == 0:
                progress = dict(pid=os.getpid(), unix=time.time(), completed=i+1, total=len(sources))
                write(DEST / 'progress.json', progress)
                print('INPUTS', i+1, '/', len(sources), flush=True)
        write(DEST / 'objects.json', rows)
        result = dict(plan_sha256=sha(DEST / 'plan.json'), objects_sha256=sha(DEST / 'objects.json'),
            completed_unix=time.time(), seconds=time.time()-started,
            quantiles=QUANTILES, axis_order=plan_record['statistics']['axis_order'],
            overall=aggregate(rows), by_source={source:aggregate([row for row in rows if row['source']==source])
                                             for source in sorted(set(sources))},
            meaning='Fractions quantify the additive floor share in each regularized input variance. They do not measure task impact, normal stability, or cause of any accuracy difference. All original methods receive this same preprocessing.',
            task_predictions_read=False, heldout_arrays_read=False, training_changed=False,
            additional_gate=False, task_effect_claim=False, fresh_confirmation=False)
        write(DEST / 'summary.json', result)
        print('COMPLETE', DEST / 'summary.json', sha(DEST / 'summary.json'), flush=True)
    except BaseException as error:
        write(DEST / 'failure.json', dict(unix=time.time(), error_type=type(error).__name__, message=str(error)))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('plan', 'run'))
    args = parser.parse_args()
    (plan if args.action == 'plan' else run)()
