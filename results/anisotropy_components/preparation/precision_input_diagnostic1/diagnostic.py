"""Describe the mean-precision factor on fixed old fitting inputs only."""
import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path('/home/liwt/GRoPE')
DEST = Path(__file__).resolve().parent
NATIVE = ROOT / 'results/round65_actual_gs/attempt1/data'
REGION = ROOT / 'results/round68_fixed_region/attempt1/data'
CURRENT = ROOT / 'results/anisotropy_components/attempt1'
Q = [0, .1, .5, .9, .99, 1]
SALT = 'GRoPE-mean-precision-inputs-20261107-v1'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_new(name, value):
    with (DEST / name).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def factor(covariance):
    eigen = np.linalg.eigvalsh(np.asarray(covariance, dtype=np.float64))
    assert np.isfinite(eigen).all() and np.all(eigen > 0)
    result = eigen.mean(-1) * (1 / eigen).mean(-1)
    assert np.isfinite(result).all() and np.all(result >= 1 - 1e-10)
    return result


def describe(values):
    values = np.asarray(values)
    return dict(tokens=int(values.size), p_quantiles=np.quantile(values, Q).tolist(),
                log10_p_quantiles=np.quantile(np.log10(values), Q).tolist(),
                fractions_above={str(t): float((values > t).mean()) for t in (2, 10, 100)})


def plan():
    assert not (DEST / 'plan.json').exists()
    manifest = read(NATIVE / 'manifest.json')
    fit = next(row for row in manifest['partitions'] if row['split'] == 'fit')
    ids = fit['ids']
    categories = [int(identity.split('/')[0]) for identity in ids]
    assert set(categories) == set(range(60))
    assert [categories.count(c) for c in range(60)] == fit['class_counts']
    selected = []
    for category in range(60):
        indices = [i for i, c in enumerate(categories) if c == category]
        indices.sort(key=lambda i: hashlib.sha256((SALT + ':' + ids[i]).encode()).digest())
        assert len(indices) >= 8
        selected.extend(dict(index=i, identity=ids[i], category=category,
                             source='MACGS' if category < 30 else 'ModelNet') for i in indices[:8])
    region = read(REGION / 'manifest.json')['caches']
    current = read(CURRENT / 'queue_status.json')
    fixture = factor(np.stack([np.eye(3), np.diag([.25, 1., 4.])]))
    np.testing.assert_allclose(fixture, [1., 3.0625], rtol=1e-14)
    sources = {str(p): sha(p) for p in (
        Path(__file__), NATIVE / 'manifest.json', REGION / 'manifest.json',
        CURRENT / 'lock.json', ROOT / 'docs/anisotropy_components_interpretation.md')}
    write_new('plan.json', dict(created_unix=time.time(),
        hypothesis='The direction-independent mean-precision factor p=r*h materially differs from one in actual fitting inputs, so F-minus-trace differences need not arise only from orientation.',
        competing_explanation='The factor stays close to one on these inputs; the algebraic distinction may then have little practical magnitude in this setting.',
        fixed_comparison='p=mean(eigenvalues)*mean(reciprocal eigenvalues) versus the spherical value one; retain source and region-realization summaries.',
        decisions=dict(large='Keep mean precision/spectrum as a concrete competing explanation when reading M/S results; a task intervention would still be needed for attribution.',
                       near_one='Give this explanation less weight on the sampled inputs, without proving orientation is responsible.',
                       mixed='Keep task/source regimes separate; do not average away differences or select a new model from this diagnostic.'),
        source_sha256=sources, native_covariance_sha256=fit['hashes']['covariance.npy'],
        native_fit_count=fit['count'], native_selected=selected, native_cached_tokens=2048,
        native_tokens_per_object=512, native_sampling_salt=SALT,
        native_token_selection='Fixed SHA256(identity)-seeded choice without replacement from 2048 cached primitives.',
        region_caches={name: region[name] for name in ('fit0', 'fit1')},
        region_selection='All 576 fitting pairs, both views, all 64 regions in each of the two existing realizations.',
        quantiles=Q, descriptive_thresholds=[2, 10, 100],
        aggregation='Native: eight objects per class, equal 512 sampled tokens per object; region: all fitting tokens, with fit0/fit1 reported separately. No confidence intervals or population-effect claims.',
        fixture_values=fixture.tolist(), covariance_semantics='Already regularized input covariance; no floor changes or estimates of unregularized physical thickness.',
        resources='One CPU thread, bounded object slices; no GPU allocation.',
        queue_stage_at_plan=current['stage'], completed_jobs_at_plan=len(current['completed']),
        selected_at_plan=(CURRENT / 'selected.json').exists(), result_at_plan=(CURRENT / 'result.json').exists(),
        heldout_inputs_or_predictions_permitted=False, current_training_changed=False,
        additional_gate=False, fresh_confirmation=False))
    print(json.dumps(dict(plan=str(DEST / 'plan.json'), native_objects=len(selected),
                          native_source_counts=dict(Counter(r['source'] for r in selected)))))


def run():
    p = read(DEST / 'plan.json')
    assert not (DEST / 'result.json').exists()
    for path, expected in p['source_sha256'].items():
        assert sha(Path(path)) == expected, path
    path = NATIVE / 'fit/covariance.npy'
    assert sha(path) == p['native_covariance_sha256']
    covariance = np.load(path, mmap_mode='r', allow_pickle=False)
    assert covariance.shape == (p['native_fit_count'], p['native_cached_tokens'], 3, 3)
    started = time.time()
    rows, groups = [], {}
    for row in p['native_selected']:
        digest = hashlib.sha256((p['native_sampling_salt'] + ':tokens:' + row['identity']).encode()).digest()
        indices = np.random.default_rng(int.from_bytes(digest[:8], 'little')).choice(
            p['native_cached_tokens'], p['native_tokens_per_object'], replace=False)
        sample = np.array(covariance[row['index'], indices], copy=True)
        values = factor(sample)
        rows.append(dict(kind='native', **row, token_indices=indices.tolist(),
                         sampled_covariance_sha256=hashlib.sha256(sample.tobytes()).hexdigest(),
                         **describe(values)))
        groups.setdefault('native_' + row['source'], []).append(values)
    for name, record in p['region_caches'].items():
        path = Path(record['path'])
        assert sha(path) == record['sha256']
        with np.load(path, allow_pickle=False) as arrays:
            cov = arrays['covariance']
            identities, scenes = arrays['ids'], arrays['scenes']
        assert cov.shape == (576, 2, 64, 3, 3)
        values = factor(cov)
        groups['region_' + name] = [values.ravel()]
        for identity, scene, item in zip(identities, scenes, values):
            rows.append(dict(kind='region', realization=name, identity=str(identity),
                             scene=str(scene), **describe(item)))
    summaries = {key: describe(np.concatenate(values)) for key, values in groups.items()}
    write_new('objects_and_pairs.json', rows)
    write_new('result.json', dict(completed_unix=time.time(), seconds=time.time()-started,
        plan_sha256=sha(DEST / 'plan.json'), details_sha256=sha(DEST / 'objects_and_pairs.json'),
        numpy_version=np.__version__, quantiles=p['quantiles'], groups=summaries,
        interpretation='p is the multiplicative gap between the direction-neutral terms h_i*d2 or 3*h_i*r_j and the trace-spherical terms before log1p. It is not the actual oriented F/T ratio, network-output ratio, or fraction of task benefit.',
        native_class_balanced_sample=True, fitting_inputs_only=True, new_predictions=0,
        training_changed=False, additional_gate=False, fresh_confirmation=False))
    print(json.dumps(dict(result=str(DEST / 'result.json'), groups=summaries)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('plan', 'run'))
    args = parser.parse_args()
    (plan if args.action == 'plan' else run)()
