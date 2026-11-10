"""Separate arithmetic for the fixed study; no inference or model selection."""
import numpy as np
from scipy.special import logsumexp


def independent_metrics(logits, labels, sources):
    x, y, s = np.asarray(logits, dtype=np.float64), np.asarray(labels), np.asarray(sources)
    correct = x.argmax(axis=1) == y
    ce = logsumexp(x, axis=1) - x[np.arange(len(y)), y]
    by_class, balanced_acc, balanced_ce, cells = [], [], [], []
    for c in sorted(set(y.tolist())):
        by_class.append(correct[y == c].mean())
        ac, loss = [], []
        for source in sorted(set(s[y == c].tolist())):
            positions = [i for i in range(len(y)) if y[i] == c and s[i] == source]
            ac.append(np.mean(correct[positions])); loss.append(np.mean(ce[positions]))
            cells.append((int(c), source, len(positions), int(correct[positions].sum()), ac[-1], loss[-1]))
        balanced_acc.append(np.mean(ac)); balanced_ce.append(np.mean(loss))
    return dict(mca=float(np.mean(by_class)), oa=float(correct.mean()), ce=float(ce.mean()),
                source_equal_accuracy=float(np.mean(balanced_acc)),
                source_equal_ce=float(np.mean(balanced_ce)), cells=cells)


def independent_pair(left, right, labels, sources, *, seed, replicates=10000):
    """Bootstrap each model's accuracy through a flat weight matrix, then subtract.

    RNG streams/block size match the declared protocol, but calculations do not
    call its metric, weighting or difference/accumulation implementations.
    """
    y, source = np.asarray(labels), np.asarray(sources)
    classes, domains = np.unique(y), np.unique(source)
    groups = [np.flatnonzero((y == c) & (source == d)) for c in classes for d in domains]
    if any(len(g) == 0 for g in groups):
        raise ValueError('Missing class/source cell.')
    membership = np.zeros((len(y), len(groups)))
    for column, index in enumerate(groups):
        membership[index, column] = 1 / len(index)
    lc, rc = 100. * (left == y), 100. * (right == y)
    cell_delta = lc @ membership - rc @ membership
    class_delta = cell_delta.reshape(len(left), len(classes), len(domains)).mean(2)
    srng, orng, crng = [np.random.default_rng(v) for v in np.random.SeedSequence(seed).spawn(3)]
    draws = {key: [] for key in ('seed', 'objects', 'crossed', 'classes')}
    for start in range(0, replicates, 128):
        size = min(128, replicates - start)
        sw = srng.multinomial(len(left), np.full(len(left), 1 / len(left)), size=size) / len(left)
        cw = crng.multinomial(len(classes), np.full(len(classes), 1 / len(classes)), size=size) / len(classes)
        weights = np.zeros((size, len(y)))
        for index in groups:
            counts = orng.multinomial(len(index), np.full(len(index), 1 / len(index)), size=size)
            weights[:, index] = counts / (len(index) * len(groups))
        effects = weights @ lc.T - weights @ rc.T
        draws['seed'].append(sw @ cell_delta.mean(1))
        draws['objects'].append(effects.mean(1))
        draws['crossed'].append(np.einsum('bs,bs->b', sw, effects))
        draws['classes'].append(cw @ class_delta.mean(0))
    return dict(mean=float(cell_delta.mean()), seeds=cell_delta.mean(1),
        cells=cell_delta.mean(0).reshape(len(classes), len(domains)),
        intervals={k: np.quantile(np.concatenate(v), [.025, .975]) for k, v in draws.items()})


def review():
    # Imports here let synthetic and old-only checks exercise this arithmetic
    # without opening new predictions or importing the inference implementation.
    from experiments import native_source_diversity_evaluation as evaluation
    from experiments.round65_common import read, write, sha
    import time
    lock = evaluation.check_evaluation_lock()
    result = read(evaluation.OUT / 'result.json')
    assert result['evaluation_lock_sha256'] == sha(evaluation.OUT / 'evaluation_lock.json')
    meta = evaluation.population_metadata()
    y, source, old = meta['labels'], meta['sources'], meta['original_native']
    seven = np.isin(y, [1, 4, 11, 12, 13, 15, 20])
    masks = dict(primary_seven=seven, seven_MACGS=seven & (source == 'MACGS'),
        seven_ModelNet=seven & (source == 'ModelNet'), original_60=old,
        original_other53=old & ~seven, merged_60=np.ones(len(y), bool))
    maximum = 0.
    predictions = {}
    for family in ('trained', 'background'):
        predictions[family] = {}
        for role in ('relative5', 'distance_mlp', 'fixed_isotropic5'):
            for seed in (11, 22, 33):
                logits, identity = evaluation.load_prediction(family, role, seed, lock, meta)
                saved = next(r for r in result['records'] if (r['family'], r['role'], r['seed']) == (family, role, seed))
                assert saved['identity'] == identity
                predictions[family][role, seed] = logits.argmax(1)
                for group, ix in masks.items():
                    actual = independent_metrics(logits[ix], y[ix], source[ix])
                    reported = saved['groups'][group]
                    assert reported['count'] == int(ix.sum())
                    for key in ('mca', 'oa', 'ce', 'source_equal_accuracy', 'source_equal_ce'):
                        maximum = max(maximum, abs(actual[key] - reported[key]))
                    for cell, written in zip(actual['cells'], reported['class_source_cells'], strict=True):
                        assert cell[:4] == (written['label'], written['source'], written['count'], written['correct'])
                        maximum = max(maximum, abs(cell[4] - written['accuracy']), abs(cell[5] - written['ce']))
                    if np.isin(y[ix], [1, 4, 11, 12, 13, 15, 20]).all():
                        outside = ~np.isin(logits[ix].argmax(1), [1, 4, 11, 12, 13, 15, 20])
                        balanced = np.mean([np.mean([outside[(y[ix] == c) & (source[ix] == d)].mean()
                            for d in np.unique(source[ix][y[ix] == c])]) for c in np.unique(y[ix])])
                        maximum = max(maximum, abs(outside.mean() - reported['outside_seven_fraction']),
                                      abs(balanced - reported['source_equal_outside_seven_fraction']))
    assert len(result['comparisons']) == len(lock['comparisons']) == 20
    for saved, spec in zip(result['comparisons'], lock['comparisons'], strict=True):
        for key, value in spec.items():
            assert saved[key] == value
        pop = predictions[spec['family']]
        ix = masks[spec['group']]
        a = np.stack([pop['relative5', s][ix] for s in (11, 22, 33)])
        b = np.stack([pop[spec['right'], s][ix] for s in (11, 22, 33)])
        domains = source[ix] if spec['group'] == 'primary_seven' else np.full(ix.sum(), 'pooled')
        actual = independent_pair(a, b, y[ix], domains, seed=spec['bootstrap_seed'])
        maximum = max(maximum, abs(actual['mean'] - saved['mean_gain_pp']),
            float(np.abs(actual['seeds'] - saved['paired_seed_gains_pp']).max()),
            float(np.abs(actual['cells'] - saved['class_source_gains_pp']).max()))
        for key, values in actual['intervals'].items():
            maximum = max(maximum, float(np.abs(values - saved['intervals_95_pp'][key]).max()))
    # Old/new differences are descriptive, not a pure source causal estimate.
    for change in result['strategy_changes']:
        for row in change['paired']:
            values = [next(r['groups'][change['group']] for r in result['records'] if
                (r['family'], r['role'], r['seed']) == (family, change['role'], row['seed']))
                for family in ('trained', 'background')]
            for output, metric, scale in [('accuracy_change_pp', 'source_equal_accuracy', 100),
                                         ('oa_change_pp', 'oa', 100), ('ce_change', 'ce', 1)]:
                maximum = max(maximum, abs(row[output] - scale * (values[0][metric] - values[1][metric])))
    main = result['comparisons'][0]
    flag = main['mean_gain_pp'] >= .5 and all(x > 0 for x in main['paired_seed_gains_pp']) and main['intervals_95_pp']['crossed'][0] > 0
    assert flag == result['primary_numerical_signal']
    assert result['scientific_review_pending'] and not result['goal_complete']
    if maximum > 1e-8:
        raise ValueError(f'Material independent arithmetic disagreement: {maximum}')
    write(evaluation.OUT / 'independent_checks.json', dict(passed=True, maximum_error=maximum,
        completed_unix=time.time(), records=18, comparisons=20, result_sha256=sha(evaluation.OUT / 'result.json'),
        implementation_sha256=sha(__file__), scientific_review_pending=True, goal_complete=False))
    print('SOURCE_DIVERSITY_ARITHMETIC_REVIEW', maximum, flush=True)


if __name__ == '__main__':
    review()
