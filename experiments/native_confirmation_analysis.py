"""Paired native confirmation estimates; no acquisition, training or selection."""
import hashlib
from pathlib import Path

import numpy as np


def load_prediction_stack(records, expected_ids, expected_labels):
    """Read selected exports in declared seed order, rejecting identity drift."""
    ids, labels = np.asarray(expected_ids), np.asarray(expected_labels)
    if ids.ndim != 1 or labels.shape != ids.shape or len(set(ids.tolist())) != len(ids):
        raise ValueError('invalid or duplicated expected population')
    if not records or len({r['seed'] for r in records}) != len(records):
        raise ValueError('one prediction per declared seed required')
    predictions = []
    for record in records:
        path = Path(record['path'])
        if hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
            raise ValueError('prediction bytes changed')
        with np.load(path, allow_pickle=False) as archive:
            if not np.array_equal(archive['ids'], ids) or not np.array_equal(archive['labels'], labels):
                raise ValueError('prediction population or order changed')
            logits = archive['logits']
            if logits.shape != (len(ids), 60) or not np.isfinite(logits).all():
                raise ValueError('invalid prediction logits')
            predictions.append(logits.argmax(axis=1))
    return np.stack(predictions)


def paired_accuracy_summary(left, right, labels, *, replicates=10000,
                            seed=76001, chunk_size=128, object_bootstrap=True):
    """Equal-class effects with paired seed and class-stratified object draws.

    Objects are crossed with seeds. Each object multiplicity is shared by
    every seed and both methods; the seed multiplicity is independent of it.
    Percentile intervals are approximate, especially with few training seeds.
    """
    a, b, y = np.asarray(left), np.asarray(right), np.asarray(labels)
    if a.ndim != 2 or a.shape != b.shape or y.ndim != 1 or a.shape[1] != len(y):
        raise ValueError('unaligned seed/object predictions')
    if a.shape[0] == 0 or len(y) == 0:
        raise ValueError('empty population or seed cohort')
    for value in (a, b, y):
        if not np.issubdtype(value.dtype, np.integer) or np.any((value < 0) | (value >= 60)):
            raise ValueError('class labels must be integers from 0 to 59')
    if replicates < 2 or chunk_size < 1:
        raise ValueError('invalid resampling size')
    classes = np.unique(y)
    indices = [np.flatnonzero(y == c) for c in classes]
    left_correct, right_correct = a == y[None, :], b == y[None, :]
    delta = 100. * (left_correct.astype(np.float64) - right_correct.astype(np.float64))
    class_delta = np.stack([delta[:, ix].mean(axis=1) for ix in indices], axis=1)
    paired = class_delta.mean(axis=1)
    nseeds, nclasses = len(paired), len(classes)
    seed_rng, object_rng, class_rng = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(3)]
    seed_draws, class_draws = np.empty(replicates), np.empty(replicates)
    object_draws = np.empty(replicates) if object_bootstrap else None
    crossed_draws = np.empty(replicates) if object_bootstrap else None
    for start in range(0, replicates, chunk_size):
        stop = min(start + chunk_size, replicates)
        count = stop - start
        sw = seed_rng.multinomial(nseeds, np.full(nseeds, 1. / nseeds), size=count) / nseeds
        seed_draws[start:stop] = sw @ paired
        cw = class_rng.multinomial(nclasses, np.full(nclasses, 1. / nclasses), size=count) / nclasses
        class_draws[start:stop] = cw @ class_delta.mean(axis=0)
        if object_bootstrap:
            per_seed = np.zeros((count, nseeds), dtype=np.float64)
            for ix in indices:
                weights = object_rng.multinomial(len(ix), np.full(len(ix), 1. / len(ix)), size=count)
                per_seed += (weights @ delta[:, ix].T) / (len(ix) * nclasses)
            object_draws[start:stop] = per_seed.mean(axis=1)
            crossed_draws[start:stop] = (sw * per_seed).sum(axis=1)
    interval = lambda values: np.quantile(values, [.025, .975]).tolist() if values is not None else None
    absolute = lambda correct: (100 * np.stack([correct[:, ix].mean(1) for ix in indices], 1).mean(1)).tolist()
    return dict(count=len(y), seed_pairs=nseeds, classes=classes.tolist(),
        class_counts=[len(ix) for ix in indices], mean_gain_pp=float(paired.mean()),
        paired_seed_gains_pp=paired.tolist(), positive_seed_pairs=int((paired > 0).sum()),
        left_seed_mca_pct=absolute(left_correct), right_seed_mca_pct=absolute(right_correct),
        paired_oa_gain_pp=delta.mean(axis=1).tolist(), class_mean_gain_pp=class_delta.mean(0).tolist(),
        paired_seed_bootstrap_95_pp=interval(seed_draws), paired_class_bootstrap_95_pp=interval(class_draws),
        within_class_object_bootstrap_95_pp=interval(object_draws),
        crossed_seed_object_bootstrap_95_pp=interval(crossed_draws),
        bootstrap_replicates=replicates, bootstrap_seed=seed, chunk_size=chunk_size,
        intervals_are_approximate=True)


def confirmation_decision(primary):
    """Only the prespecified new-five, unused-object base comparison can pass."""
    expected = dict(left='relative5', right='distance_mlp', cohort='new5',
                    seeds=[44, 55, 66, 77, 88], variant='base',
                    group='no_known_historical_source_overlap', count=3845,
                    seed_pairs=5, classes=list(range(60)))
    if any(primary.get(k) != value for k, value in expected.items()):
        raise ValueError('wrong primary cohort, comparison or population')
    mean = primary['mean_gain_pp']
    interval = primary['crossed_seed_object_bootstrap_95_pp']
    if interval is None or len(interval) != 2 or not np.isfinite([mean, *interval]).all() or interval[0] > interval[1]:
        raise ValueError('missing or invalid primary uncertainty')
    support = bool(mean >= .5 and interval[0] > 0)
    return dict(primary_support=support,
                outcome='positive_signal' if support else ('negative' if mean <= 0 else 'inconclusive'),
                overall_goal_complete=False, all_conditions_require_review=True)
