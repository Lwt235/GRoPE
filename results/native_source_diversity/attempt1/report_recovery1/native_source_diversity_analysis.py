"""Declared populations and full-head metrics for the source-diversity study."""
import numpy as np

from experiments.native_source_diversity_sampling import (
    CLASSES, equal_class_source_weights, paired_source_comparison,
)

ROLES = ('relative5', 'distance_mlp', 'fixed_isotropic5')
SEEDS = (11, 22, 33)
GROUPS = ('primary_seven', 'seven_MACGS', 'seven_ModelNet',
          'original_60', 'original_other53', 'merged_60')


def group_masks(labels, sources, original_native):
    y, s, old = np.asarray(labels), np.asarray(sources), np.asarray(original_native)
    if y.ndim != 1 or s.shape != y.shape or old.shape != y.shape or old.dtype != bool:
        raise ValueError('Unaligned population metadata.')
    if not np.array_equal(np.unique(y[old]), np.arange(60)):
        raise ValueError('The complete original 60-class population is required.')
    if not np.array_equal(s[old], np.where(y[old] < 30, 'MACGS', 'ModelNet')):
        raise ValueError('Original source metadata changed.')
    seven = np.isin(y, CLASSES)
    if not (seven[~old].all() and (s[~old] == 'ModelNet').all()):
        raise ValueError('Only the declared seven ModelNet classes may be added.')
    return dict(primary_seven=seven, seven_MACGS=seven & (s == 'MACGS'),
                seven_ModelNet=seven & (s == 'ModelNet'), original_60=old,
                original_other53=old & ~seven, merged_60=np.ones(len(y), dtype=bool))


def metrics(logits, labels, sources):
    x, y, source = np.asarray(logits, dtype=np.float64), np.asarray(labels), np.asarray(sources)
    if (y.ndim != 1 or x.shape != (len(y), 60) or source.shape != y.shape or not len(y)
            or not np.issubdtype(y.dtype, np.integer) or np.any((y < 0) | (y >= 60))
            or not np.isfinite(x).all()):
        raise ValueError('Finite full-60 logits and aligned real labels are required.')
    pred = x.argmax(1)
    correct = pred == y
    shifted = x - x.max(1, keepdims=True)
    ce = np.log(np.exp(shifted).sum(1)) - shifted[np.arange(len(y)), y]
    weights = equal_class_source_weights(y, source)
    classes = np.unique(y)
    cells = []
    for c in classes:
        for s in np.unique(source[y == c]):
            ix = (y == c) & (source == s)
            cells.append(dict(label=int(c), source=str(s), count=int(ix.sum()),
                correct=int(correct[ix].sum()), accuracy=float(correct[ix].mean()), ce=float(ce[ix].mean())))
    class_accuracy = [float(correct[y == c].mean()) if c in classes else None for c in range(60)]
    seven_only = np.isin(y, CLASSES).all()
    outside = ~np.isin(pred, CLASSES)
    return dict(count=len(y), mca=float(np.mean([class_accuracy[c] for c in classes])),
        oa=float(correct.mean()), ce=float(ce.mean()),
        source_equal_accuracy=float(weights @ correct), source_equal_ce=float(weights @ ce),
        class_accuracy=class_accuracy, class_source_cells=cells,
        outside_seven_fraction=float(outside.mean()) if seven_only else None,
        source_equal_outside_seven_fraction=float(weights @ outside) if seven_only else None)


def summaries(logits, labels, sources, original_native):
    groups = group_masks(labels, sources, original_native)
    return {name: metrics(logits[ix], labels[ix], sources[ix]) for name, ix in groups.items()}


def comparison_specs():
    """First two are the frozen primary and attribution; all others descriptive."""
    rows = []
    for family in ('trained', 'background'):
        for group in GROUPS[:-1]:
            for right in ROLES[1:]:
                rows.append(dict(family=family, group=group, left='relative5', right=right,
                                 seeds=list(SEEDS), bootstrap_seed=92001 + len(rows)))
    return rows


def compare(predictions, labels, sources, original_native, spec, *, replicates=10000):
    ix = group_masks(labels, sources, original_native)[spec['group']]
    a = np.stack([predictions['relative5', seed][ix] for seed in SEEDS])
    b = np.stack([predictions[spec['right'], seed][ix] for seed in SEEDS])
    # Only the primary has a full seven-class by two-source Cartesian population.
    # The original 60-class comparisons retain ordinary mCA: one cell per class.
    domain = sources[ix] if spec['group'] == 'primary_seven' else np.full(ix.sum(), 'pooled')
    result = paired_source_comparison(a, b, labels[ix], domain,
        seed=spec['bootstrap_seed'], replicates=replicates)
    return dict(**spec, **result, count=int(ix.sum()),
                interpretation='development' if spec['family'] == 'trained' else 'used-data background')


def primary_signal(row):
    return bool(row['mean_gain_pp'] >= .5 and min(row['paired_seed_gains_pp']) > 0
                and row['intervals_95_pp']['crossed'][0] > 0)
