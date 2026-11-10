"""Fixed class-preserving source replacement and class/source weighting."""
import copy
import numpy as np

CLASSES = (1, 4, 11, 12, 13, 15, 20)
FIELDS = ('centers', 'covariance', 'appearance')


class SourceSchedule:
    def __init__(self, seed, extra_rows, *, state=None):
        self.rng = np.random.default_rng(np.random.SeedSequence([92000, int(seed)]))
        self.labels = np.asarray([r['taxonomy_id'] for r in extra_rows], dtype=np.int64)
        self.pools = {label: np.asarray([i for i, r in enumerate(extra_rows)
                       if r['taxonomy_id'] == label and r['split'] == 'fit'], dtype=np.int64)
                      for label in CLASSES}
        if any(len(pool) == 0 for pool in self.pools.values()):
            raise ValueError('Every declared class requires an original-fit pool.')
        if state is not None:
            self.rng.bit_generator.state = copy.deepcopy(state)

    def state(self):
        return copy.deepcopy(self.rng.bit_generator.state)

    def map(self, index, old_labels):
        index = np.asarray(index, dtype=np.int64)
        labels = np.asarray(old_labels)[index]
        mapping = np.column_stack([np.zeros(len(index), dtype=np.int64), index])
        use_other_source = self.rng.random(len(index)) < .5
        for label, pool in self.pools.items():
            positions = np.flatnonzero((labels == label) & use_other_source)
            chosen = pool[self.rng.integers(len(pool), size=len(positions))]
            mapping[positions, 0] = 1
            mapping[positions, 1] = chosen
            assert np.array_equal(self.labels[chosen], labels[positions])
        return mapping


def gather(old, extra, mapping, subset):
    """Gather identical cache values on CPU before the unchanged CUDA augment."""
    mapping, subset = np.asarray(mapping), np.asarray(subset)
    if mapping.shape != (len(subset), 2) or not np.isin(mapping[:, 0], [0, 1]).all():
        raise ValueError('Invalid bank/index mapping.')
    result = {}
    for name in FIELDS:
        exemplar = old[name]
        value = np.empty((len(mapping), subset.shape[1], *exemplar.shape[2:]), dtype=exemplar.dtype)
        for bank, bundle in enumerate((old, extra)):
            selected = np.flatnonzero(mapping[:, 0] == bank)
            if len(selected):
                value[selected] = bundle[name][mapping[selected, 1, None], subset[selected]]
        result[name] = value
    labels = np.empty(len(mapping), dtype=np.int64)
    for bank, bundle in enumerate((old, extra)):
        selected = np.flatnonzero(mapping[:, 0] == bank)
        labels[selected] = bundle['labels'][mapping[selected, 1]]
    result['labels'] = labels
    return result


def equal_class_source_weights(labels, sources):
    labels, sources = np.asarray(labels), np.asarray(sources)
    if labels.ndim != 1 or labels.shape != sources.shape or len(labels) == 0:
        raise ValueError('Labels and sources must describe the same nonempty population.')
    classes = np.unique(labels)
    weights = np.zeros(len(labels), dtype=np.float64)
    for label in classes:
        available = np.unique(sources[labels == label])
        for source in available:
            selected = (labels == label) & (sources == source)
            weights[selected] = 1. / (len(classes) * len(available) * selected.sum())
    assert np.isclose(weights.sum(), 1.)
    return weights


def selection_ce(logits, labels, sources):
    x, y = np.asarray(logits, dtype=np.float64), np.asarray(labels)
    if x.shape != (len(y), 60) or not np.isfinite(x).all():
        raise ValueError('The unmasked classifier must return finite 60-class logits.')
    maximum = x.max(1)
    ce = maximum + np.log(np.exp(x - maximum[:, None]).sum(1)) - x[np.arange(len(y)), y]
    return float(equal_class_source_weights(y, sources) @ ce)


def paired_source_comparison(left, right, labels, sources, *, seed, replicates=10000):
    """Keep real 60-class decisions; resample paired objects inside each cell."""
    a, b, y, source = map(np.asarray, (left, right, labels, sources))
    if a.ndim != 2 or a.shape != b.shape or a.shape[1] != len(y) or source.shape != y.shape:
        raise ValueError('Unaligned predictions, labels or sources.')
    classes, domains = np.unique(y), np.unique(source)
    cells = [np.flatnonzero((y == c) & (source == d)) for c in classes for d in domains]
    if any(len(ix) == 0 for ix in cells):
        raise ValueError('All specified class/source cells are required.')
    delta = 100. * ((a == y).astype(float) - (b == y).astype(float))
    by_cell = np.stack([delta[:, ix].mean(1) for ix in cells], axis=1)
    by_class = by_cell.reshape(len(a), len(classes), len(domains)).mean(2)
    srng, orng, crng = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(3)]
    draws = {k: [] for k in ('seed', 'objects', 'crossed', 'classes')}
    for start in range(0, replicates, 128):
        size = min(128, replicates - start)
        sw = srng.multinomial(len(a), np.full(len(a), 1. / len(a)), size=size) / len(a)
        cw = crng.multinomial(len(classes), np.full(len(classes), 1. / len(classes)), size=size) / len(classes)
        sampled = np.zeros((size, len(a)))
        for ix in cells:
            w = orng.multinomial(len(ix), np.full(len(ix), 1. / len(ix)), size=size)
            sampled += (w @ delta[:, ix].T) / (len(ix) * len(cells))
        draws['seed'].append(sw @ by_cell.mean(1))
        draws['objects'].append(sampled.mean(1))
        draws['crossed'].append((sw * sampled).sum(1))
        draws['classes'].append(cw @ by_class.mean(0))
    intervals = {k: np.quantile(np.concatenate(v), [.025, .975]).tolist() for k, v in draws.items()}
    return dict(mean_gain_pp=float(by_cell.mean()), paired_seed_gains_pp=by_cell.mean(1).tolist(),
        class_source_gains_pp=by_cell.mean(0).reshape(len(classes), len(domains)).tolist(),
        classes=classes.tolist(), sources=domains.tolist(), intervals_95_pp=intervals,
        bootstrap_seed=seed, bootstrap_replicates=replicates,
        class_sensitivity_resamples_both_sources_together=True)
