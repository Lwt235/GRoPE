import json
import numpy as np

from experiments.native_source_diversity_sampling import (
    CLASSES, SourceSchedule, gather, equal_class_source_weights, selection_ce,
    paired_source_comparison,
)


def rows():
    return [dict(taxonomy_id=c, split=split) for c in CLASSES
            for split in ('fit', 'val', 'dev_eval', 'fit')]


def test_replacement_preserves_classes_and_excludes_nonfit_objects():
    extra = rows()
    old = np.tile(np.arange(60), 50)
    mapped = SourceSchedule(11, extra).map(np.arange(len(old)), old)
    effective = np.array([old[i] if bank == 0 else extra[i]['taxonomy_id'] for bank, i in mapped])
    np.testing.assert_array_equal(effective, old)
    assert all(extra[i]['split'] == 'fit' for bank, i in mapped if bank == 1)
    assert (mapped[~np.isin(old, CLASSES), 0] == 0).all()
    replaced = mapped[np.isin(old, CLASSES), 0].sum()
    assert 100 < replaced < 250


def test_shared_stream_and_resumed_source_rng():
    extra = rows()
    labels = np.tile(np.asarray(CLASSES), 10)
    a, b = SourceSchedule(22, extra), SourceSchedule(22, extra)
    index = np.arange(len(labels))
    np.testing.assert_array_equal(a.map(index, labels), b.map(index, labels))
    saved = json.loads(json.dumps(a.state()))
    resumed = SourceSchedule(22, extra, state=saved)
    np.testing.assert_array_equal(a.map(index, labels), resumed.map(index, labels))
    np.testing.assert_array_equal(b.map(index, labels), SourceSchedule(22, extra, state=saved).map(index, labels))


def test_source_rng_does_not_change_original_sample_rng():
    base_a, base_b = np.random.default_rng(65011), np.random.default_rng(65011)
    schedule = SourceSchedule(11, rows())
    labels = np.array(CLASSES)
    for _ in range(10):
        ia, ib = base_a.permutation(len(labels)), base_b.permutation(len(labels))
        np.testing.assert_array_equal(ia, ib)
        schedule.map(ia, labels)
        np.testing.assert_array_equal(base_a.normal(size=(16, 512, 3)), base_b.normal(size=(16, 512, 3)))


def test_cpu_gather_uses_selected_object_and_same_token_indices():
    old = dict(centers=np.arange(3*5*3, dtype=np.float32).reshape(3, 5, 3),
               covariance=np.arange(3*5*9, dtype=np.float64).reshape(3, 5, 3, 3),
               appearance=np.arange(3*5*4, dtype=np.float32).reshape(3, 5, 4), labels=np.array([1, 4, 11]))
    extra = {k: v + (1000 if k != 'labels' else 0) for k, v in old.items()}
    mapping = np.array([[0, 2], [1, 1], [0, 0]])
    subset = np.array([[4, 1], [2, 0], [3, 2]])
    got = gather(old, extra, mapping, subset)
    for row, (bank, index) in enumerate(mapping):
        expected = (old, extra)[bank]
        for key in ('centers', 'covariance', 'appearance'):
            np.testing.assert_array_equal(got[key][row], expected[key][index, subset[row]])
            assert got[key].dtype == expected[key].dtype
        assert got['labels'][row] == expected['labels'][index]


def test_selection_weights_equal_classes_then_sources_not_objects():
    labels = np.array([0, 0, 0, 1])
    sources = np.array(['MACGS', 'MACGS', 'ModelNet', 'MACGS'])
    w = equal_class_source_weights(labels, sources)
    np.testing.assert_allclose(w, [.125, .125, .25, .5])
    x = np.zeros((4, 60))
    x[np.arange(4), labels] = [0, 1, 2, 3]
    expected = w @ np.log1p(59 * np.exp(-np.arange(4)))
    assert np.isclose(selection_ce(x, labels, sources), expected)


def test_paired_stats_keep_class_source_weighting_and_true_full_head_errors():
    labels = np.array([0] * 101 + [1] * 11)
    sources = np.array(['MACGS'] * 100 + ['ModelNet'] + ['MACGS'] * 10 + ['ModelNet'])
    left = np.tile(labels, (3, 1))
    right = left.copy()
    right[:, sources == 'ModelNet'] = 59
    summary = paired_source_comparison(left, right, labels, sources, seed=92001, replicates=256)
    assert summary['mean_gain_pp'] == 50
    np.testing.assert_allclose(summary['paired_seed_gains_pp'], [50, 50, 50])
    np.testing.assert_allclose(summary['intervals_95_pp']['crossed'], [50, 50])
    same = paired_source_comparison(right, right, labels, sources, seed=92001, replicates=256)
    assert same['mean_gain_pp'] == 0
    np.testing.assert_array_equal(same['intervals_95_pp']['crossed'], [0, 0])
