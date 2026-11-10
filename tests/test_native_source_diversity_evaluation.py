import numpy as np
import pytest

from experiments import native_source_diversity_analysis as analysis
from experiments.native_source_diversity_evaluation import checked_arrays
from experiments.native_source_diversity_review import independent_metrics, independent_pair


def test_full_head_source_weighting_does_not_mask_other_classes():
    y = np.array([1] * 6 + [4] * 4)
    source = np.array(['MACGS'] * 5 + ['ModelNet', 'MACGS'] + ['ModelNet'] * 3)
    prediction = np.array([1] * 5 + [59, 4, 59, 59, 59])
    x = np.zeros((len(y), 60))
    x[np.arange(len(y)), prediction] = 10.
    result = analysis.metrics(x, y, source)
    assert result['oa'] == pytest.approx(.6)
    assert result['mca'] == pytest.approx((5 / 6 + 1 / 4) / 2)
    assert result['source_equal_accuracy'] == pytest.approx(.5)
    assert result['outside_seven_fraction'] == pytest.approx(.4)
    assert result['source_equal_outside_seven_fraction'] == pytest.approx(.5)
    assert [c['count'] for c in result['class_source_cells']] == [5, 1, 1, 3]
    separate = independent_metrics(x + 10000., y, source)
    for key in ('oa', 'mca', 'source_equal_accuracy', 'source_equal_ce', 'ce'):
        assert result[key] == pytest.approx(separate[key], abs=1e-10)


def test_paired_bootstrap_with_imbalanced_cells_and_seed_disagreement():
    cells = [(1, 'MACGS', 2), (1, 'ModelNet', 57), (4, 'MACGS', 13),
             (4, 'ModelNet', 4), (12, 'MACGS', 29), (12, 'ModelNet', 5)]
    y = np.concatenate([np.full(n, c) for c, _, n in cells])
    source = np.concatenate([np.full(n, s) for _, s, n in cells])
    rng = np.random.default_rng(137)
    a = np.where(rng.random((3, len(y))) < .65, y, 59)
    b = np.where(rng.random((3, len(y))) < .7, y, 58)
    first = analysis.paired_source_comparison(a, b, y, source, seed=92001, replicates=257)
    second = independent_pair(a, b, y, source, seed=92001, replicates=257)
    assert first['mean_gain_pp'] == pytest.approx(second['mean'], abs=1e-10)
    np.testing.assert_allclose(first['paired_seed_gains_pp'], second['seeds'], atol=1e-10)
    np.testing.assert_allclose(first['class_source_gains_pp'], second['cells'], atol=1e-10)
    for key, value in second['intervals'].items():
        np.testing.assert_allclose(first['intervals_95_pp'][key], value, atol=1e-10)
    # Dropping a domain within one class must not silently change its weight.
    keep = ~((y == 1) & (source == 'MACGS'))
    with pytest.raises(ValueError, match='cells'):
        analysis.paired_source_comparison(a[:, keep], b[:, keep], y[keep], source[keep], seed=92001)


def test_groups_retain_original_full_task_and_both_sources():
    y = np.concatenate([np.arange(60), np.repeat(analysis.CLASSES, 2)])
    source = np.concatenate([np.where(np.arange(60) < 30, 'MACGS', 'ModelNet'), np.full(14, 'ModelNet')])
    old = np.arange(len(y)) < 60
    masks = analysis.group_masks(y, source, old)
    assert {name: int(ix.sum()) for name, ix in masks.items()} == dict(
        primary_seven=21, seven_MACGS=7, seven_ModelNet=14,
        original_60=60, original_other53=53, merged_60=74)
    assert not np.any(masks['seven_ModelNet'] & old)
    bad_source = source.copy(); bad_source[0] = 'ModelNet'
    with pytest.raises(ValueError, match='source metadata'):
        analysis.group_masks(y, bad_source, old)


def test_prediction_identity_and_full_output_are_required(tmp_path):
    meta = dict(ids=np.array(['1/a.ply', '4/b.ply']), labels=np.array([1, 4]))
    path = tmp_path / 'sample.npz'
    logits = np.zeros((2, 60))
    np.savez(path, logits=logits, **meta)
    np.testing.assert_array_equal(checked_arrays(path, meta), logits)
    np.savez(path, logits=logits, ids=meta['ids'][::-1], labels=meta['labels'])
    with pytest.raises(ValueError, match='ids mismatch'):
        checked_arrays(path, meta)
    np.savez(path, logits=logits[:, :7], **meta)
    with pytest.raises(ValueError, match='full-head'):
        checked_arrays(path, meta)


def test_signal_requires_effect_size_all_seeds_and_uncertainty():
    row = dict(mean_gain_pp=.6, paired_seed_gains_pp=[.4, .6, .8], intervals_95_pp={'crossed': [.1, 1.1]})
    assert analysis.primary_signal(row)
    assert not analysis.primary_signal(dict(row, mean_gain_pp=.49))
    assert not analysis.primary_signal(dict(row, paired_seed_gains_pp=[-.1, 1., .9]))
    assert not analysis.primary_signal(dict(row, intervals_95_pp={'crossed': [0., 1.1]}))


def test_primary_and_descriptive_comparisons_have_fixed_order():
    specs = analysis.comparison_specs()
    assert len(specs) == 20
    assert [(s['family'], s['group'], s['right'], s['bootstrap_seed']) for s in specs[:2]] == [
        ('trained', 'primary_seven', 'distance_mlp', 92001),
        ('trained', 'primary_seven', 'fixed_isotropic5', 92002)]
    assert [s['bootstrap_seed'] for s in specs] == list(range(92001, 92021))
    assert all(s['seeds'] == [11, 22, 33] for s in specs)
