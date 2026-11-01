"""Synthetic counterexamples for the independent Round15 TRAIN audit math."""
from copy import deepcopy
from math import log

import numpy as np
import pytest

from experiments import round15_regional_audit_metrics as M


@pytest.fixture(scope='module')
def parts():
    return M.literal_parts()


@pytest.fixture(scope='module')
def identity_labels():
    y, x = np.mgrid[:144, :192]
    return dict(gt_xy=np.stack((x, y), -1).reshape(-1, 2).astype(np.float64),
                gt_visible=np.ones(144 * 192, dtype=bool))


def test_literal_partitions_cover_all_pixels_and_preserve_continuous_moments(parts):
    for name, part in parts.items():
        assert len(part['bounds']) == M.NATIVE_SIZES[name]
        assert part['mass'].sum() == 27648
        np.testing.assert_array_equal(np.bincount(part['owner']), part['mass'])
        np.testing.assert_array_equal(part['lookup'][4::8, 4::8].ravel(), part['atoms'])
        np.testing.assert_allclose(part['mean'] * 8, (part['bounds'][:, :2] + part['bounds'][:, 2:] - 1) / 2)
        width = part['bounds'][:, 2:] - part['bounds'][:, :2]
        np.testing.assert_allclose(np.diagonal(part['cov'], axis1=1, axis2=2) * 64, width ** 2 / 12)
        np.testing.assert_array_equal(part['cov'][:, 0, 1], 0)
    assert np.all(parts['g24']['parent'] == -1)
    np.testing.assert_array_equal(parts['mixed']['bounds'][:5],
        [[0, 0, 16, 16], [16, 0, 24, 8], [24, 0, 32, 8], [16, 8, 24, 16], [24, 8, 32, 16]])
    np.testing.assert_array_equal(parts['mixed']['bounds'][5:9],
        [[32, 0, 48, 8], [32, 8, 48, 16], [48, 0, 56, 16], [56, 0, 64, 16]])


def test_native_counts_all_fourteen_conditions_keep_the_identical_dense_population(parts, identity_labels):
    common_reference = None
    for q, k in M.CONDITIONS:
        direct, common = M.independent_counts(identity_labels, parts, q, k)
        assert direct.shape == (M.NATIVE_SIZES[q], M.NATIVE_SIZES[k])
        np.testing.assert_array_equal(direct.sum(-1), parts[q]['mass'])
        np.testing.assert_array_equal(direct.sum(0), parts[k]['mass'])
        assert common.sum() == direct.sum() == 27648
        if common_reference is None:
            common_reference = common
        else:
            np.testing.assert_array_equal(common, common_reference)
    np.testing.assert_array_equal(np.sort(common_reference[0][common_reference[0] > 0]), [64] * 4)
    direct, _ = M.independent_counts(identity_labels, parts, 'g24', 'g8')
    np.testing.assert_array_equal(np.sort(direct[0][direct[0] > 0]), [64] * 9)


def test_nearest_boundary_ties_and_invisible_nonfinite_pixels(parts):
    xy = np.full((27648, 2), np.nan)
    visible = np.zeros(27648, dtype=bool)
    visible[:5] = True
    xy[:5] = [[-.5, -.5], [7.5, 0], [7.5 - 2e-9, 0], [7.5 - .25e-9, 0], [191.5 - 1e-10, 143.5 - 1e-10]]
    direct, common = M.reconstruct_targets(xy, visible, parts['g8'], parts['g8'])
    assert direct[0, 0] == 2 and direct[0, 1] == 2 and direct[0, -1] == 1
    assert direct.sum() == common.sum() == 5
    xy[0] = [191.5, 0]
    with pytest.raises(ValueError, match='outside image'):
        M.reconstruct_targets(xy, visible, parts['g8'], parts['g8'])
    visible[0] = False
    assert M.reconstruct_targets(xy, visible, parts['g8'], parts['g8'])[0].sum() == 4


def test_common_map_preserves_constant_spatial_density_and_identity(parts):
    for q, k in M.PARTITION_CONDITIONS:
        p = np.broadcast_to(parts[k]['mass'] / 27648, (M.NATIVE_SIZES[q], M.NATIVE_SIZES[k]))
        np.testing.assert_allclose(M.common_prediction(p, parts[q], parts[k]), 1 / 432, atol=1e-17, rtol=0)
    rng = np.random.default_rng(7)
    p = rng.random((108, 432)); p /= p.sum(-1, keepdims=True)
    np.testing.assert_array_equal(M.common_prediction(p, parts['g16'], parts['g8']), p)
    with pytest.raises(ValueError, match='no exact common'):
        M.common_prediction(np.full((48, 432), 1 / 432), parts['g24'], parts['g8'])


def test_common_queries_use_pixel_area_not_token_count(parts):
    # An independently constructed valid partition with unequal areas in the
    # first g16 parent distinguishes the correct area rule from token averaging.
    bounds = np.vstack(([[0, 0, 16, 8], [0, 8, 8, 16], [8, 8, 16, 16]], parts['g16']['bounds'][1:]))
    query = M.prepare_partition(bounds)
    p = np.zeros((len(bounds), 432)); p[:, 0] = 1
    p[1] = 0; p[1, 1] = 1
    p[2] = 0; p[2, 2] = 1
    common = M.common_prediction(p, query, parts['g8'])
    np.testing.assert_array_equal(common[0, :3], [.5, .25, .25])
    assert not np.allclose(common[0, :3], 1 / 3)


def test_divergences_hand_calculation_clamp_and_zero_mass():
    p = np.array([[.25, .75], [.9, .1]])
    counts = np.array([[2, 0], [0, 0]])
    kl, js, mass = M.divergence_sums(p, counts)
    assert mass == 2
    assert kl == pytest.approx(2 * log(4))
    expected_js = log(1 / .625) + .25 * log(.25 / .625) + .75 * log(.75 / .375)
    assert js == pytest.approx(expected_js)
    p[1] = [.01, .99]
    assert M.divergence_sums(p, counts) == (kl, js, mass)
    clipped, _, _ = M.divergence_sums([[0., 1.]], [[2, 0]])
    assert clipped == pytest.approx(-2 * log(1e-15))
    with pytest.raises(ValueError, match='unnormalized'):
        M.divergence_sums([[.2, .2]], [[1, 0]])
    with pytest.raises(ValueError, match='nonnegative integers'):
        M.divergence_sums([[.5, .5]], [[.5, .5]])


def test_metric_keeps_direct_loss_distinct_from_common_loss(parts, identity_labels):
    # Swapping horizontal g8 children inside each g16 parent is wrong natively,
    # but area-averaging to the common parent cancels the permutation exactly.
    p = np.eye(432)[np.arange(432) ^ 1]
    direct, common = M.independent_counts(identity_labels, parts, 'g8', 'g8')
    values = M.measure(p, parts, 'g8', 'g8', 'partition', direct, common)
    assert values['kl_sum'] == values['js_sum'] == 0
    assert values['direct_kl_sum'] == pytest.approx(-27648 * log(1e-15))
    assert values['direct_js_sum'] == pytest.approx(27648 * log(2))
    direct, common = M.independent_counts(identity_labels, parts, 'g24', 'g8')
    values = M.measure(np.full((48, 432), 1 / 432), parts, 'g24', 'g8', 'scale', direct, common)
    assert values['kl_sum'] == values['direct_kl_sum']
    assert values['js_sum'] == values['direct_js_sum']


def minimal_metrics():
    return {family: [dict(grid_index=i, config=dict(h=h, temperature=t), J=20.)
                     for i, (h, t) in enumerate(M.GRID)] for family in M.FAMILIES}


def test_selector_keeps_primary_constraint_and_unconstrained_controls_separate():
    metrics = minimal_metrics()
    for index, value in ((0, .1), (6, 2.), (9, 3.)):
        metrics['quartic_full'][index]['J'] = value
    metrics['quartic_point'][3].update(J=.7, pck8=0.)
    metrics['gaussian_point'][4].update(J=.8, pck8=1.)
    metrics['gaussian_quad16'][1].update(J=.5, pck8=0.)
    metrics['gaussian_quad4'][7].update(J=.6, pck8=1.)
    selection = M.select_metrics(metrics, [9, 6])
    assert selection['primary']['grid_index'] == 6
    assert selection['unconstrained_primary']['grid_index'] == 0
    assert selection['strong_point']['family'] == 'quartic_point'
    assert selection['strong_point']['grid_index'] == 3
    assert selection['strong_quadrature']['family'] == 'gaussian_quad16'
    assert selection['strong_quadrature']['grid_index'] == 1
    assert selection['eligible_indices'] == [6, 9]
    assert len(selection['controls']) == 7


def test_selector_fixed_grid_and_cross_family_exact_ties():
    metrics = minimal_metrics()
    metrics['gaussian_point'][7]['J'] = metrics['gaussian_point'][2]['J'] = 1.
    metrics['quartic_point'][0]['J'] = 1.
    result = M.select_metrics(metrics, [9, 6])
    assert result['primary']['grid_index'] == 6
    assert result['strong_point']['family'] == 'gaussian_point'
    assert result['strong_point']['grid_index'] == 2
    assert result['strong_quadrature']['family'] == 'gaussian_quad4'
    assert result['strong_quadrature']['grid_index'] == 0


@pytest.mark.parametrize('eligible', [[], [0, 0], [-1], [12], [True], [6.0]])
def test_selector_rejects_invalid_or_empty_feasibility_without_fallback(eligible):
    with pytest.raises(ValueError):
        M.select_metrics(minimal_metrics(), eligible)


def test_selector_rejects_partial_or_mislabeled_control_grid():
    metrics = minimal_metrics()
    metrics['gaussian_quad16'].pop()
    with pytest.raises(ValueError, match='twelve'):
        M.select_metrics(metrics, [6, 9])
    metrics = minimal_metrics()
    metrics['gaussian_fixed'][0]['config']['h'] = .5
    with pytest.raises(ValueError, match='ordered regional grid'):
        M.select_metrics(metrics, [6, 9])


@pytest.fixture(scope='module')
def full_population():
    entries, rows = [], []
    for s, scene in enumerate(M.SCENES):
        for pair_index in range(16):
            pair = [pair_index * 15, pair_index * 15 + 5]
            entries.append(dict(scene=scene, pair=pair, split='train'))
            mass = (1, 10, 100)[s] * (pair_index + 1)
            for family_index, family in enumerate(M.FAMILIES):
                for grid_index in range(12):
                    offset = (family_index + 1) / 100 + grid_index / 100
                    for c, (query, key) in enumerate(M.CONDITIONS):
                        is_partition = c < 9
                        common_rate = (s + 1) * (pair_index + 1) / 16 + (c + 1) / 10 + offset
                        direct_rate = (common_rate + 5 if is_partition else
                            2 * (s + 1) * (pair_index + 1) / 16 + (c - 8) / 10 + offset)
                        rows.append(dict(scene=scene, pair=pair, family=family, grid_index=grid_index,
                            query=query, key=key, scenario='partition' if is_partition else 'scale',
                            kl_sum=mass * (common_rate if is_partition else direct_rate),
                            js_sum=mass * .1, direct_kl_sum=mass * direct_rate,
                            direct_js_sum=mass * .1, mass=mass,
                            n_query=M.NATIVE_SIZES[query], n_key=M.NATIVE_SIZES[key]))
    return entries, rows


def test_full_fixed_J_rejects_pooling_scenes_tokens_or_all_fourteen_conditions(full_population):
    entries, rows = full_population
    result = M.aggregate(rows, [6, 9], entries)
    assert result['population']['rows'] == 64512
    assert result['population']['pairs'] == 48
    assert result['population']['mass_by_scene'] == dict(zip(M.SCENES, [136, 1360, 13600]))
    first = result['metrics']['quartic_full'][0]
    # sum_{i=1}^16 i*(i/16) / sum_i i = .6875. Scenes remain equally
    # weighted even though their visible populations differ by factors of ten.
    assert first['common9_kl'] == pytest.approx(2 * .6875 + .5 + .01)
    assert first['direct5_kl'] == pytest.approx(4 * .6875 + .3 + .01)
    assert first['J'] == pytest.approx(2.4725)
    pooled_scene_factor = (1 * 1 + 10 * 2 + 100 * 3) / 111
    wrong_global = .5 * (pooled_scene_factor * .6875 + .51) + .5 * (2 * pooled_scene_factor * .6875 + .31)
    wrong_fourteen = (9 * first['common9_kl'] + 5 * first['direct5_kl']) / 14
    wrong_pair_equal = .5 * (2 * 8.5 / 16 + .51) + .5 * (4 * 8.5 / 16 + .31)
    assert all(abs(first['J'] - wrong) > .1 for wrong in (wrong_global, wrong_fourteen, wrong_pair_equal))
    assert first['by_scene_condition'][M.SCENES[0]]['g8_g8']['direct_kl'] - first['by_scene_condition'][M.SCENES[0]]['g8_g8']['kl'] == pytest.approx(5.)
    assert result['selection']['primary']['grid_index'] == 6
    assert result['selection']['unconstrained_primary']['grid_index'] == 0
    assert result['selection']['primary']['J'] == pytest.approx(2.5325)
    assert result['selection']['strong_point']['family'] == 'quartic_point'
    # The shared denominator is per pair, not replicated 1,344 times in this
    # reported unique-population count.
    assert result['population']['unique_visible_pixel_mass'] == 15096


@pytest.mark.parametrize('change', [
    dict(mass=2), dict(n_query=108), dict(n_key=108), dict(grid_index=12),
    dict(grid_index=True), dict(scenario='scale'), dict(scene='new_test_scene'),
    dict(family='weakened_point'), dict(kl_sum=float('nan')), dict(js_sum=1.),
    dict(condition='g16_g8'), dict(config=dict(h=.5, temperature=0.)),
])
def test_population_rejects_changed_rows(full_population, change):
    entries, rows = full_population
    changed = list(rows)
    changed[0] = dict(rows[0], **change)
    with pytest.raises(ValueError):
        M.validate_population(changed, entries)


def test_population_rejects_duplicates_missing_rows_and_nontrain_cohort(full_population):
    entries, rows = full_population
    changed = list(rows); changed[1] = changed[0]
    with pytest.raises(ValueError, match='duplicate'):
        M.validate_population(changed, entries)
    with pytest.raises(ValueError, match='64512'):
        M.validate_population(rows[:-1], entries)
    bad_entries = deepcopy(entries); bad_entries[0]['split'] = 'val'
    with pytest.raises(ValueError, match='TRAIN only'):
        M.validate_population(rows, bad_entries)
    bad_entries = list(entries); bad_entries[1] = entries[0]
    with pytest.raises(ValueError, match='48 unique'):
        M.validate_population(rows, bad_entries)


def test_population_rejects_scale_common_metric_substitution(full_population):
    entries, rows = full_population
    changed = list(rows)
    changed[9] = dict(rows[9], kl_sum=rows[9]['kl_sum'] + .01)
    with pytest.raises(ValueError, match='direct native'):
        M.validate_population(changed, entries)


def test_invalid_partitions_do_not_silently_reassign_pixels(parts):
    bounds = parts['g16']['bounds']
    with pytest.raises(ValueError, match='incomplete'):
        M.prepare_partition(bounds[:-1])
    with pytest.raises(ValueError, match='overlapping'):
        M.prepare_partition(np.vstack((bounds, bounds[:1])))
    with pytest.raises(ValueError, match='integer rectangles'):
        M.prepare_partition(bounds.astype(float) + .5)
