"""Synthetic diagnostic contracts; no experiment artifacts or real inputs."""
import copy

import numpy as np
import pytest

from experiments import round16_mechanism_metrics as M


KEY = np.array([[3.5, 3.5], [11.5, 3.5], [19.5, 3.5], [27.5, 3.5]], dtype=np.float64)


def softmax(logits):
    x = np.exp(logits - logits.max(1, keepdims=True))
    return x / x.sum(1, keepdims=True)


def make(weights, content=None, *, temperature=0., valid=None, mu=None, target=None, cov=None, h=.5):
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim == 1:
        weights = weights[None]
    n = len(weights)
    p0 = weights / weights.sum(1, keepdims=True)
    content = np.zeros_like(p0, dtype=np.float32) if content is None else np.asarray(content, dtype=np.float32)
    pT = p0.copy() if temperature == 0 else softmax(
        np.log(p0) + np.multiply(np.float32(temperature), content, dtype=np.float32).astype(np.float64))
    valid = np.ones(n, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    mu = np.tile(KEY[0], (n, 1)) if mu is None else np.asarray(mu, dtype=np.float64)
    target = np.tile(KEY[0], (n, 1)) if target is None else np.asarray(target, dtype=np.float64)
    cov = np.tile(np.eye(2), (n, 1, 1)) if cov is None else np.asarray(cov, dtype=np.float64)
    return M.analyze(p0, pT, content, mu, cov, KEY, target, valid, h=h, temperature=temperature)


def row(arrays, scene='a', pair='0', family='quartic_full', grid_index=0, query='g8'):
    return dict(scene=scene, pair=pair, family=family, grid_index=grid_index,
                query=query, stats=M.stats(arrays))


def test_exact_first_index_ties_and_stable_runner_up():
    a = make([[1, 1, 1, 1], [1, 4, 4, 2]], mu=[[7.5, 3.5], [15.5, 3.5]])
    np.testing.assert_array_equal(a['zero_winner'], [0, 1])
    np.testing.assert_array_equal(a['j2'], [1, 2])
    np.testing.assert_array_equal(a['nearest_mu'], [0, 1])
    np.testing.assert_array_equal(a['tie_count'], [4, 2])
    np.testing.assert_array_equal(a['top_geometry_gap'], [0, 0])
    assert a['winner'].dtype == np.int64 and a['flip'].dtype == np.bool_
    s = M.stats(a)['all']
    assert s['n_geometry_tie'] == 2 and s['n_flip'] == 0


def test_content_temperature_rounds_before_double_subtraction():
    c = np.array([[.12345679, .9876543, -.7, -.2]], dtype=np.float32)
    a = make([1, 1, 1, 1], c, temperature=10.)
    expected = float(np.float32(10.) * c[0, 1]) - float(np.float32(10.) * c[0, 0])
    wrong = 10. * (float(c[0, 1]) - float(c[0, 0]))
    assert expected != wrong
    assert a['content_gain'][0] == expected
    assert a['winner'][0] == 1 and a['zero_winner'][0] == 0
    assert a['identity_error'][0] < 2.e-15
    np.testing.assert_allclose(a['advantage'], a['content_gain'] - a['geom_penalty'], atol=2.e-15, rtol=0)


def test_harm_benefit_conservation_and_invalid_not_wrong():
    a = make([[7, 1, 1, 1], [1, 1, 1, 7], [1, 1, 1, 7], [1, 1, 1, 7]],
             [[0, 0, 0, 1], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
             temperature=5., valid=[True, True, True, False],
             target=[[3.5, 3.5], [3.5, 3.5], [3.5, 3.5], [np.nan, np.inf]])
    np.testing.assert_array_equal(a['correct'], [0, 1, 0, -1])
    np.testing.assert_array_equal(a['zero_correct'], [1, 0, 0, -1])
    assert np.isnan(a['mu_target_dist_px'][-1])
    for s in M.stats(a).values():
        assert s['n_valid'] == 3
        assert s['n_harm'] == 1 and s['n_benefit'] == 1
        assert s['n_both_wrong'] == 1 and s['n_both_correct'] == 0
        assert s['n_harm'] + s['n_benefit'] + s['n_both_wrong'] + s['n_both_correct'] == s['n_valid']
        assert s['n_correct'] - s['n_zero_correct'] == s['n_benefit'] - s['n_harm']
        assert s['mu_target_dist_px_count'] == 3
    summary = M.summarize([row(a)])['quartic_full']['0']['g8']
    assert summary['macro']['all']['pck8'] == 1/3
    assert summary['macro']['all']['net_pck_change'] == 0
    assert summary['by_scene']['a']['all']['additive']['n'] == 4
    assert summary['by_scene']['a']['valid']['additive']['n'] == 3


def test_pck_inclusive_eight_pixel_boundary():
    a = make([[9, 1, 1, 1], [1, 1, 9, 1], [1, 1, 1, 9]],
             target=np.tile([11.5, 3.5], (3, 1)))
    np.testing.assert_array_equal(a['correct'], [1, 1, 0])


def test_phase_fov_and_covariance_reference_units():
    a = make([[9, 1, 1, 1]] * 4,
             mu=[[-.5, -.5], [31.5, 3.5], [35.5, 3.5], [3.5, 3.5]],
             cov=[[[2, 1], [1, 2]], [[2, 0], [0, 2]], [[2, 0], [0, 1]], [[1, 0], [0, 2]]], h=.5)
    np.testing.assert_array_equal(a['mu_in_fov'], [True, False, False, True])
    np.testing.assert_array_equal(a['fov_distance_px'], [0, 0, 4, 0])
    np.testing.assert_array_equal(a['mu_phase_x'], [0, 0, .5, .5])
    np.testing.assert_array_equal(a['isotropic'], [False, True, False, False])
    np.testing.assert_allclose(a['S_condition'], [3.25/1.25, 1, 2.25/1.25, 2.25/1.25], atol=1.e-15)
    np.testing.assert_allclose(a['S_major_angle_rad'], [np.pi/4, 0, 0, np.pi/2], atol=1.e-15)
    assert np.all((a['S_major_angle_rad'] >= 0) & (a['S_major_angle_rad'] < np.pi))


def test_symmetry_domain_matches_frozen_quartic_scale_rule():
    make([9, 1, 1, 1], cov=[[[1.e6, 5.e-7], [0, 1.e6]]])
    with pytest.raises(ValueError, match='symmetric'):
        make([9, 1, 1, 1], cov=[[[1, 5.e-11], [0, 1]]])


def test_paired_four_cells_and_distances_conserve_exact_counts():
    full = make([[9, 1, 1, 1], [1, 1, 1, 9], [9, 1, 1, 1], [1, 1, 1, 9], [1, 1, 1, 9]],
                valid=[1, 1, 1, 1, 0])
    point = make([[1, 1, 1, 9], [9, 1, 1, 1], [9, 1, 1, 1], [1, 1, 1, 9], [9, 1, 1, 1]],
                 valid=[1, 1, 1, 1, 0])
    paired = M.paired_stats(full, point)
    for s in paired.values():
        assert s['n_full_correct_point_wrong'] == 1
        assert s['n_full_wrong_point_correct'] == 1
        assert s['n_both_correct'] == 1 and s['n_both_wrong'] == 1
        assert s['n_full_correct'] - s['n_point_correct'] == s['n_full_correct_point_wrong'] - s['n_full_wrong_point_correct']
    assert paired['all']['dist_mu_px_difference_sum'] == 24
    assert paired['valid']['dist_mu_px_difference_sum'] == 0
    r = dict(scene='a', pair='0', family='full_vs_point', grid_index=0, query='g8', stats=paired)
    macro = M.summarize([r])['full_vs_point']['0']['g8']['macro']
    assert macro['all']['full_minus_point_pck'] == 0
    assert macro['all']['full_harm_rate'] == .25
    assert macro['all']['dist_mu_px_difference_mean'] == 24/5


def test_paired_rejects_different_shared_targets_or_context():
    full = make([9, 1, 1, 1])
    point = copy.deepcopy(full)
    point['S_condition'][0] = 2.
    with pytest.raises(ValueError, match='shared geometry'):
        M.paired_stats(full, point)
    point = copy.deepcopy(full)
    point['valid'][0] = False
    for k in ('correct', 'zero_correct', 'nearest_correct'):
        point[k][0] = -1
    with pytest.raises(ValueError, match='valid population'):
        M.paired_stats(full, point)


def test_scene_macro_not_query_pooled_and_no_cross_scale():
    good = make([9, 1, 1, 1])
    bad = make([[1, 1, 1, 9]] * 9)
    rows = [row(good, scene='small'), row(bad, scene='large'), row(bad, scene='large', query='g24')]
    out = M.summarize(rows)['quartic_full']['0']
    assert out['g8']['macro']['all']['pck8'] == .5
    assert out['g24']['macro']['all']['pck8'] == 0
    assert out['g8']['by_scene']['large']['all']['additive']['n'] == 9
    assert 'n' not in out['g8']['macro']['all']
    assert M.summarize(rows) == M.summarize(rows[::-1])


def test_pairs_pool_inside_scene_before_scene_mean():
    rows = [row(make([9, 1, 1, 1]), scene='a', pair='0'),
            row(make([[1, 1, 1, 9]] * 3), scene='a', pair='1'),
            row(make([9, 1, 1, 1]), scene='b', pair='0')]
    out = M.summarize(rows)['quartic_full']['0']['g8']
    assert out['by_scene']['a']['all']['derived']['pck8'] == .25
    assert out['macro']['all']['pck8'] == .625
    with pytest.raises(ValueError, match='duplicate'):
        M.summarize(rows + [rows[0]])


def test_json_pair_lists_are_valid_and_duplicates_are_detected():
    first = row(make([9, 1, 1, 1]), pair=[30, 45])
    second = row(make([1, 1, 1, 9]), pair=[45, 60])
    result = M.summarize([first, second])['quartic_full']['0']['g8']
    assert result['macro']['all']['pck8'] == .5
    duplicate = copy.deepcopy(first)
    duplicate['pair'] = (30, 45)
    with pytest.raises(ValueError, match='duplicate'):
        M.summarize([first, duplicate])


def test_additive_float_reduction_keeps_small_signed_terms():
    a = make([[9, 1, 1, 1]] * 3)
    a['identity_residual'] = np.array([1.e16, 1., -1.e16])
    assert M.stats(a)['all']['identity_residual_sum'] == 1.


def test_empty_labeled_population_is_none_not_zero_or_dropped_scene():
    invalid = make([[1, 1, 1, 9]] * 2, valid=[False, False], target=[[np.nan, np.nan]] * 2)
    s = M.stats(invalid)
    assert s['all']['n'] == 2 and s['all']['n_valid'] == 0
    assert s['all']['n_both_wrong'] == 0 and s['valid']['n'] == 0
    rows = [row(invalid, scene='empty'), row(make([9, 1, 1, 1]), scene='other')]
    out = M.summarize(rows)['quartic_full']['0']['g8']
    assert out['macro']['all']['pck8'] is None
    assert out['macro']['valid']['dist_mu_px_mean'] is None
    assert out['macro']['all']['dist_mu_px_mean'] == 12
    assert out['by_scene']['empty']['valid']['derived']['flip_rate'] is None


def test_rejects_wrong_content_dtype_and_nonpositive_probabilities():
    p = np.full((1, 4), .25)
    args = [p, p, np.zeros((1, 4)), KEY[:1], np.eye(2)[None], KEY, KEY[:1], np.array([True])]
    with pytest.raises(ValueError, match='content32'):
        M.analyze(*args, h=.5, temperature=5.)
    args[2] = args[2].astype(np.float32)
    args[0] = np.array([[0., .5, .25, .25]])
    with pytest.raises(ValueError, match='strictly positive'):
        M.analyze(*args, h=.5, temperature=5.)


def test_all_numeric_accumulators_add_and_match_direct_concatenation():
    a = make([[7, 1, 1, 1], [1, 1, 1, 7]],
             [[0, 0, 0, 1], [1, 0, 0, 0]], temperature=5.)
    pieces = [{name: value[i:i+1] for name, value in a.items()} for i in range(2)]
    rows = [row(piece, pair=str(i)) for i, piece in enumerate(pieces)]
    out = M.summarize(rows)['quartic_full']['0']['g8']['by_scene']['a']
    direct = M.stats(a)
    for population in ('all', 'valid'):
        for name, value in direct[population].items():
            assert out[population]['additive'][name] == value
