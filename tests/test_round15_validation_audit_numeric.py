"""Synthetic-only coverage for the future validation numerical audit."""
from copy import deepcopy

import numpy as np
import pytest

from experiments import round15_regional_audit_metrics as PARTS
from experiments import round15_validation_audit_numeric as N


def synthetic_inputs():
    y, x = np.mgrid[:144, :192]
    z = np.full((144, 192), 2.)
    points = np.stack(((x - 95.5) * z / 100, (y - 71.5) * z / 100, z), -1).reshape(-1, 3)
    J = np.zeros((27648, 2, 3)); J[:, 0, 0] = J[:, 1, 1] = 50
    J[:, 0, 2] = -25 * points[:, 0]; J[:, 1, 2] = -25 * points[:, 1]
    covariance = np.broadcast_to(np.diag([.0001, .0002, .0003]), (27648, 3, 3)).copy()
    projected = J @ covariance @ J.transpose(0, 2, 1)
    return dict(q_to_k_xy=np.stack((x, y), -1).reshape(-1, 2).astype(float),
        q_to_k_pixel_cov=projected, q_features=np.ones((432, 3)), k_features=np.ones((432, 3)),
        q_points3=points, q_pixel_cov3=covariance, q_to_k_transform=np.eye(4),
        k_K=np.array([[100., 0., 95.5], [0., 100., 71.5], [0., 0., 1.]]),
        q_predicted_depth=z, k_predicted_depth=z.copy())


def synthetic_operators():
    return {f'o{i:02d}': dict(family=family, h=.5, temperature=5.)
            for i, family in enumerate(N.FAMILIES)}


def test_fixed_conditions_have_one_zero_copy_and_all_six_perturbations():
    assert len(N.CONDITIONS) == 68
    assert len(set(N.CONDITIONS)) == 68
    assert sum(scenario == 'scale' for scenario, _, _, _ in N.CONDITIONS) == 5
    for state in N.STATES:
        assert sum(scenario == state for scenario, _, _, _ in N.CONDITIONS) == 9
    assert all(state == 'zero' for scenario, state, _, _ in N.CONDITIONS if scenario == 'scale')
    assert all(q != 'g24' and k != 'g24' for scenario, _, q, k in N.CONDITIONS if scenario != 'scale')


def test_projection_identity_and_metric_translation_preserve_all_pixels():
    inputs = synthetic_inputs()
    zero, record = N.reference_observations(inputs, 'zero')
    np.testing.assert_allclose(zero['q_to_k_xy'], inputs['q_to_k_xy'], atol=3e-14, rtol=0)
    np.testing.assert_allclose(zero['q_to_k_pixel_cov'], inputs['q_to_k_pixel_cov'], atol=3e-15, rtol=0)
    assert record['predicted_in_front_count'] == 27648
    assert record['zero_float32_xy_equal']
    assert not record['key_depth_used_by_positional_operator']
    for state, displacement in (('tx_plus01', .5), ('tx_minus01', -.5)):
        raw, record = N.reference_observations(inputs, state)
        np.testing.assert_allclose(raw['q_to_k_xy'] - zero['q_to_k_xy'],
                                  np.broadcast_to([displacement, 0.], (27648, 2)), atol=4e-14, rtol=0)
        assert raw['q_to_k_pixel_cov'].shape == (27648, 2, 2)
        assert record['transform'][0][3] == pytest.approx(displacement / 50)


def test_depth_scaling_both_fields_does_not_claim_key_depth_sensitivity():
    inputs = synthetic_inputs()
    zero, _ = N.reference_observations(inputs, 'zero')
    for state, factor in (('depth09', .9), ('depth11', 1.1)):
        raw, record = N.reference_observations(inputs, state)
        # Identity camera makes synchronous point/covariance depth scaling cancel
        # in the perspective projection. These are still different depth states.
        np.testing.assert_allclose(raw['q_to_k_xy'], zero['q_to_k_xy'], atol=4e-14, rtol=0)
        np.testing.assert_allclose(raw['q_to_k_pixel_cov'], zero['q_to_k_pixel_cov'], atol=4e-15, rtol=0)
        assert record['depth_factor'] == factor
        for side in ('q', 'k'):
            assert record['effective_depth_sha256'][side] == N.STRESS.array_sha(np.full((144, 192), 2 * factor))
    changed = deepcopy(inputs); changed['k_predicted_depth'][:] = 19.
    raw, changed_record = N.reference_observations(changed, 'depth11')
    baseline, baseline_record = N.reference_observations(inputs, 'depth11')
    np.testing.assert_array_equal(raw['q_to_k_xy'], baseline['q_to_k_xy'])
    np.testing.assert_array_equal(raw['q_to_k_pixel_cov'], baseline['q_to_k_pixel_cov'])
    assert changed_record['effective_depth_sha256']['k'] != baseline_record['effective_depth_sha256']['k']
    assert changed_record['effective_depth_sha256']['k'] == N.STRESS.array_sha(np.full((144, 192), 19. * 1.1))


def test_target_y_rotation_uses_degrees_and_left_multiplication():
    inputs = synthetic_inputs()
    for state, sign in (('ry_plus1', 1), ('ry_minus1', -1)):
        raw, record = N.reference_observations(inputs, state)
        a = sign * np.pi / 180
        R = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
        transformed = inputs['q_points3'] @ R.T
        hom = transformed @ inputs['k_K'].T
        np.testing.assert_allclose(raw['q_to_k_xy'], hom[:, :2] / hom[:, 2, None], atol=1e-12, rtol=0)
        np.testing.assert_allclose(np.asarray(record['transform'])[:3, :3], R, atol=0, rtol=0)
    # A preexisting translation must rotate with the camera rows, not be left
    # fixed while only the rotation block is changed.
    changed = deepcopy(inputs); changed['q_to_k_transform'][:3, 3] = [.1, .2, .3]
    _, record = N.reference_observations(changed, 'ry_plus1')
    a = np.pi / 180
    expected = [np.cos(a) * .1 + np.sin(a) * .3, .2, -np.sin(a) * .1 + np.cos(a) * .3]
    np.testing.assert_allclose(np.asarray(record['transform'])[:3, 3], expected, atol=1e-16, rtol=0)


def test_behind_camera_fallback_keeps_population_and_no_label_field_is_accepted():
    inputs = synthetic_inputs(); inputs['q_to_k_transform'][2, 3] = -2.2
    raw, record = N.reference_observations(inputs, 'tx_plus01')
    assert record['predicted_in_front_count'] == 0
    assert raw['q_to_k_xy'].shape == (27648, 2)
    assert np.isfinite(raw['q_to_k_xy']).all() and np.isfinite(raw['q_to_k_pixel_cov']).all()
    assert np.abs(raw['q_to_k_xy']).max() <= 1024
    with pytest.raises(ValueError, match='ten forward fields'):
        N.reference_observations(dict(synthetic_inputs(), gt_visible=np.ones(27648, bool)), 'zero')
    broken = synthetic_inputs(); broken['q_to_k_xy'][0, 0] += .1
    with pytest.raises(ValueError, match='zero reprojection'):
        N.reference_observations(broken, 'zero')


def test_matched_trace_is_independent_full_moment_trace_without_bandwidth_or_mutation():
    covariance = np.array([[[4., .5], [.5, 1.]], [[.25, -.1], [-.1, 2.]]])
    query = dict(cov=covariance.copy())
    actual = N.quartic_covariance(query, 'quartic_isotropic_matched')
    np.testing.assert_array_equal(actual, np.array([2.5 * np.eye(2), 1.125 * np.eye(2)]))
    np.testing.assert_array_equal(np.trace(actual, axis1=1, axis2=2), np.trace(covariance, axis1=1, axis2=2))
    np.testing.assert_array_equal(query['cov'], covariance)
    np.testing.assert_array_equal(N.quartic_covariance(query, 'quartic_full'), covariance)
    np.testing.assert_array_equal(N.quartic_covariance(query, 'quartic_point'), np.zeros_like(covariance))
    # Bandwidth belongs inside the integrator once, not inside this transform.
    boxes = np.array([[-.5, -.5, .5, .5], [.5, -.5, 1.5, .5]])
    means = np.zeros((2, 2))
    p64 = N.REF.QREF.independent_mass(means, actual, boxes, .5, 64)
    p96 = N.REF.QREF.independent_mass(means, actual, boxes, .5, 96)
    np.testing.assert_allclose(p64, p96, atol=2e-15, rtol=0)


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'isotropic_h', 'invalid_h', 'invalid_family', 'bool_temperature'])
def test_manifest_rejects_missing_controls_dedup_or_config_changes(mutation):
    operators = synthetic_operators()
    if mutation == 'missing': operators.pop('o07')
    if mutation == 'duplicate': operators['o09'] = dict(operators['o02'])
    if mutation == 'isotropic_h': operators['o08']['h'] = 1.
    if mutation == 'invalid_h': operators['o02']['h'] = .7
    if mutation == 'invalid_family': operators['o02']['family'] = 'new_fitted_kernel'
    if mutation == 'bool_temperature': operators['o02']['temperature'] = False
    with pytest.raises(ValueError):
        N.validate_operators(operators)


def test_all_68_conditions_all_operators_and_numerical_failures_are_preserved(monkeypatch):
    inputs = synthetic_inputs(); parts = PARTS.literal_parts(); operators = synthetic_operators()
    operators['o09'] = dict(family='gaussian_point', h=1., temperature=0.)
    quartic_calls = []

    def quartic_zero(mean, covariance, bounds, h, order):
        quartic_calls.append((order, covariance.copy()))
        mass = np.zeros((len(mean), len(bounds)), np.float64)
        # One reference-only convergence failure, kept even though actual vs
        # GL96 agrees. All later references represent the same zero-FOV case.
        if len(quartic_calls) == 1:
            assert order == 64
            mass[0, 0] = 1e-29
        return mass

    def gaussian_zero(query, bounds, family, h):
        return np.zeros((len(query['mean']), len(bounds)), np.float64)

    seen = []

    def loader(scenario, condition, identifier):
        seen.append((scenario, condition, identifier))
        q, k = condition.split('_')
        p = np.tile(parts[k]['mass'] / parts[k]['mass'].sum(), (len(parts[q]['mass']), 1))
        if scenario == 'zero' and condition == 'g8_g8' and identifier == 'o02':
            # First native argmax remains index zero; that never exempts an
            # inaccurate probability matrix from the frozen numerical bound.
            p[0] = 0.; p[0, 0] = 1.
        return p

    monkeypatch.setattr(N.REF.QREF, 'independent_mass', quartic_zero)
    monkeypatch.setattr(N.REF, 'gaussian_reference_mass', gaussian_zero)
    result = N.audit_pair(inputs, parts, loader, operators)
    assert not result['passed'] and result['numerical_failure_count'] == 2
    assert result['matrix_count'] == len(seen) == 680
    assert len(set(seen)) == 680
    assert result['query_rows'] == 5440
    assert result['quartic_matrix_count'] == 204 and result['gaussian_matrix_count'] == 476
    assert len(result['projection_records']) == 7
    assert {c['family'] for c in result['checks']} == set(N.FAMILIES)
    reference_failure = [c for c in result['checks'] if c['reference_convergence_error'] is not None
                         and c['reference_convergence_error'] > 1e-8]
    assert len(reference_failure) == 1 and reference_failure[0]['max_probability_error'] < 1e-12
    actual_failure = [c for c in result['checks'] if c['max_probability_error'] > 1e-4]
    assert len(actual_failure) == 1 and actual_failure[0]['argmax_change_count'] == 0
    assert all(c['reference_zero_fov_rows'] == 8 for c in result['checks'])
    for c in result['checks']:
        assert c['indices'] == np.linspace(0, len(parts[c['query']]['mass']) - 1, 8, dtype=int).tolist()
        assert c['config'] == operators[c['operator']]
    assert not result['labels_decoded'] and not result['configuration_selection']
    assert not result['scientific_gate_passed']


def test_complete_array_corruption_outside_selected_queries_is_rejected():
    inputs = synthetic_inputs(); parts = PARTS.literal_parts()

    def loader(scenario, condition, identifier):
        q, k = condition.split('_')
        p = np.full((len(parts[q]['mass']), len(parts[k]['mass'])), 1 / len(parts[k]['mass']))
        p[1, 0] = np.nan  # Index1 is outside the fixed first g8 linspace sample.
        return p

    with pytest.raises(ValueError, match='invalid complete native sample'):
        N.audit_pair(inputs, parts, loader, synthetic_operators())


def test_source_closure_binds_old_independent_references_and_excludes_new_producer_model():
    assert set(N.REF.SOURCE_NAMES).issubset(N.SOURCE_NAMES)
    assert set(N.STRESS.AUDIT_SOURCES).issubset(N.SOURCE_NAMES)
    assert 'experiments/round15_validation.py' not in N.SOURCE_NAMES
    assert 'models/round15_validation_pe.py' not in N.SOURCE_NAMES
    assert 'experiments/round15_validation_metrics.py' not in N.SOURCE_NAMES
