"""R20 independent numerical references on literal synthetic arrays only."""
from copy import deepcopy

import numpy as np
import pytest

from experiments import round20_anchor_audit_numeric as A


def test_fixed_center_second_moment_psd_reduction_translation_and_units():
    mean = np.array([[1., 2.], [-3., 4.]], dtype=np.float32)
    cov = np.array([[[2., .3], [.3, .7]], [[0., 0.], [0., 0.]]], dtype=np.float32)
    a = mean.astype(np.float64) + [[.5, -.25], [1., 2.]]
    ca = A.fixed_mean_second_moment(mean, cov, a)
    assert np.linalg.eigvalsh(ca).min() >= 0.
    np.testing.assert_array_equal(A.fixed_mean_second_moment(mean, cov, mean), cov.astype(np.float64))
    np.testing.assert_array_equal(A.fixed_mean_second_moment(mean + [8., -16.], cov, a + [8., -16.]), ca)
    np.testing.assert_array_equal(A.fixed_mean_second_moment(mean * 8., cov * 64., a * 8.), ca * 64.)
    np.testing.assert_array_equal(ca[1], np.array([[1., 2.], [2., 4.]]))
    assert not np.array_equal(ca[1], np.zeros((2, 2)))  # C=0 alone is not the point limit when a differs.


@pytest.mark.parametrize('bad', ['nan', 'asymmetric', 'indefinite', 'shape'])
def test_second_moment_rejects_invalid_original_statistics(bad):
    mu = np.zeros((2, 2)); a = mu.copy(); cov = np.tile(np.eye(2), (2, 1, 1))
    if bad == 'nan': a[0, 0] = np.nan
    elif bad == 'asymmetric': cov[0, 0, 1] = 1.
    elif bad == 'indefinite': cov[0, 0, 0] = -1.
    else: cov = cov[:1]
    with pytest.raises(ValueError): A.fixed_mean_second_moment(mu, cov, a)


def test_matched_tau_float32_product_and_zero_temperature():
    content = np.array([[.1, -.7, .3], [.9, -.2, -.8]], np.float32)
    ca = np.array([np.eye(2), np.diag([.2, .7])], np.float64)
    expected_tau = (10 * .0625 / (.0625 + np.trace(ca, axis1=1, axis2=2))).astype(np.float32)
    values = [A.prepared_bias(content, ca, family, .25, 10.) for family in A.TRACE_FAMILIES]
    for tau, bias in values:
        np.testing.assert_array_equal(tau, expected_tau)
        np.testing.assert_array_equal(bias, expected_tau[:, None] * content)
    tau, bias = A.prepared_bias(content, ca, 'anchor_trace', .25, 0.)
    assert not tau.any() and not bias.any()
    tau, bias = A.prepared_bias(content, ca, 'anchor_constant', .25, 10.)
    np.testing.assert_array_equal(tau, [10., 10.])


def synthetic_sample(family):
    n, k = 8, 4
    a = np.tile([.2, .3], (n, 1)) + np.arange(n)[:, None] * [.03, -.02]
    old = (a + [.125, -.125]).astype(np.float32)
    cov = np.tile(np.array([[.1, .02], [.02, .2]], np.float32), (n, 1, 1))
    ca = A.fixed_mean_second_moment(old, cov, a)
    features = np.column_stack((np.linspace(.1, .8, n), np.linspace(.2, -.4, n)))
    key_features = np.array([[.2, .9], [.5, -.2], [-.6, .3], [-.4, -.5]])
    query = dict(mean=old.astype(np.float64), cov=cov.astype(np.float64), anchor=a,
        anchor_cov=ca, features=features, mass=np.full(n, 64., np.float64))
    key = dict(mean=np.zeros((k, 2)), features=key_features)
    bounds = np.array([[-1., -1., 0., 0.], [0., -1., 1., 0.],
                       [-1., 0., 0., 1.], [0., 0., 1., 1.]], np.float64)
    h, temperature = A.CONFIGS[family]
    geometry_cov = A.covariance_for(family, cov, ca, query['mass'])
    content = (features @ key_features.T).astype(np.float32)
    tau, bias = A.prepared_bias(content, ca, family, h, temperature)
    raw = A.integral(a, geometry_cov, bounds, h, 64, family)
    p = A.probability(raw, bias, bounds)
    prepared = dict(mean64=a, cov64=geometry_cov, old_mean32=old, original_cov32=cov,
        anchor_cov64=ca, content32=content, raw_mass64=raw, tau32=tau, bias32=bias)
    return query, key, bounds, p, prepared


@pytest.mark.parametrize('family', A.FAMILIES)
def test_each_fixed_family_keeps_eight_rows_and_full_reference_hybrid_arithmetic(family):
    query, key, bounds, p, prepared = synthetic_sample(family)
    result, arrays = A.numeric_condition(query, key, bounds, family, p, prepared)
    assert result['passed']
    assert result['query_indices'] == list(range(8))
    assert result['tau32_exact'] and result['bias32_exact']
    assert arrays['reference96'].shape == (8, 4)
    assert result['max_reference_convergence_error'] <= 1e-8
    assert result['hybrid']['max_reference_convergence_error'] <= 1e-8


def test_prepared_tau_failure_is_retained_and_geometry_tamper_rejected():
    query, key, bounds, p, prepared = synthetic_sample('anchor_trace')
    bad = deepcopy(prepared)
    bad['tau32'][0] = np.nextafter(bad['tau32'][0], np.float32(np.inf))
    result, _ = A.numeric_condition(query, key, bounds, 'anchor_trace', p, bad)
    assert not result['passed'] and not result['tau32_exact']
    bad = deepcopy(prepared); bad['anchor_cov64'][0, 0, 0] += .001
    with pytest.raises(ValueError, match='second moment'):
        A.numeric_condition(query, key, bounds, 'anchor_trace', p, bad)


def test_native_area_floor_and_tail_rows_are_not_filtered():
    bounds = np.array([[0., 0., 1., 1.], [1., 0., 3., 2.]], np.float64)
    raw = np.zeros((8, 2), np.float64)
    bias = np.zeros((8, 2), np.float32)
    probability = A.probability(raw, bias, bounds)
    np.testing.assert_allclose(probability, np.tile([.2, .8], (8, 1)), atol=1e-14, rtol=0)
    assert probability.shape == (8, 2)
    with pytest.raises(ValueError, match='frozen'):
        A.integral(np.zeros((8, 2)), np.zeros((8, 2, 2)), bounds, .25, 128, 'anchor_trace')


def test_original_mixed_bounds_reject_checkerboard_substitution():
    parts = A.REGION.literal_parts()
    A.reference_schema(parts)
    altered = deepcopy(parts)
    altered['mixed']['bounds'] = altered['g8']['bounds']
    with pytest.raises(ValueError, match='definition'):
        A.reference_schema(altered)
