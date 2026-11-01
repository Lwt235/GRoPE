"""Synthetic-only checks of the independent R21 mathematical path."""
from copy import deepcopy
import numpy as np
import pytest
from experiments import round21_matched_decomposition_audit_numeric as A


def synthetic_sample(family):
    count, k = 17, 4
    mu = np.column_stack((np.arange(count) / 64., np.arange(count) / 128.)).astype(np.float32)
    offsets = np.array([[-.125, -.125], [.125, -.125], [-.125, .125], [.125, .125]], np.float32)
    nodes = mu[:, None] + offsets[None]
    nc = np.broadcast_to(np.diag([.03125, .0625]).astype(np.float32), (count, 4, 2, 2)).copy()
    weights = np.full((count, 4), .25, np.float32)
    mm, mc = A.mixture_moments(nodes, nc, weights)
    cov = mc.astype(np.float32); anchor = mm + [.125 + 1e-10, -.0625]
    ca = A.second_moment(mu, cov, anchor)
    features = np.column_stack((np.arange(count) / 32., np.ones(count) / 4))
    kfeatures = np.array([[.25, .5], [-.5, .25], [.125, -.5], [-.25, -.125]])
    query = dict(mean=mm, cov=mc, anchor=anchor, anchor_cov=ca, nodes=nodes.astype(np.float64),
                 node_cov=nc.astype(np.float64), weights=weights.astype(np.float64), features=features)
    key = dict(mean=np.zeros((k, 2)), features=kfeatures)
    bounds = np.array([[-1., -1., 0., 0.], [0., -1., 1., 0.], [-1., 0., 0., 1.], [0., 0., 1., 1.]])
    indices = A.selected_indices(count); geometry, state = family.rsplit('_', 1)
    content = (features @ kfeatures.T).astype(np.float32)[indices]
    tau = A.REF.independent_temperature(ca[indices], .25, 10.).astype(np.float32)
    bias = tau[:, None] * content
    mean = mm[indices] if geometry == 'mu_c' else anchor[indices]
    covariance = ca[indices] if geometry == 'a_ca' else mc[indices]
    raw = A.integral(mean, covariance, query['nodes'][indices], query['node_cov'][indices],
                     query['weights'][indices], bounds, geometry, 64)
    actual = A.probability(raw, bias if state == 'content' else np.zeros_like(bias), bounds)
    prepared = dict(mu32=mu[indices], cov32=cov[indices], a64=anchor[indices], anchor_cov64=ca[indices],
                    full4_nodes32=nodes[indices], full4_node_cov32=nc[indices], full4_weights32=weights[indices],
                    content32=content, tau32=tau, bias32=bias, raw_mass64=raw)
    return query, key, bounds, actual, prepared


@pytest.mark.parametrize('family', A.FAMILIES)
def test_eight_roles_reference_hybrid_and_same_raw_arithmetic(family):
    args = synthetic_sample(family)
    result, arrays = A.numeric_condition(*args[:3], family, *args[3:])
    assert result['passed'] and result['tau32_exact'] and result['bias32_exact']
    assert result['query_indices'] == [0, 2, 4, 6, 9, 11, 13, 16]
    assert arrays['actual'].shape == (8, 4)
    assert result['same_mass_probability_error'] <= 1e-12
    assert result['hybrid']['max_reference_convergence_error'] <= 1e-8


def test_tau_failure_recorded_and_covariance_tamper_rejected():
    query, key, bounds, actual, prepared = synthetic_sample('a_ca_content')
    bad = deepcopy(prepared); bad['tau32'][0] = np.nextafter(bad['tau32'][0], np.float32(np.inf))
    result, _ = A.numeric_condition(query, key, bounds, 'a_ca_content', actual, bad)
    assert not result['passed'] and not result['tau32_exact']
    bad = deepcopy(prepared); bad['anchor_cov64'][0, 0, 0] += .001
    with pytest.raises(ValueError, match='second moment'):
        A.numeric_condition(query, key, bounds, 'a_ca_content', actual, bad)


def test_argmax_evidence_maps_sampled_indices_to_original_rows(monkeypatch):
    args = synthetic_sample('mu_c_zero')
    monkeypatch.setattr(A.REF, 'probability_check', lambda *args: dict(passed=True, argmax_changes=[dict(query_index=6)]))
    result, _ = A.numeric_condition(*args[:3], 'mu_c_zero', *args[3:])
    for item in (result, result['hybrid']):
        assert item['argmax_changes'] == [dict(sampled_row_index=6, query_index=13)]


def test_native_area_floor_once_and_no_tail_filtering():
    raw = np.zeros((8, 2), np.float64); bias = np.zeros((8, 2), np.float32)
    bounds = np.array([[0., 0., 1., 1.], [1., 0., 3., 2.]])
    np.testing.assert_allclose(A.probability(raw, bias, bounds), np.tile([.2, .8], (8, 1)), atol=1e-14, rtol=0)
    with pytest.raises(ValueError): A.probability(raw.astype(np.float32), bias, bounds)
    with pytest.raises(ValueError): A.integral(None, None, None, None, None, bounds, 'mu_c', 128)


def test_full4_real_moments_include_within_pixel_and_between_node_scatter():
    yy, xx = np.mgrid[:144, :192]
    xy = np.stack((xx + .25 * yy, yy + .125 * xx), -1).astype(np.float32)
    within = np.broadcast_to(np.array([[.125, .03125], [.03125, .25]], np.float32), (144, 192, 2, 2)).copy()
    inputs = dict(q_to_k_xy=xy.reshape(-1, 2), q_to_k_pixel_cov=within.reshape(-1, 2, 2))
    bounds = np.array([[0, 0, 24, 24], [24, 0, 48, 24]], np.int64)
    result = A.full4_reference(inputs, bounds)
    mean, cov = A.mixture_moments(result['nodes'], result['node_cov'], result['weights'])
    for index, (x0, y0, x1, y1) in enumerate(bounds):
        points = xy[y0:y1, x0:x1].astype(np.float64).reshape(-1, 2) / 8
        m = points.mean(0); d = points - m
        c = d.T @ d / len(points) + within[y0:y1, x0:x1].astype(np.float64).mean((0, 1)) / 64
        np.testing.assert_allclose(mean[index], m, rtol=0, atol=1e-15)
        np.testing.assert_allclose(cov[index], c, rtol=0, atol=1e-15)
    assert np.all(result['node_cov'][:, :, 0, 0] > .125 / 64)


@pytest.mark.parametrize('dtype', [np.int32, np.int64])
def test_original_archive_integer_bounds_use_literal_int64(dtype):
    parts = A.M.REGION.literal_parts()
    for q in A.PARTITIONS: parts[q]['bounds'] = parts[q]['bounds'].astype(dtype)
    schema = A.reference_schema(parts)
    assert all(schema[q + '_bounds'].dtype == np.int64 for q in A.PARTITIONS)


def test_float_or_modified_native_bounds_rejected():
    parts = A.M.REGION.literal_parts(); parts['g8']['bounds'] = parts['g8']['bounds'].astype(np.float64)
    with pytest.raises(ValueError, match='literal'): A.reference_schema(parts)
    parts = A.M.REGION.literal_parts(); parts['g24']['bounds'][0, 2] -= 1
    with pytest.raises(ValueError, match='literal'): A.reference_schema(parts)


@pytest.mark.parametrize('kind', ['negative', 'asymmetric', 'nonfinite', 'weight'])
def test_invalid_independent_component_statistics_rejected(kind):
    _, _, _, _, prepared = synthetic_sample('full4_zero')
    n, c, w = (prepared[k].copy() for k in ('full4_nodes32', 'full4_node_cov32', 'full4_weights32'))
    if kind == 'negative': c[0, 0, 0, 0] = -1
    elif kind == 'asymmetric': c[0, 0, 0, 1] = 1
    elif kind == 'nonfinite': n[0, 0, 0] = np.nan
    else: w[0, 0] = .5
    with pytest.raises(ValueError): A.mixture_moments(n, c, w)
