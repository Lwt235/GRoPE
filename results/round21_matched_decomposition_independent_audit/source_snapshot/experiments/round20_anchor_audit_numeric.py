"""R20 independent forward mathematics; no real-data loader or selector.

Shared-source replay belongs to the outer auditor. Here all center, support,
content and integration references follow previously frozen independent math.
The imported quartic module has unused production imports; only its independent
NumPy Cartesian GL function is called, and its complete import closure is frozen.
"""
from __future__ import annotations

import numpy as np
from scipy.special import softmax

from experiments import round10_partition_audit as SUPPORT
from experiments import round15_regional_audit_metrics as REGION
from experiments import round15_train_numerical24 as QUARTIC
from experiments import round17_trace_numerical_audit as GAUSSIAN
from experiments.round19_center_semantics_audit import center_reference


PARTITIONS = ('g8', 'g16', 'mixed', 'g24')
CONDITIONS = REGION.CONDITIONS
FAMILIES = ('anchor_trace', 'mean_only_matched_tau', 'anchor_constant',
    'anchor_isotropic_matched', 'anchor_point_regional', 'anchor_point_localizer',
    'anchor_point_legacy', 'anchor_point_matched', 'anchor_fixed_regional',
    'anchor_fixed_matched', 'anchor_quartic_regional', 'anchor_quartic_localizer')
CONFIGS = dict(zip(FAMILIES, ((.25, 10.), (.25, 10.), (.25, 10.), (.25, 10.),
    (1., 5.), (.25, 5.), (.5, 5.), (.25, 10.), (.5, 5.), (.25, 10.), (1., 5.), (.25, 5.))))
TRACE_FAMILIES = ('anchor_trace', 'mean_only_matched_tau', 'anchor_isotropic_matched')
REFERENCE_ORDERS = (64, 96)
NUMERICAL_POLICY = dict(gaussian_production_order=32, quartic_production_order=24,
    gaussian_reference_function='experiments.round17_trace_numerical_audit.independent_mass',
    quartic_reference_function='experiments.round15_train_numerical24.independent_mass',
    reference_orders=[64, 96], probability_tolerance=1e-4,
    reference_convergence_tolerance=1e-8, same_raw_mass_tolerance=1e-12,
    moment_relative_to_one_plus_reference_tolerance=1e-4, content_absolute_tolerance=1e-5,
    anchor_pixel_tolerance=1e-9, normalization_tolerance=1e-12,
    fixed_rows_per_condition=8, fixed_first_pairs_per_scene=1,
    reference_matrices=504, reference_probability_rows=4032,
    all_projection_rows=39888, exact_prepared_tau32_and_bias32=True)
REFERENCE_POLICY = NUMERICAL_POLICY


def selected_indices(count):
    if type(count) is not int or count < 8:
        raise ValueError('at least eight literal query rows required')
    return np.linspace(0, count - 1, 8, dtype=np.int64)


def fixed_mean_second_moment(mean, covariance, anchor):
    mean, covariance, anchor = [np.asarray(x, np.float64) for x in (mean, covariance, anchor)]
    if (mean.ndim != 2 or mean.shape[1] != 2 or anchor.shape != mean.shape
            or covariance.shape != (len(mean), 2, 2)
            or any(not np.isfinite(x).all() for x in (mean, covariance, anchor))):
        raise ValueError('finite matching first/second moments and anchors required')
    covariance = GAUSSIAN._covariance(covariance)
    delta = mean - anchor
    return covariance + delta[:, :, None] * delta[:, None, :]


def reference_schema(parts):
    expected = REGION.literal_parts()
    if set(parts) != set(PARTITIONS):
        raise ValueError('all four original native partitions required')
    schema = {}
    for name in PARTITIONS:
        bounds = np.asarray(parts[name]['bounds'])
        if bounds.dtype != np.int64 or not np.array_equal(bounds, expected[name]['bounds']):
            raise ValueError('native partition definition changed')
        b = expected[name]['bounds']
        size = b[:, 2:] - b[:, :2]
        covariance = np.zeros((len(b), 2, 2), np.float64)
        covariance[:, 0, 0] = size[:, 0] ** 2 / 12.
        covariance[:, 1, 1] = size[:, 1] ** 2 / 12.
        schema.update({name + '_bounds': b, name + '_area': expected[name]['area'],
            name + '_centers': (b[:, :2] + b[:, 2:] - 1) / 2.,
            name + '_covariance': covariance, name + '_parent_g16': expected[name]['parent']})
    return schema


def reference_banks(inputs, parts):
    fields = ('q_to_k_xy', 'q_to_k_pixel_cov', 'q_features', 'k_features')
    for name in fields:
        if name not in inputs or not np.isfinite(inputs[name]).all():
            raise ValueError('finite original predicted observations required')
    if (np.shape(inputs['q_to_k_xy']) != (27648, 2)
            or np.shape(inputs['q_to_k_pixel_cov']) != (27648, 2, 2)
            or np.ndim(inputs['q_features']) != 2
            or np.shape(inputs['q_features']) != np.shape(inputs['k_features'])
            or np.shape(inputs['q_features'])[0] != 432):
        raise ValueError('wrong original dense support/content population')
    schema = reference_schema(parts)
    queries, keys = {}, {}
    for name in PARTITIONS:
        full = SUPPORT.reference_support(inputs, schema, name)
        a_px, front, inside = center_reference(inputs, schema[name + '_bounds'])
        full.update(anchor=a_px / 8., anchor_px=a_px, a_front=front, a_fov=inside,
                    anchor_cov=fixed_mean_second_moment(full['mean'], full['cov'], a_px / 8.))
        queries[name] = full
        keys[name] = SUPPORT.reference_key(inputs, schema, name)
    return queries, keys, schema


def covariance_for(family, original, anchor_cov, area):
    if family not in FAMILIES:
        raise ValueError('unknown fixed operator')
    if family == 'mean_only_matched_tau':
        return np.asarray(original, np.float64)
    if family in ('anchor_trace', 'anchor_constant'):
        return np.asarray(anchor_cov, np.float64)
    if family == 'anchor_isotropic_matched':
        trace = anchor_cov[:, 0, 0] + anchor_cov[:, 1, 1]
        return trace[:, None, None] * np.eye(2)[None] / 2.
    if family.startswith('anchor_fixed_'):
        return np.asarray(area, np.float64)[:, None, None] * np.eye(2)[None] / (64. * 12.)
    return np.zeros_like(anchor_cov, dtype=np.float64)


def prepared_bias(content, anchor_cov, family, h, temperature):
    content = np.asarray(content)
    if content.dtype != np.float32 or content.ndim != 2 or not np.isfinite(content).all():
        raise ValueError('finite executed float32 content required')
    if family in TRACE_FAMILIES:
        tau = GAUSSIAN.independent_temperature(anchor_cov, h, temperature).astype(np.float32)
    else:
        tau = np.full(len(content), temperature, dtype=np.float32)
    if tau.shape != (len(content),):
        raise ValueError('temperature population mismatch')
    bias = tau[:, None] * content
    return tau, bias


def probability(raw, bias32, bounds):
    raw, bias32, bounds = np.asarray(raw), np.asarray(bias32), np.asarray(bounds)
    if (raw.dtype != np.float64 or bias32.dtype != np.float32 or raw.shape != bias32.shape
            or raw.ndim != 2 or bounds.shape != (raw.shape[1], 4)
            or not np.isfinite(raw).all() or np.any(raw < 0) or not np.isfinite(bias32).all()):
        raise ValueError('invalid independent raw mass/bias/native keys')
    area = (bounds[:, 2] - bounds[:, 0]) * (bounds[:, 3] - bounds[:, 1])
    if np.any(area <= 0):
        raise ValueError('nonpositive native key area')
    return softmax(np.log(raw + 1e-30 * area[None]) + bias32.astype(np.float64), axis=1)


def integral(mean, covariance, bounds, h, order, family):
    if order not in REFERENCE_ORDERS:
        raise ValueError('reference order is frozen')
    function = QUARTIC.independent_mass if family.startswith('anchor_quartic_') else GAUSSIAN.independent_mass
    return function(mean, covariance, bounds, h, order)


def _array(prepared, name, shape, dtype):
    value = np.asarray(prepared[name])
    if value.shape != shape or value.dtype != dtype or not np.isfinite(value).all():
        raise ValueError('invalid executed intermediate: ' + name)
    return value


def numeric_condition(query, key, bounds, family, actual, prepared):
    """One fixed eight-query/all-key sample; returns evidence, never hides fails.

    query/key are independent NumPy reference banks. prepared contains role-owned
    executed mean64,cov64,content32,raw_mass64,tau32,bias32, plus shared primary
    audit context old_mean32,original_cov32,anchor_cov64. The latter are NOT
    point/fixed execution intermediates or charged to those operators' costs.
    """
    h, temperature = CONFIGS[family]
    indices = selected_indices(len(query['mean']))
    n, k = 8, len(key['mean'])
    actual = np.asarray(actual)
    if actual.shape != (n, k) or actual.dtype != np.float64:
        raise ValueError('fixed eight-row probability sample required')
    a = query['anchor'][indices]
    original = query['cov'][indices]
    anchor_cov = query['anchor_cov'][indices]
    covariance = covariance_for(family, original, anchor_cov, query['mass'][indices])
    content = query['features'][indices] @ key['features'].T
    mean_actual = _array(prepared, 'mean64', (n, 2), np.float64)
    cov_actual = _array(prepared, 'cov64', (n, 2, 2), np.float64)
    old_mean = _array(prepared, 'old_mean32', (n, 2), np.float32)
    old_cov = _array(prepared, 'original_cov32', (n, 2, 2), np.float32)
    ca_actual = _array(prepared, 'anchor_cov64', (n, 2, 2), np.float64)
    content_actual = _array(prepared, 'content32', (n, k), np.float32)
    raw_actual = _array(prepared, 'raw_mass64', (n, k), np.float64)
    tau_actual = _array(prepared, 'tau32', (n,), np.float32)
    bias_actual = _array(prepared, 'bias32', (n, k), np.float32)
    if np.any(raw_actual < 0):
        raise ValueError('negative executed raw mass')
    expected_ca = fixed_mean_second_moment(old_mean, old_cov, mean_actual)
    if not np.array_equal(ca_actual, expected_ca):
        raise ValueError('executed fixed-center second moment changed')
    expected_cov = covariance_for(family, old_cov, ca_actual, query['mass'][indices])
    if not np.array_equal(cov_actual, expected_cov):
        raise ValueError('executed family covariance changed')
    tau_ref, bias_ref = prepared_bias(content_actual, ca_actual, family, h, temperature)
    _, independent_bias = prepared_bias(content.astype(np.float32), anchor_cov, family, h, temperature)
    refs, hybrid_raws = {}, {}
    for order in REFERENCE_ORDERS:
        refs[order] = integral(a, covariance, bounds, h, order, family)
        hybrid_raws[order] = integral(mean_actual, cov_actual, bounds, h, order, family)
    p64 = probability(refs[64], independent_bias, bounds)
    p96 = probability(refs[96], independent_bias, bounds)
    hybrid64 = probability(hybrid_raws[64], bias_ref, bounds)
    hybrid96 = probability(hybrid_raws[96], bias_ref, bounds)
    same = probability(raw_actual, bias_ref, bounds)
    check = GAUSSIAN.probability_check(actual, p64, p96, refs[96])
    hybrid_check = GAUSSIAN.probability_check(actual, hybrid64, hybrid96, hybrid_raws[96])
    for detail in (check, hybrid_check):
        for change in detail['argmax_changes']:
            local_row = change['query_index']
            change.update(sampled_row_index=local_row, query_index=int(indices[local_row]))
    moment_errors = {name: float(np.max(np.abs(value - reference) / (1. + np.abs(reference))))
        for name, value, reference in (
            ('old_mean', old_mean, query['mean'][indices]), ('original_cov', old_cov, original),
            ('anchor_cov', ca_actual, anchor_cov), ('executed_cov', cov_actual, covariance))}
    content_error = float(np.max(np.abs(content_actual - content)))
    center_error_px = float(np.max(np.abs(mean_actual - a))) * 8.
    same_error = float(np.max(np.abs(actual - same)))
    tau_exact, bias_exact = np.array_equal(tau_actual, tau_ref), np.array_equal(bias_actual, bias_ref)
    passed = bool(check['passed'] and hybrid_check['passed'] and max(moment_errors.values()) <= 1e-4
        and content_error <= 1e-5 and center_error_px <= 1e-9 and same_error <= 1e-12
        and tau_exact and bias_exact)
    check.update(passed=passed, family=family, h=h, temperature=temperature,
        query_indices=indices.tolist(), moment_errors=moment_errors, content_error=content_error,
        maximum_anchor_error_px=center_error_px, same_mass_probability_error=same_error,
        tau32_exact=bool(tau_exact), bias32_exact=bool(bias_exact),
        hybrid=hybrid_check, reference_orders=list(REFERENCE_ORDERS),
        production_order=24 if family.startswith('anchor_quartic_') else 32,
        actual_raw_fov_mass=raw_actual.sum(1).tolist(),
        actual_zero_fov_rows=int(np.count_nonzero(raw_actual.sum(1) == 0)),
        actual_tiny_fov_rows=int(np.count_nonzero(raw_actual.sum(1) < 1e-20)),
        actual_zero_raw_cells=int(np.count_nonzero(raw_actual == 0)))
    check['prepared_provenance'] = dict(
        role_owned_execution=['mean64', 'cov64', 'content32', 'raw_mass64', 'tau32', 'bias32'],
        shared_primary_audit_context=['old_mean32', 'original_cov32', 'anchor_cov64'],
        shared_context_is_control_execution_cost=False)
    arrays = dict(indices=indices, actual=actual, reference64=p64, reference96=p96,
        hybrid64=hybrid64, hybrid96=hybrid96, same_mass=same,
        raw_reference64=refs[64], raw_reference96=refs[96], raw_actual=raw_actual,
        raw_hybrid64=hybrid_raws[64], raw_hybrid96=hybrid_raws[96],
        reference_mean=a, reference_cov=covariance, reference_anchor_cov=anchor_cov,
        reference_content=content, reference_tau32=tau_ref, reference_bias32=bias_ref,
        **prepared)
    return check, arrays
