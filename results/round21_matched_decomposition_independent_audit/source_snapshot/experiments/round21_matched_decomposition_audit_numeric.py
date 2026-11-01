"""Independent NumPy Gaussian/full4 references for the fixed R21 diagnostic.

Production imports are not used by the independent mathematical path.
Gaussian native-cell GL64/96 and scaled-temperature references are inherited
from the frozen independent R17 reference, with new mixture reconstruction.
"""
from __future__ import annotations

import numpy as np
from scipy.special import softmax
from experiments import round10_partition_audit as SUPPORT
from experiments import round17_trace_numerical_audit as REF
from experiments import round19_center_semantics_audit as ANCHOR
from experiments import round21_matched_decomposition_audit_metrics as M

PARTITIONS = M.QUERIES
GEOMETRIES, FAMILIES = M.GEOMETRIES, M.FAMILIES
REFERENCE_ORDERS = (64, 96)
REFERENCE_POLICY = dict(gaussian_production_order=32, reference_orders=[64, 96],
    gaussian_reference_function='experiments.round17_trace_numerical_audit.independent_mass',
    mixture_reference='independently reconstructed four within-stratum Gaussian moments; weighted native masses before one floor',
    reference_matrices=72, reference_probability_rows=576, unique_query_identities=72,
    fixed_rows_per_condition=8, fixed_first_pairs_per_scene=1,
    probability_tolerance=1e-4, reference_convergence_tolerance=1e-8,
    same_raw_mass_tolerance=1e-12, normalization_tolerance=1e-12,
    moment_normalized_tolerance=1e-4, content_absolute_tolerance=1e-5,
    anchor_pixel_tolerance=1e-9, exact_prepared_tau32_and_bias32=True)


def selected_indices(count):
    if type(count) is not int or count < 8:
        raise ValueError('at least eight literal native query rows required')
    return np.linspace(0, count - 1, 8, dtype=np.int64)


def second_moment(mean, covariance, anchor):
    mean, anchor = np.asarray(mean, np.float64), np.asarray(anchor, np.float64)
    covariance = REF._covariance(covariance)
    if mean.ndim != 2 or mean.shape[1] != 2 or anchor.shape != mean.shape or covariance.shape != (len(mean), 2, 2):
        raise ValueError('matching finite moment population required')
    if not np.isfinite(mean).all() or not np.isfinite(anchor).all():
        raise ValueError('finite anchors/moments required')
    d = mean - anchor
    return covariance + d[:, :, None] * d[:, None, :]


def mixture_moments(nodes, covariances, weights):
    nodes, covariances, weights = [np.asarray(x, np.float64) for x in (nodes, covariances, weights)]
    if (nodes.ndim != 3 or nodes.shape[1:] != (4, 2) or covariances.shape != (*nodes.shape[:2], 2, 2)
            or weights.shape != nodes.shape[:2] or not all(np.isfinite(x).all() for x in (nodes, covariances, weights))
            or np.any(weights < 0) or np.any(abs(weights.sum(1) - 1) > 1e-12)):
        raise ValueError('four finite Gaussian components and normalized weights required')
    REF._covariance(covariances)
    mean = np.sum(weights[:, :, None] * nodes, axis=1)
    delta = nodes - mean[:, None]
    cov = np.sum(weights[:, :, None, None] * (covariances + delta[:, :, :, None] * delta[:, :, None, :]), axis=1)
    return mean, cov


def full4_reference(inputs, bounds):
    xy = np.asarray(inputs['q_to_k_xy'], np.float64).reshape(144, 192, 2) / 8.
    within = np.asarray(inputs['q_to_k_pixel_cov'], np.float64).reshape(144, 192, 2, 2) / 64.
    means, covariances, weights = [], [], []
    for x0, y0, x1, y1 in bounds:
        if (x1 - x0) % 2 or (y1 - y0) % 2:
            raise ValueError('four equal native strata required')
        sx, sy = (x1 - x0) // 2, (y1 - y0) // 2
        nodes, covs = [], []
        for y in (y0, y0 + sy):
            for x in (x0, x0 + sx):
                points = xy[y:y + sy, x:x + sx].reshape(-1, 2)
                mean = points.mean(0); delta = points - mean
                nodes.append(mean)
                covs.append(np.einsum('ni,nj->ij', delta, delta) / len(points)
                            + within[y:y + sy, x:x + sx].mean((0, 1)))
        means.append(nodes); covariances.append(covs); weights.append([.25] * 4)
    return dict(nodes=np.asarray(means), node_cov=np.asarray(covariances), weights=np.asarray(weights))


def reference_schema(parts):
    literal = M.REGION.literal_parts(); schema = {}
    for q in PARTITIONS:
        bounds = np.asarray(parts[q]['bounds'])
        if bounds.dtype.kind not in 'iu' or not np.array_equal(bounds, literal[q]['bounds']):
            raise ValueError('original literal native bounds required')
        # Old accepted archives store int32; the independent literal geometry
        # supplies int64 after an exact integer-value comparison.
        b = literal[q]['bounds']; size = b[:, 2:] - b[:, :2]
        covariance = np.zeros((len(b), 2, 2), np.float64)
        covariance[:, 0, 0], covariance[:, 1, 1] = size[:, 0] ** 2 / 12., size[:, 1] ** 2 / 12.
        schema.update({q + '_bounds': b, q + '_area': literal[q]['area'],
            q + '_centers': (b[:, :2] + b[:, 2:] - 1.) / 2., q + '_covariance': covariance,
            q + '_parent_g16': literal[q]['parent']})
    return schema


def reference_banks(inputs, parts):
    schema = reference_schema(parts)
    queries = {}
    for q in PARTITIONS:
        full = SUPPORT.reference_support(inputs, schema, q)
        a, front, fov = ANCHOR.center_reference(inputs, schema[q + '_bounds'])
        mixture = full4_reference(inputs, schema[q + '_bounds'])
        full.update(mixture, anchor=a / 8., anchor_px=a, front=front, fov=fov,
                    anchor_cov=second_moment(full['mean'], full['cov'], a / 8.))
        queries[q] = full
    return queries, SUPPORT.reference_key(inputs, schema, 'g8'), schema


def probability(raw, bias32, bounds):
    raw, bias32, bounds = map(np.asarray, (raw, bias32, bounds))
    if (raw.dtype != np.float64 or bias32.dtype != np.float32 or raw.ndim != 2 or raw.shape != bias32.shape
            or bounds.shape != (raw.shape[1], 4) or not np.isfinite(raw).all() or np.any(raw < 0)
            or not np.isfinite(bias32).all()):
        raise ValueError('valid independent native masses and float32 bias required')
    area = np.prod(bounds[:, 2:] - bounds[:, :2], axis=1)
    if np.any(area <= 0):
        raise ValueError('positive native key area required')
    return softmax(np.log(raw + 1e-30 * area[None]) + bias32.astype(np.float64), axis=1)


def integral(mean, cov, nodes, node_cov, weights, bounds, geometry, order):
    if geometry not in GEOMETRIES or order not in REFERENCE_ORDERS:
        raise ValueError('unknown geometry or unfrozen quadrature order')
    if geometry == 'full4':
        raw = REF.independent_mass(nodes, node_cov, bounds, .25, order)
        return np.sum(raw * weights[:, :, None], axis=1)
    return REF.independent_mass(mean, cov, bounds, .25, order)


def _array(prepared, name, shape, dtype):
    value = np.asarray(prepared[name])
    if value.shape != shape or value.dtype != dtype or not np.isfinite(value).all():
        raise ValueError('invalid executed intermediate: ' + name)
    return value


def numeric_condition(query, key, bounds, family, actual, prepared):
    if family not in FAMILIES:
        raise ValueError('unknown fixed R21 role')
    geometry, state = family.rsplit('_', 1)
    indices = selected_indices(len(query['mean'])); n, k = 8, len(key['mean'])
    actual = np.asarray(actual)
    if actual.dtype != np.float64 or actual.shape != (n, k):
        raise ValueError('eight-row native probability sample required')
    mu = _array(prepared, 'mu32', (n, 2), np.float32)
    c = _array(prepared, 'cov32', (n, 2, 2), np.float32)
    a = _array(prepared, 'a64', (n, 2), np.float64)
    ca = _array(prepared, 'anchor_cov64', (n, 2, 2), np.float64)
    nodes = _array(prepared, 'full4_nodes32', (n, 4, 2), np.float32)
    nc = _array(prepared, 'full4_node_cov32', (n, 4, 2, 2), np.float32)
    w = _array(prepared, 'full4_weights32', (n, 4), np.float32)
    content = _array(prepared, 'content32', (n, k), np.float32)
    tau = _array(prepared, 'tau32', (n,), np.float32)
    bias = _array(prepared, 'bias32', (n, k), np.float32)
    raw = _array(prepared, 'raw_mass64', (n, k), np.float64)
    if not np.array_equal(ca, second_moment(mu, c, a)):
        raise ValueError('executed fixed-anchor second moment changed')
    et = REF.independent_temperature(ca, .25, 10.).astype(np.float32)
    eb = et[:, None] * content
    executed_bias = eb if state == 'content' else np.zeros_like(eb)
    reference_ca = query['anchor_cov'][indices]
    reference_content = query['features'][indices] @ key['features'].T
    rt = REF.independent_temperature(reference_ca, .25, 10.).astype(np.float32)
    rb = rt[:, None] * reference_content.astype(np.float32)
    if state == 'zero':
        rb = np.zeros_like(rb)
    rm = query['mean'][indices] if geometry == 'mu_c' else query['anchor'][indices]
    rc = reference_ca if geometry == 'a_ca' else query['cov'][indices]
    em = mu.astype(np.float64) if geometry == 'mu_c' else a
    ec = ca if geometry == 'a_ca' else c.astype(np.float64)
    refs, hybrids = {}, {}
    for order in REFERENCE_ORDERS:
        refs[order] = integral(rm, rc, query['nodes'][indices], query['node_cov'][indices], query['weights'][indices], bounds, geometry, order)
        hybrids[order] = integral(em, ec, nodes.astype(np.float64), nc.astype(np.float64), w.astype(np.float64), bounds, geometry, order)
    ps = {o: probability(refs[o], rb, bounds) for o in REFERENCE_ORDERS}
    hs = {o: probability(hybrids[o], executed_bias, bounds) for o in REFERENCE_ORDERS}
    check = REF.probability_check(actual, ps[64], ps[96], refs[96])
    hybrid = REF.probability_check(actual, hs[64], hs[96], hybrids[96])
    for result in (check, hybrid):
        for change in result['argmax_changes']:
            local = change['query_index']
            change['sampled_row_index'] = local
            change['query_index'] = int(indices[local])
    comparisons = [('mu', mu, query['mean'][indices]), ('cov', c, query['cov'][indices]),
        ('anchor_cov', ca, reference_ca), ('nodes', nodes, query['nodes'][indices]),
        ('node_cov', nc, query['node_cov'][indices]), ('weights', w, query['weights'][indices])]
    errors = {name: float(np.max(abs(x - y) / (1 + abs(y)))) for name, x, y in comparisons}
    mm, mc = mixture_moments(nodes, nc, w)
    mix_errors = dict(mean=float(np.max(abs(mm - mu) / (1 + abs(mu)))),
                      covariance=float(np.max(abs(mc - c) / (1 + abs(c)))))
    center_error = float(np.max(abs(a - query['anchor'][indices]))) * 8.
    content_error = float(np.max(abs(content - reference_content)))
    same = probability(raw, executed_bias, bounds); arithmetic_error = float(np.max(abs(actual - same)))
    exact_tau, exact_bias = np.array_equal(tau, et), np.array_equal(bias, eb)
    passed = bool(check['passed'] and hybrid['passed'] and max(errors.values()) <= 1e-4
        and max(mix_errors.values()) <= 1e-4 and center_error <= 1e-9 and content_error <= 1e-5
        and arithmetic_error <= 1e-12 and exact_tau and exact_bias)
    check.update(passed=passed, family=family, geometry=geometry, content_enabled=state == 'content',
        query_indices=indices.tolist(), hybrid=hybrid, moment_errors=errors, mixture_moment_errors=mix_errors,
        maximum_anchor_error_px=center_error, content_error=content_error,
        same_mass_probability_error=arithmetic_error, tau32_exact=bool(exact_tau), bias32_exact=bool(exact_bias),
        actual_raw_fov_mass=raw.sum(1).tolist(), actual_zero_fov_rows=int(np.count_nonzero(raw.sum(1) == 0)),
        actual_tiny_fov_rows=int(np.count_nonzero(raw.sum(1) < 1e-20)), actual_zero_raw_cells=int(np.count_nonzero(raw == 0)),
        reference_orders=list(REFERENCE_ORDERS), production_order=32)
    arrays = dict(actual=actual, reference64=ps[64], reference96=ps[96], hybrid64=hs[64], hybrid96=hs[96],
        same_mass=same, indices=indices, raw_reference64=refs[64], raw_reference96=refs[96],
        raw_hybrid64=hybrids[64], raw_hybrid96=hybrids[96], mixture_mean=mm, mixture_cov=mc,
        mixture_mean_residual=mm - mu, mixture_cov_residual=mc - c, **prepared)
    return check, arrays
