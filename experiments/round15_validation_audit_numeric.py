"""Independent future-validation numerical checks, developed on synthetic data.

The caller owns cohort/phase authorization and loads only the two predeclared
first validation pairs after the new audit is frozen. No new producer/model,
selector, label loader, or metric function is imported. All operators in the
frozen manifest are checked at every one of the 68 conditions, eight literal
linspace queries and all native keys. No configuration is selected here.
"""
from __future__ import annotations

import numpy as np

from experiments import round12_stress_result_audit as STRESS
from experiments import round15_regional_audit_numeric as REF


STATES = ('zero', 'depth09', 'depth11', 'ry_plus1', 'ry_minus1', 'tx_plus01', 'tx_minus01')
PARTITIONS = ('g8', 'g16', 'mixed', 'g24')
ORIGINAL = tuple((q, k) for q in PARTITIONS[:3] for k in PARTITIONS[:3])
SCALE = (('g8', 'g24'), ('g24', 'g8'), ('g16', 'g24'), ('g24', 'g16'), ('g24', 'g24'))
CONDITIONS = tuple((state, state, q, k) for state in STATES for q, k in ORIGINAL) + tuple(
    ('scale', 'zero', q, k) for q, k in SCALE)
FAMILIES = REF.FAMILIES + ('quartic_isotropic_matched',)
FIELDS = ('q_to_k_xy', 'q_to_k_pixel_cov', 'q_features', 'k_features',
          'q_points3', 'q_pixel_cov3', 'q_to_k_transform', 'k_K',
          'q_predicted_depth', 'k_predicted_depth')
REFERENCE_ORDERS = (64, 96)
PROBABILITY_TOLERANCE = 1e-4
REFERENCE_TOLERANCE = 1e-8
SOURCE_NAMES = tuple(dict.fromkeys((
    'experiments/round15_validation_audit_numeric.py',
    'tests/test_round15_validation_audit_numeric.py',
) + REF.SOURCE_NAMES + STRESS.AUDIT_SOURCES))


def validate_operators(operators):
    """Accept the frozen family/h/T manifest, never a score-selected subset."""
    if not isinstance(operators, dict) or not 9 <= len(operators) <= 18:
        raise ValueError('expected nine to eighteen unique frozen operators')
    seen, families = set(), []
    for identifier, config in operators.items():
        if not isinstance(identifier, str) or not identifier:
            raise ValueError('operator identifier must be a nonempty string')
        if not isinstance(config, dict) or set(config) != {'family', 'h', 'temperature'}:
            raise ValueError('operator must have only family/h/temperature fields')
        family, h, temperature = config['family'], config['h'], config['temperature']
        if (family not in FAMILIES or isinstance(h, (bool, np.bool_))
                or isinstance(temperature, (bool, np.bool_))
                or h not in (.25, .5, 1., 2.) or temperature not in (0., 5., 10.)):
            raise ValueError('operator outside declared family/configuration domain')
        identity = (family, h, temperature)
        if identity in seen:
            raise ValueError('identical operators must be deduplicated before audit')
        seen.add(identity)
        families.append(family)
    if set(families) != set(FAMILIES):
        raise ValueError('all nine numerical families must be represented')
    if families.count('quartic_full') != 1 or families.count('quartic_isotropic_matched') != 1:
        raise ValueError('expected one primary and one matched isotropic operator')
    full = next(c for c in operators.values() if c['family'] == 'quartic_full')
    iso = next(c for c in operators.values() if c['family'] == 'quartic_isotropic_matched')
    if (full['h'], full['temperature']) != (iso['h'], iso['temperature']):
        raise ValueError('isotropic attribution must inherit primary h/T')


def reference_observations(inputs, state):
    """Reproject the ten allowed fields through the frozen independent R12 math.

    Every source pixel remains present, including behind-camera fallbacks. Both
    depth fields are scaled for provenance; only query geometry enters the PE.
    Zero always uses reprojection, with the old cache-consistency checks.
    """
    if set(inputs) != set(FIELDS) or state not in STATES:
        raise ValueError('expected ten forward fields and a declared state')
    if (np.shape(inputs['q_points3']) != (27648, 3)
            or np.shape(inputs['q_pixel_cov3']) != (27648, 3, 3)
            or np.shape(inputs['q_to_k_transform']) != (4, 4)
            or np.shape(inputs['k_K']) != (3, 3)
            or np.shape(inputs['q_predicted_depth']) != (144, 192)
            or np.shape(inputs['k_predicted_depth']) != (144, 192)):
        raise ValueError('wrong forward geometry/depth population')
    # Validates cached field/features shapes without passing depth to support
    # references. The output used below comes from independent reprojection.
    REF.forward_inputs({name: inputs[name] for name in REF.FIELDS})
    raw, provenance = STRESS.perturb_reference(inputs, state)
    raw = REF.forward_inputs(raw)
    factor = {'depth09': .9, 'depth11': 1.1}.get(state, 1.)
    depth = np.nan_to_num(np.asarray(inputs['q_predicted_depth'], np.float64),
                          nan=10., posinf=20., neginf=.1).clip(.1, 20.)
    if not np.allclose(np.asarray(inputs['q_points3'])[:, 2], depth.ravel(), atol=1e-12, rtol=1e-12):
        raise ValueError('3D query points disagree with clipped predicted depth')
    diagnostics = dict(state=state, depth_factor=factor,
        transform=provenance['transform'].tolist(),
        effective_depth_sha256=provenance['effective'],
        projected_xy_sha256=STRESS.array_sha(raw['q_to_k_xy']),
        projected_cov_sha256=STRESS.array_sha(raw['q_to_k_pixel_cov']),
        predicted_in_front_count=provenance['front'],
        key_depth_used_by_positional_operator=False,
        depth_policy='scale both clipped RGB depth fields; no post-scaling reclip')
    if state == 'zero':
        for field, atol in (('q_to_k_xy', 1e-12), ('q_to_k_pixel_cov', 1e-10)):
            if not np.allclose(raw[field], inputs[field], atol=atol, rtol=1e-12):
                raise ValueError('zero reprojection differs from cached geometry: ' + field)
        diagnostics.update(
            zero_xy_max_error=float(np.max(np.abs(raw['q_to_k_xy'] - inputs['q_to_k_xy']))),
            zero_cov_max_error=float(np.max(np.abs(raw['q_to_k_pixel_cov'] - inputs['q_to_k_pixel_cov']))),
            zero_float32_xy_equal=bool(np.array_equal(np.asarray(raw['q_to_k_xy'], np.float32),
                                                    np.asarray(inputs['q_to_k_xy'], np.float32))),
            zero_float32_cov_equal=bool(np.array_equal(np.asarray(raw['q_to_k_pixel_cov'], np.float32),
                                                     np.asarray(inputs['q_to_k_pixel_cov'], np.float32))))
    return raw, diagnostics


def quartic_covariance(query, family):
    """Independent full-moment trace for the production trace-only comparison."""
    covariance = np.asarray(query['cov'], np.float64)
    if family == 'quartic_full':
        return covariance
    if family == 'quartic_point':
        return np.zeros_like(covariance)
    if family == 'quartic_isotropic_matched':
        return np.trace(covariance, axis1=-2, axis2=-1)[:, None, None] * np.eye(2) / 2
    raise ValueError('not a quartic family')


def audit_pair(inputs, parts, sample_loader, operators):
    """Check one fixed first-pair sample without loading a path or any labels.

    ``sample_loader(scenario, condition, operator_id)`` returns the complete
    native float64 array for that sample. The caller verifies artifact hashes.
    ``operators`` is roles['operators'], mapping IDs to {family,h,temperature}.
    All 68*U matrices are checked; numerical failures remain in the returned
    report. Structural corruption is an exception, never a reduced population.
    """
    validate_operators(operators)
    if not callable(sample_loader):
        raise ValueError('sample_loader must return the retained native arrays')
    checks, projection_records = [], []
    for state in STATES:
        observations, projection = reference_observations(inputs, state)
        projection_records.append(projection)
        queries, keys, indices, schema = REF.reference_banks(observations, parts)
        cache = {}
        for scenario, actual_state, q, k in CONDITIONS:
            if actual_state != state:
                continue
            condition = q + '_' + k
            bounds = (schema[k + '_bounds'] - .5) / 8
            content = queries[q]['full']['features'] @ keys[k]['features'].T
            native_shape = (len(schema[q + '_bounds']), len(bounds))
            for identifier, config in operators.items():
                family, h, temperature = config['family'], config['h'], config['temperature']
                probability = np.asarray(sample_loader(scenario, condition, identifier))
                if (probability.shape != native_shape or probability.dtype != np.float64
                        or not np.isfinite(probability).all() or np.any(probability < 0)
                        or np.max(np.abs(probability.sum(1) - 1)) > 1e-12):
                    raise ValueError('invalid complete native sample: ' + str((scenario, condition, identifier)))
                actual = probability[indices[q]]
                del probability
                support_name = {'gaussian_full4': 'full4', 'gaussian_quad4': 'quad4',
                                'gaussian_quad16': 'quad16'}.get(family, 'full')
                query = queries[q][support_name]
                cache_key = q, k, family, h
                if cache_key not in cache:
                    if family.startswith('quartic_'):
                        covariance = quartic_covariance(query, family)
                        cache[cache_key] = tuple(REF.QREF.independent_mass(
                            query['mean'], covariance, bounds, h, order) for order in REFERENCE_ORDERS)
                    else:
                        mass = REF.gaussian_reference_mass(query, bounds, family, h)
                        cache[cache_key] = mass, mass
                raw64, raw96 = cache[cache_key]
                reference = REF.probabilities(raw96, content, temperature, bounds)
                reference64 = (REF.probabilities(raw64, content, temperature, bounds)
                               if family.startswith('quartic_') else reference)
                result = REF.probability_check(actual, reference, reference64, raw96, bounds, indices[q])
                # Thresholds are explicitly asserted to prevent an inherited
                # helper from silently changing this newly declared contract.
                passed = (result['max_probability_error'] <= PROBABILITY_TOLERANCE
                          and (result['reference_convergence_error'] is None
                               or result['reference_convergence_error'] <= REFERENCE_TOLERANCE))
                if passed != result['passed']:
                    raise ValueError('inherited numerical threshold changed')
                checks.append(dict(operator=identifier, config=dict(config), family=family,
                    h=h, temperature=temperature, scenario=scenario, state=state,
                    query=q, key=k, condition=condition, indices=indices[q].tolist(),
                    query_count=native_shape[0], key_count=native_shape[1],
                    reference_orders=list(REFERENCE_ORDERS) if family.startswith('quartic_') else [64],
                    **result))
    identities = [(c['scenario'], c['condition'], c['operator']) for c in checks]
    expected = {(scenario, q + '_' + k, op) for scenario, _, q, k in CONDITIONS for op in operators}
    if len(checks) != 68 * len(operators) or len(set(identities)) != len(checks) or set(identities) != expected:
        raise ValueError('independent numerical sample lost declared conditions/operators')
    return dict(passed=all(c['passed'] for c in checks), checks=checks,
        matrix_count=len(checks), query_rows=8 * len(checks), condition_count=68,
        unique_operator_count=len(operators), projection_records=projection_records,
        quartic_matrix_count=sum(c['family'].startswith('quartic_') for c in checks),
        gaussian_matrix_count=sum(c['family'].startswith('gaussian_') for c in checks),
        max_probability_error=max(c['max_probability_error'] for c in checks),
        max_reference_convergence_error=max(c['reference_convergence_error'] for c in checks
                                           if c['reference_convergence_error'] is not None),
        numerical_failure_count=sum(not c['passed'] for c in checks),
        argmax_change_count=sum(c['argmax_change_count'] for c in checks),
        reference_tiny_fov_rows=sum(c['reference_tiny_fov_rows'] for c in checks),
        labels_decoded=False, geometry_prepared_independently=True,
        configuration_selection=False, scientific_gate_passed=False)
