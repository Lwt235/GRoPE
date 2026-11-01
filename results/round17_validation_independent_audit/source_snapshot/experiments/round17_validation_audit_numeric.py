"""Independent R17 validation numerical checks, developed on synthetic data.

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
from experiments import round17_trace_numerical_audit as TRACE


STATES = ('zero', 'depth09', 'depth11', 'ry_plus1', 'ry_minus1', 'tx_plus01', 'tx_minus01')
PARTITIONS = ('g8', 'g16', 'mixed', 'g24')
ORIGINAL = tuple((q, k) for q in PARTITIONS[:3] for k in PARTITIONS[:3])
SCALE = (('g8', 'g24'), ('g24', 'g8'), ('g16', 'g24'), ('g24', 'g16'), ('g24', 'g24'))
CONDITIONS = tuple((state, state, q, k) for state in STATES for q, k in ORIGINAL) + tuple(
    ('scale', 'zero', q, k) for q, k in SCALE)
TRACE_FAMILIES = ('trace_full', 'trace_isotropic_matched')
FAMILIES = TRACE_FAMILIES + REF.FAMILIES[1:]
FIELDS = ('q_to_k_xy', 'q_to_k_pixel_cov', 'q_features', 'k_features',
          'q_points3', 'q_pixel_cov3', 'q_to_k_transform', 'k_K',
          'q_predicted_depth', 'k_predicted_depth')
REFERENCE_ORDERS = (64, 96)
PROBABILITY_TOLERANCE = 1e-4
SAME_MASS_TOLERANCE = 1e-12
REFERENCE_TOLERANCE = 1e-8
SOURCE_NAMES = tuple(dict.fromkeys((
    'experiments/round17_validation_audit_numeric.py',
    'tests/test_round17_validation_audit_numeric.py',
) + REF.SOURCE_NAMES + STRESS.AUDIT_SOURCES + TRACE.SOURCE_NAMES))


def validate_operators(operators):
    """Accept the frozen family/h/T manifest, never a score-selected subset."""
    if not isinstance(operators, dict) or not 9 <= len(operators) <= 19:
        raise ValueError('expected nine to nineteen unique frozen operators')
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
    if families.count('trace_full') != 1 or families.count('trace_isotropic_matched') != 1:
        raise ValueError('expected one primary and one matched isotropic operator')
    full = next(c for c in operators.values() if c['family'] == 'trace_full')
    iso = next(c for c in operators.values() if c['family'] == 'trace_isotropic_matched')
    if (full['h'], full['temperature']) != (iso['h'], iso['temperature']):
        raise ValueError('isotropic attribution must inherit primary h/T')
    if not any(c['family'] == 'gaussian_full1' and (c['h'], c['temperature']) ==
               (full['h'], full['temperature']) for c in operators.values()):
        raise ValueError('matched constant-temperature full1 control missing')


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


def trace_covariance(query, family):
    """Geometry may be isotropic; attenuation always uses original full C."""
    covariance = np.asarray(query['cov'], np.float64)
    if family == 'trace_full':
        return covariance
    if family == 'trace_isotropic_matched':
        return np.eye(2)*np.trace(covariance, axis1=-2, axis2=-1)[:, None, None]/2
    raise ValueError('not a trace family')


def _array(record, name, shape, dtype):
    value = np.asarray(record[name])
    if value.shape != shape or value.dtype != dtype or not np.isfinite(value).all():
        raise ValueError('invalid prepared sample ' + name)
    return value


def prepared_check(prepared, actual, indices, bounds, family, h, temperature):
    """Audit re-created actual intermediates; these were not stored at forward.

    The outer auditor first replays the complete native matrix and matches its
    committed hash. This callback supplies its eight selected rows. Independent
    NumPy arithmetic checks the float32 tau and product exactly; independent
    integration over this actual support separates integration from preparation.
    A second fully independent support/integration reference is checked by
    audit_pair. This function makes no independent-actual-preparation claim.
    """
    fields = {'indices', 'mean', 'cov', 'tau_cov', 'content', 'raw', 'tau32', 'bias32',
              'probability', 'full_probability_sha256', 'committed_probability_sha256',
              'recomputed_not_original_intermediates'}
    if set(prepared) != fields or family not in TRACE_FAMILIES:
        raise ValueError('wrong shared-preparation evidence schema')
    selected = np.asarray(prepared['indices'])
    if selected.dtype != np.int64 or not np.array_equal(selected, indices):
        raise ValueError('prepared query sample differs from fixed indices')
    digest = prepared['full_probability_sha256']
    if (not isinstance(digest, str) or len(digest) != 64
            or any(c not in '0123456789abcdef' for c in digest)
            or digest != prepared['committed_probability_sha256']
            or prepared['recomputed_not_original_intermediates'] is not True):
        raise ValueError('shared replay not bound to complete forward probability')
    q, k = actual.shape
    mean = _array(prepared, 'mean', (q, 2), np.float32)
    covariance = _array(prepared, 'cov', (q, 2, 2),
                        np.float64 if family == 'trace_isotropic_matched' else np.float32)
    tau_covariance = _array(prepared, 'tau_cov', (q, 2, 2), np.float32)
    content = _array(prepared, 'content', (q, k), np.float32)
    raw = _array(prepared, 'raw', (q, k), np.float64)
    tau32 = _array(prepared, 'tau32', (q,), np.float32)
    bias32 = _array(prepared, 'bias32', (q, k), np.float32)
    probability = _array(prepared, 'probability', (q, k), np.float64)
    if np.any(raw < 0) or not np.array_equal(probability, actual):
        raise ValueError('prepared probability differs from retained sample or raw mass is negative')
    expected_covariance = (tau_covariance if family == 'trace_full' else
        trace_covariance({'cov': tau_covariance}, family))
    if not np.array_equal(covariance, expected_covariance):
        raise ValueError('geometry covariance is not the declared transformation of original tau_cov')
    independent_tau = TRACE.independent_temperature(tau_covariance, h, temperature).astype(np.float32)
    independent_bias = (independent_tau[:, None]*content).astype(np.float32)
    same = TRACE.independent_probability(raw, content, tau_covariance, h, temperature, bounds)
    hybrid_raw = TRACE.independent_mass(mean, covariance, bounds, h, 96)
    hybrid = TRACE.independent_probability(hybrid_raw, content, tau_covariance, h, temperature, bounds)
    hybrid_result = REF.probability_check(actual, hybrid, hybrid, hybrid_raw, bounds, indices)
    exact_tau = bool(np.array_equal(tau32, independent_tau))
    exact_bias = bool(np.array_equal(bias32, independent_bias))
    same_error = float(np.max(np.abs(actual-same)))
    return dict(
        same_prepared_tau32_exact=exact_tau, same_prepared_bias32_exact=exact_bias,
        max_tau32_error=float(np.max(np.abs(tau32.astype(np.float64)-independent_tau))),
        max_bias32_error=float(np.max(np.abs(bias32.astype(np.float64)-independent_bias))),
        same_mass_probability_error=same_error,
        hybrid_max_probability_error=hybrid_result['max_probability_error'],
        hybrid_argmax_change_count=hybrid_result['argmax_change_count'],
        hybrid_argmax_changes=hybrid_result['argmax_changes'],
        hybrid_reference_raw_fov_mass=hybrid_result['reference_raw_fov_mass'],
        hybrid_reference_zero_fov_rows=hybrid_result['reference_zero_fov_rows'],
        hybrid_reference_tiny_fov_rows=hybrid_result['reference_tiny_fov_rows'],
        same_prepared_passed=exact_tau and exact_bias and same_error <= SAME_MASS_TOLERANCE,
        hybrid_passed=hybrid_result['max_probability_error'] <= PROBABILITY_TOLERANCE,
        intermediates_recomputed_from_locked_production=True,
        intermediates_saved_by_original_forward=False,
        actual_preparation_is_independent=False,
        full_probability_sha256=digest)


def audit_pair(inputs, parts, sample_loader, operators, *, prepared_loader):
    """Check one first VAL pair, every frozen operator and all 68 conditions.

    sample_loader(scenario,condition,operator) returns the full native float64
    probability. prepared_loader with the same arguments returns the selected
    eight trace-family rows and replay provenance defined by prepared_check.
    No path, target, score or selection is opened here. Numerical failures are
    retained, while structural corruption raises rather than reducing coverage.
    """
    validate_operators(operators)
    if not callable(sample_loader) or not callable(prepared_loader):
        raise ValueError('both complete probability and trace preparation callbacks required')
    checks, projection_records = [], []
    for state in STATES:
        observations, projection = reference_observations(inputs, state)
        projection_records.append(projection)
        queries, keys, indices, schema = REF.reference_banks(observations, parts)
        cache = {}
        for scenario, actual_state, q, k in CONDITIONS:
            if actual_state != state:
                continue
            condition = q+'_'+k
            bounds = (schema[k+'_bounds']-.5)/8
            content = queries[q]['full']['features'] @ keys[k]['features'].T
            native_shape = (len(schema[q+'_bounds']), len(bounds))
            tau_evidence = {}
            for identifier, config in operators.items():
                family, h, temperature = config['family'], config['h'], config['temperature']
                probability = np.asarray(sample_loader(scenario, condition, identifier))
                if (probability.shape != native_shape or probability.dtype != np.float64
                        or not np.isfinite(probability).all() or np.any(probability < 0)
                        or np.max(np.abs(probability.sum(1)-1)) > 1e-12):
                    raise ValueError('invalid complete native sample: '+str((scenario,condition,identifier)))
                actual = probability[indices[q]]
                del probability
                support_name = {'gaussian_full4':'full4', 'gaussian_quad4':'quad4',
                                'gaussian_quad16':'quad16'}.get(family, 'full')
                query = queries[q][support_name]
                cache_key = q,k,family,h
                if cache_key not in cache:
                    if family in TRACE_FAMILIES:
                        covariance = trace_covariance(query, family)
                        cache[cache_key] = tuple(TRACE.independent_mass(query['mean'], covariance,
                            bounds, h, order) for order in REFERENCE_ORDERS)
                    elif family == 'quartic_point':
                        cache[cache_key] = tuple(REF.QREF.independent_mass(query['mean'],
                            np.zeros_like(query['cov']), bounds, h, order) for order in REFERENCE_ORDERS)
                    else:
                        mass = REF.gaussian_reference_mass(query, bounds, family, h)
                        cache[cache_key] = mass,mass
                raw64,raw96 = cache[cache_key]
                prepared_result = {}
                if family in TRACE_FAMILIES:
                    reference = TRACE.independent_probability(raw96,content.astype(np.float32),query['cov'],h,temperature,bounds)
                    reference64 = TRACE.independent_probability(raw64,content.astype(np.float32),query['cov'],h,temperature,bounds)
                    prepared = prepared_loader(scenario,condition,identifier)
                    prepared_result = prepared_check(prepared,actual,indices[q],bounds,family,h,temperature)
                    tau_evidence[family] = {name:np.asarray(prepared[name]) for name in ('mean','tau_cov','tau32','content')}
                else:
                    reference = REF.probabilities(raw96,content,temperature,bounds)
                    reference64 = REF.probabilities(raw64,content,temperature,bounds) if family=='quartic_point' else reference
                result = REF.probability_check(actual,reference,reference64,raw96,bounds,indices[q])
                passed = (result['max_probability_error'] <= PROBABILITY_TOLERANCE
                    and (result['reference_convergence_error'] is None or result['reference_convergence_error'] <= REFERENCE_TOLERANCE))
                if passed != result['passed']:
                    raise ValueError('inherited reference threshold changed')
                if family in TRACE_FAMILIES:
                    passed = passed and prepared_result['same_prepared_passed'] and prepared_result['hybrid_passed']
                result['passed'] = bool(passed)
                checks.append(dict(operator=identifier,config=dict(config),family=family,h=h,temperature=temperature,
                    scenario=scenario,state=state,query=q,key=k,condition=condition,indices=indices[q].tolist(),
                    query_count=native_shape[0],key_count=native_shape[1],
                    reference_orders=list(REFERENCE_ORDERS) if family in TRACE_FAMILIES or family=='quartic_point' else [64],
                    **result,**prepared_result))
            if set(tau_evidence) != set(TRACE_FAMILIES):
                raise ValueError('missing matched tau evidence')
            full,iso=(tau_evidence[f] for f in TRACE_FAMILIES)
            if any(not np.array_equal(full[name],iso[name]) for name in full):
                raise ValueError('matched isotropic role changed original full support, content or tau')
    identities=[(c['scenario'],c['condition'],c['operator']) for c in checks]
    expected={(scenario,q+'_'+k,op) for scenario,_,q,k in CONDITIONS for op in operators}
    if len(checks)!=68*len(operators) or len(set(identities))!=len(checks) or set(identities)!=expected:
        raise ValueError('independent numerical sample lost declared population')
    trace=[c for c in checks if c['family'] in TRACE_FAMILIES]
    return dict(passed=all(c['passed'] for c in checks),checks=checks,matrix_count=len(checks),
        query_rows=8*len(checks),condition_count=68,unique_operator_count=len(operators),projection_records=projection_records,
        quartic_matrix_count=sum(c['family']=='quartic_point' for c in checks),
        gaussian_matrix_count=sum(c['family'].startswith('gaussian_') for c in checks),trace_matrix_count=len(trace),
        max_probability_error=max(c['max_probability_error'] for c in checks),
        max_reference_convergence_error=max(c['reference_convergence_error'] for c in checks if c['reference_convergence_error'] is not None),
        max_hybrid_probability_error=max(c['hybrid_max_probability_error'] for c in trace),
        max_same_mass_probability_error=max(c['same_mass_probability_error'] for c in trace),
        same_prepared_tau32_exact=all(c['same_prepared_tau32_exact'] for c in trace),
        same_prepared_bias32_exact=all(c['same_prepared_bias32_exact'] for c in trace),
        matched_original_tau_cov_and_tau32_exact=True,
        numerical_failure_count=sum(not c['passed'] for c in checks),
        argmax_change_count=sum(c['argmax_change_count'] for c in checks),
        hybrid_argmax_change_count=sum(c['hybrid_argmax_change_count'] for c in trace),
        reference_tiny_fov_rows=sum(c['reference_tiny_fov_rows'] for c in checks),
        labels_decoded=False,geometry_prepared_independently=True,
        actual_preparation_is_independent=False,actual_intermediates_recomputed=True,
        configuration_selection=False,scientific_gate_passed=False)
