"""Independent Stage B probability checks; no regional labels or scores read.

Only literal NumPy support references and the already frozen independent
Gaussian/quartic integrators are called. The selection argument controls the
predeclared extra Gaussian checks, never the fixed quartic or Gaussian sample.
No producer forward/preparation function is called. QREF's unused production
imports are included in SOURCE_NAMES through its complete frozen source list.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
from scipy.special import softmax

from experiments import round10_partition_audit as AREF
from experiments import round10_cell_result_audit as BREF
from experiments import round11_result_audit as FOURREF
from experiments import round15_train_numerical24 as QREF

ROOT = Path(__file__).resolve().parents[1]
PARTITIONS = ('g8', 'g16', 'mixed', 'g24')
CONDITIONS = tuple((q, k) for q in PARTITIONS[:3] for k in PARTITIONS[:3]) + (
    ('g8', 'g24'), ('g24', 'g8'), ('g16', 'g24'), ('g24', 'g16'), ('g24', 'g24'))
FAMILIES = ('quartic_full', 'quartic_point', 'gaussian_point', 'gaussian_fixed',
            'gaussian_full1', 'gaussian_full4', 'gaussian_quad4', 'gaussian_quad16')
GAUSSIAN_METHODS = dict(gaussian_point='point', gaussian_fixed='fixed',
                       gaussian_full1='full', gaussian_full4='full4',
                       gaussian_quad4='quad4', gaussian_quad16='quad16')
GRID = tuple((h, t) for h in (.25, .5, 1., 2.) for t in (0., 5., 10.))
FIELDS = ('q_to_k_xy', 'q_to_k_pixel_cov', 'q_features', 'k_features')
REFERENCE_ORDERS = (64, 96)
PROBABILITY_TOLERANCE = 1e-4
REFERENCE_TOLERANCE = 1e-8
FLOOR = 1e-30
SOURCE_NAMES = tuple(dict.fromkeys((
    'experiments/round15_regional_audit_numeric.py',
    'tests/test_round15_regional_audit_numeric.py',
    'experiments/round10_partition_audit.py',
    'experiments/round10_cell_result_audit.py',
    'experiments/round11_result_audit.py',
    'experiments/round15_train_numerical24.py',
) + QREF.SOURCE_NAMES))


def absolute(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT/path


def selected_indices(count):
    if count < 8:
        raise ValueError('the declared sample needs at least eight query cells')
    return np.linspace(0, count-1, 8, dtype=np.int64)


def reference_partition_schema(parts):
    """Convert native rectangles to R10 references using literal pixel units.

    Stored Gaussian integration children are deliberately not used. In
    particular g24 key integration covers its original 24px rectangle.
    """
    if set(parts) != set(PARTITIONS):
        raise ValueError('expected all four native partitions')
    output = {}
    for name in PARTITIONS:
        part = parts[name]
        bounds = np.asarray(part['bounds'])
        if (bounds.ndim != 2 or bounds.shape[1] != 4 or len(bounds) < 8
                or not np.issubdtype(bounds.dtype, np.integer)):
            raise ValueError('integer native rectangle bounds required')
        bounds = bounds.astype(np.int64, copy=False)
        size = bounds[:, 2:]-bounds[:, :2]
        if (np.any(bounds % 8) or np.any(bounds[:, :2] < 0)
                or np.any(bounds[:, 2:] > [192, 144])
                or np.any(size <= 0) or np.any(size > 24)):
            raise ValueError('invalid native rectangle extent')
        owners = np.full((144, 192), -1, np.int64)
        for index, (x0, y0, x1, y1) in enumerate(bounds):
            if np.any(owners[y0:y1, x0:x1] != -1):
                raise ValueError('overlapping native rectangles')
            owners[y0:y1, x0:x1] = index
        if np.any(owners < 0):
            raise ValueError('incomplete native pixel field')
        centers = (bounds[:, :2]+bounds[:, 2:]-1)/2
        area = size.prod(-1).astype(np.float64)
        covariance = np.zeros((len(bounds), 2, 2), np.float64)
        covariance[:, 0, 0] = size[:, 0]**2/12
        covariance[:, 1, 1] = size[:, 1]**2/12
        # These checks prevent the reference8/physical-pixel conversion from
        # silently changing, while the reference values come from bounds.
        for field, expected in (('mean', centers/8), ('cov', covariance/64), ('mass', area)):
            value = np.asarray(part[field])
            if value.shape != expected.shape or not np.allclose(value, expected, rtol=0, atol=1e-6):
                raise ValueError('native geometry units or area changed: '+field)
        parent = np.asarray(part['parent'])
        if parent.shape != (len(bounds),):
            raise ValueError('invalid parent shape')
        if name == 'g24' and np.any(parent != -1):
            raise ValueError('q24 has no common16 parent')
        output.update({name+'_bounds': bounds, name+'_centers': centers,
                       name+'_covariance': covariance, name+'_area': area,
                       name+'_pixel_index': owners, name+'_parent_g16': parent})
    return output


def forward_inputs(inputs):
    if set(inputs) != set(FIELDS):
        raise ValueError('only the four forward observation fields are accepted')
    output = {k: np.asarray(inputs[k]) for k in FIELDS}
    if (output['q_to_k_xy'].shape != (144*192, 2)
            or output['q_to_k_pixel_cov'].shape != (144*192, 2, 2)):
        raise ValueError('unexpected dense forward geometry shape')
    qf, kf = output['q_features'], output['k_features']
    if qf.ndim != 2 or qf.shape != kf.shape or qf.shape[0] != 432:
        raise ValueError('expected corresponding base8 feature banks')
    if any(not np.isfinite(value).all() for value in output.values()):
        raise ValueError('nonfinite forward observations')
    return output


def slice_support(support, indices):
    return {name: np.asarray(value)[indices] for name, value in support.items()}


def reference_banks(inputs, parts):
    inputs = forward_inputs(inputs)
    schema = reference_partition_schema(parts)
    queries, keys, indices = {}, {}, {}
    for name in PARTITIONS:
        indices[name] = selected_indices(len(schema[name+'_bounds']))
        queries[name] = {
            'full': slice_support(AREF.reference_support(inputs, schema, name), indices[name]),
            'quad4': slice_support(AREF.reference_support(inputs, schema, name, 4), indices[name]),
            'quad16': slice_support(AREF.reference_support(inputs, schema, name, 16), indices[name]),
            'full4': slice_support(FOURREF.reference_four_support(inputs, schema, name), indices[name]),
        }
        keys[name] = AREF.reference_key(inputs, schema, name)
    return queries, keys, indices, schema


def planned_configurations(selection):
    """The six fixed Gaussian configurations cannot be replaced by winners."""
    controls = selection['controls']
    if set(controls) != set(FAMILIES[1:]):
        raise ValueError('expected all seven Stage B control selections')
    planned = {family: tuple(range(12)) for family in FAMILIES[:2]}
    for family in FAMILIES[1:]:
        chosen = controls[family]
        index = chosen['grid_index']
        if type(index) is not int or index not in range(12):
            raise ValueError('selected grid index outside fixed grid')
        h, temperature = GRID[index]
        if chosen['family'] != family or chosen['config'] != dict(h=h, temperature=temperature):
            raise ValueError('selection family/configuration identity mismatch')
        if family in GAUSSIAN_METHODS:
            planned[family] = (4,) if index == 4 else (4, index)
    return planned


def gaussian_reference_mass(query, bounds, family, h):
    """Frozen independent SciPy Gaussian integration, over original keys."""
    method = GAUSSIAN_METHODS[family]
    if method in ('full4', 'quad4', 'quad16'):
        children = BREF.reference_mass(query['nodes'], query['node_cov'], bounds, h)
        return np.sum(children*query['node_weights'][:, :, None], axis=1)
    covariance = BREF.transformed_covariance(query, method)
    return BREF.reference_mass(query['mean'], covariance, bounds, h)


def probabilities(raw, content, temperature, bounds):
    raw = np.asarray(raw, np.float64)
    if raw.shape != content.shape or not np.isfinite(raw).all() or np.any(raw < 0):
        raise ValueError('invalid independent raw mass')
    area = np.prod(bounds[:, 2:]-bounds[:, :2], axis=1)
    return softmax(np.log(raw+FLOOR*area[None])+temperature*content, axis=-1)


def probability_check(actual, reference, reference64, raw, bounds, indices):
    for value in (actual, reference, reference64):
        if (value.shape != reference.shape or value.dtype != np.float64
                or not np.isfinite(value).all() or np.any(value < 0)
                or np.max(np.abs(value.sum(1)-1)) > 1e-12):
            raise ValueError('invalid saved or independent probability')
    error = float(np.max(np.abs(actual-reference)))
    convergence = float(np.max(np.abs(reference64-reference))) if reference64 is not reference else None
    winner = actual.argmax(1); ref_winner = reference.argmax(1)
    disagreements = []
    for row in np.flatnonzero(winner != ref_winner):
        ordered = np.sort(reference[row])
        disagreements.append(dict(query_index=int(indices[row]), actual_argmax=int(winner[row]),
            reference_argmax=int(ref_winner[row]), reference_top_gap=float(ordered[-1]-ordered[-2]),
            reference_winner_advantage=float(reference[row, ref_winner[row]]-reference[row, winner[row]]),
            actual_winner_advantage=float(actual[row, winner[row]]-actual[row, ref_winner[row]])))
    fov = raw.sum(1)
    return dict(max_probability_error=error, reference_convergence_error=convergence,
        passed=error <= PROBABILITY_TOLERANCE and (convergence is None or convergence <= REFERENCE_TOLERANCE),
        argmax_change_count=len(disagreements), argmax_changes=disagreements,
        reference_raw_fov_mass=fov.tolist(), reference_raw_fov_min=float(fov.min()),
        reference_raw_fov_max=float(fov.max()), reference_zero_fov_rows=int((fov == 0).sum()),
        reference_tiny_fov_rows=int((fov < 1e-20).sum()),
        reference_zero_fov_indices=indices[fov == 0].tolist(),
        reference_tiny_fov_indices=indices[fov < 1e-20].tolist(),
        reference_floor_fov_mass=float(FLOOR*np.prod(bounds[:, 2:]-bounds[:, :2], axis=1).sum()),
        reference_raw_zero_cells=int((raw == 0).sum()))


def numeric_pair(inputs, parts, samples, selection):
    """Audit the fixed first-pair sample; return all checks, including failures.

    `parts` is the nested native partition mapping, `samples` maps each of the
    fourteen `query_key` strings to an npz containing all native query rows,
    and `selection` is the unmodified producer summary's selection object.
    No label or score data are loaded; selection J values are never examined.
    """
    if set(samples) != {q+'_'+k for q, k in CONDITIONS}:
        raise ValueError('expected all fourteen probability sample conditions')
    planned = planned_configurations(selection)
    queries, keys, indices, schema = reference_banks(inputs, parts)
    fields = {f'{family}__c{i:02d}' for family in FAMILIES for i in range(12)}
    checks = []
    for q, k in CONDITIONS:
        bounds = (schema[k+'_bounds']-.5)/8
        content = queries[q]['full']['features']@keys[k]['features'].T
        cache = {}
        with np.load(absolute(samples[q+'_'+k]), allow_pickle=False) as archive:
            if set(archive.files) != fields:
                raise ValueError('incomplete native probability sample archive')
            for family in FAMILIES:
                support_name = ('full4' if family == 'gaussian_full4' else
                                'quad4' if family == 'gaussian_quad4' else
                                'quad16' if family == 'gaussian_quad16' else 'full')
                query = queries[q][support_name]
                for grid_index in planned[family]:
                    h, temperature = GRID[grid_index]
                    field = f'{family}__c{grid_index:02d}'
                    saved = archive[field]
                    if saved.shape != (len(schema[q+'_bounds']), len(bounds)):
                        raise ValueError('sample array has wrong native shape')
                    actual = saved[indices[q]]
                    del saved
                    key = family, h
                    if key not in cache:
                        if family.startswith('quartic_'):
                            covariance = query['cov'] if family == 'quartic_full' else np.zeros_like(query['cov'])
                            cache[key] = tuple(QREF.independent_mass(query['mean'], covariance, bounds, h, order)
                                               for order in REFERENCE_ORDERS)
                        else:
                            raw = gaussian_reference_mass(query, bounds, family, h)
                            cache[key] = raw, raw
                    raw64, raw96 = cache[key]
                    reference = probabilities(raw96, content, temperature, bounds)
                    reference64 = (probabilities(raw64, content, temperature, bounds)
                                   if family.startswith('quartic_') else reference)
                    check = probability_check(actual, reference, reference64, raw96, bounds, indices[q])
                    checks.append(dict(family=family, grid_index=grid_index, h=h, temperature=temperature,
                        query=q, key=k, condition=q+'_'+k, indices=indices[q].tolist(),
                        query_count=len(schema[q+'_bounds']), key_count=len(bounds), field=field,
                        reference_orders=list(REFERENCE_ORDERS) if family.startswith('quartic_') else [64],
                        **check))
    expected = 14*sum(len(values) for values in planned.values())
    if len(checks) != expected:
        raise ValueError('incomplete independent numerical population')
    return dict(passed=all(row['passed'] for row in checks), checks=checks, matrix_count=len(checks),
        quartic_matrix_count=sum(row['family'].startswith('quartic_') for row in checks),
        gaussian_matrix_count=sum(row['family'].startswith('gaussian_') for row in checks),
        max_probability_error=max(row['max_probability_error'] for row in checks),
        max_reference_convergence_error=max(row['reference_convergence_error'] for row in checks
                                             if row['reference_convergence_error'] is not None),
        argmax_change_count=sum(row['argmax_change_count'] for row in checks),
        reference_tiny_fov_rows=sum(row['reference_tiny_fov_rows'] for row in checks),
        labels_decoded=False, geometry_prepared_independently=True, scientific_gate_passed=False)
