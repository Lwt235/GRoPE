"""Synthetic-only checks for the unrun R17 validation operator draft."""
import numpy as np
import pytest
import torch

from models import round17_validation_pe as P
from models import round15_validation_pe as OLD
from models import round17_trace_pe as TRACE
from models import round12_stress_pe as LEGACY
from experiments import round17_regional as REGIONAL


@pytest.fixture(scope='module', autouse=True)
def four_threads():
    original = torch.get_num_threads()
    torch.set_num_threads(4)
    yield
    torch.set_num_threads(original)


@pytest.fixture(scope='module')
def synthetic():
    yy, xx = np.mgrid[:48, :48]
    xy = np.stack((.8*xx+.25*yy+.3*np.sin(yy/4), .2*xx+1.1*yy), -1).astype(np.float32)
    pixel_cov = np.broadcast_to(np.array([[.12, .03], [.03, .23]], np.float32), (48, 48, 2, 2)).copy()
    rng = np.random.default_rng(1701)
    inputs = dict(q_to_k_xy=xy.reshape(-1, 2), q_to_k_pixel_cov=pixel_cov.reshape(-1, 2, 2),
        q_features=rng.normal(size=(36, 6)).astype(np.float32),
        k_features=rng.normal(size=(36, 6)).astype(np.float32))
    observations = {n: torch.from_numpy(v) for n, v in inputs.items()}
    parts = {}
    for name, side in (('g8', 8), ('g16', 16), ('mixed', 16), ('g24', 24)):
        boxes = []
        for y in range(0, 48, side):
            for x in range(0, 48, side):
                mode = (x//16+2*(y//16)) % 4 if name == 'mixed' else 0
                if mode == 1:
                    boxes.extend([[x, y, x+8, y+16], [x+8, y, x+16, y+16]])
                elif mode == 2:
                    boxes.extend([[x, y, x+16, y+8], [x, y+8, x+16, y+16]])
                elif mode == 3:
                    boxes.extend([[a, b, a+8, b+8] for b in (y, y+8) for a in (x, x+8)])
                else:
                    boxes.append([x, y, x+side, y+side])
        bounds = np.asarray(boxes, np.int64)
        owner = np.empty((48, 48), np.int64)
        for index, (x0, y0, x1, y1) in enumerate(boxes):
            owner[y0:y1, x0:x1] = index
        size = bounds[:, 2:]-bounds[:, :2]
        mean = (bounds[:, :2]+bounds[:, 2:]-1)/2
        cov = np.zeros((len(bounds), 2, 2))
        cov[:, 0, 0], cov[:, 1, 1] = size[:, 0]**2/768, size[:, 1]**2/768
        children, child_owner = LEGACY.split_cell_bounds(bounds)
        parent = (mean[:, 1].astype(int)//16)*3+mean[:, 0].astype(int)//16
        if name == 'g24':
            parent[:] = -1
        parts[name] = dict(bounds=bounds, owner=owner.ravel(), atoms=owner[4::8, 4::8].ravel(),
            parent=parent, mean=mean/8, cov=cov, mass=size.prod(-1),
            cell_bounds=children, cell_owner=child_owner)
    return inputs, observations, LEGACY.partition_tensors(parts, 'cpu')


def equal_query(left, right):
    assert set(left) == set(right)
    for name, value in left.items():
        if isinstance(value, torch.Tensor):
            assert value.dtype == right[name].dtype and torch.equal(value, right[name]), name
        else:
            assert value == right[name]


def test_each_trace_role_prepares_full_covariance_and_keeps_original_cross_terms(synthetic, monkeypatch):
    _, observations, parts = synthetic
    calls = []
    original = OLD.prepare_support
    def spy(*args, **kwargs):
        calls.append(kwargs['include_full_moments'])
        return original(*args, **kwargs)
    monkeypatch.setattr(OLD, 'prepare_support', spy)
    for part in parts.values():
        full = P.prepare_query(observations, part, 'trace_full')
        iso = P.prepare_query(observations, part, 'trace_isotropic_matched')
        assert calls[-2:] == [True, True]
        assert full['tau_cov'].dtype == iso['tau_cov'].dtype == torch.float32
        assert torch.equal(full['cov'], full['tau_cov'])
        assert torch.equal(full['tau_cov'], iso['tau_cov'])
        assert full['tau_cov'].data_ptr() != iso['tau_cov'].data_ptr()
        assert bool((iso['tau_cov'][:, 0, 1] != 0).all())
        variance = full['cov'].double().diagonal(dim1=-2, dim2=-1).sum(-1)/2
        assert iso['cov'].dtype == torch.float64
        assert torch.equal(iso['cov'], variance[:, None, None]*torch.eye(2, dtype=torch.float64))
        for name in ('mean', 'mass', 'features', 'parent'):
            assert torch.equal(full[name], iso[name])


@pytest.mark.parametrize('qname,kname', [('g8', 'g16'), ('g16', 'mixed'), ('mixed', 'g24'), ('g24', 'g8')])
def test_trace_full_is_exact_frozen_stage_b_path_on_same_synthetic_inputs(synthetic, qname, kname):
    inputs, observations, parts = synthetic
    queries, keys = REGIONAL.prepare_pair(inputs, parts)
    query = P.prepare_query(observations, parts[qname], 'trace_full')
    key = P.prepare_key(observations, parts[kname], 'trace_full')
    for name in ('mean', 'cov', 'features', 'mass', 'parent'):
        assert torch.equal(query[name], queries[qname][name])
    for name in key:
        assert torch.equal(key[name], keys[kname][name])
    with torch.no_grad():
        for _, index, expected, expected_diagnostic in REGIONAL.condition_predictions(queries[qname], keys[kname]):
            h, temperature = (.25, .5, 1., 2.)[index//3], (0., 5., 10.)[index%3]
            actual, diagnostic = P.predict(query, key, 'trace_full', h, temperature)
            assert np.array_equal(actual.numpy(), expected)
            assert {n: diagnostic[n] for n in expected_diagnostic} == expected_diagnostic
            assert diagnostic['all_finite'] and actual.dtype == torch.float64
            assert diagnostic['max_normalization_error'] == float((actual.sum(-1)-1).abs().max())


def test_isotropic_geometry_and_tau_use_distinct_exact_covariances(synthetic, monkeypatch):
    _, observations, parts = synthetic
    query = P.prepare_query(observations, parts['g16'], 'trace_isotropic_matched')
    key = P.prepare_key(observations, parts['g24'], 'trace_isotropic_matched')
    full = P.prepare_query(observations, parts['g16'], 'trace_full')
    original = TRACE.probabilities
    recorded = []
    def spy(geometry, content, covariance, h, temperature):
        recorded.append(covariance)
        return original(geometry, content, covariance, h, temperature)
    monkeypatch.setattr(TRACE, 'probabilities', spy)
    for h in (.25, .5, 1., 2.):
        raw = LEGACY.tiled_cell_mass(query['mean'], query['cov'], key, h)
        expected_geometry = (raw+1.e-30*(key['mass'].double()/64)[None]).log()
        actual_geometry, _ = P.geometry(query, key, 'trace_isotropic_matched', h)
        assert torch.equal(actual_geometry, expected_geometry)
        for temperature in (0., 5., 10.):
            full_tau = TRACE.trace_temperature(full['tau_cov'], h, temperature)
            iso_tau = TRACE.trace_temperature(query['tau_cov'], h, temperature)
            assert torch.equal(full_tau, iso_tau)
            content = query['features']@key['features'].T
            expected = original(expected_geometry, content, full['tau_cov'], h, temperature)
            actual, _ = P.predict(query, key, 'trace_isotropic_matched', h, temperature)
            assert recorded[-1] is query['tau_cov']
            assert torch.equal(actual, expected)


@pytest.mark.parametrize('family', P.CONTROL_FAMILIES)
def test_all_seven_historical_controls_and_constant_temperature_path_are_unchanged(synthetic, family):
    _, observations, parts = synthetic
    query = P.prepare_query(observations, parts['g16'], family)
    old_query = OLD.prepare_query(observations, parts['g16'], family)
    equal_query(query, old_query)
    key = P.prepare_key(observations, parts['g24'], family)
    old_key = OLD.prepare_key(observations, parts['g24'], family)
    equal_query(key, old_key)
    for h, temperature in ((.25, 0.), (.5, 5.), (2., 10.)):
        actual, diagnostics = P.predict(query, key, family, h, temperature)
        expected, expected_diagnostics = OLD.predict(old_query, old_key, family, h, temperature)
        assert torch.equal(actual, expected) and diagnostics == expected_diagnostics
    assert 'matched_constant_temperature' not in P.FAMILIES


def test_point_covariance_and_temperature_zero_recover_original_gaussian_probability(synthetic):
    _, observations, parts = synthetic
    query = P.prepare_query(observations, parts['g16'], 'trace_full')
    key = P.prepare_key(observations, parts['g24'], 'trace_full')
    old_query = OLD.prepare_query(observations, parts['g16'], 'gaussian_full1')
    for h in (.25, .5, 1., 2.):
        actual, _ = P.predict(query, key, 'trace_full', h, 0.)
        expected, _ = OLD.predict(old_query, key, 'gaussian_full1', h, 0.)
        assert torch.equal(actual, expected)
    zero_inputs = dict(observations, q_to_k_xy=torch.full_like(observations['q_to_k_xy'], 12.),
                       q_to_k_pixel_cov=torch.zeros_like(observations['q_to_k_pixel_cov']))
    point = OLD.prepare_query(zero_inputs, parts['g16'], 'gaussian_point')
    for family in P.TRACE_FAMILIES:
        new = P.prepare_query(zero_inputs, parts['g16'], family)
        assert not bool(new['tau_cov'].any())
        for temperature in (0., 5., 10.):
            actual, _ = P.predict(new, key, family, .25, temperature)
            expected, _ = OLD.predict(point, key, 'gaussian_point', .25, temperature)
            assert torch.equal(actual, expected)


def test_native_24_gaussian_children_and_role_only_key_bounds(synthetic):
    _, observations, parts = synthetic
    for family in P.TRACE_FAMILIES:
        key = P.prepare_key(observations, parts['g24'], family)
        assert set(key) == {'mean', 'mass', 'cell_bounds', 'cell_owner', 'features'}
        assert key['cell_bounds'].shape == (16, 4) and key['mass'].shape == (4,)
        assert torch.all(key['cell_bounds'][:, 2:]-key['cell_bounds'][:, :2] == 1.5)
        assert torch.equal(key['cell_owner'], torch.arange(4).repeat_interleave(4))
        equal_query(key, OLD.prepare_key(observations, parts['g24'], 'gaussian_full1'))
    equal_query(P.prepare_key(observations, parts['g24']), OLD.prepare_key(observations, parts['g24']))


def test_floor_once_and_raw_tail_population_retained(synthetic, monkeypatch):
    _, observations, parts = synthetic
    query = P.prepare_query(observations, parts['g24'], 'trace_full')
    key = P.prepare_key(observations, parts['mixed'], 'trace_full')
    def zero_integral(mean, covariance, boxes, **kwargs):
        assert kwargs['quadrature_order'] == 32
        return torch.zeros((len(mean), len(boxes)), dtype=torch.float64)
    monkeypatch.setattr(OLD, 'gaussian_cell_mass_fast', zero_integral)
    probability, diagnostics = P.predict(query, key, 'trace_full', .5, 0.)
    geometry, _ = P.geometry(query, key, 'trace_full', .5)
    expected_geometry = (1.e-30*key['mass'].double()/64).log()[None].expand_as(geometry)
    assert torch.equal(geometry, expected_geometry)
    assert torch.equal(probability, expected_geometry.softmax(-1))
    assert diagnostics['raw_zero_fov_rows'] == diagnostics['raw_tiny_fov_rows'] == len(query['mean'])


def test_original_cross_covariance_psd_checks_execute_even_for_iso_and_t0(synthetic, monkeypatch):
    _, observations, parts = synthetic
    query = P.prepare_query(observations, parts['g24'], 'trace_isotropic_matched')
    key = P.prepare_key(observations, parts['g24'], 'trace_isotropic_matched')
    bad = dict(query, tau_cov=query['tau_cov'].clone())
    bad['tau_cov'][:, 0, 1] = 1.e6
    bad['tau_cov'][:, 1, 0] = 1.e6
    # Isotropic integration remains valid; the original full covariance must
    # still fail the independently timed temperature validation, including T0.
    P.geometry(bad, key, 'trace_isotropic_matched', .5)
    for temperature in (0., 5.):
        with pytest.raises(ValueError, match='positive semidefinite'):
            P.predict(bad, key, 'trace_isotropic_matched', .5, temperature)
    monkeypatch.setattr(torch, 'set_num_threads', lambda *_: pytest.fail('model must not change runtime'))
    P.predict(query, key, 'trace_isotropic_matched', .5, 0.)
    poisoned = dict(query, features=torch.full_like(query['features'], float('nan')))
    with pytest.raises(ValueError, match='content must be finite'):
        P.predict(poisoned, key, 'trace_isotropic_matched', .5, 0.)


def test_trace_gradient_flows_through_geometry_original_tau_covariance_and_content(synthetic):
    _, observations, parts = synthetic
    inputs = {n: value.clone().requires_grad_() for n, value in observations.items()}
    query = P.prepare_query(inputs, parts['g24'], 'trace_isotropic_matched')
    key = P.prepare_key(inputs, parts['g24'], 'trace_isotropic_matched')
    probability, _ = P.predict(query, key, 'trace_isotropic_matched', .5, 5.)
    probability[:, 0].sum().backward()
    for field in P.OBSERVATION_FIELDS:
        assert inputs[field].grad is not None and torch.isfinite(inputs[field].grad).all()
        assert bool((inputs[field].grad != 0).any())


def test_schema_guards_exclude_gt_unrequested_families_and_mismatched_queries(synthetic):
    _, observations, parts = synthetic
    for family in P.TRACE_FAMILIES:
        with pytest.raises(ValueError, match='four predicted'):
            P.prepare_query(dict(observations, gt_visible=torch.ones(1)), parts['g8'], family)
    for family in ('quartic_full', 'quartic_isotropic_matched', 'matched_constant_temperature'):
        with pytest.raises(ValueError, match='undeclared'):
            P.prepare_query(observations, parts['g8'], family)
    query = P.prepare_query(observations, parts['g24'], 'trace_full')
    key = P.prepare_key(observations, parts['g24'], 'trace_full')
    with pytest.raises(ValueError, match='different family'):
        P.geometry(query, key, 'trace_isotropic_matched', .5)
