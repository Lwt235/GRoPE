"""Synthetic-only R20 adapter checks; no dataset, cache or label is opened."""
import json

import numpy as np
import pytest
import torch

from models import round15_validation_pe as OLD
from models import round17_trace_pe as TRACE
from models import round17_validation_pe as R17
from models import round20_anchor_pe as P


@pytest.fixture(scope='module', autouse=True)
def four_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(4)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope='module')
def synthetic():
    yy, xx = np.mgrid[:48, :48]
    xy = np.stack((xx + .25 * yy, yy + .125 * xx), -1).astype(np.float32)
    pixel_cov = np.broadcast_to(np.array([[.125, .03125], [.03125, .25]], np.float32),
                                (48, 48, 2, 2)).copy()
    rng = np.random.default_rng(2001)
    observations = dict(q_to_k_xy=torch.from_numpy(xy.reshape(-1, 2)),
        q_to_k_pixel_cov=torch.from_numpy(pixel_cov.reshape(-1, 2, 2)),
        q_features=torch.from_numpy(rng.normal(size=(36, 7)).astype(np.float32)),
        k_features=torch.from_numpy(rng.normal(size=(36, 7)).astype(np.float32)))
    parts = {}
    for name, side in (('g8', 8), ('g16', 16), ('mixed', 16), ('g24', 24)):
        boxes = []
        for y in range(0, 48, side):
            for x in range(0, 48, side):
                mode = (x // 16 + 2 * (y // 16)) % 4 if name == 'mixed' else 0
                if mode == 1:
                    boxes.extend([[x, y, x + 8, y + 16], [x + 8, y, x + 16, y + 16]])
                elif mode == 2:
                    boxes.extend([[x, y, x + 16, y + 8], [x, y + 8, x + 16, y + 16]])
                elif mode == 3:
                    boxes.extend([[a, b, a + 8, b + 8]
                                  for b in (y, y + 8) for a in (x, x + 8)])
                else:
                    boxes.append([x, y, x + side, y + side])
        bounds = np.asarray(boxes, np.int64)
        owner = np.empty((48, 48), np.int64)
        for i, (x0, y0, x1, y1) in enumerate(bounds):
            owner[y0:y1, x0:x1] = i
        center = (bounds[:, :2] + bounds[:, 2:] - 1) / 2
        parent = center[:, 1].astype(int) // 16 * 3 + center[:, 0].astype(int) // 16
        if name == 'g24':
            parent[:] = -1
        parts[name] = dict(bounds=torch.from_numpy(bounds), owner=torch.from_numpy(owner.ravel()),
            atoms=torch.from_numpy(owner[4::8, 4::8].ravel()), parent=torch.from_numpy(parent))
    return observations, parts


def anchor_for(observations, partition):
    mean = OLD.prepare_query(observations, partition, 'gaussian_point')['mean'].double()
    return mean + mean.new_tensor([.125 + 1.e-10, -.0625])


def test_declared_twelve_complete_fixed_operators_are_json_serializable():
    assert len(P.FAMILIES) == 12
    assert list(json.loads(json.dumps(P.ROLE_CONFIGS))) == list(P.FAMILIES)
    assert P.FULL_FAMILIES == ('anchor_trace', 'mean_only_matched_tau', 'anchor_constant',
                               'anchor_isotropic_matched')
    assert len(P.TRACE_FAMILIES) == 3
    assert P.ROLE_CONFIGS['anchor_trace']['h'] == .25
    assert P.ROLE_CONFIGS['anchor_trace']['temperature'] == 10.


def test_second_moment_is_about_anchor_and_never_rounds_anchor_to_float32():
    mean = torch.tensor([[1., 2.], [-1., 3.]], dtype=torch.float32)
    cov = torch.tensor([[[.25, .125], [.125, .5]], [[1., -.25], [-.25, .5]]])
    anchor = mean.double() + torch.tensor([[.125 + 1.e-10, -.5], [.25, .125]], dtype=torch.float64)
    delta, actual = P.anchor_second_moment(mean, cov, anchor)
    expected_delta = mean.numpy().astype(np.float64) - anchor.numpy()
    expected = cov.numpy().astype(np.float64) + np.einsum('ni,nj->nij', expected_delta, expected_delta)
    assert np.array_equal(delta.numpy(), expected_delta)
    assert np.array_equal(actual.numpy(), expected)
    assert actual.dtype == torch.float64 and delta.dtype == torch.float64
    assert not torch.equal(anchor, anchor.float().double())
    assert bool((torch.linalg.eigvalsh(actual - cov.double()) >= -1.e-15).all())


def test_second_moment_translation_rotation_and_joint_unit_scaling():
    mean = torch.tensor([[1., 2.], [4., -2.]], dtype=torch.float32)
    cov = torch.tensor([[[.5, .125], [.125, .25]], [[1., -.25], [-.25, .5]]])
    anchor = mean.double() + torch.tensor([[.25, -.5], [-.125, .25]], dtype=torch.float64)
    _, base = P.anchor_second_moment(mean, cov, anchor)
    shift = mean.new_tensor([16., -8.])
    _, translated = P.anchor_second_moment(mean + shift, cov, anchor + shift.double())
    assert torch.equal(translated, base)
    _, scaled = P.anchor_second_moment(mean * 2, cov * 4, anchor * 2)
    assert torch.equal(scaled, base * 4)
    assert torch.equal(TRACE.trace_temperature(scaled, .5, 10.),
                       TRACE.trace_temperature(base, .25, 10.))
    rotation = mean.new_tensor([[0., -1.], [1., 0.]])
    _, rotated = P.anchor_second_moment(mean @ rotation.T, rotation @ cov @ rotation.T,
                                        anchor @ rotation.double().T)
    assert torch.equal(rotated, rotation.double() @ base @ rotation.double().T)


def test_full_roles_have_double_geometry_and_identical_primary_tau(synthetic):
    observations, parts = synthetic
    a = anchor_for(observations, parts['mixed'])
    queries = {n: P.prepare_query(observations, parts['mixed'], n, a) for n in P.FULL_FAMILIES}
    main = queries['anchor_trace']
    assert main['mean'] is a and main['cov'].dtype == torch.float64
    for name, query in queries.items():
        assert query['old_mean'].dtype == query['original_cov'].dtype == torch.float32
        assert query['cov'].dtype == query['tau_cov'].dtype == torch.float64
        assert torch.equal(query['tau_cov'], main['tau_cov'])
        assert torch.equal(query['anchor_delta'], main['anchor_delta'])
        assert query['tau_cov'].data_ptr() != main['tau_cov'].data_ptr() or name == 'anchor_trace'
    assert torch.equal(queries['mean_only_matched_tau']['cov'], main['original_cov'].double())
    variance = main['tau_cov'].diagonal(dim1=-2, dim2=-1).sum(-1) / 2
    assert torch.equal(queries['anchor_isotropic_matched']['cov'],
                       torch.diag_embed(variance[:, None].expand(-1, 2)))
    key = P.prepare_key(observations, parts['g24'], 'anchor_trace')
    terms = {n: P.prepared_terms(queries[n], key, n) for n in P.TRACE_FAMILIES}
    for n in terms:
        assert torch.equal(terms[n]['tau32'], terms['anchor_trace']['tau32'])
        assert torch.equal(terms[n]['bias32'], terms['anchor_trace']['bias32'])
    # Independent expression of the frozen scaled-double arithmetic.
    c = main['tau_cov'].numpy()
    scale = np.maximum(.25 ** 2, np.maximum(c[:, 0, 0], c[:, 1, 1]))
    r = .25 ** 2 / scale
    expected = (10. * (r / (r + (c[:, 0, 0] / scale + c[:, 1, 1] / scale)))).astype(np.float32)
    assert np.array_equal(terms['anchor_trace']['tau32'].numpy(), expected)


def test_point_and_fixed_do_not_prepare_or_borrow_full_moments(synthetic, monkeypatch):
    observations, parts = synthetic
    a = anchor_for(observations, parts['g16'])
    original = OLD.prepare_support
    calls = []
    def spy(*args, **kwargs):
        calls.append(kwargs['include_full_moments'])
        return original(*args, **kwargs)
    monkeypatch.setattr(OLD, 'prepare_support', spy)
    for name in P.FAMILIES[4:]:
        query = P.prepare_query(observations, parts['g16'], name, a)
        assert calls[-1] is False
        assert 'tau_cov' not in query and 'original_cov' not in query
        if P.ROLE_CONFIGS[name]['old_family'] == 'gaussian_fixed':
            expected = query['mass'].double()[:, None, None] / (64 * 12) * torch.eye(2, dtype=torch.float64)
            assert torch.equal(query['cov'], expected)
        else:
            assert not bool(query['cov'].any())


@pytest.mark.parametrize('family', P.FAMILIES)
def test_all_twelve_paths_match_old_integrals_and_declared_content_arithmetic(synthetic, family):
    observations, parts = synthetic
    a = anchor_for(observations, parts['g24'])
    query = P.prepare_query(observations, parts['g24'], family, a)
    key = P.prepare_key(observations, parts['g8'], family)
    config = P.ROLE_CONFIGS[family]
    adapted = dict(query, family=config['old_family'])
    expected_geometry, old_diagnostic = OLD.geometry(adapted, key, config['old_family'], config['h'])
    content = query['features'] @ key['features'].T
    if config['content_policy'] == 'trace':
        expected_bias = TRACE.content_bias(query['tau_cov'], config['h'], config['temperature'], content)
    else:
        expected_bias = config['temperature'] * content
    expected = (expected_geometry + expected_bias.double()).softmax(-1)
    actual, diagnostics = P.predict(query, key, family)
    terms = P.prepared_terms(query, key, family)
    assert torch.equal(actual, expected)
    assert torch.equal(terms['geometry'], expected_geometry)
    assert torch.equal(terms['bias32'], expected_bias)
    assert terms['content32'].dtype == terms['bias32'].dtype == terms['tau32'].dtype == torch.float32
    assert terms['raw_mass'].dtype == actual.dtype == torch.float64
    assert {n: diagnostics[n] for n in old_diagnostic} == old_diagnostic
    assert actual.shape == (len(a), len(key['mass']))
    assert diagnostics['all_finite'] and diagnostics['max_normalization_error'] <= 1.e-12


@pytest.mark.parametrize('qname,kname', [('g8', 'g16'), ('g16', 'mixed'), ('mixed', 'g24'), ('g24', 'g8')])
def test_equal_anchor_recovers_old_r17_and_t0_geometry(synthetic, qname, kname):
    observations, parts = synthetic
    old = R17.prepare_query(observations, parts[qname], 'trace_full')
    new = P.prepare_query(observations, parts[qname], 'anchor_trace', old['mean'].double())
    assert torch.equal(new['cov'], old['cov'].double())
    key = P.prepare_key(observations, parts[kname], 'anchor_trace')
    for temperature in (0., 10.):
        expected, _ = R17.predict(old, key, 'trace_full', .25, temperature)
        actual, _ = P.predict(new, key, 'anchor_trace', .25, temperature)
        assert torch.equal(actual, expected)
    geometry, _ = P.geometry(new, key, 'anchor_trace')
    actual, _ = P.predict(new, key, 'anchor_trace', temperature=0.)
    assert torch.equal(actual, geometry.softmax(-1))


def test_zero_old_covariance_only_reduces_to_point_when_anchor_equals_old_mean(synthetic):
    observations, parts = synthetic
    zero = dict(observations, q_to_k_xy=torch.full_like(observations['q_to_k_xy'], 12.),
                q_to_k_pixel_cov=torch.zeros_like(observations['q_to_k_pixel_cov']))
    a = torch.full((len(parts['g24']['bounds']), 2), 1.5, dtype=torch.float64)
    main = P.prepare_query(zero, parts['g24'], 'anchor_trace', a)
    point = P.prepare_query(zero, parts['g24'], 'anchor_point_matched', a)
    key = P.prepare_key(zero, parts['g8'], 'anchor_trace')
    assert not bool(main['cov'].any())
    assert torch.equal(P.predict(main, key, 'anchor_trace')[0],
                       P.predict(point, key, 'anchor_point_matched')[0])
    shifted = P.prepare_query(zero, parts['g24'], 'anchor_trace', a + a.new_tensor([1., -.5]))
    assert not bool(shifted['original_cov'].any())
    assert bool(shifted['cov'].any())
    expected = torch.tensor([[1., -.5], [-.5, .25]], dtype=torch.float64).expand_as(shifted['cov'])
    assert torch.equal(shifted['cov'], expected)


def test_native24_children_are_summed_before_one_floor(synthetic, monkeypatch):
    observations, parts = synthetic
    family = 'anchor_trace'
    query = P.prepare_query(observations, parts['g24'], family,
                            anchor_for(observations, parts['g24']))
    key = P.prepare_key(observations, parts['g24'], family)
    assert len(key['cell_bounds']) == 4 * len(key['mass'])
    assert torch.equal(key['cell_owner'], torch.arange(4).repeat_interleave(4))
    def child_mass(mean, covariance, boxes, **kwargs):
        assert kwargs['quadrature_order'] == 32
        return (torch.arange(1, len(boxes) + 1, dtype=torch.float64) * 1.e-3)[None].expand(len(mean), -1)
    monkeypatch.setattr(OLD, 'gaussian_cell_mass_fast', child_mass)
    terms = P.prepared_terms(query, key, family, temperature=0.)
    expected_raw = (torch.arange(1, 17, dtype=torch.float64) * 1.e-3).reshape(4, 4).sum(-1)
    expected_geometry = (expected_raw + 1.e-30 * key['mass'].double() / 64).log()[None].expand(len(query['mean']), -1)
    assert torch.equal(terms['raw_mass'], expected_raw[None].expand_as(terms['raw_mass']))
    assert torch.equal(terms['geometry'], expected_geometry)


@pytest.mark.parametrize('family', ['anchor_trace', 'anchor_quartic_localizer'])
def test_floor_and_exterior_population_are_not_filtered(synthetic, monkeypatch, family):
    observations, parts = synthetic
    a = torch.full((len(parts['mixed']['bounds']), 2), 128., dtype=torch.float64)
    query = P.prepare_query(observations, parts['mixed'], family, a)
    key = P.prepare_key(observations, parts['mixed'], family)
    def zero_mass(mean, covariance, boxes, **kwargs):
        return torch.zeros((len(mean), len(boxes)), dtype=torch.float64)
    monkeypatch.setattr(OLD, 'gaussian_cell_mass_fast', zero_mass)
    monkeypatch.setattr(OLD, 'quartic_cell_mass', zero_mass)
    actual, diagnostic = P.predict(query, key, family, temperature=0.)
    expected_geometry = (1.e-30 * key['mass'].double() / 64).log()[None].expand_as(actual)
    assert torch.equal(actual, expected_geometry.softmax(-1))
    assert actual.shape[0] == len(parts['mixed']['bounds'])
    assert diagnostic['raw_zero_fov_rows'] == diagnostic['raw_tiny_fov_rows'] == len(a)


def test_gradient_reaches_anchor_original_geometry_and_content(synthetic):
    observations, parts = synthetic
    inputs = {n: value.clone().requires_grad_() for n, value in observations.items()}
    a = anchor_for(observations, parts['g24']).clone().requires_grad_()
    query = P.prepare_query(inputs, parts['g24'], 'anchor_trace', a)
    key = P.prepare_key(inputs, parts['g8'], 'anchor_trace')
    probability, _ = P.predict(query, key, 'anchor_trace')
    probability[:, 0].sum().backward()
    for field in inputs:
        assert inputs[field].grad is not None and torch.isfinite(inputs[field].grad).all()
        assert bool((inputs[field].grad != 0).any())
    assert a.grad is not None and torch.isfinite(a.grad).all() and bool((a.grad != 0).any())


def test_native_geometry_translation_and_tau_shape_contract(synthetic):
    observations, parts = synthetic
    family = 'anchor_trace'
    query = P.prepare_query(observations, parts['g24'], family,
                            anchor_for(observations, parts['g24']))
    key = P.prepare_key(observations, parts['g24'], family)
    shift = torch.tensor([8., -16.], dtype=torch.float64)
    translated_query = dict(query, mean=query['mean'] + shift)
    translated_key = dict(key, cell_bounds=key['cell_bounds'] + shift.repeat(2))
    original = P.prepared_terms(query, key, family)
    translated = P.prepared_terms(translated_query, translated_key, family)
    torch.testing.assert_close(translated['geometry'], original['geometry'], atol=1.e-12, rtol=0.)
    assert torch.equal(translated['tau32'], original['tau32'])
    bad = dict(query, tau_cov=query['tau_cov'][:1])
    with pytest.raises(ValueError, match='matching covariance shape/device'):
        P.predict(bad, key, family)


@pytest.mark.parametrize('family', ['anchor_trace', 'anchor_constant'])
def test_content_temperature_rejects_invalid_or_overflowing_values(synthetic, family):
    observations, parts = synthetic
    query = P.prepare_query(observations, parts['g24'], family,
                            anchor_for(observations, parts['g24']))
    key = P.prepare_key(observations, parts['g8'], family)
    for temperature in (-1., float('nan'), 1.e300):
        with pytest.raises(ValueError):
            P.predict(query, key, family, temperature=temperature)


@pytest.mark.parametrize('bad_anchor', ['float32', 'nan', 'wrong_shape'])
def test_anchor_schema_and_full_input_rejections(synthetic, bad_anchor):
    observations, parts = synthetic
    a = anchor_for(observations, parts['g24'])
    if bad_anchor == 'float32':
        a = a.float()
    elif bad_anchor == 'nan':
        a[0, 0] = float('nan')
    else:
        a = a[:1]
    with pytest.raises(ValueError, match='anchor must be finite float64'):
        P.prepare_query(observations, parts['g24'], 'anchor_trace', a)


def test_original_covariance_and_tau_covariance_cannot_hide_invalid_cross_terms(synthetic):
    mean = torch.zeros((1, 2), dtype=torch.float32)
    a = torch.ones((1, 2), dtype=torch.float64)
    for cov in (torch.tensor([[[1., 2.], [2., 1.]]]),
                torch.tensor([[[1., 0.], [.1, 1.]]]),
                torch.tensor([[[float('nan'), 0.], [0., 1.]]])):
        with pytest.raises(ValueError):
            P.anchor_second_moment(mean, cov, a)
    observations, parts = synthetic
    query = P.prepare_query(observations, parts['g24'], 'anchor_isotropic_matched',
                            anchor_for(observations, parts['g24']))
    key = P.prepare_key(observations, parts['g8'], 'anchor_isotropic_matched')
    query['tau_cov'] = query['tau_cov'].clone()
    query['tau_cov'][:, 0, 1] = query['tau_cov'][:, 1, 0] = 1.e6
    for temperature in (0., 10.):
        with pytest.raises(ValueError, match='positive semidefinite'):
            P.predict(query, key, 'anchor_isotropic_matched', temperature=temperature)


def test_extra_gt_fields_wrong_family_and_invalid_content_are_rejected(synthetic, monkeypatch):
    observations, parts = synthetic
    a = anchor_for(observations, parts['g24'])
    with pytest.raises(ValueError, match='four predicted'):
        P.prepare_query(dict(observations, gt_depth=torch.ones(1)), parts['g24'], 'anchor_trace', a)
    with pytest.raises(ValueError, match='undeclared'):
        P.prepare_query(observations, parts['g24'], 'search_another_shape', a)
    query = P.prepare_query(observations, parts['g24'], 'anchor_trace', a)
    key = P.prepare_key(observations, parts['g8'], 'anchor_trace')
    with pytest.raises(ValueError, match='different family'):
        P.predict(query, key, 'mean_only_matched_tau')
    bad = dict(query, features=torch.full_like(query['features'], float('nan')))
    with pytest.raises(ValueError, match='content must be finite'):
        P.predict(bad, key, 'anchor_trace', temperature=0.)
    monkeypatch.setattr(torch, 'set_num_threads', lambda *_: pytest.fail('model must not alter runtime'))
    P.predict(query, key, 'anchor_trace')
