"""Synthetic-only draft checks; no dataset or result archive is opened."""
import numpy as np
import pytest
import torch

from models import round15_validation_pe as P
from models import round12_stress_pe as LEGACY
from experiments import round15_regional as REGIONAL
from experiments import round15_train_numerical24 as REFERENCE


@pytest.fixture(scope='module', autouse=True)
def four_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(4)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope='module')
def synthetic():
    width = height = 48
    yy, xx = np.mgrid[:height, :width]
    xy = np.stack((.8*xx+.2*yy+.3*np.sin(yy/4), -.15*xx+1.1*yy+.2*np.cos(xx/7)), -1)
    covariance = np.empty((height, width, 2, 2))
    covariance[..., 0, 0] = .1+.01*(xx%3)
    covariance[..., 1, 1] = .2+.01*(yy%5)
    covariance[..., 0, 1] = covariance[..., 1, 0] = .025
    rng = np.random.default_rng(34)
    inputs = dict(q_to_k_xy=xy.reshape(-1, 2).astype(np.float32),
        q_to_k_pixel_cov=covariance.reshape(-1, 2, 2).astype(np.float32),
        q_features=rng.normal(size=(36, 12)).astype(np.float32),
        k_features=rng.normal(size=(36, 12)).astype(np.float32))
    observations = {k:torch.from_numpy(v) for k, v in inputs.items()}
    parts = {}
    for name, side in (('g8', 8), ('g16', 16), ('mixed', 16), ('g24', 24)):
        boxes = []
        for y in range(0, height, side):
            for x in range(0, width, side):
                mode = (x//16+2*(y//16)) % 4 if name == 'mixed' else 0
                if mode == 1:
                    boxes.extend([[x, y, x+8, y+16], [x+8, y, x+16, y+16]])
                elif mode == 2:
                    boxes.extend([[x, y, x+16, y+8], [x, y+8, x+16, y+16]])
                elif mode == 3:
                    boxes.extend([[a, b, a+8, b+8] for b in (y, y+8) for a in (x, x+8)])
                else:
                    boxes.append([x, y, x+side, y+side])
        bounds = np.array(boxes, np.int64)
        owner = np.empty((height, width), np.int64)
        for i, (x0, y0, x1, y1) in enumerate(boxes):
            owner[y0:y1, x0:x1] = i
        size = bounds[:, 2:]-bounds[:, :2]
        mean = (bounds[:, :2]+bounds[:, 2:]-1)/2
        cov = np.zeros((len(bounds), 2, 2))
        cov[:, 0, 0], cov[:, 1, 1] = size[:, 0]**2/(12*64), size[:, 1]**2/(12*64)
        children, child_owner = LEGACY.split_cell_bounds(bounds)
        parent = (mean[:, 1].astype(int)//16)*3+mean[:, 0].astype(int)//16
        if name == 'g24':
            parent[:] = -1
        parts[name] = dict(bounds=bounds, owner=owner.ravel(), atoms=owner[4::8, 4::8].ravel(),
            parent=parent, mean=mean/8, cov=cov, mass=size.prod(-1),
            cell_bounds=children, cell_owner=child_owner)
    return inputs, observations, LEGACY.partition_tensors(parts, 'cpu')


def test_role_specific_preparation_preserves_all_original_needed_statistics(synthetic):
    inputs, observations, parts = synthetic
    old_queries, old_keys = REGIONAL.prepare_pair(inputs, parts)
    for name, part in parts.items():
        key = P.prepare_key(observations, part)
        for field in ('mean', 'mass', 'native_cell_bounds', 'cell_bounds', 'cell_owner', 'features'):
            assert torch.equal(key[field], old_keys[name][field]), (name, field)
        for family in P.FAMILIES[:-1]:
            query = P.prepare_query(observations, part, family)
            old = old_queries[name][REGIONAL.PREPARATION[family]]
            for field in ('mean', 'mass', 'parent', 'features'):
                assert torch.equal(query[field], old[field]), (name, family, field)
            if family in ('quartic_full', 'quartic_point', 'gaussian_point', 'gaussian_full1'):
                assert torch.equal(query['cov'], old['cov'])
            elif family == 'gaussian_fixed':
                assert 'cov' not in query
            else:
                assert 'cov' not in query
                for field in ('nodes', 'node_cov', 'node_weights'):
                    assert torch.equal(query[field], old[field]), (name, family, field)


@pytest.mark.parametrize('qname,kname', [('g8', 'g16'), ('g16', 'mixed'), ('mixed', 'g24'), ('g24', 'g8')])
def test_all_families_and_h_t_preserve_stage_b_probabilities(synthetic, qname, kname):
    inputs, observations, parts = synthetic
    old_queries, old_keys = REGIONAL.prepare_pair(inputs, parts)
    key = P.prepare_key(observations, parts[kname])
    for family in P.FAMILIES[:-1]:
        query = P.prepare_query(observations, parts[qname], family)
        role_key = P.prepare_key(observations, parts[kname], family)
        old_query = old_queries[qname][REGIONAL.PREPARATION[family]]
        for h in (.25, .5, 1., 2.):
            old_logits, old_diagnostic = REGIONAL.geometry(old_query, old_keys[kname], family, h)
            for temperature in (0., 5., 10.):
                expected = (old_logits+temperature*(old_query['features']@old_keys[kname]['features'].T)).softmax(-1)
                actual, diagnostics = P.predict(query, key, family, h, temperature)
                role_actual, role_diagnostics = P.predict(query, role_key, family, h, temperature)
                assert torch.equal(role_actual, actual)
                assert role_diagnostics == diagnostics
                assert torch.equal(actual, expected), (qname, kname, family, h, temperature)
                assert actual.dtype == torch.float64
                assert {k:diagnostics[k] for k in old_diagnostic} == old_diagnostic
                assert diagnostics['all_finite'] is True
                assert diagnostics['max_normalization_error'] == float((actual.sum(-1)-1).abs().max())
                assert diagnostics['max_normalization_error'] < 1.e-14


def test_point_fixed_quad_omit_unused_scatter_and_iso_ignores_off_diagonal(synthetic, monkeypatch):
    _, observations, parts = synthetic
    recorded = []
    original = P.prepare_support
    def spy(*args, **kwargs):
        recorded.append(dict(kwargs, pixel_covariance_argument=args[1]))
        return original(*args, **kwargs)
    monkeypatch.setattr(P, 'prepare_support', spy)
    for family in ('quartic_point', 'gaussian_point', 'gaussian_fixed', 'gaussian_quad4', 'gaussian_quad16',
                   'quartic_isotropic_matched'):
        query = P.prepare_query(observations, parts['g16'], family)
        assert recorded[-1]['include_full_moments'] is False
        if 'quad' in family:
            assert 'cov' not in query and query['node_cov'].abs().max() < .01
        else:
            assert recorded[-1]['pixel_covariance_argument'] is None
    # A nonfinite off-diagonal is deliberately diagnostic poison: only the
    # independent trace-only role can ignore it while still reading diagonals.
    changed = dict(observations, q_to_k_pixel_cov=observations['q_to_k_pixel_cov'].clone())
    changed['q_to_k_pixel_cov'][:, 0, 1] = float('nan')
    changed['q_to_k_pixel_cov'][:, 1, 0] = float('nan')
    for family in ('quartic_point', 'gaussian_point', 'gaussian_fixed', 'quartic_isotropic_matched'):
        a = P.prepare_query(observations, parts['g16'], family)
        b = P.prepare_query(changed, parts['g16'], family)
        for name in a:
            if isinstance(a[name], torch.Tensor):
                assert torch.equal(a[name], b[name])


def test_trace_only_moments_and_probabilities_match_independent_numpy(synthetic):
    inputs, observations, parts = synthetic
    key = P.prepare_key(observations, parts['g24'])
    role_key = P.prepare_key(observations, parts['g24'], 'quartic_isotropic_matched')
    xy = inputs['q_to_k_xy'].astype(float)/8
    pixel_cov = inputs['q_to_k_pixel_cov'].astype(float)/64
    for name, part in parts.items():
        query = P.prepare_query(observations, part, 'quartic_isotropic_matched')
        full = P.prepare_query(observations, part, 'quartic_full')
        # The two-column float32 reduction matches the original diagonal
        # reduction on this fixture, although this is not assumed universally.
        expected_trace = full['cov'].double().diagonal(dim1=-2, dim2=-1).mean(-1)
        torch.testing.assert_close(query['cov'][:, 0, 0], expected_trace, rtol=1.e-6, atol=1.e-7)
        means, covariances, qfeatures = [], [], []
        for group in range(len(part['bounds'])):
            group_xy = xy[part['owner'].numpy()==group]
            mean = group_xy.mean(0)
            delta = group_xy-mean
            cov = delta.T@delta/len(delta)+pixel_cov[part['owner'].numpy()==group].mean(0)
            means.append(mean)
            covariances.append(np.eye(2)*np.trace(cov)/2)
            feature = inputs['q_features'][part['atoms'].numpy()==group].astype(float).mean(0)
            qfeatures.append(feature/max(np.linalg.norm(feature), 1.e-12))
        np.testing.assert_allclose(query['mean'].numpy(), means, atol=6.e-6, rtol=1.e-6)
        np.testing.assert_allclose(query['cov'].numpy(), covariances, atol=2.e-6, rtol=2.e-6)
        kfeatures = []
        for group in range(len(key['mass'])):
            feature = inputs['k_features'][parts['g24']['atoms'].numpy()==group].astype(float).mean(0)
            kfeatures.append(feature/max(np.linalg.norm(feature), 1.e-12))
        bounds = key['native_cell_bounds'].numpy()
        content = np.asarray(qfeatures)@np.asarray(kfeatures).T
        for h in (.25, .5, 1., 2.):
            ref64 = REFERENCE.independent_mass(means, covariances, bounds, h, 64)
            ref96 = REFERENCE.independent_mass(means, covariances, bounds, h, 96)
            for temperature in (0., 5., 10.):
                p64 = REFERENCE.probability(ref64, content, temperature, bounds)
                p96 = REFERENCE.probability(ref96, content, temperature, bounds)
                actual, _ = P.predict(query, key, 'quartic_isotropic_matched', h, temperature)
                role_actual, _ = P.predict(query, role_key, 'quartic_isotropic_matched', h, temperature)
                assert torch.equal(role_actual, actual)
                assert np.max(np.abs(p64-p96)) < 1.e-8
                assert np.max(np.abs(actual.numpy()-p96)) < 1.e-4


def test_zero_mass_native_area_floor_once_after_integral(synthetic, monkeypatch):
    _, observations, parts = synthetic
    query = P.prepare_query(observations, parts['g24'], 'quartic_point')
    key = P.prepare_key(observations, parts['mixed'])
    def zero_mass(mean, covariance, boxes, **kwargs):
        assert kwargs['quadrature_order'] == 24
        return torch.zeros((len(mean), len(boxes)), dtype=torch.float64)
    monkeypatch.setattr(P, 'quartic_cell_mass', zero_mass)
    actual, diagnostics = P.predict(query, key, 'quartic_point', .25, 0.)
    expected = key['mass'].double()/key['mass'].double().sum()
    torch.testing.assert_close(actual, expected[None].expand_as(actual), atol=1.e-15, rtol=1.e-14)
    assert diagnostics['raw_zero_fov_rows'] == len(query['mean'])
    assert diagnostics['raw_tiny_fov_rows'] == len(query['mean'])
    assert diagnostics['raw_floor_fov_mass'] == 1.e-30*36


def test_native_24_bounds_and_gaussian_children_remain_distinct(synthetic, monkeypatch):
    _, observations, parts = synthetic
    key = P.prepare_key(observations, parts['g24'])
    assert len(key['cell_bounds']) == 4*len(key['native_cell_bounds'])
    assert torch.all(key['native_cell_bounds'][:, 2:]-key['native_cell_bounds'][:, :2] == 3.)
    assert torch.all(key['cell_bounds'][:, 2:]-key['cell_bounds'][:, :2] == 1.5)
    assert torch.equal(key['native_cell_bounds'][0], torch.tensor([-.0625, -.0625, 2.9375, 2.9375]))
    stale = dict(parts['g24'], mass=torch.zeros(4), mean=torch.zeros((4, 2)), cell_bounds=torch.ones((1, 4)))
    rebuilt = P.prepare_key(observations, stale)
    for field in key:
        assert torch.equal(key[field], rebuilt[field])


def test_role_specific_key_builds_only_needed_bounds_with_exact_shared_values(synthetic, monkeypatch):
    _, observations, parts = synthetic
    shared = {'mean', 'mass', 'features'}
    for part in parts.values():
        superset = P.prepare_key(observations, part)
        for family in P.FAMILIES:
            key = P.prepare_key(observations, part, family)
            fields = {'native_cell_bounds'} if family.startswith('quartic_') else {'cell_bounds', 'cell_owner'}
            assert set(key) == shared | fields
            for name, value in key.items():
                assert torch.equal(value, superset[name])
    # The quartic preparation must not construct Gaussian children and merely
    # discard them afterward: no repeat_interleave call is needed or allowed.
    monkeypatch.setattr(torch, 'repeat_interleave', lambda *_args, **_kw:pytest.fail('unused Gaussian children'))
    for family in ('quartic_full', 'quartic_point', 'quartic_isotropic_matched'):
        key = P.prepare_key(observations, parts['g24'], family)
        assert key['native_cell_bounds'].shape == (4, 4)
        assert torch.all(key['native_cell_bounds'][:, 2:]-key['native_cell_bounds'][:, :2] == 3.)


def test_temperature_zero_still_executes_content_and_model_does_not_set_threads(synthetic, monkeypatch):
    _, observations, parts = synthetic
    query = P.prepare_query(observations, parts['g24'], 'quartic_point')
    key = P.prepare_key(observations, parts['g24'])
    monkeypatch.setattr(torch, 'set_num_threads', lambda n: pytest.fail('model owns no runtime configuration'))
    P.predict(query, key, 'quartic_point', .5, 0.)
    bad = dict(query, features=torch.full_like(query['features'], float('nan')))
    with pytest.raises(ValueError, match='invalid normalized'):
        P.predict(bad, key, 'quartic_point', .5, 0.)


def test_schema_rejects_gt_extra_family_mismatch_and_wrong_precision(synthetic):
    _, observations, parts = synthetic
    with pytest.raises(ValueError, match='four predicted'):
        P.prepare_query(dict(observations, gt_visible=torch.ones(1)), parts['g8'], 'quartic_full')
    with pytest.raises(ValueError, match='float32'):
        P.prepare_key({k:v.double() for k, v in observations.items()}, parts['g8'])
    with pytest.raises(ValueError, match='undeclared'):
        P.prepare_query(observations, parts['g8'], 'quartic_isotropic')
    with pytest.raises(ValueError, match='undeclared'):
        P.prepare_key(observations, parts['g8'], 'quartic_isotropic')
    query = P.prepare_query(observations, parts['g8'], 'gaussian_fixed')
    with pytest.raises(ValueError, match='different family'):
        P.geometry(query, P.prepare_key(observations, parts['g8']), 'gaussian_full1', .5)
