from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from scipy.special import ndtr

from experiments import round15_regional_audit_numeric as N


def synthetic_parts():
    output = {}
    for name in N.PARTITIONS:
        stride = 8 if name == 'g8' else 24 if name == 'g24' else 16
        bounds = []
        for y in range(0, 144, stride):
            for x in range(0, 192, stride):
                if name == 'mixed' and (x//16+y//16) % 2:
                    bounds.extend([[x, y, x+8, y+16], [x+8, y, x+16, y+16]])
                else:
                    bounds.append([x, y, x+stride, y+stride])
        b = np.asarray(bounds, np.int64); size = b[:, 2:]-b[:, :2]
        mean = (b[:, :2]+b[:, 2:]-1)/2
        covariance = np.zeros((len(b), 2, 2))
        covariance[:, 0, 0] = size[:, 0]**2/(12*64)
        covariance[:, 1, 1] = size[:, 1]**2/(12*64)
        parent = (mean[:, 1].astype(int)//16)*12+mean[:, 0].astype(int)//16
        if name == 'g24':
            parent[:] = -1
        output[name] = dict(bounds=b, mean=mean/8, cov=covariance,
                            mass=size.prod(1).astype(float), parent=parent,
                            cell_bounds=np.array([[999., 999., 1000., 1000.]]))
    return output


def synthetic_inputs():
    y, x = np.mgrid[:144, :192]
    return dict(q_to_k_xy=np.stack((x, y), -1).reshape(-1, 2).astype(float),
        q_to_k_pixel_cov=np.broadcast_to(np.eye(2)/12, (144*192, 2, 2)).copy(),
        q_features=np.ones((432, 3)), k_features=np.ones((432, 3)))


def synthetic_selection(selected=4):
    h, temperature = N.GRID[selected]
    return dict(controls={family: dict(family=family, grid_index=selected,
        config=dict(h=h, temperature=temperature), J=float('nan')) for family in N.FAMILIES[1:]})


def test_reference_native_units_and_original_g24_rectangle():
    parts = synthetic_parts(); schema = N.reference_partition_schema(parts)
    np.testing.assert_array_equal(schema['g24_bounds'][0], [0, 0, 24, 24])
    np.testing.assert_allclose((schema['g24_bounds'][0]-.5)/8, [-.0625, -.0625, 2.9375, 2.9375])
    np.testing.assert_allclose(schema['g24_centers'][0], [11.5, 11.5])
    np.testing.assert_allclose(schema['g24_covariance'][0], 48*np.eye(2))
    for name in N.PARTITIONS:
        assert schema[name+'_area'].sum() == 144*192
    bad = deepcopy(parts); bad['g8']['mean'] *= 8
    with pytest.raises(ValueError, match='units'):
        N.reference_partition_schema(bad)
    bad = deepcopy(parts); bad['g24']['parent'][0] = 0
    with pytest.raises(ValueError, match='common16'):
        N.reference_partition_schema(bad)


def test_literal_pixel_moments_and_stratum_scatter_are_distinct():
    inputs = synthetic_inputs(); parts = synthetic_parts()
    queries, keys, indices, schema = N.reference_banks(inputs, parts)
    np.testing.assert_array_equal(indices['g24'], np.linspace(0, 47, 8, dtype=int))
    full = queries['g24']['full']; local = queries['g24']['full4']
    np.testing.assert_allclose(full['mean'][0], [11.5/8, 11.5/8])
    np.testing.assert_allclose(full['cov'], np.broadcast_to(.75*np.eye(2), (8, 2, 2)), atol=1e-14)
    np.testing.assert_allclose(local['node_cov'], np.broadcast_to(.1875*np.eye(2), (8, 4, 2, 2)), atol=1e-14)
    np.testing.assert_allclose(queries['g24']['quad4']['node_cov'],
        np.broadcast_to(np.eye(2)/(12*64), (8, 4, 2, 2)), atol=1e-14)
    mixed_mean, mixed_cov = N.FOURREF.mixture_moments(local)
    np.testing.assert_allclose(mixed_mean, full['mean'], atol=1e-14)
    np.testing.assert_allclose(mixed_cov, full['cov'], atol=1e-14)
    np.testing.assert_allclose(keys['g24']['mass'], 576)
    with pytest.raises(ValueError, match='four forward'):
        N.forward_inputs(dict(inputs, gt_visible=np.ones(144*192, bool)))


def test_gaussian_point_reference_matches_exact_cdf_in_native_24px_cell():
    bounds = np.array([[-.0625, -.0625, 2.9375, 2.9375], [2.9375, -.0625, 5.9375, 2.9375]])
    means = np.array([[1.3, 1.7], [-.7, 1.1]])
    query = dict(mean=means, cov=np.broadcast_to(np.eye(2), (2, 2, 2)).copy(), mass=np.full(2, 576.))
    actual = N.gaussian_reference_mass(query, bounds, 'gaussian_point', .5)
    expected = np.empty((2, 2))
    for q, mean in enumerate(means):
        expected[q] = ((ndtr((bounds[:, 2]-mean[0])/.5)-ndtr((bounds[:, 0]-mean[0])/.5))
                      *(ndtr((bounds[:, 3]-mean[1])/.5)-ndtr((bounds[:, 1]-mean[1])/.5)))
    np.testing.assert_allclose(actual, expected, atol=2e-15, rtol=1e-12)


def test_quartic_reference_converges_and_native_mass_is_additive():
    means = np.array([[.13, -.17], [1.4, 1.4]])
    covariance = np.array([[[.3, .12], [.12, .7]], [[.04, -.015], [-.015, .1]]])
    large = np.array([[-.0625, -.0625, 2.9375, 2.9375]])
    edges = np.array([-.0625, .9375, 1.9375, 2.9375])
    small = np.array([[x, y, edges[i+1], edges[j+1]]
                     for j, y in enumerate(edges[:-1]) for i, x in enumerate(edges[:-1])])
    raw64 = N.QREF.independent_mass(means, covariance, large, .25, 64)
    raw96 = N.QREF.independent_mass(means, covariance, large, .25, 96)
    split96 = N.QREF.independent_mass(means, covariance, small, .25, 96).sum(1)
    np.testing.assert_allclose(raw64, raw96, atol=1e-12, rtol=0)
    np.testing.assert_allclose(raw96[:, 0], split96, atol=1e-12, rtol=0)


def test_zero_fov_floor_uses_area_once_and_failures_remain_visible():
    boxes = np.array([[0., 0., 1., 1.], [1., 0., 4., 3.]])
    raw = np.zeros((8, 2)); content = np.zeros_like(raw)
    p = N.probabilities(raw, content, 10., boxes)
    np.testing.assert_allclose(p, np.tile([.1, .9], (8, 1)), atol=1e-15)
    result = N.probability_check(p, p, p.copy(), raw, boxes, np.arange(8))
    assert result['passed'] and result['reference_zero_fov_rows'] == 8
    assert result['reference_tiny_fov_indices'] == list(range(8))
    assert result['reference_floor_fov_mass'] == pytest.approx(1e-29)
    bad = p.copy(); bad[0] = [.95, .05]
    result = N.probability_check(bad, p, p.copy(), raw, boxes, np.arange(8))
    assert not result['passed'] and result['argmax_change_count'] == 1
    assert result['argmax_changes'][0]['reference_top_gap'] == pytest.approx(.8)
    reference64 = p.copy(); reference64[0] += [2e-8, -2e-8]
    result = N.probability_check(p, p, reference64, raw, boxes, np.arange(8))
    assert not result['passed'] and result['reference_convergence_error'] > 1e-8


def test_fixed_gaussian_sample_supplemented_by_selected_configuration():
    fixed = N.planned_configurations(synthetic_selection(4))
    changed = N.planned_configurations(synthetic_selection(1))
    for family in N.FAMILIES[:2]:
        assert fixed[family] == changed[family] == tuple(range(12))
    for family in N.GAUSSIAN_METHODS:
        assert fixed[family] == (4,) and changed[family] == (4, 1)
    bad = synthetic_selection(1); bad['controls']['gaussian_full4']['config']['h'] = 2.
    with pytest.raises(ValueError, match='identity'):
        N.planned_configurations(bad)


def test_relative_paths_resolve_against_repository_not_filesystem_root(tmp_path, monkeypatch):
    monkeypatch.setattr(N, 'ROOT', tmp_path)
    assert N.absolute('results/a.npz') == tmp_path/'results/a.npz'
    assert N.absolute(Path('/absolute/a.npz')) == Path('/absolute/a.npz')


def test_pair_keeps_all_conditions_configurations_and_failed_matrices(monkeypatch):
    parts = synthetic_parts(); inputs = synthetic_inputs()
    paths = {q+'_'+k: Path(q+'_'+k+'.npz') for q, k in N.CONDITIONS}
    fields = [f'{family}__c{i:02d}' for family in N.FAMILIES for i in range(12)]
    seen = []

    def zero_quartic(mean, covariance, bounds, h, order):
        assert order in (64, 96)
        return np.zeros((len(mean), len(bounds)), np.float64)

    def zero_gaussian(query, bounds, family, h):
        return np.zeros((len(query['mean']), len(bounds)), np.float64)

    class SyntheticArchive:
        files = fields

        def __init__(self, path):
            self.q, self.k = path.stem.split('_')

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def __getitem__(self, field):
            seen.append((self.q, self.k, field))
            mass = parts[self.k]['mass']
            p = np.tile(mass/mass.sum(), (len(parts[self.q]['bounds']), 1))
            if self.q == 'g8' and self.k == 'g8' and field == 'quartic_full__c00':
                p[0] = 0.; p[0, 0] = 1.
            return p

    monkeypatch.setattr(N.QREF, 'independent_mass', zero_quartic)
    monkeypatch.setattr(N, 'gaussian_reference_mass', zero_gaussian)
    monkeypatch.setattr(N.np, 'load', lambda path, allow_pickle: SyntheticArchive(path))
    result = N.numeric_pair(inputs, parts, paths, synthetic_selection(1))
    assert not result['passed']
    assert result['matrix_count'] == len(seen) == 14*(24+12)
    assert result['quartic_matrix_count'] == 14*24
    assert result['gaussian_matrix_count'] == 14*12
    assert len({(r['query'], r['key']) for r in result['checks']}) == 14
    assert sum(not r['passed'] for r in result['checks']) == 1
    assert all(r['reference_zero_fov_rows'] == 8 for r in result['checks'])
    assert not result['labels_decoded'] and not result['scientific_gate_passed']
    for r in result['checks']:
        assert r['indices'] == np.linspace(0, len(parts[r['query']]['bounds'])-1, 8, dtype=int).tolist()
    with pytest.raises(ValueError, match='fourteen'):
        N.numeric_pair(inputs, parts, dict(list(paths.items())[:-1]), synthetic_selection(1))
