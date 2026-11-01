"""Synthetic-only coverage for the future validation numerical audit."""
from copy import deepcopy

import numpy as np
import pytest

from experiments import round15_regional_audit_metrics as PARTS
from experiments import round17_validation_audit_numeric as N


def synthetic_inputs():
    y, x = np.mgrid[:144, :192]
    z = np.full((144, 192), 2.)
    points = np.stack(((x - 95.5) * z / 100, (y - 71.5) * z / 100, z), -1).reshape(-1, 3)
    J = np.zeros((27648, 2, 3)); J[:, 0, 0] = J[:, 1, 1] = 50
    J[:, 0, 2] = -25 * points[:, 0]; J[:, 1, 2] = -25 * points[:, 1]
    covariance = np.broadcast_to(np.diag([.0001, .0002, .0003]), (27648, 3, 3)).copy()
    projected = J @ covariance @ J.transpose(0, 2, 1)
    return dict(q_to_k_xy=np.stack((x, y), -1).reshape(-1, 2).astype(float),
        q_to_k_pixel_cov=projected, q_features=np.ones((432, 3)), k_features=np.ones((432, 3)),
        q_points3=points, q_pixel_cov3=covariance, q_to_k_transform=np.eye(4),
        k_K=np.array([[100., 0., 95.5], [0., 100., 71.5], [0., 0., 1.]]),
        q_predicted_depth=z, k_predicted_depth=z.copy())


def synthetic_operators():
    return {f'o{i:02d}': dict(family=family, h=.5, temperature=5.)
            for i, family in enumerate(N.FAMILIES)}


def test_fixed_conditions_have_one_zero_copy_and_all_six_perturbations():
    assert len(N.CONDITIONS) == 68
    assert len(set(N.CONDITIONS)) == 68
    assert sum(scenario == 'scale' for scenario, _, _, _ in N.CONDITIONS) == 5
    for state in N.STATES:
        assert sum(scenario == state for scenario, _, _, _ in N.CONDITIONS) == 9
    assert all(state == 'zero' for scenario, state, _, _ in N.CONDITIONS if scenario == 'scale')
    assert all(q != 'g24' and k != 'g24' for scenario, _, q, k in N.CONDITIONS if scenario != 'scale')


def test_projection_identity_and_metric_translation_preserve_all_pixels():
    inputs = synthetic_inputs()
    zero, record = N.reference_observations(inputs, 'zero')
    np.testing.assert_allclose(zero['q_to_k_xy'], inputs['q_to_k_xy'], atol=3e-14, rtol=0)
    np.testing.assert_allclose(zero['q_to_k_pixel_cov'], inputs['q_to_k_pixel_cov'], atol=3e-15, rtol=0)
    assert record['predicted_in_front_count'] == 27648
    assert record['zero_float32_xy_equal']
    assert not record['key_depth_used_by_positional_operator']
    for state, displacement in (('tx_plus01', .5), ('tx_minus01', -.5)):
        raw, record = N.reference_observations(inputs, state)
        np.testing.assert_allclose(raw['q_to_k_xy'] - zero['q_to_k_xy'],
                                  np.broadcast_to([displacement, 0.], (27648, 2)), atol=4e-14, rtol=0)
        assert raw['q_to_k_pixel_cov'].shape == (27648, 2, 2)
        assert record['transform'][0][3] == pytest.approx(displacement / 50)


def test_depth_scaling_both_fields_does_not_claim_key_depth_sensitivity():
    inputs = synthetic_inputs()
    zero, _ = N.reference_observations(inputs, 'zero')
    for state, factor in (('depth09', .9), ('depth11', 1.1)):
        raw, record = N.reference_observations(inputs, state)
        # Identity camera makes synchronous point/covariance depth scaling cancel
        # in the perspective projection. These are still different depth states.
        np.testing.assert_allclose(raw['q_to_k_xy'], zero['q_to_k_xy'], atol=4e-14, rtol=0)
        np.testing.assert_allclose(raw['q_to_k_pixel_cov'], zero['q_to_k_pixel_cov'], atol=4e-15, rtol=0)
        assert record['depth_factor'] == factor
        for side in ('q', 'k'):
            assert record['effective_depth_sha256'][side] == N.STRESS.array_sha(np.full((144, 192), 2 * factor))
    changed = deepcopy(inputs); changed['k_predicted_depth'][:] = 19.
    raw, changed_record = N.reference_observations(changed, 'depth11')
    baseline, baseline_record = N.reference_observations(inputs, 'depth11')
    np.testing.assert_array_equal(raw['q_to_k_xy'], baseline['q_to_k_xy'])
    np.testing.assert_array_equal(raw['q_to_k_pixel_cov'], baseline['q_to_k_pixel_cov'])
    assert changed_record['effective_depth_sha256']['k'] != baseline_record['effective_depth_sha256']['k']
    assert changed_record['effective_depth_sha256']['k'] == N.STRESS.array_sha(np.full((144, 192), 19. * 1.1))


def test_target_y_rotation_uses_degrees_and_left_multiplication():
    inputs = synthetic_inputs()
    for state, sign in (('ry_plus1', 1), ('ry_minus1', -1)):
        raw, record = N.reference_observations(inputs, state)
        a = sign * np.pi / 180
        R = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
        transformed = inputs['q_points3'] @ R.T
        hom = transformed @ inputs['k_K'].T
        np.testing.assert_allclose(raw['q_to_k_xy'], hom[:, :2] / hom[:, 2, None], atol=1e-12, rtol=0)
        np.testing.assert_allclose(np.asarray(record['transform'])[:3, :3], R, atol=0, rtol=0)
    # A preexisting translation must rotate with the camera rows, not be left
    # fixed while only the rotation block is changed.
    changed = deepcopy(inputs); changed['q_to_k_transform'][:3, 3] = [.1, .2, .3]
    _, record = N.reference_observations(changed, 'ry_plus1')
    a = np.pi / 180
    expected = [np.cos(a) * .1 + np.sin(a) * .3, .2, -np.sin(a) * .1 + np.cos(a) * .3]
    np.testing.assert_allclose(np.asarray(record['transform'])[:3, 3], expected, atol=1e-16, rtol=0)


def test_behind_camera_fallback_keeps_population_and_no_label_field_is_accepted():
    inputs = synthetic_inputs(); inputs['q_to_k_transform'][2, 3] = -2.2
    raw, record = N.reference_observations(inputs, 'tx_plus01')
    assert record['predicted_in_front_count'] == 0
    assert raw['q_to_k_xy'].shape == (27648, 2)
    assert np.isfinite(raw['q_to_k_xy']).all() and np.isfinite(raw['q_to_k_pixel_cov']).all()
    assert np.abs(raw['q_to_k_xy']).max() <= 1024
    with pytest.raises(ValueError, match='ten forward fields'):
        N.reference_observations(dict(synthetic_inputs(), gt_visible=np.ones(27648, bool)), 'zero')
    broken = synthetic_inputs(); broken['q_to_k_xy'][0, 0] += .1
    with pytest.raises(ValueError, match='zero reprojection'):
        N.reference_observations(broken, 'zero')


def test_geometry_isotropic_keeps_original_attenuation_covariance():
    covariance=np.array([[[4.,.5],[.5,1.]],[[.25,-.1],[-.1,2.]]])
    iso=N.trace_covariance({'cov':covariance},'trace_isotropic_matched')
    np.testing.assert_array_equal(iso,np.array([2.5*np.eye(2),1.125*np.eye(2)]))
    np.testing.assert_array_equal(N.trace_covariance({'cov':covariance},'trace_full'),covariance)
    assert not np.array_equal(iso,covariance)
    with pytest.raises(ValueError):N.trace_covariance({'cov':covariance},'gaussian_full1')


@pytest.mark.parametrize('mutation',['missing','duplicate','isotropic_h','invalid_h','invalid_family','bool_temperature','constant_missing'])
def test_manifest_requires_all_nine_families_and_both_matched_controls(mutation):
    operators=synthetic_operators()
    if mutation=='missing':operators.pop('o07')
    elif mutation=='duplicate':operators['o09']=dict(operators['o02'])
    elif mutation=='isotropic_h':operators['o01']['h']=1.
    elif mutation=='invalid_h':operators['o02']['h']=.7
    elif mutation=='invalid_family':operators['o02']['family']='new_fitted_kernel'
    elif mutation=='bool_temperature':operators['o02']['temperature']=False
    else:operators['o05']['h']=1.  # Full1 must also include matched h/T.
    with pytest.raises(ValueError):N.validate_operators(operators)


def actual_prepared(family='trace_full',T=10.,far=False):
    """Tiny synthetic actual path; no filesystem data or real predictions."""
    import torch
    from models import round15_validation_pe as G
    from models import round17_trace_pe as P
    native=np.array([[0,0,8,8],[8,0,16,8],[16,0,40,24]],np.int64)
    bounds=(native-.5)/8
    tau_cov=np.tile(np.array([[[.37,.12],[.12,.19]]],np.float32),(8,1,1))
    mean=np.tile(np.array([[.41,.33]],np.float32),(8,1))
    if far:mean[:]=1000.
    geometry_cov=(tau_cov if family=='trace_full' else N.trace_covariance({'cov':tau_cov},family))
    content=np.random.default_rng(94).normal(size=(8,3)).astype(np.float32)
    C=torch.from_numpy(tau_cov);F=torch.from_numpy(content);h=.25
    key=G._key_cells(torch.from_numpy(native),'gaussian_full1')
    raw=G._gaussian_mass(torch.from_numpy(mean),torch.from_numpy(geometry_cov).double(),key,h).numpy()
    area=np.prod(bounds[:,2:]-bounds[:,:2],axis=1)
    logits=torch.from_numpy(np.log(raw+1e-30*area))
    probability=P.probabilities(logits,F,C,h,T).numpy()
    record=dict(indices=np.arange(8,dtype=np.int64),mean=mean,cov=geometry_cov,tau_cov=tau_cov,
        content=content,raw=raw,tau32=P.trace_temperature(C,h,T).numpy(),
        bias32=P.content_bias(C,h,T,F).numpy(),probability=probability,
        full_probability_sha256='a'*64,committed_probability_sha256='a'*64,
        recomputed_not_original_intermediates=True)
    return record,probability,bounds,h,T


@pytest.mark.parametrize('family',['trace_full','trace_isotropic_matched'])
@pytest.mark.parametrize('temperature',[0.,10.])
def test_real_synthetic_torch_arithmetic_matches_independent_same_mass_and_hybrid(family,temperature):
    record,probability,bounds,h,T=actual_prepared(family,temperature)
    result=N.prepared_check(record,probability,record['indices'],bounds,family,h,T)
    assert result['same_prepared_tau32_exact'] and result['same_prepared_bias32_exact']
    assert result['same_mass_probability_error']<1e-12
    assert result['hybrid_max_probability_error']<1e-12
    assert result['same_prepared_passed'] and result['hybrid_passed']
    assert result['intermediates_recomputed_from_locked_production']
    assert not result['intermediates_saved_by_original_forward'] and not result['actual_preparation_is_independent']


def test_zero_fov_area_floor_and_argmax_population_not_filtered():
    record,probability,bounds,h,T=actual_prepared(far=True)
    result=N.prepared_check(record,probability,record['indices'],bounds,'trace_full',h,T)
    assert result['same_prepared_passed'] and result['hybrid_passed']
    assert result['hybrid_reference_zero_fov_rows']==8
    assert result['hybrid_reference_tiny_fov_rows']==8
    assert len(result['hybrid_reference_raw_fov_mass'])==8
    assert np.any(probability[:,2]>probability[:,0])


@pytest.mark.parametrize('field',['tau32','bias32'])
def test_exact_intermediate_rule_detects_single_ulp_error(field):
    record,probability,bounds,h,T=actual_prepared()
    value=record[field].flat[0]
    record[field].flat[0]=np.nextafter(value,np.float32(np.inf))
    result=N.prepared_check(record,probability,record['indices'],bounds,'trace_full',h,T)
    assert not result['same_prepared_passed']
    assert not result['same_prepared_'+field+'_exact']
    assert result['same_mass_probability_error']<1e-12


@pytest.mark.parametrize('kind',['hash','stored_claim','index','dtype','geometry','probability','extra_label'])
def test_shared_preparation_cannot_hide_changed_contract(kind):
    record,probability,bounds,h,T=actual_prepared('trace_isotropic_matched')
    if kind=='hash':record['committed_probability_sha256']='b'*64
    elif kind=='stored_claim':record['recomputed_not_original_intermediates']=False
    elif kind=='index':record['indices'][0]=99
    elif kind=='dtype':record['tau_cov']=record['tau_cov'].astype(np.float64)
    elif kind=='geometry':record['cov'][0,0,0]+=.01
    elif kind=='probability':record['probability']=np.roll(probability,1,axis=1)
    else:record['sensor_target']=np.zeros(8)
    with pytest.raises(ValueError):N.prepared_check(record,probability,np.arange(8,dtype=np.int64),bounds,'trace_isotropic_matched',h,T)


def test_all_68_conditions_with_all_operators_keep_failures_and_tiny_fov(monkeypatch):
    inputs=synthetic_inputs();parts=PARTS.literal_parts();operators=synthetic_operators()
    operators['o09']=dict(family='gaussian_point',h=1.,temperature=0.)
    calls=[]
    def quartic_mass(mean,cov,bounds,h,order):
        calls.append(order);mass=np.zeros((len(mean),len(bounds)),np.float64)
        if len(calls)==1:mass[0,0]=1e-29
        return mass
    def gaussian_mass(query,bounds,family,h):return np.zeros((len(query['mean']),len(bounds)),np.float64)
    def trace_mass(mean,cov,bounds,h,order):return np.zeros((len(mean),len(bounds)),np.float64)
    seen=[]
    def loader(scenario,condition,identifier):
        seen.append((scenario,condition,identifier));q,k=condition.split('_')
        p=np.tile(parts[k]['mass']/parts[k]['mass'].sum(),(len(parts[q]['mass']),1))
        if (scenario,condition,identifier)==('zero','g8_g8','o03'):p[0]=0.;p[0,0]=1.
        return p
    def prepared(scenario,condition,identifier):
        q,k=condition.split('_');family=operators[identifier]['family']
        covariance=np.tile(np.array([[[.3,.1],[.1,.2]]],np.float32),(8,1,1))
        geometry=covariance if family=='trace_full' else N.trace_covariance({'cov':covariance},family)
        content=np.ones((8,len(parts[k]['mass'])),np.float32)
        tau=N.TRACE.independent_temperature(covariance,.5,5.).astype(np.float32)
        probability=np.tile(parts[k]['mass']/parts[k]['mass'].sum(),(8,1))
        return dict(indices=np.linspace(0,len(parts[q]['mass'])-1,8,dtype=np.int64),
            mean=np.zeros((8,2),np.float32),cov=geometry,tau_cov=covariance,content=content,
            raw=np.zeros_like(probability),tau32=tau,bias32=tau[:,None]*content,probability=probability,
            full_probability_sha256='a'*64,committed_probability_sha256='a'*64,
            recomputed_not_original_intermediates=True)
    monkeypatch.setattr(N.REF.QREF,'independent_mass',quartic_mass)
    monkeypatch.setattr(N.REF,'gaussian_reference_mass',gaussian_mass)
    monkeypatch.setattr(N.TRACE,'independent_mass',trace_mass)
    result=N.audit_pair(inputs,parts,loader,operators,prepared_loader=prepared)
    assert len(seen)==len(set(seen))==result['matrix_count']==680
    assert result['query_rows']==5440 and len(result['projection_records'])==7
    assert result['trace_matrix_count']==136 and result['quartic_matrix_count']==68
    assert result['gaussian_matrix_count']==476 and result['numerical_failure_count']==2
    assert not result['passed']
    assert all(c['reference_zero_fov_rows']==8 for c in result['checks'])
    assert result['same_prepared_tau32_exact'] and result['same_prepared_bias32_exact']
    assert result['matched_original_tau_cov_and_tau32_exact']
    assert not result['actual_preparation_is_independent'] and not result['scientific_gate_passed']
    assert not result['labels_decoded'] and not result['configuration_selection']
    assert len([c for c in result['checks'] if (c['reference_convergence_error'] or 0)>1e-8])==1
    failure=next(c for c in result['checks'] if c['max_probability_error']>1e-4)
    assert failure['argmax_change_count']==0


def test_corruption_outside_eight_selected_queries_is_rejected():
    inputs=synthetic_inputs();parts=PARTS.literal_parts()
    def loader(scenario,condition,identifier):
        q,k=condition.split('_');p=np.full((len(parts[q]['mass']),len(parts[k]['mass'])),1/len(parts[k]['mass']))
        p[1,0]=np.nan;return p
    with pytest.raises(ValueError,match='invalid complete native sample'):
        N.audit_pair(inputs,parts,loader,synthetic_operators(),prepared_loader=lambda *_:None)


def test_source_closure_binds_all_old_and_trace_references_no_new_producer_import():
    import ast
    from pathlib import Path
    for sources in (N.REF.SOURCE_NAMES,N.STRESS.AUDIT_SOURCES,N.TRACE.SOURCE_NAMES):
        assert set(sources).issubset(N.SOURCE_NAMES)
    tree=ast.parse(Path(N.__file__).read_text())
    imported=[alias.name for node in ast.walk(tree) if isinstance(node,ast.ImportFrom) for alias in node.names]
    assert not any(name in imported for name in ('round17_validation','round17_validation_metrics','round17_validation_pe'))
