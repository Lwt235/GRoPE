"""Actual partition conservation, perspective moments and sensor isolation."""
import inspect
from pathlib import Path

import numpy as np
import pytest

from experiments.round10_partition_data import (
    PARTITIONS, choose_pairs, containing_partition, dense_observations, export,
    native_support, partition_bounds, partition_index, partition_moments,
    pixel_centers, pool_base_features, projected_3d_moments,
)


def cache(depth=None, pose=None):
    rng = np.random.default_rng(7)
    return dict(predicted_depth=np.full((144,192), 2.) if depth is None else depth,
        features=rng.normal(size=(432,128)).astype(np.float32),
        K=np.array([[100.,3.,95.5],[0.,110.,71.5],[0.,0.,1.]]),
        c2w=np.eye(4) if pose is None else pose)


@pytest.mark.parametrize('name', PARTITIONS)
def test_true_partitions_cover_same_pixels_and_only_merge_within_parents(name):
    bounds = partition_bounds(name)
    lookup = partition_index(bounds)
    centers, covariance, area = native_support(bounds)
    np.testing.assert_array_equal(np.bincount(lookup.ravel()), area)
    assert area.sum() == 192*144
    assert np.unique(lookup).size == len(bounds)
    assert np.all(bounds[:, :2]//16 == (bounds[:, 2:]-1)//16)
    assert np.linalg.eigvalsh(covariance).min() > 0
    assigned, inside = containing_partition(pixel_centers(), bounds)
    assert inside.all()
    np.testing.assert_array_equal(assigned, lookup.ravel())
    at_centers, valid = containing_partition(centers, bounds)
    assert valid.all()
    np.testing.assert_array_equal(at_centers, np.arange(len(bounds)))


def test_mixed_has_all_four_real_cell_shapes():
    bounds = partition_bounds('mixed')
    assert set(map(tuple,bounds[:,2:]-bounds[:,:2])) == {(8,8),(16,16),(8,16),(16,8)}
    lookup = partition_index(bounds)
    # Actual distinct source pixels, never duplicated query rows.
    samples = [set(np.flatnonzero(lookup.ravel()==i)) for i in range(len(bounds))]
    assert sum(map(len,samples)) == len(set.union(*samples)) == 27648


@pytest.mark.parametrize('bad', [np.array([[0,0,8,8]]),
    np.array([[0,0,16,16],[0,0,16,16]]), np.array([[0,0,16.1,16]])])
def test_invalid_partitions_rejected(bad):
    with pytest.raises(ValueError): partition_index(bad,(16,16))


def test_boundary_half_open_tie_and_nonfinite_contract():
    bounds = partition_bounds('g8',(16,16))
    index, valid = containing_partition(np.array([[7.5,0],[7.5-1e-12,0],[7.5-1e-7,0],
        [15.5,0],[-.5,0],[np.nan,0]]),bounds,(16,16))
    np.testing.assert_array_equal(index,[1,1,0,-1,0,-1])
    np.testing.assert_array_equal(valid,[True,True,True,False,True,False])


@pytest.mark.parametrize('name', PARTITIONS)
def test_constant_plane_identity_recovers_continuous_native_rectangle(name):
    inputs = dense_observations(cache(),cache())
    bounds = partition_bounds(name)
    native_m,native_c,_ = native_support(bounds)
    dense_m,dense_c = partition_moments(inputs['q_to_k_xy'],inputs['q_to_k_pixel_cov'],bounds)
    linear_m,linear_c,in_front = projected_3d_moments(inputs,bounds)
    assert in_front.all() and inputs['q_to_k_in_front'].all()
    np.testing.assert_allclose(dense_m,native_m,atol=1e-12,rtol=0)
    np.testing.assert_allclose(dense_c,native_c,atol=1e-12,rtol=0)
    np.testing.assert_allclose(linear_m,native_m,atol=1e-12,rtol=0)
    np.testing.assert_allclose(linear_c,native_c,atol=1e-12,rtol=0)


def test_mixture_moments_reassemble_children_with_parallel_axis_term():
    rng=np.random.default_rng(4)
    xy=rng.normal(size=(256,2)); a=rng.normal(size=(256,2,2)); cov=a@a.transpose(0,2,1)
    small=partition_bounds('g8',(16,16)); large=partition_bounds('g16',(16,16))
    cm,cc=partition_moments(xy,cov,small,(16,16))
    pm,pc=partition_moments(xy,cov,large,(16,16))
    delta=cm-cm.mean(0)
    np.testing.assert_allclose(pm[0],cm.mean(0),atol=1e-15)
    np.testing.assert_allclose(pc[0],cc.mean(0)+delta.T@delta/4,atol=1e-15)
    # Averaging child covariance alone loses between-child support.
    assert np.linalg.norm(pc[0]-cc.mean(0))>1e-4


def test_perspective_projection_and_moment_fitting_do_not_commute():
    depth=np.full((144,192),2.);depth[:,4::8]=4.
    pose=np.eye(4);pose[0,3]=.2;pose[2,3]=.4
    inputs=dense_observations(cache(depth),cache(pose=pose))
    bounds=partition_bounds('g16')
    dm,dc=partition_moments(inputs['q_to_k_xy'],inputs['q_to_k_pixel_cov'],bounds)
    jm,jc,_=projected_3d_moments(inputs,bounds)
    assert np.max(np.abs(dm-jm))>.1
    assert np.max(np.abs(dc-jc))>.1


def test_finite_predicted_fallback_keeps_all_pixels_without_sensor_masks():
    depth=np.full((144,192),2.);depth.flat[:4]=[np.nan,np.inf,-np.inf,0]
    pose=np.eye(4);pose[2,3]=3.
    q,k=cache(depth),cache(pose=pose)
    clean=dense_observations(q,k)
    q.update(sensor_depth=np.full((144,192),np.nan),gt_visible=np.zeros(27648,bool))
    k.update(sensor_depth=np.zeros((144,192)),gt_visible=np.zeros(27648,bool))
    poisoned=dense_observations(q,k)
    for field in clean:
        np.testing.assert_array_equal(clean[field],poisoned[field])
    assert not clean['q_to_k_in_front'].all()
    assert len(clean['q_to_k_xy'])==27648
    assert np.isfinite(clean['q_to_k_xy']).all() and np.isfinite(clean['q_to_k_pixel_cov']).all()
    bounds=partition_bounds('g16')
    m,c=partition_moments(clean['q_to_k_xy'],clean['q_to_k_pixel_cov'],bounds)
    assert len(m)==108 and np.isfinite(c).all()
    assert 'gt' not in inspect.signature(dense_observations).parameters


def test_native_feature_area_pool_has_parent_mass_conservation():
    rng=np.random.default_rng(9);features=rng.normal(size=(432,3))
    parent=pool_base_features(features,partition_bounds('g16'),normalize=False)
    mixed=partition_bounds('mixed');small=pool_base_features(features,mixed,normalize=False)
    centers,_,area=native_support(mixed)
    indices=(centers[:,1].astype(int)//16)*12+centers[:,0].astype(int)//16
    recon=np.zeros_like(parent)
    for i in range(len(mixed)):recon[indices[i]]+=area[i]*small[i]/256
    np.testing.assert_allclose(parent,recon,atol=1e-7)
    np.testing.assert_allclose(pool_base_features(features,partition_bounds('g8')),
        features/np.linalg.norm(features,axis=-1,keepdims=True),atol=1e-7)


def test_fixed_metadata_cohort_keeps_both_offsets_and_rejects_unseen_scans():
    row=dict(scene='harvard_c3/hv_c3_1',split='train',
             pairs=[[q,q+d] for q in range(30,1030,10) for d in (15,60)])
    pairs=choose_pairs(row,8)
    assert len(pairs)==16 and pairs[0]==[30,45] and pairs[-1]==[1020,1080]
    assert {k-q for q,k in pairs}=={15,60}
    with pytest.raises(ValueError):choose_pairs(dict(row,scene='secret/test'),8)
    with pytest.raises(ValueError):choose_pairs(dict(row,split='test'),8)
    with pytest.raises(ValueError):choose_pairs(row,101)


def test_export_refuses_existing_path_before_reading_any_source(tmp_path):
    (tmp_path/'marker').write_text('preserve')
    with pytest.raises(ValueError,match='refuse overwrite'):export(tmp_path)
    assert (tmp_path/'marker').read_text()=='preserve'


def test_export_forward_archives_are_written_before_first_sensor_open():
    source=inspect.getsource(export)
    assert source.index("out/'forward_complete.json'")<source.index('load_sensor_depth(')
    assert source.index("dest/'inputs.npz'")<source.index('load_sensor_depth(')
