"""Region-area conservation, mixed grids, physical support, and GT isolation."""
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.round5_rgbd_data import (
    QUERY_DEPTH_MISSING, KEY_DEPTH_MISSING, DEPTH_INCONSISTENT, OUTSIDE_IMAGE,
    pair_supervision, sha256,
)
from experiments.round6_region_data import (
    make_grid, containing_cells, region_supervision, fit_predicted_support,
    projected_support, pool_features, prepare_frame, make_pair_inputs,
    _authorize_splits,
)


def camera(hw=(32, 32)):
    h, w = hw
    return np.array([[100., 0., (w-1)/2], [0., 100., (h-1)/2], [0., 0., 1.]])


@pytest.mark.parametrize('qs,ks', [(8,8),(8,16),(16,8),(16,16)])
def test_identity_area_mass_all_resolutions(qs, ks):
    hw = (32, 32)
    depth = np.full((64, 64), 2.)
    K = camera(depth.shape)
    labels = region_supervision(depth, depth, K, K, np.eye(4), np.eye(4), qs, ks, hw)
    qgrid, kgrid = make_grid(qs, hw), make_grid(ks, hw)
    assert labels.soft_labels.shape == (qgrid.token_count, kgrid.token_count)
    assert labels.gt_valid.all() and labels.point_gt_valid.all()
    np.testing.assert_array_equal(labels.sample_count, np.full(qgrid.token_count, qs**2))
    np.testing.assert_array_equal(labels.target_counts.sum(-1), labels.sample_count)
    np.testing.assert_array_equal(labels.visible_fraction, np.ones(qgrid.token_count))
    np.testing.assert_array_equal(labels.counts_by_reason[:, 0], labels.sample_count)
    assert labels.target_counts.sum() == np.prod(hw)
    np.testing.assert_allclose(labels.point_pixcontinuous, qgrid.centers, atol=1e-6)
    if qs <= ks:
        expected_index, _ = containing_cells(qgrid.centers, kgrid)
        expected = np.eye(kgrid.token_count)[expected_index]
        np.testing.assert_array_equal(labels.soft_labels, expected)
    else:
        assert (labels.soft_labels == .25).sum() == qgrid.token_count*4
        assert (labels.soft_labels != 0).sum() == qgrid.token_count*4


def test_known_half_patch_motion_splits_source_area_not_center_point():
    hw = (32, 32)
    depth = np.full(hw, 2.)
    key_pose = np.eye(4)
    key_pose[0, 3] = -.08  # z2,f100 -> +4px target motion.
    labels = region_supervision(depth, depth, camera(hw), camera(hw), np.eye(4), key_pose, 8, 8, hw)
    np.testing.assert_array_equal(labels.target_counts[0, :2], [32, 32])
    np.testing.assert_array_equal(labels.soft_labels[0, :2], [.5, .5])
    assert labels.point_key_index[0] == 1  # The independent center lies at x7.5.
    assert labels.gt_valid.all()
    # Last-column regions retain half of their mass despite invalid center points.
    np.testing.assert_array_equal(labels.visible_fraction.reshape(4, 4)[:, -1], [.5]*4)
    assert not labels.point_gt_valid.reshape(4, 4)[:, -1].any()
    np.testing.assert_array_equal(labels.counts_by_reason.reshape(4, 4, -1)[:, -1, OUTSIDE_IMAGE], [32]*4)


def test_sensor_holes_are_unknown_and_center_mask_does_not_filter_area():
    depth = np.full((32, 32), 2.)
    query = depth.copy()
    query[4, 4] = 0
    labels = region_supervision(query, depth, camera(), camera(), np.eye(4), np.eye(4), 8, 8, depth.shape)
    assert labels.gt_valid[0] and not labels.point_gt_valid[0]
    assert labels.visible_fraction[0] == 63/64
    assert labels.target_counts[0, 0] == 63 and labels.soft_labels[0, 0] == 1
    assert labels.counts_by_reason[0, QUERY_DEPTH_MISSING] == 1
    assert labels.point_key_index[0] == -1
    key = depth.copy()
    key[:8, :8] = 0
    labels = region_supervision(depth, key, camera(), camera(), np.eye(4), np.eye(4), 8, 8, depth.shape)
    assert not labels.gt_valid[0]
    assert labels.counts_by_reason[0, KEY_DEPTH_MISSING] == 64
    assert not labels.soft_labels[0].any() and labels.visible_fraction[0] == 0


def test_depth_mismatch_has_own_reason_and_no_fabricated_mass():
    depth = np.full((32, 32), 2.)
    labels = region_supervision(depth, np.ones_like(depth), camera(), camera(),
        np.eye(4), np.eye(4), 8, 16, depth.shape)
    assert not labels.gt_valid.any() and not labels.soft_labels.any()
    np.testing.assert_array_equal(labels.counts_by_reason[:, DEPTH_INCONSISTENT], [64]*16)
    np.testing.assert_array_equal(labels.counts_by_reason.sum(-1), labels.sample_count)


def test_target_cell_half_open_boundary_and_tie_roundoff():
    grid = make_grid(8, (32, 32))
    index, inside = containing_cells(np.array([[7.5, 0], [7.5-1e-12, 0],
                                               [7.5-1e-7, 0], [31.5, 0]]), grid)
    np.testing.assert_array_equal(index, [1, 1, 0, -1])
    np.testing.assert_array_equal(inside, [1, 1, 1, 0])


@pytest.mark.parametrize('stride', [8,16])
def test_continuous_footprint_recovers_exact_constant_plane_rectangle(stride):
    hw = (32, 32)
    K = camera(hw)
    K[0, 1] = 3.  # Generic intrinsics including skew.
    mean, covariance, _ = fit_predicted_support(np.full(hw, 2.), K, stride)
    xy, C, _ = projected_support(mean, covariance, np.eye(4), np.eye(4), K)
    grid = make_grid(stride, hw)
    np.testing.assert_allclose(xy, grid.centers/8, atol=1e-7)
    expected = np.broadcast_to(np.eye(2)*stride**2/(12*64), C.shape)
    np.testing.assert_allclose(C, expected, rtol=1e-7, atol=1e-8)


def frame_cache(seed=2):
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(432,128)).astype(np.float32)
    features /= np.linalg.norm(features, axis=-1, keepdims=True)
    return {'features':features, 'predicted_depth':np.full((144,192),2.,np.float32),
            'K':camera((144,192)), 'c2w':np.eye(4)}


def test_feature_area_pool_then_l2_and_full_image_coverage():
    cache = frame_cache()
    np.testing.assert_array_equal(pool_features(cache['features'],8), cache['features'])
    features16 = pool_features(cache['features'],16)
    assert features16.shape == (108,128)
    first = cache['features'][[0,1,24,25]].astype(np.float64).mean(0)
    first /= np.linalg.norm(first)
    np.testing.assert_allclose(features16[0], first, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(np.linalg.norm(features16,axis=-1),1,atol=1e-7)
    grid16 = make_grid(16)
    assert grid16.token_count == 108
    np.testing.assert_array_equal(grid16.bounds[-1,2:], [191.5,143.5])


def test_mixed_resolution_forward_shapes_and_common_reference_units():
    inputs, diag = make_pair_inputs(frame_cache(), frame_cache(3), 8, 16)
    assert set(inputs) == {'q','k','qmean','qcov','kmean','kcov','k_to_q','k_to_qcov','q_to_k','q_to_kcov'}
    for name in ['q','qmean','qcov','q_to_k','q_to_kcov']:
        assert len(inputs[name]) == 432
    for name in ['k','kmean','kcov','k_to_q','k_to_qcov']:
        assert len(inputs[name]) == 108
    np.testing.assert_allclose(inputs['qcov'][:,0,0],1/12)
    np.testing.assert_allclose(inputs['kcov'][:,0,0],1/3)
    np.testing.assert_allclose(inputs['kmean'][0],[7.5/8,7.5/8])
    np.testing.assert_allclose(inputs['k_to_q'],inputs['kmean'],atol=1e-6)
    np.testing.assert_allclose(inputs['k_to_qcov'],inputs['kcov'],atol=1e-7)
    assert not diag['query_prediction']['gt_or_sensor_inputs_used']


def test_nonfinite_predicted_depth_uses_fixed_fallback_without_dropping_tokens():
    q, k = frame_cache(), frame_cache(3)
    q['predicted_depth'][:] = np.nan
    k['predicted_depth'][:] = -1
    inputs, diag = make_pair_inputs(q,k,16,8)
    assert len(inputs['q']) == 108 and len(inputs['k']) == 432
    assert all(np.isfinite(value).all() for value in inputs.values())
    assert diag['query_prediction']['nonfinite_depth_pixels'] == 144*192
    assert diag['key_prediction']['depth_pixels_clipped'] == 144*192
    assert not any('mask' in key or 'valid' in key for key in inputs)


def test_gt_buffer_mutation_cannot_change_forward_inputs():
    q,k = frame_cache(),frame_cache(3)
    before,_ = make_pair_inputs(q,k,8,16)
    depth = np.full((144,192),2.)
    good = region_supervision(depth,depth,q['K'],k['K'],q['c2w'],k['c2w'],8,16)
    depth[:] = 0
    unknown = region_supervision(depth,depth,q['K'],k['K'],q['c2w'],k['c2w'],8,16)
    assert good.gt_valid.all() and not unknown.gt_valid.any()
    after,_ = make_pair_inputs(q,k,8,16)
    for key in before:
        np.testing.assert_array_equal(before[key],after[key])
    signature = inspect.signature(make_pair_inputs)
    assert set(signature.parameters) == {'q','k','q_stride','k_stride'}


@pytest.mark.parametrize('qs,ks',[(8,16),(16,8)])
def test_independent_center_targets_match_frozen_round5_math(qs,ks):
    depth=np.full((32,32),2.)
    depth[4,4]=0
    reference=pair_supervision(depth,depth,camera(),camera(),np.eye(4),np.eye(4),depth.shape,qs)
    region=region_supervision(depth,depth,camera(),camera(),np.eye(4),np.eye(4),qs,ks,depth.shape)
    np.testing.assert_array_equal(region.point_gt_valid,reference.gt_valid)
    np.testing.assert_array_equal(region.point_pixcontinuous,reference.pixcontinuous)


def test_test_export_requires_frozen_selection_and_source_lock(tmp_path):
    import experiments.round6_region_data as module
    selection=tmp_path/'selection.json'
    selection.write_text('{}')
    args=SimpleNamespace(splits=['train','val'],allow_test=False,protocol_lock=None)
    assert _authorize_splits(args,selection) is None
    args.splits=['test']
    with pytest.raises(ValueError,match='frozen'):
        _authorize_splits(args,selection)
    args.allow_test=True
    args.protocol_lock=tmp_path/'lock.json'
    args.protocol_lock.write_text(json.dumps({'selection_sha256':sha256(selection),
        'exporter_sha256':sha256(module.__file__),'test_export_authorized':True}))
    assert _authorize_splits(args,selection)['sha256']==sha256(args.protocol_lock)
    selection.write_text('{"changed":true}')
    with pytest.raises(ValueError,match='exact'):
        _authorize_splits(args,selection)
