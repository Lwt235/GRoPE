"""Truth interventions preserve geometry units, partial sensor coverage and masks."""
import numpy as np
import pytest

from experiments.round5_rgbd_data import OUTSIDE_IMAGE, KEY_DEPTH_MISSING
from experiments.round6_region_data import make_grid
from experiments.round8_oracle_data import (
    ALLOWLIST, bilinear_correspondence, oracle_correspondence,
    sensor_patch_points, transformed_value_depth, selected_rows,
    bilinear_depth, direct_corresponding_depths,
)


def camera(hw=(16, 16)):
    h, w = hw
    return np.array([[100., 0., (w-1)/2], [0., 100., (h-1)/2], [0., 0., 1.]])


def dense(indices, weights, count):
    result = np.zeros((len(indices), count))
    for row in range(len(indices)):
        np.add.at(result[row], indices[row], weights[row])
    return result


def test_exact_centers_and_half_cell_bilinear():
    pixels = np.concatenate((make_grid(8, (16, 16)).centers, [[7.5, 7.5]]))
    indices, weights = bilinear_correspondence(pixels, np.ones(5, bool), (16, 16))
    actual = dense(indices, weights, 4)
    np.testing.assert_array_equal(actual[:4], np.eye(4))
    np.testing.assert_allclose(actual[4], .25)
    np.testing.assert_allclose(weights.sum(-1), 1.)


def test_border_repetition_and_invalid_safe_rows():
    pixels = np.array([[-.49, -.49], [15.49, 15.49], [15.49, 7.5], [np.nan, np.nan]])
    indices, weights = bilinear_correspondence(pixels, [True, True, True, False], (16, 16))
    actual = dense(indices, weights, 4)
    np.testing.assert_array_equal(actual[0], [1, 0, 0, 0])
    np.testing.assert_array_equal(actual[1], [0, 0, 0, 1])
    np.testing.assert_array_equal(actual[2], [0, .5, 0, .5])
    assert len(set(indices[1])) == 1
    np.testing.assert_array_equal(actual[3], 0.)
    with pytest.raises(ValueError, match='finite'):
        bilinear_correspondence([[np.nan, 0]], [True], (16, 16))
    with pytest.raises(ValueError, match='inside'):
        bilinear_correspondence([[15.5, 0]], [True], (16, 16))


def test_native_rescaled_identity_correspondence():
    depth = np.full((32, 32), 2.)
    result = oracle_correspondence(depth, depth, camera(depth.shape), camera(depth.shape),
                                  np.eye(4), np.eye(4), (16, 16))
    assert result['oracle_valid'].all()
    np.testing.assert_allclose(result['oracle_pixel'], make_grid(8, (16, 16)).centers, atol=1e-12)
    np.testing.assert_allclose(dense(result['oracle_indices'], result['oracle_weights'], 4), np.eye(4), atol=1e-12)


def test_translated_correspondence_and_no_visibility_population_filter():
    depth = np.full((16, 16), 2.)
    keypose = np.eye(4); keypose[0, 3] = -.16  # u_key = u_query + 8 pixels.
    result = oracle_correspondence(depth, depth, camera(), camera(), np.eye(4), keypose, (16, 16))
    np.testing.assert_allclose(result['oracle_pixel'][:, 0], make_grid(8, (16, 16)).centers[:, 0]+8.)
    np.testing.assert_array_equal(result['oracle_valid'], [True, False, True, False])
    assert (result['oracle_status'][[1, 3]] == OUTSIDE_IMAGE).all()
    # Invalid source correspondence is recorded without removing a query row.
    assert len(result['oracle_valid']) == 4
    missing = oracle_correspondence(depth, np.zeros_like(depth), camera(), camera(),
                                   np.eye(4), np.eye(4), (16, 16))
    assert not missing['oracle_valid'].any()
    assert (missing['oracle_status'] == KEY_DEPTH_MISSING).all()
    assert not missing['oracle_weights'].any()


def test_partial_sensor_fallback_before_patch_3d_mean():
    depth = np.zeros((16, 16)); depth[:4, :8] = 4.
    pred = np.full((16, 16), 2.)
    points, fraction = sensor_patch_points(depth, pred, camera())
    np.testing.assert_array_equal(fraction, [.5, 0, 0, 0])
    np.testing.assert_allclose(points[:, 2], [3., 2., 2., 2.])
    # Asymmetric sensor coverage changes y mean: use every backprojected pixel.
    pixels = make_grid(1, (16, 16)).centers
    mixed = np.where(depth > 0, depth, pred)
    ray = np.column_stack((pixels, np.ones(len(pixels)))) @ np.linalg.inv(camera()).T
    expected = (ray*mixed.reshape(-1, 1)).reshape(2, 8, 2, 8, 3).mean((1, 3)).reshape(-1, 3)
    np.testing.assert_allclose(points, expected)
    keypose = np.eye(4); keypose[2, 3] = 1.
    np.testing.assert_allclose(transformed_value_depth(points, keypose, np.eye(4)), [4., 3., 3., 3.])


def test_rotation_camera_transform_and_clamps():
    pose = np.eye(4); pose[:3, :3] = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
    points = np.array([[-2., 0., 3.], [2., 0., 3.], [-100., 0., 3.]])
    np.testing.assert_allclose(transformed_value_depth(points, pose, np.eye(4)), [2., .1, 20.])
    sensor = np.zeros((16, 16)); sensor[:8, :8] = 100.; sensor[8:, :8] = np.nan
    pred = np.full((16, 16), np.nan); pred[:8, 8:] = np.inf; pred[8:, 8:] = -np.inf
    result, fraction = sensor_patch_points(sensor, pred, camera())
    np.testing.assert_allclose(result[:, 2], [20., 20., 10., .1])
    np.testing.assert_array_equal(fraction, [1., 0., 0., 0.])


def test_direct_pixel_values_separate_pooling_and_preserve_invalid_fallback():
    depth = np.indices((16, 16))[1]/100.+2.
    pixels = np.array([[3.5, 3.5], [7.5, 7.5], [-.4, 4.], [np.nan, np.nan]])
    valid = np.array([True, True, True, False])
    fallback = np.array([9., 9., 9., 7.], np.float32)
    keypose = np.eye(4); keypose[2, 3] = 1.
    result = direct_corresponding_depths(pixels, valid, depth, depth, camera(), camera(),
                                        keypose, np.eye(4), fallback)
    np.testing.assert_allclose(result['direct_corresponding_pred_depth'], [3.035, 3.075, 3., 7.])
    np.testing.assert_allclose(result['direct_corresponding_sensor_depth'], [3.04, 3.08, 3., 7.])
    np.testing.assert_allclose(bilinear_depth(depth, [[15.4, 4.], [-.4, 4.]]), [2.15, 2.])


def test_direct_sensor_native_resize_and_visibility_edge_clip():
    native = np.full((32, 32), 2.)
    pred = np.full((16, 16), 3.)
    pix = make_grid(8, (16, 16)).centers
    result = direct_corresponding_depths(pix, np.ones(4, bool), pred, native,
        camera(), camera(native.shape), np.eye(4), np.eye(4), np.ones(4))
    np.testing.assert_array_equal(result['direct_corresponding_sensor_depth'], 2.)
    np.testing.assert_array_equal(result['direct_corresponding_pred_depth'], 3.)
    pose = np.eye(4); pose[0, 3] = -.02  # Shift target pixels +1: last column outside center hull but inside image.
    corr = oracle_correspondence(np.full((16, 16), 2.), np.full((16, 16), 2.),
        camera(), camera(), np.eye(4), pose, (16, 16))
    assert corr['oracle_valid'].all()
    np.testing.assert_array_equal(corr['route_clipped'], [False, True, False, True])


def test_fixed_allowlist_has_no_test_and_cli_rejects_arbitrary_data():
    assert len(ALLOWLIST) == 5
    assert {x[1] for x in ALLOWLIST.values()} == {'train', 'val'}
    assert sum(x[2] for x in ALLOWLIST.values()) == 704
    for invalid in (['test'], [], ['old', 'old'], ['round7_test']):
        with pytest.raises(ValueError, match='fixed old development'):
            selected_rows(invalid)
