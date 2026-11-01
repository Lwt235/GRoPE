"""Independent input-only support, physical-unit and projection checks."""
import numpy as np
from PIL import Image
import torch
from torch import nn

from experiments.round5_cache import frame_inputs, projected_support, resize_K, support_from_prediction
from experiments.round5_pairs import input_pair
from experiments.round5_rgbd_data import load_sensor_depth, pair_supervision, token_grid


def camera():
    return np.array([[180., 0., 95.5], [0., 175., 71.5], [0., 0., 1.]])


def rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    return np.eye(3)+np.sin(angle)*skew+(1-np.cos(angle))*(skew @ skew)


def true_projection(points_source, source_c2w, target_c2w, K):
    # Independently transform homogeneous points, then project and convert units.
    homogeneous = np.column_stack([points_source, np.ones(len(points_source))])
    target = (np.linalg.solve(target_c2w, source_c2w @ homogeneous.T)).T[:, :3]
    pixel = target @ K.T
    return pixel[:, :2]/pixel[:, 2, None]/8


def test_constant_plane_support_has_exact_discrete_pixel_moments():
    K = camera()
    z = 2.3
    mean, covariance = support_from_prediction(np.full((144, 192), z), K)
    pixels = token_grid()
    expected_mean = np.column_stack([
        (pixels[:, 0]-K[0, 2])*z/K[0, 0],
        (pixels[:, 1]-K[1, 2])*z/K[1, 1], np.full(432, z)])
    expected_covariance = np.diag([(z/K[0, 0])**2*5.25,
                                  (z/K[1, 1])**2*5.25, 0.])
    np.testing.assert_allclose(mean, expected_mean, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(covariance,
        np.broadcast_to(expected_covariance, covariance.shape), rtol=1e-7, atol=1e-10)
    projected_mean, projected_covariance, diagnostics = projected_support(
        mean, covariance, np.eye(4), np.eye(4), K)
    np.testing.assert_allclose(projected_mean, pixels/8, rtol=1e-7, atol=2e-6)
    np.testing.assert_allclose(projected_covariance,
        np.broadcast_to(np.eye(2)*(5.25/64), (432, 2, 2)), rtol=2e-7, atol=1e-8)
    assert diagnostics == {'behind_or_near': 0, 'cov_clipped': 0, 'mean_clipped': 0}


def test_rotated_projection_covariance_matches_finite_difference_source_jacobian():
    source = np.eye(4)
    source[:3, :3] = rotation([.2, 1., -.1], .12)
    source[:3, 3] = [.08, -.04, .1]
    target = np.eye(4)
    target[:3, :3] = rotation([1., -.3, .4], -.09)
    target[:3, 3] = [-.04, .03, .02]
    mean = np.array([[.1, -.2, 2.], [-.4, .2, 3.], [.3, .1, 2.4]])
    factor = np.array([[.012, .003, 0.], [0., .009, .001], [.002, 0., .02]])
    covariance = np.broadcast_to(factor @ factor.T, (3, 3, 3))
    pixel, actual_covariance, diagnostics = projected_support(mean, covariance, source, target, camera())
    expected_pixel = true_projection(mean, source, target, camera())
    jacobian = np.empty((3, 2, 3))
    epsilon = 1e-5
    for coordinate in range(3):
        shift = np.eye(3)[coordinate]*epsilon
        jacobian[:, :, coordinate] = (true_projection(mean+shift, source, target, camera())
            - true_projection(mean-shift, source, target, camera()))/(2*epsilon)
    expected_covariance = jacobian @ covariance @ jacobian.transpose(0, 2, 1)
    np.testing.assert_allclose(pixel, expected_pixel, rtol=1e-7, atol=2e-6)
    np.testing.assert_allclose(actual_covariance, expected_covariance, rtol=2e-6, atol=2e-8)
    assert diagnostics == {'behind_or_near': 0, 'cov_clipped': 0, 'mean_clipped': 0}


def test_projection_is_invariant_to_world_frame_and_consistent_metric_rescaling():
    mean = np.array([[.1, -.2, 2.], [-.4, .2, 3.]])
    covariance = np.broadcast_to(np.diag([.002, .001, .006]), (2, 3, 3))
    source, target = np.eye(4), np.eye(4)
    target[:3, :3] = rotation([0., 1., .2], .08)
    target[:3, 3] = [.12, -.02, .03]
    expected = projected_support(mean, covariance, source, target, camera())
    world = np.eye(4)
    world[:3, :3] = rotation([1., 2., -1.], .7)
    world[:3, 3] = [2., -4., 7.]
    moved = projected_support(mean, covariance, world @ source, world @ target, camera())
    for value, reference in zip(moved[:2], expected[:2]):
        np.testing.assert_allclose(value, reference, rtol=1e-7, atol=1e-7)
    for scale in (.5, 4.):
        scaled_source, scaled_target = source.copy(), target.copy()
        scaled_source[:3, 3] *= scale
        scaled_target[:3, 3] *= scale
        scaled = projected_support(mean*scale, covariance*scale**2,
            scaled_source, scaled_target, camera())
        for value, reference in zip(scaled[:2], expected[:2]):
            np.testing.assert_allclose(value, reference, rtol=1e-7, atol=1e-7)


def test_native_inputs_are_known_grid_even_when_depth_edge_moves_gaussian_centroid():
    predicted = np.ones((144, 192))
    predicted[:, 4:8] = 4.
    mean, covariance = support_from_prediction(predicted, camera())
    frame = dict(mean3=mean, cov3=covariance, c2w=np.eye(4), K=camera(),
                 features=np.ones((432, 128), dtype=np.float32))
    approximation, _, _ = projected_support(mean, covariance, np.eye(4), np.eye(4), camera())
    # Moment matching in 3D then linearizing projection need not recover the
    # exact native uniform patch: the first center moves from 3.5 to 4.7 pixels.
    assert approximation[0, 0]*8 > 4.69
    data, _ = input_pair(frame, frame)
    for side in ('q', 'k'):
        np.testing.assert_array_equal(data[side+'mean'], (token_grid()/8).astype(np.float32))
        np.testing.assert_allclose(data[side+'cov'],
            np.broadcast_to(np.eye(2)*(63/(12*64)), (432, 2, 2)), rtol=1e-7, atol=1e-9)
    np.testing.assert_array_equal(data['k_to_q'], approximation)
    np.testing.assert_array_equal(data['q_to_k'], approximation)


class _CoordinateFeatures(nn.Module):
    def forward(self, value):
        yy, xx = torch.meshgrid(torch.arange(18, device=value.device),
            torch.arange(24, device=value.device), indexing='ij')
        result = value.new_zeros((1, 128, 18, 24))
        result[:, 0] = xx
        result[:, 1] = yy
        result[:, 2] = 1.
        return result


class _Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        for name in ('conv1', 'bn1', 'relu', 'maxpool', 'layer1'):
            setattr(self, name, nn.Identity())
        self.layer2 = _CoordinateFeatures()


class _DepthModel:
    def infer_image(self, bgr, input_size):
        assert input_size == 518
        np.testing.assert_array_equal(bgr[0, 0], [30, 20, 10])
        return np.full(bgr.shape[:2], 2., dtype=np.float32)


def _write_sensor_plane(path, depth_meters):
    millimeters = np.full((48, 64), round(depth_meters*1000), dtype=np.uint32)
    encoded = (((millimeters << 3) | (millimeters >> 13)) & 65535).astype(np.uint16)
    Image.fromarray(encoded).save(path)


def test_frame_input_feature_alignment_and_gt_mutation_cannot_change_forward_arrays(tmp_path):
    rgb_path, gt_path = tmp_path/'image.png', tmp_path/'sensor.png'
    Image.fromarray(np.broadcast_to(np.array([10, 20, 30], dtype=np.uint8), (48, 64, 3))).save(rgb_path)
    K = np.array([[50., 0., 31.5], [0., 50., 23.5], [0., 0., 1.]])
    pose = np.eye(4)
    target_pose = np.eye(4)
    target_pose[0, 3] = .12
    backbone, predictor = _Backbone(), _DepthModel()
    _write_sensor_plane(gt_path, 2.)
    first = frame_inputs(rgb_path, K, pose, predictor, backbone, 'cpu')
    sensor = load_sensor_depth(gt_path)
    labels_a = pair_supervision(sensor, sensor, K, K, pose, target_pose)
    arrays_a, _ = input_pair(first, dict(first, c2w=target_pose))
    _write_sensor_plane(gt_path, 3.)
    second = frame_inputs(rgb_path, K, pose, predictor, backbone, 'cpu')
    sensor = load_sensor_depth(gt_path)
    labels_b = pair_supervision(sensor, sensor, K, K, pose, target_pose)
    arrays_b, _ = input_pair(second, dict(second, c2w=target_pose))
    assert not np.allclose(labels_a.pixcontinuous, labels_b.pixcontinuous, equal_nan=True)
    assert first.keys() == second.keys()
    assert arrays_a.keys() == arrays_b.keys()
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
    for key in arrays_a:
        np.testing.assert_array_equal(arrays_a[key], arrays_b[key])
    # Independent half-pixel calculation for a 3x resize of this native camera.
    expected_K = np.array([[150., 0., 95.5], [0., 150., 71.5], [0., 0., 1.]])
    np.testing.assert_array_equal(first['K'], expected_K)
    np.testing.assert_array_equal(resize_K(K, (64, 48)), expected_K)
    # Ratios undo L2 normalization of the controlled feature coordinates.
    observed_x = (first['features'][:, 0]/first['features'][:, 2]).reshape(18, 24)
    observed_y = (first['features'][:, 1]/first['features'][:, 2]).reshape(18, 24)
    expected_x = np.broadcast_to(np.minimum(np.arange(24)+.4375, 23), (18, 24))
    expected_y = np.broadcast_to(np.minimum(np.arange(18)+.4375, 17)[:, None], (18, 24))
    np.testing.assert_allclose(observed_x, expected_x, rtol=1e-6, atol=2e-6)
    np.testing.assert_allclose(observed_y, expected_y, rtol=1e-6, atol=2e-6)
