"""CPU checks for the geometry shared by all Round 9 readout controls.

No scene files or target images are opened. Quaternion references use SciPy;
projection covariance references differentiate the full homogeneous camera map.
The renderer boundary is stubbed only to inspect its pixel-centre contract.
"""
import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from experiments import round9_gs_readout as runner
from experiments.round9_context_data import scaled_intrinsics


def fixture_geometry():
    dtype = torch.float64
    means = torch.tensor([[.15, -.2, 2.], [-.3, .1, 3.], [.7, .2, 4.]], dtype=dtype)
    logs = torch.tensor([[.02, .08, .005], [.1, .015, .04], [.03, .03, .008]], dtype=dtype).log()
    quaternions = torch.tensor([[1., .2, -.1, .4], [.3, -.5, .2, .7], [.8, .5, -.6, .3]], dtype=dtype)
    K = torch.tensor([[173., 0., 93.7], [0., 169., 70.2], [0., 0., 1.]], dtype=dtype)
    R = torch.tensor(Rotation.from_rotvec([.12, -.07, .03]).as_matrix(), dtype=dtype)
    t = torch.tensor([-.1, .2, .3], dtype=dtype)
    return means, logs, quaternions, K, R, t


def test_covariance_matches_independent_scalar_first_quaternion_rotation():
    _, logs, quaternions, _, _, _ = fixture_geometry()
    # SciPy takes x,y,z,w, whereas GS storage is w,x,y,z.
    rotations = Rotation.from_quat(quaternions[:, [1, 2, 3, 0]].numpy()).as_matrix()
    reference = np.stack([r @ np.diag(s*s) @ r.T for r, s in zip(rotations, logs.exp().numpy())])
    actual = runner.covariance(logs, quaternions)
    torch.testing.assert_close(actual, torch.from_numpy(reference), rtol=2e-14, atol=1e-17)
    expected_eigenvalues = logs.exp().square().sort(-1).values
    torch.testing.assert_close(torch.linalg.eigvalsh(actual), expected_eigenvalues, rtol=3e-13, atol=1e-17)


def test_covariance_quaternion_scale_sign_and_spherical_invariance():
    _, logs, quaternions, _, _, _ = fixture_geometry()
    original = runner.covariance(logs, quaternions)
    for factor in (-1., 3.7, -.013):
        torch.testing.assert_close(runner.covariance(logs, quaternions * factor), original,
                                   rtol=3e-14, atol=1e-17)
    spherical = torch.full_like(logs, .04).log()
    expected = torch.eye(3, dtype=logs.dtype)[None].expand(3, -1, -1) * .04**2
    torch.testing.assert_close(runner.covariance(spherical, quaternions), expected, rtol=1e-14, atol=1e-17)


def test_projection_covariance_matches_homogeneous_map_jacobian():
    means, logs, quaternions, K, R, t = fixture_geometry()
    covariance = runner.covariance(logs, quaternions)
    actual_xy, actual_covariance, actual_z = runner.project(means, covariance, K, R, t)
    references = []
    for point, cov in zip(means, covariance):
        def camera_map(x):
            homogeneous = K @ (R @ x + t)
            return homogeneous[:2] / homogeneous[2]
        jacobian = torch.autograd.functional.jacobian(camera_map, point)
        references.append(jacobian @ cov @ jacobian.T)
    expected_xy = torch.stack([(K @ (R @ x + t))[:2] / (K @ (R @ x + t))[2] for x in means])
    torch.testing.assert_close(actual_xy, expected_xy, rtol=2e-15, atol=2e-14)
    torch.testing.assert_close(actual_covariance, torch.stack(references), rtol=2e-14, atol=2e-14)
    torch.testing.assert_close(actual_z, torch.stack([(R @ x + t)[2] for x in means]), rtol=0, atol=0)
    assert (torch.linalg.eigvalsh(actual_covariance) > 0).all()


def test_projection_is_invariant_under_rigid_world_coordinate_change():
    means, logs, quaternions, K, R, t = fixture_geometry()
    covariance = runner.covariance(logs, quaternions)
    a = torch.tensor(Rotation.from_rotvec([.4, -.2, .3]).as_matrix(), dtype=means.dtype)
    b = torch.tensor([1., -.7, 2.], dtype=means.dtype)
    changed_means = means @ a.T + b
    changed_covariance = a @ covariance @ a.T
    changed_R = R @ a.T
    changed_t = t - changed_R @ b
    before = runner.project(means, covariance, K, R, t)
    after = runner.project(changed_means, changed_covariance, K, changed_R, changed_t)
    for original, transformed in zip(before, after):
        torch.testing.assert_close(original, transformed, rtol=2e-13, atol=2e-13)


def test_context_scene_normalization_preserves_image_geometry():
    means, logs, quaternions, K, R, t = fixture_geometry()
    covariance = runner.covariance(logs, quaternions)
    center = torch.tensor([.5, -.2, -.3], dtype=means.dtype)
    scale = 2.7
    before = runner.project(means, covariance, K, R, t)
    after = runner.project((means-center)/scale, covariance/scale**2, K, R, (R@center+t)/scale)
    torch.testing.assert_close(before[0], after[0], rtol=2e-14, atol=2e-14)
    torch.testing.assert_close(before[1], after[1], rtol=2e-14, atol=2e-14)
    torch.testing.assert_close(before[2]/scale, after[2], rtol=2e-15, atol=2e-15)


def test_projection_has_finite_shape_and_position_gradients():
    means, logs, quaternions, K, R, t = fixture_geometry()
    means.requires_grad_(); logs.requires_grad_(); quaternions.requires_grad_()

    def function(position, log_scale, quaternion):
        xy, cov, z = runner.project(position, runner.covariance(log_scale, quaternion), K, R, t)
        return torch.cat((xy.reshape(-1)/100, cov.reshape(-1)/10, z))

    assert torch.autograd.gradcheck(function, (means, logs, quaternions), rtol=1e-4, atol=2e-6)


def test_projection_preserves_negative_depth_for_external_visibility_filter():
    _, _, _, K, _, _ = fixture_geometry()
    means = torch.tensor([[0., 0., -1.], [0., 0., 0.], [0., 0., 2.]], dtype=K.dtype)
    cov = torch.eye(3, dtype=K.dtype)[None].repeat(3, 1, 1) * .001
    xy, covariance, z = runner.project(means, cov, K, torch.eye(3, dtype=K.dtype), torch.zeros(3, dtype=K.dtype))
    assert torch.equal(z > 1e-4, torch.tensor([False, False, True]))
    assert torch.isfinite(xy).all() and torch.isfinite(covariance).all()


@pytest.mark.parametrize("depth", [False, True])
def test_renderer_half_pixel_adapter_matches_integer_center_projection(monkeypatch, depth):
    means, logs, quaternions, K, R, t = fixture_geometry()
    state = dict(means=means, log_scales=logs, quaternions=quaternions,
                 opacity_logits=torch.zeros(3, dtype=K.dtype), color_logits=torch.zeros((3, 3), dtype=K.dtype))
    observed = {}

    def fake_rasterizer(**kwargs):
        observed.update(kwargs)
        channels = 4 if kwargs['render_mode'] == 'RGB+ED' else 3
        return (K.new_zeros((1, runner.HEIGHT, runner.WIDTH, channels)),
                K.new_ones((1, runner.HEIGHT, runner.WIDTH, 1)), {})

    monkeypatch.setattr(runner, 'runtime', lambda: fake_rasterizer)
    K_before = K.clone()
    rgb = torch.tensor([[.1, .2, .3], [.4, .5, .6], [.7, .8, .9]], dtype=K.dtype)
    result, alpha = runner.render(state, K, R, t, colors=rgb, depth=depth)
    # gsplat evaluates integer array pixel (row,col) at (col+.5,row+.5).
    # Subtracting that offset must recover our source-image centre convention.
    camera_points = means @ observed['viewmats'][0, :3, :3].T + observed['viewmats'][0, :3, 3]
    renderer_homogeneous = camera_points @ observed['Ks'][0].T
    renderer_integer_xy = renderer_homogeneous[:, :2] / renderer_homogeneous[:, 2, None] - .5
    readout_xy, _, _ = runner.project(means, runner.covariance(logs, quaternions), K, R, t)
    torch.testing.assert_close(renderer_integer_xy, readout_xy, rtol=2e-15, atol=2e-14)
    torch.testing.assert_close(K, K_before, rtol=0, atol=0)
    torch.testing.assert_close(observed['colors'], rgb, rtol=0, atol=0)
    assert result.shape == (runner.HEIGHT, runner.WIDTH, 4 if depth else 3)
    assert alpha.shape == (runner.HEIGHT, runner.WIDTH)


def test_resize_intrinsics_and_renderer_pixel_conventions_compose():
    means, logs, quaternions, K, R, t = fixture_geometry()
    covariance = runner.covariance(logs, quaternions)
    old_xy, old_covariance, _ = runner.project(means, covariance, K, R, t)
    old_size, new_size = (640, 480), (192, 144)
    resized_K = torch.from_numpy(scaled_intrinsics(K.numpy(), old_size, new_size))
    new_xy, new_covariance, _ = runner.project(means, covariance, resized_K, R, t)
    scale = torch.tensor(np.array(new_size)/old_size, dtype=K.dtype)
    torch.testing.assert_close(new_xy, (old_xy+.5)*scale-.5, rtol=2e-15, atol=2e-14)
    transform = torch.diag(scale)
    torch.testing.assert_close(new_covariance, transform@old_covariance@transform,
                               rtol=2e-14, atol=2e-14)
