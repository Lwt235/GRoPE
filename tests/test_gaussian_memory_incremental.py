"""Scientific invariants for the new state/projection path (CPU, no training)."""
import numpy as np
import torch

from models.gaussian_memory_vggt_adapter import project_memory
from models.gaussian_memory_incremental import blend, association, update_memory


def example(n=2):
    means = torch.zeros(n, 3); means[:, 2] = torch.arange(n) + 1.
    memory = {'means': means, 'covars': torch.eye(3)[None].repeat(n, 1, 1) * .01,
              'opacity': torch.full((n,), .7), 'features': torch.zeros(n, 32)}
    camera = torch.eye(4)[None]
    intrinsic = torch.tensor([[[10., 0, 10.5], [0, 10., 7.5], [0, 0, 1.]]])
    return memory, camera, intrinsic


def test_alpha_order_and_no_coverage():
    memory, camera, K = example()
    p = project_memory(memory, camera, K)
    pixel = 7 * 20 + 10
    np.testing.assert_allclose(p['coverage'][0, pixel], .7 + .3 * .7, atol=1e-7)
    np.testing.assert_allclose(p['depth'][0, pixel], (.7 + .3 * .7 * 2) / .91, atol=1e-6)
    memory['opacity'].zero_()
    assert torch.count_nonzero(project_memory(memory, camera, K)['pe']) == 0


def test_common_rigid_change_and_memory_order():
    memory, camera, K = example(7)
    original = project_memory(memory, camera, K)['pe']
    theta = .37
    R = torch.tensor([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0],
                      [-np.sin(theta), 0, np.cos(theta)]], dtype=torch.float32)
    translation = torch.tensor([.2, -.3, .4])
    transformed = dict(memory, means=memory['means'] @ R.T + translation,
                       covars=R[None] @ memory['covars'] @ R.T[None])
    pose = camera.clone(); pose[0, :3, :3] = R; pose[0, :3, 3] = translation
    torch.testing.assert_close(project_memory(transformed, pose, K)['pe'], original, atol=2e-5, rtol=2e-5)
    permutation = torch.tensor([4, 2, 6, 1, 0, 5, 3])
    shuffled = {k: v[permutation] for k, v in memory.items()}
    torch.testing.assert_close(project_memory(shuffled, camera, K)['pe'], original)


def test_fused_covariance_contains_between_center_extent():
    mu, cov, _ = blend(torch.tensor([-1., 0, 0]), torch.eye(3), torch.zeros(32), torch.tensor(1.),
                      torch.tensor([1., 0, 0]), torch.eye(3), torch.ones(32), torch.tensor(1.))
    torch.testing.assert_close(mu, torch.zeros(3))
    torch.testing.assert_close(cov, torch.diag(torch.tensor([2., 1., 1.])))


def test_occluded_background_is_not_negative_evidence():
    memory, camera, K = example(256)
    memory['means'][:, 2] = 2.
    observation = {'camera': camera, 'K': K, 'depth_patch': torch.ones(300),
                   'valid': torch.ones(300, dtype=torch.bool)}
    assert not association(memory, observation)[2].any()
    memory['means'][:, 2] = .5
    assert association(memory, observation)[2].all()


def test_readonly_query_cannot_enter_writer():
    try:
        update_memory({}, {'source_frame': torch.tensor([5])}, [])
    except AssertionError:
        pass
    else:
        raise AssertionError('Read-only query was accepted as update evidence.')
