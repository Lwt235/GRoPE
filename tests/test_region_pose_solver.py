from itertools import combinations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import torch

from experiments.region_pose_solver import estimate, recover_truth, rigid_torch, score_transform


def test_proper_rotation_exact_reflection_and_degenerate_geometry():
    rng = np.random.default_rng(14)
    source = rng.normal(size=(24, 3))
    rotation = Rotation.from_rotvec([.3, -.7, .2]).as_matrix()
    translation = np.array([.4, -.2, 1.])
    target = source @ rotation.T + translation
    r, t, valid = rigid_torch(torch.from_numpy(source), torch.from_numpy(target))
    assert valid
    np.testing.assert_allclose(r, rotation, atol=1e-12)
    np.testing.assert_allclose(t, translation, atol=1e-12)
    r, t, valid = rigid_torch(torch.from_numpy(source), torch.from_numpy(-source))
    assert valid and float(torch.linalg.det(r)) == pytest.approx(1.)
    assert np.linalg.norm(source @ r.numpy().T + t.numpy() + source) > 1.
    line = torch.arange(24, dtype=torch.float64)[:, None] * torch.ones((1, 3), dtype=torch.float64)
    assert not rigid_torch(line, line)[2]


def test_exhaustive_consensus_with_outliers_matches_separate_scipy_solver():
    rng = np.random.default_rng(27)
    source = rng.normal(size=(10, 3))
    rotation = Rotation.from_rotvec([.7, -.2, -.4]).as_matrix()
    translation = np.array([.2, -.5, .4])
    target = source @ rotation.T + translation
    target[6:] = rng.normal(size=(4, 3)) * 7
    result = estimate(torch.from_numpy(source), torch.from_numpy(target), torch.arange(10), radius=.01, chunk_size=17)
    assert result['valid'] and result['consensus_count'] == 6 and result['hypotheses_total'] == 120
    np.testing.assert_allclose(result['rotation'], rotation, atol=1e-10)
    np.testing.assert_allclose(result['translation'], translation, atol=1e-10)
    maximum = 0
    for index in combinations(range(10), 3):
        a, b = source[list(index)], target[list(index)]
        r, _ = Rotation.align_vectors(b - b.mean(0), a - a.mean(0))
        t = b.mean(0) - r.apply(a.mean(0))
        maximum = max(maximum, int((np.linalg.norm(r.apply(source) + t - target, axis=1) <= .01).sum()))
    assert maximum == result['consensus_count']
    collapsed = estimate(torch.from_numpy(source), torch.from_numpy(target), torch.zeros(10, dtype=torch.long))
    assert not collapsed['valid'] and collapsed['distinct_target_hypotheses'] == 0


def test_truth_recovery_units_and_failed_estimates_count_as_failures():
    rng = np.random.default_rng(93)
    local = rng.normal(size=(2, 64, 3))
    r = [Rotation.from_rotvec(v).as_matrix() for v in ([.3, -.2, .5], [-.4, .1, .6])]
    t = np.array([[1., -.2, .7], [.2, -.5, -.4]])
    world = np.stack([local[i] @ r[i].T + t[i] for i in range(2)])
    actual_r, actual_t, error = recover_truth(local, world)
    np.testing.assert_allclose(actual_r, r[1].T @ r[0], atol=1e-12)
    np.testing.assert_allclose(actual_t, r[1].T @ (t[0] - t[1]), atol=1e-12)
    assert error < 1e-12
    exact = score_transform(local[0], actual_r, actual_t, actual_r, actual_t, 2.)
    assert exact['success'] and exact['rmse_m'] == 0
    shifted = score_transform(local[0], actual_r, actual_t + [.15, 0, 0], actual_r, actual_t, 2.)
    assert not shifted['success'] and shifted['rmse_m'] == pytest.approx(.3)
    failed = score_transform(local[0], actual_r, actual_t, actual_r, actual_t, 2., valid=False)
    assert not failed['success'] and np.isinf(failed['rmse_m'])
