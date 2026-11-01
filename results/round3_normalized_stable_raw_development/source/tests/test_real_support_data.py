import unittest

import numpy as np

from experiments.real_support_data import (fit_regions, make_pairs,
                                           project_gaussian, region_indices,
                                           soft_track_labels)


class RealSupportDataTests(unittest.TestCase):
    def test_half_pixel_region_partition(self):
        pixels = np.array([[-.5, -.5], [23.4999, 0], [23.5, 0], [191.499, 143.499], [191.5, 0]])
        indices, inside = region_indices(pixels, 192, 144, (8, 6))
        np.testing.assert_array_equal(inside, [True, True, True, True, False])
        np.testing.assert_array_equal(indices[:4], [0, 0, 1, 47])

    def test_camera_splits_have_no_overlap(self):
        pairs = make_pairs(list(range(177)), 8)
        sets = {s: set() for s in ('train', 'validation', 'test')}
        for pair in pairs:
            sets[pair['split']].update((pair['q_id'], pair['k_id']))
        self.assertFalse(sets['train'] & sets['validation'])
        self.assertFalse(sets['train'] & sets['test'])
        self.assertFalse(sets['validation'] & sets['test'])
        self.assertEqual(len(pairs), 48)

    def test_observed_tracks_produce_soft_labels_and_not_occlusion_truth(self):
        a = {1: 0, 2: 0, 3: 0, 4: 1, 5: 1}
        b = {1: 0, 2: 1, 4: 1}
        result = soft_track_labels(a, b, token_count=2, min_tracks=2)
        np.testing.assert_allclose(result['soft_labels'], [[.5, .5], [0, 1]])
        np.testing.assert_array_equal(result['label_valid'], [True, False])
        np.testing.assert_allclose(result['track_observation_fraction'], [2/3, .5])
        renamed = soft_track_labels({i+17: v for i,v in a.items()}, {i+17:v for i,v in b.items()}, 2, 2)
        np.testing.assert_array_equal(result['soft_labels'], renamed['soft_labels'])

    def test_fit_includes_internal_covariance_without_track_input(self):
        means = np.array([[0., 0, 2.], [.01, 0, 2.]])
        covariance = np.broadcast_to(np.eye(3) * .01, (2, 3, 3))
        K = np.array([[10., 0, 5], [0, 10, 5], [0, 0, 1]])
        fit = fit_regions(means, covariance, np.array([.8, .8]), K, np.eye(3), np.zeros(3),
                          width=12, height=12, grid=(1,1), min_points=2)
        self.assertTrue(fit['valid'][0])
        np.testing.assert_allclose(fit['mean'][0], [.005, 0, 2])
        np.testing.assert_allclose(fit['cov'][0], np.diag([.010025, .01, .01]))

    def test_projected_covariance_world_rotation(self):
        mean = np.array([[.1, .2, 2.]])
        C = np.diag([.01, .02, .03])[None]
        K = np.array([[10., 0, 5], [0, 12., 6], [0, 0, 1]])
        R = np.array([[0., -1., 0.], [1., 0., 0.], [0,0,1.]])
        pix, c2, front = project_gaussian(mean, C, K, R, np.zeros(3))
        Jc = np.array([[5., 0, .5], [0, 6., -.3]])
        np.testing.assert_allclose(c2[0], Jc @ R @ C[0] @ R.T @ Jc.T)
        np.testing.assert_allclose(pix[0], [4, 6.6])
        self.assertTrue(front[0])


if __name__ == '__main__':
    unittest.main()
