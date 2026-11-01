import unittest
from unittest.mock import patch
import numpy as np
import torch

from experiments.rendered_scene import pair, WIDTH, HEIGHT, PATCH
from experiments.region_support_data import (grid_rectangles, rectangle_moments,
    pixel_region_map, visible_region_labels, make_example, make_dataset)


class RegionSupportTests(unittest.TestCase):
    def test_independent_partitions_cover_image(self):
        for jitter in [0, 2, 4]:
            q = grid_rectangles(np.random.default_rng(11), jitter)
            k = grid_rectangles(np.random.default_rng(22), jitter)
            ids = pixel_region_map(q)
            self.assertEqual(ids.shape, (HEIGHT, WIDTH))
            self.assertEqual(len(np.unique(ids)), 48)
            self.assertTrue(np.all(q[:, 2:]-q[:, :2] >= 4))
            if jitter:
                self.assertFalse(np.array_equal(q, k))

    def test_known_query_moments_match_actual_pixels(self):
        rects = grid_rectangles(np.random.default_rng(13), 4)
        mean, cov = rectangle_moments(rects)
        for index, (x0, y0, x1, y1) in enumerate(rects):
            yy, xx = np.mgrid[y0:y1, x0:x1]
            points = np.stack([xx.reshape(-1), yy.reshape(-1)], -1)/PATCH
            np.testing.assert_allclose(mean[index], points.mean(0), atol=1e-12)
            delta = points-points.mean(0)
            np.testing.assert_allclose(cov[index], delta.T@delta/len(delta), atol=1e-12)

    def test_identical_view_exact_region_membership(self):
        a, _, _ = pair(1081001)
        rects = grid_rectangles(np.random.default_rng(21), 2)
        soft, valid, fraction = visible_region_labels(a, a, rects, rects)
        self.assertTrue(valid.all())
        np.testing.assert_allclose(soft, np.eye(48), atol=0)
        np.testing.assert_allclose(fraction, 1, atol=0)

    def test_different_partition_overlap_labels_are_actual_pixel_area(self):
        a, _, _ = pair(1081002)
        rq = grid_rectangles(np.random.default_rng(1), 2)
        rk = grid_rectangles(np.random.default_rng(2), 4)
        soft, valid, fraction = visible_region_labels(a, a, rq, rk)
        qids, kids = pixel_region_map(rq), pixel_region_map(rk)
        exact = np.zeros((48, 48))
        np.add.at(exact, (qids.reshape(-1), kids.reshape(-1)), 1)
        exact /= exact.sum(-1, keepdims=True)
        np.testing.assert_allclose(soft, exact, atol=3e-8)
        self.assertTrue(valid.all())
        np.testing.assert_allclose(fraction, 1, atol=0)
        self.assertTrue(np.any((soft > 0).sum(-1) > 1))

    def test_no_visible_area_is_invalid_without_dustbin(self):
        a, b, _ = pair(1081003)
        b = dict(b)
        b['translation'] = b['translation']+np.array([1000, 0, 0])
        rects = grid_rectangles(np.random.default_rng(23), 2)
        soft, valid, fraction = visible_region_labels(a, b, rects, rects)
        self.assertFalse(valid.any())
        self.assertEqual(soft.shape, (48, 48))
        np.testing.assert_array_equal(soft, 0)
        np.testing.assert_array_equal(fraction, 0)

    def test_geometry_changes_do_not_generate_or_change_labels(self):
        cache = {}
        data, _ = make_example(1081004, cache=cache)
        altered, _ = make_example(1081004, cache=cache, geometry_floor=.1)
        self.assertFalse(torch.equal(data['projected_kcov'], altered['projected_kcov']))
        for name in ['q', 'k', 'pixq', 'projected_qcov', 'soft_labels', 'label_valid', 'visible_fraction']:
            self.assertTrue(torch.equal(data[name], altered[name]), name)
        # Label implementation is not an input to the geometry/RGB functions.
        with patch('experiments.region_support_data.visible_region_labels',
                   return_value=(np.zeros((48, 48), np.float32), np.zeros(48, bool), np.zeros(48, np.float32))):
            unrelated, _ = make_example(1081004, cache=cache)
        for name in data:
            if name not in ['soft_labels', 'label_valid', 'visible_fraction']:
                self.assertTrue(torch.equal(data[name], unrelated[name]), name)

    def test_dataset_shape_spd_and_target_conservation(self):
        data, stats = make_dataset(1082, 2, 'grid_ood')
        self.assertEqual(data['q'].shape, (2, 48, 108))
        self.assertEqual(data['soft_labels'].shape, (2, 48, 48))
        self.assertEqual(data['label_valid'].dtype, torch.bool)
        for field in ['cq', 'ck', 'projected_qcov', 'projected_kcov']:
            self.assertTrue((torch.linalg.eigvalsh(data[field]) > 0).all(), field)
        target_sum = data['soft_labels'].sum(-1)
        torch.testing.assert_close(target_sum, data['label_valid'].float(), atol=1e-6, rtol=0)
        self.assertTrue(((data['visible_fraction'] >= 0) & (data['visible_fraction'] <= 1)).all())
        self.assertEqual(len(stats), 2)
        self.assertTrue(all(s['label_query_pixels'] == WIDTH*HEIGHT for s in stats))


if __name__ == '__main__':
    unittest.main()
