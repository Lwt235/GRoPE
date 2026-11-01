from pathlib import Path

import numpy as np
import pytest

from experiments.round9_context_data import (
    bilinear_rgb, deduplicate_and_limit, normalize_context_cameras,
    read_context_image, scaled_intrinsics, select_ids, triangulate_matches,
)


def test_split_uses_names_old_development_and_fixed_positions():
    # IDs intentionally reverse chronological name order.
    rows = [{'id': 200 - i, 'name': f'image_{i:03d}.png'} for i in range(177)]
    split = select_ids(rows, [r['id'] for r in rows])
    assert len(split['development_ids']) == 118
    assert len(split['context_ids']) == 24
    assert len(split['target_ids']) == 8
    assert split['selected_ids'][3::4] == split['target_ids']
    assert set(split['selected_ids']).isdisjoint(split['sealed_third_block_ids'])
    assert split['selected_ids'][0] == 200
    assert split['selected_ids'][-1] == 83
    assert set(split['context_ids']).isdisjoint(split['target_ids'])


def test_targets_are_rejected_before_any_image_or_hash_open(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('An image access happened')
    monkeypatch.setattr(Path, 'open', forbidden)
    frame = {'id': 8, 'image_path': '/does/not/exist.png'}
    with pytest.raises(ValueError, match='non-context'):
        read_context_image(frame, {1, 2}, {8})
    with pytest.raises(ValueError, match='non-context'):
        read_context_image(frame, {1, 8}, {8})


def test_camera_normalization_preserves_projection():
    R = np.repeat(np.eye(3)[None], 3, axis=0)
    t = np.array([[0., 0., 0.], [-1., 0., 0.], [0., -.7, 0.]])
    center, scale, nt = normalize_context_cameras(R, t)
    points = np.array([[.2, .1, 3.], [1., -.2, 5.]])
    original = points[None] + t[:, None]
    normalized = (points - center)[None] / scale + nt[:, None]
    np.testing.assert_allclose(original[..., :2] / original[..., 2, None],
                               normalized[..., :2] / normalized[..., 2, None], atol=1e-15)


def test_intrinsics_resize_composes_with_half_pixel_mapping():
    K = np.array([[500., 8., 320.], [0., 510., 240.], [0., 0., 1.]])
    twice = scaled_intrinsics(scaled_intrinsics(K, (640, 480), (192, 144)),
                              (192, 144), (500, 300))
    np.testing.assert_allclose(twice, scaled_intrinsics(K, (640, 480), (500, 300)))


def project(points, K, t):
    p = (points + t) @ K.T
    return p[:, :2] / p[:, 2, None]


def test_known_camera_triangulation_recovers_only_geometrically_valid_points():
    pytest.importorskip('cv2')
    K = np.array([[400., 0., 320.], [0., 400., 240.], [0., 0., 1.]])
    R = np.eye(3)
    a, b = np.zeros(3), np.array([-1., 0., 0.])
    points = np.array([[.2, .1, 3.], [.1, .2, 1000.], [.1, .2, -4.], [.5, .5, 4.]])
    x1, x2 = project(points, K, a), project(points, K, b)
    x2[-1, 1] += 30.
    found, valid, stats = triangulate_matches(x1, x2, K, K, R, a, R, b)
    assert valid.tolist() == [True, False, False, False]
    np.testing.assert_allclose(found[0], points[0], rtol=1e-12, atol=1e-12)
    assert stats['accepted'] == 1


def test_empty_matches_preserve_shapes():
    pytest.importorskip('cv2')
    points, valid, stats = triangulate_matches(np.empty((0, 2)), np.empty((0, 2)),
                                              np.eye(3), np.eye(3), np.eye(3), np.zeros(3),
                                              np.eye(3), np.zeros(3))
    assert points.shape == (0, 3) and valid.shape == (0,) and stats['accepted'] == 0


def test_voxel_deduplication_deterministically_keeps_first_then_subsamples():
    points = np.array([[0., 0., 0.], [.01, .01, .01], [1., 0., 0.], [2., 0., 0.], [3., 0., 0.]])
    colors = np.arange(15).reshape(5, 3) / 15
    result, selected, unique = deduplicate_and_limit(points, colors, maximum=3, voxel_width=.1)
    assert unique == 4
    np.testing.assert_array_equal(result, points[[0, 2, 4]])
    np.testing.assert_array_equal(selected, colors[[0, 2, 4]])


def test_bilinear_context_color_and_boundary_clamp():
    rgb = np.zeros((2, 2, 3))
    rgb[1, 1] = 1
    values = bilinear_rgb(rgb, np.array([[.5, .5], [2., 2.], [-1., -1.]]))
    np.testing.assert_allclose(values, np.array([[.25] * 3, [1.] * 3, [0.] * 3]))
