"""Coordinate, format and leakage-isolation regressions for real RGB-D labels."""
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from experiments.round5_rgbd_data import (
    BEHIND_CAMERA, DEPTH_INCONSISTENT, KEY_DEPTH_MISSING, OUTSIDE_IMAGE,
    QUERY_DEPTH_MISSING, Sun3DFrame, bilinear_token_labels, decode_sun3d_depth,
    load_sensor_depth, load_sun3d_scene, pair_supervision, project_covariance,
    project_points_and_jacobian, read_sun3d_extrinsics, read_sun3d_intrinsics,
    resize_intrinsics, resize_pixel_coordinates, sample_nearest_depth, sha256,
    token_grid,
)


def camera(hw=(144, 192)):
    h, w = hw
    return np.array([[120., 0., (w-1)/2], [0., 120., (h-1)/2], [0., 0., 1.]])


def encode_depth(mm):
    values = np.asarray(mm, dtype=np.uint32)
    return (((values << 3) | (values >> 13)) & 65535).astype(np.uint16)


def test_sun3d_bit_rotation_units_and_holes(tmp_path):
    millimeters = np.array([[0, 1, 1000, 4095, 8192, 65535]], dtype=np.uint16)
    encoded = encode_depth(millimeters)
    decoded = decode_sun3d_depth(encoded)
    np.testing.assert_allclose(decoded, millimeters / 1000, atol=3e-6)
    assert decoded[0, 0] == 0
    assert encoded[0, 2] != millimeters[0, 2]
    path = tmp_path / "depth.png"
    Image.fromarray(encoded).save(path)
    np.testing.assert_array_equal(load_sensor_depth(path), decoded)
    with pytest.raises(ValueError):
        decode_sun3d_depth(np.array([[65536]]))


def test_one_based_K_and_half_pixel_resize(tmp_path):
    K1 = np.array([[520., 2., 320.5], [0., 510., 240.5], [0., 0., 1.]])
    path = tmp_path / "intrinsics.txt"
    np.savetxt(path, K1)
    K0 = read_sun3d_intrinsics(path)
    np.testing.assert_array_equal(K0[:2, 2], [319.5, 239.5])
    point = np.array([[.3, -.2, 2.]])
    native, _, _ = project_points_and_jacobian(point, K0)
    resized, _, _ = project_points_and_jacobian(point, resize_intrinsics(K0, (480, 640)))
    np.testing.assert_allclose(resized, resize_pixel_coordinates(native, (480, 640), (144, 192)), atol=1e-13)
    np.testing.assert_allclose(resize_intrinsics(K0, (480, 640))[:2, 2], [95.5, 71.5])


def test_full_token_grid_pixel_centers():
    centers = token_grid()
    assert centers.shape == (432, 2)
    np.testing.assert_array_equal(centers[[0, 1, 24, -1]],
                                  [[3.5, 3.5], [11.5, 3.5], [3.5, 11.5], [187.5, 139.5]])
    with pytest.raises(ValueError):
        token_grid((145, 192))


def test_c2w_extrinsics_consecutive_rows(tmp_path):
    poses = np.broadcast_to(np.eye(4), (3, 4, 4)).copy()
    poses[1, :3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    poses[:, :3, 3] = [[0, 0, 0], [1, 2, 3], [-2, .3, 1]]
    path = tmp_path / "poses.txt"
    np.savetxt(path, poses[:, :3].reshape(-1, 4))
    np.testing.assert_allclose(read_sun3d_extrinsics(path), poses)
    np.savetxt(path, np.ones((4, 4)))
    with pytest.raises(ValueError):
        read_sun3d_extrinsics(path)


def test_native_nearest_depth_never_interpolates_holes():
    values = np.array([[1., 0., 9., np.nan]])
    z, valid = sample_nearest_depth(values, np.array([[.49, 0], [.5, 0], [1.49, 0], [1.5, 0], [-.51, 0]]))
    np.testing.assert_array_equal(valid, [True, False, False, True, False])
    np.testing.assert_allclose(z[[0, 3]], [1, 9])
    assert np.isnan(z[[1, 2, 4]]).all()


def test_native_nearest_half_pixel_tie_stable_to_camera_roundoff():
    depth = np.array([[1., 9.]])
    z, known = sample_nearest_depth(depth, np.array([[.5, 0], [.5-1e-12, 0], [.5-1e-7, 0]]))
    assert known.all()
    np.testing.assert_array_equal(z, [9, 9, 1])


def test_project_jacobian_and_covariance_resize_units():
    points = np.array([[.2, -.3, 1.7], [-1., .5, 3.]])
    K = camera()
    K[0, 1] = 3
    pixels, J, valid = project_points_and_jacobian(points, K)
    assert valid.all()
    for axis in range(3):
        shift = np.zeros_like(points)
        shift[:, axis] = 1e-6
        plus = project_points_and_jacobian(points + shift, K)[0]
        minus = project_points_and_jacobian(points - shift, K)[0]
        np.testing.assert_allclose(J[..., axis], (plus-minus)/2e-6, rtol=1e-8, atol=1e-8)
    covariance = np.array([[.02, .001, 0], [.001, .01, .002], [0, .002, .03]])
    _, C, _ = project_covariance(points, covariance, K)
    np.testing.assert_allclose(C, J @ covariance @ J.transpose(0, 2, 1))
    _, C_small, _ = project_covariance(points, covariance, resize_intrinsics(K, (144, 192), (72, 96)))
    np.testing.assert_allclose(C_small, C / 4)
    bad_pixels, bad_J, valid = project_points_and_jacobian(np.array([[0., 0., -1.]]), K)
    assert not valid[0] and np.isnan(bad_pixels).all() and not bad_J.any()


def test_identity_plane_is_identity_labels_at_native_resolution():
    depth = np.full((480, 640), 2., dtype=np.float32)
    K = camera((480, 640))
    labels = pair_supervision(depth, depth, K, K, np.eye(4), np.eye(4))
    assert labels.gt_valid.all()
    np.testing.assert_allclose(labels.pixcontinuous, token_grid(), atol=1e-5)
    np.testing.assert_allclose(labels.dense_labels(), np.eye(432), atol=1e-12)
    assert labels.metadata["nearest_grid_oracle_pck8"] == 1


def test_nonplanar_identity_reprojection_keeps_every_sensor_sample():
    # Native .5 coordinates occur after 640->192 half-pixel resizing. Alternating
    # depths expose a wrong adjacent-pixel selection that constant planes hide.
    depth = np.broadcast_to(2. + 3*(np.arange(640) % 2), (480, 640)).copy()
    pose = np.eye(4)
    theta = .3
    pose[:3, :3] = [[np.cos(theta), 0, np.sin(theta)], [0, 1, 0],
                    [-np.sin(theta), 0, np.cos(theta)]]
    pose[:3, 3] = [.1, .2, .3]
    K = camera((480, 640))
    labels = pair_supervision(depth, depth, K, K, pose, pose)
    assert labels.gt_valid.all()
    np.testing.assert_allclose(labels.pixcontinuous, token_grid(), atol=1e-5)


def test_bilinear_four_labels_and_edge_clamp():
    targets = np.array([[7.5, 7.5], [-.49, 3.5], [191.49, 143.49], [np.nan, np.nan]])
    indices, weights = bilinear_token_labels(targets, np.array([1, 1, 1, 0], bool))
    np.testing.assert_array_equal(indices[0], [0, 1, 24, 25])
    np.testing.assert_array_equal(weights[0], [.25] * 4)
    np.testing.assert_array_equal(weights[1], [1, 0, 0, 0])
    assert indices[2, 0] == 431 and weights[2, 0] == 1
    assert (indices[3] == -1).all() and not weights[3].any()


def test_pose_direction_translation_and_soft_labels():
    depth = np.full((144, 192), 2.)
    K = camera()
    # A key camera translated -1/15 m in world shifts projected pixels +4.
    key_c2w = np.eye(4)
    key_c2w[0, 3] = -1/15
    labels = pair_supervision(depth, depth, K, K, np.eye(4), key_c2w)
    expected = token_grid() + [4, 0]
    np.testing.assert_allclose(labels.pixcontinuous, expected, atol=1e-5)
    assert labels.gt_valid.reshape(18, 24)[:, :-1].all()
    assert (labels.status.reshape(18, 24)[:, -1] == OUTSIDE_IMAGE).all()
    np.testing.assert_array_equal(labels.key_indices[0, :2], [0, 1])
    np.testing.assert_array_equal(labels.key_weights[0], [.5, .5, 0, 0])


def test_sensor_holes_and_depth_mismatch_remain_explicit_unknown():
    qdepth = np.full((144, 192), 2.)
    kdepth = qdepth.copy()
    # Query pixel center 3.5 is nearest native pixel 4.
    qdepth[4, 4] = 0
    kdepth[4, 12] = 0
    kdepth[4, 20] = 1
    labels = pair_supervision(qdepth, kdepth, camera(), camera(), np.eye(4), np.eye(4))
    assert labels.status[0] == QUERY_DEPTH_MISSING
    assert labels.status[1] == KEY_DEPTH_MISSING
    assert labels.status[2] == DEPTH_INCONSISTENT
    assert not labels.gt_valid[:3].any()
    assert (labels.key_indices[:3] == -1).all()
    assert not labels.key_weights[:3].any()
    assert labels.gt_valid[3:].all()
    # Depth tolerance is .05 + .05*sensor_z, including equality.
    kdepth.fill(2.1)
    assert pair_supervision(np.full_like(qdepth, 2.), kdepth, camera(), camera(), np.eye(4), np.eye(4)).gt_valid.all()


def test_behind_and_outside_not_sensor_holes():
    depth = np.full((144, 192), 2.)
    key_c2w = np.eye(4)
    key_c2w[0, 3] = 10
    labels = pair_supervision(depth, depth, camera(), camera(), np.eye(4), key_c2w)
    assert (labels.status == OUTSIDE_IMAGE).all()
    key_c2w = np.eye(4)
    key_c2w[2, 3] = 3
    labels = pair_supervision(depth, depth, camera(), camera(), np.eye(4), key_c2w)
    assert (labels.status == BEHIND_CAMERA).all()


def make_sparse_scene(tmp_path):
    scene = "room/sequence"
    directory = tmp_path / scene
    directory.mkdir(parents=True)
    rgbs = [f"{i+1:07d}-{100+i*10:012d}.jpg" for i in range(5)]
    depths = [f"{i+90:07d}-{101+i*10:012d}.png" for i in range(5)]
    for names, filename in [(rgbs, "image_index.html"), (depths, "depth_index.html")]:
        (directory / filename).write_text("\n".join(f'<a href="{n}">{n}</a>' for n in names))
    Kpath = directory / "intrinsics.txt"
    K = camera()
    K[:2, 2] += 1
    np.savetxt(Kpath, K)
    poses = np.broadcast_to(np.eye(4), (5, 4, 4)).copy()
    poses[:, 0, 3] = np.arange(5)
    posepath = directory / "extrinsics.txt"
    np.savetxt(posepath, poses[:, :3].reshape(-1, 4))
    row = {"scene": scene, "split": "train", "rgb_count": 5, "depth_count": 5,
           "pose_count": 5, "pose_file": str(posepath.relative_to(tmp_path)),
           "K_sha256": sha256(Kpath), "pose_sha256": sha256(posepath),
           "pairs": [[1, 3]], "frames": [
               {"index": i, "rgb": rgbs[i], "depth": depths[i], "time_gap_us": 1} for i in [1, 3]]}
    (tmp_path / "selection.json").write_text(json.dumps({"scenes": [row]}))
    return scene, row


def test_sparse_reader_original_pose_index_and_nearest_timestamps(tmp_path):
    scene, row = make_sparse_scene(tmp_path)
    frames, _ = load_sun3d_scene(tmp_path, scene)
    assert list(frames) == [1, 3]
    assert frames[1].c2w[0, 3] == 1 and frames[3].c2w[0, 3] == 3
    assert frames[1].depth_path.name.startswith("0000091-")
    row["frames"][0]["index"] = 0
    (tmp_path / "selection.json").write_text(json.dumps({"scenes": [row]}))
    with pytest.raises(ValueError, match="original full-list index"):
        load_sun3d_scene(tmp_path, scene)


def test_observation_cannot_read_depth_or_gt_mask(tmp_path):
    rgbpath = tmp_path / "rgb.jpg"
    depthpath = tmp_path / "not_present.png"
    image = np.arange(144*192*3, dtype=np.uint8).reshape(144, 192, 3)
    Image.fromarray(image).save(rgbpath)
    frame = Sun3DFrame("frame", 7, rgbpath, depthpath, camera(), np.eye(4), 0)
    before = frame.observation()
    # Observation succeeds even without any depth file. Corrupt depth also cannot affect it.
    depthpath.write_bytes(b"not a depth image")
    after = frame.observation()
    assert set(vars(before)) == {"frame_id", "rgb", "K", "c2w", "pixels"}
    for key in ["rgb", "K", "c2w", "pixels"]:
        np.testing.assert_array_equal(getattr(before, key), getattr(after, key))


def test_metadata_hash_tampering_rejected(tmp_path):
    scene, _ = make_sparse_scene(tmp_path)
    with (tmp_path / scene / "intrinsics.txt").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_sun3d_scene(tmp_path, scene)
