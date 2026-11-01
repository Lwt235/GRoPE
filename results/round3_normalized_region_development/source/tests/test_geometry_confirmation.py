"""Regressions for the estimand, masks and kernels used in confirmation.

These test statistical and information-flow correctness, not effectiveness.
The heterogeneous cluster fixture prevents replacing row-weighted summaries
with equal-camera confidence intervals while leaving the summary unchanged.
"""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.real_support_geometry_probe import (
    build_examples, cluster_difference, log_bc, normalised_log_probability,
    summarise,
)
from experiments.isometric_confirmation_analysis import (
    function_ast_hashes, load_completed_manifest, paired_camera_seed_statistics, paired_point_statistics,
)


class GeometryConfirmationTests(unittest.TestCase):
    def test_cluster_bootstrap_preserves_row_weighted_estimand(self):
        a, b = [], []
        # Unequal numbers of rows make the estimands substantially different.
        for camera, differences in ((1, [6.]), (2, [-1.] * 9), (3, [2.] * 2)):
            for i, difference in enumerate(differences):
                identity = dict(scene="s", q_id=camera, k_id=8, query_region=i)
                a.append(dict(identity, soft_ce=difference + 10.))
                b.append(dict(identity, soft_ce=10.))
        result = cluster_difference(a, b, "soft_ce")
        self.assertAlmostEqual(result["mean"], (6. - 9. + 4.) / 12.)
        self.assertNotAlmostEqual(result["mean"], (6. - 1. + 2.) / 3.)
        self.assertEqual([c["rows"] for c in result["camera_differences"]], [1, 9, 2])
        # Independently replay complete-camera resampling as a ratio of sums.
        draws = np.random.default_rng(219782).integers(0, 3, size=(4000, 3))
        sums, counts = np.array([6., -9., 4.]), np.array([1, 9, 2])
        expected = sums[draws].sum(1) / counts[draws].sum(1)
        np.testing.assert_allclose(result["ci95"], np.quantile(expected, [.025, .975]))
        # Changing the order of the paired comparison must not change results.
        self.assertEqual(result, cluster_difference(a[::-1], b[::-1], "soft_ce"))

    def test_cluster_mean_equals_difference_of_summaries(self):
        metrics = ["soft_ce", "kl", "top1_target_mass", "top1_max_bin_accuracy",
                   "oracle_max_target_mass", "retained_target_mass"]
        a, b = [], []
        for camera, count in ((1, 1), (2, 5)):
            for i in range(count):
                row = dict(scene="s", q_id=camera, k_id=7, query_region=i)
                a.append(dict(row, **{m: camera + i * .1 for m in metrics}))
                b.append(dict(row, **{m: i * .2 for m in metrics}))
        for metric in metrics:
            mean = cluster_difference(a, b, metric)["mean"]
            self.assertAlmostEqual(mean, summarise(a)[metric] - summarise(b)[metric])

    def test_bc_invariant_to_shared_affine_coordinate_change(self):
        means = np.array([[.2, -.5, .4], [-.3, .2, 1.1]])
        C = np.array([np.diag([.03, .2, .8]), np.diag([.4, .1, .6])])
        A = np.array([[2., .3, .2], [.1, 1.3, -.4], [.2, .1, .8]])
        t = np.array([8., -3., 1.])
        original = log_bc(means, C, means[::-1], C[::-1])
        changed_mean = means @ A.T + t
        changed_C = A @ C @ A.T
        changed = log_bc(changed_mean, changed_C,
                         changed_mean[::-1], changed_C[::-1])
        np.testing.assert_allclose(original, changed, atol=2e-14)
        np.testing.assert_allclose(np.exp(original).diagonal()[::-1],
                                   np.exp(changed).diagonal()[::-1], atol=2e-14)

    def test_direct_kernel_normalisation_is_not_softmax_of_kernel(self):
        log_kernel = np.log(np.array([[.8, .2, .5]]))
        valid = np.array([True, True, False])
        direct = normalised_log_probability(log_kernel, valid, "normalise_kernel")
        softmax = normalised_log_probability(log_kernel, valid, "softmax_kernel", 8.)
        np.testing.assert_allclose(np.exp(direct)[0], [.8, .2, 0.])
        self.assertFalse(np.allclose(np.exp(direct), np.exp(softmax)))
        self.assertTrue(np.isneginf(direct[0, 2]))

    @staticmethod
    def _write_fixture(root, soft_labels):
        scene = root / "scene"
        scene.mkdir(exist_ok=True)
        means = np.array([[0., 0, 2.], [.1, .2, 2.], [.3, -.1, 3.]])
        C = np.broadcast_to(np.diag([.01, .04, .2]), (3, 3, 3)).copy()
        pixel_mean = np.array([[12., 12.], [36., 12.], [60., 12.]])
        pixel_C = np.broadcast_to(np.eye(2) * 48., (3, 2, 2)).copy()
        data = dict(q_mean=means, k_mean=means + .01, q_cov=C, k_cov=C * 1.2,
                    q_pixel_mean=pixel_mean, q_pixel_cov=pixel_C,
                    q_projected_mean=pixel_mean + .1, k_projected_mean=pixel_mean + .2,
                    q_projected_cov=pixel_C, k_projected_cov=pixel_C * 1.2,
                    q_valid=np.ones(3, bool), k_valid=np.array([True, True, False]),
                    q_valid_alt=np.array([True, False, True]),
                    k_valid_alt=np.array([True, False, True]),
                    label_valid=np.ones(3, bool), soft_labels=np.array(soft_labels),
                    track_counts=np.ones((3, 3), int))
        for side in ("q", "k"):
            for field in ("mean", "cov", "projected_mean", "projected_cov"):
                data[f"{side}_{field}_alt"] = data[f"{side}_{field}"] * 1.01
        np.savez(scene / "pair_test.npz", **data)
        (scene / "manifest.json").write_text(json.dumps(dict(
            scene="s", normalization=dict(scale=2.), pairs=[dict(
                split="test", file="pair_test.npz", q_id=1, k_id=2,
                baseline_group="near")],
        )))

    def test_joint_parity_masks_have_identical_targets_and_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_fixture(root, [[.2, .6, .2]] * 3)
            primary, _, _ = build_examples(root, ("test",), joint_validity=True)
            alternate, _, _ = build_examples(root, ("test",), alternate=True,
                                               joint_validity=True)
            np.testing.assert_array_equal(primary[0]["row_indices"], [0, 2])
            np.testing.assert_array_equal(primary[0]["k_valid"], [True, False, False])
            np.testing.assert_array_equal(primary[0]["row_indices"], alternate[0]["row_indices"])
            np.testing.assert_array_equal(primary[0]["k_valid"], alternate[0]["k_valid"])
            np.testing.assert_allclose(primary[0]["targets"], alternate[0]["targets"])
            np.testing.assert_allclose(primary[0]["targets"], [[1., 0., 0.]] * 2)
            np.testing.assert_allclose(primary[0]["retained"], [.2, .2])

    def test_changing_labels_does_not_change_any_kernel(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_fixture(root, [[.2, .6, .2]] * 3)
            before, _, _ = build_examples(root, ("test",))
            self._write_fixture(root, [[.6, .1, .3]] * 3)
            after, _, _ = build_examples(root, ("test",))
            self.assertFalse(np.allclose(before[0]["targets"], after[0]["targets"]))
            for method, kernels in before[0]["kernels"].items():
                for bandwidth, values in kernels.items():
                    np.testing.assert_array_equal(values, after[0]["kernels"][method][bandwidth])


class IsometricAnalysisTests(unittest.TestCase):
    def test_helper_hash_ignores_unused_factory_but_detects_executed_body_changes(self):
        first = "def helper(x):\n    return x + 1\n\ndef train():\n    old_factory()\n"
        changed_factory = first.replace("old_factory", "new_factory")
        changed_helper = first.replace("x + 1", "x + 2")
        self.assertEqual(function_ast_hashes(first, ["helper"]),
                         function_ast_hashes(changed_factory, ["helper"]))
        self.assertNotEqual(function_ast_hashes(first, ["helper"]),
                            function_ast_hashes(changed_helper, ["helper"]))

    def test_incomplete_run_guard_requires_terminal_marker_and_every_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = dict(config=dict(seeds="11,22,33,44,55", phase="confirmation",
                                        domain="real", methods="isometric_shape"), records=[])
            path = root / "manifest.json"
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "elapsed_seconds absent"):
                load_completed_manifest(root, "real")
            manifest["elapsed_seconds"] = 1.
            manifest["records"] = [dict(seed=s, method="isometric_shape") for s in [11, 22, 33, 44]]
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "missing or duplicate"):
                load_completed_manifest(root, "real")
            manifest["records"].append(dict(seed=55, method="isometric_shape"))
            path.write_text(json.dumps(manifest))
            self.assertEqual(len(load_completed_manifest(root, "real")["records"]), 5)
            manifest["records"].append(dict(seed=55, method="isometric_shape"))
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "missing or duplicate"):
                load_completed_manifest(root, "real")

    def test_crossed_bootstrap_keeps_same_camera_draw_across_identical_seeds(self):
        identities = [dict(scene="s", q_id=q) for q in [1, 2, 2, 2]]
        a = np.repeat([[4., -1., -1., -1.]], 5, axis=0)
        result = paired_camera_seed_statistics(a, np.zeros_like(a), identities)
        self.assertAlmostEqual(result["mean"], .25)
        np.testing.assert_allclose(result["ci95_seed"], [.25, .25])
        np.testing.assert_array_equal(result["ci95_camera"], result["ci95_seed_camera"])
        self.assertEqual(result["query_cameras"], 2)

    def test_crossed_bootstrap_is_seed_only_when_camera_differences_constant(self):
        identities = [dict(scene="s", q_id=q) for q in [1, 2, 2]]
        a = np.repeat(np.arange(1., 6.)[:, None], 3, axis=1)
        result = paired_camera_seed_statistics(a, np.zeros_like(a), identities)
        self.assertAlmostEqual(result["mean"], 3.)
        np.testing.assert_allclose(result["ci95_camera"], [3., 3.])
        np.testing.assert_array_equal(result["ci95_seed"], result["ci95_seed_camera"])

    def test_nested_point_bootstrap_preserves_ratio_not_scene_mean(self):
        numerator = np.repeat([[4., -3.]], 5, axis=0)
        denominator = np.repeat([[1., 3.]], 5, axis=0)
        result = paired_point_statistics(numerator, denominator)
        self.assertAlmostEqual(result["mean"], .25)
        self.assertNotAlmostEqual(result["mean"], (4. - 1.) / 2)
        np.testing.assert_allclose(result["ci95_seed"], [.25, .25])


if __name__ == "__main__":
    unittest.main()
