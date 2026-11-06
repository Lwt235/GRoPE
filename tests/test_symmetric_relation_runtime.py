"""Verify the actual count adapter and analysis additions before new fits."""
import unittest

import numpy as np
import torch

from experiments import symmetric_relation as exp
from experiments.symmetric_relation_report import region_metrics, scene_summary
from experiments.native_seed_replication_report import paired_summary
from models.symmetric_gaussian_relation import SymmetricNativeClassifier


class SymmetricRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        exp.native.configure()

    def test_actual_native_factory_preserves_selected_prefix(self):
        args = exp.native.synthetic_inputs(1, 512, 'cpu')
        for case in ('native256', 'native512'):
            for role in exp.NATIVE_ROLES:
                torch.manual_seed(11)
                adapted = exp.native_model(case, role).train()
                torch.manual_seed(11)
                plain = SymmetricNativeClassifier(role).train()
                self.assertEqual(list(adapted.state_dict()), list(plain.state_dict()))
                count = int(case[6:])
                with torch.no_grad():
                    torch.manual_seed(123)
                    actual = adapted(*args)
                    torch.manual_seed(123)
                    expected = plain(*(x[:, :count] for x in args))
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    adapted.eval()
                    plain.eval()
                    torch.testing.assert_close(adapted(*args), plain(*args), rtol=0, atol=0)

    def test_validation_count_and_explicit_condition_semantics(self):
        class Probe(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(1))
                self.observed = []

            def forward(self, centers, covariance, appearance):
                self.observed.append((centers.shape[1], self.training))
                return torch.zeros(len(centers), 60) + self.anchor

        bundle = dict(centers=np.zeros((4, 2048, 3), dtype='float32'),
            covariance=np.tile(np.eye(3), (4, 2048, 1, 1)),
            appearance=np.zeros((4, 2048, 4), dtype='float32'), labels=np.arange(4), ids=np.arange(4))
        for case, default in (('native256', 256), ('native512', 512)):
            probe = Probe()
            exp.native_evaluate(case, probe, bundle)
            for condition in ('base', 'sparse256', 'dense1024'):
                exp.native_evaluate(case, probe, bundle, condition)
            self.assertEqual(probe.observed, [(default, False), (512, False), (256, False), (1024, False)])
            self.assertEqual(exp.native_selection_batch(case, bundle, 0, 4, 'base', 'cpu')[0].shape[1], default)

    def test_hard_epe_uses_valid_queries_and_scene_macro(self):
        logits = np.array([[[3., 0.], [0., 2.]], [[0., 2.], [3., 0.]], [[2., 0.], [0., 3.]]])
        overlap = np.array([[[.8, 0.], [0., .7]], [[.9, 0.], [0., 0.]], [[0., .6], [0., 0.]]])
        world = np.array([[[[0., 0., 0.], [1., 0., 0.]], [[.2, 0., 0.], [1.1, 0., 0.]]],
                          [[[0., 0., 0.], [2., 0., 0.]], [[.5, 0., 0.], [3., 0., 0.]]],
                          [[[0., 0., 0.], [4., 0., 0.]], [[1., 0., 0.], [5., 0., 0.]]]])
        cache = dict(overlap=overlap, world_centers=world, scenes=np.array(['A', 'A', 'B']),
                     strata=np.arange(12).reshape(3, 2, 2) % 4)
        result = region_metrics(logits, cache)
        old = exp.region.engine.grouped_metrics(logits, cache)
        independent = exp.region.engine.metric_torch(logits, overlap, world)
        self.assertAlmostEqual(result['micro']['hard_epe'], independent['hard_epe'], places=12)
        expected_by_scene = {'A': [], 'B': []}
        for index in range(3):
            for direction in (0, 1):
                scores = logits[index] if direction == 0 else logits[index].T
                truth = overlap[index] if direction == 0 else overlap[index].T
                for query in range(2):
                    if truth[query].max() >= .3:
                        picked = scores[query].argmax()
                        expected_by_scene[cache['scenes'][index]].append(float(np.linalg.norm(
                            world[index, direction, query] - world[index, 1 - direction, picked])))
        self.assertEqual(result['micro']['valid_queries'], 8)
        self.assertAlmostEqual(result['macro']['hard_epe'], np.mean([np.mean(v) for v in expected_by_scene.values()]))
        self.assertNotAlmostEqual(result['macro']['hard_epe'], result['micro']['hard_epe'])
        for group in (None, '0', '1', '2', '3'):
            a, b = (result, old) if group is None else (result['strata'][group], old['strata'][group])
            for key in ('region_recall', 'center_pck', 'soft_epe', 'ce'):
                self.assertEqual(a['macro'][key], b['macro'][key])
                self.assertEqual(a['micro'][key], b['micro'][key])

    def test_scene_bootstrap_matches_existing_crossed_formula(self):
        values = np.random.default_rng(50).normal(size=(3, 60))
        scenes = scene_summary(values, 79001, 1000)
        classes = paired_summary(values, 1000, 79001)
        for a, b in (('mean_gain_pp', 'mean_gain_pp'), ('paired_seed_gains_pp', 'paired_seed_gains_pp'),
                     ('paired_scene_bootstrap_95_pp', 'paired_class_bootstrap_95_pp'),
                     ('crossed_seed_scene_bootstrap_95_pp', 'crossed_seed_class_bootstrap_95_pp')):
            self.assertEqual(scenes[a], classes[b])
        zero = scene_summary(np.zeros((3, 5)), 79001, 100)
        self.assertEqual(zero['crossed_seed_scene_bootstrap_95_pp'], [0., 0.])

    def test_sources_include_actual_training_and_data_modules(self):
        sources = {str(p.relative_to(exp.ROOT)) for p in exp.source_files()}
        self.assertTrue({'experiments/round65_run.py', 'experiments/round65_audit.py',
            'experiments/round65_data.py', 'experiments/round68_fixed_region_run.py',
            'experiments/round68_fixed_region_data.py', 'models/symmetric_gaussian_relation.py',
            'experiments/symmetric_relation_report.py'} <= sources)


if __name__ == '__main__':
    unittest.main()
