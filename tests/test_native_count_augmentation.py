"""Tests for changed sampling and decisions, not a repeated PE-formula audit."""
import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from experiments import native_count_augmentation as exp
from experiments import native_count_augmentation_report as reporting


class CountAugmentationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        exp.runner.configure()

    def tearDown(self):
        exp.SCHEDULE = None

    def test_formula_initialization_rng_and_explicit_prefix(self):
        args = exp.runner.synthetic_inputs(2, 512, 'cpu')
        for role, factory in (('relative5', exp.CompactNativeClassifier), ('distance_mlp', exp.NativeGSClassifier)):
            for seed in exp.SEEDS:
                torch.manual_seed(seed)
                model = exp.model_for(role)
                state = torch.get_rng_state().clone()
                torch.manual_seed(seed)
                reference = factory(role)
                self.assertTrue(torch.equal(state, torch.get_rng_state()))
                self.assertEqual(list(model.state_dict()), list(reference.state_dict()))
                self.assertTrue(all(torch.equal(v, reference.state_dict()[k]) for k, v in model.state_dict().items()))
            exp.SCHEDULE = exp.CountSchedule(33)
            with torch.no_grad():
                for n in (256, 512):
                    exp.SCHEDULE.set_tokens(n)
                    torch.manual_seed(123)
                    actual = model.train()(*args)
                    torch.manual_seed(123)
                    expected = reference.train()(*(x[:, :n] for x in args))
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                    self.assertEqual(exp.SCHEDULE.calls, 1)
                torch.testing.assert_close(model.eval()(*args), reference.eval()(*args), atol=0, rtol=0)

    def test_full_epoch_stream_and_actual_prefix_are_paired(self):
        class Probe:
            training = True
        for seed in exp.SEEDS:
            schedule = exp.CountSchedule(seed)
            rng = np.random.default_rng(seed + 65000)
            reference_rng = np.random.default_rng(seed + 65000)
            digest = hashlib.sha256(('R65-sampler-v1:' + str(seed)).encode()).hexdigest()
            counts = []
            for actual, expected in zip(schedule.batches(rng, 6227), exp.ORIGINAL_EPOCH_BATCHES(reference_rng, 6227)):
                for a, b in zip(actual, expected):
                    np.testing.assert_array_equal(a, b)
                index, subset, quaternion, jitter = actual
                observed = schedule.prefix(Probe(), (subset, jitter, subset))
                n = 256 if len(counts) % 2 == 0 else 512
                for a, b in zip(observed, (subset[:, :n], jitter[:, :n], subset[:, :n])):
                    np.testing.assert_array_equal(a, b)
                counts.append(observed[0].shape[1])
                digest = exp.runner.stream_update(digest, *actual)
            self.assertEqual(counts.count(256), 195)
            self.assertEqual(counts.count(512), 195)
            self.assertEqual(rng.bit_generator.state, reference_rng.bit_generator.state)
            self.assertEqual(schedule.completed, 1)
            for role, root in exp.PRIOR.items():
                old = exp.read(root / 'runs' / role / str(seed) / 'history.json')
                self.assertEqual(digest, old['training'][0]['stream_sha256'])

    def test_resume_reproduces_effective_input_record(self):
        class Probe:
            training = True
        def consume(schedule, rng):
            for index, subset, quaternion, jitter in schedule.batches(rng, 32):
                schedule.prefix(Probe(), (subset, jitter, subset))
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            rng = np.random.default_rng(65011)
            a = exp.CountSchedule(11, directory)
            consume(a, rng)
            saved = rng.bit_generator.state
            consume(a, rng)
            record = exp.read(directory / 'epoch_0002.json')
            recovered_rng = np.random.default_rng()
            recovered_rng.bit_generator.state = saved
            b = exp.CountSchedule(11, directory, completed=1)
            consume(b, recovered_rng)
            self.assertEqual(record, exp.read(directory / 'epoch_0002.json'))
            self.assertEqual(record['updates_by_tokens'], {'256': 1, '512': 1})
            self.assertEqual(record['token_exposures'], 16 * (256 + 512))
            self.assertEqual(rng.bit_generator.state, recovered_rng.bit_generator.state)

    def test_selection_stays_512_and_evaluation_does_not_advance_schedule(self):
        class Probe(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(1))
                self.counts = []
                self.register_forward_pre_hook(exp.training_prefix)
            def forward(self, c, s, a):
                self.counts.append(c.shape[1])
                return torch.zeros(len(c), 60) + self.anchor
        bundle = dict(centers=np.zeros((4, 2048, 3), dtype='float32'),
            covariance=np.tile(np.eye(3), (4, 2048, 1, 1)),
            appearance=np.zeros((4, 2048, 4), dtype='float32'), labels=np.arange(4), ids=np.arange(4))
        exp.SCHEDULE = exp.CountSchedule(11)
        exp.SCHEDULE.set_tokens(256)
        model = Probe()
        exp.runner.evaluate(model, bundle)
        for variant in exp.VARIANTS:
            exp.runner.evaluate(model, bundle, variant)
        self.assertEqual(model.counts, [512, 512, 256, 1024])
        self.assertEqual(exp.SCHEDULE.calls, 0)

    def test_unconsumed_batch_cannot_be_reported_as_training(self):
        schedule = exp.CountSchedule(11)
        stream = schedule.batches(np.random.default_rng(1), 32)
        next(stream)
        with self.assertRaises(AssertionError):
            next(stream)

    def test_contrast_signs_baseline_catchup_and_fixed_statistics(self):
        rows = []
        for regime in reporting.REGIMES:
            for role in exp.ROLES:
                for seed in exp.SEEDS:
                    for variant in exp.VARIANTS:
                        value = .70 + (.04 if role == 'relative5' else 0)
                        if regime == 'mixed256512':
                            value += .01 if role == 'relative5' else .03
                        if variant == 'sparse256':
                            value -= .01 if role == 'relative5' else .02
                        rows.append(dict(regime=regime, role=role, seed=seed, variant=variant,
                            metrics=dict(class_accuracy=[value] * 60, mca=value, oa=value, ce=1-value)))
        result = reporting.contrasts(rows, replicates=1000)
        self.assertEqual(len(result), 18)
        self.assertEqual([r['bootstrap_seed'] for r in result], list(range(84001, 84019)))
        for r in result:
            if r['kind'] == 'method_gap':
                expected = (4 if r['regime'] == 'fixed512' else 2) + (1 if r['variant'] == 'sparse256' else 0)
            elif r['kind'] == 'gap_change':
                expected = -2
            elif r['kind'] == 'training_effect':
                expected = 1 if r['role'] == 'relative5' else 3
            elif r['kind'] == 'sparse_extra_gap':
                expected = 1
            else:
                expected = 0
            self.assertAlmostEqual(r['mean_gain_pp'], expected)
            np.testing.assert_allclose(r['crossed_seed_class_bootstrap_95_pp'], [expected] * 2, atol=1e-12)
            np.testing.assert_allclose(r['paired_oa_gain_pp'], [expected] * 3, atol=1e-12)
            np.testing.assert_allclose(r['paired_ce_difference'], [-expected / 100] * 3, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
