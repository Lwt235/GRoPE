import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.native_confirmation_analysis import (
    confirmation_decision, load_prediction_stack, paired_accuracy_summary,
)


class NativeConfirmationAnalysisTests(unittest.TestCase):
    def test_equal_class_endpoint_is_not_object_accuracy(self):
        result = paired_accuracy_summary(np.array([[0, 1, 1, 1]]),
            np.array([[0, 0, 0, 0]]), np.array([0, 0, 0, 1]), replicates=200)
        self.assertAlmostEqual(result['mean_gain_pp'], 100 / 6)
        self.assertAlmostEqual(result['paired_oa_gain_pp'][0], -25)
        self.assertEqual(result['class_counts'], [3, 1])

    def test_identical_seed_effects_retain_shared_object_uncertainty(self):
        y = np.array([0, 0, 0, 1, 1])
        a = np.tile([0, 1, 1, 1, 0], (3, 1))
        b = np.tile([1, 0, 0, 0, 1], (3, 1))
        r = paired_accuracy_summary(a, b, y, replicates=2000)
        np.testing.assert_allclose(r['crossed_seed_object_bootstrap_95_pp'],
                                   r['within_class_object_bootstrap_95_pp'], atol=1e-12)
        np.testing.assert_allclose(r['paired_seed_bootstrap_95_pp'], [r['mean_gain_pp']] * 2)
        self.assertLess(r['crossed_seed_object_bootstrap_95_pp'][0], r['mean_gain_pp'])
        self.assertGreater(r['crossed_seed_object_bootstrap_95_pp'][1], r['mean_gain_pp'])

    def test_homogeneous_objects_leave_only_seed_uncertainty(self):
        y = np.array([0, 0, 0, 1, 1])
        a = np.stack([y, 1-y, y])
        b = np.tile(1-y, (3, 1))
        r = paired_accuracy_summary(a, b, y, replicates=2000)
        np.testing.assert_allclose(r['crossed_seed_object_bootstrap_95_pp'], r['paired_seed_bootstrap_95_pp'])
        np.testing.assert_allclose(r['within_class_object_bootstrap_95_pp'], [100 * 2 / 3] * 2)
        equal = paired_accuracy_summary(a, a, y, replicates=100)
        for key, value in equal.items():
            if key.endswith('bootstrap_95_pp'):
                self.assertEqual(value, [0., 0.])

    def test_prediction_pairing_rejects_reordering_and_changed_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'predictions.npz'
            ids, labels = np.array(['a', 'b']), np.array([0, 1])
            logits = np.zeros((2, 60)); logits[0, 0] = 1; logits[1, 1] = 1
            np.savez(path, ids=ids, labels=labels, logits=logits)
            row = dict(seed=11, path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            np.testing.assert_array_equal(load_prediction_stack([row], ids, labels), [[0, 1]])
            with self.assertRaises(ValueError):
                load_prediction_stack([row], ids[::-1], labels[::-1])
            with self.assertRaises(ValueError):
                load_prediction_stack([row, row], ids, labels)
            with self.assertRaises(ValueError):
                load_prediction_stack([dict(row, sha256='wrong')], ids, labels)

    def test_primary_population_and_seed_cohort_cannot_be_replaced(self):
        row = dict(left='relative5', right='distance_mlp', cohort='new5',
            seeds=[44, 55, 66, 77, 88], variant='base',
            group='no_known_historical_source_overlap', count=3845,
            seed_pairs=5, classes=list(range(60)), mean_gain_pp=.8,
            crossed_seed_object_bootstrap_95_pp=[.1, 1.5])
        self.assertTrue(confirmation_decision(row)['primary_support'])
        self.assertFalse(confirmation_decision(dict(row, mean_gain_pp=.4))['primary_support'])
        self.assertFalse(confirmation_decision(dict(row, crossed_seed_object_bootstrap_95_pp=[-.1, 1.5]))['primary_support'])
        with self.assertRaises(ValueError):
            confirmation_decision(dict(row, group='full_population', count=6090))
        with self.assertRaises(ValueError):
            confirmation_decision(dict(row, cohort='all8'))
        with self.assertRaises(ValueError):
            paired_accuracy_summary(np.zeros((2, 0), int), np.zeros((2, 0), int), np.array([], int))


if __name__ == '__main__':
    unittest.main()
