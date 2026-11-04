"""Known-label fixtures for interpretation of cross-source classification errors."""
import unittest
import numpy as np
from experiments.round65_source_diagnostic import decompose, paired_contrasts
from experiments.round65_common import ROLES, SEEDS, VARIANTS


class SourceDiagnosticTests(unittest.TestCase):
    def fixture(self):
        labels = np.arange(60)
        logits = np.eye(60)*2
        logits[0, 31] = 4  # Wrong source; correct after source restriction.
        logits[1, 2] = 4  # Wrong class in the correct source.
        logits[30, 1] = 5; logits[30, 31] = 4  # Wrong even after source restriction.
        return logits, labels

    def test_error_partition_and_source_known_limit(self):
        logits, labels = self.fixture(); before = logits.copy()
        d = decompose(logits, labels)
        for key, count in [('accuracy', 57), ('cross_source_error', 2),
                           ('within_source_error', 1), ('source_known_accuracy', 58)]:
            self.assertAlmostEqual(d['mca'][key], count/60)
        for c in range(60):
            self.assertAlmostEqual(sum(d['class_rates'][k][c] for k in
                ('accuracy', 'cross_source_error', 'within_source_error')), 1.)
        self.assertAlmostEqual(d['domains']['MACGS']['mca']['accuracy'], 28/30)
        self.assertAlmostEqual(d['domains']['ModelNet']['mca']['accuracy'], 29/30)
        np.testing.assert_array_equal(logits, before)

    def test_class_weighting_is_not_object_weighting(self):
        x, y = self.fixture()
        d = decompose(np.concatenate([x, np.repeat(x[:1], 9, axis=0)]), np.r_[y, np.zeros(9, dtype=int)])
        self.assertAlmostEqual(d['mca']['accuracy'], 57/60)
        self.assertAlmostEqual(d['object_mean']['accuracy'], 57/69)

    def test_paired_signs_and_complete_coverage(self):
        x, y = self.fixture(); control = decompose(x, y)
        good = decompose(np.eye(60)*2, y)
        bad_x = x.copy(); bad_x[2, 32] = 4; bad = decompose(bad_x, y)
        lookup = {(r, s, v): control for r in ROLES for s in SEEDS for v in VARIANTS}
        lookup['gaussian_relation', 11, 'base'] = good
        lookup['gaussian_relation', 22, 'base'] = bad
        row = next(r for r in paired_contrasts(lookup) if r['control'] == 'direct_mlp' and r['variant'] == 'base')
        np.testing.assert_allclose([r['gain_pp'] for r in row['paired']], [5, -100/60, 0], atol=1e-12)
        for r in row['paired']:
            self.assertAlmostEqual(r['gain_pp'], r['cross_source_error_reduction_pp']+r['within_source_error_reduction_pp'])
        del lookup['nope', 33, 'tilt']
        with self.assertRaises(ValueError): paired_contrasts(lookup)

    def test_missing_classes_or_nonfinite_predictions_are_rejected(self):
        x, y = self.fixture()
        with self.assertRaises(ValueError): decompose(x[:30], y[:30])
        x[0, 0] = np.nan
        with self.assertRaises(ValueError): decompose(x, y)


if __name__ == '__main__':
    unittest.main()
