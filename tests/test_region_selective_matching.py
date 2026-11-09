"""Meaningful checks for added all-query selection semantics."""
import unittest
import numpy as np
from experiments import region_selective_matching as m


class SelectiveMatchingTests(unittest.TestCase):
    def test_both_directions_and_shift(self):
        logits = np.array([[[4., 1., 2.], [3., 5., 0.], [2., 3., 4.]]])
        overlap = np.array([[[.8, 0., 0.], [0., .5, 0.], [0., 0., 0.]]])
        q = m.queries(logits, overlap)
        for direction in range(2):
            for i in range(3):
                scores = logits[0, i] if direction == 0 else logits[0, :, i]
                expected = np.exp(scores.max()) / np.exp(scores).sum()
                self.assertAlmostEqual(q['confidence'][0, direction, i], expected)
        shifted = m.queries(logits + 1000, overlap)
        np.testing.assert_allclose(q['confidence'], shifted['confidence'])
        self.assertEqual(int(q['valid'].sum()), 4)
        self.assertEqual(int(q['correct'].sum()), 4)

    def test_threshold_keeps_ties_and_maximal_feasible_set(self):
        scores = np.array([.9, .8, .8, .7, .6])
        correct = np.array([True, False, True, False, True])
        self.assertEqual(m.threshold_for_budget(scores, correct, .2), .8)
        self.assertEqual(m.threshold_for_budget(scores, correct, 0), .9)
        self.assertEqual(m.threshold_for_budget(scores, correct, .4), 0.)
        self.assertGreater(m.threshold_for_budget(np.ones(4), np.zeros(4, bool), .5), 1.)

    def test_unmatched_and_wrong_valid_are_errors(self):
        q = dict(valid=np.array([True, True, False, False]), correct=np.array([True, False, False, False]))
        r = m.statistics(q, np.array([1., 1., 1., 0.]))
        self.assertEqual(r['correct_yield'], .25)
        self.assertEqual(r['wrong_yield'], .5)
        self.assertEqual(r['unmatched_acceptance'], .5)
        self.assertEqual(r['wrong_valid_accepted'], 1.)
        self.assertAlmostEqual(r['precision'], 1 / 3)
        self.assertEqual(r['valid_recall'], .5)

    def test_zero_acceptance_and_zero_denominators(self):
        q = dict(valid=np.zeros(3, bool), correct=np.zeros(3, bool))
        r = m.statistics(q, np.zeros(3))
        self.assertIsNone(r['precision'])
        self.assertIsNone(r['valid_recall'])
        self.assertEqual(r['correct_yield'], 0.)
        self.assertEqual(r['unmatched_acceptance'], 0.)

    def test_fixed_coverage_ties_are_permutation_invariant(self):
        score = np.array([.9, .8, .8, .8, .1])
        w = m.fixed_coverage_weights(score, .4)
        np.testing.assert_allclose(w, [1, 1/3, 1/3, 1/3, 0])
        perm = np.array([3, 0, 4, 2, 1])
        np.testing.assert_allclose(m.fixed_coverage_weights(score[perm], .4), w[perm])
        self.assertEqual(m.fixed_coverage_weights(score, 0).sum(), 0)
        self.assertEqual(m.fixed_coverage_weights(score, 1).sum(), 5)

    def test_scene_macro_does_not_weight_large_scenes_more(self):
        q = dict(valid=np.ones((3, 1, 2), bool), correct=np.array([[[1, 1]], [[0, 0]], [[0, 0]]], bool))
        r = m.group_statistics(q, np.array(['a', 'b', 'b']), np.ones((3, 1, 2)))
        self.assertEqual(r['macro']['correct_yield'], .5)
        self.assertAlmostEqual(r['micro']['correct_yield'], 1/3)

    def test_paired_contrast_has_correct_direction_and_budget(self):
        rows = []
        for cohort in m.COHORTS:
            for condition in m.CONDITIONS:
                for role in m.ROLES:
                    for seed in m.SEEDS:
                        values = dict(correct_yield=.12 if role == 'relative5' else .10,
                                      wrong_yield=.04 if role == 'relative5' else .05)
                        scenes = {s: dict(values) for s in ('a', 'b')}
                        rows.append(dict(cohort=cohort, condition=condition, role=role, seed=seed,
                            operating_points=[{}, dict(metrics=dict(by_scene=scenes)), {}]))
        comparisons = m.compare(rows)
        self.assertEqual(len(comparisons), 16)
        for row in comparisons:
            self.assertEqual(row['budget'], .05)
            np.testing.assert_allclose(row['mean_differences_pp'], [2., -1.])
            np.testing.assert_allclose(row['crossed_seed_scene_95_pp'], [[2., 2.], [-1., -1.]])


if __name__ == '__main__':
    unittest.main()
