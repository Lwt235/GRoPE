"""Pure toy checks; never decode a saved archive or execute a model."""
from __future__ import annotations

import copy
import math
import unittest
from unittest.mock import patch

import numpy as np
from scipy.special import logsumexp

from experiments import round38_saved_output_diagnostic as P
from experiments import round38_saved_output_diagnostic_audit as A


def toy(mu=(.4375, .4375, 1.), weights=(.7, .1, .1, .1), bias=(0., 0., 0., 0.),
        constant=False, zero=False):
    pixels = np.asarray([[x, y, x + 8, y + 8] for y in range(0, 144, 8)
                         for x in range(0, 192, 8)], np.int64)
    bounds = (pixels.astype(np.float64) - .5) / 8.
    mu = np.asarray(mu, np.float64)
    factor = np.diag([.2, .2, 0. if constant else .3]).astype(np.float64)
    mass = np.full(432, 1e-100, np.float64)
    mass[[0, 1, 24, 25]] = weights
    raw = np.full(432, -np.inf, np.float64) if zero else np.log(mass)
    floor = np.logaddexp(raw, math.log(1e-30))
    given_bias = bias
    bias = np.zeros(432, np.float64)
    bias[[0, 1, 24, 25]] = given_bias
    logits = floor + bias
    logp = logits - logsumexp(logits)
    probability = np.exp(logp)
    rms = math.hypot(float(mu[2]), *map(float, factor[2]))
    branch = ('zero_depth_rms' if rms == 0 else
              'nonpositive_constant_depth' if constant and mu[2] <= 0 else
              'positive_constant_depth_gaussian' if constant else 'variable_depth')
    spec = dict(id='toy__g24__q0000', scene=P.SCENES[0], pair=[0, 1],
                pair_id='toy', query_index=0)
    return dict(
        spec=spec,
        record=dict(**spec, status='ok', rms_depth=rms,
                    structural_branch=branch, structural_zero=zero,
                    front_mass=0. if zero else 1. if constant else .99),
        arrays=dict(input__mu=mu, input__factor=factor, input__bounds=bounds,
                    input__content_bias=bias, log_raw_mass=raw, log_floored_mass=floor,
                    log_probability=logp, probability=probability,
                    argmax=np.asarray(probability.argmax(), np.int32)),
        pair=dict(anchor_homogeneous=mu.copy(), anchored_factor=factor.copy(),
                  old_a2=np.asarray([.4375, .4375], np.float64), tau32=np.float32(.25),
                  key_bounds=bounds.copy(), key_bounds_px=pixels))


class SavedOutputDiagnostics(unittest.TestCase):
    def setUp(self):
        # A real archive load in any nominally pure calculation is a hard test
        # failure, even if it comes through a helper imported by either module.
        self.loader_guard = patch.object(np, 'load', side_effect=AssertionError('toy cannot load archives'))
        self.loader_guard.start()
        self.addCleanup(self.loader_guard.stop)

    def check_both(self, payload):
        produced = P.diagnose_row(payload)
        independent = A.independent_row(payload)
        self.assertEqual(A.compare(independent, produced), [])
        return produced

    def test_closed_boundary_ties_have_four_anchor_owners(self):
        row = self.check_both(toy(mu=(.9375, .9375, 1.), weights=(.25,) * 4))
        self.assertEqual(row['projection']['anchor_cells'], [0, 1, 24, 25])
        self.assertTrue(row['projection']['cell_boundary'])
        self.assertEqual(row['stages']['raw']['maximizers'], [0, 1, 24, 25])
        self.assertTrue(row['stages']['raw']['anchor_first_contains'])
        self.assertEqual(row['stages']['raw']['rectangle_distance_px'], 0.)
        self.assertAlmostEqual(row['stages']['raw']['cell_center_distance_px'], 8. / math.sqrt(2.))

    def test_content_flip_has_exact_geometry_content_account(self):
        row = self.check_both(toy(bias=(0., 3., 0., 0.)))
        self.assertEqual(row['content']['set_transition'], '10')
        self.assertEqual(row['content']['first_transition'], '10')
        self.assertTrue(row['content']['winner_changed'])
        self.assertAlmostEqual(row['content']['geometry_loss'], math.log(7.))
        self.assertEqual(row['content']['content_gain'], 3.)
        self.assertAlmostEqual(row['content']['logit_advantage'], 3. - math.log(7.))
        self.assertAlmostEqual(row['content']['bookkeeping_residual'], 0.)
        self.assertEqual(row['stages']['raw']['rectangle_distance_px'], 0.)
        self.assertEqual(row['stages']['logits']['rectangle_distance_px'], 4.)

    def test_set_membership_distinguishes_first_index_tie(self):
        row = self.check_both(toy(mu=(1.4375, .4375, 1.), weights=(.4, .4, .1, .1)))
        self.assertTrue(row['stages']['raw']['anchor_set_contains'])
        self.assertFalse(row['stages']['raw']['anchor_first_contains'])
        self.assertEqual(row['stages']['raw']['rectangle_distance_px'], 4.)
        self.assertEqual(row['stages']['raw']['min_rectangle_distance_px'], 0.)

    def test_structural_zero_has_no_raw_winner_and_keeps_content(self):
        row = self.check_both(toy(mu=(0., 0., 0.), constant=True, zero=True,
                                  bias=(0., 1., 0., 0.)))
        self.assertEqual(row['projection']['projection_status'], 'zero_depth')
        self.assertFalse(row['projection']['primary_anchor_defined'])
        self.assertFalse(row['stages']['raw']['defined'])
        self.assertIsNone(row['stages']['raw']['first'])
        self.assertIsNone(row['distribution']['raw_entropy'])
        self.assertIsNone(row['arithmetic']['raw_to_floor_winner_changed'])
        self.assertTrue(row['content']['winner_changed'])

    def test_nonfront_outside_and_overflow_preserve_general_winners(self):
        cases = [
            (toy(mu=(-.4375, -.4375, -1.)), 'finite', 'interior'),
            (toy(mu=(.4375, .4375, 0.)), 'zero_depth', 'undefined'),
            (toy(mu=(25., .4375, 1.)), 'finite', 'outside'),
            (toy(mu=(1e308, .5, 1e-308)), 'nonfinite_division', 'undefined'),
        ]
        for payload, projection, fov in cases:
            with self.subTest(projection=projection, fov=fov):
                row = self.check_both(payload)
                self.assertEqual(row['projection']['projection_status'], projection)
                self.assertEqual(row['projection']['fov_status'], fov)
                self.assertFalse(row['projection']['primary_anchor_defined'])
                self.assertIsNotNone(row['stages']['probability']['first'])
                self.assertIsNone(row['stages']['probability']['rectangle_distance_px'])

    def test_materialization_tie_keeps_saved_actual_first_winner(self):
        # Locate adjacent representable toy log-probabilities whose exponentials
        # coincide.  No model parameters, dataset cases or task metrics are used.
        pair = None
        for index in range(2048):
            low = np.float64(-.72 - index * 1e-6)
            high = np.nextafter(low, np.inf)
            if np.exp(low) == np.exp(high):
                pair = (low, high)
                break
        self.assertIsNotNone(pair, 'expected an elementary binary64 exp rounding tie')
        low, high = pair
        remainder = (1. - float(np.exp(low)) - float(np.exp(high))) / 430.
        payload = toy()
        logs = np.full(432, math.log(remainder), np.float64)
        logs[:2] = (low, high)
        arrays = payload['arrays']
        arrays['log_raw_mass'] = logs.copy()
        arrays['log_floored_mass'] = np.logaddexp(logs, math.log(1e-30))
        arrays['log_probability'] = logs.copy()
        arrays['probability'] = np.exp(logs)
        arrays['argmax'] = np.asarray(arrays['probability'].argmax(), np.int32)
        row = self.check_both(payload)
        self.assertEqual(row['stages']['log_probability']['first'], 1)
        self.assertEqual(row['stages']['probability']['first'], 0)
        self.assertTrue(row['arithmetic']['log_probability_to_probability_winner_changed'])

    def test_floor_rounding_tie_is_separate_from_content_flip(self):
        payload = toy()
        arrays = payload['arrays']
        raw = np.full(432, -1001., np.float64)
        raw[1] = -1000.
        arrays['log_raw_mass'] = raw
        arrays['log_floored_mass'] = np.logaddexp(raw, math.log(1e-30))
        arrays['log_probability'] = arrays['log_floored_mass'] - logsumexp(arrays['log_floored_mass'])
        arrays['probability'] = np.exp(arrays['log_probability'])
        arrays['argmax'] = np.asarray(arrays['probability'].argmax(), np.int32)
        row = self.check_both(payload)
        self.assertEqual(row['stages']['raw']['first'], 1)
        self.assertEqual(row['stages']['floor']['tie_count'], 432)
        self.assertTrue(row['arithmetic']['raw_to_floor_winner_changed'])
        self.assertTrue(row['arithmetic']['raw_to_floor_maximizers_changed'])
        self.assertFalse(row['content']['winner_changed'])

    def test_all_transition_cells_and_failed_population_denominators(self):
        rows = [self.check_both(payload) for payload in (
            toy(), toy(bias=(0., 3., 0., 0.)),
            toy(mu=(1.4375, .4375, 1.), bias=(0., 3., 0., 0.)),
            toy(mu=(1.4375, .4375, 1.)),
            toy(mu=(0., 0., 0.), constant=True, zero=True))]
        for index, row in enumerate(rows):
            row['id'] = 'toy_' + str(index)
        for status in ('failed', 'unattempted'):
            rows.append(dict(id='toy_' + status, scene=P.SCENES[0],
                             pair=[0, 1], pair_id='toy', query_index=len(rows),
                             status=status, reason='synthetic retained failure'))
        actual, expected = P.summarize(rows), A.independent_summary(rows)
        self.assertEqual(A.compare(expected, actual), [])
        self.assertEqual(actual['all']['population'], 7)
        self.assertEqual(actual['all']['successful_rows'], 5)
        self.assertEqual(actual['all']['numerics']['distribution.raw_entropy']['n'], 4)
        self.assertEqual(actual['all']['numerics']['distribution.floor_entropy']['n'], 5)
        self.assertEqual(actual['by_scene'][P.SCENES[0]]['population'], 7)
        for kind in ('set_transition', 'first_transition'):
            self.assertEqual([actual[kind][state]['population'] for state in ('11', '10', '01', '00')], [1] * 4)
        self.assertAlmostEqual(actual['all']['numerics']['depth.sigma_z']['mean'], .24)
        self.assertEqual(actual['all']['numerics']['depth.sigma_z']['p50'], .3)

    def test_identity_corruption_and_invalid_saved_algebra_fail(self):
        payload = toy()
        payload['pair']['key_bounds_px'][0, 0] += 1
        for calculate in (P.diagnose_row, A.independent_row):
            with self.assertRaises(ValueError):
                calculate(payload)
        payload = toy()
        payload['arrays']['log_floored_mass'][0] += 1e-5
        for calculate in (P.diagnose_row, A.independent_row):
            with self.assertRaises(ValueError):
                calculate(payload)

    def test_auditor_detects_wrong_tie_even_if_numeric_drift_is_tiny(self):
        expected = A.independent_row(toy(weights=(.25,) * 4))
        changed = copy.deepcopy(expected)
        changed['stages']['raw']['maximizers'] = [0]
        changed['stages']['raw']['tie_count'] = 1
        self.assertTrue(A.compare(expected, changed))


if __name__ == '__main__':
    unittest.main()
