import copy
import itertools
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.round5_analysis import (analyze_rows, crossed_bootstrap,
    DEFAULT_SEEDS, leave_out, load_evaluation_file, validate_rows)


def row(method, seed, split, scene, pair, count, pck, ce=1.):
    return dict(method=method, seed=seed, split=split, scene=scene, pair_id=pair,
                q_index=pair*2, k_index=pair*2+1, n_valid=count,
                ce_sum=ce*count, pck8_count=pck, pck4_count=pck,
                epe_argmax_sum=(count-pck)*8, epe_soft_sum=(count-pck)*6)


def fixture():
    rows = []
    test_places = ['harvard_c11', 'hotel_umd', 'mit_lab_hj', 'mit_76_studyroom', 'mit_dorm_next_sj']
    for seed in DEFAULT_SEEDS:
        for method, pck in [('point', 60), ('fixed', 70), ('normalized', 75)]:
            rows.append(row(method, seed, 'val', 'harvard_c8/scan', 0, 100, pck))
        for place in test_places:
            # Point is stronger on test; validation must still select fixed.
            for method, pck in [('point', 90), ('fixed', 70), ('normalized', 85)]:
                rows.append(row(method, seed, 'test', place+'/scan', 1, 100, pck, .9 if method == 'normalized' else 1.))
    return rows


class Round5AnalysisTests(unittest.TestCase):
    def test_loader_accepts_bare_and_trainer_wrapper_and_preserves_provenance(self):
        rows = fixture()
        metadata = dict(selected_step=300, input_sources=[dict(scene='x', input_sha256='abc')],
                        sourcehash={'experiments/round5_train.py': 'def'})
        with tempfile.TemporaryDirectory() as directory:
            for kind, payload in [('bare_list', rows), ('trainer_wrapper', dict(rows=rows, **metadata))]:
                path = Path(directory)/f'{kind}.json'
                raw = (json.dumps(payload, indent=2)+'\n').encode()
                path.write_bytes(raw)
                parsed, audit = load_evaluation_file(path)
                self.assertEqual(parsed, rows)
                self.assertEqual(audit['payload_kind'], kind)
                self.assertEqual(audit['row_count'], len(rows))
                self.assertEqual(audit['sha256'], hashlib.sha256(raw).hexdigest())
                self.assertEqual(audit['metadata'], metadata if kind == 'trainer_wrapper' else {})
            for payload in [{'selected_step': 300}, {'rows': {}}, {'rows': [1]}, 7]:
                path = Path(directory)/'invalid.json'
                path.write_text(json.dumps(payload))
                with self.assertRaises(ValueError):
                    load_evaluation_file(path)

    def test_crossed_bootstrap_identity_permutation_invariance(self):
        values = np.array([[0., 2., 1.], [3., -.5, 2.], [-2., 1., .5]])
        seeds, places = [33, 11, 22], ['mit', 'hotel', 'harvard']
        expected = crossed_bootstrap(values, seeds, places, draws=20000)
        sr, pr = [1, 2, 0], [2, 0, 1]
        got = crossed_bootstrap(values[sr][:, pr], [seeds[i] for i in sr], [places[i] for i in pr], draws=20000)
        self.assertEqual(expected, got)

    def test_crossed_interval_matches_small_independent_enumeration(self):
        values = np.array([[0., 0.], [0., .04]])
        # Enumerate every ordered seed draw and place draw: 4x4 outcomes.
        outcomes = []
        for sd in itertools.product(range(2), repeat=2):
            for pd in itertools.product(range(2), repeat=2):
                outcomes.append(np.mean([values[s, p] for s in sd for p in pd]))
        result = crossed_bootstrap(values, [11, 22], ['a', 'b'], draws=20000)
        self.assertAlmostEqual(result['mean'], .01)
        # These 16 values enumerate a probability distribution; use its inverse
        # CDF, rather than interpolation between 16 ordered sample positions.
        exact_quantiles = np.quantile(outcomes, [.025, .975], method='inverted_cdf')
        np.testing.assert_allclose(result['ci95_crossed'], exact_quantiles, atol=0)

    def test_validation_selection_and_all_controls_are_preserved(self):
        result = analyze_rows(fixture(), draws=20000)
        self.assertEqual(result['baseline_selection']['selected'], 'fixed')
        self.assertFalse(result['baseline_selection']['test_values_used'])
        self.assertEqual(set(result['comparisons']), {'point', 'fixed'})
        self.assertTrue(result['primary_gate']['passed'])
        self.assertLess(result['comparisons']['point']['pck8']['mean'], 0)
        self.assertAlmostEqual(result['primary_gate']['pck8_difference'], .15)
        loo = result['comparisons']['fixed']['pck8']['leave_institution_out']
        self.assertEqual(set(loo), {'Harvard_c11', 'Hotel', 'MIT'})
        self.assertEqual(loo['MIT']['remaining_places'], 2)
        self.assertNotIn('ci95_crossed', loo['MIT'])

    def test_row_permutation_does_not_change_analysis(self):
        rows = fixture()
        expected = analyze_rows(rows, draws=1000)
        shuffled = [rows[i] for i in np.random.default_rng(9).permutation(len(rows))]
        self.assertEqual(expected, analyze_rows(shuffled, draws=1000))

    def test_equal_place_macro_is_not_query_micro(self):
        rows = []
        for seed in DEFAULT_SEEDS:
            for method in ['point', 'fixed', 'normalized']:
                rows.append(row(method, seed, 'val', 'validation/scan', 0, 100, 50))
                rows.append(row(method, seed, 'test', 'large/scan', 1, 100,
                                60 if method == 'normalized' else 50))
                rows.append(row(method, seed, 'test', 'small/scan', 2, 1,
                                0 if method == 'normalized' else 1))
        result = analyze_rows(rows, min_test_places=2, draws=1000)
        self.assertAlmostEqual(result['comparisons']['point']['pck8']['mean'], -.45)
        self.assertFalse(result['primary_gate']['passed'])

    def test_duplicate_rows_and_mask_changes_are_rejected(self):
        rows = fixture()
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            validate_rows(rows+[rows[0]], DEFAULT_SEEDS)
        changed = copy.deepcopy(rows)
        changed[1]['n_valid'] += 1
        with self.assertRaisesRegex(ValueError, 'n_valid differ'):
            validate_rows(changed, DEFAULT_SEEDS)

    def test_physical_place_split_leak_is_rejected(self):
        rows = fixture()
        rows[0]['scene'] = 'harvard_c11/another_scan'
        with self.assertRaisesRegex(ValueError, 'multiple supplied splits'):
            validate_rows(rows, DEFAULT_SEEDS)

    def test_leaveout_is_descriptive_and_checks_remaining_places(self):
        output = leave_out(np.array([[1., 2., 6.], [3., 4., 8.]]),
                          [11, 22], ['a', 'b', 'c'], {'a': 'X', 'b': 'X', 'c': 'Y'})
        self.assertEqual(output['X']['mean'], 7.)
        self.assertEqual(output['Y']['mean'], 2.5)
        self.assertTrue(output['X']['descriptive_only'])


if __name__ == '__main__':
    unittest.main()
