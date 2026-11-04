"""Old-data checks of the new wrapper and missing-population accounting."""
import copy
import unittest

import numpy as np
import torch

from experiments.region_confirmation_common import ROOT, PARENT, PREFIXES, read, select_places
from experiments.region_confirmation_data import build_arrays
from experiments.region_confirmation_report import population_metrics


class ConfirmationTests(unittest.TestCase):
    def test_new_data_wrapper_reproduces_old_validation(self):
        lock = read(ROOT / 'results/round60_real_scene/attempt1/lock.json')
        entry = copy.deepcopy(next(e for e in lock['entries'] if e['split'] == 'val'))
        entry['split'], entry['pairs'] = 'old_eval', entry['pairs'][:2]
        local = dict(lock, entries=[entry])
        thresholds = read(PARENT / 'data/strata_thresholds.json')['thresholds']
        base, sparse, sample, record = build_arrays(local, thresholds)
        with np.load(PARENT / 'data/val.npz', allow_pickle=False) as old:
            for key in old.files:
                if key in ('ids', 'scenes', 'strata', 'permutation', 'overlap'):
                    np.testing.assert_array_equal(base[key], old[key][:2])
                else:
                    np.testing.assert_allclose(base[key], old[key][:2], atol=2e-5, rtol=2e-5)
        self.assertEqual(record['excluded'], [])
        for key in ('ids', 'overlap', 'world_centers', 'strata'):
            np.testing.assert_array_equal(base[key], sparse[key])
        self.assertTrue(np.all(sample['sparse_retained'].sum(-1) == 256))
        self.assertTrue(np.any(base['covariance'] != sparse['covariance']))

    def test_missing_scene_and_empty_stratum_are_not_zero_scores(self):
        with np.load(PARENT / 'data/val.npz', allow_pickle=False) as z:
            cache = {k: z[k][:2].copy() for k in z.files}
        cache['strata'][:] = 0
        metrics = population_metrics(np.zeros_like(cache['overlap']), cache,
                                     [str(cache['scenes'][0]), 'missing/place'])
        self.assertEqual(metrics['by_scene']['missing/place']['valid_queries'], 0)
        self.assertIsNone(metrics['by_scene']['missing/place']['region_recall'])
        self.assertEqual(metrics['evaluable_scenes'], 1)
        self.assertIsNone(metrics['strata']['1']['macro']['region_recall'])

    def test_metadata_selection_excludes_all_used_places_and_is_order_independent(self):
        places = [prefix + '_' + str(i) for prefix in PREFIXES for i in range(6)]
        excluded = places[::3]
        selected = select_places(places, excluded)
        self.assertEqual(selected, select_places(list(reversed(places)), excluded))
        self.assertFalse(set(selected) & set(excluded))
        self.assertEqual(len(set(selected)), 10)
        self.assertTrue(all(sum(p.startswith(prefix + '_') for p in selected) == 2 for prefix in PREFIXES))


if __name__ == '__main__':
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    unittest.main()
