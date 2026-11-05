import unittest
import numpy as np
from experiments.native_confirmation_runtime import population_groups, summarize_logits


class NativeConfirmationRuntimeTests(unittest.TestCase):
    def test_reuse_groups_follow_ids_not_manifest_order(self):
        rows = [dict(file_path='0/a.ply', model_id='a', source='MACGS', taxonomy_id=0),
                dict(file_path='30/b.ply', model_id='b', source='ModelNet', taxonomy_id=30),
                dict(file_path='31/c.ply', model_id='c', source='ModelNet', taxonomy_id=31)]
        statuses = ['not_ModelNet', 'used_original_train', 'original_test_not_used_in_R48_R58']
        membership = [dict(row, historical_status=status) for row, status in zip(rows, statuses)][::-1]
        groups = population_groups(rows, membership)
        self.assertEqual(groups['historical_source_reused'].tolist(), [False, True, False])
        self.assertEqual(groups['no_known_historical_source_overlap'].tolist(), [True, False, True])
        with self.assertRaises(ValueError):
            population_groups(rows, membership[:-1])
        broken = [dict(r) for r in membership]
        broken[0]['historical_status'] = 'unresolved'
        with self.assertRaises(ValueError):
            population_groups(rows, broken)

    def test_class_equal_accuracy_and_empty_strata(self):
        labels = np.array([0, 0, 0, 1])
        logits = np.zeros((4, 60))
        logits[np.arange(4), [0, 1, 1, 1]] = 2.
        groups = dict(all_official=np.ones(4, bool), absent=np.zeros(4, bool))
        result = summarize_logits(logits, labels, groups)
        self.assertAlmostEqual(result['all_official']['metrics']['mca'], 2 / 3)
        self.assertAlmostEqual(result['all_official']['metrics']['oa'], .5)
        self.assertEqual(result['all_official']['class_count'], 2)
        self.assertEqual(result['absent']['count'], 0)
        self.assertIsNone(result['absent']['metrics'])
        with self.assertRaises(ValueError):
            summarize_logits(logits, labels, dict(wrong=np.ones(3, bool)))


if __name__ == '__main__':
    unittest.main()
