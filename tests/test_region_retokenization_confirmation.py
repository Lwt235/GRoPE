"""New cache path and missing-cohort handling, using only old observations."""
import unittest
import numpy as np
import torch

from experiments.region_retokenization_confirmation import old_conversion_check, convert, PARENT, load_arrays
from experiments.region_retokenization_confirmation_report import population_metrics


class ConfirmationTests(unittest.TestCase):
    def test_three_constructions_reproduce_existing_old_data(self):
        record=old_conversion_check()
        self.assertTrue(record['passed'])
        self.assertEqual(len(record['records']),3)
        self.assertTrue(all(r['errors']['overlap']==0 for r in record['records']))

    def test_labels_do_not_change_forward_geometry(self):
        sample=load_arrays(PARENT/'data/val_sample.npz')
        ids=load_arrays(PARENT/'data/val.npz')['ids'][:2]
        first=convert(sample,ids)
        altered=dict(sample,world=sample['world']*2+7,clean=sample['clean']*2+7)
        second=convert(altered,ids)
        for condition in first:
            for key in ('features','centers','covariance','permutation'):
                np.testing.assert_array_equal(first[condition][key],second[condition][key])

    def test_unobserved_scene_is_not_an_observed_zero_score(self):
        truth=np.zeros((2,4,4));truth[0,0,0]=1
        cache=dict(overlap=truth,world_centers=np.zeros((2,2,4,3)),scenes=np.array(['a','b']))
        result=population_metrics(np.broadcast_to(10*np.eye(4),(2,4,4)),cache,['a','b','missing'])
        self.assertEqual(result['macro']['region_recall'],1)
        self.assertEqual(result['macro']['valid_and_correct_fraction'],.125)
        self.assertEqual(result['by_scene']['b']['valid_and_correct_fraction'],0)
        self.assertIsNone(result['by_scene']['missing']['valid_and_correct_fraction'])
        self.assertTrue(result['by_scene']['missing']['unobserved'])
        self.assertEqual(result['missing_scenes'],['missing'])


if __name__=='__main__':
    torch.set_num_threads(4)
    unittest.main()
