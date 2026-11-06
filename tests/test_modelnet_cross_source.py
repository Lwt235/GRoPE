"""Scientific boundary checks for exact archive identity and the full class head."""
import unittest
import numpy as np
from experiments.modelnet_cross_source import choose_member, virtual_path
from experiments.native_confirmation_runtime import summarize_logits
from experiments.native_confirmation_analysis import paired_accuracy_summary


class CrossSourceTests(unittest.TestCase):
    def test_exact_object_and_latest_numeric_iteration(self):
        names = ['chair/chair_0001/point_cloud/iteration_9000/point_cloud.ply',
                 'chair/chair_0001/point_cloud/iteration_10000/point_cloud.ply',
                 'chair/chair_00010/point_cloud/iteration_90000/point_cloud.ply',
                 'chair/chair_0001/input.ply']
        self.assertEqual(choose_member(names,'chair_0001'),names[1])
        self.assertEqual(virtual_path(dict(taxonomy_id=12,model_id='chair_0001')),'12/chair_0001.ply')

    def test_ambiguous_missing_and_path_escape_are_rejected(self):
        with self.assertRaises(ValueError):
            choose_member(['x/chair_0001.ply','y/chair_0001.ply'],'chair_0001')
        with self.assertRaises(ValueError):
            choose_member(['chair_00010.ply'],'chair_0001')
        with self.assertRaises(ValueError):
            choose_member(['../chair_0001.ply'],'chair_0001')

    def test_full_head_does_not_mask_outside_classes(self):
        labels = np.array([1,1,15])
        logits = np.zeros((3,60));logits[:,42]=10;logits[np.arange(3),labels]=9
        result = summarize_logits(logits,labels,dict(all=np.ones(3,dtype=bool)))
        self.assertEqual(result['all']['metrics']['mca'],0.)
        self.assertEqual(result['all']['metrics']['oa'],0.)

    def test_class_equal_weighting_and_seed_pairing(self):
        labels = np.array([1,1,1,15])
        a = np.array([[1,1,1,42],[1,1,42,15]])
        b = np.array([[42,42,42,15],[1,1,42,15]])
        result = paired_accuracy_summary(a,b,labels,replicates=20,seed=80001)
        self.assertAlmostEqual(result['mean_gain_pp'],0.)
        self.assertEqual(result['paired_seed_gains_pp'],[0.,0.])
        self.assertEqual(result['paired_oa_gain_pp'],[50.,0.])


if __name__ == '__main__':
    unittest.main()
