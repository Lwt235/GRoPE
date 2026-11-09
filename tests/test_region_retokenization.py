"""Denominator and source checks for the finite regional scope test."""
import unittest
import numpy as np
import torch

from experiments.region_retokenization import metrics, paired_summary, load_arrays, R60


class RetokenizationTests(unittest.TestCase):
    def test_empty_scene_counts_in_all_query_metric(self):
        overlap=np.zeros((2,4,4));overlap[0,0,0]=1
        cache=dict(overlap=overlap,world_centers=np.zeros((2,2,4,3)),scenes=np.array(['a','b']))
        scores=np.broadcast_to(10*np.eye(4),(2,4,4))
        result=metrics(scores,cache)
        self.assertEqual(result['micro']['valid_queries'],2)
        self.assertEqual(result['micro']['total_queries'],16)
        self.assertEqual(result['macro']['region_recall'],1)
        self.assertEqual(result['macro']['coverage'],.125)
        self.assertEqual(result['macro']['valid_and_correct_fraction'],.125)
        self.assertIsNone(result['by_scene']['b']['region_recall'])
        self.assertEqual(result['zero_valid_pairs'],1)

    def test_conditional_gain_can_coexist_with_lower_absolute_output(self):
        # Different query populations must not be compared as paired successes.
        dense=np.broadcast_to(np.eye(4),(1,4,4)).copy()
        sparse=np.zeros((1,4,4));sparse[0,0,0]=1
        world=np.zeros((1,2,4,3));scenes=np.array(['a'])
        score=10*np.eye(4)[None];score[0,1,1]=0;score[0,1,2]=11
        a=metrics(score,dict(overlap=dense,world_centers=world,scenes=scenes))
        b=metrics(score,dict(overlap=sparse,world_centers=world,scenes=scenes))
        self.assertGreater(b['macro']['region_recall'],a['macro']['region_recall'])
        self.assertLess(b['macro']['valid_and_correct_fraction'],a['macro']['valid_and_correct_fraction'])

    def test_comparison_sign_and_zero_uncertainty_for_constant_effect(self):
        for effect in (-2.,0.,3.):
            result=paired_summary(np.full((3,5),effect),88001)
            self.assertEqual(result['mean_gain_pp'],effect)
            self.assertEqual(result['crossed_seed_scene_bootstrap_95_pp'],[effect,effect])
            self.assertEqual(result['positive_seed_pairs'],3 if effect>0 else 0)

    def test_existing_sparse_input_has_no_dense_anchor_protection(self):
        base=load_arrays(R60/'data/eval_base_sample.npz')
        sparse=load_arrays(R60/'data/eval_sparse256_sample.npz')
        for key in ('points','sensor','source_ids','world','clean'):
            np.testing.assert_array_equal(sparse[key],base[key][:,:,:256])
        self.assertTrue(np.any(base['token_anchors']>=256))
        self.assertLess(int(sparse['token_neighbors'].max()),256)
        self.assertEqual(sparse['token_neighbors'].shape,(2,2,64,32))


if __name__=='__main__':
    torch.set_num_threads(4)
    unittest.main()
