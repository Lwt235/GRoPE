import unittest
import numpy as np
from experiments.round68_fixed_region_run import query_values,aggregate,grouped_metrics
from experiments.round58_run import metric_torch


class SceneMetrics(unittest.TestCase):
    def test_scene_macro_and_query_micro_are_distinct(self):
        world=np.zeros((3,2,4,3));world[:,:,:,0]=np.arange(4)
        truth=np.broadcast_to(np.eye(4),(3,4,4)).copy()
        scores=10*truth.copy();scores[1]=np.roll(scores[1],1,axis=1)
        cache=dict(overlap=truth,world_centers=world,scenes=np.array(['a','a','b']),
                   strata=np.broadcast_to(np.array([0,1,0,1]),(3,2,4)))
        result=grouped_metrics(scores,cache)
        self.assertAlmostEqual(result['micro']['region_recall'],2/3)
        self.assertAlmostEqual(result['macro']['region_recall'],.75)
        self.assertEqual(result['strata']['2']['evaluable_scenes'],0)
        self.assertIsNone(result['strata']['2']['macro']['region_recall'])
        independent=metric_torch(scores,truth,world)
        for key in ('region_recall','center_pck','soft_epe','ce'):
            self.assertAlmostEqual(result['micro'][key],independent[key],places=12)

    def test_unscorable_queries_remain_in_coverage_denominator(self):
        truth=np.zeros((1,4,4));truth[0,0,0]=1
        world=np.zeros((1,2,4,3));scores=np.broadcast_to(np.eye(4)*10,(1,4,4))
        cache=dict(overlap=truth,world_centers=world)
        result=aggregate(query_values(scores,cache))
        self.assertEqual(result['valid_queries'],2)
        self.assertEqual(result['total_queries'],8)
        self.assertEqual(result['region_recall'],1)
        self.assertEqual(result['coverage'],.25)
        self.assertEqual(result['valid_and_correct_fraction'],.25)


if __name__=='__main__':unittest.main()
