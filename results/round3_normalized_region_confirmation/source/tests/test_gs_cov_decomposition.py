import unittest

import numpy as np

from experiments.gs_cov_decomposition import (
    covariance_statistics, fit_regions_decomposed, floor_covariance,
)
from experiments.real_support_data import fit_regions


class GSCovarianceDecompositionTests(unittest.TestCase):
    def setUp(self):
        self.K=np.array([[10.,0,5],[0,10.,5],[0,0,1.]])
        self.settings=dict(width=12,height=12,grid=(1,1),min_points=2,eigen_floor=1e-8)

    def test_exact_law_of_total_covariance_and_original_fit(self):
        means=np.array([[0.,0,2.],[.01,0,2.]])
        cov=np.stack([np.diag([.01,.02,.03]),np.diag([.04,.05,.06])])
        opacity=np.array([.8,.4])
        args=(means,cov,opacity,self.K,np.eye(3),np.zeros(3))
        result=fit_regions_decomposed(*args,**self.settings)
        original=fit_regions(*args,**self.settings)
        np.testing.assert_array_equal(result['mean'],original['mean'])
        np.testing.assert_allclose(result['full'],original['cov'],atol=1e-15)
        np.testing.assert_array_equal(result['raw_total'],result['raw_center']+result['raw_internal'])
        np.testing.assert_allclose(result['raw_internal'][0],np.diag([.02,.03,.04]))
        np.testing.assert_allclose(result['raw_center'][0],np.diag([2/9*.01**2,0,0]))
        np.testing.assert_allclose(result['centeronly'][0],np.diag([2/9*.01**2,1e-8,1e-8]))

    def test_visibility_and_weights_independent_of_covariance(self):
        rng=np.random.default_rng(744)
        means=np.c_[rng.uniform(-.4,.4,(30,2)),rng.uniform(1.9,2.1,30)]
        opacity=rng.uniform(.1,1,30)
        C=np.broadcast_to(np.eye(3)*.01,(30,3,3)).copy()
        a=fit_regions_decomposed(means,C,opacity,self.K,np.eye(3),np.zeros(3),**self.settings)
        b=fit_regions_decomposed(means,C*100,opacity,self.K,np.eye(3),np.zeros(3),**self.settings)
        for field in ['mean','point_count','valid','raw_center','zbuffer']:
            np.testing.assert_array_equal(a[field],b[field])
        np.testing.assert_allclose(b['raw_internal'],a['raw_internal']*100)

    def test_fixed_floor_and_unchanged_mean_all_counterfactuals(self):
        means=np.array([[0.,0,2.],[0.,0,2.]])
        cov=np.stack([np.diag([0.,.02,.03])]*2)
        result=fit_regions_decomposed(means,cov,np.ones(2),self.K,np.eye(3),np.zeros(3),**self.settings)
        np.testing.assert_allclose(result['centeronly'],result['flooronly'])
        np.testing.assert_allclose(result['full'],result['internalonly'])
        self.assertGreaterEqual(np.linalg.eigvalsh(result['full']).min(),.999e-8)
        stats=covariance_statistics(result)
        self.assertAlmostEqual(stats['internal_trace_fraction_mean'],.05/(.05+1e-8))

    def test_random_fit_replays_original_including_invalid_regions(self):
        rng=np.random.default_rng(443)
        means=np.c_[rng.uniform(-.9,.9,(100,2)),rng.uniform(1.,4.,100)]
        base=rng.normal(size=(100,3,3))
        cov=base@base.swapaxes(-1,-2)*.001
        opacity=rng.uniform(0,1,100)
        settings=dict(width=12,height=12,grid=(3,3),min_points=5,eigen_floor=1e-7)
        a=fit_regions_decomposed(means,cov,opacity,self.K,np.eye(3),np.zeros(3),**settings)
        b=fit_regions(means,cov,opacity,self.K,np.eye(3),np.zeros(3),**settings)
        for field in ['mean','point_count','valid','zbuffer']:
            np.testing.assert_array_equal(a[field],b[field])
        np.testing.assert_allclose(a['full'],b['cov'],atol=1e-15)
        self.assertTrue((~a['valid']).any())


if __name__=='__main__':
    unittest.main()
