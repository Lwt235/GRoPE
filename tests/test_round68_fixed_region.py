import unittest
import numpy as np
import torch
from experiments.round58_data import tokenize,random_rotation
from experiments.round68_fixed_region_data import thin_regions,shuffle_indices,geometry_scores
from models.round68_fixed_region import FixedRegionMatcher,ROLES
from experiments.round58_run import shared_digest


class FixedRegionChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        rng=np.random.default_rng(68001)
        points=torch.tensor(rng.normal(size=(1,2,512,3)),dtype=torch.float32)
        cls.batch=dict(points=points,world=points.clone(),clean=points.clone())
        cls.reference=tokenize(cls.batch)

    def test_sparse_keeps_reference_and_matches_direct_moments(self):
        sparse=thin_regions(self.batch,self.reference,['unit-pair'])
        self.assertTrue(torch.equal(sparse['overlap'],self.reference['overlap']))
        self.assertTrue(torch.equal(sparse['world_centers'],self.reference['world_centers']))
        self.assertTrue(torch.equal(sparse['retained'].sum(-1),torch.full((1,2),256)))
        for view in range(2):
            keep=sparse['retained'][0,view].numpy()
            self.assertTrue(keep[self.reference['anchors'][0,view].numpy()].all())
            for token in (0,17,63):
                ids=self.reference['neighbors'][0,view,token].numpy()
                points=self.batch['points'][0,view].numpy()[ids[keep[ids]]].astype(np.float64)
                center=points.mean(0);delta=points-center
                expected=delta.T@delta/len(points)+.02**2*np.eye(3)
                np.testing.assert_allclose(center,sparse['centers'][0,view,token].numpy(),atol=3e-7)
                np.testing.assert_allclose(expected,sparse['covariance'][0,view,token].numpy(),atol=3e-7)

    def test_geometry_strata_scores_ignore_independent_view_frames(self):
        c=self.reference['centers'].double();s=self.reference['covariance'].double()
        rotation=torch.tensor(np.stack([random_rotation(np.random.default_rng(v+10)) for v in range(2)]))
        moved=c@rotation.transpose(-1,-2)+torch.tensor([.2,-.5,.7])
        covariance=rotation[:,None]@s@rotation[:,None].transpose(-1,-2)
        np.testing.assert_allclose(geometry_scores(c,s),geometry_scores(moved,covariance),atol=2e-7,rtol=2e-7)

    def test_shared_content_and_identity_shuffle_control(self):
        common=[];models=[]
        for role in ROLES:
            torch.manual_seed(11);model=FixedRegionMatcher(role).eval();common.append(shared_digest(model));models.append(model)
        self.assertEqual(len(set(common)),1)
        f,c,s=(self.reference[k] for k in ('features','centers','covariance'))
        identity=torch.arange(64).expand(1,2,64)
        with torch.no_grad():
            a=models[3](f,c,s,identity);b=models[4](f,c,s,identity)
        torch.testing.assert_close(a,b,atol=0,rtol=0)
        permutation=shuffle_indices(['unit-pair'])
        self.assertTrue((permutation!=np.arange(64)).all())
        np.testing.assert_array_equal(np.sort(permutation,axis=-1),np.broadcast_to(np.arange(64),(1,2,64)))


if __name__=='__main__':unittest.main()
