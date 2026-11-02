"""New data/adapter checks; already-verified positional formulas are not repeated."""
import unittest
import numpy as np
import torch
from experiments.round5_rgbd_data import decode_sun3d_depth
from experiments.round60_data import backproject,sample_inputs,attach_supervision
from experiments.round58_data import tokenize,random_rotation


class NativeTests(unittest.TestCase):
    def test_native_decode_and_backprojection(self):
        mm=np.array([[1000,2000,0],[3000,4000,5000]],dtype=np.uint32)
        encoded=(((mm<<3)|(mm>>13))&65535).astype(np.uint16)
        depth=decode_sun3d_depth(encoded)
        k=np.array([[2.,0,1.],[0,4.,.5],[0,0,1.]])
        c=backproject(depth,k)
        np.testing.assert_array_equal(c['pixel_ids'],[0,1,3,4,5])
        np.testing.assert_allclose(c['points'],[[-.5,-.125,1.],[0,-.25,2.],[-1.5,.375,3.],[0,.5,4.],[2.5,.625,5.]])
        bad=depth.copy();bad[0,0]=np.nan;bad[1,0]=-1
        np.testing.assert_array_equal(backproject(bad,k)['pixel_ids'],[1,4,5])

    def clouds(self):
        rng=np.random.default_rng(60001)
        return [dict(points=rng.normal(size=(700,3))+[0,0,3],pixel_ids=np.arange(700)) for _ in range(2)]

    def test_nested_density_and_pose_boundary(self):
        clouds=self.clouds()
        for augment in (False,True):
            base=sample_inputs(clouds,'synthetic',0,512,augment)
            sparse=sample_inputs(clouds,'synthetic',0,256,augment)
            for key in ('points','sensor','source_ids'):
                np.testing.assert_array_equal(base[key][:,:256],sparse[key])
            self.assertEqual(float(base['scale']),float(sparse['scale']))
        native=sample_inputs(clouds,'synthetic',0)
        poses=np.tile(np.eye(4),(2,1,1));other=poses.copy();other[1,:3,3]=100.
        a,b=attach_supervision(native,poses),attach_supervision(native,other)
        ta=tokenize({k:torch.from_numpy(v[None]) for k,v in a.items()})
        tb=tokenize({k:torch.from_numpy(v[None]) for k,v in b.items()})
        for key in ('features','centers','covariance'):
            torch.testing.assert_close(ta[key],tb[key],atol=0,rtol=0)
        self.assertGreater(float(ta['overlap'].sum()),0)
        self.assertEqual(float(tb['overlap'].sum()),0)

    def test_coordinate_normalization_and_scoring(self):
        rng=np.random.default_rng(60002);clouds=self.clouds()
        base=sample_inputs(clouds,'synthetic',1)
        moved=[];rotations=[];translations=[]
        for c in clouds:
            r=random_rotation(rng);t=rng.normal(size=3)
            rotations.append(r);translations.append(t)
            moved.append(dict(points=c['points']@r.T+t,pixel_ids=c['pixel_ids']))
        transformed=sample_inputs(moved,'synthetic',1)
        self.assertAlmostEqual(float(base['scale']),float(transformed['scale']),places=12)
        for v,r in enumerate(rotations):
            np.testing.assert_allclose(transformed['points'][v],base['points'][v]@r.T,atol=2e-7,rtol=2e-7)
        poses=np.tile(np.eye(4),(2,1,1));poses[1,:3,:3]=rotations[1];poses[1,:3,3]=translations[1]
        labeled=attach_supervision(base,poses)
        expected=np.stack([base['sensor'][0],base['sensor'][1]@rotations[1].T+translations[1]])/base['scale']
        np.testing.assert_allclose(labeled['world'],expected,atol=3e-7,rtol=2e-7)

    def test_adapter_restores_source_loop_and_config(self):
        from experiments import round58_run as original
        from experiments.round60_run import fitting_adapter
        from experiments.round60_common import OUT,CONFIG
        names=('OUT','CONFIG','ROLES','checked','load_cache','SupportMatcher')
        before={n:getattr(original,n) for n in names};function=original.train
        sentinel=lambda role:None
        with self.assertRaises(RuntimeError):
            with fitting_adapter(11,factory=sentinel):
                self.assertIs(original.train,function);self.assertIs(original.CONFIG,CONFIG)
                self.assertEqual(original.OUT,OUT);self.assertIs(original.SupportMatcher,sentinel)
                raise RuntimeError('exercise restoration')
        for n in names:self.assertIs(getattr(original,n),before[n])


if __name__=='__main__':
    torch.set_num_threads(4);unittest.main()
