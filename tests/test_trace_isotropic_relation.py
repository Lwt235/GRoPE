import unittest
import numpy as np
import torch
from models.round69_compact_relation import CompactMatcher, descriptors
from models.round73_native_relative5 import CompactNativeClassifier
from models.trace_isotropic_relation import trace_covariance, TraceIsotropicMatcher, TraceIsotropicNative


class TraceIsotropicTests(unittest.TestCase):
    @staticmethod
    def geometry():
        rng = np.random.default_rng(81201)
        c = rng.normal(size=(2, 9, 3)) * .2
        a = rng.normal(size=(2, 9, 3, 3)) * .12
        s = a @ a.swapaxes(-1, -2) + .002 * np.eye(3)
        return torch.from_numpy(c), torch.from_numpy(s)

    def test_independent_scalar_formula_and_trace(self):
        c, s = self.geometry()
        iso = trace_covariance(s)
        np.testing.assert_allclose(iso.diagonal(dim1=-2,dim2=-1).sum(-1),s.diagonal(dim1=-2,dim2=-1).sum(-1),atol=1e-15)
        d2 = np.square(c.numpy()[:,:,None] - c.numpy()[:,None,:]).sum(-1)
        r = np.trace(s.numpy(),axis1=-2,axis2=-1) / 3
        expected = np.stack((np.sqrt(d2)/.2,np.log1p(d2/r[:,:,None]),np.log1p(d2/r[:,None,:]),
            np.log1p(3*r[:,None,:]/r[:,:,None]),np.log1p(3*r[:,:,None]/r[:,None,:])),-1)
        np.testing.assert_allclose(descriptors('relative5',c,iso).numpy(),expected,rtol=1e-11,atol=1e-11)
        torch.testing.assert_close(trace_covariance(iso),iso)
        # No covariance is modified in place, and distinct local scales survive.
        self.assertGreater(float((iso[...,0,0]-iso[:,0:1,0,0]).abs().max()),0)

    def test_rotations_and_anisotropic_information_removed(self):
        c,s = self.geometry()
        q = torch.linalg.qr(torch.tensor([[1.,2.,3.],[2.,-1.,4.],[3.,1.,-1.]],dtype=torch.float64)).Q
        torch.testing.assert_close(trace_covariance(q@s@q.T),trace_covariance(s),rtol=1e-12,atol=1e-12)
        a = descriptors('relative5',c,trace_covariance(s))
        b = descriptors('relative5',c@q.T+.7,trace_covariance(q@s@q.T))
        torch.testing.assert_close(a,b,rtol=1e-10,atol=1e-10)
        self.assertGreater(float((a-descriptors('relative5',c,s)).abs().max()),.01)

    def test_full_initial_state_and_rng_match(self):
        for original,new in [(CompactMatcher,TraceIsotropicMatcher),(CompactNativeClassifier,TraceIsotropicNative)]:
            for seed in (11,22,33):
                torch.manual_seed(seed);a=original('relative5');rng=torch.get_rng_state()
                torch.manual_seed(seed);b=new()
                self.assertTrue(torch.equal(rng,torch.get_rng_state()))
                self.assertEqual(a.state_dict().keys(),b.state_dict().keys())
                for key,value in a.state_dict().items():
                    self.assertTrue(torch.equal(value,b.state_dict()[key]),key)
                self.assertEqual(sum(p.numel() for p in b.position.parameters()),9792)

    def test_common_content_unchanged_and_position_learns(self):
        c,s = self.geometry();c=c.float();appearance=torch.zeros(2,9,4)
        torch.manual_seed(11);a=CompactNativeClassifier('relative5')
        torch.manual_seed(11);b=TraceIsotropicNative()
        inputs=[]
        handles=[m.stem.register_forward_pre_hook(lambda mod,args:inputs.append(args[0].detach().clone())) for m in (a,b)]
        frozen_s=s.clone();a(c,s,appearance);out=b(c,s,appearance)
        torch.testing.assert_close(inputs[0],inputs[1],atol=0,rtol=0)
        torch.testing.assert_close(s,frozen_s,atol=0,rtol=0)
        out.square().mean().backward()
        for p in b.position.parameters():
            self.assertTrue(torch.isfinite(p.grad).all());self.assertGreater(float(p.grad.norm()),0)
        for handle in handles:handle.remove()


if __name__ == '__main__':
    torch.set_num_threads(4)
    unittest.main()
