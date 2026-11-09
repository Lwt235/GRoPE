import itertools
import unittest
import numpy as np
import torch

from models.normalized_precision_relation import (
    NormalizedPrecisionMatcher, NormalizedPrecisionNative, normalized_precision_descriptors)
from models.round69_compact_relation import CompactMatcher, descriptors
from models.round73_native_relative5 import CompactNativeClassifier


class NormalizedPrecisionTests(unittest.TestCase):
    @staticmethod
    def geometry():
        rng = np.random.default_rng(87001)
        c = rng.normal(size=(2, 7, 3)) * .2
        a = rng.normal(size=(2, 7, 3, 3)) * .12
        s = a @ a.swapaxes(-1, -2) + .002*np.eye(3)
        return torch.from_numpy(c), torch.from_numpy(s)

    def test_independent_solve_formula_and_scalar_factorization(self):
        c, s = self.geometry()
        expected = np.empty((2, 7, 7, 5))
        for batch in range(2):
            for i in range(7):
                for j in range(7):
                    a, b = s[batch, i].numpy(), s[batch, j].numpy()
                    pa, pb = [np.trace(x)/3*np.mean(1/np.linalg.eigvalsh(x)) for x in (a,b)]
                    delta = (c[batch, i] - c[batch, j]).numpy()
                    terms = [delta@np.linalg.solve(a,delta)/pa, delta@np.linalg.solve(b,delta)/pb,
                             np.trace(np.linalg.solve(a,b))/pa if i != j else 3.,
                             np.trace(np.linalg.solve(b,a))/pb if i != j else 3.]
                    expected[batch,i,j] = np.r_[np.linalg.norm(delta)/.2, np.log1p(terms)]
        n = normalized_precision_descriptors(c, s)
        np.testing.assert_allclose(n.numpy(), expected, rtol=1e-11, atol=1e-11)
        p = s.diagonal(dim1=-2,dim2=-1).mean(-1)*torch.linalg.inv(s).diagonal(dim1=-2,dim2=-1).mean(-1)
        f = descriptors('relative5', c, s)
        off = ~torch.eye(7, dtype=torch.bool)
        for slot in (1,3):
            torch.testing.assert_close(torch.expm1(n[...,slot])[:,off]*p.unsqueeze(-1).expand(2,7,7)[:,off],
                                       torch.expm1(f[...,slot])[:,off], rtol=1e-11, atol=1e-11)

    def test_sphere_self_and_finite_direction_average(self):
        c, s = self.geometry()
        n = normalized_precision_descriptors(c,s)
        f = descriptors('relative5',c,s)
        torch.testing.assert_close(n.diagonal(dim1=-3,dim2=-2),f.diagonal(dim1=-3,dim2=-2),rtol=0,atol=0)
        r = s.diagonal(dim1=-2,dim2=-1).mean(-1)
        sphere = r[...,None,None]*torch.eye(3,dtype=s.dtype)
        torch.testing.assert_close(normalized_precision_descriptors(c,sphere),descriptors('relative5',c,sphere))
        rotations = []
        for permutation in itertools.permutations(range(3)):
            for signs in itertools.product((-1.,1.),repeat=3):
                matrix = np.eye(3)[list(permutation)]*np.asarray(signs)[:,None]
                if np.linalg.det(matrix) > .5:
                    rotations.append(torch.from_numpy(matrix))
        self.assertEqual(len(rotations),24)
        for i,j in ((0,1),(2,4),(5,6)):
            values = []
            for rotation in rotations:
                rotated = s.clone()
                rotated[:,i] = rotation@s[:,i]@rotation.T
                values.append(torch.expm1(normalized_precision_descriptors(c,rotated)[0,i,j,[1,3]]))
            target = torch.stack(((c[0,i]-c[0,j]).square().sum()/r[0,i],3*r[0,j]/r[0,i]))
            torch.testing.assert_close(torch.stack(values).mean(0),target,rtol=1e-11,atol=1e-11)

    def test_rigid_permutation_and_direction_sensitivity(self):
        c,s = self.geometry()
        q = torch.linalg.qr(torch.tensor([[1.,2.,3.],[2.,-1.,4.],[3.,1.,-1.]],dtype=s.dtype)).Q
        value = normalized_precision_descriptors(c,s)
        torch.testing.assert_close(value,normalized_precision_descriptors(c@q.T+.7,q@s@q.T),rtol=1e-10,atol=1e-10)
        order = torch.tensor([3,0,6,1,5,2,4])
        torch.testing.assert_close(value[:,order][:,:,order],normalized_precision_descriptors(c[:,order],s[:,order]))
        rng = np.random.default_rng(87002)
        rotations = torch.from_numpy(np.linalg.qr(rng.normal(size=(2,7,3,3)))[0])
        changed = normalized_precision_descriptors(c,rotations@s@rotations.transpose(-1,-2))
        self.assertGreater(float((value-changed).abs().max()),.01)

    def test_matched_state_rng_and_budget(self):
        for original,new in ((CompactMatcher,NormalizedPrecisionMatcher),(CompactNativeClassifier,NormalizedPrecisionNative)):
            for seed in (11,22,33):
                torch.manual_seed(seed)
                reference = original('relative5')
                rng = torch.get_rng_state()
                torch.manual_seed(seed)
                model = new()
                self.assertTrue(torch.equal(rng,torch.get_rng_state()))
                self.assertEqual(reference.state_dict().keys(),model.state_dict().keys())
                for key,value in reference.state_dict().items():
                    self.assertTrue(torch.equal(value,model.state_dict()[key]),key)
                self.assertEqual(sum(p.numel() for p in model.position.parameters()),9792)

    def test_common_content_finite_gradients(self):
        c,s = self.geometry()
        c = c.float()
        frozen = s.clone()
        native_args = (c,s,torch.zeros(2,7,4))
        region_args = (torch.zeros(1,2,7,11),c[None],s[None],torch.arange(7)[None,None].expand(1,2,7))
        for original,new,args in ((CompactNativeClassifier,NormalizedPrecisionNative,native_args),
                                  (CompactMatcher,NormalizedPrecisionMatcher,region_args)):
            torch.manual_seed(11)
            reference = original('relative5')
            torch.manual_seed(11)
            model = new()
            seen = []
            hooks = [m.stem.register_forward_pre_hook(lambda module,inputs:seen.append(inputs[0].detach().clone()))
                     for m in (reference,model)]
            reference(*args)
            result = model(*args)
            torch.testing.assert_close(seen[0],seen[1],rtol=0,atol=0)
            torch.testing.assert_close(frozen,s,rtol=0,atol=0)
            result.square().mean().backward()
            for parameter in model.position.parameters():
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(float(parameter.grad.norm()),0.)
            for hook in hooks:
                hook.remove()


if __name__ == '__main__':
    torch.set_num_threads(4)
    unittest.main()
