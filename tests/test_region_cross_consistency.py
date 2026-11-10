import unittest
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from models.region_cross_consistency import CrossConsistencyMatcher, ConsistencyPosition, relation_energy, ROLES
from models.region_fixed_hrope import FixedHRegionMatcher


class CrossConsistencyChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(99500)

    def test_zero_coefficients_restore_backbone_and_common_gradients(self):
        b,n=2,7
        features=torch.randn(b,2,n,11);centers=torch.randn(b,2,n,3)
        z=torch.randn(b,2,n,3,3);cov=z@z.transpose(-1,-2)+torch.eye(3)*.2
        permutation=torch.arange(n).expand(b,2,n)
        intrinsics=torch.eye(3).expand(b,2,3,3).clone();intrinsics[...,0,0]=.8;intrinsics[...,1,1]=.7
        xy=torch.rand(b,2,n,2)
        args=(features,centers,cov,permutation,intrinsics,xy)
        torch.manual_seed(11);base=FixedHRegionMatcher();expected=base(*args)
        expected.square().mean().backward()
        common=dict(base.named_parameters())
        for role in ROLES:
            torch.manual_seed(11);model=CrossConsistencyMatcher(role);actual=model(*args)
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)
            actual.square().mean().backward()
            for name,param in model.named_parameters():
                if name.startswith('position.'):continue
                torch.testing.assert_close(param,common[name],rtol=0,atol=0)
                torch.testing.assert_close(param.grad,common[name].grad,rtol=0,atol=1e-6)
            gradient=model.position.coefficients.grad
            self.assertTrue(torch.isfinite(gradient).all());self.assertGreater(float(gradient.norm()),0)

    def test_explicit_reference_sum_and_gradient_without_detach(self):
        n=5;c=3
        source=torch.rand(1,n,n,c,dtype=torch.float64)
        target=torch.rand(1,n,n,c,dtype=torch.float64)
        logits=torch.randn(1,n,n,dtype=torch.float64,requires_grad=True)
        p=logits.softmax(-1);actual=relation_energy(source,target,p)
        rows=[]
        for i in range(n):
            columns=[]
            for k in range(n):
                total=torch.zeros(c,dtype=torch.float64);mass=0
                for j in range(n):
                    if j==i:continue
                    for ell in range(n):
                        total=total+p[0,j,ell]*(source[0,i,j]-target[0,k,ell]).square()
                        mass=mass+p[0,j,ell]
                columns.append(total/mass)
            rows.append(torch.stack(columns))
        expected=torch.stack(rows)[None]
        torch.testing.assert_close(actual,expected,rtol=1e-10,atol=1e-12)
        a=torch.autograd.grad(actual.square().sum(),logits,retain_graph=True)[0]
        b=torch.autograd.grad(expected.square().sum(),logits,retain_graph=True)[0]
        torch.testing.assert_close(a,b,rtol=1e-9,atol=1e-12)
        self.assertGreater(float(a.norm()),0)
        own=torch.autograd.grad(actual[0,1].sum(),logits)[0]
        torch.testing.assert_close(own[0,1],torch.zeros(n,dtype=torch.float64),atol=1e-12,rtol=0)

    def test_relation_rigid_invariance_and_permutation(self):
        rng=np.random.default_rng(99501);n=7
        centers=torch.from_numpy(rng.normal(size=(2,n,3)))
        z=rng.normal(size=(2,n,3,3));cov=torch.from_numpy(z@z.swapaxes(-1,-2)+np.eye(3)*.3)
        position=ConsistencyPosition('cross_full5').double();phi=position(centers,cov)
        changed=[]
        for v in range(2):
            r=torch.from_numpy(Rotation.from_rotvec(rng.normal(size=3)).as_matrix());t=torch.from_numpy(rng.normal(size=3))
            changed.append(position(centers[v]@r.T+t,r@cov[v]@r.T))
        torch.testing.assert_close(torch.stack(changed),phi,rtol=1e-10,atol=1e-10)
        p=torch.randn(1,n,n,dtype=torch.float64).softmax(-1)
        before=relation_energy(phi[None,0],phi[None,1],p)
        a,b=torch.randperm(n),torch.randperm(n)
        after=relation_energy(phi[None,0][:,a][:,:,a],phi[None,1][:,b][:,:,b],p[:,a][:,:,b])
        torch.testing.assert_close(after,before[:,a][:,:,b],rtol=1e-10,atol=1e-10)


if __name__=='__main__':unittest.main()
