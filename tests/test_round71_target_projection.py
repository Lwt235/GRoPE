"""Independent geometry examples, input boundary and upstream adaptation checks."""
import unittest
import numpy as np
import torch

from models.round70_nvs import CompactNVS, TOKENS, HD, HEADS
from models.round71_target_projection import (TargetProjectionNVS, ROLES,
    project_reference, projection_descriptors, prepare_urope, apply_urope)
from third_party.round71_urope.urope_compatible import urope_dot_product_attention
from experiments.round70_nvs_run import digest


def cameras():
    K=torch.tensor([[128.,0,64.],[0,96.,48.],[0,0,1.]],dtype=torch.float64)[None,None].repeat(1,3,1,1)
    c=torch.eye(4,dtype=torch.float64)[None,None].repeat(1,3,1,1)
    c[:,1,0,3]=.75
    return K,c


class Tests(unittest.TestCase):
    def test_projection_independent_jacobian(self):
        K,c=cameras();m=torch.tensor([[[.2,-.3,2.]]],dtype=torch.float64)
        s=torch.diag(torch.tensor([.01,.02,.03],dtype=torch.float64))[None,None]
        u,v,valid=project_reference(m,s,K,c,torch.ones(1,1))
        def project(x):return np.array([x[0]/x[2]+.5,x[1]/x[2]+.5])
        point=m.numpy()[0,0];eps=1e-6
        jac=np.stack([(project(point+np.eye(3)[j]*eps)-project(point-np.eye(3)[j]*eps))/(2*eps) for j in range(3)],axis=-1)
        np.testing.assert_allclose(u[0,0],project(point),atol=1e-12)
        np.testing.assert_allclose(v[0,0],jac@s.numpy()[0,0]@jac.T+np.eye(2)*1e-8,atol=1e-11)
        self.assertTrue(valid.all())
        # World-frame change cancels between points, covariance and cameras.
        a=torch.tensor([[0.,-1,0],[1,0,0],[0,0,1.]],dtype=torch.float64)
        t=torch.tensor([.3,-.5,.7],dtype=torch.float64)
        g=torch.eye(4,dtype=torch.float64);g[:3,:3]=a;g[:3,3]=t
        uu,vv,_=project_reference(m@a.T+t,a@s@a.T,K,g@c,torch.ones(1,1))
        torch.testing.assert_close(u,uu);torch.testing.assert_close(v,vv)

    def test_relation_scalar_formula(self):
        u=torch.tensor([[[.4,.3]]],dtype=torch.float64)
        c=torch.tensor([[[[.02,.004],[.004,.03]]]],dtype=torch.float64)
        got=projection_descriptors(u,c,'relative5')[0,0,0].numpy()
        delta=np.array([4/128,4/96])-u.numpy()[0,0]
        q=np.diag([(8/128)**2/12,(8/96)**2/12]);s=c.numpy()[0,0]
        expected=[np.linalg.norm(delta)/.2,np.log1p(delta@np.linalg.solve(q,delta)),
                  np.log1p(delta@np.linalg.solve(s,delta)),np.log1p(np.trace(np.linalg.solve(q,s))),
                  np.log1p(np.trace(np.linalg.solve(s,q)))]
        np.testing.assert_allclose(got,expected,rtol=2e-7)

    def test_invalid_projection_has_finite_fallback(self):
        K,c=cameras()
        m=torch.tensor([[[0.,0.,0.],[1.,0.,-1.],[0.,0.,1.]]],dtype=torch.float64)
        cov=torch.eye(3,dtype=torch.float64)[None,None].repeat(1,3,1,1)*.01
        u,s,valid=project_reference(m,cov,K,c,torch.tensor([[1.,1.,0.]]))
        self.assertFalse(valid.any())
        for kind in ('distance','direct','relative5'):
            self.assertTrue(torch.isfinite(projection_descriptors(u,s,kind)).all())

    def test_urope_known_projection_and_wrapper(self):
        K,c=cameras();K=K.float();c=c.float()
        funcs=prepare_urope(K,c)
        torch.manual_seed(71)
        q,k,v=[torch.randn(1,HEADS,3*TOKENS,HD) for _ in range(3)]
        got=apply_urope(q,k,v,funcs)
        flip=torch.diag(torch.tensor([1.,-1,-1,1.]))
        expected=urope_dot_product_attention(q,k,v,viewmats=flip@torch.linalg.inv(c),Ks=K,
            patches_x=16,patches_y=12,image_width=128,image_height=96,
            group_size=1,depth_num=4,min_depth=2/3,max_depth=20/3)
        torch.testing.assert_close(got,expected,atol=2e-6,rtol=2e-5)
        # Independent pinhole geometry for source view 1 -> query view 0.
        data=torch.ones(3,4,3*TOKENS,HD)
        rotated=funcs[1](data)
        for h,depth in enumerate([2/3,2/3+1.5,2/3+3,2/3+4.5]):
            x=((4-64)/128*depth+.75);y=(4-48)/96*depth
            px=((128*x+64*depth)/(depth+1e-5)/8)
            py=((96*y+48*depth)/(depth+1e-5)/8)
            frequency=100**(-np.arange(12)/12)
            ref=np.concatenate([np.cos(px*frequency)+np.sin(px*frequency),
                                np.cos(px*frequency)-np.sin(px*frequency),
                                np.cos(py*frequency)+np.sin(py*frequency),
                                np.cos(py*frequency)-np.sin(py*frequency)])
            np.testing.assert_allclose(rotated[0,h,TOKENS].numpy(),ref,atol=3e-6)

    def test_same_common_initialization_and_input_rejection(self):
        for seed in (11,22,33):
            torch.manual_seed(seed);old=CompactNVS('observed_prope');reference=digest(old)
            for role in ROLES:
                torch.manual_seed(seed);model=TargetProjectionNVS(role)
                self.assertEqual(digest(model),reference)
                with self.assertRaises(ValueError):model({'target_rgb':torch.zeros(1)})


if __name__=='__main__':
    unittest.main()
