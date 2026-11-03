import math
import unittest
import numpy as np
import torch
from experiments.round51_data import camera_fields, pixels
from experiments.round55_train import tensor_hash, common
from models.round64_depth_field import (make_model, NEW_ROLES, validate_grid,
    depth_derivatives, depth_fields, local_gaussian, FLOOR)


def fixture(equal_focal=False):
    torch.manual_seed(6401)
    intrinsics=np.array([[170.,0,94.7],[0,170. if equal_focal else 172.,73.1],[0,0,1.]])
    pose=np.eye(4);a=.2
    pose[:3,:3]=[[math.cos(a),0,math.sin(a)],[0,1,0],[-math.sin(a),0,math.cos(a)]]
    pose[:3,3]=[.3,-.2,.1]
    data={x:torch.randn(1,432,128,dtype=torch.float64)*.1 for x in ('q','k')}
    for prefix,p in (('q',np.eye(4)),('k',pose)):
        for name,value in camera_fields(dict(K=intrinsics),p).items():
            data[prefix+name]=torch.from_numpy(value).double()[None]
    return data


class DepthFieldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(4)

    def test_affine_depth_derivatives_and_jacobian(self):
        d=fixture();validate_grid(d)
        uv=torch.from_numpy(pixels()).double()[None]
        z=1+.001*uv[...,0]-.0007*uv[...,1]
        du,dv=depth_derivatives(z,d['qpos'],d['qgeom'])
        torch.testing.assert_close(du,torch.full_like(z,.001),atol=1e-15,rtol=0)
        torch.testing.assert_close(dv,torch.full_like(z,-.0007),atol=1e-15,rtol=0)
        raw=torch.log(torch.expm1(z-1/30))-math.log(math.expm1(29/30))
        fields=depth_fields(raw[...,None],d,'q');mean,cov=local_gaussian(*fields)
        z0,rays,_,_,fx,fy=fields
        expected=[]
        for i in range(432):
            r=rays.numpy()[0,i];zi=z.numpy()[0,i]
            ju=np.array([zi/float(fx[0,0]),0,0])+r*.001
            jv=np.array([0,zi/float(fy[0,0]),0])+r*(-.0007)
            expected.append(5.25*(np.outer(ju,ju)+np.outer(jv,jv))+FLOOR**2*np.eye(3))
        np.testing.assert_allclose(cov.numpy()[0],expected,atol=1e-14,rtol=1e-12)
        torch.testing.assert_close(mean,z0[...,None]*rays,atol=0,rtol=0)
        self.assertGreaterEqual(float(torch.linalg.eigvalsh(cov).min()),FLOOR**2-1e-14)
        # A perturbed single interior pixel reaches the expected finite-difference neighbors.
        impulse=torch.zeros_like(z);impulse[0,8*24+10]=1
        a,b=depth_derivatives(impulse,d['qpos'],d['qgeom'])
        self.assertEqual(int((a!=0).sum()),2);self.assertEqual(int((b!=0).sum()),2)
        self.assertEqual(float(a[0,8*24+9]),1/16)
        self.assertEqual(float(b[0,7*24+10]),1/16)

    def test_full_grid_contract_and_geometry_gradient(self):
        d=fixture();bad=dict(d,qpos=d['qpos'].clone())
        bad['qpos'][:,0]=bad['qpos'][:,1]
        with self.assertRaises(ValueError):validate_grid(bad)
        with self.assertRaises(ValueError):depth_derivatives(torch.ones(1,7),d['qpos'][:,:7],d['qgeom'][:,:7])
        raw=(torch.randn(1,432,1,dtype=torch.float64)*.1).requires_grad_()
        weight=torch.randn(1,432,3,3,dtype=torch.float64)
        def loss(x):
            mu,cov=local_gaussian(*depth_fields(x,d,'q'))
            return mu.square().mean()+100*(cov*weight).sum()
        loss(raw).backward();self.assertTrue(torch.isfinite(raw.grad).all())
        index=(0,200,0);analytic=float(raw.grad[index]);eps=1e-6
        with torch.no_grad():
            plus=raw.clone();minus=raw.clone();plus[index]+=eps;minus[index]-=eps
            numeric=float((loss(plus)-loss(minus))/(2*eps))
        self.assertLess(abs(analytic-numeric),1e-7+1e-6*abs(numeric))

    def test_initial_equivalence_shared_weights_and_input_exclusion(self):
        d=fixture()
        torch.manual_seed(11);ref=make_model('camera_only').double().eval()
        torch.manual_seed(11);surface=make_model('depth_surface').double().eval()
        torch.manual_seed(11);footprint=make_model('depth_footprint').double().eval()
        self.assertEqual(tensor_hash(surface.state_dict()),tensor_hash(footprint.state_dict()))
        with torch.no_grad():
            torch.testing.assert_close(surface(d),footprint(d),atol=0,rtol=0)
            expected=ref.encode(d)
            for role in NEW_ROLES:
                torch.manual_seed(11);model=make_model(role).double().eval()
                self.assertEqual(tensor_hash(common(model)),tensor_hash(common(ref)))
                actual=model.encode(d,zero_relation=True)
                for a,b in zip(actual,expected):torch.testing.assert_close(a,b,atol=0,rtol=0)
                base=model(d)
                poisoned=dict(d,gt_valid=torch.zeros(1,432,dtype=torch.bool),
                    qobserved_mean=torch.full((1,432,3),float('nan')),
                    kobserved_cov=torch.full((1,432,3,3),float('nan')),
                    key_weights=torch.full((1,432,4),float('nan')))
                torch.testing.assert_close(model(poisoned),base,atol=0,rtol=0)
                n=sum(p.numel() for n,p in model.named_parameters() if n.startswith('field.'))
                self.assertEqual(n,10369 if role=='depth_local_mlp' else 10465)

    def test_independent_token_permutations_and_learning_gradient(self):
        d=fixture(equal_focal=True)
        torch.manual_seed(6402);pq,pk=torch.randperm(432),torch.randperm(432)
        perm={k:(v if k in ('qK','kK') else v[:,pq if k.startswith('q') else pk]) for k,v in d.items()}
        validate_grid(perm)
        for role in NEW_ROLES:
            torch.manual_seed(22);model=make_model(role).double()
            # Nonconstant depth, so grid-dependent derivatives are exercised.
            with torch.no_grad():model.field['predictor'].weight.normal_(0,.002)
            with torch.no_grad():
                base=model(d);changed=model(perm)
                torch.testing.assert_close(changed,base[:,pq][:,:,pk],atol=2e-9,rtol=2e-9)
            del model
            # Finite eigvalsh backward at the declared constant/equal-focal initialization.
            torch.manual_seed(22);model=make_model(role).double()
            loss=model(d)[0,100].log_softmax(-1)[200];loss.backward()
            for n,p in model.named_parameters():
                if n.startswith('field.'):
                    self.assertIsNotNone(p.grad,n);self.assertTrue(torch.isfinite(p.grad).all(),n)
            self.assertGreater(float(model.field['predictor'].weight.grad.abs().max()),1e-8)


if __name__=='__main__':unittest.main()
