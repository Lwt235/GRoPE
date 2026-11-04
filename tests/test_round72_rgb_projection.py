"""New RGB path: paired state, input boundary and differentiable geometry."""
import unittest
import numpy as np
import torch
from models.round70_nvs import CompactNVS, Camera, patchify, PATCH, WIDTH, HEIGHT, TOKENS
from models.round62_rgb_relations import gaussian_from_raw
from models.round71_target_projection import TargetProjectionNVS, project_reference, projection_descriptors
from models.round72_rgb_projection import RGBTargetProjectionNVS, ROLES
from experiments.round70_nvs_run import digest
from tests.test_round71_target_projection import cameras


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def test_common_state_and_strict_rgb_interface(self):
        for seed in (11,22,33):
            torch.manual_seed(seed);old=CompactNVS('rgb_prope')
            for role in ROLES:
                torch.manual_seed(seed);model=RGBTargetProjectionNVS(role)
                self.assertEqual(digest(model),digest(old))
                for name,value in old.geometry_head.state_dict().items():
                    torch.testing.assert_close(value,model.geometry_head.state_dict()[name],atol=0,rtol=0)
                for field in ('ref_mean','ref_cov','ref_coverage','target_rgb','target_depth'):
                    with self.assertRaises(ValueError):
                        model(dict(ref_rgb=None,K=None,c2w=None,**{field:None}))

    def test_same_explicit_geometry_matches_observed_implementation(self):
        K,c=cameras();K=K.float();c=c.float()
        torch.manual_seed(72)
        data=dict(ref_rgb=torch.rand(1,2,HEIGHT,WIDTH,3),K=K,c2w=c)
        for role in ROLES:
            torch.manual_seed(11);model=RGBTargetProjectionNVS(role).eval()
            # Nonconstant predicted geometry exercises the whole new path.
            with torch.no_grad():model.geometry_head.weight.normal_(std=.003)
            old=TargetProjectionNVS(role.replace('rgb_','observed_',1)).eval()
            old.load_state_dict({n:v for n,v in model.state_dict().items() if not n.startswith('geometry_head.')})
            with torch.no_grad():
                rays=patchify(model.create_rays(Camera(K,c,WIDTH,HEIGHT)),PATCH)
                x=model.input_tokenizer(torch.cat((patchify(data['ref_rgb'],PATCH),rays[:,:2]),-1))
                mean,cov,coverage=model.geometry(x,data)
                rt=c[:,:2,:3,:3].transpose(-1,-2)
                local_mean=torch.einsum('bvij,bvnj->bvni',rt,mean.reshape(1,2,TOKENS,3)-c[:,:2,None,:3,3])
                local_cov=rt[:,:,None]@cov.reshape(1,2,TOKENS,3,3)@rt[:,:,None].transpose(-1,-2)
                explicit=dict(data,ref_mean=local_mean,ref_cov=local_cov,ref_coverage=coverage.reshape(1,2,TOKENS))
                actual=model(data);expected=old(explicit)
            torch.testing.assert_close(actual,expected,atol=3e-5,rtol=3e-5)

    def test_predicted_geometry_projection_derivative(self):
        K,c=cameras();c[:,2,0,3]=.25
        raw=torch.tensor([[[.15,.1,-.12,.08,.05,-.04,.03]]],dtype=torch.float64,requires_grad=True)
        ray=torch.tensor([[[.12,-.07,1.]]],dtype=torch.float64)
        for kind in ('distance','direct','relative5'):
            def value(z):
                mean,cov=gaussian_from_raw(z,ray)
                u,s,valid=project_reference(mean,cov,K,c,torch.ones(1,1))
                self.assertTrue(valid.all())
                return projection_descriptors(u,s,kind)[0,37,0].sum()
            analytic=torch.autograd.grad(value(raw),raw)[0].detach().numpy().reshape(-1)
            finite=[];eps=1e-5
            for j in range(7):
                plus=raw.detach().clone();minus=raw.detach().clone()
                plus[...,j]+=eps;minus[...,j]-=eps
                finite.append(float((value(plus)-value(minus))/(2*eps)))
            np.testing.assert_allclose(analytic,finite,rtol=2e-5,atol=2e-6)
            self.assertGreater(abs(analytic[0]),0.)
            if kind!='distance':self.assertGreater(np.linalg.norm(analytic[1:]),0.)


if __name__=='__main__':unittest.main()
