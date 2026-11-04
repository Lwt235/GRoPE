import unittest
import numpy as np
import torch
from models.projected_shape_attribution import FixedShapeNVS,ROLES,fixed_shape_inputs
from models.round71_target_projection import TargetProjectionNVS,projection_descriptors,HEIGHT,WIDTH,PATCH
from models.round72_rgb_projection import RGBTargetProjectionNVS


class FixedShapeTests(unittest.TestCase):
    def test_all_initial_parameters_match_parent(self):
        for seed in (11,22,33):
            for role,parent,original in ((ROLES[0],TargetProjectionNVS,'observed_target_relative5'),
                                         (ROLES[1],RGBTargetProjectionNVS,'rgb_target_relative5')):
                torch.manual_seed(seed);a=parent(original)
                torch.manual_seed(seed);b=FixedShapeNVS(role)
                self.assertEqual(set(a.state_dict()),set(b.state_dict()))
                for key,value in a.state_dict().items():self.assertTrue(torch.equal(value,b.state_dict()[key]),key)
                self.assertEqual(sum(p.numel() for p in a.parameters()),sum(p.numel() for p in b.parameters()))
                self.assertEqual(sum(p.numel() for p in b.relative.parameters()),19008)

    def test_control_equals_explicit_reference_C_equal_Q(self):
        torch.manual_seed(77);u=torch.randn(2,7,2);a=torch.randn(2,7,2,2)
        covariance=a@a.transpose(-1,-2)+.01*torch.eye(2)
        q=torch.diag(u.new_tensor([(PATCH/WIDTH)**2/12,(PATCH/HEIGHT)**2/12])).expand(2,7,2,2)
        actual=fixed_shape_inputs(projection_descriptors(u,covariance,'relative5'))
        expected=projection_descriptors(u,q,'relative5')
        torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-6)

    def test_only_PE_reference_shape_dependence_is_removed(self):
        torch.manual_seed(78);u=torch.randn(1,7,2,requires_grad=True)
        covariance=(torch.randn(1,7,2,2));covariance=(covariance@covariance.transpose(-1,-2)+.01*torch.eye(2)).requires_grad_()
        original=projection_descriptors(u,covariance,'relative5');fixed=fixed_shape_inputs(original)
        torch.testing.assert_close(fixed[...,:2],original[...,:2],atol=0,rtol=0)
        fixed.square().mean().backward()
        self.assertTrue(torch.isfinite(u.grad).all());self.assertGreater(float(u.grad.norm()),0)
        self.assertEqual(float(covariance.grad.abs().max()),0)


if __name__=='__main__':unittest.main()
