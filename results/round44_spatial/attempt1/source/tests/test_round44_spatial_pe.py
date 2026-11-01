import unittest
import numpy as np
import torch
from models.round44_spatial_pe import ROLES, SpatialField, gaussian_response, make_model
from models.round5_cross_attention import Round5CrossAttentionMatcher
from experiments.round44_spatial_train import parameter_digest
from experiments.round44_spatial_audit import numpy_position


class SpatialPETests(unittest.TestCase):
    def test_initialization_capacity_and_fixed_energy(self):
        torch.manual_seed(731)
        reference = parameter_digest(Round5CrossAttentionMatcher('point'))
        yy, xx = torch.meshgrid(torch.arange(18), torch.arange(24), indexing='ij')
        grid = torch.stack((xx.flatten()+.4375, yy.flatten()+.4375), -1).double()
        projection = None
        gaussian = None
        for role, count in zip(ROLES, (1908,1872,1836,1728,1836,1728,1782,1836)):
            torch.manual_seed(731)
            model = make_model(role)
            self.assertEqual(parameter_digest(model), reference)
            self.assertEqual(sum(p.numel() for p in model.parameters()), 166611+count)
            field = model.position_field.double()
            if projection is None:
                projection = field.projection.detach().clone()
            torch.testing.assert_close(field.projection, projection, atol=0, rtol=0)
            x = (grid-field.chart_center)/field.chart_extent
            energy = (field.basis(x)/field.initial_rms).square().sum(-1).mean()
            self.assertAlmostEqual(float(energy), 1., places=12)
            if role.startswith('spatial_'):
                if gaussian is None:
                    gaussian = field(grid).detach()
                torch.testing.assert_close(field(grid), gaussian, atol=0, rtol=0)

    def test_independent_nontrivial_basis_formulas(self):
        for role in ROLES:
            torch.manual_seed(119)
            model = make_model(role).double()
            field = model.position_field
            with torch.no_grad():
                for name, parameter in field.named_parameters():
                    if name != 'projection':
                        parameter.add_(torch.linspace(-.08,.08,parameter.numel()).reshape_as(parameter))
            mean = torch.tensor([[0.,0.],[5.,7.],[20.,15.],[32.,-2.]],dtype=torch.float64)
            actual = field(mean).detach().numpy()
            expected = numpy_position(role,model.state_dict(),mean)
            np.testing.assert_allclose(actual,expected,atol=2e-12,rtol=2e-12)

    def test_exact_gaussian_joint_frame_change_in_2d_and_3d(self):
        generator = torch.Generator().manual_seed(166)
        for d in (2,3):
            points = torch.randn(13,d,generator=generator,dtype=torch.float64)
            centers = torch.randn(4,d,generator=generator,dtype=torch.float64)
            L = torch.eye(d).expand(4,d,d).clone() + .2*torch.tril(torch.randn(4,d,d,generator=generator,dtype=torch.float64))
            angle = torch.tensor(.73,dtype=torch.float64)
            R = torch.eye(d,dtype=torch.float64)
            R[:2,:2] = torch.stack((torch.stack((angle.cos(),-angle.sin())),torch.stack((angle.sin(),angle.cos()))))
            offset = torch.arange(d,dtype=torch.float64)+.4
            P = L@L.transpose(-1,-2)
            rotated = torch.linalg.cholesky(R@P@R.T)
            torch.testing.assert_close(gaussian_response(points,centers,L),
                gaussian_response(points@R.T+offset,centers@R.T+offset,rotated),atol=1e-12,rtol=1e-12)

    def test_shape_gradients_at_isotropy_and_far_inputs(self):
        torch.manual_seed(29)
        field = SpatialField('spatial_full').double()
        x = torch.randn(23,2,dtype=torch.float64)*6+field.chart_center
        weight = torch.randn(23,48,dtype=torch.float64)
        loss = (field(x)*weight).sum()
        loss.backward()
        for parameter in field.parameters():
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.norm()),0.)
        analytic = float(field.lower.grad[10,0])
        with torch.no_grad():
            field.lower[10,0] += 1e-6
            plus = float((field(x)*weight).sum())
            field.lower[10,0] -= 2e-6
            minus = float((field(x)*weight).sum())
            field.lower[10,0] += 1e-6
        self.assertAlmostEqual((plus-minus)/2e-6,analytic,places=7)
        far = torch.full((3,2),1e6,dtype=torch.float64,requires_grad=True)
        field(far).sum().backward()
        self.assertTrue(torch.isfinite(far.grad).all())

    def test_input_scope_and_position_path(self):
        torch.manual_seed(159)
        model = make_model('spatial_full').double()
        data = {k:torch.randn(2,7,128,dtype=torch.float64) for k in ('q','k')}
        data.update({k:torch.randn(2,7,2,dtype=torch.float64)*6+torch.tensor([12.,9.])
            for k in ('qmean','kmean','k_to_q','q_to_k')})
        data.update({k:torch.eye(2,dtype=torch.float64).expand(2,7,2,2).clone()
            for k in ('qcov','kcov','k_to_qcov','q_to_kcov')})
        changed = {k:v.clone() for k,v in data.items()}
        for k in ('qcov','kcov','k_to_qcov','q_to_kcov'):
            changed[k] *= 123
        changed['gt_valid'] = torch.zeros(2,7,dtype=torch.bool)
        with torch.no_grad():
            before = model(data)
            torch.testing.assert_close(model(changed),before,atol=0,rtol=0)
            changed['qmean'] += 4
            self.assertGreater(float((model(changed)-before).abs().max()),1e-5)


if __name__=='__main__':
    unittest.main()
