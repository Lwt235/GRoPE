import unittest
import numpy as np
import torch
from models.round45_phase_warp import ROLES, BASIS_ROLES, CoordinateWarp, make_model
from models.round5_cross_attention import Round5CrossAttentionMatcher
from models.geometry import rotary
from experiments.round45_phase_warp_train import parameter_digest
from experiments.round45_phase_warp_audit import numpy_phase


def toy_data():
    torch.manual_seed(159)
    data = {k:torch.randn(2,7,128) for k in ('q','k')}
    data.update({k:torch.randn(2,7,2)*6+torch.tensor([12.,9.])
        for k in ('qmean','kmean','k_to_q','q_to_k')})
    data.update({k:torch.eye(2).expand(2,7,2,2).clone()
        for k in ('qcov','kcov','k_to_qcov','q_to_kcov')})
    return data


class PhaseWarpTests(unittest.TestCase):
    def test_exact_complete_network_initialization_and_capacity(self):
        data = toy_data()
        for seed in (11,22,33,44,55):
            torch.manual_seed(seed)
            point = Round5CrossAttentionMatcher('point').eval()
            with torch.no_grad():
                expected = point(data)
            for role,count in zip(ROLES,(252,180,72,180,126,180,4,48)):
                torch.manual_seed(seed)
                model = make_model(role).eval()
                self.assertEqual(parameter_digest(model),parameter_digest(point))
                self.assertEqual(sum(p.numel() for p in model.parameters()),166611+count)
                with torch.no_grad():
                    torch.testing.assert_close(model(data),expected,atol=0,rtol=0)

    def test_nonzero_independent_phase_and_block_norm(self):
        for role in ROLES:
            torch.manual_seed(119)
            model = make_model(role).double()
            with torch.no_grad():
                for parameter in model.position_field.parameters():
                    parameter.add_(torch.linspace(-.08,.08,parameter.numel()).reshape_as(parameter))
            mean = torch.tensor([[0.,0.],[5.,7.],[20.,15.],[32.,-2.]],dtype=torch.float64)
            coordinates,frequency = model.position_arguments(mean)
            phase = numpy_phase(role,model.state_dict(),mean)
            np.testing.assert_allclose((coordinates@frequency.T).detach(),phase,atol=2e-12,rtol=2e-12)
            content = torch.randn(1,2,4,48,dtype=torch.float64)
            actual = model._position_encode(content,mean[None],None)
            z=content.numpy()[...,0::2]+1j*content.numpy()[...,1::2]
            reference = z*np.exp(1j*phase)
            expected = np.stack((reference.real,reference.imag),-1).reshape(1,2,4,48)
            np.testing.assert_allclose(actual.detach(),expected,atol=2e-12,rtol=2e-12)
            torch.testing.assert_close(actual.reshape(1,2,4,24,2).square().sum(-1),
                content.reshape(1,2,4,24,2).square().sum(-1),atol=2e-12,rtol=2e-12)

    def test_relative_phase_is_in_mapped_coordinates(self):
        torch.manual_seed(174)
        model = make_model('warp_full').double()
        with torch.no_grad():
            model.position_field.projection.normal_(std=.04)
        x=torch.tensor([[[4.,6.]]],dtype=torch.float64)
        y=torch.tensor([[[12.,9.]]],dtype=torch.float64)
        q=torch.randn(1,2,1,48,dtype=torch.float64)
        k=torch.randn_like(q)
        left=(model._position_encode(q,x,None)*model._position_encode(k,y,None)).sum(-1)
        ux,w=model.position_arguments(x);uy,_=model.position_arguments(y)
        right=(q*rotary(k,(uy-ux).unsqueeze(1),w)).sum(-1)
        torch.testing.assert_close(left,right,atol=2e-12,rtol=2e-12)
        # A nonlinear field is not silently treated as raw translation invariant.
        shifted=(model._position_encode(q,x+5,None)*model._position_encode(k,y+5,None)).sum(-1)
        self.assertGreater(float((left-shifted).abs().max()),1e-4)

    def test_zero_then_nonzero_basis_gradients(self):
        for role in BASIS_ROLES:
            torch.manual_seed(29)
            field=CoordinateWarp(role).double()
            x=torch.randn(23,2,dtype=torch.float64)*6+field.chart_center
            weight=torch.randn(23,2,dtype=torch.float64)
            (field(x)*weight).sum().backward()
            self.assertGreater(float(field.projection.grad.norm()),0.)
            for name,parameter in field.named_parameters():
                self.assertTrue(torch.isfinite(parameter.grad).all())
                if name!='projection':
                    self.assertEqual(float(parameter.grad.norm()),0.)
            with torch.no_grad():
                field.projection.add_(-.01*field.projection.grad)
            field.zero_grad()
            (field(x)*weight).sum().backward()
            for parameter in field.parameters():
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(float(parameter.grad.norm()),0.)
            if role=='warp_full':
                analytic=float(field.lower.grad[10,0])
                with torch.no_grad():
                    field.lower[10,0]+=1e-6
                    plus=float((field(x)*weight).sum())
                    field.lower[10,0]-=2e-6
                    minus=float((field(x)*weight).sum())
                    field.lower[10,0]+=1e-6
                np.testing.assert_allclose((plus-minus)/2e-6,analytic,rtol=2e-6,atol=2e-7)

    def test_affine_frequency_equivalence_and_far_gaussian(self):
        torch.manual_seed(45)
        model=make_model('affine_warp').double()
        with torch.no_grad():
            model.position_field.affine_offset.copy_(torch.tensor([[.1,.2],[-.1,-.15]]))
        field=model.position_field;e=field.chart_extent
        M=torch.eye(2,dtype=torch.float64)+torch.diag(e)@field.affine_offset@torch.diag(1/e)
        x=torch.randn(13,2,dtype=torch.float64)*8
        u,w=model.position_arguments(x)
        torch.testing.assert_close((u[1:]-u[:1])@w.T,
            (x[1:]-x[:1])@(w@M).T,atol=2e-12,rtol=2e-12)
        gaussian=make_model('warp_full').double()
        with torch.no_grad():
            gaussian.position_field.projection.normal_()
        far=torch.full((3,2),1e6,dtype=torch.float64,requires_grad=True)
        mapped,_=gaussian.position_arguments(far)
        torch.testing.assert_close(mapped,far,atol=0,rtol=0)
        mapped.sum().backward()
        torch.testing.assert_close(far.grad,torch.ones_like(far),atol=0,rtol=0)

    def test_input_scope_and_position_path(self):
        data=toy_data();torch.manual_seed(159)
        model=make_model('warp_full')
        with torch.no_grad():
            model.position_field.projection.normal_(std=.02)
        changed={k:v.clone() for k,v in data.items()}
        for k in ('qcov','kcov','k_to_qcov','q_to_kcov'):
            changed[k]*=123
        changed['gt_valid']=torch.zeros(2,7,dtype=torch.bool)
        with torch.no_grad():
            before=model(data)
            torch.testing.assert_close(model(changed),before,atol=0,rtol=0)
            changed['qmean']+=4
            self.assertGreater(float((model(changed)-before).abs().max()),1e-5)


if __name__=='__main__':
    unittest.main()
