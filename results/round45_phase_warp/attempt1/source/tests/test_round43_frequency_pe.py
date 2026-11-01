import unittest
import numpy as np
import torch
from models.round43_frequency_pe import FrequencyBank,NEW_ROLES,make_model


def fixture():
    rng=torch.Generator().manual_seed(731)
    q=torch.randn(2,7,128,generator=rng,dtype=torch.float64)
    means={k:torch.randn(2,7,2,generator=rng,dtype=torch.float64) for k in ('qmean','kmean','k_to_q','q_to_k')}
    cov={k:torch.eye(2,dtype=torch.float64).expand(2,7,2,2).clone() for k in ('qcov','kcov','k_to_qcov','q_to_kcov')}
    return dict(q=q,k=q.flip(1).clone(),**means,**cov)


class FrequencyTests(unittest.TestCase):
    def test_common_initialization_and_counts(self):
        reference=torch.randn(24,2,dtype=torch.float64)
        base=FrequencyBank('fixed_gaussian',reference)()
        for role,count in zip(NEW_ROLES,(20,16,12,0,6,24,48)):
            bank=FrequencyBank(role,reference)
            torch.testing.assert_close(bank(),base,atol=1e-14,rtol=1e-14)
            self.assertEqual(sum(p.numel() for p in bank.parameters()),count)
            np.testing.assert_allclose(bank.template.numpy().mean(1),0,atol=1e-14)
            np.testing.assert_allclose(np.einsum('gni,gnj->gij',bank.template.numpy(),bank.template.numpy())/6,
                                       np.broadcast_to(np.eye(2),(4,2,2)),atol=1e-14)

    def test_gaussian_has_an_exact_group_affine_control(self):
        ref=torch.randn(24,2,dtype=torch.float64)
        one=FrequencyBank('gaussian_full',ref);other=FrequencyBank('group_affine',ref)
        with torch.no_grad():
            one.mean_offset.copy_(torch.arange(8).reshape(4,2)/17)
            one.log_scale.copy_(torch.linspace(-.3,.4,8).reshape(4,2));one.lower.fill_(.2)
            L=torch.diag_embed(one.log_scale.exp());L[:,1,0]=.2
            other.mean_offset.copy_(one.mean_offset);other.matrix_offset.copy_(L-torch.eye(2))
        torch.testing.assert_close(one(),other(),atol=1e-14,rtol=1e-14)

    def test_norm_translation_and_input_scope(self):
        data=fixture();model=make_model('gaussian_full').double()
        content=torch.randn(2,2,7,48,dtype=torch.float64)
        pe=model._position_encode(content,data['qmean'],data['qcov'])
        torch.testing.assert_close(pe.norm(dim=-1),content.norm(dim=-1),atol=1e-12,rtol=1e-12)
        with torch.no_grad():before=model(data)
        changed={k:v.clone() for k,v in data.items()}
        for key in ('qmean','k_to_q'):changed[key]+=torch.tensor([7.,-3.])
        for key in ('kmean','q_to_k'):changed[key]+=torch.tensor([-2.,4.])
        for key in ('qcov','kcov','k_to_qcov','q_to_kcov'):changed[key]*=123
        changed['gt_valid']=torch.zeros(2,7,dtype=torch.bool)
        with torch.no_grad():after=model(changed)
        torch.testing.assert_close(after,before,atol=1e-11,rtol=1e-11)

    def test_finite_nonzero_frequency_gradient_at_isotropy(self):
        bank=FrequencyBank('gaussian_full',torch.randn(24,2,dtype=torch.float64))
        x=torch.randn(9,2,dtype=torch.float64);weight=torch.randn(9,24,dtype=torch.float64)
        loss=((x@bank().T).sin()*weight).sum();loss.backward()
        for p in bank.parameters():
            self.assertTrue(torch.isfinite(p.grad).all());self.assertGreater(float(p.grad.norm()),0)
        original=bank.lower.detach().clone();epsilon=1e-6
        with torch.no_grad():
            bank.lower[0,0]+=epsilon;plus=float(((x@bank().T).sin()*weight).sum())
            bank.lower[0,0]-=2*epsilon;minus=float(((x@bank().T).sin()*weight).sum());bank.lower.copy_(original)
        self.assertAlmostEqual((plus-minus)/(2*epsilon),float(bank.lower.grad[0,0]),places=6)


if __name__=='__main__':unittest.main()
