import unittest
import torch
from models.gaussian_pe import spectral_rule,density_features,lift,bhattacharyya,GaussianPEMatcher


class GaussianPETest(unittest.TestCase):
    def test_factorization_norm_and_translation(self):
        torch.manual_seed(17)
        omega,lp=spectral_rule(256,dtype=torch.float64)
        means=torch.randn(5,2,dtype=torch.float64)
        a=torch.randn(5,2,2,dtype=torch.float64)
        cov=a@a.transpose(-1,-2)*.05+torch.eye(2)*.03
        p=density_features(means,cov,omega,lp)
        c=torch.randn(5,6,dtype=torch.float64)
        extended=lift(c,p)
        torch.testing.assert_close(extended.norm(dim=-1),c.norm(dim=-1))
        torch.testing.assert_close(extended@extended.T,(c@c.T)*(p@p.T))
        p2=density_features(means+torch.tensor([2.,-4.]),cov,omega,lp)
        torch.testing.assert_close(p@p.T,p2@p2.T)

    def test_approximation_and_gauge(self):
        omega,lp=spectral_rule(4096,dtype=torch.float64)
        m=torch.tensor([[0.,0.],[.2,.3],[.8,-.1]],dtype=torch.float64)
        c=torch.tensor([[[.08,.02],[.02,.12]],[[.15,-.04],[-.04,.06]],[[.05,0.],[0.,.2]]],dtype=torch.float64)
        p=density_features(m,c,omega,lp)
        self.assertLess((p@p.T-bhattacharyya(m,c,m,c)).abs().max().item(),.025)
        # The map takes covariance, so equivalent R sign flips/permutations
        # have exactly the same input. Gradients remain finite at isotropy.
        ci=(torch.eye(2,dtype=torch.float64)[None]*.1).requires_grad_()
        density_features(m[:1],ci,omega,lp).square().sum().backward()
        self.assertTrue(torch.isfinite(ci.grad).all())

    def test_reference_affine_invariance(self):
        m=torch.tensor([[0.,0.],[1.,.2]],dtype=torch.float64)
        c=torch.stack([torch.eye(2)*.1,torch.diag(torch.tensor([.3,.1]))]).double()
        a=torch.tensor([[1.3,.4],[-.2,.7]],dtype=torch.float64)
        transformed=a@c@a.T
        torch.testing.assert_close(bhattacharyya(m,c,m,c),bhattacharyya(m@a.T,transformed,m@a.T,transformed))

if __name__=='__main__': unittest.main()
