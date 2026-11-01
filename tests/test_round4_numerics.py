"""Independent checks of the explicitly opted-in normalized-PE fast path."""
import unittest

import torch

from models.gaussian_pe import normalized_rotary,GaussianPEMatcher
from models.gaussian_pe_fast import eligibility,normalized_rotary_fast


class NormalizedFastPathTests(unittest.TestCase):
    def fixture(self,count=16):
        g=torch.Generator().manual_seed(32771)
        f=torch.randn(count,48,generator=g)
        f[:3,:4]=0
        m=torch.randn(count,2,generator=g)
        a=torch.randn(count,2,2,generator=g)
        c=a@a.transpose(-1,-2)*.03+torch.eye(2)*.1
        w=GaussianPEMatcher('normalized_shape',16).freq2
        return f,m,c,w

    def test_ordinary_value_and_all_input_gradients_match_reference(self):
        inputs=self.fixture()
        self.assertTrue(eligibility(*inputs).all())
        weight=torch.linspace(-1,1,inputs[0].numel()).reshape_as(inputs[0])
        results=[];gradients=[]
        for function in [normalized_rotary,normalized_rotary_fast]:
            args=[x.clone().requires_grad_() for x in inputs]
            result=function(*args);results.append(result)
            gradients.append(torch.autograd.grad((result*weight).sum(),args))
        torch.testing.assert_close(results[0],results[1],rtol=2e-5,atol=3e-6)
        for a,b in zip(*gradients):torch.testing.assert_close(a,b,rtol=4e-4,atol=2e-5)
        self.assertGreater(float(gradients[1][0][:3,:4].abs().sum()),.1)

    def test_mixed_fast_and_fallback_keep_correct_gradients(self):
        f,m,c,w=self.fixture()
        c[0]=torch.eye(2)*3  # outside variance bound, yet reference Jacobian representable
        f[0,:]=0;f[0,-2]=1
        mask=eligibility(f,m,c,w)
        self.assertFalse(mask[0]);self.assertTrue(mask[1:].all())
        actual,info=normalized_rotary_fast(f,m,c,w,return_diagnostics=True)
        expected=normalized_rotary(f,m,c,w)
        torch.testing.assert_close(actual[0],expected[0],rtol=0,atol=0)
        torch.testing.assert_close(actual,expected,rtol=2e-5,atol=3e-6)
        self.assertEqual(info['eligible_tokens'],len(f)-1)
        gradients=[]
        weight=torch.linspace(-1,1,f.numel()).reshape_as(f)
        for function in [normalized_rotary,normalized_rotary_fast]:
            args=[x.clone().requires_grad_() for x in [f,m,c]]
            output=function(*args,w)
            gradients.append(torch.autograd.grad((output*weight).sum(),args))
        for a,b in zip(*gradients):torch.testing.assert_close(a,b,rtol=4e-4,atol=2e-5)

    def test_underflow_zero_and_tiny_content_fall_back_exactly(self):
        w=torch.tensor([[1.,0.],[2.,0.]])
        f=torch.tensor([[0.,0.,1.,0.],[0.,0.,0.,0.],[1e-35,0.,0.,0.]])
        m=torch.zeros(3,2);c=torch.eye(2).expand(3,2,2)*1000
        self.assertFalse(eligibility(f,m,c,w).any())
        expected=normalized_rotary(f,m,c,w)
        actual=normalized_rotary_fast(f,m,c,w)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        self.assertEqual(float(actual[0].norm()),1.)

    def test_float64_and_broadcast_shapes_use_reference(self):
        inputs=self.fixture()
        inputs=[x.double() for x in inputs]
        actual,info=normalized_rotary_fast(*inputs,return_diagnostics=True)
        torch.testing.assert_close(actual,normalized_rotary(*inputs),rtol=0,atol=0)
        self.assertTrue(info['unsupported_shape_or_dtype'])

    def test_point_limit_and_common_translation_within_float32_tolerance(self):
        f,m,c,w=self.fixture()
        c=torch.zeros_like(c)
        a=normalized_rotary_fast(f,m,c,w)
        b=normalized_rotary_fast(f,m+torch.tensor([.2,-.3]),c,w)
        torch.testing.assert_close(a@a.T,b@b.T,rtol=2e-5,atol=2e-5)
        torch.testing.assert_close(a.norm(dim=-1),f.norm(dim=-1),rtol=2e-6,atol=2e-6)


if __name__=='__main__':unittest.main()
