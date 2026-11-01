"""Independent contracts and preserved pre-fix counterexamples for normalized PE."""
import unittest

import torch
import torch.nn.functional as F

from models.gaussian_pe import GaussianPEMatcher, normalized_rotary
from models.geometry import rotary


DTYPE=torch.float64


def ordinary_fixture():
    g=torch.Generator().manual_seed(80933)
    features=torch.randn(5,48,generator=g,dtype=DTYPE)
    mean=torch.randn(5,2,generator=g,dtype=DTYPE)
    base=torch.randn(5,2,2,generator=g,dtype=DTYPE)
    covariance=base@base.transpose(-1,-2)*.1+torch.eye(2,dtype=DTYPE)*.02
    frequency=GaussianPEMatcher('normalized_shape',count=16).freq2.double()
    return features,mean,covariance,frequency


def sparse_fixture(scale=1000.,dtype=DTYPE):
    features=torch.tensor([[0.,0.,1.,0.]],dtype=dtype)
    mean=torch.zeros(1,2,dtype=dtype)
    covariance=torch.eye(2,dtype=dtype)[None]*scale
    frequency=torch.tensor([[1.,0.],[2.,0.]],dtype=dtype)
    return features,mean,covariance,frequency


def legacy_normalized_rotary(features,mean,covariance,frequency):
    """The original faulty implementation, kept only as regression evidence."""
    logamp=-torch.einsum('fd,...de,fe->...f',frequency,covariance,frequency)/2
    amplitude=(logamp-logamp.amax(-1,keepdim=True)).exp()
    phase=mean@frequency.T
    x,y=features.reshape(*features.shape[:-1],-1,2).unbind(-1)
    transformed=torch.stack([(x*phase.cos()-y*phase.sin())*amplitude,
                             (x*phase.sin()+y*phase.cos())*amplitude],-1).flatten(-2)
    return F.normalize(transformed,dim=-1)*features.norm(dim=-1,keepdim=True)


def model_fixture(count=4):
    g=torch.Generator().manual_seed(1974)
    a=torch.randn(1,count,2,2,generator=g)
    cov=a@a.transpose(-1,-2)*.1+torch.eye(2)*.02
    mean=torch.randn(1,count,2,generator=g)
    q=torch.rand(1,count,108,generator=g)
    k=torch.rand(1,count,108,generator=g)
    return dict(q=q,k=k,pixq=mean,k_to_q=mean+.2,
                projected_qcov=torch.eye(2).expand(1,count,2,2)/12,
                projected_kcov=cov)


class NormalizedRotaryOrdinaryDomainTests(unittest.TestCase):
    def test_shape_and_norm_preserved_for_ordinary_nonzero_content(self):
        features,mean,cov,frequency=ordinary_fixture()
        result=normalized_rotary(features,mean,cov,frequency)
        self.assertEqual(result.shape,features.shape)
        torch.testing.assert_close(result.norm(dim=-1),features.norm(dim=-1),rtol=1e-12,atol=1e-12)

    def test_equivalent_to_normalized_expected_rotary_when_direct_norm_exceeds_epsilon(self):
        features,mean,cov,frequency=ordinary_fixture()
        raw=rotary(features,mean,frequency,cov)
        self.assertGreater(float(raw.norm(dim=-1).min()),1e-12)
        reference=F.normalize(raw,dim=-1)*features.norm(dim=-1,keepdim=True)
        torch.testing.assert_close(normalized_rotary(features,mean,cov,frequency),reference,rtol=2e-12,atol=2e-12)

    def test_zero_covariance_point_limit_for_ordinary_content(self):
        features,mean,cov,frequency=ordinary_fixture()
        result=normalized_rotary(features,mean,torch.zeros_like(cov),frequency)
        torch.testing.assert_close(result,rotary(features,mean,frequency),rtol=1e-12,atol=1e-12)

    def test_common_translation_is_shared_rotation_and_preserves_pair_scores(self):
        features,mean,cov,frequency=ordinary_fixture()
        shift=torch.tensor([1.2,-.7],dtype=DTYPE)
        original=normalized_rotary(features,mean,cov,frequency)
        translated=normalized_rotary(features,mean+shift,cov,frequency)
        shared=rotary(original,shift.expand_as(mean),frequency)
        torch.testing.assert_close(translated,shared,rtol=2e-12,atol=2e-12)
        torch.testing.assert_close(translated@translated.T,original@original.T,rtol=2e-12,atol=2e-12)

    def test_repeated_covariance_eigenvalues_have_finite_correct_gradients(self):
        features=torch.tensor([[.3,-.7,.8,.4]],dtype=DTYPE,requires_grad=True)
        mean=torch.tensor([[.2,-.1]],dtype=DTYPE,requires_grad=True)
        cov=(torch.eye(2,dtype=DTYPE)[None]*.3).requires_grad_()
        # Equal frequency norms tie max(logamp) at this isotropic covariance.
        # The common-offset derivative must cancel after normalization.
        frequency=torch.eye(2,dtype=DTYPE)
        self.assertTrue(torch.autograd.gradcheck(
            lambda x,m,c:normalized_rotary(x,m,c,frequency),(features,mean,cov),
            eps=1e-6,atol=1e-5,rtol=1e-4,fast_mode=True))

    def test_ordinary_zero_blocks_keep_direct_reference_content_gradients(self):
        features=torch.tensor([[0.,0.,.4,-.7],[.3,0.,0.,.5]],dtype=DTYPE)
        mean=torch.tensor([[.2,-.1],[-.3,.7]],dtype=DTYPE)
        cov=torch.tensor([[[.4,.05],[.05,.3]],[[.2,-.02],[-.02,.5]]],dtype=DTYPE)
        frequency=torch.tensor([[1.,0.],[2.,.3]],dtype=DTYPE)
        weight=torch.tensor([[1.,-.3,.7,.2],[-.4,.8,.3,-.6]],dtype=DTYPE)
        gradients=[];outputs=[]
        for stable in [False,True]:
            args=[t.clone().requires_grad_() for t in [features,mean,cov]]
            if stable:result=normalized_rotary(*args,frequency)
            else:
                raw=rotary(args[0],args[1],frequency,args[2])
                result=F.normalize(raw,dim=-1)*args[0].norm(dim=-1,keepdim=True)
            outputs.append(result)
            gradients.append(torch.autograd.grad((result*weight).sum(),args))
        torch.testing.assert_close(outputs[0],outputs[1],rtol=1e-11,atol=1e-12)
        for reference,actual in zip(*gradients):
            self.assertTrue(torch.isfinite(actual).all())
            torch.testing.assert_close(actual,reference,rtol=2e-10,atol=2e-11)
        self.assertGreater(float(gradients[0][0][0,:2].abs().sum()),.1)

    def test_tiny_and_large_finite_content_preserve_representable_norm(self):
        for dtype,scales in [(torch.float32,[1e-35,1e-20,1.,1e20]),
                             (torch.float64,[1e-300,1e-200,1.,1e200])]:
            for scale in scales:
                with self.subTest(dtype=dtype,scale=scale):
                    features=torch.tensor([[.3,-.7,.8,.4]],dtype=dtype)*scale
                    mean=torch.tensor([[.2,-.1]],dtype=dtype)
                    covariance=torch.eye(2,dtype=dtype)[None]*1000
                    frequency=torch.tensor([[1.,0.],[2.,0.]],dtype=dtype)
                    result=normalized_rotary(features,mean,covariance,frequency)
                    def stable_norm(value):
                        value=value.double();unit=value.abs().amax(-1,keepdim=True)
                        return (value/unit).norm(dim=-1)*unit[...,0]
                    self.assertTrue(torch.isfinite(result).all())
                    torch.testing.assert_close(stable_norm(result),stable_norm(features),rtol=2e-6,atol=0)

    def test_extreme_covariance_dense_content_survives_with_finite_gradients(self):
        for dtype in [torch.float32,torch.float64]:
            for scale in [1e2,1e6,1e20]:
                with self.subTest(dtype=dtype,scale=scale):
                    features=torch.tensor([[.3,-.7,.8,.4]],dtype=dtype,requires_grad=True)
                    mean=torch.tensor([[.2,-.1]],dtype=dtype,requires_grad=True)
                    cov=(torch.eye(2,dtype=dtype)[None]*scale).requires_grad_()
                    frequency=torch.tensor([[1.,0.],[2.,0.]],dtype=dtype)
                    result=normalized_rotary(features,mean,cov,frequency)
                    objective=(result*torch.tensor([[1.,-2.,.3,.4]],dtype=dtype)).sum()
                    objective.backward()
                    self.assertTrue(torch.isfinite(result).all())
                    for value in [features.grad,mean.grad,cov.grad]:self.assertTrue(torch.isfinite(value).all())
                    torch.testing.assert_close(result.norm(dim=-1),features.norm(dim=-1),rtol=2e-6,atol=2e-6)

    def test_large_random_spd_covariances_with_sparse_content_keep_forward_norm(self):
        features,mean,cov,frequency=ordinary_fixture()
        features[:,:24]=0  # remove all lower-frequency occupied blocks
        for dtype in [torch.float32,torch.float64]:
            for scale in [1e-3,1.,1e6,1e20,1e30]:
                with self.subTest(dtype=dtype,scale=scale):
                    current=features.to(dtype)
                    result=normalized_rotary(current,mean.to(dtype),(cov*scale).to(dtype),frequency.to(dtype))
                    self.assertTrue(torch.isfinite(result).all())
                    torch.testing.assert_close(result.double().norm(dim=-1),current.double().norm(dim=-1),rtol=2e-6,atol=2e-6)

    def test_zero_content_returns_zero_with_finite_gradients(self):
        features=torch.zeros(1,4,dtype=DTYPE,requires_grad=True)
        result=normalized_rotary(features,torch.zeros(1,2,dtype=DTYPE),torch.eye(2,dtype=DTYPE)[None],torch.eye(2,dtype=DTYPE))
        torch.testing.assert_close(result,torch.zeros_like(result),rtol=0,atol=0)
        result.sum().backward()
        self.assertTrue(torch.isfinite(features.grad).all())


class NormalizedRotaryMeaningTests(unittest.TestCase):
    def test_map_is_nonlinear_and_does_not_preserve_same_geometry_inner_products(self):
        mean=torch.zeros(1,2,dtype=DTYPE);cov=torch.eye(2,dtype=DTYPE)[None]
        frequency=torch.tensor([[1.,0.],[2.,0.]],dtype=DTYPE)
        a=torch.tensor([[1.,0.,0.,0.]],dtype=DTYPE)
        b=torch.tensor([[0.,0.,1.,0.]],dtype=DTYPE)
        fa=normalized_rotary(a,mean,cov,frequency)
        fb=normalized_rotary(b,mean,cov,frequency)
        fab=normalized_rotary(a+b,mean,cov,frequency)
        self.assertGreater(float((fab-fa-fb).norm()),.5)
        original_inner=(a*(a+b)).sum()
        transformed_inner=(fa*fab).sum()
        self.assertGreater(float(abs(transformed_inner-original_inner)),.3)

    def test_normalization_removes_common_uncertainty_attenuation(self):
        features=torch.tensor([[.3,-.7,.8,.4]],dtype=DTYPE)
        mean=torch.tensor([[.2,-.1]],dtype=DTYPE)
        frequency=torch.eye(2,dtype=DTYPE)  # both frequencies have norm 1
        ordinary=rotary(features,mean,frequency)
        for variance in [.1,100.,1e6,1e20]:
            cov=torch.eye(2,dtype=DTYPE)[None]*variance
            torch.testing.assert_close(normalized_rotary(features,mean,cov,frequency),ordinary,rtol=1e-12,atol=1e-12)
        raw=rotary(features,mean,frequency,torch.eye(2,dtype=DTYPE)[None]*.1)
        self.assertLess(float(raw.norm()),float(ordinary.norm()))

    def test_raw_covariance_sum_property_is_lost_after_content_normalization(self):
        frequency=torch.tensor([[1.,0.],[2.,0.]],dtype=DTYPE)
        mean=torch.zeros(1,2,dtype=DTYPE)
        q=F.normalize(torch.tensor([[1.,0.,1.,0.]],dtype=DTYPE),dim=-1)
        k=F.normalize(torch.tensor([[1.,0.,2.,0.]],dtype=DTYPE),dim=-1)
        zero=torch.zeros(1,2,2,dtype=DTYPE);one=torch.eye(2,dtype=DTYPE)[None]
        raw_a=(rotary(q,mean,frequency,zero)*rotary(k,mean,frequency,one*2)).sum()
        raw_b=(rotary(q,mean,frequency,one)*rotary(k,mean,frequency,one)).sum()
        torch.testing.assert_close(raw_a,raw_b,rtol=1e-12,atol=1e-12)
        normalized_a=(normalized_rotary(q,mean,zero,frequency)*normalized_rotary(k,mean,one*2,frequency)).sum()
        normalized_b=(normalized_rotary(q,mean,one,frequency)*normalized_rotary(k,mean,one,frequency)).sum()
        self.assertGreater(float(abs(normalized_a-normalized_b)),.1)


class NormalizedRotaryRegressionTests(unittest.TestCase):
    def test_pre_fix_inactive_max_amplitude_channel_counterexample_is_preserved(self):
        for dtype in [torch.float32,torch.float64]:
            inputs=sparse_fixture(dtype=dtype)
            result=legacy_normalized_rotary(*inputs)
            self.assertEqual(float(inputs[0].norm()),1.)
            torch.testing.assert_close(result,torch.zeros_like(result),rtol=0,atol=0)
            torch.testing.assert_close(normalized_rotary(*inputs),inputs[0],rtol=1e-12,atol=0)

    def test_sparse_content_norm_at_huge_covariance_regression(self):
        inputs=sparse_fixture()
        torch.testing.assert_close(normalized_rotary(*inputs).norm(dim=-1),inputs[0].norm(dim=-1),rtol=1e-12,atol=0)

    def test_pre_fix_epsilon_floor_counterexample_is_preserved(self):
        inputs=sparse_fixture(scale=20.)
        result=legacy_normalized_rotary(*inputs)
        # Active amplitude is exp(-30), still representable in float64, but
        # below F.normalize's default 1e-12 denominator clamp.
        expected=torch.exp(torch.tensor(-30.,dtype=DTYPE))/1e-12
        self.assertGreater(float(result.norm()),0.)
        torch.testing.assert_close(result.norm(),expected,rtol=1e-12,atol=0)
        self.assertLess(float(result.norm()),.1)
        torch.testing.assert_close(normalized_rotary(*inputs),inputs[0],rtol=1e-12,atol=0)

    def test_sparse_content_norm_before_underflow_regression(self):
        inputs=sparse_fixture(scale=20.)
        torch.testing.assert_close(normalized_rotary(*inputs).norm(dim=-1),inputs[0].norm(dim=-1),rtol=1e-12,atol=0)

    def test_tiny_nonzero_content_point_limit_regression(self):
        features=torch.tensor([[1e-15,0.,0.,0.]],dtype=DTYPE)
        mean=torch.zeros(1,2,dtype=DTYPE);cov=torch.zeros(1,2,2,dtype=DTYPE)
        frequency=torch.eye(2,dtype=DTYPE)
        torch.testing.assert_close(normalized_rotary(features,mean,cov,frequency),features,rtol=1e-12,atol=0)

    def test_offset_is_not_floating_point_equivalent_when_raw_expectation_underflows(self):
        features=torch.tensor([[1.,0.,1.,0.]],dtype=DTYPE)
        mean=torch.zeros(1,2,dtype=DTYPE);cov=torch.eye(2,dtype=DTYPE)[None]*2000
        frequency=torch.eye(2,dtype=DTYPE)
        raw=rotary(features,mean,frequency,cov)
        torch.testing.assert_close(raw,torch.zeros_like(raw),rtol=0,atol=0)
        shifted=normalized_rotary(features,mean,cov,frequency)
        torch.testing.assert_close(shifted,features,rtol=1e-12,atol=0)

    def test_extreme_sparse_case_has_unrepresentable_analytic_content_jacobian(self):
        inputs=list(sparse_fixture(dtype=torch.float32))
        inputs[0]=inputs[0].requires_grad_()
        result=normalized_rotary(*inputs)
        self.assertTrue(torch.isfinite(result).all())
        # At the empty low-frequency block the exact derivative coefficient is
        # exp((4-1)*1000/2)=exp(1500), outside float64 as well as float32.
        self.assertGreater(1500.,float(torch.tensor(torch.finfo(torch.float64).max,dtype=DTYPE).log()))
        result[0,0].backward()
        self.assertTrue(torch.isinf(inputs[0].grad[0,0]))


class NormalizedMatcherTests(unittest.TestCase):
    def test_no_extra_parameters_and_all_normalized_variants_execute(self):
        data=model_fixture()
        baseline=GaussianPEMatcher('projected_rope',count=16)
        expected={name:tuple(p.shape) for name,p in baseline.named_parameters()}
        for name in ['normalized_fixed','normalized_shape','normalized_trace','normalized_area','normalized_shuffled']:
            with self.subTest(method=name):
                model=GaussianPEMatcher(name,count=16,resolution=.25)
                self.assertEqual({n:tuple(p.shape) for n,p in model.named_parameters()},expected)
                self.assertEqual(model(data).shape,(1,4,5))
                self.assertTrue(torch.isfinite(model(data)).all())

    def test_counterfactual_inputs_keep_means_query_covariance_and_resolution(self):
        data=model_fixture()
        for name in ['normalized_shape','normalized_trace','normalized_area','normalized_shuffled']:
            model=GaussianPEMatcher(name,count=16,resolution=.25)
            mq,cq,mk,ck=model.positional_inputs(data)
            torch.testing.assert_close(mq,data['pixq'],rtol=0,atol=0)
            torch.testing.assert_close(mk,data['k_to_q'],rtol=0,atol=0)
            torch.testing.assert_close(cq,data['projected_qcov']+torch.eye(2)*.25**2)
            original=data['projected_kcov']
            if name=='normalized_shape':expected=original
            elif name=='normalized_trace':expected=torch.eye(2)*(original.diagonal(dim1=-2,dim2=-1).sum(-1)/2)[...,None,None]
            elif name=='normalized_area':expected=torch.eye(2)*original.det().sqrt()[...,None,None]
            else:expected=original.roll(13,dims=-3)
            torch.testing.assert_close(ck,expected+torch.eye(2)*.25**2)

    def test_model_common_pixel_translation_leaves_logits_unchanged(self):
        data=model_fixture()
        changed=dict(data)
        changed['pixq']=data['pixq']+torch.tensor([.6,-1.2])
        changed['k_to_q']=data['k_to_q']+torch.tensor([.6,-1.2])
        model=GaussianPEMatcher('normalized_shape',count=16,resolution=.25)
        torch.testing.assert_close(model(data),model(changed),rtol=2e-6,atol=2e-6)


if __name__=='__main__':unittest.main()
