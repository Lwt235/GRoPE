import unittest
import numpy as np
import torch
from models.round40_decoupled_pe import DecoupledDepthAttention, NEW_ROLES, shape_descriptor, squared_distance
from experiments.round40_decoupled_audit import toy_data, numpy_logits
from experiments.round40_decoupled_data import frame_support


class DecoupledTests(unittest.TestCase):
    def test_independent_scores(self):
        for role in NEW_ROLES:
            d=toy_data(432 if role=='shape_shuffle' else 12)
            torch.manual_seed(11);m=DecoupledDepthAttention(role).double()
            got=m.attention_logits(d).detach().numpy()
            expected=numpy_logits(d,m.state_dict(),role)
            np.testing.assert_allclose(got,expected,rtol=0,atol=1e-9)

    def test_zero_shape_and_translation(self):
        d=toy_data();torch.manual_seed(11);m=DecoupledDepthAttention('decoupled').double()
        torch.manual_seed(11);center=DecoupledDepthAttention('center_rope').double()
        torch.testing.assert_close(m.attention_logits(d,True),center.attention_logits(d),rtol=0,atol=0)
        translated=dict(d,q_anchor3=d['q_anchor3']+7.,k_anchor3=d['k_anchor3']+7.)
        torch.testing.assert_close(m.attention_logits(d),m.attention_logits(translated),rtol=0,atol=1e-10)

    def test_shape_rotation_scale_and_repeated_spectrum(self):
        d=toy_data();c=d['q_cov3'];k=d['k_cov3']
        rot=torch.tensor([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]],dtype=torch.float64)
        base=squared_distance(shape_descriptor(c),shape_descriptor(k))
        changed=squared_distance(shape_descriptor(49*(rot@c@rot.T)),shape_descriptor(49*(rot@k@rot.T)))
        torch.testing.assert_close(base,changed,atol=1e-11,rtol=0)
        repeated=torch.eye(3,dtype=torch.float64,requires_grad=True)
        shape_descriptor(repeated).square().sum().backward()
        self.assertTrue(torch.isfinite(repeated.grad).all())

    def test_real_v_gradient_and_label_isolation(self):
        d=toy_data();m=DecoupledDepthAttention('decoupled').double()
        pred=m(d);loss=(pred-d['target']).square().mean();loss.backward()
        for name in ('qk.weight','value.weight','strength_raw'):
            grad=dict(m.named_parameters())[name].grad
            self.assertTrue(torch.isfinite(grad).all());self.assertGreater(float(grad.norm()),0.)
        injected=dict(d,target=torch.full_like(d['target'],float('nan')),valid=~d['valid'])
        torch.testing.assert_close(m(injected),m(d),atol=0,rtol=0)
        self.assertGreater(float((m(d,intervention='zero_v')-m(d)).abs().max()),0.)

    def test_active_parameter_capacity(self):
        counts={r:sum(p.numel() for p in DecoupledDepthAttention(r).parameters()) for r in NEW_ROLES}
        self.assertEqual(counts['center_rope'],41345)
        self.assertEqual(counts['decoupled'],41346)
        self.assertEqual(counts['center_rbf'],counts['decoupled'])

    def test_central_covariance_not_anchor_second_moment(self):
        K=np.array([[100.,0.,95.5],[0.,100.,71.5],[0.,0.,1.]])
        depth=np.full((144,192),2.,np.float32)
        a,c=frame_support(depth,K)
        expected=np.diag([64/12,64/12,0.])*(2/100)**2
        np.testing.assert_allclose(c,np.broadcast_to(expected,c.shape),atol=1e-14,rtol=0)
        np.testing.assert_array_equal(a[:,2],np.full(432,2.))
        # Within a patch, central four pixels differ from the patch majority:
        # central moments must not include the displacement from the anchor.
        depth[:8,:8]=3.;depth[3:5,3:5]=2.
        a,c=frame_support(depth,K)
        self.assertEqual(a[0,2],2.)
        self.assertAlmostEqual(c[0,2,2],(4/64)*(60/64),places=12)


if __name__=='__main__':
    unittest.main()
