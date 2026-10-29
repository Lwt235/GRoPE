import unittest
import numpy as np
import torch
import torch.nn.functional as F
from experiments.rendered_data import make_dataset
from experiments.rendered_scene import WIDTH,HEIGHT,FOCAL,PATCH
from experiments.rendered_metrics import average_precision
from models.rendered_attention import RenderedMatcher,homography_features,mixture_rotary,METHODS,GATE_METHODS
from models.geometry import rotary


class RenderedTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.data,cls.stats=make_dataset(902,3,'oracle')

    def test_correspondence_inside_target_patch(self):
        d=self.data
        visible=d['labels']<48
        error=d['pixel_error'].gather(-1,d['labels'].clamp(max=47)[...,None]).squeeze(-1)
        self.assertTrue(bool((error[visible] <= PATCH/2*2**.5+1e-4).all()))
        self.assertTrue(bool(visible.any() and (~visible).any()))

    def test_perturbations_keep_rgb_labels_and_unrelated_geometry_fixed(self):
        base=self.data
        for condition in ['depth_noise','normal_noise','scale_noise','mixed','depth_fit']:
            altered,_=make_dataset(902,3,condition)
            for field in ['q','k','labels','pixel_error']:
                torch.testing.assert_close(base[field],altered[field],atol=0,rtol=0)
            if condition in ['normal_noise','scale_noise']:
                torch.testing.assert_close(base['mq'],altered['mq'],atol=0,rtol=0)
            if condition=='scale_noise':
                torch.testing.assert_close(base['nq'],altered['nq'],atol=0,rtol=0)

    def test_labels_and_errors_do_not_enter_model(self):
        d=self.data
        changed={**d,'labels':torch.zeros_like(d['labels']),
                 'pixel_error':torch.rand_like(d['pixel_error'])*1000}
        for method in METHODS+GATE_METHODS:
            model=RenderedMatcher(method)
            torch.testing.assert_close(model(d),model(changed),atol=0,rtol=0)

    def test_all_operator_losses_and_gradients_finite(self):
        d,_=make_dataset(903,3,'mixed')
        for method in METHODS+GATE_METHODS:
            model=RenderedMatcher(method)
            logits=model(d)
            loss=F.cross_entropy(logits.flatten(0,1),d['labels'].flatten())
            loss.backward()
            self.assertTrue(bool(torch.isfinite(loss)),method)
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None),method)

    def test_homography_query_key_factorization(self):
        torch.manual_seed(3)
        hq=torch.eye(3).repeat(2,1,1)+torch.randn(2,3,3)*.1
        hk=torch.eye(3).repeat(2,1,1)+torch.randn(2,3,3)*.1
        q,k=torch.randn(2,24),torch.randn(2,24)
        actual=(homography_features(q,hq,True)*homography_features(k,hk,False)).sum(-1)
        explicit=torch.einsum('nci,nij,ncj->n',q.reshape(2,8,3),hq@torch.linalg.inv(hk),k.reshape(2,8,3))
        torch.testing.assert_close(actual,explicit,atol=2e-5,rtol=2e-5)

    def test_oracle_plane_homography_reprojects_visible_points(self):
        d=self.data
        errors=[]
        for scene in range(3):
            for token in range(48):
                target=int(d['labels'][scene,token])
                if target==48: continue
                point=d['mq'][scene,token]
                relative=d['hk'][scene,target]@torch.linalg.inv(d['hq'][scene,token])
                projected=relative@(point/point[2])
                # Independent projection of the physical point is available via
                # the center-to-target displacement magnitudes in pixel_error.
                uv=projected[:2]/projected[2]*FOCAL+torch.tensor([WIDTH/2,HEIGHT/2])
                center=d['pixk'][scene]*PATCH
                expected=d['pixel_error'][scene,token]
                errors.append((torch.linalg.norm(uv-center,dim=-1)-expected).abs().max())
        self.assertLess(float(torch.stack(errors).max()),.003)

    def test_mixture_zero_variance_limit(self):
        torch.manual_seed(3)
        feature=torch.randn(1,2,48)
        mean=torch.randn(1,2,2)
        freq=torch.eye(2)[torch.arange(24)%2]
        actual=mixture_rotary(feature,mean[:,:,None].expand(-1,-1,3,-1),torch.zeros(1,2,3,2,2),freq)
        torch.testing.assert_close(actual,rotary(feature,mean,freq))

    def test_constant_unmatched_score_ap_equals_prevalence(self):
        labels=np.array([0,1,0,0,1],dtype=bool)
        self.assertAlmostEqual(average_precision(labels,np.ones(5)),.4)

    def test_dustbin_offset_does_not_change_ranking_or_ap(self):
        torch.manual_seed(45)
        logits=torch.randn(100,9,dtype=torch.float64)
        changed=logits.clone()
        changed[:,-1]-=2.
        truth=np.arange(100)%3==0
        self.assertTrue(torch.equal(logits[:,:-1].argmax(-1),changed[:,:-1].argmax(-1)))
        self.assertAlmostEqual(average_precision(truth,logits.softmax(-1)[:,-1].numpy()),
                               average_precision(truth,changed.softmax(-1)[:,-1].numpy()))


if __name__=='__main__':
    unittest.main()
