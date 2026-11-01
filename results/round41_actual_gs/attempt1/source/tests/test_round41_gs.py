import unittest
import numpy as np
import torch
from models.round41_gs_readout import GSReadout,ROLES
from experiments.round41_gs_audit import toy,numpy_logits
from experiments.round41_gs_data import pixel_support,query,TRAIN_VIEWS


class ActualGSChecks(unittest.TestCase):
    def test_independent_scores_and_gradients(self):
        q,k=toy()
        for role in ROLES:
            model=GSReadout(role,12,17).double()
            score=model.logits(q,k)
            np.testing.assert_allclose(score.detach().numpy(),numpy_logits(q,k,model.state_dict(),role),rtol=1e-10,atol=1e-10)
            loss=(model(q,k)-q['rgb']).square().mean();loss.backward()
            self.assertTrue(torch.isfinite(model.qk.weight.grad).all())
            self.assertGreater(float(model.qk.weight.grad.norm()),0)
            if hasattr(model,'strength_raw'):self.assertTrue(torch.isfinite(model.strength_raw.grad))

    def test_constant_value_and_true_mean_shape_activity(self):
        q,k=toy();model=GSReadout('decoupled',12,17).double()
        changed={**q,'cov':q['cov']*3}
        self.assertGreater(float((model.logits(q,k).softmax(-1)-model.logits(changed,k).softmax(-1)).abs().max()),1e-5)
        k={**k,'rgb':torch.ones_like(k['rgb'])*.37}
        torch.testing.assert_close(model(q,k),torch.ones((12,3),dtype=torch.float64)*.37,rtol=1e-12,atol=1e-12)

    def test_flat_plane_pixel_covariance(self):
        z=torch.full((5,7),2.,dtype=torch.float64);K=torch.diag(torch.tensor([20.,30.,1.],dtype=torch.float64))
        a,c,diag=pixel_support(z,torch.ones_like(z),K)
        expected=torch.diag(torch.tensor([(2/20)**2/12,(2/30)**2/12,0.],dtype=torch.float64)).expand(35,3,3)
        torch.testing.assert_close(c,expected,rtol=1e-12,atol=1e-12)
        self.assertEqual(diag['invalid_pixels'],0);self.assertEqual(a.shape,(35,3))

    def test_target_feature_exclusion(self):
        q,k=toy(12,17);model=GSReadout('decoupled',12,17).double()
        views=[]
        for i in range(24):
            feature=q['feature'].clone();feature[:,i]+=i*.1
            views.append(dict(center_feature=feature,visible=torch.ones(12,dtype=torch.bool),
                q_anchor=q['anchor'],q_cov=q['cov'],q_xy=q['xy'],q_projected_cov=q['projected_cov']))
        data=dict(views=views,train_sum=sum(views[i]['center_feature'] for i in TRAIN_VIEWS),train_count=torch.full((12,),20))
        ids=torch.arange(12);before=query(data,1,ids,model,target=0)['feature']
        change=torch.randn_like(views[0]['center_feature'])*50
        views[0]['center_feature']+=change;data['train_sum']+=change
        after=query(data,1,ids,model,target=0)['feature']
        torch.testing.assert_close(before,after,rtol=1e-12,atol=1e-12)


if __name__=='__main__':unittest.main()
