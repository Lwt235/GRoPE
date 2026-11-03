import math
import unittest
import numpy as np
import torch
from scipy.linalg import expm
from experiments.round51_data import camera_fields
from experiments.round62_data import observed_geometry
from models.round58_support_pe import SupportPosition
from models.round62_rgb_relations import (make_model,NEW_ROLES,relation_inputs,
    gaussian_from_raw,common_frame,FLOOR,POINT_VARIANCE,select_inputs,REVERSE)


def fixture():
    torch.manual_seed(6201)
    k=np.array([[170.,0,94.7],[0,172.,73.1],[0,0,1.]])
    c=np.eye(4);c[:3,:3]=expm(np.array([[0.,-.3,.2],[.3,0.,-.1],[-.2,.1,0.]]))
    c[:3,3]=[.3,-.2,.1]
    data={x:torch.randn(1,7,128,dtype=torch.float64) for x in ('q','k')}
    for prefix,pose in (('q',np.eye(4)),('k',c)):
        for name,value in camera_fields(dict(K=k),pose).items():
            data[prefix+name]=torch.from_numpy(value if name=='K' else value[:7]).double()[None]
        data[prefix+'observed_mean']=torch.randn(1,7,3,dtype=torch.float64)*.1
        data[prefix+'observed_mean'][...,2]+=1
        l=torch.randn(1,7,3,3,dtype=torch.float64)*.03
        data[prefix+'observed_cov']=l@l.transpose(-1,-2)+.0001*torch.eye(3,dtype=torch.float64)
    return data


def numpy_relations(qm,qs,km,ks):
    result=np.empty((len(qm),len(km),11))
    for i in range(len(qm)):
        for j in range(len(km)):
            delta=qm[i]-km[j]
            pi,pj=np.linalg.inv(qs[i]),np.linalg.inv(ks[j])
            result[i,j]=[np.linalg.norm(delta)/.2,
                *(.5*np.log(np.linalg.eigvalsh(qs[i])/.04)),
                *(.5*np.log(np.linalg.eigvalsh(ks[j])/.04)),
                *np.log1p([delta@pi@delta,delta@pj@delta,np.trace(pi@ks[j]),np.trace(pj@qs[i])])]
    return result


class RGBRelationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(4)

    def test_relation_numpy_reverse_rigid_and_original_map(self):
        d=fixture();qm,qs,km,ks=[d[k] for k in ('qobserved_mean','qobserved_cov','kobserved_mean','kobserved_cov')]
        actual=relation_inputs(qm,qs,km,ks)
        np.testing.assert_allclose(actual.numpy()[0],numpy_relations(*[x.numpy()[0] for x in (qm,qs,km,ks)]),atol=2e-11)
        reverse=relation_inputs(km,ks,qm,qs)
        torch.testing.assert_close(reverse,actual[...,list(REVERSE)].transpose(1,2),atol=2e-11,rtol=1e-11)
        r=torch.from_numpy(expm(np.array([[0.,-.2,.4],[.2,0.,-.1],[-.4,.1,0.]])))
        shifted=[qm@r.T+.7,r@qs@r.T,km@r.T+.7,r@ks@r.T]
        torch.testing.assert_close(actual,relation_inputs(*shifted),atol=2e-11,rtol=1e-11)
        old=SupportPosition('invariant_mlp').double()
        with torch.no_grad():
            old_result=old(torch.cat((qm,km),1),torch.cat((qs,ks),1))[:, :7,7:]
            new_result=old.mlp(actual)
        torch.testing.assert_close(old_result,new_result,atol=1e-11,rtol=1e-11)

    def test_predicted_geometry_and_gradient(self):
        raw=torch.tensor([[[.2,.1,-.2,.3,.12,-.23,.07]]],dtype=torch.float64,requires_grad=True)
        rays=torch.tensor([[[.3,-.2,1.]]],dtype=torch.float64)
        mu,cov=gaussian_from_raw(raw,rays)
        a=raw.detach().numpy()[0,0]
        depth=1/30+np.logaddexp(0,a[0]+np.log(np.expm1(29/30)))
        l=np.diag(np.array([.025,.030,.010])*np.exp(2*np.tanh(a[1:4]/2)))
        l[np.tril_indices(3,-1)]=.03*np.tanh(a[4:])
        np.testing.assert_allclose(mu.detach().numpy()[0,0],depth*rays.numpy()[0,0],atol=1e-12)
        np.testing.assert_allclose(cov.detach().numpy()[0,0],l@l.T+FLOOR**2*np.eye(3),atol=1e-12)
        self.assertGreater(float(torch.linalg.eigvalsh(cov).min()),FLOOR**2)
        (mu.sum()+100*cov.sum()).backward()
        self.assertTrue(torch.isfinite(raw.grad).all())
        self.assertTrue((raw.grad.abs()>1e-9).all())

    def test_observed_plane_and_missing_fallback(self):
        k=np.array([[170.,0,94.7],[0,172.,73.1],[0,0,1.]])
        output,counts=observed_geometry(np.full((144,192),3.),k)
        expected=np.diag([5.25/170**2,5.25/172**2,0.])+FLOOR**2*np.eye(3)
        np.testing.assert_allclose(output['cov'],np.broadcast_to(expected,(432,3,3)),atol=1e-10)
        self.assertEqual(counts['missing_centers'],0)
        empty,counts=observed_geometry(np.zeros((144,192)),k)
        np.testing.assert_allclose(empty['cov'],np.broadcast_to(POINT_VARIANCE*np.eye(3),(432,3,3)),atol=1e-10)
        np.testing.assert_allclose(empty['mean'],output['mean'],atol=1e-7)
        self.assertEqual(counts['empty_patches'],432)

    def test_backbone_zero_reduction_permutation_and_input_contract(self):
        data=fixture()
        torch.manual_seed(11);reference=make_model('camera_only').double().eval()
        common={n:v for n,v in reference.state_dict().items() if not n.startswith('field.')}
        with torch.no_grad():expected=reference.encode(data)
        for role in NEW_ROLES:
            torch.manual_seed(11);model=make_model(role).double().eval()
            for n,v in common.items():torch.testing.assert_close(v,model.state_dict()[n],atol=0,rtol=0)
            with torch.no_grad():
                reduction=model.encode(data,zero_relation=True)
                for a,b in zip(expected,reduction):torch.testing.assert_close(a,b,atol=0,rtol=0)
                base=model(data)
                perm={k:(v.flip(1) if k not in ('qK','kK') else v) for k,v in data.items()}
                torch.testing.assert_close(model(perm).flip(1,2),base,atol=1e-10,rtol=1e-10)
                poisoned=dict(data,gt_valid=torch.zeros(1,7,dtype=torch.bool),key_weights=torch.randn(1,7,4))
                if not role.startswith('observed_'):
                    poisoned.update({k:torch.full_like(v,float('nan')) for k,v in data.items() if 'observed' in k})
                torch.testing.assert_close(model(poisoned),base,atol=0,rtol=0)
                self.assertFalse(any('observed' in k for k in select_inputs(data,role)) and not role.startswith('observed_'))

    def test_actual_network_prediction_and_relation_finite_difference(self):
        data=fixture()
        for role in ('predicted_full','predicted_point','free_relation','camera_mlp','observed_full'):
            torch.manual_seed(22);model=make_model(role).double()
            weight=torch.randn(1,7,7,dtype=torch.float64)
            loss=(model(data)*weight).sum();loss.backward()
            params=[(n,p) for n,p in model.named_parameters() if n.startswith('field.')]
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for _,p in params))
            chosen=[(n,p) for n,p in params if p.grad.abs().max()>1e-8]
            self.assertTrue(chosen)
            if role=='predicted_full':
                g=model.field['predictor'].weight.grad
                self.assertGreater(float(g[0].abs().max()),1e-8)
                self.assertGreater(float(g[1:].abs().max()),1e-8)
            n,p=chosen[0];index=np.unravel_index(int(p.grad.abs().argmax()),tuple(p.shape))
            analytic=float(p.grad[index]);original=float(p.detach()[index]);eps=1e-5
            with torch.no_grad():
                p[index]=original+eps;plus=float((model(data)*weight).sum())
                p[index]=original-eps;minus=float((model(data)*weight).sum())
                p[index]=original
            numeric=(plus-minus)/(2*eps)
            self.assertLess(abs(analytic-numeric),1e-5+1e-5*abs(numeric),(role,n,analytic,numeric))


if __name__=='__main__':unittest.main()
