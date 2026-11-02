import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch

from models.round47_conditional_gaussian import ROLES, dimension, make_model, GaussianPosition
from models.round46_native3d_pe import make_model as old_model
from experiments.round47_conditional_train import parameter_digest, score_sample
from experiments.round47_conditional_audit import check_sample


def toy(dim, batch=1, count=7, dtype=torch.float32):
    data={key:torch.randn(batch,count,128,dtype=dtype) for key in ('q','k')}
    for key in ('qmean','kmean','k_to_q','q_to_k'):
        data[key]=torch.randn(batch,count,dim,dtype=dtype)*6
    for key in ('qcov','kcov','k_to_qcov','q_to_kcov'):
        data[key]=torch.randn(batch,count,dim,dim,dtype=dtype)
    return data


class ConditionalGaussianTest(unittest.TestCase):
    def test_paired_initialization_and_counts(self):
        counts={'conditional2d':970,'diagonal2d':776,'isotropic2d':582,'static2d':10,
                'fixed2d':0,'free_quadratic2d':970,'conditional3d':1746,'diagonal3d':1164,
                'isotropic3d':776,'static3d':18,'fixed3d':0,'free_quadratic3d':1746,'nope':0}
        for seed in (11,22,33,44,55):
            torch.manual_seed(seed);identity=parameter_digest(old_model('center2d'))
            for role in ROLES:
                with self.subTest(seed=seed,role=role):
                    data=toy(dimension(role))
                    torch.manual_seed(seed);model=make_model(role)
                    self.assertEqual(parameter_digest(model),identity)
                    self.assertEqual(sum(p.numel() for p in model.position_field.parameters()),counts[role])
                    self.assertEqual(sum(p.numel() for p in model.parameters()),166611+counts[role])
                    torch.manual_seed(seed)
                    fixed=make_model('nope' if role=='nope' else 'fixed'+str(dimension(role))+'d')
                    with torch.no_grad():
                        self.assertTrue(torch.equal(model(data),fixed(data)))

    def test_direct_numpy_nonzero_probabilities(self):
        torch.manual_seed(4721)
        with tempfile.TemporaryDirectory() as folder:
            for role in ROLES:
                model=make_model(role)
                with torch.no_grad():
                    for p in model.position_field.parameters():p.normal_(0,.04)
                data=toy(dimension(role))
                path=Path(folder)/(role+'.npz')
                score_sample(model,data,path)
                result=check_sample(role,model.state_dict(),path)
                self.assertLess(result['p_max_error'],5e-5,role)
                self.assertEqual(result['winner_changes'],0,role)

    def test_shape_gradients_spd_and_finite_difference(self):
        torch.manual_seed(4722)
        for dim in (2,3):
            field=GaussianPosition('conditional'+str(dim)+'d').double()
            features=torch.randn(1,5,96,dtype=torch.float64)
            x=torch.randn(1,5,dim,dtype=torch.float64)*3
            y=torch.randn(1,9,dim,dtype=torch.float64)*3
            weight=torch.randn(1,2,5,9,dtype=torch.float64)
            def loss():return (field.bias(features,x,y).softmax(-1)*weight).sum()
            loss().backward()
            grad=field.predictor.bias.grad.reshape(2,field.size)
            self.assertTrue((grad.abs().sum(0)>1e-9).all())
            index=field.size-1;analytic=field.predictor.bias.grad[index].item();eps=1e-5
            with torch.no_grad():
                field.predictor.bias[index]+=eps;plus=loss().item()
                field.predictor.bias[index]-=2*eps;minus=loss().item()
                field.predictor.bias[index]+=eps
                field.predictor.weight.normal_(0,.08)
                field.predictor.bias.normal_(0,.1)
            self.assertAlmostEqual(analytic,(plus-minus)/(2*eps),places=7)
            precision,_=field.coefficients(features)
            self.assertTrue((torch.linalg.eigvalsh(precision)>0).all())

    def test_free_quadratic_matches_initial_value_and_jacobian(self):
        torch.manual_seed(4723)
        for dim in (2,3):
            fields=[GaussianPosition(kind+str(dim)+'d').double() for kind in ('conditional','free_quadratic')]
            features=torch.randn(1,5,96,dtype=torch.float64)
            x=torch.randn(1,5,dim,dtype=torch.float64)*4
            y=torch.randn(1,8,dim,dtype=torch.float64)*4
            weight=torch.randn(1,2,5,8,dtype=torch.float64)
            outputs=[]
            for field in fields:
                p=field.bias(features,x,y).softmax(-1);outputs.append(p)
                (p*weight).sum().backward()
            torch.testing.assert_close(outputs[0],outputs[1],atol=0,rtol=0)
            for left,right in zip(fields[0].parameters(),fields[1].parameters()):
                torch.testing.assert_close(left.grad,right.grad,atol=2e-12,rtol=2e-12)

    def test_translation_permutation_and_input_scope(self):
        torch.manual_seed(4724)
        for dim in (2,3):
            model=make_model('conditional'+str(dim)+'d').double()
            with torch.no_grad():
                for p in model.position_field.parameters():p.normal_(0,.03)
            data=toy(dim,dtype=torch.float64)
            expected=model(data)
            shifted={k:v.clone() for k,v in data.items()}
            a=torch.randn(dim,dtype=torch.float64)*10;b=torch.randn(dim,dtype=torch.float64)*10
            for key in ('qmean','k_to_q'):shifted[key]+=a
            for key in ('kmean','q_to_k'):shifted[key]+=b
            torch.testing.assert_close(model(shifted),expected,atol=1e-11,rtol=1e-11)
            qi=torch.randperm(7);ki=torch.randperm(7)
            permuted=dict(data)
            for key in ('q','qmean','q_to_k','qcov','q_to_kcov'):permuted[key]=data[key][:,qi]
            for key in ('k','kmean','k_to_q','kcov','k_to_qcov'):permuted[key]=data[key][:,ki]
            torch.testing.assert_close(model(permuted),expected[:,qi][:,:,ki],atol=1e-11,rtol=1e-11)
            unused=dict(data)
            for key in ('qcov','kcov','k_to_qcov','q_to_kcov'):unused[key]=data[key]*1e6
            unused.update(gt_valid=torch.zeros(1,7,dtype=torch.bool),pixcontinuous=torch.ones(1,7,2))
            torch.testing.assert_close(model(unused),expected,atol=0,rtol=0)
            self.assertGreater(float((model(data,zero_values=True)-expected).abs().max()),1e-4)

    def test_joint_coordinate_and_precision_transform(self):
        rng=np.random.default_rng(4725)
        for dim in (2,3):
            x=rng.normal(size=(4,dim));y=rng.normal(size=(6,dim));m=rng.normal(size=(4,dim))
            factor=rng.normal(size=(4,dim,dim));precision=factor@factor.transpose(0,2,1)+np.eye(dim)
            rotation=np.linalg.qr(rng.normal(size=(dim,dim)))[0];scale=2.7;offset=rng.normal(size=dim)
            delta=y[None]-x[:,None]-m[:,None]
            score=-.5*np.einsum('qki,qij,qkj->qk',delta,precision,delta)
            xt=scale*x@rotation.T+offset;yt=scale*y@rotation.T+offset;mt=scale*m@rotation.T
            pt=rotation@precision@rotation.T/scale**2
            dt=yt[None]-xt[:,None]-mt[:,None]
            transformed=-.5*np.einsum('qki,qij,qkj->qk',dt,pt,dt)
            np.testing.assert_allclose(score,transformed,atol=3e-12,rtol=3e-12)


if __name__=='__main__':unittest.main()
