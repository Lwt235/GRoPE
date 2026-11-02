import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch

from models.round49_joint_gaussian import ROLES, dimension, base_role, make_model
from models.round46_native3d_pe import make_model as make_rope
from models.round47_conditional_gaussian import make_model as make_gaussian
from experiments.round49_joint_train import parameter_digest, score_sample
from experiments.round49_joint_audit import check_sample


def toy(dim, dtype=torch.float32):
    data={key:torch.randn(1,7,128,dtype=dtype) for key in ('q','k')}
    for key in ('qmean','kmean','k_to_q','q_to_k'):
        data[key]=torch.randn(1,7,dim,dtype=dtype)*6
    for key in ('qcov','kcov','k_to_qcov','q_to_kcov'):
        data[key]=torch.randn(1,7,dim,dim,dtype=dtype)
    return data


class JointGaussianTest(unittest.TestCase):
    def test_paired_initialization_counts_and_initial_function(self):
        counts={'conditional2d':970,'diagonal2d':776,'isotropic2d':582,'static2d':10,
                'fixed2d':0,'free_quadratic2d':970,'conditional3d':1746,'diagonal3d':1164,
                'isotropic3d':776,'static3d':18,'fixed3d':0,'free_quadratic3d':1746}
        for seed in (11,22,33,44,55):
            torch.manual_seed(seed);expected=parameter_digest(make_rope('center2d'))
            for role in ROLES:
                with self.subTest(seed=seed,role=role):
                    data=toy(dimension(role))
                    torch.manual_seed(seed);model=make_model(role)
                    self.assertEqual(parameter_digest(model),expected)
                    self.assertEqual(sum(p.numel() for p in model.position_field.parameters()),counts[base_role(role)])
                    self.assertEqual(sum(p.numel() for p in model.parameters()),166611+counts[base_role(role)])
                    torch.manual_seed(seed);fixed=make_model('joint_fixed'+str(dimension(role))+'d')
                    with torch.no_grad():
                        torch.testing.assert_close(model(data),fixed(data),atol=0,rtol=0)

    def test_disabling_either_part_recovers_old_implementation(self):
        torch.manual_seed(4901)
        for dim in (2,3):
            data=toy(dim)
            torch.manual_seed(11);joint=make_model('joint_conditional'+str(dim)+'d')
            torch.manual_seed(11);rope=make_rope('center2d' if dim==2 else 'point3d')
            self.assertTrue(torch.equal(joint.frequency,rope.frequency))
            with patch.object(joint.position_field,'bias',side_effect=lambda f,x,y:
                              f.new_zeros(f.shape[0],2,x.shape[1],y.shape[1])):
                torch.testing.assert_close(joint(data),rope(data),atol=1e-6,rtol=1e-6)
            torch.manual_seed(11);gaussian=make_gaussian('conditional'+str(dim)+'d')
            with torch.no_grad():
                for p in joint.position_field.parameters():p.normal_(0,.03)
            gaussian.position_field.load_state_dict(joint.position_field.state_dict())
            with patch.object(joint,'rotate_content',side_effect=lambda content,mean:content):
                torch.testing.assert_close(joint(data),gaussian(data),atol=0,rtol=0)

    def test_nonzero_direct_complex_phase_and_gaussian_reference(self):
        torch.manual_seed(4902)
        with tempfile.TemporaryDirectory() as folder:
            for role in ROLES:
                model=make_model(role)
                with torch.no_grad():
                    for p in model.position_field.parameters():p.normal_(0,.04)
                path=Path(folder)/(role+'.npz')
                score_sample(model,toy(dimension(role)),path)
                result=check_sample(role,model.state_dict(),path)
                self.assertLess(result['p_max_error'],5e-5,role)
                self.assertEqual(result['winner_changes'],0,role)

    def test_initial_free_quadratic_function_and_derivatives(self):
        torch.manual_seed(4903)
        for dim in (2,3):
            data=toy(dim,dtype=torch.float64);weight=torch.randn(1,7,7,dtype=torch.float64)
            models=[];outputs=[]
            for kind in ('conditional','free_quadratic'):
                torch.manual_seed(11);model=make_model('joint_'+kind+str(dim)+'d').double()
                output=model(data);(output*weight).sum().backward()
                models.append(model);outputs.append(output)
            torch.testing.assert_close(outputs[0],outputs[1],atol=0,rtol=0)
            for left,right in zip(models[0].position_field.parameters(),models[1].position_field.parameters()):
                torch.testing.assert_close(left.grad,right.grad,atol=2e-10,rtol=2e-10)

    def test_position_and_value_gradients(self):
        torch.manual_seed(4904)
        for dim in (2,3):
            model=make_model('joint_conditional'+str(dim)+'d').double()
            data=toy(dim,dtype=torch.float64);weight=torch.randn(1,7,7,dtype=torch.float64)
            def loss():return (model(data)*weight).sum()
            loss().backward()
            field=model.position_field
            self.assertTrue((field.predictor.bias.grad.reshape(2,field.size).abs().sum(0)>1e-9).all())
            for layer in model.layers:
                self.assertGreater(float(layer.qkv.weight.grad[192:].norm()),1e-8)
            index=field.size-1;analytic=float(field.predictor.bias.grad[index]);eps=1e-5
            with torch.no_grad():
                field.predictor.bias[index]+=eps;plus=float(loss())
                field.predictor.bias[index]-=2*eps;minus=float(loss())
                field.predictor.bias[index]+=eps
            np.testing.assert_allclose(analytic,(plus-minus)/(2*eps),atol=1e-7,rtol=1e-4)
            self.assertGreater(float((model(data)-model(data,zero_values=True)).abs().max()),1e-4)

    def test_translation_permutation_and_no_label_or_covariance_input(self):
        torch.manual_seed(4905)
        for dim in (2,3):
            model=make_model('joint_conditional'+str(dim)+'d').double()
            with torch.no_grad():
                for p in model.position_field.parameters():p.normal_(0,.03)
            data=toy(dim,dtype=torch.float64);expected=model(data)
            shifted={k:v.clone() for k,v in data.items()}
            a=torch.randn(dim,dtype=torch.float64)*10;b=torch.randn(dim,dtype=torch.float64)*10
            for key in ('qmean','k_to_q'):shifted[key]+=a
            for key in ('kmean','q_to_k'):shifted[key]+=b
            torch.testing.assert_close(model(shifted),expected,atol=2e-10,rtol=2e-10)
            qi=torch.randperm(7);ki=torch.randperm(7);permuted=dict(data)
            for key in ('q','qmean','q_to_k','qcov','q_to_kcov'):permuted[key]=data[key][:,qi]
            for key in ('k','kmean','k_to_q','kcov','k_to_qcov'):permuted[key]=data[key][:,ki]
            torch.testing.assert_close(model(permuted),expected[:,qi][:,:,ki],atol=2e-10,rtol=2e-10)
            unused=dict(data)
            for key in ('qcov','kcov','k_to_qcov','q_to_kcov'):unused[key]=data[key]*1e6
            unused.update(gt_valid=torch.zeros(1,7,dtype=torch.bool),pixcontinuous=torch.ones(1,7,2))
            torch.testing.assert_close(model(unused),expected,atol=0,rtol=0)


if __name__=='__main__':
    torch.set_num_threads(4)
    unittest.main()
