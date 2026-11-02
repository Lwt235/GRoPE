import unittest
import numpy as np
import torch
from models.round46_native3d_pe import ROLES,TWO_D,SpatialMap3D,make_model
from experiments.round45_phase_warp_train import parameter_digest
from experiments.round46_native3d_data import center_points,pair_geometry
from experiments.round46_native3d_audit import numpy_phase,reference_geometry


def toy_data(dim):
    torch.manual_seed(159)
    data={k:torch.randn(2,7,128) for k in ('q','k')}
    data.update({k:torch.randn(2,7,dim)*6 for k in ('qmean','kmean','k_to_q','q_to_k')})
    data.update({k:torch.zeros(2,7,dim,dim) for k in ('qcov','kcov','k_to_qcov','q_to_kcov')})
    return data


def frame_pair():
    y,x=np.mgrid[:144,:192]
    K=np.array([[160.,0.,95.5],[0.,155.,71.5],[0.,0.,1.]])
    q=dict(predicted_depth=(2+.03*x+.01*y).astype('float32'),K=K,c2w=np.eye(4))
    pose=np.eye(4);angle=.31
    pose[:3,:3]=[[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]]
    pose[:3,3]=[.4,-.2,.15]
    k=dict(predicted_depth=(3+.015*x+.02*y).astype('float32'),K=K.copy(),c2w=pose)
    return q,k


class Native3DTests(unittest.TestCase):
    def test_complete_initialization_and_active_capacity(self):
        counts=(768,448,192,576,320,448,9,72,0,0,252)
        for seed in (11,22,33,44,55):
            for role,count in zip(ROLES,counts):
                dim=2 if role in TWO_D else 3;data=toy_data(dim)
                torch.manual_seed(seed);point=make_model('center2d' if dim==2 else 'point3d')
                torch.manual_seed(seed);model=make_model(role)
                self.assertEqual(parameter_digest(model),parameter_digest(point))
                self.assertEqual(sum(p.numel() for p in model.parameters()),166611+count)
                with torch.no_grad():torch.testing.assert_close(model(data),point(data),atol=0,rtol=0)

    def test_independent_nonzero_phases_and_norm(self):
        for role in ROLES:
            torch.manual_seed(119);model=make_model(role).double()
            dim=2 if role in TWO_D else 3
            with torch.no_grad():
                for p in model.position_field.parameters():p.add_(torch.linspace(-.04,.05,p.numel()).reshape_as(p))
            mean=torch.randn(9,dim,dtype=torch.float64)*7
            coords,freq=model.position_arguments(mean)
            phase=numpy_phase(role,model.state_dict(),mean)
            np.testing.assert_allclose((coords@freq.T).detach(),phase,rtol=2e-12,atol=2e-12)
            h=torch.randn(1,2,9,48,dtype=torch.float64)
            actual=model._position_encode(h,mean[None],None)
            z=(h.numpy()[...,::2]+1j*h.numpy()[...,1::2])*np.exp(1j*phase)
            reference=np.stack((z.real,z.imag),-1).reshape(1,2,9,48)
            np.testing.assert_allclose(actual.detach(),reference,rtol=2e-12,atol=2e-12)
            torch.testing.assert_close(actual.reshape(1,2,9,24,2).square().sum(-1),
                h.reshape(1,2,9,24,2).square().sum(-1),rtol=2e-12,atol=2e-12)

    def test_three_dimensional_shape_gradients(self):
        torch.manual_seed(29);field=SpatialMap3D('gaussian3d').double()
        x=torch.randn(37,3,dtype=torch.float64)*5;weight=torch.randn_like(x)
        (field(x)*weight).sum().backward()
        self.assertGreater(float(field.projection.grad.norm()),0.)
        for name,p in field.named_parameters():
            if name!='projection':self.assertEqual(float(p.grad.norm()),0.)
        with torch.no_grad():field.projection.add_(-.01*field.projection.grad)
        field.zero_grad();(field(x)*weight).sum().backward()
        for p in field.parameters():
            self.assertTrue(torch.isfinite(p.grad).all());self.assertGreater(float(p.grad.norm()),0.)
        for axis in range(3):self.assertGreater(float(field.lower.grad[:,axis].norm()),0.)
        derivative=float(field.lower.grad[31,2])
        with torch.no_grad():
            field.lower[31,2]+=1e-6;plus=float((field(x)*weight).sum())
            field.lower[31,2]-=2e-6;minus=float((field(x)*weight).sum())
            field.lower[31,2]+=1e-6
        np.testing.assert_allclose((plus-minus)/2e-6,derivative,atol=2e-7,rtol=2e-6)
        far=torch.full((2,3),1e6,dtype=torch.float64,requires_grad=True)
        response=field(far);torch.testing.assert_close(response,torch.zeros_like(response),atol=0,rtol=0)
        response.sum().backward();self.assertTrue(torch.isfinite(far.grad).all())

    def test_center_geometry_reference_and_centroid_difference(self):
        q,k=frame_pair();actual=pair_geometry(q,k);expected=reference_geometry(q,k)
        for key in actual:np.testing.assert_allclose(actual[key],expected[key],atol=3e-5,rtol=3e-7)
        both=np.concatenate((actual['qmean'],actual['kmean'])).astype(float)/12
        np.testing.assert_allclose(both.mean(0),0,atol=2e-8)
        self.assertAlmostEqual(float(np.square(both).sum(-1).mean()),1.,places=7)
        np.testing.assert_array_equal(actual['qmean'],actual['q_to_k'])
        np.testing.assert_array_equal(actual['kmean'],actual['k_to_q'])
        yy,xx=np.mgrid[:144,:192];z=q['predicted_depth'];K=q['K']
        xyz=np.stack(((xx-K[0,2])*z/K[0,0],(yy-K[1,2])*z/K[1,1],z),-1)
        centroid=xyz.reshape(18,8,24,8,3).transpose(0,2,1,3,4).reshape(432,64,3).mean(1)
        self.assertGreater(float(abs(center_points(q)-centroid).max()),1e-4)

    def test_common_world_gauge_and_scale(self):
        q,k=frame_pair();original=pair_geometry(q,k)
        H=np.eye(4);H[:3,:3]=[[0,-1,0],[1,0,0],[0,0,1]];H[:3,3]=[7.,-2.,3.]
        changed=pair_geometry(dict(q,c2w=H@q['c2w']),dict(k,c2w=H@k['c2w']))
        for key in original:np.testing.assert_allclose(changed[key],original[key],atol=2e-5,rtol=2e-7)
        # Keep predicted depths inside the fixed clamp in this unit-change test.
        q=dict(q,predicted_depth=np.full((144,192),3.,dtype=np.float32))
        k=dict(k,predicted_depth=np.full((144,192),4.,dtype=np.float32))
        original=pair_geometry(q,k)
        qq=dict(q,c2w=q['c2w'].copy(),predicted_depth=2*q['predicted_depth'])
        kk=dict(k,c2w=k['c2w'].copy(),predicted_depth=2*k['predicted_depth'])
        qq['c2w'][:3,3]*=2;kk['c2w'][:3,3]*=2
        scaled=pair_geometry(qq,kk)
        for key in original:np.testing.assert_allclose(scaled[key],original[key],atol=2e-5,rtol=2e-7)

    def test_input_scope_and_affine_equivalence(self):
        data=toy_data(3);torch.manual_seed(159);model=make_model('gaussian3d')
        changed={k:v.clone() for k,v in data.items()}
        for k in ('qcov','kcov','k_to_qcov','q_to_kcov'):changed[k].fill_(123)
        changed['gt_valid']=torch.zeros(2,7,dtype=torch.bool)
        with torch.no_grad():
            before=model(data);torch.testing.assert_close(model(changed),before,atol=0,rtol=0)
            changed['qmean']+=2;self.assertGreater(float((model(changed)-before).abs().max()),1e-5)
        affine=make_model('affine3d').double()
        with torch.no_grad():affine.position_field.affine_offset.normal_(std=.1)
        x=torch.randn(13,3,dtype=torch.float64)*7;u,w=affine.position_arguments(x)
        M=torch.eye(3,dtype=torch.float64)+affine.position_field.affine_offset
        torch.testing.assert_close((u[1:]-u[:1])@w.T,(x[1:]-x[:1])@(w@M).T,atol=2e-12,rtol=2e-12)


if __name__=='__main__':unittest.main()
