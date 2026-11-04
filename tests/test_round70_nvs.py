import numpy as np
import torch

from experiments.round70_nvs_data import (NVSData, edge_intrinsics, observed_moments,
                                         HEIGHT, WIDTH, PATCH)
from models.round70_nvs import (CompactNVS, ROLES, _SymmetricLog, descriptors,
                                hrope_matrices)
from models.round69_compact_relation import descriptors as old_descriptors


def test_camera_and_observed_patch_alignment():
    K = np.array([[520.,0,321.],[0,519.,239.],[0,0,1.]])
    newK = edge_intrinsics(K,(480,640))
    pixels = np.array([[.5,.5,1.],[4.,4.,1.],[127.5,95.5,1.]])
    old = pixels.copy(); old[:,:2] *= [640/WIDTH,480/HEIGHT]; old[:,:2] -= .5
    np.testing.assert_allclose(pixels@np.linalg.inv(newK).T,old@np.linalg.inv(K).T,atol=1e-14)
    mean,cov,coverage = observed_moments(np.full((480,640),3.,np.float32),K)
    yy,xx = np.meshgrid((np.arange(HEIGHT//PATCH)+.5)*PATCH,
                        (np.arange(WIDTH//PATCH)+.5)*PATCH,indexing='ij')
    expect = np.c_[xx.ravel(),yy.ravel(),np.ones(xx.size)]@np.linalg.inv(newK).T
    np.testing.assert_allclose(mean,expect,atol=4e-8)
    assert np.all(coverage==1) and np.all(np.linalg.eigvalsh(cov)>0)


def test_matrix_log_gradient_and_r69_descriptor():
    torch.manual_seed(45)
    a = torch.randn(2,3,3,dtype=torch.double)
    s = (a@a.transpose(-1,-2)+.1*torch.eye(3)).requires_grad_()
    assert torch.autograd.gradcheck(_SymmetricLog.apply,(s,),eps=1e-6,atol=2e-5)
    iso = (torch.eye(3,dtype=torch.double)*.02).requires_grad_()
    g = torch.autograd.grad(_SymmetricLog.apply(iso).sum(),iso)[0]
    torch.testing.assert_close(g,torch.ones_like(g)*25.)
    mean = torch.randn(2,7,3)
    a = torch.randn(2,7,3,3); cov = a@a.transpose(-1,-2)+.1*torch.eye(3)
    new = descriptors(mean,cov,'relative5')
    old = old_descriptors('relative5',mean,cov)
    torch.testing.assert_close(new,old,rtol=2e-6,atol=2e-6)


def test_common_initialization_and_hrope_inverse():
    for seed in (11,22,33):
        reference = None
        for role in ROLES:
            torch.manual_seed(seed); model = CompactNVS(role)
            shared = {k:v for k,v in model.state_dict().items()
                      if not k.startswith(('relative.','hfield.','geometry_head.'))}
            if reference is None: reference = shared
            else:
                assert shared.keys()==reference.keys()
                for key in shared: assert torch.equal(shared[key],reference[key]),(role,key)
    raw = torch.randn(2,11,13)*.1
    raw[...,:6] += torch.tensor([1.,0,0,0,1.,0]); raw[...,11] += 1
    K = torch.tensor([[.8,0,.02],[0,1.1,-.03],[0,0,1.]]).expand(2,11,3,3)
    h,hi = hrope_matrices(raw,K)
    torch.testing.assert_close(h@hi,torch.eye(3).expand_as(h),atol=1e-6,rtol=1e-6)


def test_input_boundary():
    ds = NVSData('fit')
    inputs,labels = ds.batch([0],'rgb_prope','cpu')
    assert set(inputs)=={'ref_rgb','K','c2w'}
    assert labels.shape == (1,HEIGHT,WIDTH,3)
    for role in ('rgb_prope','observed_prope'):
        data,_ = ds.batch([0],role,'cpu')
        data['target_rgb'] = labels
        try: CompactNVS(role)(data)
        except ValueError: pass
        else: raise AssertionError('target RGB accepted')
    # Mutating cached sensor geometry cannot alter RGB-only inference inputs.
    for key in ('mean','cov','coverage'): ds.arrays[key].fill(np.nan)
    other,_ = ds.batch([0],'rgb_relative5','cpu')
    assert all(torch.equal(inputs[k],other[k]) for k in inputs)
