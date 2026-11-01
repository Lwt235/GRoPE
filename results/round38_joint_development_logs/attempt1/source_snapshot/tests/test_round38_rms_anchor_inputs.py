"""Analytic geometry and content boundary tests, using synthetic arrays only."""
import unittest
from unittest.mock import patch
import numpy as np
import torch
from experiments import round38_rms_anchor_inputs as I


def observations():
    k=np.array([[100.,0.,96.],[0.,100.,72.],[0.,0.,1.]])
    y,x=np.mgrid[:144,:192]
    points=np.stack(((x-96.)*.02,(y-72.)*.02,np.full_like(x,2.,dtype=float)),axis=-1)
    covariance=np.zeros((144*192,3,3));covariance[:,0,0]=covariance[:,1,1]=.02**2/12
    return dict(q_points3=points.reshape(-1,3),q_pixel_cov3=covariance,
        q_K=k.copy(),k_K=k.copy(),q_to_k_transform=np.eye(4),q_predicted_depth=np.full((144,192),2.))


def cache():
    rng=np.random.default_rng(38)
    cov=np.tile(np.diag([.25,.5]),(48,1,1))
    h2=torch.tensor(.25**2,dtype=torch.float64);diag=torch.from_numpy(cov).diagonal(dim1=-2,dim2=-1)
    scale=torch.maximum(diag.amax(-1),h2);h=h2/scale
    tau=(10*(h/(h+(diag/scale[:,None]).sum(-1)))).float().numpy()
    return {'g24__a':np.zeros((48,2)),'g24__anchor_cov':cov,'g24__tau32':tau,
        'g24__q_features':rng.normal(size=(48,128)).astype(np.float32),
        'g8__k_features':rng.normal(size=(432,128)).astype(np.float32),'g8__bounds_px':I.native_bounds(8)}


class InputTests(unittest.TestCase):
    def setUp(self):
        self.guard=patch.object(np,'load',side_effect=AssertionError('no archive access in input tests'))
        self.guard.start();self.addCleanup(self.guard.stop)

    def test_constant_plane_exact_support_moments(self):
        b=I.prepare_geometry(observations())
        self.assertEqual(b['source_mean'].shape,(48,3))
        np.testing.assert_allclose(b['source_mean'][0],[-1.69,-1.21,2.],rtol=0,atol=2e-14)
        expected=np.diag([24**2/12*.02**2,24**2/12*.02**2,0.])
        np.testing.assert_allclose(b['source_covariance'],np.broadcast_to(expected,(48,3,3)),rtol=2e-13,atol=8*np.finfo(np.float64).eps*np.max(expected))
        np.testing.assert_allclose(b['source_covariance_direct'],b['source_covariance'],rtol=2e-13,atol=8*np.finfo(np.float64).eps*np.max(expected))
        np.testing.assert_allclose(b['anchor_delta'],0.,rtol=0,atol=5e-13)
        np.testing.assert_allclose(b['anchor_target'][0],[-1.69,-1.21,2.],atol=1e-15)

    def test_rank_deficiency_no_jitter(self):
        for factor in (np.zeros((3,3)),np.diag([2.,0.,0.]),np.diag([2.,3.,0.])):
            mean=np.zeros(3);anchor=np.array([0.,0.,4.])
            f=I.augment_factor(mean,factor,anchor)
            np.testing.assert_allclose(f@f.T,factor@factor.T+np.diag([0.,0.,16.]),rtol=0,atol=0)
        self.assertTrue(np.array_equal(I.augment_factor(np.zeros(3),np.zeros((3,3)),np.zeros(3)),np.zeros((3,3))))

    def test_augmentation_translation_and_factor_rotation(self):
        rng=np.random.default_rng(20);f=rng.normal(size=(3,3));m=rng.normal(size=3);a=rng.normal(size=3)
        rotation=np.linalg.qr(rng.normal(size=(3,3)))[0]
        base=I.augment_factor(m,f,a);rot=I.augment_factor(m,f@rotation,a)
        moved=I.augment_factor(m+np.array([3.,4.,5.]),f,a+np.array([3.,4.,5.]))
        np.testing.assert_allclose(base@base.T,rot@rot.T,rtol=2e-14,atol=2e-14)
        np.testing.assert_allclose(base@base.T,moved@moved.T,rtol=2e-14,atol=2e-14)

    def test_anchor_retains_negative_target_depth(self):
        o=observations();o['q_to_k_transform'][2,3]=-3.
        b=I.prepare_geometry(o)
        np.testing.assert_array_equal(b['anchor_target'][:,2],-np.ones(48))
        self.assertFalse(b['anchor_strict_front'].any())
        self.assertEqual(len(b['anchored_factor']),48)

    def test_anchor_never_uses_screen_clamp(self):
        o=observations();o['q_to_k_transform'][0,3]=1e6
        b=I.prepare_geometry(o)
        self.assertTrue((b['anchor_homogeneous'][:,0]>1e6).all())

    def test_center_depth_cleanup_then_bilinear_mean(self):
        o=observations();o['q_predicted_depth'][11:13,11:13]=[[np.nan,np.inf],[-np.inf,-7.]]
        a=I.center_anchor(o,I.native_bounds(24))
        self.assertEqual(a['anchor_depth'][0],(10.+20.+.1+.1)/4)
        self.assertEqual(a['anchor_depth'][1],2.)
        self.assertTrue(np.isnan(o['q_predicted_depth'][11,11]))

    def test_nonrigid_supplied_camera_is_preserved(self):
        o=observations();o['q_to_k_transform'][0,0]=1.0001
        b=I.prepare_geometry(o)
        self.assertGreater(b['camera_rotation_diagnostics'][0],0)
        np.testing.assert_allclose(b['target_mean'][:,0],b['source_mean'][:,0]*1.0001)

    def test_homogeneous_mapping_precedes_projection(self):
        o=observations();o['q_to_k_transform'][2,3]=.7
        b=I.prepare_geometry(o);a=b['homogeneous_transform']
        np.testing.assert_allclose(b['anchor_homogeneous'],b['anchor_target']@a.T,rtol=1e-14,atol=1e-13)
        np.testing.assert_allclose(b['anchor_homogeneous'][:,2],2.7)
        for f,q in zip(b['anchored_factor'],b['anchor_covariance']):
            np.testing.assert_allclose(f@f.T,q,rtol=5e-13,atol=5e-13)

    def test_pixel_moment_disagreement_is_retained_for_auditor(self):
        o=observations();o['q_pixel_cov3']*=2
        b=I.prepare_geometry(o)
        self.assertGreater(np.max(abs(b['source_covariance_direct']-b['source_covariance'])),1e-5)

    def test_full_matrix_float32_bias_exact(self):
        c=cache();b=I.prepare_content(c)
        expected=torch.from_numpy(c['g24__q_features'])@torch.from_numpy(c['g8__k_features']).T
        bias=torch.from_numpy(c['g24__tau32'])[:,None]*expected
        np.testing.assert_array_equal(b['content32'],expected.numpy())
        np.testing.assert_array_equal(b['bias32'],bias.numpy())
        np.testing.assert_array_equal(b['content_bias64'],bias.double().numpy())
        self.assertEqual(b['content32'].dtype,np.float32)

    def test_tau_corruption_rejected(self):
        c=cache();c['g24__tau32'][0]=np.nextafter(c['g24__tau32'][0],np.float32(np.inf))
        with self.assertRaisesRegex(ValueError,'temperature'):I.prepare_content(c)

    def test_native_order_corruption_rejected(self):
        c=cache();c['g8__bounds_px'][[0,1]]=c['g8__bounds_px'][[1,0]]
        with self.assertRaisesRegex(ValueError,'identity'):I.prepare_content(c)

    def test_forbidden_observation_or_cache_field_rejected(self):
        o=observations();o['gt_xy']=np.zeros((1,2))
        with self.assertRaisesRegex(ValueError,'allowlist'):I.prepare_geometry(o)
        c=cache();c['sensor_depth']=np.zeros(1)
        with self.assertRaisesRegex(ValueError,'allowlist'):I.prepare_content(c)

    def test_model_inputs_are_four_owned_buffers(self):
        b=I.prepare_inputs(observations(),cache());v=I.model_inputs(b,0)
        self.assertEqual(set(v),{'mu','factor','bounds','content_bias'})
        np.testing.assert_array_equal(v['mu'],b['anchor_homogeneous'][0])
        v['mu'][0]=999
        self.assertNotEqual(b['anchor_homogeneous'][0,0],999)
        for invalid in (-1,48,True,np.int64(0)):
            with self.assertRaises(ValueError):I.model_inputs(b,invalid)

    def test_invalid_intrinsics_and_geometry_rejected(self):
        o=observations();o['q_K'][2,2]=2
        with self.assertRaisesRegex(ValueError,'intrinsics'):I.prepare_geometry(o)
        o=observations();o['q_points3'][0,0]=np.nan
        with self.assertRaisesRegex(ValueError,'finite'):I.prepare_geometry(o)

    def test_world_unit_scaling_of_second_moment(self):
        o=observations();b=I.prepare_geometry(o)
        scaled={k:v.copy() for k,v in o.items()}
        scaled['q_points3']*=3;scaled['q_pixel_cov3']*=9;scaled['q_predicted_depth']*=3
        s=I.prepare_geometry(scaled)
        np.testing.assert_allclose(s['anchor_homogeneous'],b['anchor_homogeneous']*3,rtol=1e-14,atol=1e-13)
        np.testing.assert_allclose(s['anchor_covariance'],b['anchor_covariance']*9,rtol=1e-12,atol=1e-12)


if __name__=='__main__':unittest.main()
