import unittest
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from experiments.region_frame_transform import frames,transform
from models.round69_compact_relation import descriptors


class FrameTransformChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_frame_identity_is_order_independent(self):
        ids=np.array(['old-a','old-b','old-c']);r,t=frames(ids)
        other_r,other_t=frames(ids[[2,0,1]])
        np.testing.assert_array_equal(r[[2,0,1]],other_r)
        np.testing.assert_array_equal(t[[2,0,1]],other_t)
        np.testing.assert_allclose(r.swapaxes(-1,-2)@r,np.broadcast_to(np.eye(3),r.shape),atol=1e-14)
        np.testing.assert_allclose(np.linalg.det(r),1,atol=1e-14)
        self.assertTrue(np.all(np.abs(t)<=.5))

    def test_direct_patch_moments_features_and_relations(self):
        rng=np.random.default_rng(99702)
        patches=rng.normal(size=(2,2,7,32,3));centers=patches.mean(-2)
        delta=patches-centers[...,None,:]
        cov=np.einsum('bvnki,bvnkj->bvnij',delta,delta)/32+np.eye(3)*.0004
        r,t=frames(np.array(['fit-a','fit-b']))
        a,b=transform(centers,cov,r,t)
        direct=np.einsum('bvij,bvnkj->bvnki',r,patches)+t[:,:,None,None,:]
        new_center=direct.mean(-2);new_delta=direct-new_center[...,None,:]
        new_cov=np.einsum('bvnki,bvnkj->bvnij',new_delta,new_delta)/32+np.eye(3)*.0004
        np.testing.assert_allclose(a,new_center,atol=1e-14)
        np.testing.assert_allclose(b,new_cov,atol=1e-14)
        np.testing.assert_allclose(np.linalg.eigvalsh(b),np.linalg.eigvalsh(cov),atol=1e-14)
        np.testing.assert_allclose(np.sort(np.linalg.norm(new_delta,axis=-1)),np.sort(np.linalg.norm(delta,axis=-1)),atol=1e-14)
        before=descriptors('relative5',torch.from_numpy(centers),torch.from_numpy(cov))
        after=descriptors('relative5',torch.from_numpy(a),torch.from_numpy(b))
        torch.testing.assert_close(after,before,rtol=1e-11,atol=1e-11)

    def test_known_two_frame_pose_composition(self):
        rng=np.random.default_rng(99703);points=rng.normal(size=(13,3))
        rotation=Rotation.from_rotvec(rng.normal(size=3)).as_matrix();translation=rng.normal(size=3)
        target=points@rotation.T+translation
        r,t=frames(np.array(['pair']))
        a=points@r[0,0].T+t[0,0];b=target@r[0,1].T+t[0,1]
        composed_r=r[0,1]@rotation@r[0,0].T
        composed_t=r[0,1]@translation+t[0,1]-composed_r@t[0,0]
        np.testing.assert_allclose(a@composed_r.T+composed_t,b,atol=1e-13)
        np.testing.assert_allclose((a-t[0,0])@r[0,0],points,atol=1e-13)


if __name__=='__main__':unittest.main()
