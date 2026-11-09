import unittest
import numpy as np
import torch

from experiments.native_region_moments_diagnostic import mixture_moments


class MixtureMomentTests(unittest.TestCase):
    def setUp(self):
        self.centers=torch.tensor([[1.,2.,-1.],[-2.,.5,3.]],dtype=torch.float64)
        a=torch.tensor([[[1.,0.,0.],[.3,2.,0.],[-.2,.4,.5]],
                        [[.6,0.,0.],[.4,.8,0.],[.2,-.3,1.2]]],dtype=torch.float64)
        self.covariance=a@a.transpose(-1,-2)
        self.weights=torch.tensor([.3,.7],dtype=torch.float64)

    def test_single_component_identity(self):
        mean,within,between,total=mixture_moments(self.centers[:1],self.covariance[:1])
        torch.testing.assert_close(mean,self.centers[0])
        torch.testing.assert_close(total,self.covariance[0])
        self.assertEqual(torch.count_nonzero(between),0)

    def test_independent_exact_sigma_point_moments(self):
        c,s,w=self.centers.numpy(),self.covariance.numpy(),self.weights.numpy()
        points=[];weights=[]
        for i in range(2):
            axes=np.sqrt(3)*np.linalg.cholesky(s[i]).T
            points.extend(c[i]+np.concatenate((axes,-axes)))
            weights.extend([w[i]/6]*6)
        points,weights=np.asarray(points),np.asarray(weights)
        mean=np.average(points,axis=0,weights=weights)
        centered=points-mean
        expected=(centered*weights[:,None]).T@centered
        actual,_,_,total=mixture_moments(self.centers,self.covariance,self.weights)
        np.testing.assert_allclose(actual.numpy(),mean,rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(total.numpy(),expected,rtol=1e-12,atol=1e-12)

    def test_rigid_equivariance(self):
        angle=.73
        rotation=torch.tensor([[np.cos(angle),-np.sin(angle),0.],[np.sin(angle),np.cos(angle),0.],[0.,0.,1.]],dtype=torch.float64)
        translation=torch.tensor([2.,-4.,.3],dtype=torch.float64)
        mean,_,_,total=mixture_moments(self.centers,self.covariance,self.weights)
        moved,_,_,moved_total=mixture_moments(self.centers@rotation.T+translation,
                                              rotation@self.covariance@rotation.T,self.weights)
        torch.testing.assert_close(moved,mean@rotation.T+translation)
        torch.testing.assert_close(moved_total,rotation@total@rotation.T)

    def test_mass_preserving_component_split(self):
        mean,_,_,total=mixture_moments(self.centers,self.covariance,self.weights)
        indices=torch.tensor([0,0,1])
        split_weights=torch.tensor([.15,.15,.7],dtype=torch.float64)
        split_mean,_,_,split_total=mixture_moments(self.centers[indices],self.covariance[indices],split_weights)
        torch.testing.assert_close(split_mean,mean)
        torch.testing.assert_close(split_total,total)


if __name__=='__main__':unittest.main()
