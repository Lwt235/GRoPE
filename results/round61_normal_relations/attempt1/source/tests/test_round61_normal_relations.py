"""Bounded independent geometry and learning checks for new relations only."""
import unittest
import numpy as np
import torch
from models.round58_support_pe import SupportMatcher
from models.round61_normal_relations import make_model,pair_inputs,spectral_geometry,NEW_ROLES
from experiments.round58_run import shared_digest
from experiments.round58_data import loss_function,random_rotation


def independent(centers,covariances,role):
    spectra=[];normals=[];axial=[]
    for s in covariances:
        w,v=np.linalg.eigh(s);chosen=w-w[0]<=1e-6*w[-1]
        pi=sum(np.outer(v[:,k],v[:,k]) for k in range(3) if chosen[k]);count=int(chosen.sum())
        normals.append(pi/count);spectra.append(.5*np.log(w/.04))
        if count==3:axial.append(np.eye(3)*np.mean(w))
        else:axial.append(np.mean(w[chosen])*pi+np.mean(w[~chosen])*(np.eye(3)-pi))
    output=[]
    for i,a in enumerate(centers):
        row=[]
        for j,b in enumerate(centers):
            delta=a-b;d=np.linalg.norm(delta);u=delta/d if d else np.zeros(3)
            if role=='normal_ppf_mlp':
                relation=[u@normals[i]@u,u@normals[j]@u,np.trace(normals[i]@normals[j])]
            else:
                pi,pj=np.linalg.inv(axial[i]),np.linalg.inv(axial[j])
                relation=np.log1p([delta@pi@delta,delta@pj@delta,np.trace(pi@axial[j]),np.trace(pj@axial[i])])
            row.append(np.r_[d/.2,spectra[i],spectra[j],relation])
        output.append(row)
    return np.asarray(output)


class NormalRelationTests(unittest.TestCase):
    def inputs(self):
        rng=np.random.default_rng(6101)
        f=torch.from_numpy(rng.normal(size=(1,2,6,11)))
        c=torch.from_numpy(rng.normal(scale=.2,size=(1,2,6,3)))
        matrices=[]
        for view in range(2):
            one=[]
            for i in range(6):
                r=random_rotation(rng)
                values=np.array(([1,2,5],[1,2,2],[1,1,5],[1,1,1],[1,1+1e-7,5],[1,1.1,3])[i])*.02
                one.append(r@np.diag(values)@r.T)
            matrices.append(one)
        return f,c,torch.from_numpy(np.array(matrices)[None])

    def test_independent_relations_and_degenerate_projectors(self):
        f,c,s=self.inputs()
        for role in NEW_ROLES:
            expected=np.stack([independent(c[0,v].numpy(),s[0,v].numpy(),role) for v in range(2)])[None]
            np.testing.assert_allclose(pair_inputs(role,c,s).numpy(),expected,rtol=5e-12,atol=5e-12)
        w,n,a,k=spectral_geometry(s)
        self.assertEqual(k[0,0].tolist(),[1,1,2,3,2,1])
        torch.testing.assert_close(n.diagonal(dim1=-2,dim2=-1).sum(-1),torch.ones_like(k,dtype=torch.float64))
        torch.testing.assert_close(a[0,:,3],torch.eye(3,dtype=torch.float64).expand(2,3,3)*.02,rtol=1e-12,atol=1e-12)
        self.assertTrue((torch.linalg.eigvalsh(a)>0).all())

    def test_axisymmetric_equivalence_and_tangent_intervention(self):
        f,c,s=self.inputs();_,_,axial,_=spectral_geometry(s)
        torch.manual_seed(11);full=SupportMatcher('invariant_mlp').double()
        torch.manual_seed(11);control=make_model('axial_mlp').double()
        torch.testing.assert_close(full.position(c,axial),control.position(c,axial),rtol=1e-11,atol=1e-11)
        # Rotate only a tangent eigenframe of an anisotropic region; normals/spectra stay fixed.
        r=torch.tensor([[1.,0,0],[0,0,-1],[0,1,0]],dtype=torch.float64)
        fixed=torch.diag(torch.tensor([.02,.04,.1],dtype=torch.float64)).expand(1,2,6,3,3).clone()
        moved=fixed.clone();moved[:,:,0]=r@fixed[:,:,0]@r.T
        for role in NEW_ROLES:
            torch.testing.assert_close(pair_inputs(role,c,fixed),pair_inputs(role,c,moved),rtol=1e-12,atol=1e-12)
        self.assertGreater(float((full.position(c,fixed)-full.position(c,moved)).abs().max()),1e-5)

    def test_geometry_and_common_initialization(self):
        f,c,s=self.inputs();rng=np.random.default_rng(6102)
        r=torch.from_numpy(np.stack([random_rotation(rng) for _ in range(2)]))
        moved=c@r.transpose(-1,-2)+c.new_tensor([.7,-.3,.2])
        rotated=r[:,None]@s@r[:,None].transpose(-1,-2)
        for seed in (11,22,33):
            torch.manual_seed(seed);expected=shared_digest(SupportMatcher('invariant_mlp'))
            for role in NEW_ROLES:
                torch.manual_seed(seed);model=make_model(role)
                self.assertEqual(shared_digest(model),expected)
                self.assertEqual(sum(p.numel() for p in model.position.parameters()),10272 if role=='normal_ppf_mlp' else 10368)
        for role in NEW_ROLES:
            torch.manual_seed(11);model=make_model(role).double().eval()
            output=model(f,c,s)
            torch.testing.assert_close(output,model(f,moved,rotated),rtol=2e-10,atol=2e-10)
            torch.testing.assert_close(output,model(f.flip(-2),c.flip(-2),s.flip(-3)).flip(-2,-1),rtol=2e-10,atol=2e-10)

    def test_actual_learning_gradient_finite_difference(self):
        f,c,s=self.inputs();truth=torch.from_numpy(np.random.default_rng(6103).uniform(size=(1,6,6)))
        for role in NEW_ROLES:
            torch.manual_seed(11);model=make_model(role).double()
            loss=loss_function(model(f,c,s),truth);loss.backward()
            for p in model.position.parameters():
                self.assertTrue(torch.isfinite(p.grad).all());self.assertGreater(float(p.grad.norm()),0)
            p=model.position.mlp[-1].weight;analytic=float(p.grad[1,2]);epsilon=1e-6
            with torch.no_grad():
                initial=float(p[1,2]);p[1,2]=initial+epsilon;plus=float(loss_function(model(f,c,s),truth))
                p[1,2]=initial-epsilon;minus=float(loss_function(model(f,c,s),truth));p[1,2]=initial
            self.assertAlmostEqual(analytic,(plus-minus)/(2*epsilon),delta=2e-7)

    def test_scoped_adapter_restoration(self):
        from experiments import round58_run as original
        from experiments.round61_run import fitting_adapter
        from experiments.round61_common import OUT,CONFIGS
        names=('OUT','CONFIG','ROLES','SupportMatcher','checked','load_cache')
        before={n:getattr(original,n) for n in names};function=original.train
        for domain in ('synthetic','real'):
            with self.assertRaisesRegex(RuntimeError,'restore'):
                with fitting_adapter(domain,11,factory=make_model):
                    self.assertEqual(original.OUT,OUT/domain);self.assertIs(original.CONFIG,CONFIGS[domain])
                    self.assertIs(original.train,function);raise RuntimeError('restore')
            for n in names:self.assertIs(getattr(original,n),before[n])


if __name__=='__main__':
    torch.set_num_threads(4);unittest.main()
