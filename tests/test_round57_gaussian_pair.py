"""Bounded checks of Gaussian-pair algebra, controls and actual optimization."""
import io
import unittest
import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import softmax
from models.round57_gaussian_pair import make_model, NEW_ROLES, gaussian_vectors
from models.round56_native_joint import make_model as reference_model
from experiments.round48_train import shared_digest
from experiments.round48_audit import phase, rotate
from experiments.round57_audit import direct_pair_energy
from experiments.round56_optimization import snapshot, restore


class GaussianPairTests(unittest.TestCase):
    def test_nonzero_direct_formula_and_finite_difference(self):
        for role in NEW_ROLES:
            torch.manual_seed(571)
            model=make_model(role).double().eval()
            with torch.no_grad():
                for parameter in model.position_field.parameters():parameter.normal_(0,.02)
            q,k=[F.normalize(torch.randn(1,2,7,48,dtype=torch.float64),dim=-1) for _ in range(2)]
            features=torch.randn(1,7,96,dtype=torch.float64)
            positions=torch.randn(1,7,3,dtype=torch.float64)*4
            scores=model.scores(model.layers[0],q,k,features,positions)
            state=model.state_dict();angle=phase('rope',state,positions[0].numpy())
            qr,kr=rotate(q[0].numpy(),angle),rotate(k[0].numpy(),angle)
            temperature=np.exp(min(float(model.layers[0].log_temperature),np.log(60)))
            geometry=direct_pair_energy(role,state,features[0].numpy(),features[0].numpy(),
                                       positions[0].numpy(),positions[0].numpy())
            direct=temperature*(qr@kr.swapaxes(-1,-2))+geometry
            np.testing.assert_allclose(scores[0].detach().numpy(),direct,atol=2e-11,rtol=2e-11)
            np.testing.assert_allclose(scores.softmax(-1)[0].detach().numpy(),softmax(direct,axis=-1),atol=2e-12)
            pq,pk,row=model.position_field.vectors(features,positions)
            self.assertEqual(pq.shape[-1],19)
            joined_q=np.concatenate((np.sqrt(temperature)*qr,pq[0].detach().numpy()),-1)
            joined_k=np.concatenate((np.sqrt(temperature)*kr,pk[0].detach().numpy()),-1)
            np.testing.assert_allclose(softmax(joined_q@joined_k.swapaxes(-1,-2),axis=-1),
                                       softmax(direct,axis=-1),atol=2e-12)
            weight=model.position_field.predictor.weight
            gradient=torch.autograd.grad(scores.softmax(-1)[0,0,0,2],weight)[0]
            entries=[(0,1)] if role=='pair_center' else [(0,1),(3,2),(6,3)]
            for i,j in entries:
                initial=weight[i,j].item();values=[]
                for change in (1e-5,-1e-5):
                    with torch.no_grad():weight[i,j]=initial+change
                    values.append(model.scores(model.layers[0],q,k,features,positions).softmax(-1)[0,0,0,2].item())
                with torch.no_grad():weight[i,j]=initial
                self.assertAlmostEqual(gradient[i,j].item(),(values[0]-values[1])/2e-5,places=7)

    def test_geometric_relation_properties_at_fixed_parameters(self):
        rng=np.random.default_rng(572)
        center=rng.normal(size=(6,3));factor=rng.normal(size=(6,3,3))
        covariance=factor@factor.swapaxes(-1,-2)+np.eye(3)*.4
        covariance[0]=np.eye(3);covariance[1]=np.eye(3)
        def lifted(c,s,include=True):
            precision=np.linalg.inv(s)
            q,k,row=gaussian_vectors(torch.from_numpy(c),torch.from_numpy(precision),
                                     torch.from_numpy(s) if include else None)
            return (q@k.T-row).numpy()
        actual=lifted(center,covariance);precision=np.linalg.inv(covariance)
        direct=np.empty((6,6))
        for i in range(6):
            for j in range(6):
                delta=center[i]-center[j]
                direct[i,j]=-.25*(delta@(precision[i]+precision[j])@delta+
                    np.trace(precision[i]@covariance[j]+precision[j]@covariance[i])-6)
        np.testing.assert_allclose(actual,direct,atol=2e-12)
        np.testing.assert_allclose(actual,actual.T,atol=2e-12)
        np.testing.assert_allclose(np.diag(actual),0,atol=2e-12)
        self.assertLess(actual.max(),2e-12)
        self.assertLess((actual-lifted(center,covariance,False)).max(),2e-12)
        transform=np.array([[1.2,.3,-.1],[.1,.8,.2],[0,-.2,1.1]])
        shifted=center@transform.T+np.array([3.,-5.,2.])
        changed=transform@covariance@transform.T
        np.testing.assert_allclose(lifted(shifted,changed),actual,atol=2e-11)
        equal=np.broadcast_to(np.eye(3)*16,(6,3,3)).copy()
        distance=-np.square(center[:,None]-center[None]).sum(-1)/32
        np.testing.assert_allclose(lifted(center,equal),distance,atol=2e-12)
        np.testing.assert_allclose(lifted(center,equal,False),distance,atol=2e-12)

    def test_initial_controls_gradient_identity_and_diagnostic(self):
        torch.manual_seed(573);points=torch.randn(2,17,3,dtype=torch.float64);labels=torch.tensor([1,7])
        models=[];outputs=[]
        for role in NEW_ROLES:
            torch.manual_seed(11);model=make_model(role).double().eval()
            torch.manual_seed(11);reference=reference_model('joint_fixed').double().eval()
            self.assertEqual(shared_digest(model),shared_digest(reference))
            expected=582 if role=='pair_center' else 1746
            self.assertEqual(sum(p.numel() for p in model.position_field.parameters()),expected)
            outputs.append(model(points));models.append(model)
            torch.testing.assert_close(outputs[-1],reference(points),atol=2e-11,rtol=2e-11)
        for output in outputs[1:]:torch.testing.assert_close(output,outputs[0],atol=2e-11,rtol=2e-11)
        for model,output in zip(models[:2],outputs[:2]):F.cross_entropy(output,labels).backward()
        for (name,a),(other,b) in zip(models[0].named_parameters(),models[1].named_parameters()):
            self.assertEqual(name,other)
            torch.testing.assert_close(a.grad,b.grad,atol=2e-10,rtol=2e-9)
        mean_bias=models[0].position_field.predictor.bias.grad.reshape(2,9)[:,:3]
        self.assertLess(float(mean_bias.abs().max()),2e-12)
        full=make_model('pair_full','shape_off').double().eval()
        with torch.no_grad():
            for parameter in full.position_field.parameters():parameter.normal_(0,.02)
        distance=make_model('pair_distance').double().eval();distance.load_state_dict(full.state_dict())
        torch.testing.assert_close(full(points),distance(points),atol=0,rtol=0)
        with self.assertRaises(ValueError):full.train()(points)

    def test_optimizer_serialization_continues_actual_pair_model(self):
        torch.manual_seed(574);model=make_model('pair_full')
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
        points=torch.randn(2,11,3);labels=torch.tensor([1,7])
        def step(m,opt,number):
            torch.manual_seed(1100000000+number);m.train();opt.zero_grad(set_to_none=True)
            F.cross_entropy(m(points),labels,label_smoothing=.1).backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);opt.step()
        step(model,optimizer,0)
        blob=io.BytesIO();torch.save(snapshot(model,optimizer),blob);blob.seek(0)
        saved=torch.load(blob,map_location='cpu',weights_only=False)
        resumed=make_model('pair_full');optimizer2=torch.optim.AdamW(resumed.parameters(),lr=.001)
        restore(resumed,optimizer2,saved);step(model,optimizer,1);step(resumed,optimizer2,1)
        for a,b in zip(model.state_dict().values(),resumed.state_dict().values()):
            torch.testing.assert_close(a,b,atol=0,rtol=0)
        self.assertEqual(optimizer.state_dict()['param_groups'],optimizer2.state_dict()['param_groups'])


if __name__=='__main__':unittest.main()
