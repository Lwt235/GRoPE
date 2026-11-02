"""Bounded scientific tests: direct formula, controls, interventions, continuation."""
import hashlib
import io
import unittest
import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import softmax
from models.round48_point_transformer import make_model as old_model
from models.round56_native_joint import make_model, NEW_ROLES, intervene_coefficients
from experiments.round48_audit import pair_energy, phase, rotate
from experiments.round48_train import shared_digest
from experiments.round56_optimization import epoch_batches, snapshot, restore


class NativeJointTests(unittest.TestCase):
    def test_nonzero_formula_lift_and_gradients(self):
        for role in NEW_ROLES:
            torch.manual_seed(561)
            m = make_model(role).double()
            with torch.no_grad():
                for p in m.position_field.parameters():
                    p.normal_(0,.08)
            q,k = [F.normalize(torch.randn(1,2,7,48,dtype=torch.float64),dim=-1) for _ in range(2)]
            features = torch.randn(1,7,96,dtype=torch.float64)
            positions = torch.randn(1,7,3,dtype=torch.float64)*4
            scores = m.scores(m.layers[0],q,k,features,positions)
            state=m.state_dict()
            a=phase('rope',state,positions[0].numpy())
            qr,kr=rotate(q[0].numpy(),a),rotate(k[0].numpy(),a)
            temperature=np.exp(min(float(m.layers[0].log_temperature),np.log(60)))
            direct=temperature*(qr@kr.swapaxes(-1,-2))+pair_energy(
                role.removeprefix('joint_'),state,features[0].numpy(),positions[0].numpy(),positions[0].numpy())
            np.testing.assert_allclose(scores.softmax(-1)[0].detach().numpy(),softmax(direct,axis=-1),atol=2e-12)
            pq,pk=m.position_field.vectors(features,positions,positions)
            qcat=np.concatenate((np.sqrt(temperature)*qr,pq[0].detach().numpy()),-1)
            kcat=np.concatenate((np.sqrt(temperature)*kr,np.broadcast_to(pk[0].detach().numpy(),(2,7,9))),-1)
            np.testing.assert_allclose(scores[0].detach().numpy(),qcat@kcat.swapaxes(-1,-2),atol=2e-12)
            # Relative phase/regional preference is invariant to a common position translation
            # when content/predicted region is fixed; this is not full-model SO(3) invariance.
            shifted=m.scores(m.layers[0],q,k,features,positions+17)
            torch.testing.assert_close(scores.softmax(-1),shifted.softmax(-1),atol=2e-12,rtol=2e-12)
            if role != 'joint_fixed':
                weight=m.position_field.predictor.weight
                value=scores.softmax(-1)[0,0,0,2]
                derivative=torch.autograd.grad(value,weight)[0][5,3].item()
                original=weight[5,3].item(); outputs=[]
                for change in (1e-5,-1e-5):
                    with torch.no_grad(): weight[5,3]=original+change
                    outputs.append(m.scores(m.layers[0],q,k,features,positions).softmax(-1)[0,0,0,2].item())
                with torch.no_grad(): weight[5,3]=original
                self.assertAlmostEqual(derivative,(outputs[0]-outputs[1])/2e-5,places=7)

    def test_shared_initialization_and_initial_control_functions(self):
        points=torch.randn(2,17,3)
        values=[]
        for role in NEW_ROLES:
            torch.manual_seed(11);m=make_model(role).eval()
            torch.manual_seed(11);old=old_model('rope').eval()
            self.assertEqual(shared_digest(m),shared_digest(old))
            with torch.no_grad():values.append(m(points))
        for value in values[1:]:torch.testing.assert_close(value,values[0],atol=0,rtol=0)

    def test_intervention_preserves_center_and_declared_shape_properties(self):
        torch.manual_seed(562)
        a=torch.randn(2,5,3,3,dtype=torch.float64)
        p=a@a.transpose(-1,-2)+torch.eye(3,dtype=torch.float64)
        mean=torch.randn(2,5,3,dtype=torch.float64);b=(p@mean.unsqueeze(-1)).squeeze(-1)
        for mode in ('isotropic_trace','rotate_z90'):
            changed,linear=intervene_coefficients(p,b,mode)
            torch.testing.assert_close(torch.linalg.solve(changed,linear.unsqueeze(-1)).squeeze(-1),mean)
            original_cov,changed_cov=torch.linalg.inv(p),torch.linalg.inv(changed)
            if mode=='isotropic_trace':
                torch.testing.assert_close(original_cov.diagonal(dim1=-2,dim2=-1).sum(-1),
                                           changed_cov.diagonal(dim1=-2,dim2=-1).sum(-1))
                self.assertTrue(torch.allclose(torch.linalg.eigvalsh(changed_cov).std(-1),torch.zeros(2,5,dtype=torch.float64)))
            else:torch.testing.assert_close(torch.linalg.eigvalsh(original_cov),torch.linalg.eigvalsh(changed_cov))

    def test_original_sampler_stream_and_optimizer_continuation(self):
        r1=np.random.default_rng(48011);r2=np.random.default_rng(48011)
        h1,h2=hashlib.sha256(),hashlib.sha256()
        for _ in range(2):
            actual=list(epoch_batches(r1,7,h1,batch_size=3,points=13))
            order=r2.permutation(7);h2.update(order.astype('<i8').tobytes());expected=[]
            for begin in range(0,7,3):
                ids=order[begin:begin+3]
                subset=np.stack([r2.choice(2048,13,replace=False) for _ in ids])
                angle=r2.uniform(0,2*np.pi,len(ids)).astype(np.float32)
                jitter=r2.normal(0,.01,(len(ids),13,3)).clip(-.05,.05).astype(np.float32)
                for value in (ids.astype('<i8'),subset.astype('<i8'),angle,jitter):h2.update(value.tobytes())
                expected.append((ids,subset,angle,jitter))
            for row,ref in zip(actual,expected):
                for x,y in zip(row,ref):np.testing.assert_array_equal(x,y)
        self.assertEqual(h1.hexdigest(),h2.hexdigest())
        self.assertEqual(r1.bit_generator.state,r2.bit_generator.state)
        torch.manual_seed(563);m=make_model('joint_conditional');opt=torch.optim.AdamW(m.parameters(),lr=.001)
        x=torch.randn(2,11,3);y=torch.tensor([1,7])
        def step(model,optimizer,number):
            torch.manual_seed(1100000000+number);model.train();optimizer.zero_grad(set_to_none=True)
            F.cross_entropy(model(x),y,label_smoothing=.1).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        step(m,opt,0)
        blob=io.BytesIO();torch.save(snapshot(m,opt),blob);blob.seek(0)
        state=torch.load(blob,map_location='cpu',weights_only=False)
        restored=make_model('joint_conditional');opt2=torch.optim.AdamW(restored.parameters(),lr=.001)
        restore(restored,opt2,state)
        step(m,opt,1);step(restored,opt2,1)
        for a,b in zip(m.state_dict().values(),restored.state_dict().values()):torch.testing.assert_close(a,b,atol=0,rtol=0)
        self.assertEqual(opt.state_dict()['param_groups'],opt2.state_dict()['param_groups'])


if __name__=='__main__':unittest.main()
