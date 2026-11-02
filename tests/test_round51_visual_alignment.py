import unittest
import numpy as np
import torch
from experiments.round51_data import camera_fields, pixels
from models.round51_visual_alignment import VisualAlignmentMatcher, ROLES


class VisualTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(4)

    def sample(self):
        torch.manual_seed(815)
        d = {k:torch.randn(2, 9, 128) for k in ('q', 'k')}
        d.update({k:torch.randn(2, 9, 18) for k in ('qgeom', 'kgeom')})
        d.update({k:torch.randn(2, 9, 2) for k in ('qpos', 'kpos')})
        d.update({k:torch.diag(torch.tensor([.8, 1.2, 1.])).expand(2, 3, 3) for k in ('qK', 'kK')})
        return d

    def test_camera_chart(self):
        K = np.array([[170., 0, 94.7], [0, 172., 73.1], [0, 0, 1.]])
        q = dict(K=K, c2w=np.eye(4)); relative = np.eye(4); relative[:3, 3] = [.3, -.2, .1]
        fields = camera_fields(q, relative)
        rays = np.column_stack((pixels(), np.ones(432)))@np.linalg.inv(K).T
        projected = rays@fields['K'].T
        np.testing.assert_allclose(projected[:, :2]*[192, 144]/8, fields['pos'], atol=2e-6)
        np.testing.assert_allclose(fields['geom'][0, 9:12], relative[:3, 3]/3, atol=1e-8)
        world = np.eye(4); theta=.2
        world[:3, :3] = [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]]
        world[:3, 3] = [1, 2, 3]
        np.testing.assert_allclose(np.linalg.inv(world@q['c2w'])@(world@relative), relative, atol=1e-12)

    def test_shared_input_and_permutation(self):
        data = self.sample(); common = None; permutation = torch.randperm(9)
        for role in ROLES:
            torch.manual_seed(11); model = VisualAlignmentMatcher(role).eval()
            state = {k:v for k,v in model.state_dict().items() if not k.startswith('field.')}
            if common is None: common = state
            else:
                self.assertEqual(common.keys(), state.keys())
                for key in state: self.assertTrue(torch.equal(common[key], state[key]), (role,key))
            pred = model(data)
            forbidden = dict(data, gt_valid=torch.randn(2,9), pixcontinuous=torch.randn(2,9,2),
                predicted_depth=torch.randn(2,9), mean3=torch.randn(2,9,3))
            torch.testing.assert_close(pred, model(forbidden), rtol=0, atol=0)
            perm = {k:v[:,permutation] if k in ('k','kgeom','kpos') else v for k,v in data.items()}
            torch.testing.assert_close(model(perm), pred[:,:,permutation], atol=6e-6, rtol=3e-5)
            loss = pred.square().mean(); loss.backward()
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_paper_operator_nonzero(self):
        torch.manual_seed(72); model = VisualAlignmentMatcher('hrope_paper').double()
        with torch.no_grad(): model.field.weight.normal_(0, .04)
        feat = torch.randn(1,5,96,dtype=torch.float64)
        intr = torch.diag(torch.tensor([.9,1.1,1.],dtype=torch.float64)).unsqueeze(0)
        a, inv = model.matrices(feat,intr); raw = model.field(feat).detach().numpy()[0]
        expected=[]
        for x in raw:
            e=x[:3]/np.linalg.norm(x[:3]); b=x[3:6]-e*np.dot(e,x[3:6]); b/=np.linalg.norm(b)
            r=np.stack((e,b,np.cross(e,b)),axis=-1); n=x[9:12]/np.linalg.norm(x[9:12])
            expected.append(intr.numpy()[0]@(r+np.clip(x[12],-.5,.5)*np.outer(np.clip(x[6:9],-.5,.5),n)))
        expected=np.array(expected)
        np.testing.assert_allclose(a.detach().numpy()[0],expected,rtol=1e-11,atol=1e-11)
        np.testing.assert_allclose(inv.detach().numpy()[0],np.linalg.inv(expected),rtol=1e-10,atol=1e-10)
        value=torch.randn(1,2,5,48,dtype=torch.float64);pos=torch.randn(1,5,2,dtype=torch.float64)
        full=np.zeros((5,48,48))
        for j in range(5):
            for block in range(8):full[j,3*block:3*block+3,3*block:3*block+3]=expected[j]
            phase=(pos.numpy()[0,j,:,None]*np.logspace(-1,.6,6)).ravel()
            for b,t in enumerate(phase): full[j,24+2*b:26+2*b,24+2*b:26+2*b]=[[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]]
        ops={'query':(a.transpose(-1,-2),full.transpose(0,2,1)), 'key':(inv,np.linalg.inv(full)),
             'value':(inv,np.linalg.inv(full)), 'output':(a,full)}
        for name,(matrix,reference) in ops.items():
            result=model.transport(value,matrix,pos,name).detach().numpy()[0]
            np.testing.assert_allclose(result,np.einsum('nij,hnj->hni',reference,value.numpy()[0]),atol=2e-11,rtol=1e-10)


if __name__=='__main__': unittest.main()
