import unittest
import math
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from models.round54_calibrated_plane import make_model, ROLES, NEW_ROLES, plane_frame
from models.round52_camera_frame import CameraFrameMatcher, camera_lift
from tests.test_round52_camera_frame import CameraTests, full_operator
from third_party import round54_prope_torch as official


def numpy_plane_frame(model, features):
    raw=features@model.field.weight.detach().numpy().T+model.field.bias.detach().numpy()
    frames=[]
    for x in raw:
        a,b=(0.,0.) if model.plane_variant=='camera_offset' else x[:2]
        r=Rotation.from_quat([-b,a,0.,1.]).as_matrix()
        z=np.tanh(x[-1]);f=np.eye(4);f[:3,:3]=r;f[:3,3]=z*r[:,2]
        frames.append(f)
    return np.array(frames)


def numpy_plane_operator(model,features,K,geometry):
    camera=np.eye(4);camera[:3,:3]=geometry[0,:9].reshape(3,3);camera[:3,3]=geometry[0,9:12]
    lift=np.eye(4);lift[:3,:3]=K
    return (lift@np.linalg.inv(camera))@numpy_plane_frame(model,features)


class PlaneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(4)

    def test_independent_frame_and_common_frame_reduction(self):
        data=CameraTests().sample();feat=torch.randn(1,7,96,dtype=torch.float64)
        for role in NEW_ROLES:
            torch.manual_seed(18);model=make_model(role).double()
            with torch.no_grad():model.field.weight.normal_(0,.08)
            a,ai=model.matrices(feat,data['kK'],camera_lift(data['kgeom'],data['kK']))
            ref=numpy_plane_operator(model,feat.numpy()[0],data['kK'].numpy()[0],data['kgeom'].numpy()[0])
            np.testing.assert_allclose(a.detach().numpy()[0],ref,atol=2e-12)
            np.testing.assert_allclose(ai.detach().numpy()[0],np.linalg.inv(ref),atol=2e-12)
            p,pi=camera_lift(data['qgeom'],data['qK']);k,ki=camera_lift(data['kgeom'],data['kK'])
            f,fi=plane_frame(model.field(feat),role=='camera_offset')
            torch.testing.assert_close((p[:,None]@f)@(fi@ki[:,None]),(p@ki)[:,None].expand_as(a),atol=2e-12,rtol=2e-12)
            values=torch.randn(1,2,7,48,dtype=torch.float64);full=full_operator(ref,data['kpos'].numpy()[0])
            for op,mat,nmat in (('query',a.transpose(-1,-2),full.transpose(0,2,1)),
                ('key',ai,np.linalg.inv(full)),('value',ai,np.linalg.inv(full)),('output',a,full)):
                result=model.transport(values,mat,data['kpos'],op).detach().numpy()[0]
                np.testing.assert_allclose(result,np.einsum('nij,hnj->hni',nmat,values.numpy()[0]),atol=2e-11)

    def test_matched_initialization_inputs_and_gradients(self):
        data=CameraTests().sample();common=None;perm=torch.tensor([3,1,0,6,5,2,4])
        torch.manual_seed(11);reference=CameraFrameMatcher('camera_only').double()(data).detach()
        for role in ROLES:
            torch.manual_seed(11);model=make_model(role).double()
            state={k:v for k,v in model.state_dict().items() if not k.startswith('field.')}
            if common is None:common=state
            else:
                self.assertEqual(state.keys(),common.keys())
                for key in state:self.assertTrue(torch.equal(state[key],common[key]),(role,key))
            pred=model(data)
            if role in NEW_ROLES:torch.testing.assert_close(pred,reference,atol=0,rtol=0)
            else:
                torch.manual_seed(11);old=CameraFrameMatcher(role).double()
                torch.testing.assert_close(old(data),pred,atol=0,rtol=0)
            extra=dict(data,gt_valid=torch.randn(1,7),mean3=torch.randn(1,7,3),normal=torch.randn(1,7,3))
            torch.testing.assert_close(model(extra),pred,atol=0,rtol=0)
            shuffled={k:(v[:,perm] if k in ('k','kgeom','kpos') else v) for k,v in data.items()}
            torch.testing.assert_close(model(shuffled),pred[:,:,perm],atol=1e-10,rtol=1e-10)
            if role not in NEW_ROLES:continue
            probe=torch.randn_like(pred);(pred*probe).sum().backward()
            self.assertTrue((model.field.weight.grad.abs().sum(-1)>1e-9).all())
            model.zero_grad(set_to_none=True)
            with torch.no_grad():model.field.weight.normal_(0,.015)
            (model(data)*probe).sum().backward();p=model.field.weight
            for row in range(len(p)):
                ix=(row,int(p.grad[row].abs().argmax()));analytic=float(p.grad[ix]);original=float(p.detach()[ix]);eps=1e-5
                with torch.no_grad():
                    p[ix]=original+eps;plus=float((model(data)*probe).sum())
                    p[ix]=original-eps;minus=float((model(data)*probe).sum());p[ix]=original
                numeric=(plus-minus)/(2*eps)
                self.assertLess(abs(analytic-numeric)/max(1,abs(analytic),abs(numeric)),2e-5)

    def test_official_prope_transport_with_declared_chart_and_phase_adapter(self):
        data={k:v.float() for k,v in CameraTests().sample().items()}
        geometry=torch.stack((data['qgeom'],data['kgeom']),1)
        intrinsics=torch.stack((data['qK'],data['kK']),1)
        view=[]
        for geom in (data['qgeom'],data['kgeom']):
            c=torch.eye(4)[None];c[:,:3,:3]=geom[:,0,:9].reshape(-1,3,3);c[:,:3,3]=geom[:,0,9:12]
            view.append(torch.linalg.inv(c))
        view=torch.stack(view,1)
        # Official implementation centers at image midpoint; R52 centers at
        # principal point. Express R52's already-centered chart for the oracle.
        pixel_K=intrinsics.clone();pixel_K[...,0,0]*=192;pixel_K[...,1,1]*=144
        pixel_K[...,0,2]=96;pixel_K[...,1,2]=72
        xy=torch.cat((data['qpos'],data['kpos']),1)[0]
        omega=torch.logspace(-1,.6,6)
        coeff=[((xy[:,i,None]*omega).cos()[None,None],-(xy[:,i,None]*omega).sin()[None,None]) for i in (0,1)]
        oq,ok,oo=official._prepare_apply_fns(48,view,pixel_K,7,1,192,144,coeff[0],coeff[1])
        values=torch.randn(1,2,14,48)
        def to_split(v):
            return torch.cat((v[...,:24],*[torch.cat((v[...,s:s+12:2],v[...,s+1:s+12:2]),-1) for s in (24,36)]),-1)
        def to_interleaved(v):
            return torch.cat((v[...,:24],*[v[...,s:s+12].reshape(*v.shape[:-1],2,6).transpose(-1,-2).flatten(-2) for s in (24,36)]),-1)
        model=make_model('camera_only')
        for operation,fn in (('query',oq),('key',ok),('value',ok),('output',oo)):
            expected=to_interleaved(fn(to_split(values)));parts=[]
            for j,prefix in enumerate(('q','k')):
                p,pi=camera_lift(data[prefix+'geom'],data[prefix+'K'])
                m=p.transpose(-1,-2) if operation=='query' else p if operation=='output' else pi
                parts.append(model.transport(values[:,:,j*7:(j+1)*7],m[:,None],data[prefix+'pos'],operation))
            torch.testing.assert_close(torch.cat(parts,2),expected,atol=2e-6,rtol=2e-6)


if __name__=='__main__':unittest.main()
