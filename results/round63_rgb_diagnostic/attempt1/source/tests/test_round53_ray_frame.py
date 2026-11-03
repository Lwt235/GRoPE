import unittest
import numpy as np
import torch
from scipy.linalg import expm
from experiments.round51_data import camera_fields, pixels
from models.round51_visual_alignment import VisualAlignmentMatcher
from models.round53_ray_frame import RayFrameMatcher, ROLES, FIELDS, camera_pose, center_rays, make_model
from models.round52_camera_frame import CameraFrameMatcher


def fixture():
    torch.manual_seed(5310)
    K = np.array([[170.,0,94.7],[0,172.,73.1],[0,0,1.]])
    C = np.eye(4); C[:3,:3] = expm(np.array([[0.,-.3,.2],[.3,0.,-.1],[-.2,.1,0.]])); C[:3,3] = [.3,-.2,.1]
    data = {k:torch.randn(1,7,128,dtype=torch.float64) for k in ('q','k')}
    for prefix, pose in (('q',np.eye(4)),('k',C)):
        for key,value in camera_fields(dict(K=K),pose).items():
            data[prefix+key] = torch.from_numpy(value if key=='K' else value[:7]).double()[None]
    return data,K,C


def numpy_ray_frame(model, features, rays):
    if model.experiment_role=='pose_only':
        return np.broadcast_to(np.eye(4),(len(features),4,4)).copy()
    raw = features@model.field.weight.detach().numpy().T+model.field.bias.detach().numpy()
    frames = []
    for x,ray in zip(raw,rays):
        depth = .1+np.logaddexp(0.,x[0]+np.log(np.expm1(.9)))
        if model.experiment_role=='ray_point': L=np.eye(3)
        elif model.experiment_role=='ray_iso': L=np.eye(3)*np.exp(.5*np.tanh(x[1]))
        elif model.experiment_role in ('ray_gaussian','ray_gaussian_hybrid'):
            L=np.diag(np.exp(.5*np.tanh(x[1:4]))); L[np.tril_indices(3,-1)]=.5*np.tanh(x[4:7])
        else:L=expm(.5*np.tanh(x[1:10]).reshape(3,3))
        g=np.eye(4);g[:3,:3]=L;g[:3,3]=depth*ray;frames.append(g)
    return np.array(frames)


class RayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(4)

    def test_ray_projection_frame_inverse_and_world_gauge(self):
        data,K,C=fixture();rays=center_rays(data['kpos'],data['kK'])
        reference=np.column_stack((pixels()[:7],np.ones(7)))@np.linalg.inv(K).T
        np.testing.assert_allclose(rays.numpy()[0],reference,atol=7e-8)
        feat=torch.randn(1,7,96,dtype=torch.float64);camera=camera_pose(data['kgeom'])
        W=np.eye(4);W[:3,:3]=expm(np.array([[0.,.2,-.1],[-.2,0.,.3],[.1,-.3,0.]]));W[:3,3]=[.4,-.3,.1]
        W=torch.from_numpy(W)[None]
        for role in ('pose_only',*FIELDS):
            torch.manual_seed(17);model=RayFrameMatcher(role).double()
            if role in FIELDS:
                with torch.no_grad():model.field.weight.normal_(0,.02)
            a,ai=model.ray_matrices(feat,rays,(camera,torch.linalg.inv(camera)))
            g=numpy_ray_frame(model,feat.numpy()[0],rays.numpy()[0]);f=camera.numpy()[0]@g
            np.testing.assert_allclose(ai.detach().numpy()[0],f,atol=1e-11)
            np.testing.assert_allclose(a.detach().numpy()[0],np.linalg.inv(f),atol=1e-11)
            if role!='ray_gaussian_hybrid':
                values=torch.randn(1,2,7,48,dtype=torch.float64)
                full=np.zeros((7,48,48))
                for index in range(0,48,4):full[:,index:index+4,index:index+4]=np.linalg.inv(f)
                for op,mat,ref in (('query',a.transpose(-1,-2),full.transpose(0,2,1)),
                                  ('key',ai,np.linalg.inv(full)),('value',ai,np.linalg.inv(full)),('output',a,full)):
                    got=model.transport(values,mat,data['kpos'],op).detach().numpy()[0]
                    np.testing.assert_allclose(got,np.einsum('nij,hnj->hni',ref,values.numpy()[0]),atol=2e-11)
            other=camera.clone();other[:,:3,3]+=torch.tensor([.1,.2,-.1])
            b,bi=model.ray_matrices(feat*.8,rays,(other,torch.linalg.inv(other)))
            aw,awinv=model.ray_matrices(feat,rays,(W@camera,torch.linalg.inv(W@camera)))
            bw,bwinv=model.ray_matrices(feat*.8,rays,(W@other,torch.linalg.inv(W@other)))
            torch.testing.assert_close(a@bi,aw@bwinv,atol=1e-11,rtol=1e-11)

    def test_shape_initialization_and_gradients(self):
        data,_,_=fixture();rays=center_rays(data['qpos'],data['qK']);feat=torch.randn(1,7,96,dtype=torch.float64)
        initial=None
        for role in FIELDS:
            torch.manual_seed(11);model=RayFrameMatcher(role).double()
            g,gi=model.gaussian_frame(feat,rays)
            if initial is None:initial=g
            else:torch.testing.assert_close(g,initial,atol=0,rtol=0)
            torch.testing.assert_close(g[...,:3,3],rays,atol=1e-12,rtol=1e-12)
            if role=='ray_gaussian':
                probe=torch.randn_like(g);(g*probe).sum().backward()
                self.assertTrue((model.field.weight.grad[1:].abs().sum(-1)>1e-8).all())
                model.zero_grad(set_to_none=True)
            with torch.no_grad():model.field.weight.normal_(0,.01)
            probe=torch.randn(1,7,7,dtype=torch.float64);loss=(model(data)*probe).sum();loss.backward()
            p=model.field.weight;idx=np.unravel_index(int(p.grad.abs().argmax()),p.shape)
            analytic=float(p.grad[idx]);original=float(p.detach()[idx]);eps=1e-5
            with torch.no_grad():
                p[idx]=original+eps;plus=float((model(data)*probe).sum())
                p[idx]=original-eps;minus=float((model(data)*probe).sum());p[idx]=original
            numeric=(plus-minus)/(2*eps)
            self.assertGreater(abs(analytic),1e-8)
            self.assertLess(abs(analytic-numeric)/max(1,abs(analytic),abs(numeric)),2e-5,role)

    def test_shared_controls_input_exclusion_and_permutation(self):
        data,_,_=fixture();common=None;perm=torch.tensor([3,1,0,6,5,2,4])
        for role in ROLES:
            torch.manual_seed(11);model=make_model(role).double().eval()
            state={k:v for k,v in model.state_dict().items() if not k.startswith('field.')}
            if common is None:common=state
            else:
                self.assertEqual(common.keys(),state.keys())
                for key in state:self.assertTrue(torch.equal(state[key],common[key]),(role,key))
            result=model(data)
            extra=dict(data,gt_valid=torch.randn(1,7),predicted_depth=torch.randn(1,7),mean3=torch.randn(1,7,3))
            torch.testing.assert_close(result,model(extra),atol=0,rtol=0)
            permuted={k:(v[:,perm] if k in ('k','kpos','kgeom') else v) for k,v in data.items()}
            torch.testing.assert_close(model(permuted),result[:,:,perm],atol=1e-10,rtol=1e-10)
            if role in ('nope','mlp','hrope'):
                torch.manual_seed(11);old=VisualAlignmentMatcher(role).double().eval()
                torch.testing.assert_close(old(data),result,atol=0,rtol=0)
            elif role=='projection_only':
                torch.manual_seed(11);old=CameraFrameMatcher('camera_only').double().eval()
                torch.testing.assert_close(old(data),result,atol=0,rtol=0)


if __name__=='__main__':unittest.main()
