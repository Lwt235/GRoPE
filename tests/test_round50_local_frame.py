import unittest
import numpy as np
import torch
from scipy.linalg import expm
from experiments.round50_data import scene, view, intersect, centers, example, labels
from models.round50_local_frame import LocalFrameMatcher, ROLES


def fixture(count=7):
    ex = example(4123, 'slant')
    return {k:torch.from_numpy(v[None, :count] if v.ndim and v.shape[0]==64 else v[None]).double()
            for k,v in ex.items()}


def numpy_frame(model, features, intrinsic):
    role = model.role
    if role == 'frame_identity': return np.broadcast_to(np.eye(4), (*features.shape[:-1], 4, 4)).copy()
    raw = features@model.field.weight.detach().numpy().T+model.field.bias.detach().numpy()
    out=[]
    for x in raw.reshape(-1, raw.shape[-1]):
        if role in ('gs_frame', 'hrope'):
            u=x[:3]/np.linalg.norm(x[:3]);v=x[3:6]-u*np.dot(u,x[3:6]);v/=np.linalg.norm(v)
            r=np.column_stack((u,v,np.cross(u,v)))
        if role == 'hrope':
            t=.5*np.tanh(x[6:9]);n=x[9:12]/np.linalg.norm(x[9:12]);kap=.5*np.tanh(x[12])
            out.append(r+kap*np.outer(t,n));continue
        if role == 'gs_frame': l=r@np.diag(np.exp(.5*np.tanh(x[6:9])))@r.T;mu=np.tanh(x[9:12])
        elif role == 'gs_iso': l=np.eye(3)*np.exp(.5*np.tanh(x[0]));mu=np.tanh(x[1:4])
        else: l=expm(.5*np.tanh(x[:9]).reshape(3,3));mu=np.tanh(x[9:12])
        g=np.eye(4);g[:3,:3]=l;g[:3,3]=mu;out.append(np.linalg.inv(g))
    d=3 if role=='hrope' else 4
    result=np.stack(out).reshape(*features.shape[:-1],d,d)
    if role=='hrope':result=intrinsic[:,None]@result
    return result


class TestLocalFrame(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_geometry_and_differential(self):
        s=scene(4123,'slant');q=view(s,np.zeros(3),np.eye(3),48,3)
        k=view(s,s['camera'],s['rotation'],s['focal'],4)
        lab=labels(s,q,k);valid=lab['valid']
        self.assertGreater(valid.sum(),12)
        expected=(centers()[lab['indices']]*lab['weights'][...,None]).sum(1)
        np.testing.assert_allclose(expected[valid],lab['target'][valid],atol=1e-12)
        # Direct plane-induced homographies and independent finite differences.
        K=np.array([[48,0,31.5],[0,48,31.5],[0,0,1.]])
        Kk=K.copy();Kk[0,0]=Kk[1,1]=s['focal']
        r=s['rotation'].T;t=-r@s['camera']
        for i in np.flatnonzero(valid)[:12]:
            n=q['normal'][i];d=np.dot(n,q['xyz'][i])
            H=Kk@(r+np.outer(t,n)/d)@np.linalg.inv(K)
            p=H@np.r_[centers()[i],1];np.testing.assert_allclose(p[:2]/p[2],lab['target'][i],atol=1e-12)
            jac=[]
            for axis in range(2):
                e=np.eye(2)[axis]*1e-4
                a=H@np.r_[centers()[i]+e,1];b=H@np.r_[centers()[i]-e,1]
                jac.append((a[:2]/a[2]-b[:2]/b[2])/2e-4)
            np.testing.assert_allclose(np.stack(jac,-1),lab['warp_jac'][i],rtol=1e-7,atol=1e-8)

    def test_nonzero_operator_and_transport(self):
        for role in ('hrope','gs_frame','gs_iso','affine','frame_identity'):
            torch.manual_seed(11);m=LocalFrameMatcher(role).double()
            if hasattr(m,'field'):
                with torch.no_grad():m.field.weight.normal_(0,.015)
            f=torch.randn(1,3,96,dtype=torch.float64);K=torch.diag(torch.tensor([.8,.8,1.],dtype=torch.float64))[None]
            a,ai=m.matrices(f,K);reference=numpy_frame(m,f.numpy(),K.numpy())
            np.testing.assert_allclose(a.detach(),reference,rtol=1e-10,atol=1e-10)
            np.testing.assert_allclose((a@ai).detach(),np.broadcast_to(np.eye(a.shape[-1]),a.shape),atol=1e-10)
            q,k,v=[torch.randn(1,2,3,48,dtype=torch.float64) for _ in range(3)]
            pos=torch.randn(1,3,2,dtype=torch.float64)
            full=[];d=a.shape[-1];omega=np.logspace(-1,.6,6)
            for i in range(3):
                mat=np.eye(48)
                for j in range(0,24,d):mat[j:j+d,j:j+d]=reference[0,i]
                phases=(pos.numpy()[0,i,:,None]*omega).ravel()
                for j,p in enumerate(phases):
                    c,z=np.cos(p),np.sin(p);mat[24+2*j:26+2*j,24+2*j:26+2*j]=[[c,z],[-z,c]]
                full.append(mat)
            full=np.stack(full)
            tq=m.transport(q,a.transpose(-1,-2),pos,'query').detach().numpy()[0]
            tk=m.transport(k,ai,pos,'key').detach().numpy()[0]
            tv=m.transport(v,ai,pos,'value').detach().numpy()[0]
            for h in range(2):
                eq=np.stack([full[i].T@q.numpy()[0,h,i] for i in range(3)])
                ek=np.stack([np.linalg.solve(full[i],k.numpy()[0,h,i]) for i in range(3)])
                ev=np.stack([np.linalg.solve(full[i],v.numpy()[0,h,i]) for i in range(3)])
                np.testing.assert_allclose(tq[h]@tk[h].T,eq@ek.T,atol=1e-10)
                np.testing.assert_allclose(tv[h],ev,atol=1e-10)
            aggregate=torch.randn_like(q)
            actual=m.transport(aggregate,a,pos,'output').detach().numpy()[0]
            for h in range(2):
                expected=np.stack([full[i]@aggregate.numpy()[0,h,i] for i in range(3)])
                np.testing.assert_allclose(actual[h],expected,atol=1e-10)

    def test_shared_parameters_labels_and_permutation(self):
        data=fixture();reference=None;zero=None
        for role in ROLES:
            torch.manual_seed(22);m=LocalFrameMatcher(role).double().eval()
            shared={k:v for k,v in m.state_dict().items() if not k.startswith('field.')}
            if reference is None:reference=shared
            else:
                for k in shared:torch.testing.assert_close(shared[k],reference[k],rtol=0,atol=0)
            with torch.no_grad():
                out=m(data);z=m(data,zero_values=True)
                if zero is None:zero=z
                else:torch.testing.assert_close(z,zero,atol=1e-12,rtol=1e-12)
                clean={k:v for k,v in data.items() if k in ('qrgb','krgb','qgeom','kgeom','qnoisy','knoisy','qpos','kpos','qK','kK')}
                torch.testing.assert_close(m(clean),out,atol=0,rtol=0)
                perm=torch.tensor([3,1,0,6,5,2,4]);modified={k:(v[:,perm] if v.ndim>2 and v.shape[1]==7 else v) for k,v in clean.items()}
                torch.testing.assert_close(m(modified),out[:,perm][:,:,perm],atol=1e-10,rtol=1e-10)

    def test_shape_gauge_and_gradient(self):
        torch.manual_seed(33);m=LocalFrameMatcher('gs_frame').double()
        with torch.no_grad():m.field.bias[6:9].copy_(torch.tensor([.2,-.3,.1]))
        f=torch.randn(1,3,96,dtype=torch.float64);K=torch.eye(3,dtype=torch.float64)[None]
        a,_=m.matrices(f,K)
        with torch.no_grad():m.field.bias[:6]*=-1
        b,_=m.matrices(f,K)
        torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-12)
        for role in ROLES:
            torch.manual_seed(33);m=LocalFrameMatcher(role).double()
            if role in ('rope','frame_identity'):continue
            with torch.no_grad():
                for p in m.field.parameters():p.add_(.01*torch.randn_like(p))
            d=fixture(4);probe=torch.randn(1,4,4,dtype=torch.float64)
            loss=(m(d)*probe).sum();loss.backward()
            ps=[p for p in m.field.parameters() if p.grad is not None]
            p=max(ps,key=lambda t:float(t.grad.abs().max()));index=np.unravel_index(int(p.grad.abs().argmax()),p.shape)
            analytic=float(p.grad[index]);original=float(p.detach()[index]);eps=1e-5
            with torch.no_grad():
                p[index]=original+eps;plus=float((m(d)*probe).sum())
                p[index]=original-eps;minus=float((m(d)*probe).sum());p[index]=original
            numeric=(plus-minus)/(2*eps)
            self.assertTrue(np.isfinite(analytic));self.assertGreater(abs(analytic),1e-8)
            self.assertLess(abs(analytic-numeric)/max(1,abs(analytic),abs(numeric)),2e-5,role)


if __name__=='__main__':unittest.main()
