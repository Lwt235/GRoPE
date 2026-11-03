"""Data-free checks of native GS conversion and matched Transformer PE."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.linalg import expm
from scipy.special import erf
import torch
import torch.nn.functional as F
from models.round65_actual_gs import NativeGSClassifier, ROLES, common_features
from experiments.round65_data import quaternion_rotation, native_arrays, read_vertices, split_rows


def synthetic():
    rng = np.random.default_rng(6502)
    centers = torch.tensor(rng.normal(size=(2,9,3)),dtype=torch.float32)
    a = rng.normal(size=(2,9,3,3))*.08
    covariance = torch.tensor(a@a.swapaxes(-1,-2)+.003*np.eye(3),dtype=torch.float64)
    appearance = torch.tensor(rng.normal(size=(2,9,4)),dtype=torch.float32)
    return centers,covariance,appearance


def numpy_relation(centers,covariance):
    c,s=np.asarray(centers,dtype=np.float64),np.asarray(covariance,dtype=np.float64)
    p=np.linalg.inv(s)
    spectrum=.5*np.log(np.linalg.eigvalsh(s)/.04)
    result=np.zeros((len(c),len(c),11))
    for i in range(len(c)):
        for j in range(len(c)):
            d=c[i]-c[j]
            result[i,j]=np.r_[np.linalg.norm(d)/.2,spectrum[i],spectrum[j],
                np.log1p(d@p[i]@d),np.log1p(d@p[j]@d),
                np.log1p(np.trace(p[i]@s[j])),np.log1p(np.trace(p[j]@s[i]))]
    return result


def numpy_project(x,state):
    x=x@state['mlp.0.weight'].detach().numpy().T+state['mlp.0.bias'].detach().numpy()
    x=.5*x*(1+erf(x/np.sqrt(2)))
    return x@state['mlp.2.weight'].detach().numpy().T


def unvectorize(vector):
    result=np.zeros((3,3));i,j=np.triu_indices(3)
    result[i,j]=vector/np.where(i==j,1,np.sqrt(2))
    result[j,i]=result[i,j]
    return result


class ActualGSTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def test_native_conversion_and_ply(self):
        rng=np.random.default_rng(6503);q=rng.normal(size=(13,4))
        expected=Rotation.from_quat(q[:,[1,2,3,0]]).as_matrix()
        np.testing.assert_allclose(quaternion_rotation(q),expected,atol=1e-14)
        np.testing.assert_allclose(quaternion_rotation(-q),expected,atol=1e-14)
        names=['x','y','z','opacity','f_dc_0','f_dc_1','f_dc_2']
        names += [f'scale_{i}' for i in range(3)]+[f'rot_{i}' for i in range(4)]
        vertex=np.zeros(13,dtype=[(n,'<f4') for n in names])
        xyz=rng.normal(size=(13,3));logs=rng.normal(-3,.2,size=(13,3))
        for k,n in enumerate(['x','y','z']):vertex[n]=xyz[:,k]
        for k in range(3):vertex[f'scale_{k}']=logs[:,k]
        for k in range(4):vertex[f'rot_{k}']=q[:,k]
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'synthetic.ply'
            header='ply\nformat binary_little_endian 1.0\nelement vertex 13\n'
            header+=''.join('property float '+n+'\n' for n in names)+'end_header\n'
            path.write_bytes(header.encode()+vertex.tobytes())
            parsed=read_vertices(path)
            np.testing.assert_array_equal(parsed,vertex)
            out,stats=native_arrays(parsed,'synthetic',11)
        indices=out['indices'];radius=stats['rms_radius']
        raw_xyz=np.column_stack([vertex[n] for n in ('x','y','z')]).astype(float)
        np.testing.assert_allclose(out['centers'],(raw_xyz[indices]-raw_xyz.mean(0))/radius,atol=1e-7)
        ref_rotation=Rotation.from_quat(np.column_stack([vertex[f'rot_{i}'] for i in (1,2,3,0)])).as_matrix()
        for k,i in enumerate(indices):
            variances=np.exp(2*np.array([vertex[f'scale_{d}'][i] for d in range(3)],dtype=float))/radius**2
            ref=ref_rotation[i]@np.diag(variances)@ref_rotation[i].T+1e-8*np.eye(3)
            np.testing.assert_allclose(out['covariance'][k],ref,atol=1e-14)
        np.testing.assert_array_equal(out['appearance'],np.zeros((11,4)))

    def test_relation_reference_and_invariance(self):
        c,s,a=synthetic();torch.manual_seed(11);model=NativeGSClassifier('gaussian_relation')
        actual=model.position(c,s).detach().numpy()
        for b in range(2):
            ref=numpy_project(numpy_relation(c[b].numpy(),s[b].numpy()),model.position.state_dict())
            np.testing.assert_allclose(actual[b],ref,atol=3e-6,rtol=2e-5)
        rot=torch.tensor(Rotation.from_euler('xyz',[.3,-.5,.2]).as_matrix())
        transformed_c=(c.double()@rot.T+torch.tensor([1.,-2.,.5])).float()
        transformed_s=rot@s@rot.T
        torch.testing.assert_close(model.position(transformed_c,transformed_s),
                                   model.position(c,s),atol=1e-5,rtol=2e-5)

    def test_direct_control_retains_geometry_information(self):
        c,s,a=synthetic();model=NativeGSClassifier('direct_mlp')
        descriptors=model.position.descriptors(c,s).numpy()
        i,j=2,5;pair=descriptors[0,i,j]
        np.testing.assert_allclose(pair[:3]*.2,(c[0,i]-c[0,j]).numpy(),atol=2e-7)
        np.testing.assert_allclose(.04*expm(2*unvectorize(pair[4:10])),s[0,i].numpy(),atol=2e-8)
        np.testing.assert_allclose(.04*expm(2*unvectorize(pair[10:16])),s[0,j].numpy(),atol=2e-8)

    def test_shared_content_initialization_gradient_and_permutation(self):
        c,s,a=synthetic();shared=None;features=common_features(c,s,a)
        perm=torch.tensor([7,0,3,8,2,6,1,5,4])
        for role in ROLES:
            torch.manual_seed(11);model=NativeGSClassifier(role)
            current={k:v.clone() for k,v in model.state_dict().items() if not k.startswith('position.')}
            if shared is None:shared=current
            else:
                for k,v in current.items():torch.testing.assert_close(v,shared[k],atol=0,rtol=0)
            received=[]
            hook=model.stem.register_forward_pre_hook(lambda m,args:received.append(args[0].detach().clone()))
            model.train();loss=F.cross_entropy(model(c,s,a),torch.tensor([0,59]));loss.backward()
            hook.remove();torch.testing.assert_close(received[0],features,atol=0,rtol=0)
            self.assertTrue(torch.isfinite(loss))
            for name,p in model.named_parameters():
                self.assertIsNotNone(p.grad,name);self.assertTrue(torch.isfinite(p.grad).all(),name)
            for name,p in model.position.named_parameters():self.assertGreater(float(p.grad.norm()),0.,name)
            model.eval()
            with torch.no_grad():
                torch.testing.assert_close(model(c,s,a),model(c[:,perm],s[:,perm],a[:,perm]),atol=3e-6,rtol=3e-5)

    def test_split_population_and_duplicates(self):
        rows=[dict(taxonomy_id=c,file_path=f'{c}/object_{i}.ply',sha256=f'{c}_{i}')
              for c in range(60) for i in range(20)]
        partitions=split_rows(rows)
        self.assertEqual({k:len(v) for k,v in partitions.items()},dict(fit=960,val=120,dev_eval=120))
        self.assertEqual({r['file_path'] for records in partitions.values() for r in records},
                         {r['file_path'] for r in rows})
        p0=partitions['val'][0]['file_path'];p1=partitions['fit'][0]['file_path']
        for r in rows:
            if r['file_path'] in (p0,p1):r['sha256']='duplicate'
        with self.assertRaisesRegex(ValueError,'byte-identical'):split_rows(rows)


if __name__=='__main__':
    unittest.main()
