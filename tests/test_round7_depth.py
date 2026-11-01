import numpy as np
import torch
from models.round7_depth_attention import DepthAttention, CONFIGS, depth_loss
from experiments.round7_depth_data import observations, query_labels, center_prediction
from experiments.round7_depth_train import metric_rows, macro


def cache(depth=2.,pose=None):
    rng=np.random.default_rng(3);f=rng.normal(size=(432,128)).astype('float32');f/=np.linalg.norm(f,axis=-1,keepdims=True)
    return dict(features=f,predicted_depth=np.full((144,192),depth,'float32'),
        K=np.array([[100.,0,95.5],[0,100.,71.5],[0,0,1.]]),c2w=np.eye(4) if pose is None else pose)

def tensor_data():
    x,_=observations(cache(),cache());d={k:torch.from_numpy(v[None]) for k,v in x.items()}
    d.update(target=torch.full((1,432),2.1),valid=torch.ones((1,432),dtype=torch.bool),boundary=torch.zeros((1,432),dtype=torch.bool))
    return d

def test_v_camera_transform_and_same_camera_identity():
    pose=np.eye(4);pose[2,3]=1
    x,_=observations(cache(),cache(3.,pose))
    np.testing.assert_allclose(x['k_value_depth'],4.)
    same,_=observations(cache(),cache())
    np.testing.assert_allclose(same['q_to_k'],same['qmean'],atol=1e-6)
    np.testing.assert_allclose(same['q_to_kcov'],same['qcov'],atol=1e-6)

def test_sensor_missing_does_not_create_boundary_or_forward_mask():
    z=np.ones((144,192))*2;z[:8,:8]=0
    y=query_labels(z);assert y['valid'].sum()==431;assert y['boundary'].sum()==0
    x,_=observations(cache(),cache());assert not set(y)&set(x)
    a={k:v.copy() for k,v in x.items()};query_labels(np.ones((144,192))*9)
    for k in a:np.testing.assert_array_equal(a[k],x[k])

def test_center_sampling_and_edge_labels():
    z=np.indices((144,192))[1].astype(float)/10+1
    np.testing.assert_allclose(center_prediction(z)[1:24],(np.arange(1,24)*8+3.5)/10+1)
    z=np.full((144,192),2.);z[:,96:]=3.
    y=query_labels(z);assert y['boundary'].sum()==36;assert y['valid'].sum()==432

def test_all_controls_equal_capacity_and_singleview_ignores_key():
    torch.set_num_threads(1);d=tensor_data();counts=[]
    for role in CONFIGS:
        torch.manual_seed(11);model=DepthAttention(role);counts.append(sum(p.numel() for p in model.parameters()))
    assert len(set(counts))==1
    model=DepthAttention('singleview');a=model(d)
    altered=dict(d,k=d['k']*3,k_value_depth=d['k_value_depth']*3,q_to_k=d['q_to_k']*3)
    torch.testing.assert_close(a,model(altered),rtol=0,atol=0)

def test_gt_isolation_actual_values_and_gradients():
    torch.set_num_threads(1);d=tensor_data();model=DepthAttention('full');a=model(d)
    changed=dict(d,target=torch.full_like(d['target'],float('nan')),valid=~d['valid'],boundary=~d['boundary'])
    torch.testing.assert_close(a,model(changed),rtol=0,atol=0)
    assert (model(d,intervention='zero_v')-a).abs().max()>0
    assert (model(d,intervention='reverse_v')-a).abs().max()>0
    d['target'][0,0]=float('nan');d['valid'][0,0]=False
    loss=depth_loss(a,d['target'],d['valid']);loss.backward()
    for name,p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(),name
        assert p.grad.abs().sum()>0,name

def test_metric_population_includes_all_query_valid():
    data=dict(target=torch.tensor([[2.,0.,4.]]),valid=torch.tensor([[True,False,True]]),boundary=torch.tensor([[True,False,False]]))
    rows=metric_rows(torch.tensor([[3.,9.,2.]]),data,[dict(scene='x',pair_id='1:2')])
    assert rows[0]['count']==2 and rows[0]['absrel_sum']==1
    result=macro(rows);assert result['metrics']['AbsRel']==.5
    assert np.isclose(result['metrics']['RMSE'],np.sqrt(2.5))
