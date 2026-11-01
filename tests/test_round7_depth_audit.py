import numpy as np
import pytest
from experiments import round7_depth_audit as a


def test_numpy_metrics_handle_unknown_nan_and_strict_delta1():
    pred=np.array([[3.,9.,2.,5.]]);target=np.array([[2.,np.nan,4.,4.]])
    valid=np.array([[1,0,1,1]],bool);boundary=np.array([[1,1,0,0]],bool)
    r=a.numpy_rows(pred,target,valid,boundary,[dict(scene='s')])[0]
    assert r['count']==3 and r['boundary_count']==1
    assert r['absrel_sum']==1.25 and r['abs_sum']==4 and r['sq_sum']==6
    assert r['boundary_absrel_sum']==.5
    # 5/4 equals1.25 and is excluded, as are1.5 and2.
    assert r['delta1_count']==0
    assert np.isclose(r['log_sq_sum'],np.log(1.5)**2+np.log(.5)**2+np.log(1.25)**2)


def test_identity_and_all_unknown_rows_are_finite_zero():
    pred=np.array([[2.,3.],[2.,3.]]);target=np.array([[2.,3.],[np.nan,0.]])
    valid=np.array([[1,1],[0,0]],bool);b=np.zeros_like(valid)
    r=a.numpy_rows(pred,target,valid,b,[dict(scene='s'),dict(scene='s')])
    assert r[0]['delta1_count']==2 and r[0]['absrel_sum']==0
    assert all(r[1][key]==0 for key in a.SUM_FIELDS)


def test_scene_macro_rmse_is_not_global_rmse():
    pred=np.array([[2.,2.],[5.,5.]])
    target=np.ones_like(pred);valid=np.array([[1,1],[1,0]],bool)
    rows=a.numpy_rows(pred,target,valid,valid,[dict(scene='a'),dict(scene='b')])
    macro=a.numpy_macro(rows,2)
    assert macro['metrics']['AbsRel']==2.5
    assert macro['metrics']['RMSE']==2.5
    assert not np.isclose(macro['metrics']['RMSE'],np.sqrt(18/3))
    assert macro['total_count']==3 and macro['total_positions']==4


def test_boundary_holes_dont_make_edges_and_only_valid_neighbors_count():
    z=np.full((1,18,24),2.);v=np.ones_like(z,bool);z[:,:,12:]=3.;v[:,0,0]=False;z[:,0,0]=np.nan
    b=a.boundary_from_labels(z.reshape(1,-1),v.reshape(1,-1)).reshape(z.shape)
    assert b.sum()==36 and not b[0,0,1] and not b[0,1,0]


def test_no_fresh_test_path_accepted_before_read():
    with pytest.raises(ValueError,match='only exact old validation'):
        a.load_old_validation([{'path':'results/round7_depth_newtest/test/anything'}])


def test_independent_attention_uniform_identity_and_visual_temperature():
    import torch
    x=dict(q=np.array([[[1.,0.],[0.,1.]]]),k=np.array([[[1.,0.],[0.,1.]]]),
        q_to_k=np.array([[[0.,0.],[1.,0.]]]),kmean=np.array([[[0.,0.],[1.,0.]]]),
        qcov=np.tile(np.eye(2),(1,2,1,1)),q_to_kcov=np.tile(np.eye(2),(1,2,1,1)),
        kcov=np.tile(np.eye(2),(1,2,1,1)))
    state={'qk.weight':torch.eye(2)}
    p=a.numpy_attention(x,state,'point',1.,2.)
    expected=1/(1+np.exp(-2.5))
    np.testing.assert_allclose(p[0].diagonal(),expected)
    np.testing.assert_allclose(p.sum(-1),1)


def test_raw_sensor_center_decoder_and_execution_guard_do_not_open_test():
    from experiments import round7_depth_test_audit as t
    mm=np.full((480,640),2000,np.uint32);mm[13,13]=0
    encoded=((mm*8)%65536)+(mm//8192)
    labels=t.independent_sensor_centers(encoded.astype(np.uint16))
    assert labels['valid'].sum()==431
    assert labels['target'][0]==0 and np.all(labels['target'][1:]==2.)
    assert labels['native_hw']==[480,640] and labels['boundary'].sum()==0
    with pytest.raises(ValueError,match='root test-complete'):
        t.audit(None,None,None,False)
