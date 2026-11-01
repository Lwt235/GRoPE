import numpy as np
import pytest
from experiments import round7_scale_data as d


def cache():
    rng=np.random.default_rng(18)
    f=rng.normal(size=(432,128));f/=np.linalg.norm(f,axis=-1,keepdims=True)
    return {'features':f.astype(np.float32),'predicted_depth':np.full((144,192),2.),
        'K':np.array([[120.,0.,95.5],[0.,120.,71.5],[0.,0.,1.]]),'c2w':np.eye(4)}


def test_three_by_three_pool_exact_members_and_l2():
    c=cache();out=d.pool_features(c['features'],24)
    assert out.shape==(48,128)
    ids=[row*24+col for row in range(3) for col in range(3)]
    expected=c['features'][ids].astype(np.float64).mean(0);expected/=np.linalg.norm(expected)
    np.testing.assert_allclose(out[0],expected,atol=1e-8,rtol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(out,axis=-1),1,atol=1e-7)
    for stride in (8,16):
        np.testing.assert_array_equal(d.pool_features(c['features'],stride),d.frozen_data.pool_features(c['features'],stride))


@pytest.mark.parametrize('qs,ks',d.CONDITIONS)
def test_every_new_condition_full_image_mass_and_common_units(qs,ks):
    c=cache();inputs,_=d.make_inputs(c,c,qs,ks)
    assert len(inputs['q'])==(144//qs)*(192//qs)
    assert len(inputs['k'])==(144//ks)*(192//ks)
    np.testing.assert_allclose(inputs['qcov'][:,0,0],qs**2/(12*64))
    np.testing.assert_allclose(inputs['kcov'][:,0,0],ks**2/(12*64))
    np.testing.assert_allclose(inputs['q_to_k'],inputs['qmean'],atol=1e-6)
    np.testing.assert_allclose(inputs['q_to_kcov'],inputs['qcov'],atol=1e-7)
    depth=np.full((48,48),2.)
    K=np.array([[100.,0.,23.5],[0.,100.,23.5],[0.,0.,1.]])
    labels=d.region_labels(depth,depth,K,K,np.eye(4),np.eye(4),qs,ks,image_hw=(48,48))
    assert labels.target_counts.sum()==48**2
    np.testing.assert_array_equal(d.exact_visible_mass(labels),np.full((48//qs)**2,qs**2))
    assert labels.gt_valid.all()
    np.testing.assert_allclose(labels.soft_labels.sum(-1),1,atol=1e-7)


def test_nonnested24_to16_uniform_area_counts_are_not_uniform_four_neighbors():
    depth=np.full((48,48),2.);K=np.array([[100.,0.,23.5],[0.,100.,23.5],[0.,0.,1.]])
    y=d.region_labels(depth,depth,K,K,np.eye(4),np.eye(4),24,16,(48,48))
    np.testing.assert_array_equal(y.target_counts[0,[0,1,3,4]],[256,128,128,64])
    np.testing.assert_allclose(y.soft_labels[0,[0,1,3,4]],[4/9,2/9,2/9,1/9],atol=1e-7)


def test24_fraction_rounding_does_not_define_raw_area_mass():
    depth=np.full((48,48),2.);depth[0,0]=0
    K=np.array([[100.,0.,23.5],[0.,100.,23.5],[0.,0.,1.]])
    y=d.region_labels(depth,depth,K,K,np.eye(4),np.eye(4),24,24,(48,48))
    assert d.exact_visible_mass(y)[0]==575
    assert y.counts_by_reason[0,1]==1


def test_old_grids_are_not_rerun_and_no_gt_inputs_enter_forward():
    c=cache()
    with pytest.raises(ValueError,match='NEW24'):d.make_inputs(c,c,8,16)
    before,_=d.make_inputs(c,c,24,8)
    c['sensor_depth']=np.zeros((480,640));c['gt_valid']=np.zeros(48,bool)
    after,_=d.make_inputs(c,c,24,8)
    for name in before:np.testing.assert_array_equal(before[name],after[name])
