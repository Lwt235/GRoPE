"""Contract and analytic numerical tests, using no dataset or label files."""
import numpy as np
import pytest
import torch
from experiments import round12_depth_geometry as G


def observation():
    y,x=np.mgrid[:144,:192]
    xy=np.stack([x,y],-1).reshape(-1,2).astype(np.float32)
    cov=np.broadcast_to(np.eye(2,dtype=np.float32)/12,(27648,2,2)).copy()
    return dict(q_to_k_xy=torch.from_numpy(xy),q_to_k_pixel_cov=torch.from_numpy(cov))


def test_actual_physical_bounds_cover_native_area():
    p=G.g8_partition('cpu')
    assert tuple(p['bounds'][0].tolist())==(0,0,8,8)
    assert tuple(p['cell_bounds'][0].tolist())==(-.0625,-.0625,.9375,.9375)
    assert tuple(p['cell_bounds'][-1].tolist())==(22.9375,16.9375,23.9375,17.9375)
    assert torch.equal(torch.bincount(p['owner']),torch.full((432,),64))
    assert torch.equal(torch.bincount(p['parent']),torch.full((108,),4))


def test_join_rejects_reordered_or_substituted_features():
    caches={i:{'features':np.full((432,128),i,np.float32)} for i in range(3)}
    pairs=[[0,1],[1,2]]
    q=np.stack([caches[i]['features'] for i,j in pairs]);k=np.stack([caches[j]['features'] for i,j in pairs])
    G.check_pair_join(q,k,pairs,caches)
    with pytest.raises(ValueError,match='order'):G.check_pair_join(q[::-1],k,pairs,caches)
    with pytest.raises(ValueError,match='duplicate'):G.check_pair_join(q,k,[[0,1],[0,1]],caches)


def test_geometry_rejects_gt_or_content_fields_and_reconfiguration():
    obs=observation()
    for name in ('gt_valid','q_to_k_in_front','q_features'):
        with pytest.raises(ValueError,match='only predicted'):G.compute_geometry({**obs,name:None},'full4')
    with pytest.raises(ValueError,match='configuration'):G.compute_geometry(obs,'full4',dict(method='full4',h=1.,temperature=5.))


def test_identity_point_matches_separable_normal_rectangle():
    torch.set_num_threads(4)
    from scipy.special import ndtr
    p=G.g8_partition('cpu');obs=observation()
    with torch.inference_mode():logit=G.compute_geometry(obs,'point_pck').numpy()
    mean=(p['bounds'].numpy()[:,:2]+3.5)/8;bounds=p['cell_bounds'].numpy()
    delta_lo=(bounds[None,:,:2]-mean[:,None,:])/.5
    delta_hi=(bounds[None,:,2:]-mean[:,None,:])/.5
    mass=(ndtr(delta_hi)-ndtr(delta_lo)).prod(-1)
    # Compare probabilities, including exact floor convention, away from CDF cancellation tails.
    expected=mass+1e-30
    np.testing.assert_allclose(np.exp(logit),expected,atol=2e-15,rtol=1e-10)
    assert logit.shape==(432,432) and np.isfinite(logit).all()


def test_common_query_aggregation_preserves_mass():
    p=torch.arange(432*432,dtype=torch.float64).reshape(432,432)+1
    p=p/p.sum(-1,keepdim=True)
    actual=G.common_g8(p).numpy()
    owners=G.g8_partition('cpu')['parent'].numpy()
    expected=np.stack([p.numpy()[owners==i].mean(0) for i in range(108)])
    np.testing.assert_allclose(actual,expected,atol=1e-16,rtol=0)
    np.testing.assert_allclose(actual.sum(-1),1,atol=1e-15,rtol=0)


def test_stage_output_refuses_overwrite(tmp_path):
    p=tmp_path/'record.json';G.write_new(p,{'ok':True})
    with pytest.raises(FileExistsError):G.write_new(p,{'ok':False})
    with pytest.raises(ValueError,match='overwrite'):G.freeze(tmp_path)
