import numpy as np
import pytest
import torch

from experiments import round14_temperature_probe as P


def synthetic_supports():
    mean=torch.tensor([[.2,.3],[1.4,.2]],dtype=torch.float64)
    covariance=torch.tensor([[[.12,.02],[.02,.2]],[[.25,-.04],[-.04,.15]]],dtype=torch.float64)
    features=torch.eye(2,dtype=torch.float64)
    full=dict(mean=mean,cov=covariance,mass=torch.full((2,),64.,dtype=torch.float64),features=features)
    query={'full':full}
    for role,n in [('full4',4),('quad16',16)]:
        query[role]=dict(mean=mean,features=features,nodes=mean[:,None].expand(-1,n,-1),
            node_cov=covariance[:,None].expand(-1,n,-1,-1),
            node_weights=torch.full((2,n),1/n,dtype=torch.float64))
    native=np.array([[0,0,8,8],[8,0,16,8],[16,0,24,8]],float)
    key=dict(cell_bounds=torch.tensor((native-.5)/8),cell_owner=torch.arange(3),
        mass=torch.full((3,),64.,dtype=torch.float64),
        features=torch.tensor([[-1.,0.],[0.,1.],[1.,0.]],dtype=torch.float64))
    return query,key


def test_fixed_seven_roles_share_five_geometries_and_matched_reduction():
    query,key=synthetic_supports()
    with torch.no_grad():probability,diagnostics=P.condition_predictions(query,key)
    assert len(probability)==7 and len(diagnostics)==5
    assert set(probability)==set(P.CONFIGS)
    for t in ('t0','t5'):
        torch.testing.assert_close(probability['full4_'+t],probability['full1_'+t],atol=1.e-14,rtol=1.e-13)
    torch.testing.assert_close(probability['quad16'],probability['full4_t5'],atol=1.e-14,rtol=1.e-13)
    assert not torch.allclose(probability['full1_frozen'],probability['full1_t0'])


def test_zero_temperature_invariant_to_content_while_t5_responds():
    query,key=synthetic_supports()
    with torch.no_grad():first,_=P.condition_predictions(query,key)
    changed=dict(key,features=-key['features'])
    with torch.no_grad():second,_=P.condition_predictions(query,changed)
    for role,cfg in P.CONFIGS.items():
        if cfg['temperature']==0:
            torch.testing.assert_close(first[role],second[role],atol=0,rtol=0)
        else:assert not torch.allclose(first[role],second[role])


def test_metrics_independent_hand_calculation():
    p=np.array([[.25,.75]],np.float64);indices=np.array([1]);mean=np.array([[7.5,3.5]])
    keys=np.array([[3.5,3.5],[11.5,3.5]]);counts=np.array([[1,3]])
    row,correct=P.measure(p,indices,mean,keys,counts,np.array([[11.5,3.5]]),np.array([True]))
    assert row['kl_sum']==0 and row['mass']==4 and row['pck_count']==row['n_point']==1
    assert correct.tolist()==[True]
    assert row['winner_to_mean_sum_all']==4
    assert row['expected_to_mean_sum_all']==2
    assert row['within8_of_mean_probability_sum_all']==1
    assert row['predicted_mean_to_sensor_center_sum']==4
    with pytest.raises(ValueError,match='argmax'):
        P.measure(p,np.array([0]),mean,keys,counts,np.array([[11.5,3.5]]),np.array([True]))


def population_rows():
    entry=dict(scene=P.SCENES[0],pair=[1,2]);rows=[]
    for role in P.CONFIGS:
        for q,n in zip(P.QUERIES,(432,108,48)):
            rows.append(dict(scene=entry['scene'],pair=entry['pair'],query=q,role=role,mass=100,n_point=10,n_query=n))
    return [entry],rows


def test_population_guards_role_and_scale_changes():
    entries,rows=population_rows();P.verify_population(rows,entries)
    rows[0]['n_point']=9
    with pytest.raises(ValueError,match='population changed'):P.verify_population(rows,entries)
    entries,rows=population_rows()
    for r in rows:
        if r['query']=='g24':r['mass']=99
    with pytest.raises(ValueError,match='visible pixel mass'):P.verify_population(rows,entries)


def test_all_three_forward_shards_required_before_target_reader(tmp_path,monkeypatch):
    P.A.write(tmp_path/'lock.json',{})
    for scene in P.SCENES[:2]:
        d=tmp_path/'forward'/P.S.B.scene_slug(scene);d.mkdir(parents=True)
        P.S.complete_stage(d,tmp_path,'forward',scene,0.)
    monkeypatch.setattr(P,'verify_lock',lambda out:{})
    monkeypatch.setattr(P,'scene_entries',lambda lock,scene:[])
    def forbidden(path):raise AssertionError('a target array was opened before all forward shards')
    monkeypatch.setattr(P.A,'load_npz',forbidden)
    with pytest.raises(FileNotFoundError):P.evaluate(tmp_path,P.SCENES[0])
    d=tmp_path/'forward'/P.S.B.scene_slug(P.SCENES[2]);d.mkdir(parents=True)
    P.S.complete_stage(d,tmp_path,'forward',P.SCENES[2],0.)
    assert len(P.all_forward_complete(tmp_path))==3


def test_equal_scene_contrasts_do_not_pool_unequal_center_counts():
    rows=[];paired=[]
    for si,scene in enumerate(P.SCENES):
        for pair in range(16):
            for q in P.QUERIES:
                npoint=10*(si+1);nquery=432;mass=100*(si+1)
                for role in P.CONFIGS:
                    pck=(.8-.1*si) if role=='full4_t0' else .9 if role=='point_pck' else .6
                    kl=.11 if role=='full4_t0' else .10
                    rows.append(dict(scene=scene,pair=[pair,pair+15],query=q,role=role,
                        kl_sum=kl*mass,mass=mass,pck_count=round(pck*npoint),n_point=npoint,n_query=nquery,
                        winner_to_mean_sum_all=2*nquery,winner_to_mean_sum_center_valid=2*npoint,
                        expected_to_mean_sum_all=nquery,expected_to_mean_sum_center_valid=npoint,
                        within8_of_mean_probability_sum_all=.5*nquery,
                        within8_of_mean_probability_sum_center_valid=.5*npoint,predicted_mean_to_sensor_center_sum=3*npoint))
                for name,(left,right) in P.CONTRASTS.items():
                    paired.append(dict(scene=scene,pair=[pair,pair+15],query=q,contrast=name,left=left,right=right,
                        n_query=nquery,n_point=npoint,winner_changed_all=nquery//2,
                        winner_changed_center_valid=npoint//2,winner_displacement_sum_all=8*nquery,
                        left_correct_right_wrong=npoint//10,right_correct_left_wrong=0))
    summary=P.summarize(rows,paired)
    value=summary['contrasts']['full4_content_off']['g24']
    assert value['pck_difference_pp']==pytest.approx(10.)
    assert value['direct_kl_relative_change']==pytest.approx(.1)
    assert value['paired_equal_scene']['left_correct_right_wrong_fraction']==pytest.approx(.1)
    assert all(v['center_valid_winner_offset_difference_px']==0 for v in value['by_scene'].values())
    assert summary['full4_train_pck_gap']['g24']['train_gap_recovered_fraction']==pytest.approx(1/3)
