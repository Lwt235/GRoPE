import pytest
import torch
import torch.nn.functional as F
from models.round7_depth_attention import DepthAttention,depth_loss
from models.round8_oracle_attention import OracleDepthAttention,ROLES,replace_routing


def data():
    torch.manual_seed(21);torch.set_num_threads(1)
    qmean=torch.rand(2,3,2);kmean=torch.rand(2,4,2)
    return dict(q=F.normalize(torch.randn(2,3,128),dim=-1),k=F.normalize(torch.randn(2,4,128),dim=-1),
        qmean=qmean,kmean=kmean,qcov=torch.eye(2).expand(2,3,2,2)/12,kcov=torch.eye(2).expand(2,4,2,2)/12,
        q_to_k=qmean+.2,q_to_kcov=torch.eye(2).expand(2,3,2,2)/10,
        q_depth=torch.ones(2,3)*2,q_value_depth=torch.ones(2,3)*2,
        k_value_depth=torch.tensor([[1.,2.,3.,4.],[2.,3.,4.,5.]]),
        k_oracle_value_depth=torch.tensor([[1.1,2.1,3.1,4.1],[2.1,3.1,4.1,5.1]]),
        oracle_indices=torch.tensor([0,1,1,2]).expand(2,3,4),
        oracle_weights=torch.tensor([.4,.1,.2,.3]).expand(2,3,4),
        oracle_valid=torch.tensor([[True,False,True],[False,True,False]]),
        valid=torch.ones(2,3,dtype=torch.bool),target=torch.ones(2,3)*2.3,boundary=torch.zeros(2,3,dtype=torch.bool))


@pytest.mark.parametrize('role,old',[('pred_pred','full'),('singleview','singleview')])
def test_baseline_bitwise_equal_original(role,old):
    d=data();torch.manual_seed(11);a=OracleDepthAttention(role)
    torch.manual_seed(11);b=DepthAttention(old)
    for k,v in a.state_dict().items():torch.testing.assert_close(v,b.state_dict()[k],rtol=0,atol=0)
    torch.testing.assert_close(a(d),b(d),rtol=0,atol=0)
    altered=dict(d,oracle_valid=~d['oracle_valid'],k_oracle_value_depth=d['k_oracle_value_depth']*3)
    torch.testing.assert_close(a(altered),b(d),rtol=0,atol=0)


def test_duplicate_border_indices_conserve_route_and_fallback():
    d=data();pred=torch.softmax(torch.randn(2,3,4),-1)
    a=replace_routing(pred,d['oracle_indices'],d['oracle_weights'],d['oracle_valid'])
    torch.testing.assert_close(a.sum(-1),torch.ones(2,3))
    torch.testing.assert_close(a[d['oracle_valid']],torch.tensor([.4,.3,.3,0.]).expand(3,4))
    torch.testing.assert_close(a[~d['oracle_valid']],pred[~d['oracle_valid']],rtol=0,atol=0)


def test_neutral_oracle_exactly_matches_predicted_model():
    d=data();d['oracle_valid'].zero_();d['k_oracle_value_depth']=d['k_value_depth']
    torch.manual_seed(11);base=OracleDepthAttention('pred_pred')
    for role in ROLES[:-1]:
        m=OracleDepthAttention(role);m.load_state_dict(base.state_dict())
        torch.testing.assert_close(m(d),base(d),rtol=0,atol=0)


def test_label_and_diagnostic_fields_never_enter_forward():
    d=data()
    for role in ROLES:
        m=OracleDepthAttention(role);p=m(d)
        altered=dict(d,target=torch.full_like(d['target'],float('nan')),valid=~d['valid'],boundary=~d['boundary'],
            k_sensor_fraction=torch.full((2,4),float('nan')),direct_corresponding_pred_depth=torch.ones(2,3)*19,
            direct_corresponding_sensor_depth=torch.ones(2,3)*19,route_clipped=~d['oracle_valid'])
        torch.testing.assert_close(m(altered),p,rtol=0,atol=0)


def test_routing_and_value_interventions_are_separate():
    d=data();base=OracleDepthAttention('pred_pred');_,a,v=base(d,return_attention=True)
    for role in ('gt_route','gt_value','gt_both'):
        m=OracleDepthAttention(role);m.load_state_dict(base.state_dict());_,aa,vv=m(d,return_attention=True)
        if role=='gt_route':torch.testing.assert_close(vv,v,rtol=0,atol=0)
        if role=='gt_value':torch.testing.assert_close(aa,a,rtol=0,atol=0)
        if role!='gt_value':torch.testing.assert_close(aa[~d['oracle_valid']],a[~d['oracle_valid']],rtol=0,atol=0)


def test_gt_routing_blocks_qk_gradient_only_on_eligible_rows():
    d=data();m=OracleDepthAttention('gt_route')
    eligible=d['oracle_valid'];prediction=m(d);depth_loss(prediction,d['target'],eligible).backward()
    assert m.qk.weight.grad.abs().sum()==0
    assert m.value.weight.grad.abs().sum()>0 and m.head[-1].weight.grad.abs().sum()>0
    m.zero_grad(set_to_none=True);d['target'][0,0]=float('nan')
    depth_loss(m(d),d['target'],~eligible).backward()
    assert torch.isfinite(m.qk.weight.grad).all() and m.qk.weight.grad.abs().sum()>0


def test_same_stored_parameter_capacity():
    assert {sum(p.numel() for p in OracleDepthAttention(r).parameters()) for r in ROLES}=={41345}
