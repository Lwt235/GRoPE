import torch

from models.round6_measure_attention import Round6MeasureMatcher, position_bias, METHODS


def inputs():
    torch.manual_seed(82)
    q=torch.randn(2,5,128);k=torch.randn(2,3,128)
    qm=torch.randn(2,5,2);km=torch.randn(2,3,2)
    qc=torch.eye(2).expand(2,5,2,2)/12
    kc=torch.eye(2).expand(2,3,2,2)/3
    aq=torch.randn(2,5,2,2);ak=torch.randn(2,3,2,2)
    return dict(q=q,k=k,qmean=qm,kmean=km,qcov=qc,kcov=kc,
        q_to_k=qm+.2,k_to_q=km-.2,q_to_kcov=aq@aq.transpose(-1,-2)+qc,
        k_to_qcov=ak@ak.transpose(-1,-2)+kc)


def test_variable_token_count_equal_parameters_actual_value_and_geometry_gradients():
    torch.set_num_threads(1)
    d=inputs();counts=[]
    for method in METHODS:
        model=Round6MeasureMatcher(method)
        counts.append(sum(p.numel() for p in model.parameters()))
        score=model(d)
        assert score.shape==(2,5,3) and torch.isfinite(score).all()
        loss=-score.log_softmax(-1)[...,0].mean()
        loss.backward()
        assert torch.isfinite(model.log_bandwidth.grad) and model.log_bandwidth.grad.abs()>0
        for layer in model.layers:
            assert layer.qkv.weight.grad[192:].abs().sum()>0
        assert not torch.allclose(model(d),model(d,zero_values=True))
    assert len(set(counts))==1


def test_fixed_known_size_and_point_controls_ignore_projected_covariance():
    d=inputs();changed=dict(d,q_to_kcov=d['q_to_kcov']*30)
    for method in ('point','fixed'):
        torch.testing.assert_close(position_bias(d,method,.7),position_bias(changed,method,.7))
    assert not torch.allclose(position_bias(d,'full',.7),position_bias(changed,'full',.7))
    sizes=dict(d,qcov=d['qcov']*4)
    assert not torch.allclose(position_bias(d,'fixed',.7),position_bias(sizes,'fixed',.7))


def test_native_key_constant_mass_and_determinant_are_row_constants():
    d=inputs();b=position_bias(d,'full',.7)
    s=d['q_to_kcov']+d['kcov'][:,:1]+.7**2*torch.eye(2)
    delta=d['q_to_k'][:,:,None]-d['kmean'][:,None]
    quad=torch.einsum('bqki,bqij,bqkj->bqk',delta,torch.linalg.inv(s),delta)
    torch.testing.assert_close(b.softmax(-1),(-.5*quad).softmax(-1),atol=2e-6,rtol=2e-6)


def test_forward_does_not_read_extra_supervision_fields():
    model=Round6MeasureMatcher('full');d=inputs()
    a=model(d)
    d.update(gt_valid=torch.zeros(2,5,dtype=torch.bool),soft_labels=torch.randn(2,5,3),
             target_depth=torch.tensor(float('nan')))
    torch.testing.assert_close(a,model(d),rtol=0,atol=0)
