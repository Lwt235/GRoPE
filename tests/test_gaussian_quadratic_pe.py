import pytest
import torch

from models.gaussian_quadratic_pe import GaussianQuadraticMatcher,quadratic_features
from models.gaussian_measure_pe import gaussian_measure_log_bias
from models.round6_measure_attention import position_bias


def sample(dtype=torch.float64):
    torch.manual_seed(2)
    qm=torch.randn(2,7,2,dtype=dtype)*5
    km=torch.randn(2,9,2,dtype=dtype)*5
    a=torch.randn(2,7,2,2,dtype=dtype)
    qc=a@a.transpose(-1,-2)+torch.eye(2,dtype=dtype)*.1
    kc=torch.tensor([[.4,.03],[.03,.2]],dtype=dtype).expand(2,2,2)
    return qm,qc,km,kc


def test_five_channel_exact_probability_and_attention_with_common_anisotropic_keys():
    qm,qc,km,kc=sample();qg,kg=quadratic_features(qm,qc,km,kc,.3)
    logits=qg@kg.transpose(-1,-2)
    reference=gaussian_measure_log_bias(qm,qc,km,kc[:,None].expand(-1,9,-1,-1),
        key_mass=torch.ones(2,9,dtype=torch.float64),bandwidth=.3)
    p=logits.softmax(-1);r=reference.softmax(-1)
    torch.testing.assert_close(p,r,atol=1e-13,rtol=1e-12)
    values=torch.randn(2,9,11,dtype=torch.float64)
    torch.testing.assert_close(p@values,r@values,atol=1e-13,rtol=1e-12)


def test_position_unit_rescaling_and_translation():
    qm,qc,km,kc=sample()
    q,k=quadratic_features(qm,qc,km,kc,.3)
    a,b=quadratic_features(qm*8+200,qc*64,km*8+200,kc*64,.3*8)
    torch.testing.assert_close((q@k.transpose(-1,-2)).softmax(-1),
        (a@b.transpose(-1,-2)).softmax(-1),atol=1e-13,rtol=1e-12)


def test_all_controls_exact_with_content_and_no_learned_parameters():
    qm,qc,km,kc=sample(torch.float32)
    nativeq=torch.eye(2).expand(2,7,2,2)/12
    nativek=torch.eye(2).expand(2,9,2,2)/3
    d=dict(q_to_k=qm,q_to_kcov=qc,qcov=nativeq,kmean=km,kcov=nativek,
           q=torch.nn.functional.normalize(torch.randn(2,7,128),dim=-1),
           k=torch.nn.functional.normalize(torch.randn(2,9,128),dim=-1))
    for method in ('point','fixed','isotropic','full'):
        model=GaussianQuadraticMatcher(method,.5,5.)
        q,k=model.encode(d)
        assert q.shape[-1]==k.shape[-1]==133
        assert sum(p.numel() for p in model.parameters())==0
        direct=5*(d['q']@d['k'].transpose(-1,-2))+position_bias(d,method,.5)
        torch.testing.assert_close(model(d).softmax(-1),direct.softmax(-1),atol=2e-5,rtol=2e-4)
    unequal=dict(d,kcov=d['kcov'].clone());unequal['kcov'][:,0]*=2
    for method in ('point','fixed','isotropic','full'):
        with pytest.raises(ValueError,match='uniform'):
            GaussianQuadraticMatcher(method)(unequal)
