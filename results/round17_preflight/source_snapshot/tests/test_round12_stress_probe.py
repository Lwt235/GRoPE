import copy
import numpy as np
import pytest
import torch

from experiments import round12_stress_probe as P


def synthetic_labels(parts):
    xy=P.D.pixel_centers()
    labels={'gt_xy':xy,'gt_visible':np.ones(len(xy),bool)}
    for name,p in parts.items():
        labels[name+'_point_xy']=p['mean']*8
        labels[name+'_point_visible']=np.ones(len(p['mean']),bool)
    return labels


@pytest.mark.parametrize('qname,kname',P.M.SCALE_CONDITIONS)
def test_direct_scale_labels_conserve_all_pixels(qname,kname):
    parts=P.M.partitions();labels=synthetic_labels(parts)
    direct,common=P.target_counts(labels,parts,qname,kname)
    assert direct.sum()==common.sum()==192*144
    np.testing.assert_array_equal(direct.sum(-1),parts[qname]['mass'])
    p=direct/parts[qname]['mass'][:,None]
    row=P.measure(p,p.argmax(-1),labels,parts,qname,kname,'scale',direct,common)
    assert row['kl_sum']==0 and row['direct_kl_sum']==0
    assert row['mass']==192*144


def test_common_reconstruction_matches_frozen_torch_operator():
    rng=np.random.default_rng(1208);parts=P.M.partitions()
    for q,k in P.M.ORIGINAL_CONDITIONS:
        probability=rng.random((len(parts[q]['mass']),len(parts[k]['mass'])))
        probability/=probability.sum(-1,keepdims=True)
        common=P.common_probability(probability,parts[q],parts[k])
        expected=P.A.common_prediction(torch.tensor(probability),
            {name:torch.tensor(parts[q][name]) for name in ('parent','mass')},
            {'mass':torch.tensor(parts[k]['mass'])},torch.tensor(parts[k]['atoms']))
        np.testing.assert_allclose(common,expected.numpy(),atol=3.e-18,rtol=1.e-14)
        np.testing.assert_allclose(common.sum(-1),1.,atol=1.e-15)
    with pytest.raises(ValueError):P.common_probability(np.ones((48,432))/432,parts['g24'],parts['g8'])


def synthetic_summaries():
    output={}
    for scenario in ('scale',*P.STATES):
        conditions=P.M.SCALE_CONDITIONS if scenario=='scale' else P.M.ORIGINAL_CONDITIONS
        output[scenario]={role:dict(kl=1.,pck8=.8,by_scene={s:1. for s in P.SCENES},
            by_condition={q+'_'+k:1. for q,k in conditions}) for role in P.ROLES}
        output[scenario]['point']['kl']=1.1
        output[scenario]['point']['by_scene']={s:1.1 for s in P.SCENES}
    return output


def test_locked_gate_conjunction_rejects_scale_local_pck_and_stress_failures():
    baseline=synthetic_summaries()
    assert P.gates(baseline)['passed']
    for violation in ('scale_point','scale_condition','scale_pck','stress_kl','stress_pck'):
        summaries=copy.deepcopy(baseline)
        if violation=='scale_point':summaries['scale']['full4']['kl']=1.09
        elif violation=='scale_condition':summaries['scale']['full4']['by_condition']['g24_g8']=1.021
        elif violation=='scale_pck':summaries['scale']['full4']['pck8']=.794
        elif violation=='stress_kl':summaries['depth09']['full4']['kl']=1.021
        else:summaries['ry_plus1']['full4']['pck8']=.794
        assert not P.gates(summaries)['passed']


def test_saved_argmax_and_probability_integrity_guard():
    parts=P.M.partitions();labels=synthetic_labels(parts)
    direct,common=P.target_counts(labels,parts,'g24','g8')
    probability=direct/direct.sum(-1)[:,None]
    wrong=probability.argmax(-1)+1
    with pytest.raises(ValueError,match='argmax'):
        P.measure(probability,wrong,labels,parts,'g24','g8','scale',direct,common)


def test_stage_verification_detects_extra_and_changed_artifacts(tmp_path):
    P.A.write(tmp_path/'lock.json',{'x':1})
    directory=tmp_path/'forward';directory.mkdir();(directory/'data.txt').write_text('x')
    P.complete_stage(directory,tmp_path,'forward','scene',0.)
    assert P.verify_stage(directory,tmp_path,'forward','scene')['stage']=='forward'
    (directory/'extra.txt').write_text('extra')
    with pytest.raises(ValueError,match='extra'):P.verify_stage(directory,tmp_path,'forward','scene')
    (directory/'extra.txt').unlink();(directory/'data.txt').write_text('modified')
    with pytest.raises(ValueError,match='changed'):P.verify_stage(directory,tmp_path,'forward','scene')
