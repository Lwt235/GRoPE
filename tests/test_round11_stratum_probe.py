"""Round11 gate contract stays identical to the frozen A/B thresholds."""
import copy

import pytest

from experiments import round10_partition_probe as A
from experiments import round11_stratum_probe as R


def fixture():
    condition={q+'_'+k:1. for q in A.PARTITIONS for k in A.PARTITIONS}
    base=dict(kl=1.,pck8=.9,by_scene={'a':1.,'b':1.},by_condition=condition)
    baselines={name:copy.deepcopy(base) for name in (*A.METHODS,'point_pck')}
    candidate=copy.deepcopy(base);candidate.update(kl=.97,pck8=.897)
    candidate['by_scene']={'a':.96,'b':.98}
    candidate['by_condition']={name:.97 for name in condition}
    selection=dict(strong_point='point',strong_quadrature='quad16')
    reference=dict(summary=dict(summary=baselines),selection=selection)
    timings=[dict(method=name,total_ms=.8 if name=='full4' else 1.) for name in R.TIMING_ROLES]
    return candidate,reference,timings


def test_gate_checks_match_frozen_A_after_primary_alias():
    candidate,reference,timing=fixture()
    result=R.gates(dict(full4=candidate,iso4=candidate,iso4_matched=candidate),reference,timing)
    old=copy.deepcopy(reference['summary']['summary']);old['full']=candidate
    oldtimes=[dict(method=name,total_ms=.8 if name=='full' else 1.) for name in A.METHODS]
    expected=A.gates(old,reference['selection'],oldtimes)
    assert result['checks']==expected['checks']
    assert result['passed']==expected['passed']
    assert result['pck_difference_pp']==expected['pck_difference_pp']
    assert result['partition_gains']==expected['partition_gains']


def test_partition_or_pck_failure_cannot_be_rescued_by_average_gain():
    candidate,reference,timing=fixture()
    candidate['by_condition']['g16_g8']=1.03
    candidate['pck8']=.894
    result=R.gates(dict(full4=candidate),reference,timing)
    assert not result['passed']
    assert not result['checks']['no_partition_worse_than_quadrature_by_2pct']
    assert not result['checks']['pck_loss_no_more_than_half_pp']


def test_calibration_and_eval_roles_are_predeclared():
    configs=R.configurations()
    assert len(configs)==24
    assert R.METHODS==('full4','iso4')
    assert R.EVAL_ROLES==('full4','iso4','iso4_matched')
    assert R.TIMING_ROLES==('full4','iso4','quad16','full1')
    assert list(configs)[0]=='full4_h0.25_t0.0'


def test_paired_population_join_rejects_changed_supervision_or_duplicates():
    row=dict(scene='scene',pair=[1,2],condition='g8_g8',mass=123.,n_point=10)
    R.verify_paired_populations({'full4':[row]},{'point':[row],'full':[row]})
    changed=dict(row,mass=124.)
    with pytest.raises(ValueError,match='population mismatch'):
        R.verify_paired_populations({'full4':[changed]},{'point':[row]})
    with pytest.raises(ValueError,match='duplicate'):
        R.verify_paired_populations({'full4':[row,row]},{'point':[row]})
