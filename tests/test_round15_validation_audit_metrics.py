"""Independent/producer agreement on synthetic full populations, including failures."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import random

import pytest

from experiments import round15_validation_audit_metrics as A
from experiments import round15_validation_metrics as P
from experiments.round15_regional_audit import compare_tree

spec=importlib.util.spec_from_file_location('validation_synthetic_fixtures',Path(__file__).with_name('test_round15_validation_metrics.py'))
fixtures=importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)


@pytest.fixture(scope='module')
def case():
    roles=fixtures.role_manifest();entries=fixtures.cohort()
    rows=fixtures.rows_for(roles,entries)
    timing_entries,timing_rows=fixtures.timing_for(roles)
    return roles,entries,rows,timing_entries,timing_rows


def paired(rows,entries,roles,timing_rows,timing_entries):
    a=A.summarize(rows,entries,roles);p=P.summarize(rows,entries,roles)
    assert compare_tree(a,p,tolerance=0)==0
    at=A.aggregate_timing(timing_rows,timing_entries,roles);pt=P.aggregate_timing(timing_rows,timing_entries,roles)
    assert compare_tree(at,pt,tolerance=0)==0
    ag=A.gates(a,roles,at);pg=P.gates(p,roles,pt)
    assert compare_tree(ag,pg,tolerance=0)==0
    return ag


def test_complete_population_and_pass_match_exactly(case):
    roles,entries,rows,te,tr=case
    assert paired(rows,entries,roles,tr,te)['passed']


def test_random_unequal_scene_mass_and_shuffled_rows_match(case):
    roles,entries,original,te,tr=case
    rng=random.Random(73);rows=deepcopy(original)
    populations={(e['scene'],tuple(e['pair'])):rng.randrange(1,27649) for e in entries}
    for r in rows:
        r['mass']=populations[r['scene'],tuple(r['pair'])]
        factor=rng.uniform(.05,3)
        for field in A.FIELDS:r[field]=r['mass']*factor
        r['pck_count']=rng.randrange(r['n_point']+1)
    rng.shuffle(rows)
    paired(rows,entries,roles,tr,te)


@pytest.mark.parametrize('remove,duplicate',[(True,False),(False,True)])
def test_incomplete_or_duplicate_population_is_rejected_by_both(case,remove,duplicate):
    roles,entries,rows,_,_=case
    altered=list(rows[:-1] if remove else rows)
    if duplicate:altered[-1]=altered[0]
    for module in (A,P):
        with pytest.raises(ValueError):module.summarize(altered,entries,roles)


@pytest.mark.parametrize('mutation',[
    lambda r:r.update(mass=999),
    lambda r:r.update(n_point=24),
    lambda r:r.update(pck_count=26),
    lambda r:r.update(pair=[False,1]),
    lambda r:r['numerical'].update(all_finite=False),
    lambda r:r['numerical'].update(max_normalization_error=1.1e-12),
])
def test_corrupt_counts_or_numerical_evidence_rejected(case,mutation):
    roles,entries,rows,_,_=case
    changed=list(rows);changed[0]=deepcopy(changed[0]);mutation(changed[0])
    for module in (A,P):
        with pytest.raises(ValueError):module.summarize(changed,entries,roles)


def test_fraction_boundary_and_one_extra_error_are_not_rounded(case):
    roles,entries,original,te,tr=case
    rows=deepcopy(original);primary=roles['aliases']['primary']
    selected=[]
    for r in rows:
        if r['operator']==primary and r['scenario']=='zero' and r['condition']=='g16_g8' and r['pair']==[0,1]:
            r['pck_count']=18;selected.append(r)
    assert len(selected)==2
    exact=paired(rows,entries,roles,tr,te)
    assert exact['new_q16_protection']['passed']
    selected[0]['pck_count']=17
    failed=paired(rows,entries,roles,tr,te)
    assert failed['new_q16_protection']['passed'] is False and failed['passed'] is False


def test_legacy_quad_can_fail_independently_of_new_quad(case):
    roles,entries,original,te,tr=case
    rows=deepcopy(original);legacy=roles['aliases']['legacy_quad16']
    assert legacy!=roles['aliases'][roles['strong_quad']]
    for r in rows:
        if r['operator']==legacy:
            r['kl_sum']=.7*r['mass'];r['direct_kl_sum']=.7*r['mass']
    gates=paired(rows,entries,roles,tr,te)
    assert gates['original']['accuracy_cost']['comparisons'][roles['strong_quad']]['passed']
    assert not gates['original']['accuracy_cost']['comparisons']['legacy_quad16']['passed']
    assert not gates['passed']


def test_timing_requires_all_cells_and_15_actual_samples(case):
    roles,_,_,te,tr=case
    for bad in (tr[:-1],[dict(tr[0],samples_seconds=[1.]*14),*tr[1:]]):
        for module in (A,P):
            with pytest.raises(ValueError):module.aggregate_timing(bad,te,roles)
