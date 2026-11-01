"""Synthetic checks only: paired scan statistics, gates, identity and lock order."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments import round6_analysis as a

SCENES=['brown_a','brown_b','brown_c','mit_d','home_e']


def synthetic_rows():
    result={}
    for role,kl in [('full',.8),('fixed',1.),('point',1.2),('isotropic',.9),('point_pck',1.1)]:
        rows=[]
        for scene in SCENES:
            for grid in a.GRIDS:
                qs,ks=map(int,grid.replace('q','').replace('k','').split('_'))
                for pair,mass in [(0,1.),(1,9.)]:
                    row={'scene':scene,'grid':grid,'q_index':pair,'k_index':pair+10,
                        'pair_id':f'{pair}:{pair+10}','q_stride':qs,'k_stride':ks,
                        'n_valid':1,'visible_mass':mass,'entropy_sum':mass*.2,
                        'kl_sum':mass*kl,'ce_sum':mass*(kl+.2),'js_sum':mass*.05,
                        'row_kl_sum':kl,'n_point':1000,'pck8_count':950,'pck4_count':800,
                        'pck_stride_count':950,'point_ce_sum':100.,'point_epe_sum':200.,
                        'point_soft_epe_sum':300.}
                    for coverage in a.COVERAGE:
                        row[f'coverage_{coverage}_mass']=mass if coverage=='0.75_1.01' else 0.
                        row[f'coverage_{coverage}_kl']=mass*kl if coverage=='0.75_1.01' else 0.
                    rows.append(row)
        result[role]=rows
    return result


def test_frozen_gates_pass_analytic_constant_effect_and_report_pp_correctly():
    rows=synthetic_rows()
    for row in rows['full']:row['pck8_count']=946
    result=a.analyze_rows(rows,SCENES,bootstrap=1000)
    assert result['gates']['pass']
    effect=result['comparisons']['full_minus_fixed']['kl']['all']
    assert effect['difference']==pytest.approx(-.2)
    np.testing.assert_allclose(effect['ci95'],[-.2,-.2],atol=1e-14)
    assert effect['relative_difference']==pytest.approx(-.2)
    pck=result['comparisons']['full_minus_fixed']['pck8']
    assert pck['difference']==pytest.approx(-.004)
    assert pck['difference_pp']==pytest.approx(-.4)
    assert result['statistical_protocol']['training_seed_replicates']==0
    assert result['pair_count_per_scan']==dict.fromkeys(SCENES,2)


def test_pairing_cancels_between_scan_offsets_in_every_bootstrap_draw():
    baseline=np.array([1.,2.,4.,8.,16.])
    full=baseline-.1
    draws=np.random.default_rng(a.RNG_SEED).integers(0,5,(a.BOOTSTRAP,5))
    result=a.paired_scan_statistics(full,baseline,draws,SCENES)
    np.testing.assert_allclose(result['ci95'],[-.1,-.1],atol=1e-14)
    assert result['improved_scan_count']==5
    again=a.paired_scan_statistics(full,baseline,draws,SCENES)
    assert again==result


def test_crossscale_gate_cannot_hide_behind_better_total():
    rows=synthetic_rows()
    for row in rows['full']:
        if row['grid']=='q8_k16':row['kl_sum']=row['visible_mass']*1.05
    result=a.analyze_rows(rows,SCENES,bootstrap=200)
    gate=result['gates']
    assert gate['kl_at_least_2pct_and_ci_upper_below_zero']['all']
    assert not gate['kl_at_least_2pct_and_ci_upper_below_zero']['q8_k16']
    assert not gate['pass']


def test_same_scale_and_point_optimal_guards_are_separate_and_strict():
    rows=synthetic_rows()
    for row in rows['full']:
        if row['grid']=='q8_k8':row['kl_sum']=row['visible_mass']*1.03
    result=a.analyze_rows(rows,SCENES,bootstrap=200)
    assert not result['gates']['same_scale_relative_ci_upper_below_2pct']['q8_k8']
    rows=synthetic_rows()
    for row in rows['point_pck']:row['pck8_count']=960
    result=a.analyze_rows(rows,SCENES,bootstrap=200)
    assert result['gates']['q8_k8_pck8_ci_lower_above_minus_0_5pp']['fixed']
    assert not result['gates']['q8_k8_pck8_ci_lower_above_minus_0_5pp']['point_pck']
    assert not result['gates']['pass']


def test_macro_uses_pixel_weight_within_then_equal_scene_grid():
    rows=synthetic_rows()
    # One scene/grid has KL values 1 and3 with masses1 and9 =>2.8,
    # rather than a pair mean2. Other19 scene/grid cells remain1.
    for row in rows['fixed']:
        if row['scene']==SCENES[0] and row['grid']==a.GRIDS[0]:
            row['kl_sum']=row['visible_mass']*(1 if row['q_index']==0 else 3)
    result=a.analyze_rows(rows,SCENES,bootstrap=200)
    assert result['summaries']['fixed']['kl']['by_scan_grid'][SCENES[0]][a.GRIDS[0]]==pytest.approx(2.8)
    assert result['summaries']['fixed']['kl']['macro']==pytest.approx((19+2.8)/20)
    assert result['summaries']['fixed']['query_kl']['macro']==1.


def test_identity_alignment_shuffle_missing_duplicate_and_gt_mismatch():
    rows=synthetic_rows()
    rows['fixed'].reverse()
    assert a.analyze_rows(rows,SCENES,bootstrap=100)['gates']['pass']
    broken=copy.deepcopy(rows);broken['point'].pop()
    with pytest.raises(ValueError,match='identities'):a.analyze_rows(broken,SCENES,bootstrap=100)
    broken=copy.deepcopy(rows);broken['point'].append(broken['point'][0])
    with pytest.raises(ValueError,match='duplicate'):a.analyze_rows(broken,SCENES,bootstrap=100)
    broken=copy.deepcopy(rows);broken['fixed'][0]['n_point']+=1
    with pytest.raises(ValueError,match='counts/masks'):a.analyze_rows(broken,SCENES,bootstrap=100)
    broken=copy.deepcopy(rows);broken['fixed'][0]['entropy_sum']+=.1
    with pytest.raises(ValueError,match='entropy'):a.analyze_rows(broken,SCENES,bootstrap=100)


def test_zero_condition_is_not_silently_dropped_or_relative_floor_added():
    rows=synthetic_rows()
    for values in rows.values():
        for row in values:
            if row['scene']==SCENES[0] and row['grid']==a.GRIDS[0]:
                row['visible_mass']=row['n_valid']=0
                for field in ['entropy_sum','ce_sum','kl_sum','js_sum','row_kl_sum']:
                    row[field]=0.
                for c in a.COVERAGE:row[f'coverage_{c}_mass']=row[f'coverage_{c}_kl']=0.
    result=a.analyze_rows(rows,SCENES,bootstrap=100)
    assert result['summaries']['full']['kl']['macro'] is None
    assert not result['gates']['pass']
    assert not result['comparisons']['full_minus_fixed']['kl']['all']['evaluable']
    draw=np.zeros((10,5),int)
    undefined=a.paired_scan_statistics(np.zeros(5),np.zeros(5),draw,SCENES)
    assert undefined['relative_ci95'] is None and undefined['relative_difference'] is None


def test_optional_diagnostic_cannot_replace_full_or_break_primary_coverage():
    rows=synthetic_rows()
    rows['full_no_content']=copy.deepcopy(rows['full'])
    for row in rows['full_no_content']:row['kl_sum']=0
    result=a.analyze_rows(rows,SCENES,bootstrap=100)
    assert 'full_minus_full_no_content' in result['comparisons']
    assert result['gates']['candidate'].startswith('full only')
    rows['full_no_content'].pop()
    result=a.analyze_rows(rows,SCENES,bootstrap=100)
    assert result['gates']['pass']
    assert result['optional_diagnostics']['full_no_content'].startswith('not analyzed')


def test_group_and_leave_one_sensitivities_do_not_manufacture_three_group_ci():
    result=a.analyze_rows(synthetic_rows(),SCENES,bootstrap=100)
    difference=result['comparisons']['full_minus_fixed']['kl']['all']
    assert len(difference['leave_one_scan_out'])==5
    groups=difference['acquisition_group_sensitivity']
    assert set(groups)=={'Brown','MIT','home'}
    assert all(not any('ci' in key for key in group) for group in groups.values())


def lock_fixture(tmp_path):
    configs={role:{'config':{'method':'point' if role=='point_pck' else role,
        'bandwidth':.5,'temperature':5.},'validation_sha256':'frozen'} for role in a.ROLES}
    lock={'analysis_source_sha256':a.sha256(a.__file__),'expected_scenes':SCENES,
          'expected_grids':list(a.GRIDS),'area_baseline':'fixed','roles':configs}
    path=tmp_path/'lock.json';path.write_text(json.dumps(lock))
    return path,lock


def test_root_roles_schema_and_lock_checked_before_result_read(tmp_path):
    path,lock=lock_fixture(tmp_path)
    # Malformed result content proves an invalid source lock is rejected first.
    directory=tmp_path/'full';directory.mkdir();(directory/'test.json').write_text('not JSON')
    lock['analysis_source_sha256']='wrong';path.write_text(json.dumps(lock))
    with pytest.raises(ValueError,match='exact analysis source'):
        a.load_locked_results(tmp_path,path)


def test_locked_loader_configs_hash_contract_and_optional_absence(tmp_path):
    path,lock=lock_fixture(tmp_path)
    rows=synthetic_rows()
    for role in a.ROLES:
        directory=tmp_path/role;directory.mkdir()
        (directory/'test.json').write_text(json.dumps({'rows':rows[role],
            'config':lock['roles'][role]['config'],'lock_sha256':a.sha256(path),
            'input_sources':[{'scene':'same','grid':'q8_k8','input_sha256':'x','label_sha256':'y'}]}))
    loaded,normalized,provenance=a.load_locked_results(tmp_path,path)
    assert loaded==rows and set(normalized['method_configs'])==set(a.ROLES)
    assert provenance['shared_input_hash_contract']=='verified'
    payload=json.loads((tmp_path/'fixed/test.json').read_text())
    payload['config']['bandwidth']=2.
    (tmp_path/'fixed/test.json').write_text(json.dumps(payload))
    with pytest.raises(ValueError,match='config differs'):a.load_locked_results(tmp_path,path)


def test_cli_synthetic_roundtrip_uses_all_twenty_thousand_paired_draws(tmp_path):
    path,lock=lock_fixture(tmp_path)
    lock['expected_pairs_per_scene']=2
    path.write_text(json.dumps(lock))
    rows=synthetic_rows()
    for role in a.ROLES:
        directory=tmp_path/role;directory.mkdir()
        for row in rows[role]:row['method']=role
        (directory/'test.json').write_text(json.dumps({'rows':rows[role],
            'config':lock['roles'][role]['config'],'lock_sha256':a.sha256(path)}))
    output=tmp_path/'analysis'
    a.main(SimpleNamespace(results_root=tmp_path,lock=path,out=output))
    result=json.loads((output/'summary.json').read_text())
    assert result['gates']['pass']
    assert result['statistical_protocol']['bootstrap']==20000
    assert result['statistical_protocol']['rng_seed']==20261030
    assert a.sha256(output/'source_snapshot.py')==a.sha256(a.__file__)
