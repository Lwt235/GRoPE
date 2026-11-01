"""R29 log-arithmetic, failure retention and CLI toys; no formal integration."""
import ast
import math
from pathlib import Path
import numpy as np
import pytest
from scipy.special import logsumexp

from experiments import round29_log_rms_audit as A


def native():
    return np.asarray([[-.8,-.4,.2,.7],[.2,-.4,1.2,.7]],np.float64)


def test_log_raw_gates_reject_floor_hidden_error_below_float_range():
    reference=np.asarray([-1200.,-1201.]);actual=reference+math.log(1.01)
    check=A.compare_log_raw(actual,reference,native(),np.zeros(2),1e-30,A.C.GATES)
    assert not check['passed'] and check['maximum_probability_error']==0.
    assert math.isclose(check['raw_fov_relative'],.01,rel_tol=1e-10)
    assert math.isclose(check['raw_l1_over_reference_fov'],.01,rel_tol=1e-10)
    assert check['raw_fov_actual_materialized_diagnostic']==0.


def test_log_l1_detects_rearrangement_even_when_total_mass_matches():
    reference=np.log(np.asarray([.2,.3]));actual=np.log(np.asarray([.3,.2]))
    check=A.compare_log_raw(actual,reference,native(),np.zeros(2),1e-30,A.C.GATES)
    assert not check['passed'] and check['raw_fov_relative']<1e-14
    assert math.isclose(check['raw_l1_over_reference_fov'],.4,rel_tol=1e-14)


def test_comparison_uses_archived_probability_and_original_threshold():
    raw=np.log(np.asarray([.2,.3]));p=A.REF.probabilities(raw,native(),np.zeros(2),1e-30)['probability'];p=p.copy();p[0]+=.001
    assert not A.compare_log_raw(raw,raw,native(),np.zeros(2),1e-30,A.C.GATES,archived_probability=p)['passed']


def test_structural_zero_is_separate_from_unrepresentable_positive_mass():
    zero=np.full(2,-np.inf)
    check=A.compare_log_raw(zero,zero,native(),np.zeros(2),1e-30,A.C.GATES,structural_zero=True)
    assert check['passed'] and check['raw_fov_relative'] is None and not check['raw_relative_applicable']
    with pytest.raises(ValueError):A.compare_log_raw(zero,zero,native(),np.zeros(2),1e-30,A.C.GATES)
    with pytest.raises(ValueError):A.validate_logs(np.asarray([-1000.,-np.inf]),False)


def test_logdiff_and_direct_fov_absolute_gate_keep_original_units():
    np.testing.assert_array_equal(A.log_absolute_difference(np.asarray([-np.inf,-4.]),np.asarray([-np.inf,-np.inf])),np.asarray([-np.inf,-4.]))
    raw=np.asarray([-1200.,-1201.]);direct=np.asarray([float(logsumexp(raw))+1.])
    check=A.direct_fov_check(raw,direct,False,1e-8)
    assert check['passed'] and check['absolute_error_materialized_diagnostic']==0.
    assert math.isfinite(check['log_absolute_error'])
    assert not A.direct_fov_check(np.log(np.asarray([.2,.3])),np.log(np.asarray([.6])),False,1e-8)['passed']


def test_ordinary_materialization_distinguishes_positive_subfloat_and_true_zero():
    logs=np.asarray([-1200.,-710.,math.log(.2)])
    ordinary=np.exp(logs);indices=np.asarray([1,2],np.int64)
    bank=dict(log_raw_mass=logs,materialized_raw_indices=indices,materialized_raw_mass=ordinary[indices])
    check=A.materialization_check(bank,False)
    assert check==dict(ordinary_raw_unmaterialized_count=1,ordinary_raw_subnormal_count=1)
    bad=dict(bank,materialized_raw_indices=np.arange(3,dtype=np.int64),materialized_raw_mass=ordinary)
    with pytest.raises(ValueError):A.materialization_check(bad,False)
    zero=dict(log_raw_mass=np.full(2,-np.inf),materialized_raw_indices=np.arange(2,dtype=np.int64),materialized_raw_mass=np.zeros(2))
    assert A.materialization_check(zero,True)['ordinary_raw_unmaterialized_count']==0


def test_overflowed_ordinary_materialization_is_rejected():
    with pytest.raises(ValueError,match='overflow'):
        A.materialization_check(dict(log_raw_mass=np.asarray([1000.]),materialized_raw_indices=np.asarray([0],np.int64),materialized_raw_mass=np.asarray([np.inf])),False)


def test_merge_identity_combines_canonical_raw_and_floor_without_materializing():
    parent=dict(log_raw_mass=np.asarray([-1000.,-1001.]),log_floored_mass=np.log(np.asarray([1e-30,2e-30])),probability=np.asarray([1/3,2/3]))
    actual={k:np.asarray([float(logsumexp(v))]) if k!='probability' else np.asarray([math.fsum(v)]) for k,v in parent.items()}
    fixture=dict(id='toy_merge',identity=dict(parent='toy_parent',relation='merge',groups=[[0,1]]))
    assert A.identity_check(fixture,actual,parent,A.C.GATES)['passed']


def partial_state():
    bounds=native();counts=np.asarray([2,0],np.int64)
    bank=dict(input__mu=np.asarray([0.,0.,2.]),input__factor=np.zeros((3,3)),input__bounds=bounds,
        partial_log_raw_mass=np.asarray([-2.,-np.inf]),partial_log_raw_fov_mass=np.asarray([-np.inf]),
        partial_completed_cells=np.zeros(2,bool),partial_completed_fov=np.zeros(1,bool),
        partial_accumulated_tiles=counts,partial_accumulated_fov_tiles=np.zeros(1,np.int64))
    row=dict(structural_zero=False,work=dict(cell_node_evaluations=1152,native_tiles_completed=2,fov_tiles_completed=0))
    return row,bank


def test_partial_finite_mass_is_not_an_implicitly_completed_cell():
    row,bank=partial_state();check=A.partial_check(row,bank,A.C.POLICY)
    assert check['completed_native_cells']==0 and check['incomplete_finite_native_cells']==1
    bank['partial_completed_cells'][0]=True
    with pytest.raises(ValueError,match='completion'):A.partial_check(row,bank,A.C.POLICY)


def test_partial_tile_count_and_finite_sum_population_must_match():
    row,bank=partial_state();bank['partial_log_raw_mass'][1]=-5.
    with pytest.raises(ValueError,match='accumulated'):A.partial_check(row,bank,A.C.POLICY)


def test_local_lse_bound_only_counts_lost_finite_relative_terms():
    terms=np.asarray([[0.,-1000.,-np.inf],[0.,-800.,-801.]])
    report=A.local_lse_diagnostic(terms)
    assert report['sum_count']==2 and report['underflow_sum_count']==2 and report['underflow_term_count']==3
    assert math.isclose(report['maximum_log_scaled_omitted_bound'],math.log(2)-800,abs_tol=1e-13)
    assert report['maximum_log_relative_omitted_bound']==report['maximum_log_scaled_omitted_bound']


def test_cli_guard_is_after_all_runtime_functions_and_no_production_import():
    tree=ast.parse(Path(A.__file__).read_text());last=tree.body[-1]
    assert isinstance(last,ast.If) and '__name__' in ast.unparse(last.test)
    imports=[n.module or '' for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
    assert not any(n.startswith('models') or 'preflight' in n for n in imports)


def test_freeze_refuses_formal_attempt_before_metadata_or_arrays(tmp_path,monkeypatch):
    run=tmp_path/'run';run.mkdir();(run/'records.json').write_text('[]')
    monkeypatch.setattr(A,'verify_producer_lock',lambda *a:pytest.fail('must stop before metadata'))
    with pytest.raises(ValueError):A.freeze(run,tmp_path/'audit')


def test_complete_saved_structural_zero_path_checks_log_floor_and_metadata():
    bounds=native();logs=np.full(2,-np.inf);bias=np.zeros(2)
    normalized=A.REF.probabilities(logs,bounds,bias,1e-30)
    bank=dict(input__mu=np.zeros(3),input__factor=np.zeros((3,3)),input__bounds=bounds,input__content_bias=bias,
        log_raw_mass=logs,log_raw_fov_mass=np.asarray([-np.inf]),
        log_floored_mass=normalized['log_floored_mass'],log_probability=normalized['log_probability'],probability=normalized['probability'],
        materialized_raw_indices=np.arange(2,dtype=np.int64),materialized_raw_mass=np.zeros(2))
    report=dict(by_scope={name:dict(sum_count=0,underflow_sum_count=0,underflow_term_count=0,
        maximum_log_scaled_omitted_bound=None,maximum_log_absolute_omitted_bound=None,maximum_log_relative_omitted_bound=None) for name in A.LSE_SCOPES},
        exact_arithmetic_local_omission_only=True,is_pipeline_error_bound=False,excludes_addition_absorption_density_quadrature_and_roundoff=True)
    logfloor=normalized['log_floor_mass']
    for name,terms in dict(native_total=logs,floor=np.column_stack((logs,logfloor)),floor_total=logfloor,
                          floored_total=logfloor,content_normalization=logfloor).items():
        report['by_scope'][name]=A.local_lse_diagnostic(terms)
    row=dict(passed=True,structural_branch='zero_depth_rms',structural_zero=True,rms_depth=0.,normalized_depth_mean=None,normalized_depth_std=None,
        front_mass=0.,quadrature_order=24,maximum_tile_side_in_h=2.,h=.25,floor_density=1e-30,projection_regularization='rms_depth_xy_gaussian',
        canonical_raw_representation='log_mass',ordinary_materialization_is_canonical=False,tail_truncation_used=False,
        quadrature_error_bound_available=False,exact_arithmetic_gl_bound_available=True,native_tile_count=0,fov_tile_count=0,
        work=dict(cell_node_evaluations=0,native_tiles_completed=0,fov_tiles_completed=0),
        analytic_truncation_bound=dict(exact_arithmetic_only=True,excludes_density_evaluation_and_roundoff=True,is_relative_error_certificate=False,
            raw_native_l1_absolute=0.,raw_direct_fov_absolute=0.),
        density_diagnostics=dict(constant_depth_nodes=0,positive_branch_nodes=0,negative_branch_nodes=0,negative_erfcx_nodes=0,negative_series_nodes=0,
            maximum_log_density=None,minimum_log_density=None,maximum_log_negative_tail_relative_remainder=None,
            minimum_scaled_negative_tail_series_lower_bound=None,maximum_scaled_negative_tail_series_upper_bound=None),
        lse_diagnostics=report,log_raw_mass_sum=None,log_total_area_floor=float(logsumexp(logfloor)),
        log_normalizer=float(logsumexp(logfloor)),log_content_normalizer=normalized['log_content_normalizer'],
        ordinary_raw_unmaterialized_count=0,ordinary_raw_subnormal_count=0,zero_raw_cells=2,all_raw_mass_zero=True,area_prior_used=True,
        raw_mass_below_total_floor=True,argmax_index=0,argmax_exact_ties=2,argmax_near_ties_at_1e_12=2)
    assert A.production_check(row,bank,A.C.POLICY,A.C.GATES)['passed']
    wrong=dict(bank,log_floored_mass=bank['log_floored_mass']+math.log(2.))
    assert not A.production_check(row,wrong,A.C.POLICY,A.C.GATES)['passed']


def test_all43_reference_outcomes_survive_one_charged_reference_failure(tmp_path,monkeypatch):
    run,out=tmp_path/'run',tmp_path/'audit';run.mkdir();out.mkdir()
    for name in ('lock.json','completion.json','forward_complete.json'):A.write(run/name,{'toy':True})
    A.write(out/'lock.json',{'toy':True})
    ids=list(A.FIX.OLD_BASE_IDS)+list(A.FIX.ANALYTIC_BASE_IDS);identities=list(A.FIX.IDENTITY_IDS)
    bounds=native();logs=np.log(np.asarray([.2,.3]));bias=np.zeros(2)
    bank=dict(input__mu=np.asarray([0.,0.,2.]),input__factor=np.zeros((3,3)),input__bounds=bounds,
        input__content_bias=bias,log_raw_mass=logs,probability=A.REF.probabilities(logs,bounds,bias,1e-30)['probability'])
    fixtures=[dict(id=n,kind='base',policy_overrides={}) for n in ids]
    fixtures += [dict(id=n,kind='identity',identity=dict(parent=ids[0],relation='equal'),policy_overrides={}) for n in identities]
    rows=[dict(id=f['id'],passed=True) for f in fixtures];banks={f['id']:dict(bank) for f in fixtures}
    lock=dict(policy=A.C.POLICY,reference_policy=A.C.REFERENCE_POLICY,gates=A.C.GATES,base_ids=ids,identity_ids=identities,source_hashes={})
    seal=dict(finished_unix=0.,all_production_calls_completed=True,phase_wall_passed=True,phase_wall_seconds=0.)
    monkeypatch.setattr(A,'verify_seal',lambda *args:(lock,seal,rows,fixtures,banks))
    monkeypatch.setattr(A,'verify_joint_lock',lambda *args:None)
    monkeypatch.setattr(A,'production_check',lambda *args:dict(passed=True))
    monkeypatch.setattr(A,'reference_check',lambda *args:dict(passed=True))
    monkeypatch.setattr(A,'analytic_check',lambda *args:dict(passed=True))
    monkeypatch.setattr(A,'identity_check',lambda f,*args:dict(id=f['id'],passed=True))
    monkeypatch.setattr(A,'near_constant_check',lambda *args:dict(passed=True))
    seen=[]
    def numerical(*args,budget,**kwargs):
        budget.charge(1);seen.append(kwargs['order']);work={'cell_node_evaluations':1}
        if len(seen)==1:raise A.REF.ReferenceFailure(ValueError('synthetic charged failure'),work,{'partial_raw_mass':np.asarray([.1,0.])})
        return dict(log_raw_mass=logs,work=work)
    monkeypatch.setattr(A.REF,'integrate_old',numerical)
    monkeypatch.setattr(A.REF,'integrate_analytic',lambda *a,**k:dict(log_raw_mass=logs,work=dict(cell_node_evaluations=0,analytic_cdf_evaluations=12)))
    A.audit(run,out)
    records=A.read(out/'reference_records.json');summary=A.read(out/'summary.json')
    assert len(records)==43 and len(seen)==40 and summary['analytic_cdf_evaluations']==36
    assert summary['reference_cell_node_evaluations']==40 and summary['reference_call_failure_count']==1
    assert not summary['numerical_preflight_passed'] and records[0]['cumulative_cell_node_evaluations']==1
    assert records[0]['available_array_sha256'] and records[-1]['reference_kind']=='analytic'
    done=A.read(out/'completion.json');A.C.verify_inventory(out,done['files'],exclude=('completion.json',))
    assert done['verification_sha256']==A.sha(out/'verification.json')
