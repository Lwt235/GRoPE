"""Artifact-level closure of the fixed goal, without new numerical or efficacy gates."""
import time

from experiments.learned_projection_data import *


def main():
    p=plan();evidence={}
    assert len(p['training_catalog'])==len(p['catalog'])==12
    assert p['steps']==2000 and p['expected_fits']==9
    assert p['pe']['network']==[8,16,4] and p['primary_pairs']==[['G','P'],['G','S']]
    lock=read(OUT/'configuration_lock.json')
    assert lock['plan_sha256']==sha256(OUT/'plan.json')
    assert all(sha256(ROOT/f)==h for f,h in lock['source_sha256'].items())
    pre=read(OUT/'preflight.json')
    assert pre['complete'] and len(pre['records'])==6 and pre['pilot_weights_discarded']
    assert all(r['initial_native_max_abs']<1e-4 and len(r['steps'])==3 for r in pre['records'])
    for r in pre['records']:
        assert all(all(step['read_gradients'].get(str(i),0)>0 for i in p['read_layers']) for step in r['steps'])
        assert r['steps'][-1]['extra_column_gradient']>0 if r['kind']!='P' else all(s['extra_column_gradient']==0 for s in r['steps'])
    evidence['fixed_design_and_necessary_interface_checks']=['plan.json','configuration_lock.json','preflight.json']
    native=read(OUT/'native_identity.json');empty=read(OUT/'empty_reader_interface.json')
    assert native['complete'] and native['actual_model_sha256']==p['model_sha256']
    assert len(native['official_source_files'])==len(p['official_source_sha256'])
    assert all(sha256(f)==h for f,h in native['official_source_files'].items())
    assert empty['complete'] and empty['empty_residual_exact_zero'] and empty['positive_control_residual_max_abs']>0
    evidence['actual_native_identity_and_empty_trained_reader_interface']=['native_identity.json','empty_reader_interface.json']
    matched=read(OUT/'matched_checks.json')
    assert matched['complete'] and len(matched['records'])==9 and len(matched['source_fields'])==24
    assert all(r['actual_updates']==2000 and r['actual_samples_paired'] and r['all_three_initial_states_identical'] for r in matched['records'])
    assert matched['scientific_code_identity_unchanged'] and all(r['unchanged'] for r in matched['source_fields'])
    evidence['unchanged24_fields_and_nine_actual_matched_fits']=['matched_checks.json','geometry_complete.json']
    if (OUT/'memory_staging/validation.json').exists():
        staging=read(OUT/'memory_staging/validation.json');implementation=read(OUT/'memory_staging/implementation_record.json')
        assert staging['complete'] and len(staging['records'])==4 and staging['no_optimizer_update'] and staging['no_development_data_used']
        assert all(r['all_input_values_dtypes_shapes_strides_exact'] and r['max_abs_depth_difference']==0 for r in staging['records'])
        assert all(np.isfinite(r['max_abs_gradient_difference']) for r in staging['records'])
        assert staging['source_sha256']==sha256(ROOT/'experiments/learned_projection_stream_train.py')
        assert all(sha256(ROOT/f)==h for f,h in implementation['source_sha256'].items())
        evidence['resource_only_CPU_staging_identity_and_recorded_gradient_limit']=['memory_staging/validation.json','memory_staging/implementation_record.json']
    training=read(OUT/'training_summary.json')
    assert training['complete'] and len(training['completed'])==9 and not training['pending']
    assert all(r['updates']==2000 and r['all_logged_losses_and_gradient_norms_finite'] for r in training['completed'])
    evidence['training_and_preserved_resource_yield_cost']=['training_summary.json','resume_verification.json','resource_dispatch_update.json']
    audit=read(OUT/'scoring/independent_recomputation.json')
    assert audit['complete'] and audit['new_files']==9 and audit['new_predictions']==864
    summary=read(OUT/'scoring/summary.json')
    assert summary['complete'] and summary['plan_sha256']==sha256(OUT/'plan.json') and len(summary['places'])==12
    for endpoint in ['missing/roi/centered_log_mse','missing/roi/log_mse','full/roi/centered_log_mse','full/whole/absrel']:
        for pair in ['G-P','G-S','S-P','P-parent_L','S-parent_L','G-parent_L','G-old_fixed_G_sensor']:
            v=summary['comparisons'][endpoint+'/'+pair]
            assert v['paired_cells']==36 and len(v['per_seed_relative_percent'])==3 and len(v['per_place_relative_percent'])==12
    evidence['all864_predictions_and_fixed_primary_auxiliary_comparisons']=['scoring/independent_recomputation.json','scoring/summary.json','scoring/metrics.json','report_tables.md']
    cost=read(OUT/'cost.json')
    assert cost['complete'] and len(cost['queries'])==12 and {r['kind'] for r in cost['queries']}==set(KINDS)
    assert all(r['retained']==5 and len(r['seconds'])==5 and r['warmup']==2 for r in cost['queries'])
    assert read(OUT/'pe_description.json')['complete']
    evidence['full_fresh_query_cost_and_descriptive_PE_use']=['cost.json','pe_description.json']
    review=read(OUT/'figures/visual_review.json');manifest=read(OUT/'figures/manifest.json')
    assert review['complete'] and not manifest['visual_review_pending']
    assert set(review['actually_viewed'])==set(manifest['files'])=={'outcomes.png','depth_cases.png'}
    for f in manifest['files']:
        assert sha256(OUT/'figures'/f)==manifest['sha256'][f]==review['sha256'][f]
    evidence['actual_visual_review_of_both_final_figures']=['figures/manifest.json','figures/visual_review.json','figures/case_selection.json']
    decision=read(OUT/'method_decision.json')
    assert decision['complete'] and decision['plan_sha256']==sha256(OUT/'plan.json')
    assert decision['summary_sha256']==sha256(OUT/'scoring/summary.json')
    assert decision['report_sha256']==sha256(OUT/'report.md')
    assert decision['extra_source_sensor_diagnostic'] and not decision['new_confirmation'] and not decision['automatic_successor']
    assert decision['old_fixed_G_negative_results_preserved']
    docs=read(OUT/'documentation_sync.json')
    assert docs['complete'] and docs['outline_items']==['O2','O3','O4','O5','O6']
    required_docs={'docs/research_current.md','docs/research_outline_alignment.md','docs/research_core.md',
        'docs/gaussian_memory_evidence_scope.md','docs/research_autonomous_continuation.md',
        'docs/learned_projection_pe_execution_plan.md','docs/source_geometry_results_analysis_20261112.md'}
    assert required_docs.issubset(docs['files'])
    for f,h in docs['files'].items():assert sha256(ROOT/f)==h
    evidence['scoped_report_method_decision_and_synchronized_outline']=['report.md','method_decision.json','documentation_sync.json']
    queue=read(OUT/'queue_status.json');processes=read(OUT/'final_process_status.json')
    assert len(queue['complete'])==9 and not queue['running'] and not queue['pending'] and not queue['failed']
    assert processes['none_running']
    evidence['finite_work_finished_and_no_live_own_worker']=['queue_status.json','final_process_status.json']
    write_json(OUT/'completion_audit.json',dict(complete=True,checks=evidence,remaining_required_work=[],
        recorded_epoch=time.time(),plan_sha256=sha256(OUT/'plan.json'),
        scope='Verifies completion of the fixed diagnostic, not favorable outcomes or general efficacy. Source/sample checks and scoring implementations were inspected; no new efficacy threshold.'))
    print('Fixed diagnostic completion proven; formal goal tool update still requires the agent.',flush=True)


if __name__=='__main__':main()
