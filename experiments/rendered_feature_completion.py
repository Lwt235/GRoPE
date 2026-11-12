"""Audit finite deliverables and scope; no method-efficacy threshold is imposed."""
from experiments.rendered_feature_data import *


def main():
    p=plan();requirements=[]
    def proven(name,evidence):
        requirements.append(dict(requirement=name,status='proven',evidence=evidence))
    assert p['expected_fits']==9 and p['steps']==2000
    assert p['expected_adapted_predictions']==864 and p['expected_intervention_predictions']==288
    assert p['choices_changed']==['C3'] and not p['new_confirmation']
    proven('Original finite design and capability boundaries', ['plan.json','authorization.json'])
    pre=read(OUT/'preflight.json');fields=read(OUT/'features_complete.json')
    assert pre['complete'] and len(pre['records'])==6 and pre['pilot_weights_discarded']
    assert {r['kind'] for r in pre['records']}==set(KINDS)
    assert fields['complete'] and fields['fields']==24 and fields['queries_per_field']==4
    assert fields['source_fields_not_rebuilt'] and fields['content_not_rebound']
    assert fields['plan_sha256']==sha256(OUT/'plan.json')
    matched=read(OUT/'matched_checks.json');timing=read(OUT/'training_summary.json')
    assert matched['complete'] and len(matched['records'])==9 and len(matched['source_fields'])==24
    assert {(r['kind'],r['seed']) for r in matched['records']}=={(k,s) for s in SEEDS for k in KINDS}
    assert all(r['actual_updates']==2000 and r['parameters']==787712 and r['actual_samples_paired']
        and r['initial_state_exactly_matched'] for r in matched['records'])
    assert timing['complete'] and len(timing['completed'])==9 and not timing['pending']
    lock=read(OUT/'configuration_lock.json')
    assert lock['plan_sha256']==sha256(OUT/'plan.json')
    assert all(sha256(ROOT/name)==digest for name,digest in lock['source_sha256'].items())
    for meta in p['training_catalog']+p['catalog']:
        record=read(OUT/'features'/f'{meta["id"]}.json')
        assert record['source_only_saved_state_reused'] and record['source_content_unchanged']
        assert sha256(record['source_path'])==record['source_sha256']
        assert all(sha256(path)==digest for path,digest in record['files'].items())
    proven('Same24saved scenes, correct native interface, nine matched2000-step fits',
        ['preflight.json','features_complete.json','matched_checks.json','training_summary.json','configuration_lock.json'])
    audit=read(OUT/'scoring/independent_recomputation.json');summary=read(OUT/'scoring/summary.json')
    assert audit['complete'] and audit['new_files']==12 and audit['new_predictions']==1152
    assert len(audit['files'])==12 and all(r['predictions']==96 for r in audit['files'])
    assert sum(r['predictions'] for r in audit['files'] if not r['intervention'])==864
    assert sum(r['predictions'] for r in audit['files'] if r['intervention'])==288
    for r in audit['files']:
        marker=read(r['marker'])
        assert sha256(marker['prediction_path'])==r['prediction_sha256']
    assert summary['complete'] and summary['plan_sha256']==sha256(OUT/'plan.json')
    assert summary['main_endpoint']=='missing/roi/centered_log_mse'
    assert (OUT/'report_tables.md').is_file() and (OUT/'report_tables.md').stat().st_size>0
    for pair in ['Render-Local','Render-Global','Global-Local','Render-Render_shuffle_binding']:
        assert summary['comparisons'][p['main_endpoint']+'/'+pair]['paired_cells']==36
    proven('864adapted and288fixed-binding outputs, independent scoring and fixed contrasts',
        ['scoring/metrics.json','scoring/independent_recomputation.json','scoring/summary.json','report_tables.md'])
    cost=read(OUT/'cost.json');assert cost['complete'] and len(cost['queries'])==12
    assert {(r['kind'],r['condition'],r['id']) for r in cost['queries']}=={
        (k,c,m['id']) for k in KINDS for c in CONDITIONS for m in p['training_catalog'][:2]}
    assert all(r['warmup']==2 and r['retained']==5 and len(r['seconds'])==5 for r in cost['queries'])
    assert cost['source_scene_tensor_bytes']==p['source_state_bytes']
    assert sha256(cost['source_construction_reused_from'])==cost['source_construction_sha256']
    description=read(OUT/'cost_description.json')
    assert description['complete'] and description['raw_cost_sha256']==sha256(OUT/'cost.json')
    assert description['source_scene_tensor_bytes']==cost['source_scene_tensor_bytes']
    assert description['development_feature_cache_bytes']==cost['development_feature_cache_bytes']
    assert set(description['observations'])==set(KINDS)
    assert all(r['cases']==4 and r['retained_samples']==20 for r in description['observations'].values())
    assert all(sha256(ROOT/name)==digest for name,digest in description['cost_code_sha256'].items())
    proven('Fresh native-query cost and separately scoped inherited construction', ['cost.json','cost_description.json'])
    manifest=read(OUT/'figures/manifest.json');review=read(OUT/'figures/visual_review.json')
    assert manifest['complete'] and not manifest['visual_review_pending'] and review['complete']
    assert set(review['actually_viewed'])=={'outcomes.png','depth_cases.png'}
    for name in review['actually_viewed']:
        digest=sha256(OUT/'figures'/name)
        assert digest==manifest['sha256'][name]==review['sha256'][name]
    assert review['case_selection_sha256']==sha256(OUT/'figures/case_selection.json')
    proven('Actual outcome and fixed/counterexample image review', ['figures/manifest.json','figures/visual_review.json'])
    decision=read(OUT/'method_decision.json');docs=read(OUT/'documentation_sync.json')
    assert decision['complete'] and decision['plan_sha256']==sha256(OUT/'plan.json')
    assert decision['summary_sha256']==sha256(OUT/'scoring/summary.json')
    assert decision['report_sha256']==sha256(OUT/'report.md')
    assert not decision['new_confirmation'] and not decision['rgb_only'] and not decision['o6_pe_validated']
    assert not decision['automatic_successor']
    assert docs['complete']
    required={'docs/research_current.md','docs/research_outline_alignment.md','docs/rendered_feature_native_execution_plan.md'}
    assert required.issubset(docs['sha256'])
    assert all(sha256(ROOT/name)==digest for name,digest in docs['sha256'].items())
    processes=read(OUT/'final_process_status.json')
    assert processes['none_running'] and not processes['own_active_processes']
    proven('Scientific decision, synchronized current/outline docs and ended own jobs',
        ['report.md','method_decision.json','documentation_sync.json','final_process_status.json'])
    write_json(OUT/'completion_audit.json',dict(complete=True,requirements=requirements,
        remaining_required_work=[],plan_sha256=sha256(OUT/'plan.json'),
        scope='Finite experiment completion only. No result-dependent success threshold, new efficacy claim, or automatic successor.'))
    print('Finite validation deliverables verified; no required work remains.',flush=True)


if __name__=='__main__':main()
