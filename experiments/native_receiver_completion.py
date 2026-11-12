"""Finite-deliverable audit; no numerical efficacy gate is imposed."""
from experiments.native_receiver_data import *


def main():
    p=plan();assert p['expected_fits']==9 and p['steps']==2000 and p['expected_new_predictions']==2880
    assert p['choices_changed']==['C6'] and not p['new_confirmation'] and not p['automatic_successor']
    pre=read(OUT/'preflight.json');assert pre['complete'] and len(pre['records'])==6 and pre['pilot_weights_discarded']
    assert pre['native_parameters']>600000000 and read(OUT/'checkpoint_preflight.json')['complete']
    checks=read(OUT/'matched_checks.json');assert checks['complete'] and len(checks['records'])==9 and len(checks['source_fields'])==24
    timing=read(OUT/'training_summary.json');assert timing['complete'] and len(timing['completed'])==9
    persistence=read(OUT/'persistence_complete.json');assert persistence['complete'] and persistence['finals']==9
    for path in persistence['persistent_paths']:assert Path(path).is_file()
    score=read(OUT/'scoring/independent_recomputation.json')
    assert score['complete'] and score['new_files']==30 and score['new_predictions']==2880 and score['inherited_development_rows']==864
    for row in score['files']:
        marker=read(row['marker']);assert sha256(marker['prediction_path'])==row['prediction_sha256']
    summary=read(OUT/'scoring/summary.json');assert summary['complete'] and summary['plan_sha256']==sha256(OUT/'plan.json')
    for control in ['Local','Global']:
        assert summary['comparisons']['development/missing/roi/centered_log_mse/A_Render-A_'+control]['paired_cells']==36
        assert summary['interactions']['development/missing/roi/centered_log_mse/Render-'+control]['paired_cells']==36
    cost=read(OUT/'cost.json');assert cost['complete'] and len(cost['queries'])==12
    assert all(row['warmup']==2 and row['retained']==len(row['seconds'])==5 for row in cost['queries'])
    manifest=read(OUT/'figures/manifest.json');review=read(OUT/'figures/visual_review.json')
    assert manifest['complete'] and not manifest['visual_review_pending'] and review['complete']
    assert set(review['actually_viewed'])=={'outcomes.png','depth_cases.png'}
    assert all(sha256(OUT/'figures'/name)==digest==review['sha256'][name] for name,digest in manifest['sha256'].items())
    decision=read(OUT/'method_decision.json');assert decision['complete']
    assert decision['report_sha256']==sha256(OUT/'report.md') and decision['summary_sha256']==sha256(OUT/'scoring/summary.json')
    assert not decision['o6_pe_validated'] and not decision['new_confirmation'] and not decision['automatic_successor']
    assert (OUT/'report_tables.md').exists() and read(OUT/'cost_description.json')['complete']
    docs=read(OUT/'documentation_sync.json');assert docs['complete']
    required={'docs/research_current.md','docs/research_outline_alignment.md','docs/native_receiver_adaptation_execution_plan.md'}
    assert required.issubset(docs['sha256']) and all(sha256(ROOT/n)==h for n,h in docs['sha256'].items())
    status=read(OUT/'final_process_status.json');assert status['none_running'] and not status['own_active_processes']
    write_json(OUT/'completion_audit.json',dict(complete=True,remaining_required_work=[],plan_sha256=sha256(OUT/'plan.json'),
        scope='Finite experiment deliverables only, not success of overall scientific method.'))
    print('Finite native receiver experiment complete; no required work remains.',flush=True)


if __name__=='__main__':main()
