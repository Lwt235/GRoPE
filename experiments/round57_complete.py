"""Record a completed R57 experiment after actual terminal/visual review.

Postprocessing only; no fitting, candidate promotion or fresh data access.
Numeric warnings are retained as evidence, not silently turned into success.
"""
import argparse
from pathlib import Path
import time
import numpy as np
from experiments.round57_common import (ROOT,OUT,ROLES,NEW_ROLES,SEEDS,VARIANTS,
    checked,sha,read,write,model_path)


def complete(out,queue_chunk,report_chunk,report,visual_review):
    lock=checked(out)
    destination=out/'completion.json'
    if destination.exists():raise ValueError('preserve existing completion record')
    if not visual_review:raise ValueError('record actual plot inspection, not an inferred visual pass')
    status=read(out/'queue_status.json');children=status['completed']
    if status['state']!='complete' or status['active'] or not status['all_children_exit_zero']:
        raise ValueError('original controller is not recorded complete')
    expected_counts={'smoke':1,'train':9,'training_audit':1,'export':3,'metrics':1}
    actual_counts={name:sum(r['stage']==name for r in children) for name in expected_counts}
    if actual_counts!=expected_counts or len(children)!=15 or any(r['exit_code']!=0 for r in children):
        raise ValueError('missing/failed original worker')
    if not queue_chunk or not report_chunk:raise ValueError('actual terminal tool outputs required')
    seal=read(out/'audit_training/completion.json');summary=read(out/'metrics/summary.json')
    verification=read(out/'metrics/verification.json')
    expected={(role,seed) for role in ROLES for seed in SEEDS}
    if {(r['role'],r['seed']) for r in seal['records']}!=expected or len(seal['records'])!=48:
        raise ValueError('selection population mismatch')
    audit_end=next(r['completed_unix'] for r in children if r['stage']=='training_audit')
    export_start=min(r['started_unix'] for r in children if r['stage']=='export')
    if not seal['unix']<=audit_end<=export_start:raise ValueError('selection/export order mismatch')
    if sha(out/'data/manifest.json')!=lock['copied_manifest_sha256']:
        raise ValueError('copied data manifest changed')
    for partition in read(out/'data/manifest.json')['partitions']:
        if sha(partition['path'])!=partition['sha256']:raise ValueError('development input changed')
    for reference in lock['references']:
        path=Path(reference['path'])
        for name,digest in reference['hashes'].items():
            if sha(path/name)!=digest:raise ValueError('reused reference changed')
    runs=[];predictions={};reference_prediction_equal=0
    for selected in seal['records']:
        role,seed=selected['role'],selected['seed'];path=model_path(role,seed,out)
        run=read(path/'run.json')
        if sha(path/'run.json')!=selected['run_sha256'] or sha(path/'best.pt')!=selected['checkpoint_sha256']:
            raise ValueError('selected run/model changed')
        for name,digest in run['hashes'].items():
            if sha(path/name)!=digest:raise ValueError('training artifact changed')
        identity=[run['initial_shared_sha256'],run['batch_augmentation_sha256']]
        if identity!=lock['paired_reference_identity'][str(seed)]:raise ValueError('unpaired comparison')
        if run['epochs']!=120 or run['steps']!=29760:raise ValueError('budget mismatch')
        history=read(path/'history.json')
        selected_history=min(history['validation'],key=lambda r:r['ce'])
        if selected_history['epoch']!=run['best_epoch'] or selected_history['ce']!=run['best_val_ce']:
            raise ValueError('selection mismatch')
        exported=read(out/'evaluation'/role/(str(seed)+'.json'))
        npz=out/'evaluation'/role/(str(seed)+'.npz')
        if sha(npz)!=exported['predictions_sha256'] or exported['checkpoint_sha256']!=selected['checkpoint_sha256']:
            raise ValueError('prediction/model identity mismatch')
        if exported['selection_seal_sha256']!=sha(out/'audit_training/completion.json'):
            raise ValueError('selection seal changed')
        if role not in NEW_ROLES:
            ref=next(r for r in lock['references'] if r['role']==role and r['seed']==seed)
            if sha(npz)!=ref['prediction_sha256'] or sha(ref['prediction_path'])!=ref['prediction_sha256']:
                raise ValueError('reference predictions are not exact copies')
            if sha(ref['prediction_record_path'])!=ref['prediction_record_sha256']:
                raise ValueError('reference prediction metadata changed')
            reference_prediction_equal+=1
        predictions[str(npz)]=sha(npz)
        runs.append(dict(role=role,seed=seed,reused=role not in NEW_ROLES,
            selected_epoch=run['best_epoch'],initial_shared_sha256=identity[0],sampling_sha256=identity[1],
            checkpoint_sha256=selected['checkpoint_sha256'],run_sha256=selected['run_sha256'],
            seconds=run['seconds'],parameters=run['parameters'],positional_parameters=run['positional_parameters'],
            shared_parameters_changed=run['shared_parameters_changed'],position_parameter_changes=run['position_parameter_changes']))
    expected_metrics={(r,s,v) for r in ROLES for s in SEEDS for v in VARIANTS}
    metric_rows=verification['metric_replays']
    if len(metric_rows)!=336 or {(r['role'],r['seed'],r['variant']) for r in metric_rows}!=expected_metrics:
        raise ValueError('metric replay population mismatch')
    diagnostic_rows=summary['diagnostics']
    expected_diagnostics={(r,s,m) for r in ('pair_full',) for s in SEEDS
                          for m in ('shape_off',)}
    if len(diagnostic_rows)!=3 or {(r['role'],r['seed'],r['mode']) for r in diagnostic_rows}!=expected_diagnostics:
        raise ValueError('diagnostic population mismatch')
    for row in diagnostic_rows:
        path=out/'diagnostics'/row['role'];name=str(row['seed'])+'_'+row['mode']
        record=read(path/(name+'.json'));npz=path/(name+'.npz')
        selected=next(r for r in seal['records'] if r['role']==row['role'] and r['seed']==row['seed'])
        if sha(npz)!=record['predictions_sha256'] or record['checkpoint_sha256']!=selected['checkpoint_sha256']:
            raise ValueError('diagnostic source changed')
        predictions[str(npz)]=sha(npz)
    provenance=read(out/'report1/provenance.json')
    if provenance['summary_sha256']!=sha(out/'metrics/summary.json') or provenance['verification_sha256']!=sha(out/'metrics/verification.json'):
        raise ValueError('report input changed')
    report=Path(report).resolve()
    figures=[out/'report1/comparison.png',out/'report1/learning_curves.png']
    for file in [report,*figures]:
        if not file.is_file():raise ValueError('missing reviewed result artifact')
    discrepancies=[r for r in metric_rows if r['correct_difference']!=0]
    diagnostic_discrepancies=[r for r in diagnostic_rows if r['correct_replay_difference']!=0]
    handles=read(out/'execution_handles.json')
    write(destination,dict(completed=True,scope='R57 experiment execution and bounded checks,not research-goal achievement',
        unix=time.time(),lock_sha256=sha(out/'lock.json'),scientific_sources=len(lock['sources']),
        controller_terminal_tool_chunk=queue_chunk,report_terminal_tool_chunk=report_chunk,
        controller_session_id=handles['queue_session_id'],
        report_watch_session_id=handles.get('report_watch_session_id'),child_counts=actual_counts,
        all_children_exit_zero=True,runs=runs,new_training_runs=9,reused_training_runs=39,
        original_reference_predictions_exact=reference_prediction_equal,selected_before_new_exports=True,
        new_training_seconds=sum(r['seconds'] for r in runs if not r['reused']),
        actual_queue_wall_seconds=status['observed_unix']-status['started_unix'],
        operator_warnings=seal['warnings'],max_operator_probability_error=max(r['numeric']['p_max_error'] for r in seal['records']),
        operator_winner_changes=sum(r['numeric']['winner_changes'] for r in seal['records']),
        max_saved_validation_ce_error=max(r['val_ce_error'] for r in seal['records']),
        metric_groups=336,diagnostic_groups=3,correct_count_discrepancies=discrepancies,
        diagnostic_count_discrepancies=diagnostic_discrepancies,max_metric_ce_error=max(r['ce_error'] for r in metric_rows),
        max_diagnostic_ce_error=max(r['ce_replay_error'] for r in diagnostic_rows),
        prediction_hashes=predictions,report_path=str(report),report_sha256=sha(report),
        plots_visually_inspected=True,plot_hashes={str(f):sha(f) for f in figures},
        independent_arithmetic_same_author=True,independent_full_network_or_lab=False,
        official_test_access=False,fresh_confirmation=False,candidate_promoted=False,full_goal_achieved=False))
    print('EXPERIMENT_COMPLETE',destination,'full goal remains unachieved',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=OUT)
    parser.add_argument('--queue-chunk',required=True);parser.add_argument('--report-chunk',required=True)
    parser.add_argument('--report',type=Path,required=True);parser.add_argument('--visual-review',action='store_true')
    args=parser.parse_args();complete(args.out,args.queue_chunk,args.report_chunk,args.report,args.visual_review)
