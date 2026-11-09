"""Post-run execution review, adapted from the completed mean-precision review.

Run only after the fixed queue completes. This checks existing artifacts and
does not train, select a different checkpoint, or export new predictions.
Scientific and visual interpretation remain separate, explicit review work.
"""
import hashlib
import json
import math
from pathlib import Path
import time

ROOT = Path('/home/liwt/GRoPE')
OUT = ROOT / 'results/normalized_precision/attempt1'
HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    destination = HERE / 'execution_checks.json'
    assert not destination.exists(), 'Preserve the prior review record.'
    queue, lock, selected, result = [read(OUT / name) for name in
                                    ('queue_status.json', 'lock.json', 'selected.json', 'result.json')]
    assert queue['state'] == queue['stage'] == 'complete'
    assert not queue['active'] and not queue['pending']
    assert sha(OUT / 'lock.json') == (OUT / 'lock.sha256').read_text().strip()
    for group in ('source_sha256', 'prior_sha256', 'cache_sha256'):
        for path, digest in lock[group].items():
            assert sha(path) == digest, path
    assert selected['all_six_selected_before_development_evaluation']
    assert len(selected['records']) == 6
    assert selected['root_lock_sha256'] == sha(OUT / 'lock.json')
    assert result['lock_sha256'] == sha(OUT / 'lock.json')
    assert result['selected_sha256'] == sha(OUT / 'selected.json')
    for case, digest in selected['case_selection_sha256'].items():
        assert sha(OUT / case / 'selected.json') == digest
    rows = []
    for choice in selected['records']:
        case, role, seed = choice['case'], choice['role'], choice['seed']
        assert role == 'normalized_precision5' and seed in (11,22,33)
        current = OUT / case / 'runs' / role / str(seed)
        prior = ROOT / ('results/round69_compact_relation/attempt1' if case == 'region'
                        else 'results/round73_native_relative5/attempt1') / 'runs/relative5' / str(seed)
        run, history = read(current / 'run.json'), read(current / 'history.json')
        old_run, old_history = read(prior / 'run.json'), read(prior / 'history.json')
        epochs, updates = (60,1080) if case == 'region' else (120,46800)
        metric = 'selection_ce' if case == 'region' else 'ce'
        epoch_key = 'selected_epoch' if case == 'region' else 'best_epoch'
        pe_key = 'pe_parameters' if case == 'region' else 'position_parameters'
        best = min(history['validation'], key=lambda value: value[metric])
        streams = lambda h: [(v['epoch'],v['step'],v['stream_sha256']) for v in h['training']]
        jobs = [v for v in queue['completed'] if (v.get('action'),v.get('case'),v.get('role'),v.get('seed'))
                == ('train',case,role,seed)]
        checks = dict(
            full_budget=run['epochs']==epochs and run['steps']==updates and len(history['training'])==epochs,
            finite_training_and_selection=all(math.isfinite(v['loss']) for v in history['training'])
                and all(math.isfinite(v[metric]) for v in history['validation']),
            all_sampling_streams_match=streams(history)==streams(old_history),
            shared_initialization_matches=run['initial_shared_sha256']==old_run['initial_shared_sha256'],
            parameter_counts_match=run['parameters']==old_run['parameters'] and run[pe_key]==old_run[pe_key]==9792,
            validation_minimum_selected=run[epoch_key]==choice[epoch_key]==best['epoch']
                and run['best_val_ce']==best[metric],
            selected_identities_match=sha(current/'run.json')==choice['run_sha256']
                and sha(current/'best.pt')==choice['checkpoint_sha256'],
            normal_exit=bool(jobs) and jobs[-1]['exit_code']==0
                and all(j['exit_code']==75 and case=='native512' for j in jobs[:-1]))
        for name,digest in run['hashes'].items():
            assert sha(current/name)==digest, current/name
        if case == 'native512':
            gradients = [v for k,v in run['first_gradients'].items() if k.startswith('position.')]
            checks.update(data_and_final_stream_match=all(run[k]==old_run[k] for k in ('data_sha256','sampling_sha256')),
                positive_finite_initial_pe_gradients=len(gradients)==3
                    and all(v is not None and math.isfinite(v) and v>0 for v in gradients))
        assert all(checks.values()), (case,seed,checks)
        rows.append(dict(case=case,seed=seed,epochs=epochs,updates=updates,selected_epoch=choice[epoch_key],
            checks=checks,run_sha256=choice['run_sha256'],history_sha256=sha(current/'history.json'),
            checkpoint_sha256=choice['checkpoint_sha256'],training_exit_codes=[j['exit_code'] for j in jobs],
            existing_validation_replay={k:choice[k] for k in ('validation_ce_error','independent_ce_error',
                'gpu_logit_error','cpu_logit_error','cpu_prediction_disagreements') if k in choice}))
    exports = [j for j in queue['completed'] if j['action']=='export']
    assert len(exports)==6 and all(j['exit_code']==0 and j['started_unix']>selected['created_unix'] for j in exports)
    assert all(j['exit_code']==0 or (j['action']=='train' and j['case']=='native512' and j['exit_code']==75)
               for j in queue['completed'])
    records = result['records']
    assert len(records)==len({(r['case'],r['role'],r['seed'],r['condition']) for r in records})==120
    assert sum(bool(r['reused']) for r in records)==96
    for row in records:
        assert sha(row['prediction_path'])==row['prediction_sha256']
    assert len(result['comparisons'])==80 and len(result['primary'])==10 and len(result['error_counts'])==4
    assert [r['bootstrap_seed'] for r in result['comparisons']]==list(range(87001,87081))
    assert result['new_fits']==6 and result['new_prediction_groups']==24 and result['reused_prediction_groups']==96
    assert result['candidate']=='relative5' and not result['automatic_promotion']
    assert not result['fresh_confirmation'] and not result['official_test_access'] and not result['goal_complete']
    review = dict(checked_unix=time.time(),script_sha256=sha(Path(__file__)),
        root_lock_sha256=sha(OUT/'lock.json'),selection_sha256=sha(OUT/'selected.json'),
        result_sha256=sha(OUT/'result.json'),all_execution_checks_passed=True,records=rows,
        exports_after_global_selection=True,prediction_groups=120,
        independent_metric_error_reported=result['independent_metric_max_error'],
        permutation_records=result['permutation'],scientific_review_pending=True,visual_review_pending=True,
        scope='Existing execution and report identities only; this is not an external replication or a scientific pass.',
        goal_complete=False)
    destination.write_text(json.dumps(review,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(destination),all_execution_checks_passed=True,fits=len(rows),
                         prediction_groups=len(records),scientific_review_pending=True)))


if __name__ == '__main__':
    main()
