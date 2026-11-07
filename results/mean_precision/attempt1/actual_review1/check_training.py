"""Review completed fixed-budget fits and the shared selection seal."""
import hashlib
import json
import math
from pathlib import Path
import time

ROOT = Path('/home/liwt/GRoPE')
OUT = ROOT / 'results/mean_precision/attempt1'
DEST = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    destination = DEST / 'training_checks.json'
    assert not destination.exists()
    selected = read(OUT / 'selected.json')
    assert selected['all_six_selected_before_development_evaluation']
    assert len(selected['records']) == 6
    assert selected['root_lock_sha256'] == sha(OUT / 'lock.json')
    for case, digest in selected['case_selection_sha256'].items():
        assert sha(OUT / case / 'selected.json') == digest
    queue = read(OUT / 'queue_status.json')
    rows = []
    for choice in selected['records']:
        case, role, seed = choice['case'], choice['role'], choice['seed']
        current = OUT / case / 'runs' / role / str(seed)
        prior = ROOT / ('results/round69_compact_relation/attempt1' if case == 'region'
                        else 'results/round73_native_relative5/attempt1') / 'runs/relative5' / str(seed)
        run, history = read(current / 'run.json'), read(current / 'history.json')
        old_run, old_history = read(prior / 'run.json'), read(prior / 'history.json')
        epochs, steps = (60, 1080) if case == 'region' else (120, 46800)
        metric = 'selection_ce' if case == 'region' else 'ce'
        epoch_key = 'selected_epoch' if case == 'region' else 'best_epoch'
        pe_key = 'pe_parameters' if case == 'region' else 'position_parameters'
        best = min(history['validation'], key=lambda v: v[metric])
        streams = lambda h: [(v['epoch'], v['step'], v['stream_sha256']) for v in h['training']]
        jobs = [v for v in queue['completed'] if (v.get('action'), v.get('case'), v.get('role'), v.get('seed'))
                == ('train', case, role, seed)]
        checks = dict(full_budget=run['epochs'] == epochs and run['steps'] == steps
                      and len(history['training']) == epochs,
            finite_training_and_selection=all(math.isfinite(v['loss']) for v in history['training'])
                and all(math.isfinite(v[metric]) for v in history['validation']),
            all_sampling_streams_match=streams(history) == streams(old_history),
            shared_initialization_matches=run['initial_shared_sha256'] == old_run['initial_shared_sha256'],
            parameter_counts_match=run['parameters'] == old_run['parameters']
                and run[pe_key] == old_run[pe_key] == 9792,
            validation_minimum_selected=run[epoch_key] == choice[epoch_key] == best['epoch']
                and run['best_val_ce'] == best[metric],
            selected_run_and_checkpoint_match=sha(current / 'run.json') == choice['run_sha256']
                and sha(current / 'best.pt') == choice['checkpoint_sha256'],
            normal_exit=len(jobs) == 1 and jobs[0]['exit_code'] == 0)
        if case == 'native512':
            gradient = [v for k, v in run['first_gradients'].items() if k.startswith('position.')]
            checks.update(data_and_sampling_config_match=all(run[k] == old_run[k]
                          for k in ('data_sha256', 'sampling_sha256')),
                          positive_finite_initial_PE_gradients=len(gradient) == 3
                          and all(math.isfinite(v) and v > 0 for v in gradient))
        assert all(checks.values()), (case, role, seed, checks)
        rows.append(dict(case=case, role=role, seed=seed, epochs=epochs, steps=steps,
            selection_metric=metric, selected_epoch=choice[epoch_key], checks=checks,
            history_sha256=sha(current / 'history.json'), reference_history_sha256=sha(prior / 'history.json'),
            run_sha256=choice['run_sha256'], checkpoint_sha256=choice['checkpoint_sha256'],
            validation_replay={k: choice[k] for k in ('validation_ce_error', 'independent_ce_error',
                'gpu_logit_error', 'cpu_logit_error', 'cpu_prediction_disagreements') if k in choice}))
    exports = [j for j in queue['completed'] + queue['active'] if j['action'] == 'export']
    assert exports and all(j['started_unix'] > selected['created_unix'] for j in exports)
    result = dict(checked_unix=time.time(), script_sha256=sha(Path(__file__)),
        root_lock_sha256=sha(OUT / 'lock.json'), selection_sha256=sha(OUT / 'selected.json'),
        records=rows, exports_observed=len(exports),
        observed_exports_started_after_global_selection=True,
        scope='Completed fits and existing selection replay records; full-initial-state checks remain in preparation/startup evidence. No new predictions or model updates.',
        scientific_effectiveness_established=False)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(output=str(destination), fits=len(rows), selected_epochs={
        case: {role: [v['selected_epoch'] for v in rows if v['case'] == case and v['role'] == role]
               for role in ('mean_precision5',)}
        for case in ('region', 'native512')}, all_training_checks_passed=True)))


if __name__ == '__main__':
    main()
