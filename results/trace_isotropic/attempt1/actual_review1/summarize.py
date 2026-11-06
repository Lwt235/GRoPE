"""Post-run review of the frozen trace-isotropic comparison; no training changes."""
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent.parent
DEST = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
load = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
result, events, queue, seal = [load(OUT / name) for name in
                             ('result.json', 'error_analysis.json', 'queue_status.json', 'selected.json')]
assert queue['state'] == queue['stage'] == 'complete' and len(queue['completed']) == 18
assert all(x['exit_code'] == 0 for x in queue['completed'])
assert (len(result['records']), len(result['comparisons']), len(events['records'])) == (96, 80, 40)
assert seal['all_six_selected_before_development_evaluation']
assert result['lock_sha256'] == sha(OUT / 'lock.json')
assert result['selected_sha256'] == sha(OUT / 'selected.json')

training = []
for case, epochs, steps, prior in (
        ('region', 60, 1080, 'round69_compact_relation'),
        ('native512', 120, 46800, 'round73_native_relative5')):
    for seed in (11, 22, 33):
        directory = OUT / case / 'runs/trace_isotropic5' / str(seed)
        old = ROOT / 'results' / prior / 'attempt1/runs/relative5' / str(seed)
        run, history, old_run, old_history = map(load, (
            directory / 'run.json', directory / 'history.json', old / 'run.json', old / 'history.json'))
        assert (run['epochs'], run['steps']) == (epochs, steps)
        assert all(np.isfinite(x['loss']) for x in history['training'])
        assert run['initial_shared_sha256'] == old_run['initial_shared_sha256']
        assert [x['stream_sha256'] for x in history['training']] == [x['stream_sha256'] for x in old_history['training']]
        key = 'selection_ce' if case == 'region' else 'ce'
        best = min(history['validation'], key=lambda x: x[key])
        assert best['epoch'] == run.get('best_epoch', run.get('selected_epoch'))
        assert best[key] == run['best_val_ce']
        training.append(dict(case=case, seed=seed, epochs=epochs, steps=steps, best_epoch=best['epoch'],
            selection_field=key, finite_losses=True, full_epoch_streams_match=True,
            shared_initialization_matches=True, peak_allocated_gib=run['peak_allocated'] / 2**30,
            run_sha256=sha(directory / 'run.json')))

checks = []
for entry in events['records']:
    with np.load(entry['event_path'], allow_pickle=False) as z:
        a, b = z[entry['left'] + '_correct'], z[entry['right'] + '_correct']
        mask, groups = z['eligible'], z['groups']
        assert a.shape == b.shape and a.shape[0] == 3
        counts = dict(both_correct=int((a[:, mask] & b[:, mask]).sum()),
            corrected=int((a[:, mask] & ~b[:, mask]).sum()),
            regressed=int((~a[:, mask] & b[:, mask]).sum()),
            both_wrong=int((~a[:, mask] & ~b[:, mask]).sum()))
        assert counts == entry['groups']['all']['event_counts']
        delta = [float(np.mean([a[k, mask & (groups == g)].mean() - b[k, mask & (groups == g)].mean()
                               for g in np.unique(groups[mask])]) * 100) for k in range(3)]
        error = float(np.max(np.abs(np.asarray(delta) - entry['groups']['all']['paired_seed_gains_pp'])))
        assert error < 1e-10
        for sample in entry['examples']:
            i = sample['index']
            assert z['ids'][i] == sample['id'] and mask[i]
            assert np.array_equal(z[entry['left'] + '_picked'][:, i], sample['left_prediction'])
            assert np.array_equal(z[entry['right'] + '_picked'][:, i], sample['right_prediction'])
            expected = a[:, i] & ~b[:, i] if sample['type'] == 'corrected_all_seeds' else ~a[:, i] & b[:, i]
            assert expected.all()
        checks.append(dict(case=entry['case'], condition=entry['condition'], left=entry['left'], right=entry['right'],
            counts_verified=True, example_id_predictions_and_events_verified=True, macro_difference_error_pp=error))

core_pairs = [('relative5', 'trace_isotropic5'), ('trace_isotropic5', 'fixed_isotropic5'),
              ('trace_isotropic5', 'distance_mlp')]
base = []
for case in ('region', 'native512'):
    for role in ('relative5', 'distance_mlp', 'fixed_isotropic5', 'trace_isotropic5'):
        rows = [x for x in result['records'] if (x['case'], x['role'], x['condition']) == (case, role, 'base')]
        values = [100 * (x['metrics']['macro']['region_recall'] if case == 'region' else x['metrics']['mca']) for x in rows]
        base.append(dict(case=case, role=role, seeds=[x['seed'] for x in rows],
                         values_percent=values, mean_percent=float(np.mean(values))))

analysis = dict(created_unix=time.time(), result_sha256=sha(OUT / 'result.json'),
    error_analysis_sha256=sha(OUT / 'error_analysis.json'), lock_sha256=sha(OUT / 'lock.json'),
    selected_sha256=sha(OUT / 'selected.json'), all_jobs_completed=True, training=training, base=base,
    core_comparisons=[x for x in result['comparisons'] if x.get('stratum') is None and (x['left'], x['right']) in core_pairs],
    error_event_checks=checks, base_error_events=[x for x in events['records'] if x['condition'] == 'base'],
    selected=seal['records'], independent_metric_max_error=result['independent_metric_max_error'],
    permutation=result['permutation'], decision=dict(retain_candidate='relative5', promote_trace_isotropic5=False,
        local_size_explains_most_gain='not_supported_in_this_trace_matched_development_control',
        region_joint_orientation_axis_ratio_added_utility='supported_on_reused_development_scenes',
        native_joint_orientation_axis_ratio_added_utility='positive_mean_with_seed_and_class_uncertainty',
        isolates_orientation_alone=False, scale_volume_precision_equivalence=False,
        fresh_confirmation=False, general_effectiveness_established=False, goal_complete=False))
(DEST / 'analysis.json').write_text(json.dumps(analysis, ensure_ascii=False, indent=2) + '\n')
backup = DEST / 'automatic_report.md'
if not backup.exists():
    shutil.copy2(OUT / 'report.md', backup)
print(json.dumps(dict(fits_verified=len(training), event_comparisons_checked=len(checks),
    event_max_error_pp=max(x['macro_difference_error_pp'] for x in checks), analysis=str(DEST / 'analysis.json'))))
