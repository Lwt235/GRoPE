"""Summarize existing completed predictions; no fitting or new data access."""
from collections import Counter
import hashlib
import json
from pathlib import Path
from statistics import mean
import time

OUT = Path('/home/liwt/GRoPE/results/mean_precision/attempt1')
DEST = Path(__file__).resolve().parent


def read(name):
    return json.loads((OUT / name).read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    target = DEST / 'analysis.json'
    assert not target.exists(), 'Preserve the existing review artifact.'
    result, errors = read('result.json'), read('error_analysis.json')
    queue, selected = read('queue_status.json'), read('selected.json')
    training = read('actual_review1/training_checks.json')
    exports = [job for job in queue['completed'] if job['action'] == 'export']
    seeds = [row['bootstrap_seed'] for row in result['comparisons']]
    expected_jobs = dict(preflight=2, train=6, selections=2, seal=1, export=6, report=1)
    checks = dict(
        complete_queue=queue['state'] == queue['stage'] == 'complete'
            and not queue['pending'] and not queue['active'],
        all_18_jobs_exit_zero=len(queue['completed']) == 18
            and all(job['exit_code'] == 0 for job in queue['completed'])
            and dict(Counter(job['action'] for job in queue['completed'])) == expected_jobs,
        all_6_training_checks_passed=len(training['records']) == 6
            and all(all(row['checks'].values()) for row in training['records']),
        all_6_exports_after_global_selection=len(exports) == 6
            and all(job['started_unix'] > selected['created_unix'] for job in exports),
        identities_match=sha(OUT / 'lock.json') == result['lock_sha256']
            == selected['root_lock_sha256'] == training['root_lock_sha256']
            and sha(OUT / 'selected.json') == result['selected_sha256']
            == training['selection_sha256'],
        expected_record_counts=len(result['records']) == 168
            and result['new_prediction_groups'] == 24
            and result['reused_prediction_groups'] == 144
            and len(result['comparisons']) == 112
            and len(errors['records']) == 56,
        bootstrap_schedule=seeds == list(range(83001, 83113)),
        saved_event_files_present=all(Path(row['event_path']).is_file() for row in errors['records']),
        no_official_test_or_fresh_confirmation=not result['official_test_access']
            and not result['fresh_confirmation'],
    )
    assert all(checks.values()), checks
    base_scores = []
    for case in ('region', 'native512'):
        for role in sorted({row['role'] for row in result['records']}):
            rows = sorted((row for row in result['records'] if row['case'] == case
                           and row['role'] == role and row['condition'] == 'base'),
                          key=lambda row: row['seed'])
            values = [100 * (row['metrics']['macro']['region_recall'] if case == 'region'
                             else row['metrics']['mca']) for row in rows]
            base_scores.append(dict(case=case, role=role, seeds=[r['seed'] for r in rows],
                                    values_pct=values, mean_pct=mean(values)))
    summaries = []
    for row in result['comparisons']:
        if row.get('stratum') is not None:
            continue
        event = next(e for e in errors['records'] if all(e[k] == row[k]
                     for k in ('case', 'condition', 'left', 'right')))
        item = {key: row[key] for key in ('case', 'condition', 'left', 'right',
                                         'mean_gain_pp', 'paired_seed_gains_pp')}
        item.update(joint_interval_pp=row['crossed_seed_scene_bootstrap_95_pp'
                    if row['case'] == 'region' else 'crossed_seed_class_bootstrap_95_pp'],
                    error_groups=event['groups'], fixed_id_examples=event['examples'])
        if row['case'] == 'region':
            item.update(other_metric_means=row['mean'], paired_metrics=row['paired'],
                        scenes=row['scenes'], scene_gains_pp=row['scene_mean_gain_pp'])
        else:
            values = row['class_mean_gain_pp']
            values = list(values.values()) if isinstance(values, dict) else values
            item.update(oa_mean_gain_pp=mean(row['paired_oa_gain_pp']),
                        oa_seed_gains_pp=row['paired_oa_gain_pp'],
                        ce_mean_difference=mean(row['paired_ce_difference']),
                        ce_seed_differences=row['paired_ce_difference'],
                        source_mean_gains_pp={k: mean(v) for k, v in row['domains'].items()},
                        source_seed_gains_pp=row['domains'],
                        classes_positive_negative_zero=[sum(v > 0 for v in values),
                                                        sum(v < 0 for v in values),
                                                        sum(v == 0 for v in values)])
        summaries.append(item)
    artifact = dict(created_unix=time.time(), script_sha256=sha(Path(__file__)),
        input_sha256={name: sha(OUT / name) for name in ('result.json', 'error_analysis.json',
            'selected.json', 'queue_status.json', 'actual_review1/training_checks.json')},
        checks=checks, base_scores=base_scores, comparisons=summaries,
        interactions=[],
        numerical_summary=dict(independent_metric_max_error=result['independent_metric_max_error'],
            permutation_records=len(result['permutation']),
            permutation_prediction_changes=sum(r['prediction_changes'] for r in result['permutation']),
            permutation_max_logit_error=max(r['maximum_logit_error'] for r in result['permutation'])),
        interpretation=dict(
            region='Q loses 2.988 pp to F and 3.848 pp to M on base recall; F-Q center metrics also improve. Q does not establish a replacement.',
            native='F-Q base mCA is +2.649 pp in all three seeds, with a broad joint interval; Q is below Distance in all base seeds and all condition means.',
            mechanism='Direction-neutral mean size and precision do not preserve the main accuracy benefit. Conditional value of mean precision when directions are present remains unresolved.',
            next_question='Preserve directional relations and test whether their scalar mean-precision amplification is needed; any normalization remains an untested hypothesis.',
            promotion=False, general_effectiveness=False, fresh_confirmation=False))
    target.write_text(json.dumps(artifact, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(output=str(target), checks=checks,
                          summarized_comparisons=len(summaries), numerical_summary=artifact['numerical_summary'])))


if __name__ == '__main__':
    main()
