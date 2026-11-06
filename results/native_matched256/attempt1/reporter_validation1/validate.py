"""Exercise the new reporter on an explicit old-versus-identical-old fixture.

No current training outputs or official TEST arrays are opened. Temporary
fixture reports are removed and are never treated as new research evidence.
"""
import copy
from pathlib import Path
import tempfile
import time

import numpy as np

from experiments import native_matched256 as experiment
from experiments import native_matched256_report as reporter
from experiments.round65_common import read, write, sha

DEST = experiment.OUT / 'reporter_validation1'


def main():
    experiment.check()
    assert not (DEST / 'result.json').exists()
    sources = {str(Path(reporter.__file__)): sha(reporter.__file__), str(Path(__file__)): sha(__file__)}
    expected = {}
    for row in read(experiment.PRIOR['relative5'] / 'result.json')['comparisons']:
        if row['control'] == 'distance_mlp':
            expected['distance_mlp', row['variant']] = row
    for row in read(experiment.PRIOR['fixed_isotropic5'] / 'result.json')['comparisons']:
        if row['left'] == 'relative5':
            expected['fixed_isotropic5', row['variant']] = row
    with tempfile.TemporaryDirectory(prefix='grope-matched256-reporter-fixture-') as temporary:
        fixture = Path(temporary)
        (fixture / 'data').symlink_to((experiment.OLD / 'data').resolve(), target_is_directory=True)
        (fixture / 'lock.json').write_text((experiment.OUT / 'lock.json').read_text())
        selections = []
        for role, root in experiment.PRIOR.items():
            for row in read(root / 'selected.json')['records']:
                if row['role'] == role and row['seed'] in experiment.SEEDS:
                    selections.append(copy.deepcopy(row))
        write(fixture / 'selected.json', dict(records=selections,
            all_nine_selected_before_development_evaluation=True,
            fixture_only=True, actual_new_fits=0))
        for role, root in experiment.PRIOR.items():
            for seed in experiment.SEEDS:
                source = root / 'evaluation' / role / str(seed)
                target = fixture / 'evaluation' / role / str(seed)
                target.mkdir(parents=True)
                meta = read(source / 'record.json')
                meta['selection_sha256'] = sha(fixture / 'selected.json')
                meta['fixture_only'] = True
                write(target / 'record.json', meta)
                for variant in experiment.VARIANTS:
                    (target / (variant + '.npz')).symlink_to(source / (variant + '.npz'))
        reporter.OUT = fixture
        reporter.report()
        result = read(fixture / 'result.json')
        assert len(result['records']) == 108 and len(result['comparisons']) == 36
        assert len(result['interactions']) == 12 and len(result['primary']) == 2
        assert all(row['train_tokens'] == 256 and row['variant'] == 'sparse256' for row in result['primary'])
        worst = 0.
        for row in result['comparisons']:
            if row['left'] != 'relative5':
                continue
            old = expected[row['right'], row['variant']]
            point = old.get('mean_gain_pp', old.get('mean_mca_gain_pp'))
            pairs = old.get('paired_seed_gains_pp', old.get('paired_mca_gain_pp'))
            assert point is not None and pairs is not None, old.keys()
            error = max(abs(row['mean_gain_pp'] - point), float(np.max(np.abs(np.asarray(row['paired_seed_gains_pp']) - pairs))))
            assert error < 1e-8
            worst = max(worst, error)
        for row in result['interactions']:
            for key in ('mean_gain_pp', 'paired_seed_gains_pp', 'class_mean_gain_pp',
                        'paired_seed_bootstrap_95_pp', 'paired_class_bootstrap_95_pp', 'crossed_seed_class_bootstrap_95_pp'):
                assert np.all(np.asarray(row[key]) == 0), (row['control'], row['variant'], key)
        rows = {(r['train_tokens'], r['role'], r['seed'], r['variant']): r for r in result['records']}
        for role in experiment.ROLES:
            for seed in experiment.SEEDS:
                for variant in experiment.VARIANTS:
                    left, right = rows[256, role, seed, variant], rows[512, role, seed, variant]
                    assert left['metrics'] == right['metrics'] and left['domains'] == right['domains']
        assert (fixture / 'report.md').is_file() and (fixture / 'comparison.png').is_file()
        summary = dict(passed=True, observed_unix=time.time(), fixture='existing512 predictions on both sides',
            true_new_fits=0, true_new_predictions=0, report_cells=108, comparisons=36,
            old_point_and_paired_estimate_max_error=worst,
            independent_metric_max_error=result['independent_metric_max_error'],
            all_12_interactions_and_their_intervals_exactly_zero=True,
            primary_routes_to_sparse256=True, metric_and_domain_duplicates_exact=True,
            source_sha256=sources, official_test_access=False, current_training_outputs_accessed=False,
            fixture_reports_are_not_evidence_and_removed=True, scientific_result=False)
    write(DEST / 'result.json', summary)
    print(summary, flush=True)


if __name__ == '__main__':
    main()
