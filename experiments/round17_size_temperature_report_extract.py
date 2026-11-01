"""External, standard-library-only report of completed audited TRAIN attribution.

Never imports an experiment/model or decodes probability/label arrays. Hashes
the complete evidence inventories; trusts the accepted numerical audit rather
than running it again. No selector, scientific gate, or new statistics.
"""
from __future__ import annotations

import argparse
import csv
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
SCENES = ('harvard_c3/hv_c3_1', 'harvard_c5/hv_c5_1', 'harvard_c6/hv_c6_1')
ROLES = ('trace_full', 'size_temperature')
PARTITIONS = ('g8', 'g16', 'mixed', 'g24')
CONDITIONS = tuple(q+'_'+k for q in PARTITIONS[:3] for k in PARTITIONS[:3]) + (
    'g8_g24', 'g24_g8', 'g16_g24', 'g24_g16', 'g24_g24')
SIZES = dict(g8=((8, 8),), g16=((16, 16),), mixed=((8, 8), (8, 16), (16, 8), (16, 16)), g24=((24, 24),))
NATIVE_COUNTS = dict(g8=432, g16=108, mixed=243, g24=48)
CONFIG = dict(h=.25, temperature=10.)
AUDIT_POLICY = dict(pre_forward_design=True, scientific_gate_defined=False,
    shared_old_preparation_for_replay=True, independent_new_model_import=False,
    independent_numpy_moments_and_integral=True, metric_tolerance=1e-8,
    summary_tolerance=1e-10, probability_tolerance=1e-4,
    reference_tolerance=1e-8, same_mass_tolerance=1e-12)
SUMMARY_FALSE = ('scientific_gate_defined', 'selection_changed', 'current_validation_changed', 'efficiency_claim')
VERIFICATION_FALSE = ('scientific_gate_passed', 'scientific_gate_defined', 'independent_actual_preparation',
    'selection_changed', 'validation_data_read', 'new_sensor_data_read', 'cost_measured')


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def read(path):
    def invalid(value):
        raise ValueError('nonfinite JSON constant: '+value)
    return json.loads(Path(path).read_text(), parse_constant=invalid)


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def exact(actual, expected, message):
    require(type(actual) is type(expected) and actual == expected, message)


def inventory(root, files):
    """Exact population and byte hashes; no NPZ decoding or file writes."""
    root = root.resolve()
    require(isinstance(files, dict), 'inventory must be a mapping')
    for name in files:
        p = Path(name)
        require(not p.is_absolute() and '..' not in p.parts and str(p) == name,
                'unsafe inventory path: '+name)
    paths = [p for p in root.rglob('*') if p.is_file()]
    require(not any(p.is_symlink() for p in root.rglob('*')), 'symlinks are not sealed evidence')
    actual = {str(p.relative_to(root)) for p in paths if p != root/'completion.json'}
    require(actual == set(files), 'completed inventory population differs: '+str(root))
    for name, expected in files.items():
        require(digest(root/name) == expected, 'completed artifact changed: '+str(root/name))


def check_sources(lock, directory, repo):
    require(isinstance(lock['source_hashes'], dict), 'source hashes required')
    for name, expected in lock['source_hashes'].items():
        p = Path(name)
        require(not p.is_absolute() and '..' not in p.parts, 'unsafe source path')
        require(digest(repo/p) == expected and digest(directory/'source_snapshot'/p) == expected,
                'source or frozen snapshot changed: '+name)


def fraction_dict(value):
    return dict(numerator=value.numerator, denominator=value.denominator)


def check_fraction(value):
    if value is None:
        return
    require(set(value) == {'numerator', 'denominator'}
            and all(type(value[k]) is int for k in value) and value['denominator'] > 0,
            'invalid exact Fraction schema')
    require(fraction_dict(Fraction(value['numerator'], value['denominator'])) == value,
            'fraction must preserve canonical exact numerator/denominator')


def strata_names():
    return {c+f'__{w}x{h}' for c in CONDITIONS for w, h in SIZES[c.split('_')[0]]}


def verify_numeric(rows, verification, lock):
    first = {s:min((e for e in lock['entries'] if e['scene'] == s), key=lambda e:Path(e['path']).name)['pair']
             for s in SCENES}
    expected = {(s, tuple(first[s]), c, role) for s in SCENES for c in CONDITIONS for role in ROLES}
    found = [(r['scene'], tuple(r['pair']), r['condition'], r['role']) for r in rows]
    require(len(rows) == 84 and len(set(found)) == 84 and set(found) == expected,
            'complete fixed numerical sample required')
    for row in rows:
        require(row['passed'] is True and row['same_prepared_tau32_exact'] is True
                and row['same_prepared_bias32_exact'] is True, 'independent numerical check failed')
        count = NATIVE_COUNTS[row['query']]
        indices = [i*(count-1)//7 for i in range(8)]
        require(row['indices'] == indices, 'fixed numerical query indices changed')
        for key, limit in (('max_probability_error', 1e-4), ('reference_convergence_error', 1e-8),
                           ('same_mass_probability_error', 1e-12), ('hybrid_probability_error', 1e-4)):
            value = row[key]
            require(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= limit,
                    'numerical tolerance failed: '+key)
    exact(verification['numerical_failure_count'], 0, 'numerical failure count changed')


def verify_population(summary, evaluated, paired, strata, lock):
    entries = lock['entries']
    cohort = {(e['scene'], tuple(e['pair'])) for e in entries}
    require(len(entries) == len(cohort) == 48 and {s for s, _ in cohort} == set(SCENES)
            and all(sum(s == scene for s, _ in cohort) == 16 for scene in SCENES), 'TRAIN cohort changed')
    wanted = {(s, p, c, role) for s, p in cohort for c in CONDITIONS for role in ROLES}
    found = [(r['scene'], tuple(r['pair']), r['condition'], r['role']) for r in evaluated]
    require(len(found) == 1344 and len(set(found)) == 1344 and set(found) == wanted,
            'complete role/scene/condition rows required')
    for row in evaluated:
        q, k = row['condition'].split('_')
        require(row['config'] == CONFIG and row['shape'] == [NATIVE_COUNTS[q], NATIVE_COUNTS[k]]
                and row['dtype'] == 'float64', 'native row operator/population changed')
    for values, expected, key in (
        (paired, {(s, p, c) for s, p in cohort for c in CONDITIONS},
         lambda r:(r['scene'], tuple(r['pair']), r['condition'])),
        (strata, {(s, p, c, w, h) for s, p in cohort for c in CONDITIONS for w, h in SIZES[c.split('_')[0]]},
         lambda r:(r['scene'], tuple(r['pair']), r['condition'], r['width'], r['height']))):
        keys = [key(r) for r in values]
        require(len(keys) == len(set(keys)) == len(expected) and set(keys) == expected,
                'complete paired/native-size rows required')
    exact(summary['size_stratum_rows'], 1104, 'native size stratum count changed')
    require(set(summary['metrics']) == set(ROLES), 'both roles required')
    for metric in summary['metrics'].values():
        require(set(metric['by_scene_condition']) == set(SCENES)
                and set(metric['by_condition']) == set(CONDITIONS), 'complete role summary required')
        for scene in SCENES:
            require(set(metric['by_scene_condition'][scene]) == set(CONDITIONS), 'scene conditions omitted')
        for values in metric['by_condition'].values():
            check_fraction(values['pck_exact'])
    for name, expected in (('paired_summary', set(CONDITIONS)), ('size_strata_summary', strata_names())):
        require(set(summary[name]) == expected, 'complete paired/strata summary required')
        for group in summary[name].values():
            require(set(group['by_scene']) == set(SCENES), 'paired summary omits scene')
            for values in (*group['by_scene'].values(), group['macro']):
                check_fraction(values['pck_difference_exact'])
    support = summary['support_statistics']
    exact(support['unique_queries'], 39888, 'unique support population changed')
    expected = {(s, q, w, h) for s in SCENES for q in PARTITIONS for w, h in SIZES[q]}
    groups = [(g['scene'], g['query'], g['width'], g['height']) for g in support['groups']]
    require(len(groups) == len(set(groups)) == 21 and set(groups) == expected
            and sum(g['n'] for g in support['groups']) == 39888, 'complete support groups required')


def accepted_metadata(run, audit, *, repo=ROOT):
    """Reject incomplete/failed audits before opening any effect summary."""
    required = {run:('lock.json', 'lock.sha256', 'forward_complete.json', 'summary.json', 'completion.json'),
                audit:('lock.json', 'lock.sha256', 'started.json', 'verification.json',
                       'numerical_checks.json', 'independent_summary.json', 'completion.json')}
    require(run != audit and run not in audit.parents and audit not in run.parents, 'separate evidence roots required')
    for root, names in required.items():
        for name in names:
            require((root/name).is_file(), 'completed independent acceptance required: '+str(root/name))
    require(not (audit/'failure.json').exists(), 'independent audit failure must be preserved, not accepted')
    ac = read(audit/'completion.json')
    require(ac['passed'] is True and ac['scientific_gate_passed'] is False, 'successful descriptive audit required')
    inventory(audit, ac['files'])
    v = read(audit/'verification.json')
    require(v['passed'] is True and all(v[n] is False for n in VERIFICATION_FALSE)
            and v['descriptive_attribution_only'] is True, 'unsupported audit acceptance or scientific claim')
    for name in ('lock', 'forward_complete', 'summary', 'completion'):
        require(v['producer_'+name+'_sha256'] == digest(run/(name+'.json')),
                'audit belongs to a different producer run: '+name)
    for root in (run, audit):
        require(digest(root/'lock.json') == (root/'lock.sha256').read_text().strip(), 'lock checksum changed')
    require(v['independent_audit_lock_sha256'] == digest(audit/'lock.json'), 'audit lock binding changed')
    lock, design, pc = read(run/'lock.json'), read(audit/'lock.json'), read(run/'completion.json')
    require(design['run'] == str(run) and design['producer_lock_sha256'] == digest(run/'lock.json'),
            'independent design belongs to a different producer run')
    for key, value in AUDIT_POLICY.items():
        exact(design.get(key), value, 'fixed audit policy changed: '+key)
    require(set(design['source_hashes']) <= set(lock['source_hashes'])
            and all(lock['source_hashes'][n] == h for n, h in design['source_hashes'].items()), 'joint source binding changed')
    require(len(lock['source_hashes']) == 84 and len(design['source_hashes']) == 37, 'fixed source population changed')
    check_sources(lock, run, repo); check_sources(design, audit, repo)
    require(lock['roles'] == list(ROLES) and lock['config'] == CONFIG
            and lock['entries'] == lock['prior']['entries'], 'fixed attribution configuration changed')
    for path, expected in lock['prior']['files'].items():
        require(Path(path).is_absolute() and digest(Path(path)) == expected, 'accepted TRAIN metadata binding changed')
    inventory(run, pc['files'])
    require(pc['lock_sha256'] == digest(run/'lock.json') and pc['summary_sha256'] == digest(run/'summary.json')
            and pc['independent_audit_pending'] is True, 'producer completion binding changed')
    seal = read(run/'forward_complete.json')
    require(seal['lock_sha256'] == digest(run/'lock.json') and seal['labels_decoded'] is False
            and seal['all_probabilities_retained'] is True and seal['matrices'] == 1344
            and seal['raw_probability_bytes'] == 512434944 and set(seal['forward_chain']) == set(SCENES),
            'complete label-free global seal required')
    evaluated, paired, strata, ends = [], [], [], []
    for scene in SCENES:
        records = {}
        for stage in ('forward', 'evaluation'):
            directory = run/stage/scene.replace('/', '__')
            completion, start = read(directory/'completion.json'), read(directory/'started.json')
            inventory(directory, completion['files'])
            require(completion['stage'] == stage and completion['scene'] == scene
                    and completion['lock_sha256'] == digest(run/'lock.json') and completion['record_count'] == 448
                    and completion['runtime'] == lock['runtime'] and start['created_unix'] == completion['started_unix'],
                    'scene stage population/identity changed')
            records[stage] = (completion, start, directory)
        fc, fs, fd = records['forward']; ec, es, ed = records['evaluation']
        require(lock['created_unix'] <= design['created_unix'] <= fs['created_unix'] <= fc['finished_unix'] <= seal['sealed_unix']
                <= es['created_unix'] <= ec['finished_unix'], 'forward-before-label chronology changed')
        require(fc['labels_decoded'] is False and ec['labels_decoded'] is True
                and fc['primary_exact_matrices'] == 224 and fc['determinism_matrices'] == 28
                and fc['all_probabilities_retained'] is True and fs['lock_sha256'] == digest(run/'lock.json')
                and fs['runtime'] == lock['runtime'] and ec['forward_complete_sha256'] == digest(run/'forward_complete.json')
                and es['forward_complete_sha256'] == digest(run/'forward_complete.json')
                and seal['forward_chain'][scene] == dict(completion_sha256=digest(fd/'completion.json'), finished_unix=fc['finished_unix']),
                'scene source/seal binding changed')
        evaluated.extend(read(ed/'rows.json')); paired.extend(read(ed/'paired_rows.json')); strata.extend(read(ed/'size_strata.json'))
        ends.append(ec['finished_unix'])
    # Opening effect summaries is last, after accepted audit and full hash chain.
    summary, independent, numeric = read(run/'summary.json'), read(audit/'independent_summary.json'), read(audit/'numerical_checks.json')
    start = read(audit/'started.json')
    require(start['producer_lock_sha256'] == digest(run/'lock.json') and start['created_unix'] == v['started_unix']
            and v['original_design_created_unix'] == design['created_unix']
            and max(ends) <= summary['created_unix'] <= pc['finished_unix'] <= v['started_unix'] <= v['finished_unix'] <= ac['finished_unix'],
            'completion/audit chronology changed')
    require(summary['config'] == CONFIG and summary['roles'] == list(ROLES)
            and summary['lock_sha256'] == digest(run/'lock.json') and summary['forward_complete_sha256'] == digest(run/'forward_complete.json')
            and summary['descriptive_attribution_only'] is True and summary['independent_audit_pending'] is True
            and all(summary[n] is False for n in SUMMARY_FALSE), 'unsupported summary claim/binding')
    for name, count in dict(matrices=1344, pairs=48, paired_conditions=672).items():
        exact(summary[name], count, 'summary population changed: '+name)
    for name, count in dict(matrices=1344, primary_old_hash_matches=672, independent_count_tables=672,
        independent_numerical_matrices=84, numerical_query_rows=672, actual_prepared_bias_sample_matrices=84,
        unique_support_queries=39888).items():
        exact(v[name], count, 'audit population changed: '+name)
    for key, limit in (('maximum_metric_sum_error', 1e-8), ('maximum_summary_error', 1e-10),
                       ('maximum_complete_same_raw_error', 1e-12)):
        require(type(v[key]) in (int, float) and math.isfinite(v[key]) and 0 <= v[key] <= limit, 'audit tolerance failed: '+key)
    verify_population(summary, evaluated, paired, strata, lock)
    verify_numeric(numeric, v, lock)
    require(set(independent) == {'metrics', 'paired_summary', 'size_strata_summary', 'support_statistics'}, 'independent summary incomplete')
    return dict(summary=summary, independent_summary=independent, verification=v, numerical_checks=numeric,
                evaluated=evaluated, paired=paired, strata=strata, lock=lock, audit_lock=design)


def tables(data):
    summary = data['summary']
    detail, conditions, macros, paired, strata = [], [], [], [], []
    for role in ROLES:
        metric = summary['metrics'][role]
        macros.append(dict(role=role, **{k:v for k, v in metric.items() if k not in ('by_scene_condition', 'by_condition')}))
        for scene in SCENES:
            for condition in CONDITIONS:
                value = metric['by_scene_condition'][scene][condition]
                detail.append(dict(role=role, scene=scene, condition=condition, **value,
                    pck_exact=fraction_dict(Fraction(value['pck_count'], value['n_point']))))
        conditions.extend(dict(role=role, condition=c, **metric['by_condition'][c]) for c in CONDITIONS)
    for field, target in (('paired_summary', paired), ('size_strata_summary', strata)):
        for group, value in summary[field].items():
            target.extend(dict(group=group, scene=s, **value['by_scene'][s]) for s in SCENES)
            target.append(dict(group=group, scene='equal_scene_macro', **value['macro']))
    return dict(role_scene_conditions=detail, role_condition_macros=conditions, role_overall_macros=macros,
                paired_scene_and_macro=paired, size_strata_scene_and_macro=strata,
                unique_support_groups=summary['support_statistics']['groups'],
                all_evaluated_rows=data['evaluated'], all_paired_rows=data['paired'], all_size_stratum_rows=data['strata'])


def flattened(value, prefix=''):
    output = {}
    for key, item in value.items():
        name = prefix+key
        if isinstance(item, dict):
            output.update(flattened(item, name+'.'))
        else:
            output[name] = json.dumps(item, ensure_ascii=False, allow_nan=False) if isinstance(item, list) else item
    return output


def extract(run, audit, out, *, repo=ROOT):
    run, audit, out = (Path(p).resolve() for p in (run, audit, out))
    require(not out.exists() and out != run and out != audit and run not in out.parents and audit not in out.parents,
            'report output must be new and outside both frozen roots')
    data = accepted_metadata(run, audit, repo=repo)
    output = tables(data)
    out.mkdir(parents=True, exist_ok=False)
    for name, rows in output.items():
        write(out/(name+'.json'), rows)
        flat = [flattened(row) for row in rows]
        columns = list(dict.fromkeys(key for row in flat for key in row))
        with (out/(name+'.csv')).open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader(); writer.writerows(flat)
    for name in ('summary', 'independent_summary', 'verification', 'numerical_checks'):
        write(out/(name+'.json'), data[name])
    bound = {str(root/name):digest(root/name) for root, names in (
        (run, ('lock.json', 'lock.sha256', 'forward_complete.json', 'summary.json', 'completion.json')),
        (audit, ('lock.json', 'lock.sha256', 'started.json', 'verification.json', 'numerical_checks.json', 'independent_summary.json', 'completion.json')))
        for name in names}
    receipt = dict(created_unix=time.time(), reporting_acceptance_only=True, independent_audit_passed=True,
        scientific_gate_passed=False, scientific_gate_defined=False, descriptive_attribution_only=True,
        selection_changed=False, current_validation_changed=False, efficiency_claim=False,
        additional_audit=False, probabilities_or_labels_decoded=False, complete_evidence_inventories_hashed=True,
        actual_value_run=False, optimized_gs_run=False, fresh_confirmation_run=False, general_research_goal_achieved=False,
        source_sha256=digest(Path(__file__)), inputs_sha256=bound, table_rows={k:len(v) for k, v in output.items()},
        scope='Fixed TRAIN attribution only. Full file inventories were hashed; accepted shared replay and independent sampled references are not re-executed. No scientific gate or equivalence claim.',
        outputs_sha256={p.name:digest(p) for p in sorted(out.iterdir())})
    write(out/'acceptance.json', receipt)
    (out/'acceptance.sha256').write_text(digest(out/'acceptance.json')+'\n')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT/'results/round17_size_temperature_attribution')
    parser.add_argument('--audit', type=Path, default=ROOT/'results/round17_size_temperature_independent_audit')
    parser.add_argument('--out', type=Path, default=ROOT/'results/round17_size_temperature_report_tables')
    args = parser.parse_args()
    extract(args.run, args.audit, args.out)


if __name__ == '__main__':
    main()
