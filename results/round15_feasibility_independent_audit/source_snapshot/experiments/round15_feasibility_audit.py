"""Independent center-feasibility audit. No producer metric/selector imports.

The canonical audit table uses integer correct/valid counts for every pair,
family, global h/T configuration, scene and query scale. File adapters and
source-freeze binding are added before reading any new experimental result.
"""
from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

SCENES = ('harvard_c3/hv_c3_1', 'harvard_c5/hv_c5_1', 'harvard_c6/hv_c6_1')
QUERIES = ('g8', 'g16', 'g24')
FAMILIES = ('quartic_full', 'quartic_point', 'gaussian_point')
GRID = tuple((h, t) for h in (.25, .5, 1., 2.) for t in (0., 5., 10.))
ROOT = Path(__file__).resolve().parents[1]


def independent_center_selection(rows, *, expected_pairs=16):
    """Equal-scene PCK, one global point configuration, all three protections.

    No KL or regional winner selection occurs at this stage. A legacy point
    configuration is included even if either newly selected point is weaker
    at a particular scale. Comparison tolerance is the declared .005 only.
    """
    groups = defaultdict(list)
    populations = {}
    identities = set()
    for row in rows:
        family, cfg = row['family'], (float(row['h']), float(row['temperature']))
        scene, query, pair = row['scene'], row['query'], tuple(row['pair'])
        if family not in FAMILIES or cfg not in GRID or scene not in SCENES or query not in QUERIES:
            raise ValueError('unexpected family/configuration/cohort')
        good, total = row['correct'], row['valid']
        if not isinstance(good, int) or not isinstance(total, int) or not 0 <= good <= total:
            raise ValueError('invalid integer counts')
        identity = (family, cfg, scene, query, pair)
        if identity in identities:
            raise ValueError('duplicate metric row')
        identities.add(identity)
        population_key = (scene, query, pair)
        if population_key in populations and populations[population_key] != total:
            raise ValueError('changed center population')
        populations[population_key] = total
        groups[family, cfg, scene, query].append((pair, good, total))

    scores = {}
    for family in FAMILIES:
        for cfg in GRID:
            by_query = {}
            by_scene = {s: {} for s in SCENES}
            for query in QUERIES:
                for scene in SCENES:
                    group = groups[family, cfg, scene, query]
                    if len(group) != expected_pairs:
                        raise ValueError('missing pair population')
                    reference_pairs = {p for s, q, p in populations if s == scene and q == query}
                    if {p for p, _, _ in group} != reference_pairs:
                        raise ValueError('pair identity mismatch')
                    denominator = sum(v for _, _, v in group)
                    if denominator <= 0:
                        raise ValueError('empty center population')
                    by_scene[scene][query] = Fraction(sum(g for _, g, _ in group), denominator)
                by_query[query] = sum(by_scene[s][query] for s in SCENES) / len(SCENES)
            scores[family, cfg] = dict(by_query=by_query, by_scene=by_scene,
                                       macro=sum(by_query.values()) / len(QUERIES))

    points = {}
    for family in ('gaussian_point', 'quartic_point'):
        # Python max preserves the declared GRID order for exact ties.
        cfg = max(GRID, key=lambda c: scores[family, c]['macro'])
        points[family] = cfg
    references = (
        ('selected_gaussian_point', 'gaussian_point', points['gaussian_point']),
        ('selected_quartic_point', 'quartic_point', points['quartic_point']),
        ('legacy_point_pck', 'gaussian_point', (.5, 5.)),
    )
    eligible, checks = [], []
    for cfg in GRID:
        all_pass = True
        for query in QUERIES:
            candidate = scores['quartic_full', cfg]['by_query'][query]
            for name, family, reference_cfg in references:
                reference = scores[family, reference_cfg]['by_query'][query]
                difference = candidate - reference
                passed = difference >= -Fraction(1, 200)
                checks.append(dict(h=cfg[0], temperature=cfg[1], query=query,
                                   reference=name, difference_pp=float(100*difference), passed=passed))
                all_pass = all_pass and passed
        if all_pass:
            eligible.append(dict(h=cfg[0], temperature=cfg[1]))
    return dict(scores=scores, point_configurations=points, checks=checks,
                eligible=eligible, stop_candidate=len(eligible) == 0,
                regional_primary_selected=False, scientific_gate_passed=False)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def array_sha(value):
    a = np.ascontiguousarray(value)
    return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)


def verify_files(directory, files):
    for name, digest in files.items():
        if sha(Path(directory)/name) != digest:
            raise ValueError('changed artifact: '+name)


def verify_constraint_table(saved, expected, eligible):
    """Require all 108 distinct guards, including failed configurations."""
    reference_names = {'gaussian_point': 'selected_gaussian_point',
                       'quartic_point': 'selected_quartic_point',
                       'legacy_point_pck': 'legacy_point_pck'}
    expected_checks = {(GRID.index((c['h'], c['temperature'])), c['query'], c['reference']): c
                       for c in expected}
    if len(saved) != 12 or {v['grid_index'] for v in saved} != set(range(12)):
        raise ValueError('missing or duplicate configuration guards')
    seen = set()
    for item in saved:
        i = item['grid_index']; h, temperature = GRID[i]
        if (item['config'] != dict(h=h, temperature=temperature)
                or item['feasible'] != (i in eligible) or len(item['checks']) != 9):
            raise ValueError('invalid configuration guards')
        for c in item['checks']:
            identity = (i, c['query'], reference_names[c['reference']])
            if identity in seen or identity not in expected_checks:
                raise ValueError('duplicate or unexpected constraint')
            seen.add(identity)
            target = expected_checks[identity]
            if c['passed'] != target['passed'] or abs(c['difference_pp']-target['difference_pp']) > 1e-10:
                raise ValueError('center constraint differs')
    if seen != set(expected_checks) or len(seen) != 108:
        raise ValueError('incomplete constraint coverage')


def freeze(out, run):
    """Freeze this independent implementation before opening new results."""
    if out.exists():
        raise ValueError('refusing audit freeze overwrite')
    run = run.resolve()
    producer = read(run/'lock.json')  # Only prior source/input metadata.
    if sha(run/'lock.json') != (run/'lock.sha256').read_text().strip():
        raise ValueError('producer lock mismatch')
    paths = ('experiments/round15_feasibility_audit.py', 'tests/test_round15_feasibility_audit.py')
    sources = {p: sha(ROOT/p) for p in paths}
    out.mkdir(parents=True)
    for name in paths:
        target = out/'source_snapshot'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT/name).read_bytes())
    write(out/'lock.json', dict(created_unix=time.time(), source_hashes=sources,
        producer_sources=producer['source_hashes'], producer_lock_sha256=sha(run/'lock.json'),
        run=str(run), before_new_arrays_metrics_summary=True, scientific_gate_passed=False,
        row_counts_exact=True, rational_selection=True, float_report_tolerance=1e-12,
        production_preflight_geometry_probability_tolerance=1e-12,
        production_preflight_content_probability_tolerance=1e-5,
        independent_reference_probability_tolerance=1e-4,
        coverage='All 5184 probability arrays and center metrics; all 12 configurations and 108 constraints; all 216 frozen real-input quartic numerical matrices.'))
    (out/'lock.sha256').write_text(sha(out/'lock.json')+'\n')


def audit(out):
    started = time.time()
    lock = read(out/'lock.json')
    if sha(out/'lock.json') != (out/'lock.sha256').read_text().strip():
        raise ValueError('audit lock mismatch')
    verify_files(ROOT, lock['source_hashes'])
    verify_files(out/'source_snapshot', lock['source_hashes'])
    run = Path(lock['run']); producer = read(run/'lock.json')
    if sha(run/'lock.json') != lock['producer_lock_sha256']:
        raise ValueError('producer lock changed')
    verify_files(ROOT, lock['producer_sources'])
    verify_files(run/'source_snapshot', lock['producer_sources'])
    verify_files(Path('/'), producer['inputs']); verify_files(Path('/'), producer['manifests'])
    complete = read(run/'completion.json'); verify_files(run, complete['files'])
    if complete['lock_sha256'] != sha(run/'lock.json') or complete['summary_sha256'] != sha(run/'summary.json'):
        raise ValueError('completion chain mismatch')
    seal = read(run/'forward_complete.json')
    if seal['lock_sha256'] != sha(run/'lock.json') or seal['labels_decoded'] or seal['array_count'] != 5184:
        raise ValueError('global forward seal invalid')
    numerical_root = Path(producer['forward_preflight']['root'])
    numerical_complete = read(numerical_root/'completion.json')
    verify_files(numerical_root, numerical_complete['files'])
    if sha(numerical_root/'completion.json') != producer['forward_preflight']['completion_sha256']:
        raise ValueError('real numerical preflight changed')
    numeric_lock = read(numerical_root/'lock.json')
    # Native g8 centers reconstructed independently, without partition helpers.
    yy, xx = np.meshgrid(np.arange(18)*8+3.5, np.arange(24)*8+3.5, indexing='ij')
    centers = np.column_stack((xx.ravel(), yy.ravel()))
    rows = []; numerical_checks = []; valid_totals = {}
    maximum_probability_sum_error = 0.
    expected_fields = {f'{f}__c{i:02d}__{q}' for f in FAMILIES for i in range(12) for q in QUERIES}
    for scene in SCENES:
        slug = scene.replace('/', '__'); forward = run/'forward'/slug; evaluation = run/'evaluation'/slug
        fc, ec = read(forward/'completion.json'), read(evaluation/'completion.json')
        verify_files(forward, fc['files']); verify_files(evaluation, ec['files'])
        if fc['labels_decoded'] or not ec['center_labels_decoded'] or ec['regional_labels_decoded']:
            raise ValueError('unexpected supervision access')
        if not producer['created_unix'] <= lock['created_unix'] <= fc['started_unix'] <= fc['finished_unix'] <= seal['sealed_unix'] <= ec['started_unix'] <= ec['finished_unix']:
            raise ValueError('stage chronology mismatch')
        if seal['forward_chain'][scene]['completion_sha256'] != sha(forward/'completion.json'):
            raise ValueError('unbound forward shard')
        if ec['forward_complete_sha256'] != sha(run/'forward_complete.json'):
            raise ValueError('unbound evaluation stage')
        recorded_rows = read(evaluation/'rows.json'); recorded_forward = read(forward/'records.json')
        identities = lambda r: (tuple(r['pair']), r['family'], r['grid_index'], r['query'])
        recorded = {identities(r): r for r in recorded_rows}
        forwards = {identities(r): r for r in recorded_forward}
        if len(recorded) != len(recorded_rows) or len(forwards) != len(recorded_forward) or set(recorded) != set(forwards):
            raise ValueError('duplicate or unmatched metric records')
        entries = sorted((e for e in producer['target_contract']['entries'] if e['scene'] == scene), key=lambda e: e['path'])
        if len(entries) != 16:
            raise ValueError('expected exact training pairs')
        for pair_index, entry in enumerate(entries):
            target = Path(entry['target_path']); path = forward/(Path(entry['path']).name+'.npz')
            if sha(target) != entry['target_sha256']:
                raise ValueError('center target hash mismatch')
            digest = sha(path)
            with np.load(target, allow_pickle=False) as z:
                targets = {q: (z[q+'_point_xy'], z[q+'_point_visible']) for q in QUERIES}
            with np.load(path, allow_pickle=False) as arrays:
                if set(arrays.files) != expected_fields:
                    raise ValueError('missing or extra prediction')
                for family in FAMILIES:
                    for index, (h, temperature) in enumerate(GRID):
                        for q in QUERIES:
                            field = f'{family}__c{index:02d}__{q}'; p = arrays[field]
                            count = dict(g8=432, g16=108, g24=48)[q]
                            if p.shape != (count, 432) or p.dtype != np.float64 or not np.isfinite(p).all() or (p < 0).any():
                                raise ValueError('invalid probability array')
                            error = float(np.max(np.abs(p.sum(1)-1)))
                            maximum_probability_sum_error = max(maximum_probability_sum_error, error)
                            if error > 1e-12:
                                raise ValueError('probabilities not normalized')
                            item = recorded[(tuple(entry['pair']), family, index, q)]
                            fr = forwards[(tuple(entry['pair']), family, index, q)]
                            if any(item.get(k) != v for k, v in fr.items()):
                                raise ValueError('metric row differs from forward commitment')
                            if item['archive_sha256'] != digest or item['array_sha256'] != array_sha(p) or item['target_sha256'] != entry['target_sha256']:
                                raise ValueError('array or label provenance changed')
                            if (item['field'] != field or item['config'] != dict(h=h, temperature=temperature)
                                    or item['archive'] != path.name or item['shape'] != list(p.shape)
                                    or item['dtype'] != str(p.dtype)
                                    or item['input_sha256'] != producer['inputs'][str(Path(entry['path'])/'inputs.npz')]):
                                raise ValueError('configuration identity mismatch')
                            point, valid = targets[q]
                            if valid.dtype != np.bool_ or valid.shape != (count,) or point.shape != (count, 2) or not np.isfinite(point[valid]).all():
                                raise ValueError('invalid center targets')
                            winner = p.argmax(1)
                            squared_distance = np.einsum('ij,ij->i', centers[winner[valid]]-point[valid], centers[winner[valid]]-point[valid])
                            correct, total = int(np.count_nonzero(squared_distance <= 64)), int(np.count_nonzero(valid))
                            if item['pck_count'] != correct or item['n_point'] != total or item['n_query'] != count or item['argmax_sha256'] != array_sha(winner):
                                raise ValueError('native argmax or center metric mismatch')
                            rows.append(dict(scene=scene, pair=entry['pair'], family=family, h=h, temperature=temperature,
                                             query=q, correct=correct, valid=total))
                            valid_totals[(scene, tuple(entry['pair']), q)] = total
                            if pair_index == 0 and family != 'gaussian_point':
                                indices = np.asarray(numeric_lock['query_indices'][q[1:]])
                                archive = numerical_root/'arrays'/slug/f'{q}_{family}_h{h:g}.npz'
                                with np.load(archive, allow_pickle=False) as reference:
                                    same = float(np.max(np.abs(p[indices]-reference[f'actual_T{temperature:g}'])))
                                    independent = float(np.max(np.abs(p[indices]-reference[f'reference96_T{temperature:g}'])))
                                numerical_checks.append(dict(scene=scene, family=family, h=h, temperature=temperature, query=q,
                                    producer_difference=same, independent_difference=independent))
                                # Actual production prepares all queries while the
                                # numerical probe multiplies eight selected rows.
                                # Float32 GEMM may select a different reduction;
                                # pure geometry must still agree to 1e-12. The
                                # independent final-probability bound stays 1e-4.
                                comparison_tolerance = 1e-12 if temperature == 0 else 1e-5
                                if same > comparison_tolerance or independent > 1e-4:
                                    raise ValueError('feasibility forward differs from frozen numerical reference')
        if len(recorded) != 1728:
            raise ValueError('wrong complete scene row count')
    independent = independent_center_selection(rows)
    result = read(run/'summary.json')
    differences = []
    for family in FAMILIES:
        for i, cfg in enumerate(GRID):
            score = independent['scores'][family, cfg]; saved = result['metrics'][family][i]
            differences.append(abs(float(score['macro'])-saved['equal_scene_equal_scale_pck8']))
            for q in QUERIES:
                differences.append(abs(float(score['by_query'][q])-saved['by_query'][q]))
                for scene in SCENES:
                    differences.append(abs(float(score['by_scene'][scene][q])-saved['by_scene_query'][scene][q]['pck8']))
    if max(differences) > 1e-12:
        raise ValueError('reported PCK summary differs from independent rational calculation')
    for family, cfg in independent['point_configurations'].items():
        selection = result['point_localization_selections'][family]
        if (selection['grid_index'] != GRID.index(cfg)
                or selection['config'] != dict(h=cfg[0], temperature=cfg[1])):
            raise ValueError('global point configuration or tie order mismatch')
    eligible = [GRID.index((v['h'], v['temperature'])) for v in independent['eligible']]
    if (eligible != result['feasible_grid_indices'] or result['feasible'] is not bool(eligible)
            or result['decision_pending_independent_audit'] != (
                'complete_full_regional_calibration' if eligible else 'stop_unchanged_candidate')):
        raise ValueError('feasible set mismatch')
    if result['legacy_point'] != dict(grid_index=4, config=dict(h=.5, temperature=5.)):
        raise ValueError('legacy point reference changed')
    verify_constraint_table(result['feasibility'], independent['checks'], eligible)
    if len(rows) != 5184 or len(numerical_checks) != 216:
        raise ValueError('incomplete independent audit population')
    if result['final_primary_selected'] or result['regional_calibration_done'] or result['scientific_gate_passed'] or result['fresh_test_read']:
        raise ValueError('stage overclaims completion or scope')
    verification = dict(passed=True, feasibility_lock_sha256=sha(run/'lock.json'),
        feasibility_forward_complete_sha256=sha(run/'forward_complete.json'),
        feasibility_summary_sha256=sha(run/'summary.json'), feasibility_completion_sha256=sha(run/'completion.json'),
        independent_audit_lock_sha256=sha(out/'lock.json'), probability_arrays=len(rows),
        independent_numerical_matrices=len(numerical_checks), numerical_checks=numerical_checks,
        maximum_probability_sum_error=maximum_probability_sum_error, maximum_summary_error=max(differences),
        feasible_grid_indices=eligible, stop_candidate=not eligible, scientific_gate_passed=False,
        final_primary_selected=False, regional_labels_decoded=False, sensor_decoded=False,
        validation_or_test_read=False, seconds=time.time()-started)
    write(out/'verification.json', verification)
    write(out/'completion.json', dict(files={str(p.relative_to(out)): sha(p) for p in out.rglob('*') if p.is_file()},
                                      finished_unix=time.time(), passed=True))
    print(json.dumps({k: v for k, v in verification.items() if k != 'numerical_checks'}, indent=2))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=('freeze', 'audit'))
    parser.add_argument('--out', type=Path, default=ROOT/'results/round15_feasibility_independent_audit')
    parser.add_argument('--run', type=Path, default=ROOT/'results/round15_feasibility')
    args = parser.parse_args()
    freeze(args.out, args.run) if args.stage == 'freeze' else audit(args.out)


if __name__ == '__main__':
    main()
