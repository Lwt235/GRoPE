"""Decompose sealed R39 scores without labels, new probabilities or fitting."""
from __future__ import annotations
import argparse
import ast
import math
from pathlib import Path
import shutil
import time
import traceback
import numpy as np
from experiments import round39_dual_overlap_screen as OLD
from experiments import round39_schur_components as PRIMARY
from experiments import round39_schur_components_audit as AUDIT

ROOT = OLD.ROOT
OUT = ROOT/'results/round39_schur_diagnostic'
PROTOCOL = 'ROUND39_SCHUR_DIAGNOSTIC_PROTOCOL.md'
SOURCE_ROOTS = (PROTOCOL, 'experiments/round39_schur_diagnostic.py',
    'experiments/round39_schur_components.py', 'experiments/round39_schur_components_audit.py',
    'tests/test_round39_schur_diagnostic.py')
ROLES = ('full', 'point', 'isotropic')
FIELDS = ('td', 'tv', 'rd', 'rv', 's', 'residual', 'radial_delta')
OLD_FIELDS = ('q_anchor', 'q_cov', 'k_anchor', 'k_cov', 'gamma', 'content')+tuple(
    role+'__'+name for role in ROLES for name in ('lograw', 'logfloored', 'logp', 'p', 'winner'))
EPS = 1e-8
sha, read, write, array_sha = OLD.sha, OLD.read, OLD.write, OLD.array_sha
inventory, verify_inventory = OLD.inventory, OLD.verify_inventory


def source_names():
    found, pending = set(), list(SOURCE_ROOTS)
    while pending:
        name = pending.pop()
        if name in found: continue
        path = ROOT/name
        if not path.is_file(): raise ValueError('missing joint source: '+name)
        found.add(name)
        if path.suffix != '.py': continue
        for node in ast.walk(ast.parse(path.read_text())):
            modules = []
            if isinstance(node, ast.Import): modules = [n.name for n in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level: raise ValueError('relative import needs explicit review')
                if node.module: modules = [node.module]+[node.module+'.'+n.name for n in node.names]
            for module in modules:
                parts = module.split('.')
                paths = [Path(*parts).with_suffix('.py'), Path(*parts)/'__init__.py']
                paths += [Path(*parts[:i])/'__init__.py' for i in range(1, len(parts))]
                pending.extend(str(p) for p in paths if (ROOT/p).is_file() and str(p) not in found)
    return sorted(found)


def metadata():
    bindings = {}
    def bind(path, expected=None):
        p = Path(path).resolve()
        if p.name == 'labels.npz' or (p.suffix == '.npz' and '/targets/' in str(p)):
            raise ValueError('diagnostic never opens or hashes labels')
        digest = sha(p)
        if expected is not None and digest != expected: raise ValueError('changed accepted source: '+str(p))
        bindings[str(p)] = digest
    for name in ('results/round39_dual_overlap_root_acceptance_20261101.json',
                 'results/round39_post_result_decision_20261101.json'):
        path = ROOT/name; bind(path)
        for n, h in read(path)['files'].items(): bind(ROOT/n, h)
    accepted = read(ROOT/'results/round39_dual_overlap_root_acceptance_20261101.json')
    if not accepted['accepted_for_development_failure_interpretation'] or accepted['passed_checks'] != 0:
        raise ValueError('accepted fixed failure required')
    lock = OLD.verify(OLD.OUT)
    OLD.completed(OLD.OUT/'forward'); OLD.completed(OLD.OUT/'audit')
    records = read(OLD.OUT/'forward/records.json')
    if [r['pair_id'] for r in records] != [e['pair_id'] for e in lock['entries']]:
        raise ValueError('original ordered population changed')
    for e in lock['entries']: bind(e['input_path'], e['input_sha256'])
    return dict(entries=lock['entries'], records=records, bindings=bindings)


def freeze(out):
    if out.exists(): raise ValueError('preserve existing diagnostic attempt')
    data = metadata()
    sources = {n: sha(ROOT/n) for n in source_names()}
    out.mkdir(parents=True)
    for name, digest in sources.items():
        dst = out/'source_snapshot'/name; dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, dst)
        if sha(dst) != digest: raise ValueError('source changed during copy')
    lock = dict(created_unix=time.time(), source_hashes=sources, runtime=OLD.C.check_runtime(),
                roles=list(ROLES), population=dict(pairs=48, queries_per_role=2304, role_queries=6912, keys=432),
                labels_read=False, new_probabilities=False, scientific_gate_passed=False, goal_complete=False, **data)
    write(out/'lock.json', lock); (out/'lock.sha256').write_text(sha(out/'lock.json')+'\n')


def verify(out):
    lock = read(out/'lock.json')
    if sha(out/'lock.json') != (out/'lock.sha256').read_text().strip(): raise ValueError('changed lock')
    if set(lock['source_hashes']) != set(source_names()) or lock['runtime'] != OLD.C.check_runtime():
        raise ValueError('changed sources or runtime')
    for n, h in lock['source_hashes'].items():
        if sha(ROOT/n) != h or sha(out/'source_snapshot'/n) != h: raise ValueError('changed source: '+n)
    for n, h in lock['bindings'].items():
        if sha(n) != h: raise ValueError('changed binding: '+n)
    return lock


def close(a, b):
    if not math.isfinite(float(a)) or not math.isfinite(float(b)) or not math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-8):
        raise ValueError('gap or continuous diagnostic disagreement')
    return abs(float(a)-float(b))


def anchors(points, K):
    """Half-open original image edges; no clipping or sensor validity."""
    if K.shape != (3, 3) or not np.isfinite(K).all() or not np.array_equal(K[2], [0., 0., 1.]):
        raise ValueError('finite standard key intrinsics required')
    homogeneous = points @ K.T
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        xy = homogeneous[:, :2]/homogeneous[:, 2, None]
    result = []
    for point, pixel in zip(points, xy):
        finite = bool(np.isfinite(pixel).all())
        front = bool(point[2] > 0.)
        inside = bool(finite and -.5 <= pixel[0] < 191.5 and -.5 <= pixel[1] < 143.5)
        eligible = front and inside
        cell = (pixel+.5)/8 if finite else None
        boundary = bool(finite and np.any(cell == np.floor(cell)))
        index = int(math.floor(cell[1])*24+math.floor(cell[0])) if eligible else None
        result.append(dict(anchor_eligible=eligible, anchor_index=index,
            anchor_front=front, projection_finite=finite, anchor_in_fov=inside,
            on_cell_boundary=boundary, projected_xy=pixel.tolist() if finite else None))
    return result


def gap(terms, raw, floor, bias, winner, anchor):
    components = {name: float(terms[name][winner]-terms[name][anchor]) for name in ('td', 'tv', 'rd', 'rv')}
    rawgap = float(raw[winner]-raw[anchor])
    floor_gap = float((floor[winner]-raw[winner])-(floor[anchor]-raw[anchor]))
    content_gap = float(bias[winner]-bias[anchor])
    finalgap = float((floor[winner]+bias[winner])-(floor[anchor]+bias[anchor]))
    summed = math.fsum(components.values())
    raw_error = close(summed, rawgap)
    final_error = close(math.fsum((summed, floor_gap, content_gap)), finalgap)
    T = components['td']+components['tv']
    R = components['rd']+components['rv']
    distance = components['td']+components['rd']
    volume = components['tv']+components['rv']
    return dict(**components, rawgap=rawgap, finalgap=finalgap, T=T, R=R,
        volume=volume, floor_gap=floor_gap, content_gap=content_gap,
        raw_reconstruction_error=raw_error, final_reconstruction_error=final_error,
        radial_distance_reversal=bool(T < -EPS and T+components['rd'] > EPS),
        radial_total_reversal=bool(T < -EPS and T+R > EPS),
        volume_reversal=bool(distance < -EPS and distance+volume > EPS))


def pair_rows(data, K, terms, entry, role):
    raw, floor, lp, p, win = (data[role+'__'+name] for name in ('lograw', 'logfloored', 'logp', 'p', 'winner'))
    if raw.shape != p.shape or floor.shape != p.shape or lp.shape != p.shape or p.shape[1] != 432:
        raise ValueError('all original native key vectors required')
    if win.dtype != np.int32 or not np.array_equal(win, p.argmax(-1).astype(np.int32)):
        raise ValueError('saved winner differs from saved probability')
    reconstruct = math.log(9.)+terms['td']+terms['tv']+terms['rd']+terms['rv']
    err = OLD.normerror(reconstruct, raw)
    if err > 1e-9: raise ValueError('raw decomposition does not match saved score')
    points = anchors(data['q_anchor'], K)
    rows = []
    for index, a in enumerate(points):
        rw, fw, lw, pw, winner = (int(x) for x in (raw[index].argmax(), floor[index].argmax(),
            (floor[index]+data['content'][index]).argmax(), lp[index].argmax(), win[index]))
        row = dict(scene=entry['scene'], pair_id=entry['pair_id'], query_index=index, role=role, **a,
            raw_winner=rw, floored_winner=fw, logits_winner=lw, logp_winner=pw, final_winner=winner,
            floor_changes_first=rw != fw, content_changes_first=fw != lw,
            normalization_changes_first=lw != pw, materialization_changes_first=pw != winner,
            raw_first_miss=None, final_first_miss=None, raw_anchor_below_max=None, final_anchor_below_max=None,
            raw_comparison=None, final_comparison=None)
        vectors = dict(raw=raw[index], floored=floor[index], logits=floor[index]+data['content'][index],
                       logp=lp[index], probability=p[index])
        row['stage_tie_counts'] = {name: int(np.count_nonzero(v == v.max())) for name, v in vectors.items()}
        row['stage_anchor_is_max'] = {name: bool(v[a['anchor_index']] == v.max()) if a['anchor_eligible'] else None
                                      for name, v in vectors.items()}
        if a['anchor_eligible']:
            ai = a['anchor_index']
            row.update(raw_first_miss=rw != ai, final_first_miss=winner != ai,
                raw_anchor_below_max=bool(raw[index, ai] < raw[index, rw]),
                final_anchor_below_max=bool(p[index, ai] < p[index, winner]))
            selected = {name: value[index] for name, value in terms.items()}
            for name, selected_winner in (('raw_comparison', rw), ('final_comparison', winner)):
                row[name] = gap(selected, raw[index], floor[index], data['content'][index], selected_winner, ai)
        rows.append(row)
    return rows, err


def descriptive(values):
    a = np.asarray(values, np.float64)
    if not len(a): return dict(count=0, minimum=None, maximum=None, mean=None, p50=None, p90=None, p99=None)
    if not np.isfinite(a).all(): raise ValueError('nonfinite statistic')
    q = np.quantile(a, [.5, .9, .99], method='linear')
    return dict(count=len(a), minimum=float(a.min()), maximum=float(a.max()), mean=float(a.mean()),
                p50=float(q[0]), p90=float(q[1]), p99=float(q[2]))


def summarize(rows):
    output = {}
    for role in ROLES:
        output[role] = {}
        for scene in ('all', *OLD.C.SCENES):
            selected = [r for r in rows if r['role'] == role and (scene == 'all' or r['scene'] == scene)]
            eligible = [r for r in selected if r['anchor_eligible']]
            counts = {key: sum(bool(r[key]) for r in selected) for key in (
                'anchor_eligible', 'anchor_front', 'projection_finite', 'anchor_in_fov', 'on_cell_boundary',
                'floor_changes_first', 'content_changes_first', 'normalization_changes_first', 'materialization_changes_first',
                'raw_first_miss', 'final_first_miss', 'raw_anchor_below_max', 'final_anchor_below_max')}
            groups = {}
            for group, subset in (('eligible_all', eligible), ('raw_first_miss', [r for r in eligible if r['raw_first_miss']]),
                                  ('final_first_miss', [r for r in eligible if r['final_first_miss']])):
                groups[group] = dict(count=len(subset))
                for comparison in ('raw_comparison', 'final_comparison'):
                    v = [r[comparison] for r in subset]
                    groups[group][comparison] = dict(count=len(v),
                        reversals={k: sum(r[k] for r in v) for k in ('radial_distance_reversal', 'radial_total_reversal', 'volume_reversal')},
                        statistics={k: descriptive([r[k] for r in v]) for k in ('rawgap', 'finalgap', 'T', 'R', 'rd', 'rv', 'volume', 'floor_gap', 'content_gap')})
            stages = {name: dict(rows_with_ties=sum(r['stage_tie_counts'][name] > 1 for r in selected),
                        eligible_anchor_max_count=sum(bool(r['stage_anchor_is_max'][name]) for r in eligible))
                      for name in ('raw', 'floored', 'logits', 'logp', 'probability')}
            output[role][scene] = dict(population=len(selected), counts=counts, stages=stages, groups=groups)
    return output


def compare(a, b):
    if isinstance(a, dict):
        if not isinstance(b, dict) or set(a) != set(b): raise ValueError('schema differs')
        return max([compare(a[k], b[k]) for k in a] or [0.])
    if isinstance(a, list):
        if not isinstance(b, list) or len(a) != len(b): raise ValueError('population differs')
        return max([compare(x, y) for x, y in zip(a, b)] or [0.])
    if a is None or isinstance(a, (str, bool, int)):
        if a != b: raise ValueError('identity/count/flag differs')
        return 0.
    return close(a, b)


def run(out, independent=False):
    started = time.perf_counter()
    lock = verify(out)
    phase = 'audit' if independent else 'diagnostic'
    directory = out/phase
    if directory.exists(): raise ValueError('retain existing attempted stage')
    if independent: OLD.completed(out/'diagnostic')
    directory.mkdir()
    write(directory/'started.json', dict(created_unix=time.time(), lock_sha256=sha(out/'lock.json'), labels_read=False))
    rows, records, checks = [], [], []
    index, role = -1, None
    try:
        saved_records = {r['pair_id']: r for r in read(out/'diagnostic/records.json')} if independent else None
        for index, (entry, rec) in enumerate(zip(lock['entries'], lock['records'])):
            if time.perf_counter()-started > 1200: raise TimeoutError('phase budget')
            if entry['pair_id'] != rec['pair_id']: raise ValueError('ordered pair identity mismatch')
            data = OLD.archive(OLD.OUT/rec['archive_path'], rec['archive_sha256'], OLD_FIELDS, directory)
            for name, val in data.items():
                if array_sha(val) != rec['array_sha256'][name]: raise ValueError('saved array digest mismatch')
            K = OLD.archive(entry['input_path'], entry['input_sha256'], ('k_K',), directory)['k_K']
            if data['q_anchor'].shape != (48, 3) or data['k_anchor'].shape != (432, 3): raise ValueError('original full native population required')
            baseline = None
            if independent:
                saved = saved_records[entry['pair_id']]
                baseline = OLD.archive(out/saved['archive_path'], saved['archive_sha256'], tuple(role+'__'+n for role in ROLES for n in FIELDS), directory)
            arrays = {}
            for role in ROLES:
                function = AUDIT.audit_components if independent else PRIMARY.schur_components
                terms = function(data['q_anchor'], data['q_cov'], data['k_anchor'], data['k_cov'], data['gamma'], role)
                if set(terms) != set(FIELDS) or any(v.dtype != np.float64 or v.shape != (48, 432) or not np.isfinite(v).all() for v in terms.values()) or np.any(terms['s'] <= 0):
                    raise ValueError('invalid component matrix')
                pairrows, reconstruction = pair_rows(data, K, terms, entry, role)
                rows.extend(pairrows)
                errors = {} if baseline is None else {n: OLD.normerror(terms[n], baseline[role+'__'+n]) for n in FIELDS}
                if errors and max(errors.values()) > 1e-9: raise ValueError('independent decomposition differs')
                checks.append(dict(pair_id=entry['pair_id'], role=role, queries=48, keys=432,
                    raw_reconstruction_error=reconstruction, component_errors=errors, passed=True))
                arrays.update({role+'__'+n: v for n, v in terms.items()})
            if not independent:
                path = directory/(entry['pair_id']+'.npz'); np.savez_compressed(path, **arrays)
                records.append(dict(pair_id=entry['pair_id'], archive_path=str(path.relative_to(out)), archive_sha256=sha(path),
                    array_sha256={n: array_sha(v) for n, v in arrays.items()}))
                write(directory/'records.json', records)
            write(directory/'checks.json', checks); write(directory/'rows.json', rows)
            print('R39_SCHUR', phase, index+1, entry['pair_id'], flush=True)
        expected = [(e['pair_id'], role, i) for e in lock['entries'] for role in ROLES for i in range(48)]
        if [(r['pair_id'], r['role'], r['query_index']) for r in rows] != expected or len(rows) != 6912 or len(checks) != 144:
            raise ValueError('full ordered population required')
        summary = summarize(rows); write(directory/'summary.json', summary)
        differences = None
        if independent:
            differences = dict(rows=compare(read(out/'diagnostic/rows.json'), rows), summary=compare(read(out/'diagnostic/summary.json'), summary))
            write(directory/'verification.json', dict(passed=True, maximum_discrepancy=differences,
                independent_numerical_implementations=True, shared_row_bookkeeping=True, labels_read=False, new_probabilities=False))
        byte_count = sum(p.stat().st_size for p in (out/'diagnostic').glob('*.npz'))
        if byte_count > 512*(1<<20): raise ValueError('saved component budget exceeded')
        verify(out)
        files = inventory(directory); elapsed = time.perf_counter()-started
        if elapsed > 1200: raise TimeoutError('phase budget including verification and inventory')
        write(directory/'completion.json', dict(passed=True, files=files, finished_unix=time.time(), phase_wall_seconds=elapsed,
            role_queries=6912, queries_per_role=2304, pairs=48, role_pair_checks=144, component_npz_bytes=byte_count,
            labels_read=False, new_probabilities=False, scientific_gate_passed=False, goal_complete=False))
    except BaseException as ex:
        write(directory/'rows.json', rows); write(directory/'checks.json', checks)
        if not independent: write(directory/'records.json', records)
        write(directory/'failure.json', dict(error=str(ex), error_type=type(ex).__name__, traceback=traceback.format_exc(),
            finished_unix=time.time(), completed_role_rows=len(rows), failed_pair_index=index,
            failed_role=role, unattempted_pair_ids=[e['pair_id'] for e in lock['entries'][index+1:]],
            labels_read=False, new_probabilities=False, goal_complete=False))
        raise


def main():
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=('freeze', 'diagnostic', 'audit'))
    p.add_argument('--out', type=Path, default=OUT); args = p.parse_args(); OLD.C.configure_runtime()
    if args.stage == 'freeze': freeze(args.out)
    else: run(args.out, args.stage == 'audit')


if __name__ == '__main__': main()
