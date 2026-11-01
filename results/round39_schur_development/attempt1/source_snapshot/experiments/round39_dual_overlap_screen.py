"""Frozen R39 development stages. Predictions/numeric audit precede labels."""
from __future__ import annotations
import argparse
import ast
import json
from pathlib import Path
import shutil
import time
import traceback
import numpy as np
from models import round39_dual_overlap_pe as P
from experiments import round39_overlap_audit_numeric as N
from experiments import round38_rms_anchor_contract as C
from experiments import round38_rms_anchor_inputs as I
from experiments import round38_rms_anchor_evaluate as E
from experiments import round20_anchor_fixed_metrics as M
from experiments import round25_student_metrics as A

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/round39_dual_overlap'
PROTOCOL = 'ROUND39_DUAL_OVERLAP_PROTOCOL.md'
SOURCE_ROOTS = (PROTOCOL, 'models/round39_dual_overlap_pe.py',
    'experiments/round39_dual_overlap_screen.py',
    'experiments/round39_overlap_audit_numeric.py',
    'tests/test_round39_dual_overlap.py',
    'docs/gaussian_overlap_measure_pe_reserve_20261101.md',
    'docs/research_gate_map_20261101.md')
sha, read, write, array_sha = C.sha, C.read, C.write, C.array_sha
inventory, verify_inventory = C.inventory, C.verify_inventory
FIELDS = P.OBS_FIELDS+('q_points3', 'q_pixel_cov3')
ROLE_FIELDS = ('lograw', 'logfloored', 'logp', 'p', 'winner')


def source_names():
    found, pending = set(), list(SOURCE_ROOTS)
    while pending:
        name = pending.pop()
        if name in found: continue
        path = ROOT/name
        if not path.is_file(): raise ValueError('source missing: '+name)
        found.add(name)
        if path.suffix != '.py': continue
        for node in ast.walk(ast.parse(path.read_text())):
            modules = []
            if isinstance(node, ast.Import): modules = [x.name for x in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level: raise ValueError('relative source import')
                if node.module: modules = [node.module]+[node.module+'.'+x.name for x in node.names]
            for mod in modules:
                parts = mod.split('.')
                paths = [Path(*parts).with_suffix('.py'), Path(*parts)/'__init__.py']
                paths += [Path(*parts[:i])/'__init__.py' for i in range(1, len(parts))]
                pending.extend(str(p) for p in paths if (ROOT/p).is_file() and str(p) not in found)
    return sorted(found)


def metadata():
    bindings = {}
    def bind(path, expected=None):
        p = Path(path).resolve()
        if p.suffix == '.npz' and (p.name == 'labels.npz' or '/targets/' in str(p)):
            raise ValueError('label-byte access forbidden at freeze')
        digest = sha(p)
        if expected is not None and digest != expected: raise ValueError('changed inherited artifact: '+str(p))
        bindings[str(p)] = digest
    for name in ('results/round38_rms_anchor_root_acceptance_20261101.json',
                 'results/round38_saved_output_diagnostic_root_acceptance_20261101.json',
                 'results/round20_anchor_root_acceptance_20261031.json'):
        bind(ROOT/name)
        receipt = read(ROOT/name)
        for path, digest in receipt['files'].items(): bind(ROOT/path, digest)
    oldpath = ROOT/'results/round38_rms_anchor_screen/lock.json'
    bind(oldpath, (oldpath.parent/'lock.sha256').read_text().strip())
    old = read(oldpath)
    entries = old['entries']
    if len(entries) != 48 or len({e['pair_id'] for e in entries}) != 48 or any(sum(e['scene'] == s for e in entries) != 16 for s in C.SCENES):
        raise ValueError('complete original old TRAIN cohort required')
    for e in entries:
        for field in ('input', 'cache', 'manifest'): bind(e[field+'_path'], e[field+'_sha256'])
    bind(old['partitions_path'], old['partitions_sha256'])
    bind(ROOT/'results/round20_anchor_fixed/summary.json')
    return dict(entries=entries, partitions_path=old['partitions_path'],
                partitions_sha256=old['partitions_sha256'], bindings=bindings)


def freeze(out):
    if out.exists(): raise ValueError('preserve existing attempt')
    data = metadata()
    sources = {n: sha(ROOT/n) for n in source_names()}
    out.mkdir(parents=True)
    for n, digest in sources.items():
        dst = out/'source_snapshot'/n; dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/n, dst)
        if sha(dst) != digest: raise ValueError('snapshot changed')
    lock = dict(created_unix=time.time(), source_hashes=sources, runtime=C.check_runtime(),
                roles=list(P.ROLES), primary='full', labels_read=False,
                scientific_gate_passed=False, goal_complete=False, **data)
    write(out/'lock.json', lock)
    (out/'lock.sha256').write_text(sha(out/'lock.json')+'\n')
    return lock


def verify(out):
    lock = read(out/'lock.json')
    if sha(out/'lock.json') != (out/'lock.sha256').read_text().strip(): raise ValueError('changed lock')
    if set(lock['source_hashes']) != set(source_names()) or lock['runtime'] != C.check_runtime(): raise ValueError('changed source closure/runtime')
    for name, digest in lock['source_hashes'].items():
        if sha(ROOT/name) != digest or sha(out/'source_snapshot'/name) != digest: raise ValueError('changed source: '+name)
    for path, digest in lock['bindings'].items():
        if sha(path) != digest: raise ValueError('changed binding: '+path)
    return lock


def archive(path, digest, fields, stage):
    path = Path(path)
    with (stage/'reads.jsonl').open('a') as f:
        f.write(json.dumps(dict(event='attempt', path=str(path), fields=list(fields), unix=time.time()))+'\n')
    if sha(path) != digest: raise ValueError('changed archive: '+str(path))
    data = {}
    with np.load(path, allow_pickle=False) as z:
        for name in fields:
            data[name] = z[name]
            with (stage/'reads.jsonl').open('a') as f:
                f.write(json.dumps(dict(event='decoded', path=str(path), field=name))+'\n')
    return data


def normerror(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all(): raise ValueError('shape/finite mismatch')
    return float(np.max(abs(a-b)/np.maximum(1., np.maximum(abs(a), abs(b)))))


def load_input(e, stage):
    inputs = archive(e['input_path'], e['input_sha256'], FIELDS, stage)
    obs = {k: inputs[k] for k in P.OBS_FIELDS}
    cache = archive(e['cache_path'], e['cache_sha256'], I.CACHE_FIELDS, stage)
    for name, value in cache.items():
        if array_sha(value) != e['cache_array_sha256'][name]: raise ValueError('cache array identity differs')
    content = I.prepare_content(cache)['content_bias64']
    _, points, within, _ = P.dense(obs['q_predicted_depth'], obs['q_K'])
    error = max(normerror(points.reshape(-1, 3), inputs['q_points3']),
                normerror(within.reshape(-1, 3, 3), inputs['q_pixel_cov3']))
    if error > 1e-10: raise ValueError('archived dense geometry differs')
    return obs, content, error


def completed(directory):
    comp = read(directory/'completion.json')
    verify_inventory(directory, comp['files'], exclude=('completion.json',))
    if not comp['passed']: raise ValueError('previous stage incomplete')
    return comp


def stage(out, kind):
    started = time.perf_counter()
    lock = verify(out)
    directory = out/kind
    if directory.exists(): raise ValueError('retain existing stage attempt')
    if kind == 'audit': completed(out/'forward')
    directory.mkdir()
    write(directory/'started.json', dict(created_unix=time.time(), lock_sha256=sha(out/'lock.json'), labels_read=False))
    records, checks = [], []
    index, current_role = -1, None
    try:
        forward = {r['pair_id']: r for r in read(out/'forward/records.json')} if kind == 'audit' else None
        for index, e in enumerate(lock['entries']):
            if time.perf_counter()-started > 1200: raise TimeoutError('frozen phase budget')
            obs, content, dense_error = load_input(e, directory)
            if kind == 'forward':
                geom = P.prepare(obs)
                arrays = dict(geom)
                arrays['content'] = content
                for role in P.ROLES:
                    values = P.score(**geom, content=content, role=role)
                    arrays.update({role+'__'+k: v for k, v in values.items()})
                path = directory/(e['pair_id']+'.npz')
                np.savez_compressed(path, **arrays)
                records.append(dict(pair_id=e['pair_id'], scene=e['scene'], pair=e['pair'],
                    archive_path=str(path.relative_to(out)), archive_sha256=sha(path),
                    array_sha256={k: array_sha(v) for k, v in arrays.items()},
                    dense_geometry_error=dense_error, gamma=float(geom['gamma'])))
                write(directory/'records.json', records)
            else:
                rec = forward[e['pair_id']]
                saved = archive(out/rec['archive_path'], rec['archive_sha256'], tuple(rec['array_sha256']), directory)
                for name, val in saved.items():
                    if array_sha(val) != rec['array_sha256'][name]: raise ValueError('saved array changed')
                independent = N.audit_prepare(obs)
                geo_error = max(normerror(saved[k], v) for k, v in independent.items())
                if geo_error > 1e-10 or not np.array_equal(saved['content'], content): raise ValueError('independent input failure')
                for role in P.ROLES:
                    current_role = role
                    values = N.audit_scores(independent['q_anchor'], independent['q_cov'],
                        independent['k_anchor'], independent['k_cov'], independent['gamma'], content, role)
                    errors = {k: normerror(saved[role+'__'+k], values[k]) for k in ('lograw', 'logfloored', 'logp')}
                    p_error = float(np.max(abs(saved[role+'__p']-values['p'])))
                    normalization = float(np.max(abs(saved[role+'__p'].sum(-1)-1.)))
                    winner = np.array_equal(saved[role+'__winner'], values['winner'])
                    if max(errors.values()) > 1e-9 or p_error > 1e-10 or normalization > 2e-12 or not winner:
                        raise ValueError('independent numerical failure '+e['pair_id']+'/'+role)
                    checks.append(dict(pair_id=e['pair_id'], role=role, queries=48, keys=432,
                        geometry_error=geo_error, dense_geometry_error=dense_error, score_error=errors,
                        probability_error=p_error, normalization_error=normalization, winner_exact=bool(winner), passed=True))
                write(directory/'checks.json', checks)
            print('R39', kind, index+1, e['pair_id'], flush=True)
        if kind == 'forward': write(directory/'records.json', records)
        else: write(directory/'checks.json', checks)
        size = sum(p.stat().st_size for p in (out/'forward').glob('*.npz'))
        if size > 512*(1<<20) or time.perf_counter()-started > 1200: raise ValueError('frozen resource budget exceeded')
        verify(out)
        files = inventory(directory)
        elapsed = time.perf_counter()-started
        if elapsed > 1200: raise TimeoutError('frozen phase budget including verification/inventory')
        write(directory/'completion.json', dict(passed=True, files=files,
            started_unix=read(directory/'started.json')['created_unix'], finished_unix=time.time(),
            phase_wall_seconds=elapsed, pairs=48, queries=2304,
            roles=list(P.ROLES), role_queries=6912, role_pair_checks=len(checks),
            forward_npz_bytes=size, labels_read=False, scientific_gate_passed=False, goal_complete=False))
    except BaseException as ex:
        write(directory/'records.json' if kind == 'forward' else directory/'checks.json', records if kind == 'forward' else checks)
        write(directory/'failure.json', dict(error_type=type(ex).__name__, error=str(ex), traceback=traceback.format_exc(),
            finished_unix=time.time(), completed_pairs=len(records), completed_role_pair_checks=len(checks),
            failed_pair_index=index, failed_pair_id=lock['entries'][index]['pair_id'] if index >= 0 else None,
            failed_role=current_role, unattempted_pair_ids=[e['pair_id'] for e in lock['entries'][index+1:]],
            labels_read=False, goal_complete=False))
        raise


def evaluate(out, independent=False):
    started = time.perf_counter()
    lock = verify(out)
    forward_complete, audit_complete = completed(out/'forward'), completed(out/'audit')
    acceptance = read(out/'prelabel_root_acceptance.json')
    if acceptance.get('accepted_for_label_stage') is not True or acceptance.get('labels_read') is not False: raise ValueError('explicit root acceptance required')
    if (acceptance.get('pairs') != 48 or acceptance.get('queries') != 2304
            or acceptance.get('role_pair_checks') != 144 or acceptance.get('numerical_audit_passed') is not True):
        raise ValueError('explicit full-population numerical acceptance required')
    for path in (out/'lock.json', out/'forward/completion.json', out/'audit/completion.json'):
        if acceptance['files'].get(str(path.relative_to(ROOT))) != sha(path):
            raise ValueError('required prelabel binding missing: '+str(path))
    if (forward_complete.get('pairs') != 48 or forward_complete.get('queries') != 2304
            or audit_complete.get('pairs') != 48 or audit_complete.get('queries') != 2304
            or audit_complete.get('role_pair_checks') != 144
            or forward_complete.get('roles') != list(P.ROLES) or audit_complete.get('roles') != list(P.ROLES)):
        raise ValueError('completed population mismatch')
    for path, digest in acceptance['files'].items():
        if sha(ROOT/path) != digest: raise ValueError('prelabel acceptance changed')
    name = 'metrics_audit' if independent else 'evaluation'
    directory = out/name
    if directory.exists(): raise ValueError('retain previous metric attempt')
    if independent: completed(out/'evaluation')
    directory.mkdir()
    write(directory/'started.json', dict(created_unix=time.time(), root_acceptance_sha256=sha(out/'prelabel_root_acceptance.json')))
    try:
        z = archive(lock['partitions_path'], lock['partitions_sha256'],
                    tuple(q+'__'+f for q in ('g8', 'g16', 'g24') for f in ('bounds', 'owner')), directory)
        parts = {q: {f: z[q+'__'+f] for f in ('bounds', 'owner')} for q in ('g8', 'g16', 'g24')}
        source = {r['pair_id']: r for r in read(out/'forward/records.json')}
        rows = {role: [] for role in P.ROLES}
        module = A if independent else M
        for e in lock['entries']:
            if time.perf_counter()-started > 1200: raise TimeoutError('frozen metrics budget')
            labels = archive(e['label_path'], e['label_sha256'], ('gt_xy', 'gt_visible'), directory)
            target = archive(e['target_path'], e['target_sha256'], ('g24_point_xy', 'g24_point_visible'), directory)
            rec = source[e['pair_id']]
            data = archive(out/rec['archive_path'], rec['archive_sha256'], tuple(role+'__'+f for role in P.ROLES for f in ('p', 'winner')), directory)
            direct, common = module.target_counts(labels, parts, 'g24', 'g8')
            for role in P.ROLES:
                p, win = data[role+'__p'], data[role+'__winner']
                if not np.array_equal(win, p.argmax(-1).astype(np.int32)): raise ValueError('winner differs from sealed probability')
                rows[role].append(dict(scene=e['scene'], pair=e['pair'], **E.CONDITION,
                    region=module.region_metrics(p, parts, E.CONDITION, direct, common),
                    center=module.center_metrics(win, parts['g8']['bounds'], target['g24_point_xy'], target['g24_point_visible'])))
        old = read(ROOT/'results/round20_anchor_fixed/summary.json')
        summaries = {role: (A.summarize if independent else E.primary_summary)(rows[role], lock['entries']) for role in P.ROLES}
        checks = A.screen(summaries['full'], old) if independent else E.primary_checks(summaries['full'], old)
        simple_checks = dict(passed=checks['passed'], checks=[{k: row[k] for k in ('reference', 'metric', 'candidate_value', 'reference_value', 'bound', 'passed')} for row in checks['checks']])
        write(directory/'rows.json', rows); write(directory/'summary.json', summaries); write(directory/'screen.json', simple_checks)
        difference = None
        if independent:
            difference = {n: E.discrepancy(read(out/'evaluation'/(n+'.json')), v) for n, v in (('rows', rows), ('summary', summaries), ('screen', simple_checks))}
            write(directory/'verification.json', dict(passed=True, maximum_discrepancy=difference))
        if time.perf_counter()-started > 1200: raise TimeoutError('frozen metrics budget after evaluation')
        files = inventory(directory)
        elapsed = time.perf_counter()-started
        if elapsed > 1200: raise TimeoutError('frozen metrics budget including verification/inventory')
        write(directory/'completion.json', dict(passed=True, files=files,
            finished_unix=time.time(), phase_wall_seconds=elapsed,
            labels_read=True, primary='full', necessary_screen_passed=checks['passed'],
            decision='retain_fixed_candidate_for_remaining_validation' if checks['passed'] else 'stop_fixed_dual_overlap_design',
            scientific_gate_passed=False, goal_complete=False))
    except BaseException as ex:
        write(directory/'failure.json', dict(error_type=type(ex).__name__, error=str(ex), traceback=traceback.format_exc(), finished_unix=time.time(), labels_read=True, goal_complete=False))
        raise


def main():
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=('freeze', 'forward', 'audit', 'evaluate', 'metrics_audit'))
    p.add_argument('--out', type=Path, default=OUT); args = p.parse_args()
    C.configure_runtime()
    if args.stage == 'freeze': freeze(args.out)
    elif args.stage in ('forward', 'audit'): stage(args.out, args.stage)
    else: evaluate(args.out, args.stage == 'metrics_audit')


if __name__ == '__main__': main()
