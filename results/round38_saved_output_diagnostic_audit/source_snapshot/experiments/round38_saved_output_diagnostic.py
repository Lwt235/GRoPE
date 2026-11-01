"""One frozen, label-free diagnostic of R38 saved outputs; no model execution.

Pure API: diagnose_row({spec, record, arrays, pair}) -> JSON-compatible row.
The pair object contains the selected query's anchor/factor/old center/tau and
the complete native bounds. The runner reads only authenticated saved fields.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from contextlib import contextmanager
import csv
import importlib.abc
import json
import math
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np
from scipy.special import logsumexp
from experiments import round38_rms_anchor_contract as C

ROOT = C.ROOT
RUN = C.DEFAULT_OUT
OLD_AUDIT = C.DEFAULT_AUDIT
DEFAULT_OUT = ROOT/'results/round38_saved_output_diagnostic'
DEFAULT_AUDIT = ROOT/'results/round38_saved_output_diagnostic_audit'
ROOT_RECEIPT = 'results/round38_rms_anchor_root_acceptance_20261101.json'
ROOT_RECEIPT_SHA = 'c6ec773e2192b7cb045214047f229e762ed9bf8192c871ea599789480540da14'
DELIVERY = 'results/round38_rms_anchor_delivery_manifest_20261101.json'
DELIVERY_SHA = 'fdcd0bfb8ad71ff0149c29e4fc32c710c8219737437ca9273be1e2efd18396c2'
SOURCE_FILES = (
    'ROUND38_POSTFAILURE_DIAGNOSTIC_PROTOCOL.md',
    'docs/round38_postfailure_diagnostic_draft.md',
    'experiments/round38_saved_output_diagnostic.py',
    'experiments/round38_saved_output_diagnostic_audit.py',
    'tests/test_round38_saved_output_diagnostic.py',
)
FORWARD_FIELDS = ('input__mu', 'input__factor', 'input__bounds',
    'input__content_bias', 'log_raw_mass', 'log_floored_mass',
    'log_probability', 'probability', 'argmax')
PAIR_FIELDS = ('anchor_homogeneous', 'anchored_factor', 'old_a2',
    'key_bounds', 'key_bounds_px', 'tau32')
STAGES = ('raw', 'floor', 'logits', 'log_probability', 'probability')
SCENES = C.SCENES
TOLERANCE = 2e-12
WALL_SECONDS = 120.
SCOPE = dict(labels_read=False, new_model_calls=0, new_reference_calls=0,
    new_candidate_selected=False, new_task_metrics_computed=False,
    independent_probability_reference=False, scientific_gate_passed=False,
    goal_complete=False, continuous_density_mode_tested=False,
    tau_mismatch_causality_established=False)
STAT_PATHS = (
    'projection.old_anchor_distance_px', 'depth.a_z', 'depth.sigma_z', 'depth.rms_depth',
    'depth.normalized_mean', 'depth.normalized_std', 'depth.relative_std',
    'depth.front_mass', 'depth.tau32', 'distribution.raw_entropy',
    'distribution.raw_peak', 'distribution.floor_entropy',
    'distribution.floor_peak', 'distribution.log_raw_to_floor_total',
    'content.geometry_loss', 'content.content_gain', 'content.logit_advantage',
    'content.bookkeeping_residual',
) + tuple('stages.'+stage+'.'+field for stage in STAGES for field in (
    'top_two_gap', 'anchor_log_gap', 'rectangle_distance_px',
    'min_rectangle_distance_px', 'cell_center_distance_px'))


def _array(value, shape, dtype, name, finite=True):
    value = np.asarray(value)
    if value.dtype != dtype or value.shape != shape:
        raise ValueError('shape/dtype mismatch: '+name)
    if finite and not np.isfinite(value).all():
        raise ValueError('nonfinite array: '+name)
    return value


def _same(actual, expected, name):
    if (actual.dtype != expected.dtype or actual.shape != expected.shape
            or not np.array_equal(actual, expected)):
        raise ValueError('saved pair identity changed: '+name)


def _entropy(logs):
    values = logs-logsumexp(logs)
    with np.errstate(under='ignore'):
        weights = np.exp(values)
    return (-math.fsum(float(w)*float(x) for w, x in zip(weights, values)
                      if w > 0), float(weights.max()))


def _stage(scores, bounds, point, anchor_cells, primary, *, log_scores=True):
    if scores is None:
        return dict(defined=False, first=None, maximizers=[], tie_count=None,
            top_two_gap=None, anchor_set_contains=None, anchor_first_contains=None,
            anchor_log_gap=None, rectangle_distance_px=None,
            min_rectangle_distance_px=None, cell_center_distance_px=None)
    highest = float(np.max(scores))
    winners = np.flatnonzero(scores == highest).tolist()
    first = winners[0]
    ordered = np.sort(scores)
    result = dict(defined=True, first=first, maximizers=winners,
        tie_count=len(winners), top_two_gap=float(ordered[-1]-ordered[-2]),
        anchor_set_contains=None, anchor_first_contains=None, anchor_log_gap=None,
        rectangle_distance_px=None, min_rectangle_distance_px=None,
        cell_center_distance_px=None)
    if primary:
        def distance(index):
            box = bounds[index]
            delta = np.maximum(np.maximum(box[:2]-point, point-box[2:]), 0.)
            return 8*math.hypot(*map(float, delta))
        result.update(anchor_set_contains=bool(set(winners).intersection(anchor_cells)),
            anchor_first_contains=first in anchor_cells,
            anchor_log_gap=(highest-float(np.max(scores[anchor_cells]))) if log_scores else None,
            rectangle_distance_px=distance(first),
            min_rectangle_distance_px=min(distance(j) for j in winners),
            cell_center_distance_px=8*math.hypot(*map(float, (bounds[first,:2]+bounds[first,2:])/2-point)))
    return result


def diagnose_row(payload):
    """Analyze one already loaded row. No I/O, model, reference, or labels."""
    spec, record, bank, pair = (payload[n] for n in ('spec', 'record', 'arrays', 'pair'))
    if set(bank) != set(FORWARD_FIELDS) or set(pair) != set(PAIR_FIELDS):
        raise ValueError('exact decoded field boundaries required')
    a = _array(bank['input__mu'], (3,), np.float64, 'mu')
    factor = _array(bank['input__factor'], (3,3), np.float64, 'factor')
    n = len(bank['input__bounds'])
    if n < 2:
        raise ValueError('at least two cells required by top-two diagnostics')
    bounds = _array(bank['input__bounds'], (n,4), np.float64, 'bounds')
    if np.any(bounds[:,2:]<=bounds[:,:2]):
        raise ValueError('positive rectangle sides required')
    bias = _array(bank['input__content_bias'], (n,), np.float64, 'bias')
    raw = _array(bank['log_raw_mass'], (n,), np.float64, 'raw', False)
    floored = _array(bank['log_floored_mass'], (n,), np.float64, 'floored')
    logp = _array(bank['log_probability'], (n,), np.float64, 'log_probability')
    probability = _array(bank['probability'], (n,), np.float64, 'probability')
    argmax = _array(bank['argmax'], (), np.int32, 'argmax')
    old = _array(pair['old_a2'], (2,), np.float64, 'old_a2')
    tau = _array(pair['tau32'], (), np.float32, 'tau32')
    _same(a, _array(pair['anchor_homogeneous'], (3,), np.float64, 'pair anchor'), 'anchor')
    _same(factor, _array(pair['anchored_factor'], (3,3), np.float64, 'pair factor'), 'factor')
    _same(bounds, _array(pair['key_bounds'], (n,4), np.float64, 'pair bounds'), 'bounds')
    pixels = np.asarray(pair['key_bounds_px'])
    if pixels.dtype not in (np.dtype('int64'),np.dtype('float64')):
        raise ValueError('int64 or float64 physical bounds required')
    pixels = _array(pixels,(n,4),pixels.dtype,'pixel bounds')
    _same(bounds, (pixels.astype(np.float64)-.5)/8, 'reference units')
    sigma = math.hypot(*map(float, factor[2]))
    rms = math.hypot(float(a[2]), *map(float, factor[2]))
    if not math.isfinite(sigma) or not math.isfinite(rms):
        raise ValueError('nonfinite depth statistic')
    branch = ('zero_depth_rms' if rms == 0 else
        'nonpositive_constant_depth' if sigma == 0 and a[2] <= 0 else
        'positive_constant_depth_gaussian' if sigma == 0 else 'variable_depth')
    zero = branch in ('zero_depth_rms', 'nonpositive_constant_depth')
    if (record['structural_branch'] != branch or record['structural_zero'] is not zero
            or ('rms_depth' in record and record['rms_depth'] != rms)
            or (not np.isneginf(raw).all() if zero else not np.isfinite(raw).all())):
        raise ValueError('saved structural branch/raw semantics differ')
    if (probability < 0).any() or int(argmax) != int(np.argmax(probability)):
        raise ValueError('saved probability/first winner invalid')
    logfloor = np.log(1e-30*np.prod(bounds[:,2:]-bounds[:,:2], axis=1))
    logits = floored+bias
    if not np.isfinite(logits).all():
        raise ValueError('logit addition overflow')
    with np.errstate(under='ignore'):
        materialization = np.exp(logp)
    errors = dict(floor_error=float(np.max(abs(floored-np.logaddexp(raw, logfloor)))),
        log_probability_error=float(np.max(abs(logp-(logits-logsumexp(logits))))),
        materialization_error=float(np.max(abs(materialization-probability))),
        normalization_error=abs(math.fsum(map(float, probability))-1.))
    if any(not math.isfinite(v) or v > TOLERANCE for v in errors.values()):
        raise ValueError('saved arithmetic exceeds original absolute tolerance')
    sign = 'positive' if a[2] > 0 else 'negative' if a[2] < 0 else 'zero'
    point = None
    projection_status = 'zero_depth'
    if a[2] != 0:
        with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
            value = a[:2]/a[2]
        projection_status = 'finite' if np.isfinite(value).all() else 'nonfinite_division'
        if projection_status == 'finite':
            point = value
    fov_status, cells, boundary, old_delta, old_distance = 'undefined', [], None, None, None
    if point is not None:
        lo, hi = bounds[:,:2].min(0), bounds[:,2:].max(0)
        closed = bool(np.all(point >= lo) and np.all(point <= hi))
        fov_status = ('outer_boundary' if np.any(point == lo) or np.any(point == hi)
                      else 'interior') if closed else 'outside'
        cells = np.flatnonzero(np.all(point >= bounds[:,:2], axis=1)
                               & np.all(point <= bounds[:,2:], axis=1)).tolist()
        boundary = bool(closed and any(np.any(point == bounds[j,:2])
                                       or np.any(point == bounds[j,2:]) for j in cells))
        with np.errstate(over='ignore', invalid='ignore'):
            delta = 8*(point-old)
        if not np.isfinite(delta).all() or not math.isfinite(math.hypot(*map(float, delta))):
            raise ValueError('old anchor distance overflow')
        old_delta, old_distance = delta.tolist(), math.hypot(*map(float, delta))
    primary = bool(sign == 'positive' and point is not None and cells and not zero)
    stages = {name: _stage(scores, bounds, point, cells, primary, log_scores=name!='probability')
              for name,scores in zip(STAGES, (None if zero else raw, floored, logits, logp, probability))}
    errors.update(pair_identity_verified=True, saved_argmax_verified=True,
        raw_to_floor_winner_changed=None if zero else stages['raw']['first']!=stages['floor']['first'],
        raw_to_floor_maximizers_changed=None if zero else stages['raw']['maximizers']!=stages['floor']['maximizers'],
        logits_to_log_probability_winner_changed=stages['logits']['first']!=stages['log_probability']['first'],
        log_probability_to_probability_winner_changed=stages['log_probability']['first']!=stages['probability']['first'])
    relative, relative_status = None, 'nonpositive_mean'
    if a[2] > 0:
        with np.errstate(over='ignore', divide='ignore'):
            value = float(np.float64(sigma)/a[2])
        relative_status = 'defined' if math.isfinite(value) else 'nonfinite_division'
        if relative_status == 'defined':
            relative = value
    raw_entropy, raw_peak = (None, None) if zero else _entropy(raw)
    floor_entropy, floor_peak = _entropy(floored)
    jg, jb = stages['floor']['first'], stages['logits']['first']
    loss, gain = float(floored[jg]-floored[jb]), float(bias[jb]-bias[jg])
    advantage = float(logits[jb]-logits[jg])
    def transition(field):
        return None if not primary else ''.join('1' if stages[n][field] else '0' for n in ('floor','logits'))
    front = float(record['front_mass'])
    if not math.isfinite(front) or not 0 <= front <= 1:
        raise ValueError('saved front mass outside probability range')
    row = {k:spec[k] for k in ('id','scene','pair','pair_id','query_index')}
    row.update(status='ok', projection=dict(depth_sign=sign, projection_status=projection_status,
        xy_ref8=None if point is None else point.tolist(), fov_status=fov_status,
        cell_boundary=boundary, anchor_cells=cells, primary_anchor_defined=primary,
        old_anchor_delta_px=old_delta, old_anchor_distance_px=old_distance),
        depth=dict(a_z=float(a[2]),sigma_z=sigma, rms_depth=rms, normalized_mean=None if rms==0 else float(a[2])/rms,
            normalized_std=None if rms==0 else sigma/rms, relative_std=relative,
            relative_std_status=relative_status, front_mass=front, tau32=float(tau),
            structural_branch=branch, structural_zero=zero),
        distribution=dict(raw_entropy=raw_entropy, raw_peak=raw_peak,
            floor_entropy=floor_entropy, floor_peak=floor_peak,
            log_raw_to_floor_total=None if zero else float(logsumexp(raw)-logsumexp(logfloor))),
        arithmetic=errors, stages=stages,
        content=dict(winner_changed=jg!=jb, geometry_loss=loss, content_gain=gain,
            logit_advantage=advantage, bookkeeping_residual=advantage-(gain-loss),
            set_transition=transition('anchor_set_contains'),
            first_transition=transition('anchor_first_contains')))
    # Catch nonfinite derived fields without inventing clipping or a new floor.
    json.dumps(row, allow_nan=False)
    return row


def get_path(row, path):
    value = row
    for part in path.split('.'):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def describe(values):
    values = [float(v) for v in values if v is not None]
    if not values:
        return dict(n=0, mean=None, min=None, max=None, p50=None, p90=None, p99=None)
    if not all(math.isfinite(v) for v in values):
        raise ValueError('nonfinite summary value')
    ordered = sorted(values)
    quantiles = np.quantile(np.asarray(values,np.float64),[.5,.9,.99],method='linear')
    return dict(n=len(values), mean=math.fsum(values)/len(values), min=ordered[0],
        max=ordered[-1], p50=float(quantiles[0]), p90=float(quantiles[1]), p99=float(quantiles[2]))


def summarize_group(rows):
    ok = [r for r in rows if r['status']=='ok']
    categories = ('projection.depth_sign','projection.projection_status',
        'projection.fov_status','projection.cell_boundary','projection.primary_anchor_defined',
        'depth.structural_branch','depth.relative_std_status','content.winner_changed',
        'content.set_transition','content.first_transition')
    def categorical(path):
        counts = Counter(str(get_path(r,path)) for r in ok)
        return dict(sorted(counts.items()))
    return dict(population=len(rows), successful_rows=len(ok),
        status_counts=dict(sorted(Counter(r['status'] for r in rows).items())),
        categories={path:categorical(path) for path in categories},
        numerics={path:describe(get_path(r,path) for r in ok) for path in STAT_PATHS},
        stage_counts={name:{field:dict(sorted(Counter(str(r['stages'][name][field]) for r in ok).items()))
            for field in ('defined','anchor_set_contains','anchor_first_contains')}
            for name in STAGES},
        arithmetic_events={field:dict(sorted(Counter(str(r['arithmetic'][field]) for r in ok).items()))
            for field in ('raw_to_floor_winner_changed','raw_to_floor_maximizers_changed',
                'logits_to_log_probability_winner_changed','log_probability_to_probability_winner_changed')})


def summarize(rows):
    """Fixed population, scene and declared 2x2 summaries; no data-chosen bins."""
    return dict(quantile_method='linear_at_(n-1)*q', numerical_paths=list(STAT_PATHS),
        all=summarize_group(rows),
        by_scene={s:summarize_group([r for r in rows if r['scene']==s]) for s in SCENES},
        set_transition={k:summarize_group([r for r in rows if get_path(r,'content.set_transition')==k])
                        for k in ('11','10','01','00')},
        first_transition={k:summarize_group([r for r in rows if get_path(r,'content.first_transition')==k])
                          for k in ('11','10','01','00')}, **SCOPE)


def source_names():
    found, pending = set(), list(SOURCE_FILES)
    while pending:
        name = pending.pop()
        if name in found:
            continue
        path = ROOT/name
        if not path.is_file():
            raise ValueError('missing diagnostic source: '+name)
        found.add(name)
        if path.suffix != '.py':
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            modules = []
            if isinstance(node, ast.Import):
                modules = [n.name for n in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    raise ValueError('relative import needs explicit source review')
                if node.module:
                    modules = [node.module]+[node.module+'.'+n.name for n in node.names]
            for module in modules:
                parts = module.split('.')
                options = [Path(*parts).with_suffix('.py'), Path(*parts)/'__init__.py']
                options += [Path(*parts[:i])/'__init__.py' for i in range(1,len(parts))]
                pending.extend(str(p) for p in options if (ROOT/p).is_file() and str(p) not in found)
    return sorted(found)


def _inside(path, base):
    path, base = Path(path).resolve(), Path(base).resolve()
    if not path.is_relative_to(base):
        raise ValueError('path leaves frozen root: '+str(path))
    return path


def _bind(files, path, expected=None):
    path = _inside(path, ROOT)
    if path.suffix == '.npz':
        raise ValueError('metadata binding must not hash undeclared archive bytes')
    digest = C.sha(path)
    if expected is not None and digest != expected:
        raise ValueError('frozen evidence changed: '+str(path))
    if str(path) in files and files[str(path)] != digest:
        raise ValueError('conflicting frozen binding')
    files[str(path)] = digest
    return digest


def metadata():
    """Bind accepted JSON/text and declared archive identities; no NPZ access."""
    files = {}
    _bind(files, ROOT/ROOT_RECEIPT, ROOT_RECEIPT_SHA)
    _bind(files, ROOT/DELIVERY, DELIVERY_SHA)
    receipt, delivery = C.read(ROOT/ROOT_RECEIPT), C.read(ROOT/DELIVERY)
    if (receipt.get('accepted_exploratory_failure_delivery') is not True
            or receipt.get('completed_queries') != 2304 or receipt.get('passed_checks') != 0
            or receipt.get('total_checks') != 8 or receipt.get('dual_metric_routes_agree') is not True
            or receipt.get('all_input_and_output_arithmetic_passed') is not True
            or receipt.get('independent_probability_reference') is not False
            or receipt.get('scientific_gate_passed') is not False or receipt.get('goal_complete') is not False):
        raise ValueError('accepted complete R38 exploratory failure required')
    for manifest in (receipt, delivery):
        for name,digest in manifest['files'].items():
            _bind(files, ROOT/name, digest)
    names = ('lock.json','query_specs.json','prepared_complete.json',
        'prepared/completion.json','prepared/records.json','prepared/pair_records.json',
        'forward_complete.json','forward/completion.json','forward/records.json')
    for name in names:
        _bind(files, RUN/name)
    for name in ('lock.json','input_verification.json','forward_verification.json','completion.json'):
        _bind(files, OLD_AUDIT/name)
    lock, specs = C.read(RUN/'lock.json'), C.read(RUN/'query_specs.json')
    if C.query_specs(lock['entries']) != specs or lock['query_specs_sha256'] != C.sha(RUN/'query_specs.json'):
        raise ValueError('old locked complete ordered population changed')
    if len(specs)!=2304 or len({r['id'] for r in specs})!=2304:
        raise ValueError('2304 unique ordered queries required')
    preparation = C.read(RUN/'prepared_complete.json')
    forward = C.read(RUN/'forward_complete.json')
    verified = C.read(OLD_AUDIT/'forward_verification.json')
    links = ((preparation,'prepared_completion_sha256',RUN/'prepared/completion.json'),
        (preparation,'records_sha256',RUN/'prepared/records.json'),
        (preparation,'pair_records_sha256',RUN/'prepared/pair_records.json'),
        (forward,'forward_completion_sha256',RUN/'forward/completion.json'),
        (forward,'forward_records_sha256',RUN/'forward/records.json'),
        (forward,'prepared_complete_sha256',RUN/'prepared_complete.json'),
        (verified,'producer_lock_sha256',RUN/'lock.json'),
        (verified,'audit_lock_sha256',OLD_AUDIT/'lock.json'),
        (verified,'producer_forward_complete_sha256',RUN/'forward_complete.json'))
    if any(obj[key]!=C.sha(path) for obj,key,path in links):
        raise ValueError('old stage seal links changed')
    if (not preparation['all_inputs_prepared'] or not forward['all_production_completed']
            or forward['case_count']!=2304 or forward['completed_calls']!=2304
            or not verified['all_input_and_output_arithmetic_passed']
            or verified['checked_queries']!=2304):
        raise ValueError('full old input and forward arithmetic acceptance required')
    pairs, records, prepared = (C.read(RUN/n) for n in
        ('prepared/pair_records.json','forward/records.json','prepared/records.json'))
    if (len(pairs)!=48 or len(records)!=2304 or len(prepared)!=2304
            or [r['id'] for r in records]!=[s['id'] for s in specs]
            or [r['id'] for r in prepared]!=[s['id'] for s in specs]
            or [r['pair_id'] for r in pairs]!=[specs[i]['pair_id'] for i in range(0,2304,48)]):
        raise ValueError('saved archive population/order differs')
    inventories = {kind:C.read(RUN/kind/'completion.json')['files'] for kind in ('prepared','forward')}
    for kind,inventory in inventories.items():
        # Byte-only verification of all originally sealed artifacts; no decode.
        C.verify_inventory(RUN/kind,inventory,exclude=('completion.json',))
    old_completion = C.read(OLD_AUDIT/'completion.json')
    C.verify_inventory(OLD_AUDIT,old_completion['files'],exclude=('completion.json',))
    archives = {}
    for rows,fields,kind in ((pairs,PAIR_FIELDS,'prepared'),(records,FORWARD_FIELDS,'forward')):
        for r in rows:
            path = _inside(RUN/r['archive_path'], RUN/kind)
            if not r['passed'] or inventories[kind].get(str(path.relative_to(RUN/kind)))!=r['archive_sha256']:
                raise ValueError('accepted archive inventory differs')
            if any(n not in r['array_sha256'] for n in fields):
                raise ValueError('missing required saved field commitment')
            archives[str(path)] = dict(sha256=r['archive_sha256'], fields=list(fields),
                field_hashes={n:r['array_sha256'][n] for n in fields}, kind=kind)
    if len(archives)!=2352:
        raise ValueError('48 pair plus2304 saved output archives required')
    for spec,r,p in zip(specs,records,prepared):
        if (any(r.get(k)!=v or p.get(k)!=v for k,v in spec.items())
                or r['input_path']!=p['archive_path'] or r['input_sha256']!=p['archive_sha256']):
            raise ValueError('old prepared/forward query binding differs')
    return dict(evidence_files=files, specs=specs, archives=archives,
        pair_records=pairs, forward_records=records)


def freeze(out, role='producer', producer_root=None, audit_root=None):
    """Metadata-only joint source freeze; no archive decoding or statistics."""
    out = Path(out).resolve()
    if role not in ('producer','audit') or out.exists():
        raise ValueError('new diagnostic freeze directory and valid role required')
    data, runtime = metadata(), C.check_runtime()
    sources = {n:C.sha(ROOT/n) for n in source_names()}
    producer_root = Path(producer_root or (out if role=='producer' else DEFAULT_OUT)).resolve()
    audit_root = Path(audit_root or (out if role=='audit' else DEFAULT_AUDIT)).resolve()
    if role=='audit':
        producer = verify_before(producer_root, joint=False)
        if (producer['role']!='producer' or producer['source_hashes']!=sources
                or producer['runtime']!=runtime or any(producer[k]!=v for k,v in data.items())):
            raise ValueError('joint diagnostic metadata freeze differs')
    out.mkdir(parents=True)
    for name,digest in sources.items():
        target = out/'source_snapshot'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name,target)
        if C.sha(target)!=digest:
            raise ValueError('diagnostic source changed during capture')
    before = dict(version=1,role=role,created_unix=time.time(),runtime=runtime,
        source_hashes=sources,producer_root=str(producer_root),audit_root=str(audit_root),
        wall_seconds_budget=WALL_SECONDS,algebra_absolute_tolerance=TOLERANCE,
        stage_names=list(STAGES),stat_paths=list(STAT_PATHS),
        arrays_decoded_before_freeze=0,**SCOPE,**data)
    C.write(out/'before_execution.json',before)
    (out/'before_execution.sha256').write_text(C.sha(out/'before_execution.json')+'\n')
    return before


def verify_before(out, joint=True):
    out = Path(out).resolve()
    before = C.read(out/'before_execution.json')
    if C.sha(out/'before_execution.json')!=(out/'before_execution.sha256').read_text().strip():
        raise ValueError('diagnostic before-execution bytes changed')
    if (before['runtime']!=C.check_runtime() or before['wall_seconds_budget']!=WALL_SECONDS
            or before['algebra_absolute_tolerance']!=TOLERANCE
            or before['stage_names']!=list(STAGES) or before['stat_paths']!=list(STAT_PATHS)
            or any(before.get(k)!=v for k,v in SCOPE.items())):
        raise ValueError('diagnostic runtime/policy changed')
    current = metadata()
    if any(before[k]!=v for k,v in current.items()):
        raise ValueError('diagnostic original metadata bindings changed')
    if set(before['source_hashes'])!=set(source_names()):
        raise ValueError('diagnostic source closure changed')
    for name,digest in before['source_hashes'].items():
        if C.sha(ROOT/name)!=digest or C.sha(out/'source_snapshot'/name)!=digest:
            raise ValueError('frozen diagnostic source changed: '+name)
    if joint:
        peer_path = before['audit_root'] if before['role']=='producer' else before['producer_root']
        peer = verify_before(peer_path,joint=False)
        if (peer['role']==before['role'] or peer['source_hashes']!=before['source_hashes']
                or peer['runtime']!=before['runtime']
                or any(peer[k]!=before[k] for k in current)
                or peer['producer_root']!=before['producer_root']
                or peer['audit_root']!=before['audit_root']):
            raise ValueError('joint diagnostic freezes no longer agree')
    return before


class _NoModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname=='models' or fullname.startswith('models.'):
            raise RuntimeError('model imports forbidden during saved-output diagnostic')
        return None


class _FieldsOnly:
    def __init__(self, archive, fields):
        self.archive, self.allowed = archive, frozenset(fields)
    @property
    def files(self):
        return list(self.archive.files)
    def __getitem__(self, name):
        if name not in self.allowed:
            raise RuntimeError('unlisted saved field access forbidden: '+name)
        return self.archive[name]
    def close(self):
        self.archive.close()
    def __enter__(self):
        return self
    def __exit__(self,*args):
        self.close()


@contextmanager
def restricted_access(before):
    """Limit np.load paths AND decoded fields; prohibit new model imports."""
    original, guard = np.load, _NoModels()
    allowed = before['archives']
    def guarded(path,*args,**kwargs):
        if not isinstance(path,(str,Path)):
            raise RuntimeError('only a frozen saved archive path may be loaded')
        name = str(Path(path).resolve())
        if name not in allowed or args or kwargs.get('allow_pickle') is not False:
            raise RuntimeError('archive load outside frozen whitelist')
        if set(kwargs)-{'allow_pickle'}:
            raise RuntimeError('undeclared numpy archive loading policy')
        return _FieldsOnly(original(name,allow_pickle=False),allowed[name]['fields'])
    np.load = guarded
    sys.meta_path.insert(0,guard)
    try:
        yield
    finally:
        np.load = original
        sys.meta_path.remove(guard)


def _load(before, path, reads):
    name = str(Path(path).resolve())
    item = before['archives'].get(name)
    access = dict(path=name,archive_sha256=None if item is None else item['sha256'],
        allowed_fields=[] if item is None else item['fields'],status='started',
        started_unix=time.time(),finished_unix=None,archive_hash_verified=False,
        archive_opened=False,decoded_fields=[],verified_field_hashes={},error=None)
    reads.append(access)
    try:
        if item is None or C.sha(name)!=item['sha256']:
            raise ValueError('saved allowed archive hash changed: '+name)
        access['archive_hash_verified'] = True
        bank = {}
        with np.load(name,allow_pickle=False) as archive:
            access['archive_opened'] = True
            for field in item['fields']:
                bank[field] = archive[field]
                access['decoded_fields'].append(field)
        for field,digest in item['field_hashes'].items():
            if C.array_sha(bank[field])!=digest:
                raise ValueError('canonical saved field hash changed: '+name+': '+field)
            access['verified_field_hashes'][field] = digest
        access['status'] = 'verified'
        return bank
    except BaseException as error:
        access.update(status='failed',error=dict(type=type(error).__name__,message=str(error)))
        raise
    finally:
        access['finished_unix'] = time.time()


def iter_payloads(before, reads_log):
    """Yield all fixed positions; callers retain before.specs after exceptions."""
    pair_lookup = {p['pair_id']:p for p in before['pair_records']}
    cached_id, cached_bank = None, None
    for spec,record in zip(before['specs'],before['forward_records']):
        if spec['pair_id']!=cached_id:
            pair_record = pair_lookup[spec['pair_id']]
            cached_bank = _load(before,RUN/pair_record['archive_path'],reads_log)
            cached_id = spec['pair_id']
            for name,shape,dtype in (
                ('anchor_homogeneous',(48,3),np.float64),('anchored_factor',(48,3,3),np.float64),
                ('old_a2',(48,2),np.float64),('tau32',(48,),np.float32),
                ('key_bounds',(432,4),np.float64),('key_bounds_px',(432,4),np.int64)):
                _array(cached_bank[name],shape,dtype,name)
            literal = np.array([[x,y,x+8,y+8] for y in range(0,144,8)
                                for x in range(0,192,8)],np.int64)
            _same(cached_bank['key_bounds_px'],literal,'complete original native order')
            _same(cached_bank['key_bounds'],(literal.astype(np.float64)-.5)/8,'complete native bounds')
        i = spec['query_index']
        pair = {n:cached_bank[n] if n in ('key_bounds','key_bounds_px') else cached_bank[n][i]
                for n in PAIR_FIELDS}
        arrays = _load(before,RUN/record['archive_path'],reads_log)
        if any(record.get(k)!=v for k,v in spec.items()) or record['status']!='ok':
            raise ValueError('formal complete passed query identity required')
        yield dict(spec=spec,record=record,arrays=arrays,pair=pair)


def empty_row(spec, status, reason):
    return dict({k:spec[k] for k in ('id','scene','pair','pair_id','query_index')},
                status=status,reason=reason)


def write_csv(path, rows):
    fields = ('id','scene','pair_id','query_index','status','reason')
    extras = ('projection.depth_sign','projection.projection_status','projection.fov_status',
        'projection.primary_anchor_defined','content.set_transition','content.first_transition',
        'content.winner_changed')+STAT_PATHS
    with Path(path).open('x',newline='') as handle:
        writer = csv.DictWriter(handle,fieldnames=list(fields+extras))
        writer.writeheader()
        for row in rows:
            writer.writerow({name:get_path(row,name) for name in fields+extras})


def run(out):
    out = Path(out).resolve()
    if (out/'started.json').exists() or (out/'completion.json').exists():
        raise ValueError('preserve previous diagnostic attempt')
    started, tick = time.time(), time.perf_counter()
    deadline = tick+WALL_SECONDS
    before = C.read(out/'before_execution.json')
    specs = before['specs']
    rows, reads, failure = [], [], None
    C.write(out/'started.json',dict(started_unix=started,
        before_execution_sha256=C.sha(out/'before_execution.json'),**SCOPE))
    try:
        before = verify_before(out)
        if before['role']!='producer':
            raise ValueError('producer freeze required')
        with restricted_access(before):
            iterator = iter_payloads(before,reads)
            for i,spec in enumerate(specs):
                try:
                    if time.perf_counter()>=deadline:
                        raise TimeoutError('diagnostic stage deadline before row')
                    payload = next(iterator)
                    if payload['spec']!=spec:
                        raise ValueError('diagnostic iterator order changed')
                    rows.append(diagnose_row(payload))
                except BaseException as error:
                    failure = dict(index=i,id=spec['id'],type=type(error).__name__,message=str(error),
                                   traceback=traceback.format_exc())
                    rows.append(empty_row(spec,'failed',failure['message']))
                    break
            if failure is None:
                try:
                    next(iterator)
                    raise ValueError('diagnostic iterator contains extra positions')
                except StopIteration:
                    pass
    except BaseException as error:
        failure = dict(index=len(rows),id=specs[len(rows)]['id'] if len(rows)<len(specs) else None,
            type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
        if len(rows)<len(specs):
            rows.append(empty_row(specs[len(rows)],'failed',failure['message']))
    while len(rows)<len(specs):
        rows.append(empty_row(specs[len(rows)],'unattempted','after_first_diagnostic_failure'))
    C.write(out/'rows.json',rows)
    C.write(out/'summary.json',summarize(rows))
    write_csv(out/'rows.csv',rows)
    C.write(out/'reads.json',reads)
    if failure is not None:
        C.write(out/'failure.json',failure)
    wall = time.perf_counter()-tick
    time_ok = wall <= WALL_SECONDS
    success = failure is None and time_ok and len(rows)==2304 and all(r['status']=='ok' for r in rows)
    if [r['id'] for r in rows]!=[s['id'] for s in specs]:
        raise ValueError('diagnostic retained population changed')
    inventory_start = time.perf_counter()
    files = C.inventory(out)
    inventory_wall = time.perf_counter()-inventory_start
    C.write(out/'completion.json',dict(passed=success,complete_case_inventory=len(rows)==2304,
        completed_queries=sum(r['status']=='ok' for r in rows),total_queries=len(rows),
        started_unix=started,finished_unix=time.time(),phase_wall_seconds=wall,
        phase_wall_passed=time_ok,final_inventory_wall_seconds=inventory_wall,
        failure=failure,before_execution_sha256=C.sha(out/'before_execution.json'),
        archives_access_attempted=len(reads),archives_opened=sum(r['archive_opened'] for r in reads),
        archives_verified=sum(r['status']=='verified' for r in reads),
        fields_decoded=sum(len(r['decoded_fields']) for r in reads),files=files,**SCOPE))
    print('R38_SAVED_DIAGNOSTIC',success,sum(r['status']=='ok' for r in rows),flush=True)
    return success


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage',choices=('freeze','run'))
    parser.add_argument('--out',type=Path,default=DEFAULT_OUT)
    parser.add_argument('--role',choices=('producer','audit'),default='producer')
    parser.add_argument('--producer-root',type=Path,default=DEFAULT_OUT)
    parser.add_argument('--audit-root',type=Path,default=DEFAULT_AUDIT)
    args = parser.parse_args()
    C.configure_runtime()
    if args.stage=='freeze':
        freeze(args.out,args.role,args.producer_root,args.audit_root)
    else:
        if not run(args.out):
            raise SystemExit(1)


if __name__=='__main__':
    main()
