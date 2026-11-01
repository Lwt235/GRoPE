"""Frozen old-TRAIN center-information diagnostic; no new Gaussian operator."""
from __future__ import annotations

import argparse
import ast
from fractions import Fraction
import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from contextlib import redirect_stdout

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT/'results/round19_center_semantics'
DEFAULT_AUDIT = ROOT/'results/round19_center_semantics_independent_audit'
SCENES = ('harvard_c3/hv_c3_1', 'harvard_c5/hv_c5_1', 'harvard_c6/hv_c6_1')
QUERIES = ('g8', 'g16', 'g24')
COUNTS = dict(g8=432, g16=108, g24=48)
VALID_COUNTS = dict(g8=9521, g16=2463, g24=1096)
FIELDS = ('q_predicted_depth', 'q_K', 'k_K', 'q_to_k_transform')
THREAD_ENV = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')
SOURCE_FUNCTION = 'experiments/round13_mixture_mechanism.py'
SOURCES = ('ROUND19_CENTER_SEMANTICS_PROTOCOL.md',
           'docs/round19_center_semantics_preparation_draft.md',
           'docs/round19_center_semantics_design_review.md',
           'docs/round18_post_r17_design_review.md', 'experiments/__init__.py',
           'experiments/round19_center_semantics.py',
           'experiments/round19_center_semantics_audit.py',
           'tests/test_round19_center_semantics_audit.py', SOURCE_FUNCTION)
COMPARISONS = {'a_minus_mu': ('a', 'mu'), 'a_minus_primary': ('a', 'primary'),
               'mu_minus_primary': ('mu', 'primary')}
TRANSITIONS = tuple(f'primary{p}_mu{m}_a{a}' for p in (0, 1) for m in (0, 1) for a in (0, 1))
SCREEN_POLICY = dict(q24_gain=dict(numerator=1, denominator=200),
                     q8_q16_lower_bound=dict(numerator=-1, denominator=200),
                     q24_each_scene_strict=True, exploratory_only=True)


def absolute(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT/path


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def array_sha(value):
    value = np.ascontiguousarray(value)
    header = json.dumps(dict(dtype=str(value.dtype), shape=list(value.shape)),
                        sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(header+b'\n'+value.tobytes()).hexdigest()


def legacy_array_sha(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(str(value.dtype).encode()+str(value.shape).encode()+value.tobytes()).hexdigest()


def runtime():
    if any(os.environ.get(name) != '1' for name in THREAD_ENV):
        raise ValueError('OMP/MKL/OpenBLAS thread environment must all be one')
    pools = threadpool_info()
    if any(pool['num_threads'] != 1 for pool in pools):
        raise ValueError('all loaded BLAS pools must use one CPU thread')
    build = io.StringIO()
    with redirect_stdout(build):
        np.__config__.show()
    return dict(device='cpu', cpu_threads=1, python=sys.version, executable=sys.executable,
                numpy=np.__version__, threadpoolctl=importlib.metadata.version('threadpoolctl'),
                platform=platform.platform(), thread_environment={n:os.environ[n] for n in THREAD_ENV},
                numpy_build_sha256=hashlib.sha256(build.getvalue().encode()).hexdigest(),
                threadpools=pools)


def function_contract():
    path = ROOT/SOURCE_FUNCTION
    source = path.read_text()
    matches = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)
               and n.name == 'predicted_centers']
    if len(matches) != 1:
        raise ValueError('exactly one historical predicted_centers function required')
    node = matches[0]
    segment = ast.get_source_segment(source, node)
    return dict(path=SOURCE_FUNCTION, name='predicted_centers', file_sha256=sha(path),
                source_sha256=hashlib.sha256(segment.encode()).hexdigest(),
                ast_sha256=hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest())


def extracted_predictor():
    tree = ast.parse((ROOT/SOURCE_FUNCTION).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'predicted_centers')
    module = ast.Module(body=[node], type_ignores=[])
    namespace = {'np': np}
    exec(compile(module, SOURCE_FUNCTION, 'exec'), namespace)
    return namespace['predicted_centers']


def inventory(directory):
    return {str(p.relative_to(directory)):sha(p) for p in sorted(directory.rglob('*'))
            if p.is_file() and p != directory/'completion.json'}


def verify_inventory(directory):
    completion = read(directory/'completion.json')
    if completion['files'] != inventory(directory):
        raise ValueError('stage inventory changed: '+str(directory))
    if completion.get('passed') is not True:
        raise ValueError('successful complete stage required')
    return completion


def bind_sources(bindings, directory, lock, *, snapshot=True):
    for name, expected in lock['source_hashes'].items():
        for path in ([ROOT/name, directory/'source_snapshot'/name] if snapshot else [ROOT/name]):
            if sha(path) != expected:
                raise ValueError('inherited source changed: '+str(path))
            bindings[str(path.resolve())] = expected


def inherited_contract():
    """Only metadata and approved forward bytes; never hash target archives."""
    feasibility = ROOT/'results/round17_feasibility'
    size = ROOT/'results/round17_size_temperature_attribution'
    size_audit = ROOT/'results/round17_size_temperature_independent_audit'
    acceptance_path = ROOT/'results/round17_size_root_acceptance_20261031.json'
    f_lock, size_lock = read(feasibility/'lock.json'), read(size/'lock.json')
    targets = f_lock['target_contract']
    denied = {str(absolute(e['target_path']).resolve()) for e in targets['entries']}
    bindings = {}

    def bind(path, expected=None):
        path = absolute(path).resolve()
        if str(path) in denied or path.name == 'labels.npz':
            raise ValueError('target archive cannot be hashed before global seal')
        actual = sha(path)
        if expected is not None and actual != expected:
            raise ValueError('inherited artifact changed: '+str(path))
        if str(path) in bindings and bindings[str(path)] != actual:
            raise ValueError('inconsistent inherited binding')
        bindings[str(path)] = actual
        return actual

    acceptance = read(acceptance_path)
    if (acceptance.get('accepted_for_descriptive_reporting') is not True
            or acceptance.get('primary_selection_changed') is not False
            or acceptance.get('checked_current_sources_and_snapshots') is not True):
        raise ValueError('accepted full size-attribution evidence required')
    bind(acceptance_path)
    for path, digest in acceptance['files'].items():
        bind(path, digest)
    audit = read(size_audit/'verification.json')
    if audit.get('passed') is not True or audit.get('selection_changed') is not False:
        raise ValueError('accepted unchanged primary audit required')
    for name in ('lock', 'forward_complete', 'summary', 'completion'):
        if audit['producer_'+name+'_sha256'] != sha(size/(name+'.json')):
            raise ValueError('size producer/auditor binding changed')
    for directory, lock in ((size, size_lock), (size_audit, read(size_audit/'lock.json')),
                            (feasibility, f_lock)):
        bind(directory/'lock.json', (directory/'lock.sha256').read_text().strip())
        bind(directory/'lock.sha256')
        bind_sources(bindings, directory, lock)
    for path, digest in size_lock['prior']['files'].items():
        bind(path, digest)
    regional = read(ROOT/'results/round17_regional/lock.json')
    stage_a = regional['stage_a']
    for name in ('lock', 'forward_complete', 'summary', 'completion'):
        bind(feasibility/(name+'.json'), stage_a['feasibility_'+name+'_sha256'])
    bind(feasibility/'decision.json', stage_a['decision_sha256'])
    bind(stage_a['audit_path'], stage_a['audit_sha256'])
    if read(absolute(stage_a['audit_path'])).get('passed') is not True:
        raise ValueError('accepted Stage A audit required')
    if targets != stage_a['target_contract'] or targets != size_lock['prior']['target_contract']:
        raise ValueError('exact inherited center-target contract changed')
    target_root = absolute(targets['root'])
    bind(target_root/'lock.json', targets['lock_sha256'])
    bind(target_root/'completion.json', targets['completion_sha256'])
    bind(targets['audit_path'], targets['audit_sha256'])
    target_audit = read(absolute(targets['audit_path']))
    if (target_audit.get('passed') is not True
            or target_audit['target_lock_sha256'] != targets['lock_sha256']
            or target_audit['target_completion_sha256'] != targets['completion_sha256']):
        raise ValueError('accepted R13 target audit required')
    target_audit_source_path = ROOT/'results/round13_target_audit_source_lock.json'
    bind(target_audit_source_path, target_audit['audit_source_lock_sha256'])
    target_audit_source = read(target_audit_source_path)
    bind(target_audit_source['source_snapshot'], target_audit_source['source_snapshot_sha256'])
    if target_audit_source['target_lock_sha256'] != targets['lock_sha256']:
        raise ValueError('R13 independent audit source/target binding changed')
    for name, digest in target_audit_source['source_hashes'].items():
        bind(name, digest)
    target_files = read(target_root/'completion.json')['files']
    size_files = read(size/'completion.json')['files']
    manifests = {(r['scene'], tuple(r['pair'])):r for r in f_lock['training_manifest_records']}
    prior_entries = {(e['scene'], tuple(e['pair'])):e for e in size_lock['entries']}
    records = {}
    for scene in SCENES:
        relative = 'forward/'+scene.replace('/', '__')+'/records.json'
        bind(size/relative, size_files[relative])
        for row in read(size/relative):
            if row['role'] == 'trace_full' and row['key'] == 'g8' and row['query'] in QUERIES:
                identity = scene, tuple(row['pair']), row['query']
                if identity in records:
                    raise ValueError('duplicate old primary record')
                records[identity] = row
    entries = []
    for target in targets['entries']:
        identity = target['scene'], tuple(target['pair'])
        scene = target['scene']
        prior, manifest = prior_entries[identity], manifests[identity]
        path = absolute(target['path']).resolve()
        if (scene not in SCENES or absolute(prior['path']).resolve() != path
                or prior['target_path'] != target['target_path']
                or prior['target_sha256'] != target['target_sha256']):
            raise ValueError('old cohort identity mismatch')
        bind(manifest['path'], manifest['sha256'])
        meta = read(absolute(manifest['path']))
        if meta['split'] != 'train' or meta['forward_uses_sensor']:
            raise ValueError('only original predicted TRAIN inputs allowed')
        input_path = path/'inputs.npz'
        input_sha = f_lock['inputs'][str(input_path.relative_to(ROOT))]
        if input_sha != prior['input_sha256'] or manifest['sha256'] != prior['manifest_sha256']:
            raise ValueError('input/manifest join mismatch')
        bind(input_path, input_sha)
        support_path = size/'forward'/scene.replace('/', '__')/'arrays'/path.name/'supports.npz'
        support_sha = size_files[str(support_path.relative_to(size))]
        bind(support_path, support_sha)
        winner_files = {}
        for query in QUERIES:
            old = records[(*identity, query)]
            winner_path = size/old['archive_path']
            winner_sha = size_files[old['archive_path']]
            bind(winner_path, winner_sha)
            if (old['input_sha256'] != input_sha or old['shape'] != [COUNTS[query], 432]
                    or old['config'] != dict(h=.25, temperature=10.)
                    or old['array_sha256'] != old['primary_regional_array_sha256']):
                raise ValueError('fixed primary provenance changed')
            winner_files[query] = dict(path=str(winner_path), sha256=winner_sha,
                argmax_sha256=old['argmax_sha256'], primary_regional_array_sha256=old['primary_regional_array_sha256'])
        target_path = absolute(target['target_path'])
        if target_files[str(target_path.relative_to(target_root))] != target['target_sha256']:
            raise ValueError('expected target hash differs from accepted completion metadata')
        entries.append(dict(scene=scene, pair=target['pair'], pair_name=path.name, path=str(path),
            input_path=str(input_path), input_sha256=input_sha, manifest_path=str(absolute(manifest['path'])),
            manifest_sha256=manifest['sha256'], support_path=str(support_path), support_sha256=support_sha,
            winners=winner_files, target_path=str(target_path), target_sha256=target['target_sha256']))
    if (len(entries) != 48 or len({(e['scene'], tuple(e['pair'])) for e in entries}) != 48
            or any(sum(e['scene'] == scene for e in entries) != 16 for scene in SCENES)
            or len(records) != 144):
        raise ValueError('complete original 48 TRAIN pairs required')
    return bindings, entries


def freeze(out, audit_root=DEFAULT_AUDIT):
    if out.exists():
        raise ValueError('refusing existing producer directory')
    bindings, entries = inherited_contract()
    sources = {name:sha(ROOT/name) for name in SOURCES}
    current_runtime = runtime()
    out.mkdir(parents=True)
    for name in sources:
        target = out/'source_snapshot'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT/name).read_bytes())
    lock = dict(version=1, created_unix=time.time(), source_hashes=sources,
                source_function=function_contract(), runtime=current_runtime,
                bindings=bindings, entries=entries, scenes=list(SCENES), queries=list(QUERIES),
                counts=COUNTS, expected_valid_counts=VALID_COUNTS, field_allowlist=list(FIELDS),
                audit_root=str(audit_root.resolve()), screen_policy=SCREEN_POLICY,
                rows=28224, pairs=48, labels_decoded=False, actual_target_hashes_read=False,
                exploratory_only=True, scientific_gate_passed=False, new_gaussian_predictions=False)
    write(out/'lock.json', lock)
    (out/'lock.sha256').write_text(sha(out/'lock.json')+'\n')


def verify_lock(out):
    lock = read(out/'lock.json')
    if (sha(out/'lock.json') != (out/'lock.sha256').read_text().strip()
            or lock['runtime'] != runtime() or lock['source_function'] != function_contract()
            or lock['screen_policy'] != SCREEN_POLICY or lock['scenes'] != list(SCENES)
            or lock['queries'] != list(QUERIES) or lock['counts'] != COUNTS
            or lock['expected_valid_counts'] != VALID_COUNTS or lock['field_allowlist'] != list(FIELDS)
            or set(lock['source_hashes']) != set(SOURCES)):
        raise ValueError('producer lock, runtime or policy changed')
    forbidden = {e['target_path'] for e in lock['entries']}
    for name, expected in lock['source_hashes'].items():
        if sha(ROOT/name) != expected or sha(out/'source_snapshot'/name) != expected:
            raise ValueError('joint source/snapshot changed: '+name)
    for path, expected in lock['bindings'].items():
        if path in forbidden or Path(path).name == 'labels.npz':
            raise ValueError('label must never enter pre-seal bindings')
        if sha(path) != expected:
            raise ValueError('inherited input/source binding changed: '+path)
    return lock


def require_audit_freeze(out, lock):
    directory = Path(lock['audit_root'])
    design = read(directory/'lock.json')
    if (sha(directory/'lock.json') != (directory/'lock.sha256').read_text().strip()
            or design['producer_lock_sha256'] != sha(out/'lock.json')
            or Path(design['run']).resolve() != out.resolve()
            or design['source_hashes'] != lock['source_hashes']
            or not lock['created_unix'] <= design['created_unix'] <= time.time()):
        raise ValueError('independent audit must be frozen before forward')
    for name, expected in design['source_hashes'].items():
        if sha(ROOT/name) != expected or sha(directory/'source_snapshot'/name) != expected:
            raise ValueError('independent audit source/snapshot changed')
    return directory


def spatial_bounds(stride):
    return np.asarray([[x, y, x+stride, y+stride] for y in range(0,144,stride)
                       for x in range(0,192,stride)], dtype=np.int64)


def nearest(points, centers):
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError('all projected locations must be finite [N,2]')
    squared = ((points[:,None,:]-centers[None,:,:])**2).sum(axis=-1)
    return squared.argmin(axis=-1).astype(np.int32)


def fov(points):
    return (np.isfinite(points).all(-1) & (points[:,0]>=-.5) & (points[:,0]<191.5)
            & (points[:,1]>=-.5) & (points[:,1]<143.5))


def forward_pair(entry, predictor):
    with np.load(entry['input_path'], allow_pickle=False) as archive:
        inputs = {field:archive[field] for field in FIELDS}
    for field, shape in (('q_predicted_depth',(144,192)), ('q_K',(3,3)),
                         ('k_K',(3,3)), ('q_to_k_transform',(4,4))):
        if inputs[field].shape != shape:
            raise ValueError('unexpected legal input shape: '+field)
        if field != 'q_predicted_depth' and (inputs[field].dtype != np.float64
                                            or not np.isfinite(inputs[field]).all()):
            raise ValueError('historical camera inputs must be finite float64')
    output = {'key_bounds_px':spatial_bounds(8)}
    centers = (output['key_bounds_px'][:,:2]+output['key_bounds_px'][:,2:]-1)/2
    with np.load(entry['support_path'], allow_pickle=False) as support:
        for query in QUERIES:
            bounds = support[query+'__bounds']
            mean = support[query+'__mean']
            if (bounds.dtype != np.int64 or not np.array_equal(bounds, spatial_bounds(int(query[1:])))
                    or mean.dtype != np.float32 or mean.shape != (COUNTS[query],2)
                    or not np.isfinite(mean).all()):
                raise ValueError('accepted mean/native bounds changed')
            a, front, in_fov = predictor(inputs, bounds)
            mu = mean.astype(np.float64)*8.
            if (a.dtype != np.float64 or a.shape != mu.shape or not np.isfinite(a).all()
                    or front.dtype != np.bool_ or in_fov.dtype != np.bool_):
                raise ValueError('historical predicted center returned invalid arrays')
            with np.load(entry['winners'][query]['path'], allow_pickle=False) as archive:
                winner = archive['trace_full__argmax']
            if (winner.dtype != np.int32 or winner.shape != (COUNTS[query],)
                    or (winner < 0).any() or (winner >= 432).any()
                    or legacy_array_sha(winner) != entry['winners'][query]['argmax_sha256']):
                raise ValueError('accepted primary winner array changed')
            values = dict(bounds_px=bounds, query_index=np.arange(len(a),dtype=np.int32),
                          a_px=a, mu_px=mu, delta_px=a-mu, nearest_a=nearest(a,centers),
                          nearest_mu=nearest(mu,centers), primary_argmax=winner,
                          a_finite=np.isfinite(a).all(-1), mu_finite=np.isfinite(mu).all(-1),
                          a_front=front, a_fov=in_fov, mu_fov=fov(mu))
            output.update({query+'__'+name:value for name,value in values.items()})
    return output


def failed_stage(directory, started, error):
    write(directory/'failure.json', dict(passed=False, started_unix=started, failed_unix=time.time(),
                                       error_type=type(error).__name__, error=str(error)))
    write(directory/'completion.json', dict(passed=False, files=inventory(directory), finished_unix=time.time()))


def forward(out):
    directory = out/'forward'
    if directory.exists():
        raise ValueError('forward attempt already exists; preserve it')
    directory.mkdir(parents=True)
    started = time.time()
    write(directory/'started.json', dict(created_unix=started, producer_lock_sha256=sha(out/'lock.json')))
    try:
        lock = verify_lock(out)
        audit_root=require_audit_freeze(out, lock)
        if read(audit_root/'lock.json')['created_unix'] > started:
            raise ValueError('audit design must precede forward stage start')
        predictor = extracted_predictor()
        records = []
        for entry in lock['entries']:
            arrays = forward_pair(entry, predictor)
            path = directory/'arrays'/entry['scene'].replace('/','__')/(entry['pair_name']+'.npz')
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, **arrays)
            records.append(dict(scene=entry['scene'], pair=entry['pair'], pair_name=entry['pair_name'],
                archive_path=str(path.relative_to(out)), archive_sha256=sha(path), n_query=588,
                array_sha256={name:array_sha(value) for name,value in arrays.items()}))
            print('CENTER_FORWARD',entry['scene'],entry['pair_name'],flush=True)
        write(directory/'records.json', records)
        write(directory/'completion.json', dict(passed=True, producer_lock_sha256=sha(out/'lock.json'),
            files=inventory(directory), pairs=len(records), rows=sum(r['n_query'] for r in records),
            labels_decoded=False, actual_target_hashes_read=False,
            started_unix=started, finished_unix=time.time()))
    except Exception as error:
        failed_stage(directory, started, error)
        raise


def _seal_content(out):
    if (out/'forward_complete.json').exists():
        raise ValueError('refusing existing global seal')
    lock = verify_lock(out)
    require_audit_freeze(out, lock)
    completed = verify_inventory(out/'forward')
    records = read(out/'forward/records.json')
    if (completed['pairs'] != 48 or completed['rows'] != 28224 or completed['labels_decoded']
            or completed['actual_target_hashes_read'] or completed['producer_lock_sha256'] != sha(out/'lock.json')
            or [(r['scene'],r['pair'],r['pair_name']) for r in records]
            != [(e['scene'],e['pair'],e['pair_name']) for e in lock['entries']]):
        raise ValueError('complete fixed forward population required')
    for record in records:
        path = out/record['archive_path']
        if sha(path) != record['archive_sha256']:
            raise ValueError('forward archive changed')
        with np.load(path,allow_pickle=False) as arrays:
            if {name:array_sha(arrays[name]) for name in arrays.files} != record['array_sha256']:
                raise ValueError('forward array identity changed')
    write(out/'forward_complete.json',dict(producer_lock_sha256=sha(out/'lock.json'),
        forward_completion_sha256=sha(out/'forward/completion.json'),
        forward_records_sha256=sha(out/'forward/records.json'), pairs=48, rows=28224,
        labels_decoded=False,actual_target_hashes_read=False,created_unix=time.time()))


def seal(out):
    directory=out/'seal'
    if directory.exists() or (out/'forward_complete.json').exists():
        raise ValueError('seal attempt already exists; preserve it')
    directory.mkdir(parents=True)
    started=time.time()
    write(directory/'started.json',dict(created_unix=started,producer_lock_sha256=sha(out/'lock.json')))
    try:
        _seal_content(out)
        write(directory/'completion.json',dict(passed=True,files=inventory(directory),
            forward_complete_sha256=sha(out/'forward_complete.json'),finished_unix=time.time()))
    except Exception as error:
        failed_stage(directory,started,error)
        raise


def fraction_dict(value):
    return dict(numerator=value.numerator,denominator=value.denominator,float=float(value))


def continuous(values):
    count=len(values)
    total=math.fsum(float(v) for v in values)
    squared=math.fsum(float(v)*float(v) for v in values)
    return dict(error_sum_px=total,error_squared_sum_px2=squared,
                mean_error_px=total/count if count else None,
                median_error_px=float(np.median(values)) if count else None,
                rmse_px=math.sqrt(squared/count) if count else None)


def pair_metrics(arrays,query,target,valid):
    count=COUNTS[query]
    if (valid.dtype != np.bool_ or valid.shape != (count,) or target.shape != (count,2)
            or not np.isfinite(target[valid]).all()):
        raise ValueError('invalid exact center labels')
    centers=(arrays['key_bounds_px'][:,:2]+arrays['key_bounds_px'][:,2:]-1)/2
    indices={name:arrays[query+'__'+field] for name,field in
             (('a','nearest_a'),('mu','nearest_mu'),('primary','primary_argmax'))}
    correct={name:((centers[index][valid]-target[valid])**2).sum(-1)<=64.
             for name,index in indices.items()}
    metrics=dict(n_query=count,n_point=int(valid.sum()),correct={name:int(v.sum()) for name,v in correct.items()})
    metrics['comparisons']={name:dict(benefit=int((correct[a]&~correct[b]).sum()),
        harm=int((~correct[a]&correct[b]).sum()),flip_all=int((indices[a]!=indices[b]).sum()),
        flip_valid=int((indices[a][valid]!=indices[b][valid]).sum())) for name,(a,b) in COMPARISONS.items()}
    errors={name:np.sqrt(((arrays[query+'__'+name+'_px'][valid]-target[valid])**2).sum(-1)) for name in ('a','mu')}
    metrics['continuous']={name:continuous(value) for name,value in errors.items()}
    metrics['correctness_transitions']={f'primary{p}_mu{m}_a{a}':int(((correct['primary']==p)&
        (correct['mu']==m)&(correct['a']==a)).sum()) for p in (0,1) for m in (0,1) for a in (0,1)}
    metrics['flags']={field:{str(value).lower():dict(n_query=int((arrays[query+'__'+field]==value).sum()),
        n_point=int(((arrays[query+'__'+field]==value)&valid).sum())) for value in (True,False)}
        for field in ('a_front','a_fov','mu_fov')}
    return metrics,errors


def aggregate(rows,errors):
    value=dict(n_query=sum(r['n_query'] for r in rows),n_point=sum(r['n_point'] for r in rows),
        correct={name:sum(r['correct'][name] for r in rows) for name in ('a','mu','primary')},
        comparisons={name:{field:sum(r['comparisons'][name][field] for r in rows)
             for field in ('benefit','harm','flip_all','flip_valid')} for name in COMPARISONS},
        correctness_transitions={name:sum(r['correctness_transitions'][name] for r in rows) for name in TRANSITIONS},
        flags={field:{state:{name:sum(r['flags'][field][state][name] for r in rows)
            for name in ('n_query','n_point')} for state in ('true','false')}
            for field in ('a_front','a_fov','mu_fov')},
        continuous={name:continuous(np.concatenate([r[name] for r in errors])) for name in ('a','mu')})
    if value['n_point'] <= 0:
        raise ValueError('every scene/scale must retain valid center population')
    return value


def make_summary(rows,errors):
    by_scene={scene:{q:aggregate([r for r in rows if r['scene']==scene and r['query']==q],
             [errors[(r['scene'],tuple(r['pair']),r['query'])] for r in rows if r['scene']==scene and r['query']==q])
             for q in QUERIES} for scene in SCENES}
    macro={}
    for query in QUERIES:
        pck={name:sum((Fraction(by_scene[s][query]['correct'][name],by_scene[s][query]['n_point'])
             for s in SCENES),Fraction())/3 for name in ('a','mu','primary')}
        macro[query]=dict(pck={n:fraction_dict(v) for n,v in pck.items()},
                         deltas={name:fraction_dict(pck[a]-pck[b]) for name,(a,b) in COMPARISONS.items()})
    delta=lambda q:Fraction(macro[q]['deltas']['a_minus_mu']['numerator'],macro[q]['deltas']['a_minus_mu']['denominator'])
    per_scene={s:by_scene[s]['g24']['correct']['a']>by_scene[s]['g24']['correct']['mu'] for s in SCENES}
    conditions=dict(q24_macro_gain_at_least_1_200=delta('g24')>=Fraction(1,200),
        q24_all_scenes_strict=all(per_scene.values()),q8_loss_at_most_1_200=delta('g8')>=Fraction(-1,200),
        q16_loss_at_most_1_200=delta('g16')>=Fraction(-1,200))
    passed=all(conditions.values())
    return dict(by_scene=by_scene,macro=macro,screen=dict(conditions=conditions,
        by_scene_q24_strict=per_scene,passed=passed,
        next_action='consider_separately_frozen_gaussian_hypothesis' if passed else 'stop_current_anchor_information_branch',
        exploratory_only=True,scientific_gate_passed=False),pairs=48,rows=28224,
        scientific_gate_passed=False,new_gaussian_predictions=False,validation_or_fresh_read=False)


def evaluate(out):
    directory=out/'evaluation'
    if directory.exists() or (out/'summary.json').exists() or (out/'completion.json').exists():
        raise ValueError('evaluation attempt already exists; preserve it')
    directory.mkdir(parents=True)
    started=time.time()
    write(directory/'started.json',dict(created_unix=started,producer_lock_sha256=sha(out/'lock.json')))
    try:
        lock=verify_lock(out)
        audit_root=require_audit_freeze(out,lock)
        seal_record=read(out/'forward_complete.json')
        verify_inventory(out/'forward')
        seal_stage=verify_inventory(out/'seal')
        if (seal_record['forward_completion_sha256']!=sha(out/'forward/completion.json')
                or seal_record['forward_records_sha256']!=sha(out/'forward/records.json')
                or seal_record['producer_lock_sha256']!=sha(out/'lock.json')
                or seal_stage['forward_complete_sha256']!=sha(out/'forward_complete.json')):
            raise ValueError('global forward seal changed')
        accepted=read(audit_root/'forward_verification.json')
        forward_audit_complete=verify_inventory(audit_root/'forward_audit')
        if (accepted.get('passed') is not True or accepted['producer_lock_sha256']!=sha(out/'lock.json')
                or accepted['producer_forward_complete_sha256']!=sha(out/'forward_complete.json')
                or accepted['audit_lock_sha256']!=sha(audit_root/'lock.json')
                or forward_audit_complete['verification_sha256']!=sha(audit_root/'forward_verification.json')
                or not seal_record['created_unix']<=accepted['finished_unix']<=time.time()):
            raise ValueError('independent complete forward audit must precede labels')
        records=read(out/'forward/records.json')
        rows=[];errors={}
        for entry,record in zip(lock['entries'],records):
            # First actual target hash/decode is strictly after all forward and its independent audit.
            if sha(entry['target_path'])!=entry['target_sha256']:
                raise ValueError('exact inherited center target changed')
            with np.load(out/record['archive_path'],allow_pickle=False) as arrays, np.load(entry['target_path'],allow_pickle=False) as labels:
                for query in QUERIES:
                    metrics,raw_errors=pair_metrics(arrays,query,labels[query+'_point_xy'],labels[query+'_point_visible'])
                    rows.append(dict(scene=entry['scene'],pair=entry['pair'],query=query,**metrics))
                    errors[(entry['scene'],tuple(entry['pair']),query)]=raw_errors
        if len(rows)!=144 or any(sum(r['n_point'] for r in rows if r['query']==q)!=VALID_COUNTS[q] for q in QUERIES):
            raise ValueError('original all-valid center population changed')
        write(directory/'rows.json',rows)
        summary=make_summary(rows,errors)
        summary.update(producer_lock_sha256=sha(out/'lock.json'),
            producer_forward_complete_sha256=sha(out/'forward_complete.json'),
            forward_audit_sha256=sha(audit_root/'forward_verification.json'),independent_audit_pending=True,
            started_unix=started,finished_unix=time.time())
        write(out/'summary.json',summary)
        write(directory/'completion.json',dict(passed=True,files=inventory(directory),
            labels_decoded=True,records=len(rows),summary_sha256=sha(out/'summary.json'),finished_unix=time.time()))
        write(out/'completion.json',dict(passed=True,producer_lock_sha256=sha(out/'lock.json'),
            files=inventory(out),scientific_gate_passed=False,independent_audit_pending=True,finished_unix=time.time()))
    except Exception as error:
        failed_stage(directory,started,error)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('freeze','forward','seal','evaluate'))
    parser.add_argument('--out',type=Path,default=DEFAULT_OUT)
    parser.add_argument('--audit-root',type=Path,default=DEFAULT_AUDIT)
    args=parser.parse_args()
    for name in THREAD_ENV:
        os.environ[name]='1'
    with threadpool_limits(limits=1):
        if args.stage=='freeze':
            freeze(args.out.resolve(),args.audit_root.resolve())
        else:
            globals()[args.stage](args.out.resolve())


if __name__=='__main__':
    main()
