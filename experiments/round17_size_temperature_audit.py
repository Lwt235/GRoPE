"""Unrun independent audit of fixed TRAIN native-size temperature attribution.

No import of the new attribution producer or model. Old R17 preparation may
be shared for byte-exact replay; numerical support/integration, tau arithmetic,
targets and statistics have separate NumPy implementations. No selection or
scientific performance gate is defined by this module.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
from math import fsum
from pathlib import Path
import time

import numpy as np
from scipy.special import softmax

from experiments import round15_regional_audit_metrics as C
from experiments import round15_regional_audit_numeric as REF
from experiments import round17_trace_numerical_audit as TRACE
from experiments import round17_regional_audit as OLD
from experiments import round17_regional_audit_path_repair as REPAIR


ROOT = Path(__file__).resolve().parents[1]
SCENES, CONDITIONS = C.SCENES, OLD.CONDITIONS
ROLES = ('trace_full', 'size_temperature')
CONFIG = dict(h=.25, temperature=10.)
PARTITIONS = ('g8', 'g16', 'mixed', 'g24')
MATRICES, RAW_BYTES = 1344, 512434944
METRIC_TOLERANCE, SUMMARY_TOLERANCE = 1e-8, 1e-10
OWN_SOURCES = ('ROUND17_SIZE_TEMPERATURE_PROTOCOL.md',
    'experiments/round17_size_temperature_audit.py',
    'tests/test_round17_size_temperature_audit.py')
PRODUCER_SOURCES = ('ROUND17_SIZE_TEMPERATURE_PROTOCOL.md',
    'docs/round17_size_temperature_attribution_review.md',
    'models/round17_size_temperature_pe.py',
    'experiments/round17_size_temperature_attribution.py',
    'tests/test_round17_size_temperature_attribution.py')
# The inherited reference list omits two imports reached through its older
# tests/utility modules. Bind the actual transitive source closure explicitly.
INHERITED_IMPORT_SOURCES = ('experiments/__init__.py', 'models/__init__.py',
    'experiments/round15_quartic_numerical_audit.py',
    'experiments/round15_quartic_preflight.py')
SOURCE_NAMES = tuple(dict.fromkeys(OWN_SOURCES + REF.SOURCE_NAMES + TRACE.SOURCE_NAMES
    + OLD.OWN_SOURCES + REPAIR.REPAIR_SOURCES + INHERITED_IMPORT_SOURCES))
AUDIT_POLICY = dict(pre_forward_design=True, scientific_gate_defined=False,
    shared_old_preparation_for_replay=True, independent_new_model_import=False,
    independent_numpy_moments_and_integral=True, metric_tolerance=METRIC_TOLERANCE,
    summary_tolerance=SUMMARY_TOLERANCE, probability_tolerance=1e-4,
    reference_tolerance=1e-8, same_mass_tolerance=1e-12)
sha, read, write, array_sha, absolute = OLD.sha, OLD.read, OLD.write, OLD.array_sha, OLD.absolute


def native_trace(bounds):
    b = np.asarray(bounds)
    if (b.ndim != 2 or b.shape[1] != 4 or len(b) == 0 or b.dtype not in (np.dtype('int32'), np.dtype('int64'))
            or not np.isfinite(b).all() or not np.array_equal(b, b.astype(np.int64))):
        raise ValueError('finite integer native rectangles required')
    size = b.astype(np.float64)[:, 2:] - b[:, :2].astype(np.float64)
    if np.any(size <= 0):
        raise ValueError('native rectangles need positive width and height')
    return (size[:, 0]**2 + size[:, 1]**2) / (64.*12.)


def size_temperature(bounds, h=.25, temperature=10.):
    if (isinstance(h, (bool, np.bool_)) or isinstance(temperature, (bool, np.bool_))
            or not np.isfinite(h) or h <= 0 or not np.isfinite(h*h) or h*h <= 0
            or not np.isfinite(temperature) or temperature < 0):
        raise ValueError('positive finite bandwidth and nonnegative temperature required')
    result = (temperature*((h*h)/(h*h + native_trace(bounds)))).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError('native-size temperature overflow')
    return result


def independent_probability(raw, content, tau, key_bounds):
    """Actual float32 product followed by double log mass/floor and softmax."""
    raw, content, tau = np.asarray(raw), np.asarray(content), np.asarray(tau)
    bounds = np.asarray(key_bounds, np.float64)
    if (raw.dtype != np.float64 or content.dtype != np.float32 or tau.dtype != np.float32
            or raw.ndim != 2 or raw.shape != content.shape or tau.shape != (len(raw),)
            or bounds.shape != (raw.shape[1], 4) or np.any(bounds[:, 2:] <= bounds[:, :2])
            or any(not np.isfinite(v).all() for v in (raw, content, tau, bounds)) or np.any(raw < 0)):
        raise ValueError('invalid actual raw mass/content/tau/key populations')
    area = np.prod(bounds[:, 2:] - bounds[:, :2], axis=1)
    bias = tau[:, None] * content
    if not np.isfinite(bias).all():
        raise ValueError('float32 content product overflow')
    return softmax(np.log(raw + 1e-30*area[None]) + bias.astype(np.float64), axis=1)


def exact_fraction(value):
    return dict(numerator=value.numerator, denominator=value.denominator)


def center_pair(primary, size, key_bounds, target, visible):
    """First-index native modes, squared 8-pixel radius, original visible mask."""
    primary, size, target, visible = map(np.asarray, (primary, size, target, visible))
    if (primary.shape != size.shape or target.shape != (len(primary), 2)
            or visible.dtype != np.bool_ or visible.shape != (len(primary),)
            or not np.isfinite(target[visible]).all()):
        raise ValueError('paired center population mismatch')
    OLD.probability(primary, primary.shape); OLD.probability(size, size.shape)
    bounds = np.asarray(key_bounds)
    if bounds.shape != (primary.shape[1], 4):
        raise ValueError('native key-center population mismatch')
    centers = (bounds[:, :2].astype(np.float64) + bounds[:, 2:] - 1) / 2
    a, b = primary.argmax(1), size.argmax(1)
    ac, bc = np.zeros(len(primary), bool), np.zeros(len(primary), bool)
    ac[visible] = np.sum((centers[a[visible]] - target[visible])**2, axis=1) <= 64.
    bc[visible] = np.sum((centers[b[visible]] - target[visible])**2, axis=1) <= 64.
    flip = a != b
    counts = dict(n_query=len(a), n_point=int(visible.sum()), primary_correct=int(ac.sum()),
        size_correct=int(bc.sum()), flip_all_count=int(flip.sum()),
        flip_valid_count=int((flip & visible).sum()),
        size_benefit_count=int((~ac & bc & visible).sum()),
        size_harm_count=int((ac & ~bc & visible).sum()))
    if counts['size_correct']-counts['primary_correct'] != counts['size_benefit_count']-counts['size_harm_count']:
        raise ValueError('paired harm/benefit conservation failed')
    return ac, bc, counts


def validate_population(rows, entries):
    if (len(entries) != 48 or {e['scene'] for e in entries} != set(SCENES)
            or any(sum(e['scene'] == s for e in entries) != 16 for s in SCENES)):
        raise ValueError('expected exact three-scene 48-pair TRAIN cohort')
    cohort = {(e['scene'], tuple(e['pair'])): e for e in entries}
    if len(cohort) != 48:
        raise ValueError('duplicate TRAIN pair')
    expected = {(s, p, role, c['condition']) for s, p in cohort for role in ROLES for c in CONDITIONS}
    found = [(r['scene'], tuple(r['pair']), r['role'], r['condition']) for r in rows]
    if len(found) != MATRICES or len(set(found)) != len(found) or set(found) != expected:
        raise ValueError('incomplete, duplicate or extra attribution rows')
    specifications = {c['condition']: c for c in CONDITIONS}
    shared_mass, shared_center = {}, {}
    for r in rows:
        spec = specifications[r['condition']]
        if (r['config'] != CONFIG or any(r[k] != v for k, v in spec.items())
                or r['shape'] != [C.NATIVE_SIZES[spec['query']], C.NATIVE_SIZES[spec['key']]]
                or r['dtype'] != 'float64'
                or r['input_sha256'] != cohort[r['scene'], tuple(r['pair'])]['input_sha256']):
            raise ValueError('attribution row changed operator/input/native population')
        if 'mass' not in r:
            continue
        for field in ('mass', 'n_point', 'pck_count', 'n_query', 'n_key'):
            if type(r[field]) is not int or r[field] < 0:
                raise ValueError('integer metric population required')
        if (r['n_query'] != r['shape'][0] or r['n_key'] != r['shape'][1]
                or not 0 <= r['pck_count'] <= r['n_point'] <= r['n_query']
                or any(not np.isfinite(r[f]) for f in C.METRIC_FIELDS)):
            raise ValueError('invalid paired metric or center population')
        key = r['scene'], tuple(r['pair'])
        if shared_mass.setdefault(key, r['mass']) != r['mass']:
            raise ValueError('different visible dense population between conditions')
        qkey = key + (r['query'],)
        if shared_center.setdefault(qkey, r['n_point']) != r['n_point']:
            raise ValueError('center visibility changed with role/key partition')


def aggregate(rows, entries):
    """Independent stable sums, equal scenes, exact rational center means."""
    validate_population(rows, entries)
    output = {}
    for role in ROLES:
        by_scene = {}
        for scene in SCENES:
            by_scene[scene] = {}
            for c in CONDITIONS:
                values = [r for r in rows if (r['role'],r['scene'],r['condition']) == (role,scene,c['condition'])]
                mass, n, correct = (sum(r[k] for r in values) for k in ('mass','n_point','pck_count'))
                if len(values) != 16 or mass <= 0 or n <= 0:
                    raise ValueError('empty or incomplete scene/condition population')
                by_scene[scene][c['condition']] = dict(
                    **{f:fsum(r[f+'_sum'] for r in values)/mass for f in ('kl','js','direct_kl','direct_js')},
                    mass=mass,n_point=n,pck_count=correct,pck=float(Fraction(correct,n)))
        by_condition = {}
        for c in CONDITIONS:
            values = [by_scene[s][c['condition']] for s in SCENES]
            exact = sum((Fraction(v['pck_count'],v['n_point']) for v in values), Fraction())/3
            by_condition[c['condition']] = dict(
                **{f:fsum(v[f] for v in values)/3 for f in ('kl','js','direct_kl','direct_js')},
                pck=float(exact),pck_exact=exact_fraction(exact))
        macros = {prefix+'_'+metric:fsum(by_condition[c['condition']][metric] for c in choices)/len(choices)
            for prefix,choices in (('common9',CONDITIONS[:9]),('direct5',CONDITIONS[9:])) for metric in ('kl','js')}
        output[role] = dict(by_scene_condition=by_scene,by_condition=by_condition,**macros,
            J_descriptive=.5*(macros['common9_kl']+macros['direct5_kl']))
    return output


def numeric_pair(inputs, parts, supports, samples):
    """Fixed first TRAIN pair: independent support+GL64/96 and actual support hybrid."""
    inputs = REF.forward_inputs(inputs)
    schema = REF.reference_partition_schema(parts)
    expected_names = {c['condition'] for c in CONDITIONS}
    if set(samples) != expected_names:
        raise ValueError('all fourteen native conditions required')
    reference_queries, reference_keys, indices = {}, {}, {}
    for q in PARTITIONS:
        indices[q] = REF.selected_indices(len(parts[q]['bounds']))
        full = REF.AREF.reference_support(inputs, schema, q)
        reference_queries[q] = REF.slice_support(full, indices[q])
        reference_keys[q] = REF.AREF.reference_key(inputs, schema, q)
    checks = []
    for c in CONDITIONS:
        q,k,name = c['query'],c['key'],c['condition']
        saved = samples[name]; idx = indices[q]
        if saved['indices8'].dtype != np.int64 or not np.array_equal(saved['indices8'],idx):
            raise ValueError('fixed numerical query sample changed')
        mean = supports[q+'__mean'][idx]; cov = supports[q+'__cov'][idx]
        actual_raw, actual_content = saved['raw8'],saved['content8']
        if (mean.dtype != np.float32 or cov.dtype != np.float32 or actual_raw.dtype != np.float64
                or actual_content.dtype != np.float32 or actual_raw.shape != (8,len(parts[k]['bounds']))
                or actual_content.shape != actual_raw.shape):
            raise ValueError('retained actual sample dtype/shape mismatch')
        bounds = (parts[k]['bounds'].astype(np.float64)-.5)/8
        rq = reference_queries[q]
        content = (rq['features']@reference_keys[k]['features'].T).astype(np.float32)
        raw64,raw96 = (TRACE.independent_mass(rq['mean'],rq['cov'],bounds,.25,order) for order in (64,96))
        hybrid_raw = TRACE.independent_mass(mean,cov,bounds,.25,96)
        for role,suffix in (('trace_full','obs'),('size_temperature','size')):
            actual = saved[role][idx]
            tau = (TRACE.independent_temperature(cov,.25,10.).astype(np.float32) if role=='trace_full'
                   else size_temperature(parts[q]['bounds'][idx]))
            reference_tau = (TRACE.independent_temperature(rq['cov'],.25,10.).astype(np.float32) if role=='trace_full'
                             else size_temperature(parts[q]['bounds'][idx]))
            if (not np.array_equal(saved['tau_'+suffix+'32_8'],tau)
                    or saved['tau_'+suffix+'32_8'].dtype != np.float32
                    or saved['bias_'+suffix+'32_8'].dtype != np.float32
                    or not np.array_equal(saved['bias_'+suffix+'32_8'],tau[:,None]*actual_content)):
                raise ValueError('actual tau32/bias32 differs from independent arithmetic')
            same = independent_probability(actual_raw,actual_content,tau,bounds)
            hybrid = independent_probability(hybrid_raw,actual_content,tau,bounds)
            ref64 = independent_probability(raw64,content,reference_tau,bounds)
            ref96 = independent_probability(raw96,content,reference_tau,bounds)
            result = REF.probability_check(actual,ref96,ref64,raw96,bounds,idx)
            same_error = float(np.max(abs(same-actual))); hybrid_error=float(np.max(abs(hybrid-actual)))
            checks.append(dict(result,role=role,**c,indices=idx.tolist(),
                same_mass_probability_error=same_error,hybrid_probability_error=hybrid_error,
                hybrid_argmax_changes=int(np.count_nonzero(hybrid.argmax(1)!=actual.argmax(1))),
                same_prepared_tau32_exact=True,same_prepared_bias32_exact=True,
                passed=bool(result['passed'] and same_error<=1e-12 and hybrid_error<=1e-4)))
    if len(checks)!=28:
        raise ValueError('numerical sample population changed')
    return checks


def compare(actual, expected, tolerance=SUMMARY_TOLERANCE):
    return OLD.compare_tree(actual,expected,tolerance)


PAIR_FIELDS = ('n_query','n_point','primary_correct','size_correct','size_benefit_count',
               'size_harm_count','flip_all_count','flip_valid_count')


def aggregate_pairs(rows, *, stratified=False):
    groups = {}
    seen = set()
    for row in rows:
        name = row['condition'] + (f"__{row['width']}x{row['height']}" if stratified else '')
        identity = row['scene'],tuple(row['pair']),name
        if identity in seen:
            raise ValueError('duplicate paired attribution row')
        seen.add(identity)
        if (any(type(row[f]) is not int or row[f]<0 for f in PAIR_FIELDS)
                or not 0<=row['primary_correct']<=row['n_point']<=row['n_query']
                or not 0<=row['size_correct']<=row['n_point']
                or row['flip_all_count']>row['n_query'] or row['flip_valid_count']>row['n_point']
                or row['size_benefit_count']+row['size_harm_count']>row['flip_valid_count']
                or row['size_correct']-row['primary_correct']!=row['size_benefit_count']-row['size_harm_count']):
            raise ValueError('invalid paired counts or harm/benefit conservation')
        groups.setdefault(name,[]).append(row)
    result = {}
    for name,group in groups.items():
        scenes, differences = {},[]
        for scene in SCENES:
            values = [r for r in group if r['scene']==scene]
            if len(values)!=16:
                raise ValueError('incomplete paired scene population')
            sums = {f:sum(r[f] for r in values) for f in PAIR_FIELDS}
            n,nq = sums['n_point'],sums['n_query']
            if nq<=0:
                raise ValueError('empty native size stratum')
            delta = Fraction(sums['size_correct']-sums['primary_correct'],n) if n else None
            differences.append(delta)
            value = dict(sums,primary_pck=sums['primary_correct']/n if n else None,
                size_pck=sums['size_correct']/n if n else None,flip_all_rate=sums['flip_all_count']/nq,
                flip_valid_rate=sums['flip_valid_count']/n if n else None,
                size_minus_primary_pck_pp=float(100*delta) if delta is not None else None,
                pck_difference_exact=exact_fraction(delta) if delta is not None else None)
            if stratified:
                native = {}
                for role in ROLES:
                    mass = sum(r['native'][role]['mass'] for r in values)
                    native[role] = dict(mass=mass,**{f:fsum(r['native'][role][f+'_sum'] for r in values)/mass if mass else None
                        for f in ('direct_kl','direct_js')})
                value['native']=native
            scenes[scene]=value
        defined = all(d is not None for d in differences)
        delta = sum(differences,Fraction())/3 if defined else None
        macro = dict(size_minus_primary_pck_pp=float(100*delta) if defined else None,
            pck_difference_exact=exact_fraction(delta) if defined else None,
            flip_all_rate=fsum(v['flip_all_rate'] for v in scenes.values())/3,
            flip_valid_rate=fsum(v['flip_valid_rate'] for v in scenes.values())/3 if defined else None,
            every_scene_has_center_population=defined)
        if stratified:
            macro['native'] = {role:{f:fsum(v['native'][role][f] for v in scenes.values())/3
                if all(v['native'][role][f] is not None for v in scenes.values()) else None
                for f in ('direct_kl','direct_js')} for role in ROLES}
        result[name] = dict(by_scene=scenes,macro=macro)
    return result


def no_active_benchmark():
    directory=ROOT/'results/round17_validation/timing'
    if directory.exists() and not (directory/'completion.json').exists():
        raise RuntimeError('do not compete with the frozen R17 validation benchmark')


def inventory(directory, files, *, excluded=()):
    directory=Path(directory)
    if any(Path(p).is_absolute() or '..' in Path(p).parts for p in files):
        raise ValueError('unsafe completed artifact path')
    actual={str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file()}-set(excluded)
    if actual!=set(files):
        raise ValueError('changed completed artifact population')
    for name,digest in files.items():
        if sha(directory/name)!=digest:
            raise ValueError('completed artifact changed: '+name)


def verify_prior(prior):
    """Accepted old TRAIN provenance, including the transparent path repair.

    This checks result/source bytes and metadata. No sensor target archive is
    hashed or decoded. Accepted old metrics are not rerun or selected anew.
    """
    directory=absolute(prior['root']).resolve()
    old=OLD.H.checked_lock(directory)
    selected=read(directory/'selection.json')
    if sha(directory/'selection.json')!=(directory/'selection.sha256').read_text().strip():
        raise ValueError('old accepted selection changed')
    for path,digest in prior['files'].items():
        if not Path(path).is_absolute() or sha(path)!=digest:
            raise ValueError('old accepted TRAIN binding changed')
    complete=read(directory/'completion.json')
    inventory(directory,complete['files'],excluded=('completion.json','selection.json','selection.sha256'))
    audit_path=absolute(selected['audit_path']).resolve(); verified=read(audit_path)
    if (sha(audit_path)!=selected['audit_sha256'] or verified['passed'] is not True
            or verified['scientific_gate_passed'] is not False
            or selected['selection']!=verified['selection']
            or selected['train_screen']!=verified['train_screen']
            or selected['train_screen']['passed'] is not True
            or selected['validation_run'] or selected['scientific_gate_passed']):
        raise ValueError('independent accepted TRAIN decision required')
    for name in ('lock','forward_complete','summary','completion'):
        digest=sha(directory/(name+'.json'))
        if selected['regional_'+name+'_sha256']!=digest or verified['regional_'+name+'_sha256']!=digest:
            raise ValueError('accepted audit belongs to another regional run')
    primary=selected['selection']['primary']
    if (primary['family'],primary['grid_index'],primary['config'])!=('trace_full',2,CONFIG):
        raise ValueError('conditional attribution changed the accepted primary')
    repair_dir=audit_path.parent.parent
    repair=REPAIR.verify_repair(repair_dir); repaired_complete=read(repair_dir/'completion.json')
    inventory(repair_dir,repaired_complete['files'],excluded=('completion.json',))
    nested_complete=read(audit_path.parent/'completion.json')
    inventory(audit_path.parent,nested_complete['files'],excluded=('completion.json',))
    if (repaired_complete['passed'] is not True or nested_complete['passed'] is not True
            or repair['original_contract']['joint_lock_sha256']!=sha(directory/'lock.json')
            or repair['original_contract']['run']!=str(directory)
            or verified['path_repair']['repair_lock_sha256']!=sha(repair_dir/'lock.json')
            or verified['path_repair']['path_repair_frozen_before_regional_forward'] is not False):
        raise ValueError('accepted transparent path-repair receipt changed')
    targets=old['stage_a']['target_contract']
    if prior['target_contract']!=targets:
        raise ValueError('old center target metadata changed')
    target_entries={(e['scene'],tuple(e['pair'])):e for e in targets['entries']}
    entries=[]
    for entry in old['entries']:
        target=target_entries[entry['scene'],tuple(entry['pair'])]
        entries.append(dict(entry,target_path=target['target_path'],target_sha256=target['target_sha256']))
    if (entries!=prior['entries'] or prior['partitions_sha256']!=old['partitions_sha256']
            or prior['source_hashes']!=dict(old['source_hashes'],**repair['source_hashes'])
            or prior['selected_configuration']!=dict(family='trace_full',grid_index=2,**CONFIG)):
        raise ValueError('old source/cohort/partition/configuration binding differs')
    return old


def checked_lock(run):
    run=Path(run).resolve(); lock=read(run/'lock.json')
    if sha(run/'lock.json')!=(run/'lock.sha256').read_text().strip():
        raise ValueError('attribution producer lock changed')
    required=set(SOURCE_NAMES)|set(PRODUCER_SOURCES)|set(lock['prior']['source_hashes'])
    if set(lock['source_hashes'])!=required:
        raise ValueError('joint source lock differs from complete inherited/new source population')
    for name,digest in lock['source_hashes'].items():
        if sha(ROOT/name)!=digest or sha(run/'source_snapshot'/name)!=digest:
            raise ValueError('joint source/original snapshot changed: '+name)
    if (lock['roles']!=list(ROLES) or lock['config']!=CONFIG or lock['conditions']!=list(CONDITIONS)
            or lock['matrices']!=MATRICES or lock['raw_probability_bytes']!=RAW_BYTES
            or lock['field_allowlist']!=list(REF.FIELDS) or lock['entries']!=lock['prior']['entries']
            or lock['size_trace']!='(width_pixels**2+height_pixels**2)/(64*12)'
            or lock['size_temperature_arithmetic']!='float64 trace and T*(h*h/(h*h+trace)); cast once float32'
            or lock['content_arithmetic']!='tau32*full_GEMM_content32, then add geometry64 and softmax64'
            or lock['independent_audit_required'] is not True or lock['all_probabilities_retained'] is not True
            or any(lock[k] is not False for k in ('labels_decoded','validation_effects_read',
                'changes_current_validation','configuration_search','scientific_gate_defined','cost_measured'))):
        raise ValueError('fixed descriptive attribution contract changed')
    verify_prior(lock['prior'])
    if sha(run/'partitions.npz')!=lock['partitions_sha256']:
        raise ValueError('native partition archive changed')
    for entry in lock['entries']:
        path=absolute(entry['path'])
        if sha(path/'manifest.json')!=entry['manifest_sha256'] or sha(path/'inputs.npz')!=entry['input_sha256']:
            raise ValueError('old forward input/manifest changed')
        meta=read(path/'manifest.json')
        if (meta['split']!='train' or meta['scene']!=entry['scene'] or meta['pair']!=entry['pair']
                or meta['forward_uses_sensor'] or meta['label_sha256']!=entry['label_sha256']):
            raise ValueError('unauthorized data/label identity')
    return lock


def freeze(run,out):
    no_active_benchmark(); run,out=Path(run).resolve(),Path(out).resolve()
    if out.exists() or (run/'forward').exists() or out==run or run in out.parents:
        raise ValueError('audit must be separately frozen before any new forward')
    lock=checked_lock(run)
    out.mkdir(parents=True)
    source={name:lock['source_hashes'][name] for name in SOURCE_NAMES}
    for name in source:
        path=out/'source_snapshot'/name; path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes((ROOT/name).read_bytes())
    value=dict(created_unix=time.time(),producer_lock_sha256=sha(run/'lock.json'),run=str(run),
        source_hashes=source,**AUDIT_POLICY)
    write(out/'lock.json',value);(out/'lock.sha256').write_text(sha(out/'lock.json')+'\n')
    return value


def verify_audit_lock(run,out,lock):
    design=read(out/'lock.json')
    if (sha(out/'lock.json')!=(out/'lock.sha256').read_text().strip()
            or design['run']!=str(run) or design['producer_lock_sha256']!=sha(run/'lock.json')
            or design['source_hashes']!={n:lock['source_hashes'][n] for n in SOURCE_NAMES}
            or not lock['created_unix']<=design['created_unix']<=time.time()
            or set(design)!=set(AUDIT_POLICY)|{'created_unix','producer_lock_sha256','run','source_hashes'}
            or any(type(design.get(k)) is not type(v) or design[k]!=v for k,v in AUDIT_POLICY.items())):
        raise ValueError('independent pre-forward audit source/chronology differs')
    for name,digest in design['source_hashes'].items():
        if sha(out/'source_snapshot'/name)!=digest:
            raise ValueError('independent source snapshot changed')
    return design


def checked_stage(run,scene,stage,lock_sha):
    directory=run/stage/scene.replace('/','__')
    complete=read(directory/'completion.json')
    if (complete['stage']!=stage or complete['scene']!=scene or complete['lock_sha256']!=lock_sha
            or complete['record_count']!=448):
        raise ValueError('scene stage identity/population differs')
    inventory(directory,complete['files'],excluded=('completion.json',))
    return directory,complete,read(directory/'started.json')


def literal_parts(run):
    parts=C.literal_parts()
    with np.load(run/'partitions.npz',allow_pickle=False) as archive:
        for q in PARTITIONS:
            for field in ('bounds','owner','atoms','parent','mean','cov','mass'):
                if not np.array_equal(archive[q+'__'+field],parts[q][field]):
                    raise ValueError('saved native partition differs from literal independent construction')
    return parts


def load_labels(entry):
    """After the global seal: R10 dense/mixed, exact R13 Stage A square centers."""
    old = absolute(entry['path']) / 'labels.npz'
    target = absolute(entry['target_path'])
    if sha(old) != entry['label_sha256'] or sha(target) != entry['target_sha256']:
        raise ValueError('old TRAIN label bytes changed')
    with np.load(old, allow_pickle=False) as archive:
        labels = {field: archive[field] for field in
                  ('gt_xy', 'gt_visible', 'mixed_point_xy', 'mixed_point_visible')}
    with np.load(target, allow_pickle=False) as archive:
        for query in ('g8', 'g16', 'g24'):
            for field in ('xy', 'visible'):
                name = query + '_point_' + field
                labels[name] = archive[name]
    return labels


def rebuild_rows(entry, condition, parts, probabilities, labels, committed):
    q,k=condition['query'],condition['key']
    direct,common=C.independent_counts(labels,parts,q,k)
    visible,target=labels[q+'_point_visible'],labels[q+'_point_xy']
    ac,bc,pair=center_pair(probabilities['trace_full'],probabilities['size_temperature'],parts[k]['bounds'],target,visible)
    paired=dict(scene=entry['scene'],pair=entry['pair'],**condition,**pair)
    rows=[]
    for role,p in probabilities.items():
        measured=C.measure(p,parts,q,k,condition['scenario'],direct,common)
        rows.append(dict(committed[role],**measured,pck_count=int((ac if role=='trace_full' else bc).sum()),
            n_point=int(visible.sum()),label_sha256=entry['label_sha256'],target_sha256=entry['target_sha256']))
    sizes=parts[q]['bounds'][:,2:]-parts[q]['bounds'][:,:2];strata=[]
    flip=probabilities['trace_full'].argmax(1)!=probabilities['size_temperature'].argmax(1)
    for width,height in np.unique(sizes,axis=0):
        mask=np.all(sizes==[width,height],axis=1);valid=mask&visible
        native={}
        for role,p in probabilities.items():
            kl,js,mass=C.divergence_sums(p[mask],direct[mask])
            native[role]=dict(direct_kl_sum=kl,direct_js_sum=js,mass=mass)
        strata.append(dict(scene=entry['scene'],pair=entry['pair'],**condition,width=int(width),height=int(height),
            n_query=int(mask.sum()),n_point=int(valid.sum()),primary_correct=int(ac[mask].sum()),
            size_correct=int(bc[mask].sum()),size_benefit_count=int((~ac&bc&valid).sum()),
            size_harm_count=int((ac&~bc&valid).sum()),flip_all_count=int((flip&mask).sum()),
            flip_valid_count=int((flip&valid).sum()),native=native))
    return rows,paired,strata


def support_statistics(collection):
    """Every scene/pair/query support once, independent of key and role."""
    groups={};seen=set();count=0
    for entry,support in collection:
        for q in PARTITIONS:
            identity=entry['scene'],tuple(entry['pair']),q
            if identity in seen:
                raise ValueError('support counted more than once across native key partitions')
            seen.add(identity)
            bounds=support[q+'__bounds'];cov=support[q+'__cov']
            if len(bounds)!=C.NATIVE_SIZES[q]:
                raise ValueError('wrong unique support population')
            sizes=bounds[:,2:]-bounds[:,:2]
            observed_trace=np.trace(cov.astype(np.float64),axis1=-2,axis2=-1)
            obs=support[q+'__tau_obs32'].astype(np.float64)
            size=support[q+'__tau_size32'].astype(np.float64)
            native=native_trace(bounds);count+=len(bounds)
            for width,height in np.unique(sizes,axis=0):
                mask=np.all(sizes==[width,height],axis=1)
                key=entry['scene'],q,int(width),int(height)
                values=groups.setdefault(key,{f:[] for f in ('tau_obs32','tau_size32','tau_obs_over_tau_size',
                    'observed_trace','native_trace')})
                for name,value in (('tau_obs32',obs),('tau_size32',size),('tau_obs_over_tau_size',obs/size),
                    ('observed_trace',observed_trace),('native_trace',native)):
                    values[name].append(value[mask])
    if len(seen)!=48*4 or count!=39888 or len(groups)!=21:
        raise ValueError('incomplete unique-support descriptive population')
    output=[]
    for key,values in sorted(groups.items()):
        record=dict(scene=key[0],query=key[1],width=key[2],height=key[3])
        population=None
        for name,chunks in values.items():
            value=np.concatenate(chunks).astype(np.float64)
            if population is not None and len(value)!=population:
                raise ValueError('support descriptor population mismatch')
            population=len(value)
            quantiles=np.quantile(value,[0.,.05,.5,.95,1.],method='linear')
            record[name]={n:float(v) for n,v in zip(('min','p05','median','p95','max'),quantiles)}
        record['n']=population;output.append(record)
    return dict(unique_queries=count,grouping=['scene','query','width','height'],quantile_method='linear',groups=output,
        input_only=True,key_condition_repetition=False,bins_selected_from_effects=False,significance_test=False)


def _audit(run,out,started):
    lock=checked_lock(run);design=verify_audit_lock(run,out,lock)
    lock_sha,seal_sha=sha(run/'lock.json'),sha(run/'forward_complete.json')
    seal=read(run/'forward_complete.json');complete=read(run/'completion.json');result=read(run/'summary.json')
    inventory(run,complete['files'],excluded=('completion.json',))
    if (complete['lock_sha256']!=lock_sha or complete['summary_sha256']!=sha(run/'summary.json')
            or complete['independent_audit_pending'] is not True
            or seal['lock_sha256']!=lock_sha or seal['matrices']!=MATRICES
            or seal['raw_probability_bytes']!=RAW_BYTES or seal['labels_decoded'] is not False
            or seal['all_probabilities_retained'] is not True
            or set(seal['forward_chain'])!=set(SCENES)
            or not seal['sealed_unix']<=result['created_unix']<=complete['finished_unix']<=started):
        raise ValueError('result/seal identity or global chronology differs')
    parts=literal_parts(run);stages={};all_committed=[];all_measured=[];all_pair=[];all_strata=[]
    for scene in SCENES:
        fd,fc,fs=checked_stage(run,scene,'forward',lock_sha)
        ed,ec,es=checked_stage(run,scene,'evaluation',lock_sha)
        if (not design['created_unix']<=fs['created_unix']==fc['started_unix']<=fc['finished_unix']<=seal['sealed_unix']
                or not seal['sealed_unix']<=es['created_unix']==ec['started_unix']<=ec['finished_unix']<=result['created_unix']
                or fs['lock_sha256']!=lock_sha or fs['runtime']!=lock['runtime']
                or es['forward_complete_sha256']!=seal_sha or ec['forward_complete_sha256']!=seal_sha
                or fc['runtime']!=lock['runtime'] or ec['runtime']!=lock['runtime']
                or fc['labels_decoded'] is not False or ec['labels_decoded'] is not True
                or fc['all_probabilities_retained'] is not True or fc['primary_exact_matrices']!=224
                or seal['forward_chain'][scene]!=dict(completion_sha256=sha(fd/'completion.json'),finished_unix=fc['finished_unix'])):
            raise ValueError('all forward commits must precede every label stage')
        entries=sorted((e for e in lock['entries'] if e['scene']==scene),key=lambda e:Path(e['path']).name)
        committed,measured=read(fd/'records.json'),read(ed/'rows.json')
        paired,strata=read(ed/'paired_rows.json'),read(ed/'size_strata.json')
        all_committed.extend(committed);all_measured.extend(measured);all_pair.extend(paired);all_strata.extend(strata)
        expected_repeat=[{k:r[k] for k in ('condition','role','array_sha256')} for r in committed if r['pair']==entries[0]['pair']]
        if read(fd/'determinism.json')!=dict(pair=entries[0]['pair'],records=expected_repeat,
                passed=True,probability_count=28,labels_decoded=False):
            raise ValueError('complete first-pair deterministic check missing')
        old_records=read(absolute(lock['prior']['root'])/'forward'/scene.replace('/','__')/'records.json')
        old={(tuple(r['pair']),r['condition']):r for r in old_records if r['family']=='trace_full' and r['grid_index']==2}
        expected_old={(tuple(e['pair']),c['condition']) for e in entries for c in CONDITIONS}
        if len(old)!=224 or set(old)!=expected_old:
            raise ValueError('complete old primary commitments missing')
        stages[scene]=(fd,entries,old)
    validate_population(all_committed,lock['entries']);validate_population(all_measured,lock['entries'])
    def row_id(r):return r['scene'],tuple(r['pair']),r['role'],r['condition']
    fm={row_id(r):r for r in all_committed};em={row_id(r):r for r in all_measured}
    for key,r in fm.items():
        if any(em[key].get(n)!=v for n,v in r.items()):
            raise ValueError('evaluation changed a forward commitment')
    # Share only old frozen preparation and native Gaussian arithmetic. Never
    # import new attribution production, targets, aggregation or its verifier.
    from experiments import round17_regional as SHARED
    import torch
    if SHARED.runtime()!=lock['runtime']:
        raise ValueError('shared replay runtime differs from original primary')
    with np.load(run/'partitions.npz',allow_pickle=False) as archive:
        native={q:{f.split('__',1)[1]:archive[f] for f in archive.files if f.startswith(q+'__')} for q in PARTITIONS}
    tensors=SHARED.S.M.partition_tensors(native,'cpu')
    rebuilt=[];rebuilt_paired=[];rebuilt_strata=[];numeric=[];unique_supports=[]
    total_bytes=0;primary_exact=0;max_metric=0.;same_all=0.;sampled_bias_matrices=0
    for scene,(fd,entries,old) in stages.items():
        for pair_index,entry in enumerate(entries):
            path=absolute(entry['path'])
            with np.load(path/'inputs.npz',allow_pickle=False) as archive:
                inputs={field:archive[field] for field in REF.FIELDS}
            with torch.no_grad():queries,keys=SHARED.prepare_pair(inputs,tensors)
            directory=fd/'arrays'/path.name
            with np.load(directory/'supports.npz',allow_pickle=False) as archive:
                support={field:archive[field] for field in archive.files}
            expected_fields={q+'__'+f for q in PARTITIONS for f in ('mean','cov','bounds','native_trace','tau_obs32','tau_size32')}
            if set(support)!=expected_fields:
                raise ValueError('prepared support inventory changed')
            for q in PARTITIONS:
                for field in ('mean','cov'):
                    actual=support[q+'__'+field];shared=queries[q][field].numpy()
                    if actual.dtype!=np.float32 or array_sha(actual)!=array_sha(shared):
                        raise ValueError('retained actual support differs from original preparation')
                if (support[q+'__bounds'].dtype!=np.int64 or not np.array_equal(support[q+'__bounds'],parts[q]['bounds'])
                        or support[q+'__native_trace'].dtype!=np.float64
                        or not np.array_equal(support[q+'__native_trace'],native_trace(parts[q]['bounds']))):
                    raise ValueError('nominal support trace/bounds changed')
                observed=TRACE.independent_temperature(support[q+'__cov'],.25,10.).astype(np.float32)
                nominal=size_temperature(parts[q]['bounds'])
                if any(support[q+'__'+name].dtype!=np.float32 or not np.array_equal(support[q+'__'+name],v)
                       for name,v in (('tau_obs32',observed),('tau_size32',nominal))):
                    raise ValueError('actual complete tau vectors differ from independent arithmetic')
            unique_supports.append((entry,support))
            # Both archives are first opened only after every scene's global
            # seal/chronology/source checks above; no raw sensor image is read.
            labels=load_labels(entry)
            samples={}
            for c in CONDITIONS:
                q,k,name=c['query'],c['key'],c['condition'];query,key=queries[q],keys[k]
                archive_path=directory/(name+'.npz')
                with np.load(archive_path,allow_pickle=False) as archive:
                    saved={field:archive[field] for field in archive.files}
                base_fields=set(ROLES)|{role+'__argmax' for role in ROLES}
                sample_fields={'indices8','raw8','content8','tau_obs32_8','tau_size32_8','bias_obs32_8','bias_size32_8'}
                if set(saved)!=base_fields|(sample_fields if pair_index==0 else set()):
                    raise ValueError('full probability/actual sample archive inventory differs')
                with torch.no_grad():
                    raw=SHARED.S.M.tiled_cell_mass(query['mean'],query['cov'].double(),key,.25)
                    content=query['features']@key['features'].T
                    geometry=(raw+1e-30*(key['mass'].double()/64)[None]).log()
                    replay=SHARED.P.probabilities(geometry,content,query['cov'],.25,10.).numpy()
                raw_np,content_np=raw.numpy(),content.numpy()
                # Diagnostics originate in Torch's original reduction path.
                torch_fov=raw.sum(-1)
                diagnostics=dict(raw_zero_cells=int((raw==0).sum()),raw_zero_fov_rows=int((torch_fov==0).sum()),
                    raw_tiny_fov_rows=int((torch_fov<1e-20).sum()),raw_fov_min=float(torch_fov.min()),
                    raw_fov_max=float(torch_fov.max()),raw_floor_fov_mass=float(1e-30*(key['mass'].double()/64).sum()))
                original=old[tuple(entry['pair']),name]
                if (array_sha(replay)!=original['array_sha256'] or diagnostics!=original['numerical']
                        or original['input_sha256']!=entry['input_sha256']):
                    raise ValueError('shared original primary no longer exactly reproduces accepted R17B')
                primary_exact+=1;committed={}
                bounds=(parts[k]['bounds'].astype(np.float64)-.5)/8
                for role,suffix in (('trace_full','obs'),('size_temperature','size')):
                    fr=fm[scene,tuple(entry['pair']),role,name];p=saved[role];winner=saved[role+'__argmax']
                    OLD.probability(p,fr['shape']);total_bytes+=p.nbytes
                    if (array_sha(p)!=fr['array_sha256'] or winner.dtype!=np.int32
                            or not np.array_equal(winner,p.argmax(1)) or array_sha(winner)!=fr['argmax_sha256']
                            or fr['archive_path']!=str(archive_path.relative_to(run)) or fr['archive_field']!=role
                            or fr['primary_regional_array_sha256']!=original['array_sha256']
                            or fr['numerical']!=diagnostics):
                        raise ValueError('retained probability/argmax/source commitment mismatch')
                    if role=='trace_full' and array_sha(p)!=original['array_sha256']:
                        raise ValueError('retained primary is not the original accepted operator')
                    expected=independent_probability(raw_np,content_np,support[q+'__tau_'+suffix+'32'],bounds)
                    error=float(np.max(abs(expected-p)));same_all=max(same_all,error)
                    if error>1e-12:
                        raise ValueError('complete same-raw independent tau/softmax differs')
                    committed[role]=fr
                if pair_index==0:
                    idx=REF.selected_indices(len(query['mean']))
                    if (not np.array_equal(saved['indices8'],idx)
                            or array_sha(saved['raw8'])!=array_sha(raw_np[idx])
                            or array_sha(saved['content8'])!=array_sha(content_np[idx])):
                        raise ValueError('retained sample did not follow complete original arithmetic')
                    samples[name]=saved;sampled_bias_matrices+=2
                rows,pair,strata=rebuild_rows(entry,c,parts,{role:saved[role] for role in ROLES},labels,committed)
                for r in rows:
                    error=compare(em[row_id(r)],r,METRIC_TOLERANCE);max_metric=max(max_metric,error)
                rebuilt.extend(rows);rebuilt_paired.append(pair);rebuilt_strata.extend(strata)
            if pair_index==0:
                numeric.extend(dict(scene=scene,pair=entry['pair'],**r) for r in numeric_pair(inputs,parts,support,samples))
            print('SIZE_ATTRIBUTION_AUDIT',scene,pair_index+1,'/16',flush=True)
    validate_population(rebuilt,lock['entries'])
    def pair_key(r):return r['scene'],tuple(r['pair']),r['condition']
    def stratum_key(r):return pair_key(r)+(r['width'],r['height'])
    for actual,expected,key in ((all_pair,rebuilt_paired,pair_key),(all_strata,rebuilt_strata,stratum_key)):
        if len({key(r) for r in actual})!=len(actual) or {key(r) for r in actual}!={key(r) for r in expected}:
            raise ValueError('paired/native-size stratum population differs')
        am={key(r):r for r in actual}
        for r in expected:max_metric=max(max_metric,compare(am[key(r)],r,METRIC_TOLERANCE))
    recomputed=dict(metrics=aggregate(rebuilt,lock['entries']),paired_summary=aggregate_pairs(rebuilt_paired),
        size_strata_summary=aggregate_pairs(rebuilt_strata,stratified=True),support_statistics=support_statistics(unique_supports))
    max_summary=max(compare(result[name],value) for name,value in recomputed.items())
    if (primary_exact!=672 or total_bytes!=RAW_BYTES or len(numeric)!=84 or sampled_bias_matrices!=84
            or len(rebuilt_paired)!=672 or result['matrices']!=MATRICES or result['pairs']!=48
            or result['paired_conditions']!=672 or result['size_stratum_rows']!=len(rebuilt_strata)
            or result['config']!=CONFIG or result['roles']!=list(ROLES) or result['lock_sha256']!=lock_sha
            or result['forward_complete_sha256']!=seal_sha
            or result['descriptive_attribution_only'] is not True or result['independent_audit_pending'] is not True
            or any(result[n] is not False for n in ('scientific_gate_defined','selection_changed',
                'current_validation_changed','efficiency_claim'))):
        raise ValueError('incomplete audit population or unsupported scientific claim')
    passed=all(r['passed'] for r in numeric)
    write(out/'independent_summary.json',recomputed);write(out/'numerical_checks.json',numeric)
    verification=dict(passed=passed,scientific_gate_passed=False,scientific_gate_defined=False,
        descriptive_attribution_only=True,**{'producer_'+n+'_sha256':sha(run/(n+'.json'))
            for n in ('lock','forward_complete','summary','completion')},
        independent_audit_lock_sha256=sha(out/'lock.json'),matrices=MATRICES,primary_old_hash_matches=primary_exact,
        independent_count_tables=672,independent_numerical_matrices=len(numeric),numerical_query_rows=8*len(numeric),
        actual_prepared_bias_sample_matrices=sampled_bias_matrices,unique_support_queries=39888,
        maximum_metric_sum_error=max_metric,maximum_summary_error=max_summary,
        maximum_complete_same_raw_error=same_all,
        maximum_probability_error=max(r['max_probability_error'] for r in numeric),
        maximum_reference_convergence_error=max(r['reference_convergence_error'] for r in numeric),
        maximum_hybrid_error=max(r['hybrid_probability_error'] for r in numeric),
        numerical_failure_count=sum(not r['passed'] for r in numeric),
        argmax_change_count=sum(r['argmax_change_count'] for r in numeric),
        reference_tiny_fov_rows=sum(r['reference_tiny_fov_rows'] for r in numeric),
        shared_old_preparation_and_primary_replay=True,independent_actual_preparation=False,
        independent_reference_moments_and_integration=True,independent_tau_and_metrics=True,
        selection_changed=False,validation_data_read=False,new_sensor_data_read=False,cost_measured=False,
        original_design_created_unix=design['created_unix'],started_unix=started,finished_unix=time.time())
    write(out/'verification.json',verification)
    write(out/'completion.json',dict(passed=passed,scientific_gate_passed=False,
        files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()},finished_unix=time.time()))
    return verification


def audit(run,out):
    no_active_benchmark();run,out=Path(run).resolve(),Path(out).resolve()
    if not out.is_dir() or (out/'started.json').exists() or (out/'completion.json').exists():
        raise ValueError('one already-frozen independent audit execution required')
    started=time.time()
    write(out/'started.json',dict(created_unix=started,producer_lock_sha256=sha(run/'lock.json')))
    try:
        return _audit(run,out,started)
    except Exception as error:
        write(out/'failure.json',dict(passed=False,scientific_gate_passed=False,error_type=type(error).__name__,
            error=str(error),started_unix=started,failed_unix=time.time()))
        write(out/'completion.json',dict(passed=False,scientific_gate_passed=False,
            files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()},finished_unix=time.time()))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('freeze','audit'))
    parser.add_argument('--run',type=Path,default=ROOT/'results/round17_size_temperature_attribution')
    parser.add_argument('--out',type=Path,default=ROOT/'results/round17_size_temperature_independent_audit')
    args=parser.parse_args()
    result=freeze(args.run,args.out) if args.stage=='freeze' else audit(args.run,args.out)
    if args.stage=='audit' and not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
