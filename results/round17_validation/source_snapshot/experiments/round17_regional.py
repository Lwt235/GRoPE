"""R17 regional calibration: one new fixed family, complete audited old controls."""
from __future__ import annotations
import argparse
import copy
import math
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from experiments import round15_regional as OLD
from experiments import round17_feasibility as A
from models import round17_trace_pe as P
ROOT,S,GRID,SCENES=A.ROOT,A.S,A.GRID,A.SCENES
FAMILIES=('trace_full',)
CONDITIONS=OLD.CONDITIONS
RECORDS_PER_PAIR=14*12
TOTAL_RECORDS=48*RECORDS_PER_PAIR
PROBABILITY_BYTES_PER_PAIR=64054368
SAMPLE_BYTES=3*PROBABILITY_BYTES_PER_PAIR
REGIONAL_FIELDS=OLD.REGIONAL_FIELDS
NUMERICAL_POLICY=dict(gaussian_quadrature_order=32,probability_dtype='float64',
    preparation_dtype='float32',content_bias_dtype='float32',temperature_formula_dtype='float64',floor=1e-30)
sha,read,write,array_sha=A.sha,A.read,A.write,A.array_sha
absolute=A.absolute
runtime=OLD.runtime
read_parts=OLD.read_parts
legacy_contract=OLD.legacy_contract
SOURCE_NAMES=('experiments/round17_regional.py','tests/test_round17_regional.py',
 'experiments/round15_regional_audit_metrics.py','tests/test_round15_regional_audit_metrics.py',
 'experiments/round17_regional_audit.py','tests/test_round17_regional_audit.py',
 'experiments/round17_regional_audit_metrics.py','tests/test_round17_regional_audit_metrics.py')


def sources():
    output=A.sources()
    for name in SOURCE_NAMES: output[name]=sha(ROOT/name)
    return output


def stage_a_contract(directory):
    lock=A.verify_lock(directory)
    decision=read(directory/'decision.json');summary=read(directory/'summary.json')
    complete=read(directory/'completion.json')
    A.verify_files({str(directory/p):h for p,h in complete['files'].items()})
    audit_path=absolute(decision['audit_path']);audit=read(audit_path)
    expected={f'feasibility_{n}_sha256':sha(directory/(n+'.json')) for n in ('lock','forward_complete','summary','completion')}
    if (audit['passed'] is not True or audit['scientific_gate_passed'] is not False or
        any(decision.get(k)!=v or audit.get(k)!=v for k,v in expected.items()) or
        decision['audit_sha256']!=sha(audit_path)):
        raise ValueError('independent Stage A accepted source/hash chain required')
    eligible=decision['feasible_grid_indices']
    if (not eligible or eligible!=sorted(set(eligible)) or any(type(i) is not int or i not in range(12) for i in eligible)
        or eligible!=summary['feasible_grid_indices'] or eligible!=audit['feasible_grid_indices']
        or decision['action']!='complete_full_regional_calibration' or decision['primary_selected']
        or decision['scientific_gate_passed'] or decision['validation_authorized']):
        raise ValueError('Stage A did not authorize regional calibration')
    if decision['regional_design']!=dict(new_families=1,reused_families=8,configurations_per_family=12,conditions=14,unchanged_J=True):
        raise ValueError('changed Stage A regional contract')
    return dict(root=str(directory),decision_sha256=sha(directory/'decision.json'),audit_path=str(audit_path),
       audit_sha256=sha(audit_path),**expected,eligible_indices=eligible,
       point_localization_selections=summary['point_localization_selections'],legacy_point=summary['legacy_point'],
       training_manifest_records=lock['training_manifest_records'],inputs=lock['inputs'],manifests=lock['manifests'],
       data_root=lock['data_root'],data_completion_sha256=lock['data_completion_sha256'],
       target_contract=lock['target_contract'],numerical_contract=lock['numerical_contract'],
       baseline_bindings=lock['baseline_bindings'])


def prepare_pair(inputs,partition):
    obs=S.A.observation_tensors(inputs,'cpu')
    queries={n:A.prepare_query(obs,p,True) for n,p in partition.items()}
    keys={n:S.M.prepare_key(obs,p) for n,p in partition.items()}
    return queries,keys


def condition_predictions(query,key):
    content=query['features']@key['features'].T
    for hi,h in enumerate((.25,.5,1.,2.)):
        raw=S.M.tiled_cell_mass(query['mean'],query['cov'].double(),key,h)
        if raw.dtype!=torch.float64 or not bool(torch.isfinite(raw).all()) or bool((raw<0).any()):
            raise ValueError('invalid raw native Gaussian mass')
        fov=raw.sum(-1)
        diagnostics=dict(raw_zero_cells=int((raw==0).sum()),raw_zero_fov_rows=int((fov==0).sum()),
          raw_tiny_fov_rows=int((fov<1e-20).sum()),raw_fov_min=float(fov.min()),raw_fov_max=float(fov.max()),
          raw_floor_fov_mass=float(1e-30*(key['mass'].double()/64).sum()))
        geometry=(raw+1e-30*(key['mass'].double()/64)[None]).log()
        for ti,t in enumerate((0.,5.,10.)):
            probability=P.probabilities(geometry,content,query['cov'],h,t).numpy()
            validate_probability(probability,(len(query['mean']),len(key['mass'])))
            yield 'trace_full',hi*3+ti,probability,diagnostics


def freeze(out, stage_a):
    if out.exists():
        raise ValueError('refusing regional freeze overwrite')
    prior = stage_a_contract(stage_a.resolve())  # No output directory before passed decision.
    legacy = legacy_contract()
    entries = []
    for record in prior['training_manifest_records']:
        manifest = absolute(record['path'])
        if sha(manifest) != record['sha256']:
            raise ValueError('changed old-training manifest')
        metadata = read(manifest)
        if (metadata['split'] != 'train' or metadata['forward_uses_sensor']
                or metadata['scene'] != record['scene'] or metadata['pair'] != record['pair']):
            raise ValueError('unexpected old-training identity/information')
        entries.append(dict(scene=record['scene'], pair=record['pair'], path=str(manifest.parent),
            manifest_sha256=record['sha256'], input_sha256=metadata['input_sha256'],
            label_sha256=metadata['label_sha256']))  # Expected hash only; no label bytes opened.
    if len(entries) != 48 or len({(e['scene'], tuple(e['pair'])) for e in entries}) != 48:
        raise ValueError('regional cohort must contain 48 unique training pairs')
    parts = S.M.partitions()
    computed_bytes = sum(len(parts[c['query']]['mass'])*len(parts[c['key']]['mass']) for c in CONDITIONS)*12*8
    if computed_bytes != PROBABILITY_BYTES_PER_PAIR:
        raise ValueError('native partition/storage population changed')
    out.mkdir(parents=True)
    np.savez_compressed(out / 'partitions.npz', **{name+'__'+k: v for name, p in parts.items() for k, v in p.items()})
    hashes = sources()
    for name in hashes:
        destination = out / 'source_snapshot' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / name).read_bytes())
    lock = dict(created_unix=time.time(), source_hashes=hashes, stage_a=prior, legacy=legacy, entries=entries,
        families=list(FAMILIES), grid=list(GRID), conditions=list(CONDITIONS), runtime=runtime(),
        numerical_policy=NUMERICAL_POLICY, partitions_sha256=sha(out / 'partitions.npz'),
        field_allowlist=list(A.FIELDS), regional_label_allowlist=list(REGIONAL_FIELDS),
        objective_weights=dict(common9=.5, direct5=.5), records=TOTAL_RECORDS,
        full_raw_probability_bytes=48*PROBABILITY_BYTES_PER_PAIR, fixed_sample_raw_bytes=SAMPLE_BYTES,
        sample_pairs={s:min((e for e in entries if e['scene']==s), key=lambda e: Path(e['path']).name)['pair'] for s in SCENES},
        storage='first metadata pair per scene saved; all other probabilities hash-committed then replayed',
        stage_a_probabilities_reused=False, exact_replay_required=True, new_regional_labels_decoded=False,
        fresh_test_read=False, validation_authorized=False)
    write(out / 'lock.json', lock)
    (out / 'lock.sha256').write_text(sha(out / 'lock.json')+'\n')
    print('FROZEN_REGIONAL', sha(out / 'lock.json'), flush=True)


def verify_lock(out):
    lock = read(out / 'lock.json')
    if sha(out / 'lock.json') != (out / 'lock.sha256').read_text().strip() or sources() != lock['source_hashes']:
        raise ValueError('regional source/lock changed')
    for name, digest in lock['source_hashes'].items():
        if sha(out / 'source_snapshot' / name) != digest:
            raise ValueError('regional source snapshot changed')
    if stage_a_contract(Path(lock['stage_a']['root'])) != lock['stage_a'] or legacy_contract() != lock['legacy']:
        raise ValueError('changed Stage A decision or legacy reference chain')
    if (lock['families'] != list(FAMILIES) or lock['grid'] != list(GRID)
            or lock['conditions'] != list(CONDITIONS) or lock['numerical_policy'] != NUMERICAL_POLICY
            or lock['runtime'] != runtime() or lock['objective_weights'] != dict(common9=.5, direct5=.5)):
        raise ValueError('regional operator/grid/runtime/objective changed')
    if sha(out / 'partitions.npz') != lock['partitions_sha256']:
        raise ValueError('regional partitions changed')
    for e in lock['entries']:
        path = Path(e['path'])
        if sha(path / 'manifest.json') != e['manifest_sha256'] or sha(path / 'inputs.npz') != e['input_sha256']:
            raise ValueError('regional training observation changed')
    return lock


def scene_entries(lock, scene):
    if scene not in SCENES:
        raise ValueError('only fixed old-training scenes are authorized')
    entries = sorted((e for e in lock['entries'] if e['scene']==scene), key=lambda e: Path(e['path']).name)
    if len(entries) != 16:
        raise ValueError('expected 16 old-training pairs')
    return entries


def validate_probability(probability, shape):
    if probability.dtype != np.float64 or probability.shape != shape or not np.isfinite(probability).all() or np.any(probability<0):
        raise ValueError('invalid native probability dtype/shape/value')
    if np.max(np.abs(probability.sum(-1)-1)) > 1.e-12:
        raise ValueError('unnormalized native probability')


def identity(row):
    return row['scene'], tuple(row['pair']), row['family'], row['grid_index'], row['condition']


def field_name(family, index):
    return f'{family}__c{index:02d}'


def expected_records(entries):
    return {(e['scene'], tuple(e['pair']), f, i, c['condition'])
            for e in entries for f in FAMILIES for i in range(12) for c in CONDITIONS}


def verify_population(rows, entries, *, metrics=False):
    ids = [identity(r) for r in rows]
    if len(ids) != len(set(ids)) or set(ids) != expected_records(entries):
        raise ValueError('missing, duplicate or unauthorized regional population')
    parts = S.M.partitions()
    specs = {c['condition']: c for c in CONDITIONS}
    paired_mass = {}
    for row in rows:
        if (row['config'] != GRID[row['grid_index']]
                or any(row[k] != specs[row['condition']][k] for k in ('scenario', 'query', 'key'))):
            raise ValueError('regional configuration/condition mismatch')
        shape = [len(parts[row['query']]['mass']), len(parts[row['key']]['mass'])]
        if row['shape'] != shape or row['dtype'] != 'float64':
            raise ValueError('native probability shape/dtype population mismatch')
        if metrics:
            if (row['n_query'] != shape[0] or row['n_key'] != shape[1]
                    or type(row['mass']) is not int or row['mass'] < 0
                    or any(not math.isfinite(row[k]) for k in ('kl_sum','js_sum','direct_kl_sum','direct_js_sum'))
                    or row['replayed_array_sha256'] != row['array_sha256']):
                raise ValueError('invalid regional metric population or replay')
            key = row['scene'], tuple(row['pair'])
            if key in paired_mass and paired_mass[key] != row['mass']:
                raise ValueError('partition/configuration changed visible pixel mass')
            paired_mass[key] = row['mass']


def sample_path(directory, entry, condition):
    return directory / 'samples' / Path(entry['path']).name / (condition+'.npz')


def stage_a_commitments(lock, scene):
    path = Path(lock['stage_a']['root'])/'forward'/scene.replace('/', '__')/'records.json'
    records = read(path)
    result = {(tuple(r['pair']), r['family'], r['grid_index'], r['query']+'_g8'):r for r in records}
    if len(result) != len(records) or len(result) != 16*36:
        raise ValueError('incomplete Stage A probability commitments')
    return result


def check_stage_a_overlap(record, commitments):
    previous = commitments.get((tuple(record['pair']),record['family'],record['grid_index'],record['condition']))
    if previous is not None:
        if previous['array_sha256'] != record['array_sha256'] or previous['input_sha256'] != record['input_sha256']:
            raise ValueError('overlapping Stage A probability/input changed')
        return previous['array_sha256']
    return None


@torch.no_grad()
def forward(out, scene):
    lock = verify_lock(out)
    entries = scene_entries(lock, scene)
    directory = out / 'forward' / scene.replace('/', '__')
    if directory.exists():
        raise ValueError('refusing partial/existing regional forward')
    if shutil.disk_usage(out).free < SAMPLE_BYTES + 512*1024**2:
        raise ValueError('insufficient space for complete declared raw sample')
    directory.mkdir(parents=True)
    started = time.time()
    write(directory / 'started.json', dict(created_unix=started, runtime=runtime(), lock_sha256=sha(out/'lock.json')))
    partition = S.M.partition_tensors(read_parts(out), 'cpu')
    records, repeated_hashes = [], []
    stage_a = stage_a_commitments(lock, scene)
    overlap_count = 0
    for pair_index, entry in enumerate(entries):
        inputs = A.load_forward_inputs(Path(entry['path'])/'inputs.npz')
        queries, keys = prepare_pair(inputs, partition)
        repeated_queries, repeated_keys = prepare_pair(inputs, partition) if pair_index == 0 else (None, None)
        for condition in CONDITIONS:
            q, k = condition['query'], condition['key']
            saved = {}
            for family, index, probability, diagnostics in condition_predictions(queries[q], keys[k]):
                record = dict(scene=scene, pair=entry['pair'], family=family, grid_index=index, config=GRID[index],
                    **condition, shape=list(probability.shape), dtype='float64', array_sha256=array_sha(probability),
                    input_sha256=entry['input_sha256'], numerical=diagnostics)
                previous = check_stage_a_overlap(record, stage_a)
                if previous is not None:
                    record['stage_a_array_sha256'] = previous
                    overlap_count += 1
                if pair_index == 0:
                    saved[field_name(family, index)] = probability
                    record['sample_path'] = str(sample_path(directory, entry, condition['condition']).relative_to(out))
                    record['sample_field'] = field_name(family, index)
                records.append(record)
            if pair_index == 0:
                path = sample_path(directory, entry, condition['condition'])
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(path, **saved)
                for family, index, probability, _ in condition_predictions(repeated_queries[q], repeated_keys[k]):
                    digest = array_sha(probability)
                    if digest != array_sha(saved[field_name(family, index)]):
                        raise ValueError('bitwise first-pair regional repeat failed')
                    repeated_hashes.append(dict(condition=condition['condition'], family=family, grid_index=index,
                                                array_sha256=digest))
        print('REGIONAL_FORWARD', scene, pair_index+1, '/16', flush=True)
    verify_population(records, entries)
    if overlap_count != 16*36:
        raise ValueError('missing Stage A overlapping probability checks')
    write(directory/'records.json', records)
    write(directory/'determinism.json', dict(pair=entries[0]['pair'], records=repeated_hashes,
        passed=True, probability_count=RECORDS_PER_PAIR, labels_decoded=False))
    A.complete_stage(directory, out, 'forward', scene, started, record_count=len(records), pair_count=16,
        runtime=runtime(), regional_labels_decoded=False, all_probability_hashes_retained=True,
        all_probabilities_retained=False, saved_sample_pair=entries[0]['pair'],
        stage_a_exact_overlap_count=overlap_count)


def forward_chain(out):
    result = {}
    for scene in SCENES:
        directory = out/'forward'/scene.replace('/', '__')
        complete = A.verify_stage(directory, out, 'forward', scene)
        if (complete['record_count'] != 16*RECORDS_PER_PAIR or complete['regional_labels_decoded']
                or complete['stage_a_exact_overlap_count'] != 16*36):
            raise ValueError('incomplete or supervised regional forward')
        if complete['runtime'] != runtime():
            raise ValueError('changed runtime between regional passes')
        result[scene] = dict(completion_sha256=sha(directory/'completion.json'), finished_unix=complete['finished_unix'])
    return result


def seal_forward(out):
    lock = verify_lock(out)
    if (out/'forward_complete.json').exists():
        raise ValueError('refusing regional seal overwrite')
    chain, records = forward_chain(out), []
    for scene in SCENES:
        directory = out/'forward'/scene.replace('/', '__')
        current = read(directory/'records.json')
        entries = scene_entries(lock, scene)
        verify_population(current, entries)
        mapping = {(tuple(e['pair'])): e for e in entries}
        stage_a = stage_a_commitments(lock, scene)
        for row in current:
            if row['input_sha256'] != mapping[tuple(row['pair'])]['input_sha256']:
                raise ValueError('committed input identity mismatch')
            previous = check_stage_a_overlap(row, stage_a)
            if previous != row.get('stage_a_array_sha256'):
                raise ValueError('Stage A overlap commitment mismatch')
        first = [r for r in current if r['pair'] == entries[0]['pair']]
        others = [r for r in current if r['pair'] != entries[0]['pair']]
        if any('sample_path' in r or 'sample_field' in r for r in others):
            raise ValueError('undeclared raw sample selection')
        repeat = read(directory/'determinism.json')
        expected = {(r['condition'],r['family'],r['grid_index']):r['array_sha256'] for r in first}
        found = {(r['condition'],r['family'],r['grid_index']):r['array_sha256'] for r in repeat['records']}
        if repeat['passed'] is not True or repeat['pair'] != entries[0]['pair'] or expected != found or len(repeat['records']) != RECORDS_PER_PAIR:
            raise ValueError('regional determinism commitment mismatch')
        for condition in CONDITIONS:
            path = sample_path(directory, entries[0], condition['condition'])
            selected = [r for r in first if r['condition']==condition['condition']]
            with np.load(path, allow_pickle=False) as archive:
                if set(archive.files) != {field_name(f,i) for f in FAMILIES for i in range(12)}:
                    raise ValueError('incomplete raw audit sample')
                for row in selected:
                    if row['sample_path'] != str(path.relative_to(out)) or row['sample_field'] != field_name(row['family'], row['grid_index']):
                        raise ValueError('sample identity mismatch')
                    probability = archive[row['sample_field']]
                    validate_probability(probability, tuple(row['shape']))
                    if array_sha(probability) != row['array_sha256']:
                        raise ValueError('saved sample differs from commitment')
        records.extend(current)
    verify_population(records, lock['entries'])
    write(out/'forward_complete.json', dict(lock_sha256=sha(out/'lock.json'), forward_chain=chain,
        sealed_unix=time.time(), record_count=len(records), raw_sample_count=3*RECORDS_PER_PAIR,
        raw_sample_bytes=SAMPLE_BYTES, regional_labels_decoded=False, all_probabilities_retained=False))


def verify_forward_complete(out):
    value = read(out/'forward_complete.json')
    if (value['lock_sha256'] != sha(out/'lock.json') or value['forward_chain'] != forward_chain(out)
            or value['record_count'] != TOTAL_RECORDS or value['regional_labels_decoded']):
        raise ValueError('global regional forward seal mismatch')
    return value


def load_regional_labels(entry):
    path = Path(entry['path'])/'labels.npz'
    if sha(path) != entry['label_sha256']:
        raise ValueError('old regional labels changed')
    with np.load(path, allow_pickle=False) as archive:
        labels = {key:archive[key] for key in REGIONAL_FIELDS}
    if (labels['gt_xy'].shape != (144*192,2) or labels['gt_visible'].shape != (144*192,)
            or labels['gt_visible'].dtype != np.bool_):
        raise ValueError('unexpected dense regional label population')
    return labels


def measure(probability, parts, condition, direct, common):
    q, k = condition['query'], condition['key']
    validate_probability(probability, direct.shape)
    dkl, djs, mass = S.kl_js(probability, direct)
    if condition['scenario'] == 'partition':
        projected = S.common_probability(probability, parts[q], parts[k])
        kl, js, common_mass = S.kl_js(projected, common)
        if common_mass != mass:
            raise ValueError('common and native targets have different mass')
    elif condition['scenario'] == 'scale':
        kl, js = dkl, djs
    else:
        raise ValueError('unknown regional metric family')
    return dict(kl_sum=kl, js_sum=js, direct_kl_sum=dkl, direct_js_sum=djs,
                mass=int(mass), n_query=probability.shape[0], n_key=probability.shape[1])


def require_replay(probability, record):
    validate_probability(probability, tuple(record['shape']))
    digest = array_sha(probability)
    if digest != record['array_sha256']:
        context = {k:record.get(k) for k in ('scene','pair','family','grid_index','condition')}
        raise ValueError(f'regional probability replay is not bitwise identical: {context}; expected={record["array_sha256"]}; actual={digest}')
    return digest


@torch.no_grad()
def evaluate(out, scene):
    lock = verify_lock(out)
    entries = scene_entries(lock, scene)
    seal = verify_forward_complete(out)  # Global pass1 completion before any regional label bytes.
    directory = out/'evaluation'/scene.replace('/', '__')
    if directory.exists():
        raise ValueError('refusing partial/existing regional evaluation')
    started = time.time()
    if seal['sealed_unix'] > started:
        raise ValueError('regional forward chronology')
    directory.mkdir(parents=True)
    write(directory/'started.json', dict(created_unix=started, runtime=runtime(),
        forward_complete_sha256=sha(out/'forward_complete.json')))
    parts = read_parts(out)
    partition = S.M.partition_tensors(parts, 'cpu')
    committed = read(out/'forward'/scene.replace('/', '__')/'records.json')
    verify_population(committed, entries)
    commitments = {identity(row):row for row in committed}
    rows, labels_used = [], {}
    for pair_index, entry in enumerate(entries):
        inputs = A.load_forward_inputs(Path(entry['path'])/'inputs.npz')
        queries, keys = prepare_pair(inputs, partition)
        labels = None
        for condition in CONDITIONS:
            q, k = condition['query'], condition['key']
            direct = common = None
            for family, index, probability, diagnostics in condition_predictions(queries[q], keys[k]):
                record = commitments[(scene,tuple(entry['pair']),family,index,condition['condition'])]
                digest = require_replay(probability, record)  # Before loading/computing this record's targets.
                if diagnostics != record['numerical']:
                    raise ValueError('raw-mass diagnostics changed on replay')
                if labels is None:
                    labels = load_regional_labels(entry)
                    labels_used[str(Path(entry['path'])/'labels.npz')] = entry['label_sha256']
                if direct is None:
                    direct, common = S.target_counts(labels, parts, q, k)
                row = measure(probability, parts, condition, direct, common)
                rows.append(dict(**record, **row, label_sha256=entry['label_sha256'], replayed_array_sha256=digest))
        print('REGIONAL_EVALUATE', scene, pair_index+1, '/16', flush=True)
    verify_population(rows, entries, metrics=True)
    write(directory/'rows.json', rows)
    A.complete_stage(directory, out, 'evaluation', scene, started, record_count=len(rows),
        runtime=runtime(), forward_complete_sha256=sha(out/'forward_complete.json'), label_sources=labels_used,
        new_regional_labels_decoded=True, all_replay_hashes_exact=True)


def summarize_rows(rows, entries, eligible_indices):
    verify_population(rows, entries, metrics=True)
    if not eligible_indices or sorted(set(eligible_indices)) != eligible_indices or any(i not in range(12) for i in eligible_indices):
        raise ValueError('nonempty audited eligible set required')
    grouped = {}
    for row in rows:
        grouped.setdefault((row['family'],row['grid_index'],row['scene'],row['condition']), []).append(row)
    metrics = {}
    def mean(values):
        values = list(values)
        return math.fsum(values)/len(values)
    for family in FAMILIES:
        values = []
        for index in range(12):
            by_scene_condition, by_scene = {}, {}
            for scene in SCENES:
                by_scene_condition[scene] = {}
                for condition in CONDITIONS:
                    selected = grouped[(family,index,scene,condition['condition'])]
                    mass = sum(r['mass'] for r in selected)
                    if mass <= 0:
                        raise ValueError('empty scene/condition regional denominator')
                    by_scene_condition[scene][condition['condition']] = dict(
                        **{name:math.fsum(r[field] for r in selected)/mass for name,field in
                           (('kl','kl_sum'),('js','js_sum'),('direct_kl','direct_kl_sum'),('direct_js','direct_js_sum'))}, mass=mass)
                terms = {name:mean(by_scene_condition[scene][c['condition']]['kl'] for c in CONDITIONS if c['scenario']==scenario)
                         for name,scenario in (('common9_kl','partition'),('direct5_kl','scale'))}
                by_scene[scene] = dict(**terms, J=.5*terms['common9_kl']+.5*terms['direct5_kl'])
            common9 = mean(v['common9_kl'] for v in by_scene.values())
            direct5 = mean(v['direct5_kl'] for v in by_scene.values())
            by_condition = {c['condition']:{metric:mean(by_scene_condition[s][c['condition']][metric] for s in SCENES)
                            for metric in ('kl','js','direct_kl','direct_js')} for c in CONDITIONS}
            values.append(dict(grid_index=index, config=GRID[index], J=.5*common9+.5*direct5,
                common9_kl=common9, direct5_kl=direct5, by_scene=by_scene,
                by_scene_condition=by_scene_condition, by_condition=by_condition))
        metrics[family] = values
    def select(family, indices=range(12)):
        index = min(indices, key=lambda i:metrics[family][i]['J'])
        return dict(family=family, grid_index=index, config=GRID[index], J=metrics[family][index]['J'])
    old=read(A.OLD_B/'summary.json')
    prior=old['selection']
    selection=dict(primary=select('trace_full',eligible_indices),unconstrained_primary=select('trace_full'),
      eligible_indices=list(eligible_indices),controls=copy.deepcopy(prior['controls']),
      strong_point=copy.deepcopy(prior['strong_point']),strong_quadrature=copy.deepcopy(prior['strong_quadrature']),
      historical_primary=copy.deepcopy(prior['primary']))
    return dict(metrics=metrics,selection=selection,train_screen=train_screen(metrics['trace_full'],selection,old))



def train_screen(values,selection,old):
    primary=selection['primary'];candidate=values[primary['grid_index']]
    def reference(family,index):
        v=old['metrics'][family][index]
        return dict(family=family,grid_index=index,config=GRID[index],J=v['J'])
    refs=dict(strong_point=selection['strong_point'],legacy_point=reference('gaussian_point',7),
              strong_quadrature=selection['strong_quadrature'],legacy_quad16=reference('gaussian_quad16',4))
    checks=[]
    for names in (('strong_point','legacy_point'),('strong_quadrature','legacy_quad16')):
      for metric in ('common9_kl','direct5_kl'):
        for name in names:
            ref=refs[name]
            multiplier=.98 if 'point' in name else 1.01 if metric=='common9_kl' else 1.02
            maximum=multiplier-1
            a=candidate[metric];b=old['metrics'][ref['family']][ref['grid_index']][metric]
            bound=b*multiplier
            checks.append(dict(name=metric+'_vs_'+name,metric=metric,reference_name=name,reference=ref,
              candidate_value=a,reference_value=b,relative_difference=a/b-1 if b>0 else None,
              maximum_relative_difference=maximum,bound=bound,passed=b>0 and a<=bound))
    passed=all(c['passed'] for c in checks)
    return dict(passed=passed,fixed_primary=primary,checks=checks,scientific_gate_passed=False,
       next_action='prepare_old_validation_protocol' if passed else 'stop_selected_candidate_before_old_validation')


def summarize(out):
    lock = verify_lock(out)
    verify_forward_complete(out)
    if (out/'summary.json').exists() or (out/'completion.json').exists():
        raise ValueError('refusing regional summary overwrite')
    rows = []
    for scene in SCENES:
        directory = out/'evaluation'/scene.replace('/', '__')
        complete = A.verify_stage(directory, out, 'evaluation', scene)
        if (complete['forward_complete_sha256'] != sha(out/'forward_complete.json')
                or complete['runtime'] != runtime() or complete['all_replay_hashes_exact'] is not True):
            raise ValueError('regional evaluation replay chain mismatch')
        rows.extend(read(directory/'rows.json'))
    result = summarize_rows(rows, lock['entries'], lock['stage_a']['eligible_indices'])
    old=read(A.OLD_B/'summary.json')
    result['legacy_references'] = {name:dict(**spec, summary=old['metrics'][spec['family']][spec['grid_index']])
                                    for name,spec in lock['legacy']['configurations'].items()}
    result['old_baseline_summary_sha256']=sha(A.OLD_B/'summary.json')
    result.update(stage_a_point_localization_selections=lock['stage_a']['point_localization_selections'],
        legacy_point_localization=lock['stage_a']['legacy_point'], row_count=len(rows), pair_count=48,
        lock_sha256=sha(out/'lock.json'), forward_complete_sha256=sha(out/'forward_complete.json'),
        created_unix=time.time(), selection_pending_independent_audit=True, scientific_gate_passed=False,
        validation_run=False, fresh_confirmation=False, all_replay_hashes_exact=True)
    write(out/'summary.json', result)
    files = {str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}
    write(out/'completion.json', dict(lock_sha256=sha(out/'lock.json'), files=files,
        summary_sha256=sha(out/'summary.json'), forward_complete_sha256=sha(out/'forward_complete.json'),
        finished_unix=time.time(), independent_audit_required=True, all_replay_hashes_exact=True,
        scientific_gate_passed=False, validation_run=False))
    print('REGIONAL_SELECTION_PROPOSED', result['selection']['primary'], 'AUDIT PENDING', flush=True)


def accept_audit(out, verification_path):
    lock = verify_lock(out)
    complete, summary, audit = read(out/'completion.json'), read(out/'summary.json'), read(verification_path)
    A.verify_files({str(out/path):digest for path,digest in complete['files'].items()})
    expected = {f'regional_{name}_sha256':sha(out/(name+'.json'))
                for name in ('lock','forward_complete','summary','completion')}
    if (audit['passed'] is not True or audit['scientific_gate_passed'] is not False
            or any(audit.get(key) != value for key,value in expected.items())
            or audit['selection'] != summary['selection'] or audit['train_screen'] != summary['train_screen']):
        raise ValueError('independent regional selection/hash audit required')
    selected = dict(**expected, audit_path=str(verification_path.resolve()), audit_sha256=sha(verification_path),
        selection=summary['selection'],train_screen=summary['train_screen'], point_localization_selections=lock['stage_a']['point_localization_selections'],
        legacy_configurations=lock['legacy']['configurations'], stage_a_decision_sha256=lock['stage_a']['decision_sha256'],
        created_unix=time.time(), scientific_gate_passed=False,
        next_action=summary['train_screen']['next_action'], validation_run=False)
    write(out/'selection.json', selected)
    (out/'selection.sha256').write_text(sha(out/'selection.json')+'\n')
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('freeze','forward','seal-forward','evaluate','summarize','accept-audit'))
    parser.add_argument('--out', type=Path, default=ROOT/'results/round17_regional')
    parser.add_argument('--stage-a', type=Path, default=ROOT/'results/round17_feasibility')
    parser.add_argument('--scene', choices=SCENES)
    parser.add_argument('--audit', type=Path)
    args = parser.parse_args()
    if args.stage == 'freeze':
        freeze(args.out, args.stage_a)
    elif args.stage == 'forward':
        forward(args.out, args.scene)
    elif args.stage == 'seal-forward':
        seal_forward(args.out)
    elif args.stage == 'evaluate':
        evaluate(args.out, args.scene)
    elif args.stage == 'summarize':
        summarize(args.out)
    else:
        if args.audit is None:
            parser.error('accept-audit requires --audit')
        accept_audit(args.out, args.audit)

if __name__=='__main__': main()
