"""Conditional R15 old-development regression; draft until separately frozen.

No final configuration is embedded here. Freeze requires the accepted Stage B
selection and its independent audit. New observations/labels are never sought.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import resource
import shutil
import time

import numpy as np
import torch

from experiments import round15_regional as R
from experiments import round15_validation_metrics as V
from models import round15_validation_pe as P

A, S, ROOT = R.A, R.S, R.ROOT
SCENES = S.SCENES
FIELDS = (*A.FIELDS, 'q_points3', 'q_pixel_cov3', 'q_to_k_transform', 'k_K',
          'q_predicted_depth', 'k_predicted_depth')
STATES = S.M.STATES
CONDITIONS = tuple(dict(scenario=state, state=state, query=q, key=k, condition=q+'_'+k)
                   for state in STATES for q, k in S.M.ORIGINAL_CONDITIONS) + tuple(
    dict(scenario='scale', state='zero', query=q, key=k, condition=q+'_'+k)
    for q, k in S.M.SCALE_CONDITIONS)
LABEL_FIELDS = ('gt_xy', 'gt_visible', *(q+'_point_'+field
               for q in S.M.PARTITIONS for field in ('xy', 'visible', 'status')))
PRIMARY_BYTES = 1112516352
CONTROL_SAMPLE_BYTES = 69532272
sha, read, write, array_sha = A.sha, A.read, A.write, A.array_sha


def sources():
    values = R.sources()
    for name in ('ROUND15_VALIDATION_PROTOCOL.md', 'models/round15_validation_pe.py',
                 'experiments/round15_validation.py', 'experiments/round15_validation_metrics.py',
                 'tests/test_round15_validation.py', 'tests/test_round15_validation_pe.py',
                 'tests/test_round15_validation_metrics.py',
                 'docs/round15_validation_blueprint.md',
                 'docs/round15_validation_draft_preparation_note.md'):
        values[name] = sha(ROOT/name)
    return values


def selected_contract(directory):
    lock = R.verify_lock(directory)
    selected = read(directory/'selection.json')
    if sha(directory/'selection.json') != (directory/'selection.sha256').read_text().strip():
        raise ValueError('unsealed regional selection')
    complete, summary = read(directory/'completion.json'), read(directory/'summary.json')
    A.verify_files({str(directory/p):h for p,h in complete['files'].items()})
    audit_path = Path(selected['audit_path'])
    audit = read(audit_path)
    expected = {f'regional_{name}_sha256':sha(directory/(name+'.json'))
                for name in ('lock','forward_complete','summary','completion')}
    if (audit.get('passed') is not True or audit.get('scientific_gate_passed') is not False
            or selected['audit_sha256'] != sha(audit_path)
            or any(selected.get(k)!=v or audit.get(k)!=v for k,v in expected.items())
            or selected['selection'] != summary['selection'] or audit['selection'] != summary['selection']
            or selected['next_action'] != 'freeze_separate_old_validation_regression_protocol'
            or selected['validation_run'] or selected['scientific_gate_passed']
            or selected['point_localization_selections'] != lock['stage_a']['point_localization_selections']
            or selected['legacy_configurations'] != lock['legacy']['configurations']):
        raise ValueError('complete independently accepted Stage B selection required')
    specs = {}
    def add(alias, spec):
        specs[alias] = dict(family=spec['family'], **spec['config'])
    selection = selected['selection']
    add('primary', selection['primary'])
    for family in R.FAMILIES[1:]:
        add('regional_'+family, selection['controls'][family])
    for family in ('gaussian_point', 'quartic_point'):
        add('localizer_'+family, dict(family=family, **selected['point_localization_selections'][family]))
    for name in ('point','point_pck','fixed','full1','full4','quad4','quad16'):
        add('legacy_'+name, selected['legacy_configurations'][name])
    specs['matched_isotropic'] = dict(specs['primary'], family='quartic_isotropic_matched')
    roles = V.build_roles(specs, 'regional_'+selection['strong_point']['family'],
                          'regional_'+selection['strong_quadrature']['family'])
    return dict(root=str(directory), selection_sha256=sha(directory/'selection.json'),
                audit_path=str(audit_path), audit_sha256=sha(audit_path), **expected, roles=roles)


def old_target_contract():
    """Bind audited old targets by expected hashes; do not read label bytes."""
    directory = ROOT/'results/round12_stress'
    lock = S.verify_lock(directory)
    complete = read(directory/'completion.json')
    audit_path = ROOT/'results/round12_stress_independent_audit/verification.json'
    audit = read(audit_path)
    audit_lock_path=ROOT/'results/round12_stress_audit_source_lock.json'
    audit_lock=read(audit_lock_path)
    A.verify_files({str(ROOT/p):h for p,h in audit_lock['source_hashes'].items()})
    if sha(audit_lock_path)!=audit['audit_source_lock_sha256'] or sha(audit_lock['snapshot'])!=audit_lock['snapshot_sha256']:
        raise ValueError('old independent target audit source changed')
    if (audit['passed'] is not True or audit['scientific_gate_passed'] is not False
            or audit['stress_lock_sha256'] != sha(directory/'lock.json')
            or audit['stress_completion_sha256'] != sha(directory/'completion.json')
            or audit['stress_summary_sha256'] != complete['files']['summary.json']
            or complete['lock_sha256'] != sha(directory/'lock.json')
            or audit['independent_sensor_centers'] != 1536):
        raise ValueError('independently verified old center-target provenance required')
    entries, metadata_files = [], {}
    for scene in SCENES:
        stage = directory/'evaluation'/scene.replace('/', '__')
        stage_complete = read(stage/'completion.json')
        rel = str((stage/'completion.json').relative_to(directory))
        if sha(stage/'completion.json') != complete['files'][rel]:
            raise ValueError('old target evaluation chain changed')
        provenance_path = stage/'label_provenance.json'
        if sha(provenance_path) != stage_complete['files']['label_provenance.json']:
            raise ValueError('old target provenance changed')
        metadata_files[str(provenance_path)] = sha(provenance_path)
        metadata_files[str(stage/'completion.json')] = sha(stage/'completion.json')
        provenance = {tuple(p['pair']):p for p in read(provenance_path)}
        for entry in S.scene_entries(lock, scene):
            path = Path(entry['path'])
            meta = read(path/'manifest.json')
            source = provenance[tuple(entry['pair'])]
            target = stage/'labels'/(path.name+'.npz')
            target_rel = str(target.relative_to(stage))
            target_sha = stage_complete['files'][target_rel]
            if (meta['split']!='val' or meta['forward_uses_sensor'] or meta['scene']!=scene
                    or meta['pair']!=entry['pair'] or source['scene']!=scene
                    or source['old_labels_sha256']!=meta['label_sha256']
                    or source['sensor_sha256']!=meta['sensor_sources']
                    or source['source_selection_sha256']!=meta['source_selection_sha256']
                    or source['projected_centroid_used']
                    or complete['files'][str(target.relative_to(directory))] != target_sha):
                raise ValueError('old target pair/source population mismatch')
            entries.append(dict(**entry, manifest_sha256=sha(path/'manifest.json'),
                input_sha256=meta['input_sha256'], old_label_sha256=meta['label_sha256'],
                target_path=str(target), target_sha256=target_sha, target_provenance=source))
    if len(entries)!=32 or len({(e['scene'],tuple(e['pair'])) for e in entries})!=32:
        raise ValueError('exact old 32-pair population required')
    return dict(root=str(directory), lock_sha256=sha(directory/'lock.json'),
        completion_sha256=sha(directory/'completion.json'), audit_path=str(audit_path),
        audit_sha256=sha(audit_path), audit_source_lock_sha256=sha(audit_lock_path),metadata_files=metadata_files, entries=entries,
        old_scientific_gate_passed=False, target_archive_bytes_read=False)


def runtime():
    return R.runtime()


def sample_bytes(roles):
    count = len(roles['operators'])
    if not 9 <= count <= 18:
        raise ValueError('unexpected unique operator count')
    return PRIMARY_BYTES + (count-1)*CONTROL_SAMPLE_BYTES


def zero_path_check(regional_lock):
    rows=[]
    for scene in R.SCENES:
        entry=min((e for e in regional_lock['entries'] if e['scene']==scene),key=lambda e:Path(e['path']).name)
        path=Path(entry['path'])/'inputs.npz'
        if sha(path)!=entry['input_sha256']:raise ValueError('zero preflight input changed')
        inputs=load_inputs(path)
        raw,diagnostic=S.M.perturbed_observations(inputs,'zero')
        cached=S.A.observation_tensors(inputs,'cpu');reprojected=S.A.observation_tensors(raw,'cpu')
        fields={name:dict(cached_sha256=array_sha(cached[name].numpy()),
                         reprojected_sha256=array_sha(reprojected[name].numpy()),
                         exactly_equal=torch.equal(cached[name],reprojected[name])) for name in A.FIELDS}
        rows.append(dict(scene=scene,pair=entry['pair'],input_sha256=entry['input_sha256'],fields=fields,
                         zero_xy_max_error=diagnostic['zero_xy_max_error'],zero_cov_max_error=diagnostic['zero_cov_max_error']))
    return dict(passed=all(v['exactly_equal'] for row in rows for v in row['fields'].values()),
                rows=rows,old_training_only=True,labels_decoded=False)


def freeze(out, regional):
    if out.exists():
        raise ValueError('refusing validation freeze overwrite')
    prior = selected_contract(regional.resolve())
    targets = old_target_contract()
    rlock = read(regional/'lock.json')
    zero_preflight=zero_path_check(rlock)
    if zero_preflight['passed'] is not True:
        write(out.with_name(out.name+'_zero_path_failure.json'),zero_preflight)
        raise ValueError('zero float32 path differs; preserve failure and resolve before validation freeze')
    required = sample_bytes(prior['roles'])
    if shutil.disk_usage(out.parent).free < required+2*1024**3:
        raise ValueError('insufficient space for declared validation evidence plus reserve')
    timing_entries = []
    for scene in (*R.SCENES, *SCENES):
        source = rlock['entries'] if scene in R.SCENES else targets['entries']
        first = min((e for e in source if e['scene']==scene), key=lambda e:Path(e['path']).name)
        timing_entries.append({k:first[k] for k in ('scene','pair','path','input_sha256','manifest_sha256')})
    parts = S.M.partitions()
    scalars = sum(len(parts[c['query']]['mass'])*len(parts[c['key']]['mass']) for c in CONDITIONS)
    if len(CONDITIONS)!=68 or scalars*32*8 != PRIMARY_BYTES:
        raise ValueError('validation condition/storage population changed')
    out.mkdir(parents=True)
    np.savez_compressed(out/'partitions.npz', **{q+'__'+k:v for q,p in parts.items() for k,v in p.items()})
    hashes = sources()
    for name in hashes:
        dest = out/'source_snapshot'/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((ROOT/name).read_bytes())
    lock = dict(created_unix=time.time(), source_hashes=hashes, regional=prior, targets=targets,
        entries=targets['entries'], roles=prior['roles'], conditions=list(CONDITIONS),
        runtime=runtime(), field_allowlist=list(FIELDS), label_allowlist=list(LABEL_FIELDS),
        partitions_sha256=sha(out/'partitions.npz'), retained_probability_bytes=required,
        numerical_policy=A.NUMERICAL_POLICY, timing_entries=timing_entries,zero_path_preflight=zero_preflight,
        timing_policy=dict(warmups=3,repeats=15,cpu_threads=4,rotating_operator_order=True,
                           conditions=9,scene_count=5,include_role_preparation=True),
        sample_pairs={s:min((e for e in targets['entries'] if e['scene']==s),key=lambda e:Path(e['path']).name)['pair'] for s in SCENES},
        primary_all_arrays_retained=True, controls='first pair per scene retained; all hash committed before labels',
        new_q16_protection=True, matched_isotropic_attribution_only=True,
        validation_labels_decoded=False, fresh_test_read=False, scientific_gate_passed=False)
    write(out/'lock.json', lock)
    (out/'lock.sha256').write_text(sha(out/'lock.json')+'\n')
    print('FROZEN_VALIDATION', sha(out/'lock.json'), flush=True)


def verify_lock(out):
    lock = read(out/'lock.json')
    if sha(out/'lock.json') != (out/'lock.sha256').read_text().strip() or lock['source_hashes']!=sources():
        raise ValueError('validation source/lock changed')
    A.verify_files({str(out/'source_snapshot'/p):h for p,h in lock['source_hashes'].items()})
    if (selected_contract(Path(lock['regional']['root']))!=lock['regional']
            or old_target_contract()!=lock['targets'] or lock['runtime']!=runtime()
            or lock['conditions']!=list(CONDITIONS) or lock['roles']!=lock['regional']['roles']
            or sha(out/'partitions.npz')!=lock['partitions_sha256']):
        raise ValueError('validation upstream/operator/partition changed')
    for e in (*lock['entries'],*lock['timing_entries']):
        path = Path(e['path'])
        if sha(path/'manifest.json')!=e['manifest_sha256'] or sha(path/'inputs.npz')!=e['input_sha256']:
            raise ValueError('validation forward observation changed')
    return lock


def load_inputs(path):
    with np.load(path,allow_pickle=False) as archive:
        return {k:archive[k] for k in FIELDS}


def identity(row):
    return row['scene'],tuple(row['pair']),row['operator'],row['scenario'],row['condition']


def field_name(operator):
    return operator


def native_path(out,entry,condition):
    return out/'forward'/entry['scene'].replace('/','__')/'arrays'/Path(entry['path']).name/(condition['scenario']+'__'+condition['condition']+'.npz')


def forward_population(rows,entries,roles):
    expected={(e['scene'],tuple(e['pair']),op,c['scenario'],c['condition'])
              for e in entries for op in roles['operators'] for c in CONDITIONS}
    ids=[identity(r) for r in rows]
    if len(ids)!=len(set(ids)) or set(ids)!=expected:
        raise ValueError('missing/duplicate/unexpected validation forward population')
    parts=S.M.partitions()
    specs={(c['scenario'],c['condition']):c for c in CONDITIONS}
    for row in rows:
        spec=specs[(row['scenario'],row['condition'])]
        if (row['config']!=roles['operators'][row['operator']]
                or any(row[k]!=spec[k] for k in ('query','key','state'))
                or row['dtype']!='float64'
                or row['shape']!=[len(parts[row['query']]['mass']),len(parts[row['key']]['mass'])]):
            raise ValueError('validation forward config/shape mismatch')


def prepare_state(inputs,state,partition,operators):
    raw,diagnostic=S.M.perturbed_observations(inputs,state)
    obs=S.A.observation_tensors(raw,'cpu')
    names=S.M.PARTITIONS if state=='zero' else S.M.PARTITIONS[:3]
    families=tuple(dict.fromkeys(c['family'] for c in operators.values()))
    queries={q:{f:P.prepare_query(obs,partition[q],f) for f in families} for q in names}
    keys={k:{f:P.prepare_key(obs,partition[k],f) for f in families} for k in names}
    return queries,keys,diagnostic


def predict_prepared(queries,key_bank,config):
    key=key_bank[config['family']]
    probability,diagnostics=P.predict(queries[config['family']],key,config['family'],config['h'],config['temperature'])
    probability=probability.cpu().numpy()
    R.validate_probability(probability,(len(queries[config['family']]['mean']),len(key['mass'])))
    return probability,diagnostics


def require_audit_freeze(out):
    directory=ROOT/'results/round15_validation_independent_audit'
    lock=read(directory/'lock.json')
    if (sha(directory/'lock.json')!=(directory/'lock.sha256').read_text().strip()
            or lock['producer_lock_sha256']!=sha(out/'lock.json')
            or Path(lock['run']).resolve()!=out.resolve()):
        raise ValueError('separate independent validation audit must be frozen before forward')
    A.verify_files({str(ROOT/p):h for p,h in lock['source_hashes'].items()})
    A.verify_files({str(directory/'source_snapshot'/p):h for p,h in lock['source_hashes'].items()})
    return lock


@torch.no_grad()
def forward(out,scene):
    lock=verify_lock(out)
    require_audit_freeze(out)
    entries=S.scene_entries(lock,scene)
    directory=out/'forward'/scene.replace('/','__')
    if directory.exists():raise ValueError('refusing existing validation forward')
    if shutil.disk_usage(out).free<lock['retained_probability_bytes']+512*1024**2:
        raise ValueError('insufficient forward evidence space')
    directory.mkdir(parents=True)
    started=time.time()
    write(directory/'started.json',dict(created_unix=started,runtime=runtime(),lock_sha256=sha(out/'lock.json')))
    partition=S.M.partition_tensors(S.read_parts(out),'cpu')
    operators=lock['roles']['operators']; primary=lock['roles']['aliases']['primary']
    rows,geometries,repeats=[],[],[]
    for index,entry in enumerate(entries):
        inputs=load_inputs(Path(entry['path'])/'inputs.npz')
        for state in STATES:
            queries,keys,diagnostic=prepare_state(inputs,state,partition,operators)
            geometries.append(dict(scene=scene,pair=entry['pair'],**diagnostic))
            repeated=prepare_state(inputs,state,partition,operators) if index==0 else None
            for c in (c for c in CONDITIONS if c['state']==state):
                saved={}
                for op,cfg in operators.items():
                    probability,numeric=predict_prepared(queries[c['query']],keys[c['key']],cfg)
                    row=dict(scene=scene,pair=entry['pair'],operator=op,config=cfg,**c,
                        shape=list(probability.shape),dtype='float64',array_sha256=array_sha(probability),
                        argmax_sha256=array_sha(probability.argmax(-1).astype(np.int32)),
                        input_sha256=entry['input_sha256'],numerical=numeric)
                    if op==primary or index==0:
                        saved[op]=probability
                        row.update(sample_path=str(native_path(out,entry,c).relative_to(out)),sample_field=op)
                    rows.append(row)
                    if index==0:
                        other,other_numeric=predict_prepared(repeated[0][c['query']],repeated[1][c['key']],cfg)
                        if array_sha(other)!=row['array_sha256'] or other_numeric!=numeric:
                            raise ValueError('validation first-pair full-repeat mismatch')
                        repeats.append({k:row[k] for k in ('operator','scenario','condition','array_sha256')})
                dest=native_path(out,entry,c);dest.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(dest,**saved)
        print('VALIDATION_FORWARD',scene,index+1,'/16',flush=True)
    forward_population(rows,entries,lock['roles'])
    write(directory/'records.json',rows);write(directory/'geometry.json',geometries);write(directory/'repeat.json',repeats)
    A.complete_stage(directory,out,'forward',scene,started,record_count=len(rows),runtime=runtime(),labels_decoded=False)


def forward_chain(out):
    result={}
    for scene in SCENES:
        directory=out/'forward'/scene.replace('/','__')
        value=A.verify_stage(directory,out,'forward',scene)
        if value['runtime']!=runtime() or value['labels_decoded']:
            raise ValueError('invalid validation forward stage')
        result[scene]=dict(completion_sha256=sha(directory/'completion.json'),finished_unix=value['finished_unix'])
    return result


def seal_forward(out):
    if (out/'forward_complete.json').exists():raise ValueError('refusing global forward seal overwrite')
    lock=verify_lock(out);chain=forward_chain(out)
    rows=[];retained=0
    for scene in SCENES:
        current=read(out/'forward'/scene.replace('/','__')/'records.json')
        forward_population(current,S.scene_entries(lock,scene),lock['roles'])
        by_path={}
        for row in current:
            if 'sample_path' in row:by_path.setdefault(row['sample_path'],[]).append(row)
        for path,records in by_path.items():
            with np.load(out/path,allow_pickle=False) as archive:
                if set(archive.files)!={r['sample_field'] for r in records}:raise ValueError('wrong saved array population')
                for row in records:
                    p=archive[row['sample_field']];R.validate_probability(p,tuple(row['shape']))
                    if array_sha(p)!=row['array_sha256']:raise ValueError('saved probability hash mismatch')
                    retained+=p.nbytes
        rows.extend(current)
    forward_population(rows,lock['entries'],lock['roles'])
    if retained!=lock['retained_probability_bytes']:raise ValueError('incomplete retained raw evidence')
    write(out/'forward_complete.json',dict(lock_sha256=sha(out/'lock.json'),forward_chain=chain,
        sealed_unix=time.time(),record_count=len(rows),retained_probability_bytes=retained,labels_decoded=False))


def verify_forward_complete(out):
    value=read(out/'forward_complete.json')
    if value['lock_sha256']!=sha(out/'lock.json') or value['forward_chain']!=forward_chain(out) or value['labels_decoded']:
        raise ValueError('validation full-forward seal mismatch')
    return value


def load_labels(entry):
    target=Path(entry['target_path']);path=Path(entry['path'])
    if sha(target)!=entry['target_sha256'] or sha(path/'labels.npz')!=entry['old_label_sha256']:
        raise ValueError('bound old validation labels changed')
    A.verify_files(entry['target_provenance']['sensor_sha256'])
    with np.load(target,allow_pickle=False) as archive:
        labels={k:archive[k] for k in LABEL_FIELDS}
        with np.load(path/'labels.npz',allow_pickle=False) as old:
            if set(archive.files)!=set(old.files)|{'g24_point_xy','g24_point_visible','g24_point_status'}:
                raise ValueError('old extended label field population changed')
            for k in old.files:np.testing.assert_array_equal(archive[k],old[k])
    return labels


def measure(probability,labels,parts,condition,direct,common):
    metric_condition=dict(condition,scenario='scale' if condition['scenario']=='scale' else 'partition')
    row=R.measure(probability,parts,metric_condition,direct,common)
    q,k=condition['query'],condition['key']
    visible=labels[q+'_point_visible'];target=labels[q+'_point_xy']
    predicted=parts[k]['mean'][probability.argmax(-1)]*8
    correct=(np.linalg.norm(predicted-target,axis=-1)<=8)&visible
    return dict(**row,pck_count=int(correct.sum()),n_point=int(visible.sum()))


@torch.no_grad()
def evaluate(out,scene):
    lock=verify_lock(out);seal=verify_forward_complete(out)
    entries=S.scene_entries(lock,scene);directory=out/'evaluation'/scene.replace('/','__')
    if directory.exists():raise ValueError('refusing existing validation evaluation')
    directory.mkdir(parents=True);started=time.time()
    if started<seal['sealed_unix']:raise ValueError('invalid label chronology')
    write(directory/'started.json',dict(created_unix=started,forward_complete_sha256=sha(out/'forward_complete.json')))
    parts=S.read_parts(out);partition=S.M.partition_tensors(parts,'cpu')
    committed=read(out/'forward'/scene.replace('/','__')/'records.json')
    forward_population(committed,entries,lock['roles']);records={identity(r):r for r in committed}
    primary=lock['roles']['aliases']['primary'];operators=lock['roles']['operators']
    controls={o:c for o,c in operators.items() if o!=primary}
    rows=[]
    for index,entry in enumerate(entries):
        inputs=load_inputs(Path(entry['path'])/'inputs.npz');labels=None
        for state in STATES:
            queries,keys,diagnostic=prepare_state(inputs,state,partition,controls)
            for c in (c for c in CONDITIONS if c['state']==state):
                direct=common=None
                for op,cfg in operators.items():
                    record=records[(scene,tuple(entry['pair']),op,c['scenario'],c['condition'])]
                    if op==primary:
                        with np.load(out/record['sample_path'],allow_pickle=False) as archive:probability=archive[op]
                    else:
                        probability,numeric=predict_prepared(queries[c['query']],keys[c['key']],cfg)
                        if numeric!=record['numerical']:raise ValueError('validation numerical replay mismatch')
                    digest=R.require_replay(probability,record)
                    if labels is None:labels=load_labels(entry)
                    if direct is None:direct,common=S.target_counts(labels,parts,c['query'],c['key'])
                    rows.append(dict(**record,**measure(probability,labels,parts,c,direct,common),
                        target_sha256=entry['target_sha256'],replayed_array_sha256=digest,
                        probability_source='retained_primary' if op==primary else 'exact_control_replay'))
        print('VALIDATION_EVALUATE',scene,index+1,'/16',flush=True)
    forward_population(rows,entries,lock['roles'])
    # Full cross-scene population checks remain mandatory in summarize/audit.
    for row in rows:
        if (row['replayed_array_sha256']!=row['array_sha256']
                or not 0<=row['pck_count']<=row['n_point']<=row['n_query']
                or any(not np.isfinite(row[name]) for name in V.METRIC_FIELDS)):
            raise ValueError('invalid validation metric shard')
    write(directory/'rows.json',rows)
    A.complete_stage(directory,out,'evaluation',scene,started,record_count=len(rows),runtime=runtime(),
        forward_complete_sha256=sha(out/'forward_complete.json'),all_replay_hashes_exact=True)


@torch.no_grad()
def benchmark(out):
    lock=verify_lock(out);verify_forward_complete(out)
    evaluations={}
    for scene in SCENES:
        stage=out/'evaluation'/scene.replace('/','__')
        completed=A.verify_stage(stage,out,'evaluation',scene)
        if (completed['forward_complete_sha256']!=sha(out/'forward_complete.json')
                or completed['all_replay_hashes_exact'] is not True or completed['runtime']!=runtime()):
            raise ValueError('both complete exact evaluation stages required before timing')
        evaluations[scene]=dict(completion_sha256=sha(stage/'completion.json'),finished_unix=completed['finished_unix'])
    directory=out/'timing'
    if directory.exists():raise ValueError('refusing existing timing evidence')
    directory.mkdir();started=time.time()
    if max(v['finished_unix'] for v in evaluations.values())>started:
        raise ValueError('timing started before evaluations completed')
    parts=S.read_parts(out);partition=S.M.partition_tensors(parts,'cpu')
    operators=lock['roles']['operators'];opids=list(operators)
    rows=[];shared=[];execution_order=[]
    for scene_index,entry in enumerate(lock['timing_entries']):
        export_start=time.perf_counter();inputs=load_inputs(Path(entry['path'])/'inputs.npz')
        projection_start=time.perf_counter();raw,diag=S.M.perturbed_observations(inputs,'zero')
        projection_seconds=time.perf_counter()-projection_start
        obs=S.A.observation_tensors(raw,'cpu');export_seconds=time.perf_counter()-export_start
        shared.append(dict(scene=entry['scene'],pair=entry['pair'],projection_seconds=projection_seconds,
            export_seconds=export_seconds,input_sha256=entry['input_sha256']))
        for condition_index,(q,k) in enumerate(S.M.ORIGINAL_CONDITIONS):
            cell={o:dict(scene=entry['scene'],pair=entry['pair'],condition=q+'_'+k,query=q,key=k,operator=o,
                         config=operators[o],input_sha256=entry['input_sha256'],
                         samples_seconds=[],warmup_seconds=[],load_average=[],prepared_tensor_bytes=[]) for o in opids}
            for repeat in range(18):
                offset=(scene_index*9+condition_index+repeat)%len(opids)
                for op in opids[offset:]+opids[:offset]:
                    cfg=operators[op];tick=time.perf_counter()
                    query=P.prepare_query(obs,partition[q],cfg['family']);key=P.prepare_key(obs,partition[k],cfg['family'])
                    probability,_=P.predict(query,key,cfg['family'],cfg['h'],cfg['temperature'])
                    native=probability.cpu().numpy();common=S.common_probability(native,parts[q],parts[k])
                    elapsed=time.perf_counter()-tick
                    execution_order.append(dict(scene=entry['scene'],condition=q+'_'+k,repeat=repeat,operator=op))
                    if not np.isfinite(common).all() or abs(common.sum(-1)-1).max()>1.e-12:
                        raise ValueError('invalid timed complete common probability')
                    field='warmup_seconds' if repeat<3 else 'samples_seconds'
                    cell[op][field].append(elapsed)
                    cell[op]['load_average'].append(list(os.getloadavg()))
                    cell[op]['prepared_tensor_bytes'].append(sum(v.numel()*v.element_size()
                        for obj in (query,key) for v in obj.values() if torch.is_tensor(v)))
            rows.extend(cell.values())
            print('VALIDATION_TIMING',entry['scene'],q+'_'+k,flush=True)
    summary=V.aggregate_timing(rows,lock['timing_entries'],lock['roles'])
    write(directory/'rows.json',rows);write(directory/'summary.json',summary)
    write(directory/'execution_order.json',execution_order)
    write(directory/'shared.json',dict(rows=shared,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        memory_scope='whole timing process high-water RSS; includes shared tensors and framework; not per-role allocator peak',
        excludes_rgb_depth_predictor=True,primary_time_includes_role_preparation=True))
    A.complete_stage(directory,out,'timing','all',started,runtime=runtime(),cells=len(rows),evaluation_chain=evaluations)


def summarize(out):
    if (out/'summary.json').exists() or (out/'completion.json').exists():
        raise ValueError('refusing validation summary/completion overwrite')
    lock=verify_lock(out);verify_forward_complete(out)
    rows=[]
    for scene in SCENES:
        directory=out/'evaluation'/scene.replace('/','__')
        value=A.verify_stage(directory,out,'evaluation',scene)
        if value['forward_complete_sha256']!=sha(out/'forward_complete.json') or value['all_replay_hashes_exact'] is not True:
            raise ValueError('invalid evaluation provenance')
        rows.extend(read(directory/'rows.json'))
    A.verify_stage(out/'timing',out,'timing','all')
    metrics=V.summarize(rows,lock['entries'],lock['roles'])
    timing=V.aggregate_timing(read(out/'timing/rows.json'),lock['timing_entries'],lock['roles'])
    if timing!=read(out/'timing/summary.json'):raise ValueError('timing summary changed')
    gates=V.gates(metrics,lock['roles'],timing)
    write(out/'summary.json',dict(metrics=metrics,timing=timing,gates=gates,roles=lock['roles'],
        lock_sha256=sha(out/'lock.json'),forward_complete_sha256=sha(out/'forward_complete.json'),
        created_unix=time.time(),pair_count=32,row_count=len(rows),independent_audit_pending=True,
        scientific_gate_passed=False,actual_value_run=False,optimized_gs_run=False,fresh_test_read=False))
    files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}
    write(out/'completion.json',dict(lock_sha256=sha(out/'lock.json'),files=files,
        summary_sha256=sha(out/'summary.json'),finished_unix=time.time(),independent_audit_pending=True))
    print('VALIDATION_PRODUCER_COMPLETE_AUDIT_PENDING',gates,flush=True)


def accept_audit(out,audit_path):
    verify_lock(out)
    if (out/'decision.json').exists():raise ValueError('refusing accepted validation decision overwrite')
    complete=read(out/'completion.json');summary=read(out/'summary.json');audit=read(audit_path)
    A.verify_files({str(out/p):h for p,h in complete['files'].items()})
    directory=audit_path.parent;design=read(directory/'lock.json');audited=read(directory/'completion.json')
    if (sha(directory/'lock.json')!=(directory/'lock.sha256').read_text().strip()
            or design['producer_lock_sha256']!=sha(out/'lock.json')
            or audit['independent_audit_lock_sha256']!=sha(directory/'lock.json')
            or audit['passed'] is not True or audited['passed'] is not True):
        raise ValueError('independent validation audit did not pass')
    A.verify_files({str(ROOT/p):h for p,h in design['source_hashes'].items()})
    A.verify_files({str(directory/'source_snapshot'/p):h for p,h in design['source_hashes'].items()})
    A.verify_files({str(directory/p):h for p,h in audited['files'].items()})
    if audited['files'].get(audit_path.name)!=sha(audit_path):raise ValueError('unbound audit result')
    expected={f'validation_{name}_sha256':sha(out/(name+'.json')) for name in ('lock','forward_complete','summary','completion')}
    gate=summary['gates']['passed']
    if (any(audit.get(k)!=v for k,v in expected.items()) or audit['roles']!=summary['roles']
            or audit['gates']!=summary['gates'] or audit['metric_gates_passed']!=gate
            or audit['scientific_gate_passed']!=gate or audited['scientific_gate_passed']!=gate):
        raise ValueError('independent validation scientific decision mismatch')
    decision=dict(**expected,audit_path=str(audit_path.resolve()),audit_sha256=sha(audit_path),
        independent_audit_passed=True,scientific_gate_passed=gate,roles=summary['roles'],gates=summary['gates'],
        created_unix=time.time(),general_research_goal_achieved=False,
        next_action='review_remaining_actual_value_and_gs_designs' if gate else 'stop_unchanged_candidate_preserve_failure',
        actual_value_run=False,optimized_gs_run=False,fresh_confirmation_authorized=False)
    write(out/'decision.json',decision)
    (out/'decision.sha256').write_text(sha(out/'decision.json')+'\n')
    return decision


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=('freeze','forward','seal-forward','evaluate','benchmark','summarize','accept-audit'))
    parser.add_argument('--out',type=Path,default=ROOT/'results/round15_validation')
    parser.add_argument('--regional',type=Path,default=ROOT/'results/round15_regional')
    parser.add_argument('--scene',choices=SCENES)
    parser.add_argument('--audit',type=Path)
    args=parser.parse_args()
    if args.stage=='freeze':freeze(args.out,args.regional)
    elif args.stage in ('forward','evaluate'):globals()[args.stage](args.out,args.scene)
    elif args.stage=='seal-forward':seal_forward(args.out)
    elif args.stage=='accept-audit':
        if args.audit is None:parser.error('accept-audit requires --audit')
        accept_audit(args.out,args.audit)
    else:globals()[args.stage](args.out)


if __name__=='__main__':
    main()
