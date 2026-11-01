"""R21 fixed matched-content diagnostic: freeze, full forward, seal, evaluate.

No stage selects a model or opens validation/fresh data. Actual target bytes
are inaccessible until the complete independent forward audit has passed.
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path
import shutil
import time
import os

import numpy as np  # NumPy first in the accepted MKL/OpenMP runtime.
import torch
from threadpoolctl import threadpool_limits

from experiments import round20_anchor_fixed as R20
from experiments import round19_center_semantics as HISTORY
from experiments import round21_matched_decomposition_metrics as METRICS
from experiments import round21_matched_decomposition_audit_numeric as NUMREF
from models import round21_matched_decomposition as MODEL

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_OUT=ROOT/'results/round21_matched_decomposition'
DEFAULT_AUDIT=ROOT/'results/round21_matched_decomposition_independent_audit'
R20_ROOT=ROOT/'results/round20_anchor_fixed'
R20_ACCEPTANCE=ROOT/'results/round20_anchor_root_acceptance_20261031.json'
SCENES,CONDITIONS,COUNTS=METRICS.SCENES,METRICS.CONDITIONS,METRICS.COUNTS
QUERIES=tuple(COUNTS)
VALID_COUNTS=dict(g8=9521,g16=2463,g24=1096)
FAMILIES,GEOMETRIES,ROLE_CONFIGS=MODEL.FAMILIES,MODEL.GEOMETRIES,MODEL.ROLE_CONFIGS
CONTRASTS=METRICS.CONTRASTS
FIELDS,OBS_FIELDS,ANCHOR_FIELDS=R20.FIELDS,R20.OBS_FIELDS,R20.ANCHOR_FIELDS
THREAD_ENV=R20.THREAD_ENV
RAW_PROBABILITY_BYTES=780337152
RAW_MASS_BYTES=390168576
TOTAL_MATRICES=1152
NUMERICAL_POLICY=dict(gaussian_order=32,reference_gaussian_orders=[64,96],reference_query_rows=8,
    reference_pairs='first metadata pair in each scene',reference_matrices=72,reference_probability_rows=576,
    reference_unique_query_rows=72,probability_error=1.e-4,reference_convergence=1.e-8,
    same_raw_mass_arithmetic=1.e-12,normalization_error=1.e-12,anchor_error_px=1.e-9,
    moment_normalized_error=1.e-4,content_error=1.e-5,geometry_dtype='float64',anchor_dtype='float64',
    new_covariance_dtype='float64',old_moment_dtype='float32',content_bias_dtype='float32',floor=1.e-30)
SOURCE_ROOTS=('ROUND21_MATCHED_DECOMPOSITION_PROTOCOL.md','docs/round21_matched_decomposition_design_review.md',
    'experiments/round13_mixture_mechanism.py',
    'models/round21_matched_decomposition.py','experiments/round21_matched_decomposition.py',
    'experiments/round21_matched_decomposition_metrics.py','tests/test_round21_matched_decomposition.py',
    'experiments/round21_matched_decomposition_audit.py','experiments/round21_matched_decomposition_audit_metrics.py',
    'experiments/round21_matched_decomposition_audit_numeric.py','tests/test_round21_matched_decomposition_audit.py',
    'tests/test_round21_matched_decomposition_audit_metrics.py','tests/test_round21_matched_decomposition_audit_numeric.py')
sha,read,write,array_sha=R20.sha,R20.read,R20.write,R20.array_sha
legacy_array_sha=HISTORY.legacy_array_sha
inventory,verify_inventory,runtime=R20.inventory,R20.verify_inventory,R20.runtime
require_audit_freeze,require_forward_audit,failed_stage=R20.require_audit_freeze,R20.require_forward_audit,R20.failed_stage


def source_names():
    found=set();pending=list(SOURCE_ROOTS)
    while pending:
        name=pending.pop()
        if name in found:continue
        path=ROOT/name
        if not path.is_file():raise ValueError('joint source missing: '+name)
        found.add(name)
        if path.suffix!='.py':continue
        for node in ast.walk(ast.parse(path.read_text())):
            names=[]
            if isinstance(node,ast.Import):names=[n.name for n in node.names]
            elif isinstance(node,ast.ImportFrom):
                if node.level:raise ValueError('relative import needs explicit closure review')
                if node.module:names=[node.module]+[node.module+'.'+n.name for n in node.names]
            for module in names:
                parts=module.split('.');paths=[Path(*parts).with_suffix('.py'),Path(*parts)/'__init__.py']
                paths += [Path(*parts[:i])/'__init__.py' for i in range(1,len(parts))]
                pending.extend(str(p) for p in paths if (ROOT/p).is_file() and str(p) not in found)
    return sorted(found)


def inherited_contract():
    accepted=read(R20_ACCEPTANCE)
    if not all(accepted.get(k) is True for k in ('accepted_for_development_interpretation',
            'full_forward_and_final_independent_audits_passed','strict_forward_before_label_chronology_checked')):
        raise ValueError('complete accepted R20 development evidence required')
    oldlock=read(R20_ROOT/'lock.json');entries=[dict(e) for e in oldlock['entries']]
    denied={e[n] for e in entries for n in ('target_path','label_path')}
    bindings={}
    def bind(path,expected=None):
        path=Path(path);path=path if path.is_absolute() else ROOT/path;path=path.resolve()
        if str(path) in denied or path.name=='labels.npz':raise ValueError('actual targets forbidden before global audit')
        digest=sha(path)
        if expected is not None and digest!=expected:raise ValueError('accepted inherited file changed: '+str(path))
        if str(path) in bindings and bindings[str(path)]!=digest:raise ValueError('conflicting inherited binding')
        bindings[str(path)]=digest
        return digest
    bind(R20_ACCEPTANCE)
    for path,digest in accepted['files'].items():bind(path,digest)
    for path,digest in oldlock['bindings'].items():bind(path,digest)
    for directory in (R20_ROOT,ROOT/'results/round20_anchor_independent_audit'):
        old=read(directory/'lock.json')
        for name,digest in old['source_hashes'].items():
            bind(ROOT/name,digest);bind(directory/'source_snapshot'/name,digest)
    if oldlock['source_function']!=HISTORY.function_contract():raise ValueError('historical a law changed')
    bind(R20_ROOT/'inherited_rows.json',oldlock['inherited_rows_sha256'])
    inherited=read(R20_ROOT/'inherited_rows.json')
    r20rows=read(R20_ROOT/'evaluation/rows.json')
    replay={};conditions={c['condition'] for c in CONDITIONS}
    oldcomplete=read(ROOT/'results/round15_regional/completion.json')
    bind(ROOT/'results/round15_regional/completion.json')
    for role,source in (('mu_c_zero','regional_gaussian_full1'),('full4_zero','regional_gaussian_full4')):
        selected=[dict(r) for r in inherited[source]['regional'] if r['condition'] in conditions]
        for row in selected:
            if row['config']!=dict(h=.25,temperature=0.) or row['replayed_array_sha256']!=row['array_sha256']:
                raise ValueError('exact accepted old T0 commitment required')
            path=ROOT/'results/round15_regional'/row['sample_path'] if 'sample_path' in row else None
            digest=bind(path,oldcomplete['files'][row['sample_path']]) if path else None
            row.update(replay_kind='r15_commitment',source_archive_path=str(path) if path else None,
                       source_archive_sha256=digest,source_field=row.get('sample_field'))
        replay[role]=selected
    for role,source in (('a_c_content','mean_only_matched_tau'),('a_ca_content','anchor_trace')):
        selected=[dict(r) for r in r20rows if r['family']==source and r['condition'] in conditions]
        for row in selected:
            path=R20_ROOT/row['archive_path'];bind(path,row['archive_sha256'])
            row.update(replay_kind='r20_saved',source_archive_path=str(path),
                       source_archive_sha256=row['archive_sha256'],source_field=source)
        replay[role]=selected
    expected={(e['scene'],tuple(e['pair']),c['condition']) for e in entries for c in CONDITIONS}
    for role,rows in replay.items():
        if len(rows)!=144 or {(r['scene'],tuple(r['pair']),r['condition']) for r in rows}!=expected:
            raise ValueError('incomplete inherited replay population')
        for row in rows:
            entry=next(e for e in entries if e['scene']==row['scene'] and e['pair']==row['pair'])
            if row['input_sha256']!=entry['input_sha256'] or row['label_sha256']!=entry['label_sha256']:
                raise ValueError('inherited replay source/label identity differs')
    if sum(r['source_archive_path'] is not None for f in ('mu_c_zero','full4_zero') for r in replay[f])!=18:
        raise ValueError('exact eighteen actually retained T0 probability matrices required')
    bind(oldlock['partitions_path'],oldlock['partitions_sha256'])
    return bindings,entries,replay,oldlock['partitions_path'],oldlock['partitions_sha256']


def freeze(out,audit_root=DEFAULT_AUDIT):
    if out.exists():raise ValueError('refusing existing R21 directory')
    bindings,entries,replay,parts,digest=inherited_contract()
    source_hashes={n:sha(ROOT/n) for n in source_names()}
    if shutil.disk_usage(out.parent).free<RAW_PROBABILITY_BYTES+RAW_MASS_BYTES+(512<<20):
        raise ValueError('insufficient complete forward storage')
    lock=dict(version=1,created_unix=time.time(),source_hashes=source_hashes,source_function=HISTORY.function_contract(),
        runtime=runtime(),bindings=bindings,entries=entries,families=list(FAMILIES),geometries=list(GEOMETRIES),
        role_configs=ROLE_CONFIGS,contrasts=list(CONTRASTS),conditions=list(CONDITIONS),counts=COUNTS,
        expected_valid_counts=VALID_COUNTS,field_allowlist=list(FIELDS),partitions_path=parts,partitions_sha256=digest,
        audit_root=str(audit_root.resolve()),numerical_policy=NUMERICAL_POLICY,numerical_references=NUMREF.REFERENCE_POLICY,
        pairs=48,unique_query_rows=28224,matrices=TOTAL_MATRICES,raw_matrices=576,
        raw_probability_bytes=RAW_PROBABILITY_BYTES,raw_mass_bytes=RAW_MASS_BYTES,
        actual_target_hashes_read=False,labels_decoded=False,scientific_gate_passed=False,diagnostic_only=True,no_candidate_selection=True)
    out.mkdir(parents=True)
    for name,digest in source_hashes.items():
        path=out/'source_snapshot'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes((ROOT/name).read_bytes())
        if sha(path)!=digest:raise ValueError('source changed during freeze')
    write(out/'inherited_replay_rows.json',replay);lock['inherited_replay_rows_sha256']=sha(out/'inherited_replay_rows.json')
    write(out/'lock.json',lock);(out/'lock.sha256').write_text(sha(out/'lock.json')+'\n')


def verify_lock(out):
    lock=read(out/'lock.json')
    expected=dict(runtime=runtime(),source_function=HISTORY.function_contract(),families=list(FAMILIES),geometries=list(GEOMETRIES),
        role_configs=ROLE_CONFIGS,contrasts=list(CONTRASTS),conditions=list(CONDITIONS),counts=COUNTS,
        expected_valid_counts=VALID_COUNTS,field_allowlist=list(FIELDS),numerical_policy=NUMERICAL_POLICY,
        numerical_references=NUMREF.REFERENCE_POLICY,matrices=TOTAL_MATRICES,raw_matrices=576,
        raw_probability_bytes=RAW_PROBABILITY_BYTES,raw_mass_bytes=RAW_MASS_BYTES,pairs=48,unique_query_rows=28224,
        scientific_gate_passed=False,diagnostic_only=True,no_candidate_selection=True,
        actual_target_hashes_read=False,labels_decoded=False)
    if (sha(out/'lock.json')!=(out/'lock.sha256').read_text().strip()
            or any(lock[k]!=v for k,v in expected.items()) or set(lock['source_hashes'])!=set(source_names())):
        raise ValueError('immutable R21 contract/runtime changed')
    for name,digest in lock['source_hashes'].items():
        if sha(ROOT/name)!=digest or sha(out/'source_snapshot'/name)!=digest:raise ValueError('joint source changed')
    denied={e[n] for e in lock['entries'] for n in ('target_path','label_path')}
    for path,digest in lock['bindings'].items():
        if path in denied or Path(path).name=='labels.npz':raise ValueError('target entered pre-seal bindings')
        if sha(path)!=digest:raise ValueError('inherited evidence changed: '+path)
    if sha(out/'inherited_replay_rows.json')!=lock['inherited_replay_rows_sha256']:raise ValueError('old replay rows changed')
    return lock


def load_parts(lock):
    if sha(lock['partitions_path'])!=lock['partitions_sha256']:raise ValueError('old partitions changed')
    with np.load(lock['partitions_path'],allow_pickle=False) as archive:
        parts={q:{n.split('__')[1]:archive[n] for n in archive.files if n.startswith(q+'__')} for q in QUERIES}
    integer={'bounds','owner','atoms','parent','cell_owner'}
    tensors={q:{n:torch.as_tensor(v,dtype=torch.int64 if n in integer else torch.float64 if n=='cell_bounds' else torch.float32)
                for n,v in part.items()} for q,part in parts.items()}
    if sum(len(parts[q]['mass'])*len(parts['g8']['mass']) for q in QUERIES)*48*8*8!=RAW_PROBABILITY_BYTES:
        raise ValueError('complete native matrix population changed')
    return parts,tensors


@torch.no_grad()
def prepare_pair(entry,parts,tensors,predictor=None):
    inputs,obs=R20.load_inputs(entry);predictor=predictor or HISTORY.extracted_predictor();banks={};retained={}
    with np.load(entry['support_path'],allow_pickle=False) as old,np.load(entry['r19_archive_path'],allow_pickle=False) as r19:
        for q in QUERIES:
            bounds=tensors[q]['bounds'].numpy()
            if not np.array_equal(bounds,old[q+'__bounds']):raise ValueError('native support bounds changed')
            a,front,fov=predictor({n:inputs[n] for n in ANCHOR_FIELDS},bounds)
            for n,value in (('a_px',a),('a_front',front),('a_fov',fov)):
                if array_sha(value)!=array_sha(r19[q+'__'+n]):raise ValueError('original R19 center law changed')
            prepared=MODEL.prepare(obs,tensors[q],tensors['g8'],torch.from_numpy(a/8.));banks[q]=prepared
            if array_sha(prepared['mu32'].numpy())!=array_sha(old[q+'__mean']) or array_sha(prepared['cov32'].numpy())!=array_sha(old[q+'__cov']):
                raise ValueError('exact original full1 support changed')
            values={n:v.numpy() for n,v in prepared.items() if isinstance(v,torch.Tensor)}
            values.update(bounds_px=bounds,a_px=a,a_front=front,a_fov=fov)
            retained.update({q+'__'+n:v for n,v in values.items()})
        key=banks['g8']['key']
        retained.update({'key__'+n:key[n].numpy() for n in ('mass','cell_bounds','cell_owner','features')})
        retained['key__bounds_px']=tensors['g8']['bounds'].numpy()
    return banks,retained


def expected_identities(entries):
    return {(e['scene'],tuple(e['pair']),c['condition'],f) for e in entries for c in CONDITIONS for f in FAMILIES}


def validate_records(records,entries):
    ids={(r['scene'],tuple(r['pair']),r['condition'],r['family']) for r in records}
    if len(ids)!=len(records) or ids!=expected_identities(entries):raise ValueError('incomplete fixed diagnostic population')
    mapping={(e['scene'],tuple(e['pair'])):e for e in entries};conditions={c['condition']:c for c in CONDITIONS}
    for row in records:
        e=mapping[(row['scene'],tuple(row['pair']))];c=conditions[row['condition']]
        if (row['config']!=ROLE_CONFIGS[row['family']] or any(row[n]!=c[n] for n in ('scenario','query','key'))
                or row['input_sha256']!=e['input_sha256'] or row['shape']!=[COUNTS[c['query']],432]
                or row['dtype']!='float64' or row['pair_name']!=e['pair_name']):raise ValueError('diagnostic row identity changed')


def replay_index(out):
    return {(r['scene'],tuple(r['pair']),r['condition'],role):r
            for role,rows in read(out/'inherited_replay_rows.json').items() for r in rows}


def check_inherited_probability(probability,source):
    """All old commitments; direct array equality only when an archive exists."""
    digest=(legacy_array_sha if source['replay_kind']=='r15_commitment' else array_sha)(probability)
    if digest!=source['array_sha256']:raise ValueError('historical complete probability commitment differs')
    saved=source['source_archive_path'] is not None
    if saved:
        if sha(source['source_archive_path'])!=source['source_archive_sha256']:raise ValueError('saved historical archive changed')
        with np.load(source['source_archive_path'],allow_pickle=False) as archive:
            if not np.array_equal(probability,archive[source['source_field']]):raise ValueError('saved historical probability differs')
    return dict(kind=source['replay_kind'],historical_array_sha256=digest,commitment_exact=True,saved_array_checked=saved)


@torch.no_grad()
def forward(out,scene):
    if scene not in SCENES:raise ValueError('only frozen TRAIN scenes allowed')
    directory=out/'forward'/scene.replace('/','__')
    if directory.exists():raise ValueError('preserve existing forward attempt')
    directory.mkdir(parents=True);started=time.time()
    write(directory/'started.json',dict(created_unix=started,producer_lock_sha256=sha(out/'lock.json'),scene=scene))
    try:
        lock=verify_lock(out);require_audit_freeze(out,lock);parts,tensors=load_parts(lock);old=replay_index(out)
        if shutil.disk_usage(out).free<(RAW_PROBABILITY_BYTES+RAW_MASS_BYTES)//3+(128<<20):raise ValueError('insufficient complete-scene storage')
        entries=[e for e in lock['entries'] if e['scene']==scene]
        if len(entries)!=16:raise ValueError('sixteen complete pairs required')
        records=[];supports=[];predictor=HISTORY.extracted_predictor()
        for index,entry in enumerate(entries):
            banks,retained=prepare_pair(entry,parts,tensors,predictor)
            folder=directory/'arrays'/entry['pair_name'];folder.mkdir(parents=True)
            support_path=folder/'supports.npz';np.savez(support_path,**retained)
            supports.append(dict(scene=scene,pair=entry['pair'],pair_name=entry['pair_name'],input_sha256=entry['input_sha256'],
                archive_path=str(support_path.relative_to(out)),archive_sha256=sha(support_path),
                array_sha256={n:array_sha(v) for n,v in retained.items()},
                mixture_moment_residuals={q:MODEL.moment_diagnostics(banks[q]) for q in QUERIES}))
            for condition in CONDITIONS:
                q=condition['query'];prepared=banks[q];arrays={};pending=[]
                for geometry in GEOMETRIES:
                    raw=MODEL.raw_mass(prepared,geometry);predicted,diagnostics=MODEL.probabilities(prepared,geometry,raw)
                    arrays[geometry+'__raw_mass']=raw.numpy()
                    for mode,p in predicted.items():
                        family=geometry+'_'+mode;p=p.numpy();METRICS.probability(p,(COUNTS[q],432));winner=p.argmax(-1).astype(np.int32)
                        arrays[family]=p;arrays[family+'__argmax']=winner
                        source=old.get((scene,tuple(entry['pair']),condition['condition'],family))
                        replay=check_inherited_probability(p,source) if source else None
                        pending.append(dict(scene=scene,pair=entry['pair'],pair_name=entry['pair_name'],**condition,
                            family=family,config=ROLE_CONFIGS[family],shape=list(p.shape),dtype='float64',
                            array_sha256=array_sha(p),argmax_sha256=array_sha(winner),raw_mass_sha256=array_sha(raw.numpy()),
                            raw_mass_field=geometry+'__raw_mass',input_sha256=entry['input_sha256'],numerical=diagnostics,inherited_replay=replay))
                archive=folder/(condition['condition']+'.npz');np.savez(archive,**arrays);digest=sha(archive)
                for r in pending:r.update(archive_path=str(archive.relative_to(out)),archive_sha256=digest)
                records.extend(pending)
            print('R21_FORWARD',scene,index+1,'/16',flush=True)
        validate_records(records,entries);write(directory/'records.json',records);write(directory/'support_records.json',supports)
        write(directory/'completion.json',dict(passed=True,files=inventory(directory),records=len(records),scene=scene,
            actual_target_hashes_read=False,labels_decoded=False,producer_lock_sha256=sha(out/'lock.json'),
            started_unix=started,finished_unix=time.time()))
    except Exception as error:failed_stage(directory,started,error);raise


def seal(out):
    directory=out/'seal'
    if directory.exists() or (out/'forward_complete.json').exists():raise ValueError('preserve existing seal')
    directory.mkdir();started=time.time();write(directory/'started.json',dict(created_unix=started))
    try:
        lock=verify_lock(out);require_audit_freeze(out,lock);records=[];supports=[]
        for scene in SCENES:
            folder=out/'forward'/scene.replace('/','__');c=verify_inventory(folder)
            if c['labels_decoded'] or c['actual_target_hashes_read'] or c['producer_lock_sha256']!=sha(out/'lock.json'):
                raise ValueError('invalid pre-label forward completion')
            records.extend(read(folder/'records.json'));supports.extend(read(folder/'support_records.json'))
        validate_records(records,lock['entries'])
        if len(supports)!=48 or {(r['scene'],tuple(r['pair'])) for r in supports}!={(e['scene'],tuple(e['pair'])) for e in lock['entries']}:
            raise ValueError('complete support population required')
        grouped={}
        for r in records:grouped.setdefault(r['archive_path'],[]).append(r)
        expected={n for f in FAMILIES for n in (f,f+'__argmax')}|{g+'__raw_mass' for g in GEOMETRIES}
        for path,group in grouped.items():
            digest=sha(out/path)
            with np.load(out/path,allow_pickle=False) as a:
                if set(a.files)!=expected:raise ValueError('saved probability/raw schema changed')
                for r in group:
                    p,w,raw=a[r['family']],a[r['family']+'__argmax'],a[r['raw_mass_field']]
                    METRICS.probability(p,r['shape'])
                    if (digest!=r['archive_sha256'] or array_sha(p)!=r['array_sha256'] or w.dtype!=np.int32
                            or array_sha(w)!=r['argmax_sha256'] or not np.array_equal(w,p.argmax(-1))
                            or raw.dtype!=np.float64 or raw.shape!=p.shape or not np.isfinite(raw).all() or (raw<0).any()
                            or array_sha(raw)!=r['raw_mass_sha256']):raise ValueError('saved arrays changed')
        for r in supports:
            if sha(out/r['archive_path'])!=r['archive_sha256']:raise ValueError('saved support archive changed')
            with np.load(out/r['archive_path'],allow_pickle=False) as a:
                if {n:array_sha(a[n]) for n in a.files}!=r['array_sha256']:raise ValueError('saved support array changed')
        write(out/'forward/records.json',records);write(out/'forward/support_records.json',supports)
        write(out/'forward/completion.json',dict(passed=True,files=inventory(out/'forward'),matrices=len(records),
            all_raw_masses_retained=True,
            actual_target_hashes_read=False,labels_decoded=False,finished_unix=time.time()))
        write(out/'forward_complete.json',dict(producer_lock_sha256=sha(out/'lock.json'),
            forward_completion_sha256=sha(out/'forward/completion.json'),forward_records_sha256=sha(out/'forward/records.json'),
            support_records_sha256=sha(out/'forward/support_records.json'),pairs=48,matrices=TOTAL_MATRICES,raw_matrices=576,
            raw_probability_bytes=RAW_PROBABILITY_BYTES,raw_mass_bytes=RAW_MASS_BYTES,all_probabilities_retained=True,
            all_raw_masses_retained=True,
            actual_target_hashes_read=False,labels_decoded=False,created_unix=time.time()))
        write(directory/'completion.json',dict(passed=True,files=inventory(directory),all_raw_masses_retained=True,
            forward_complete_sha256=sha(out/'forward_complete.json'),finished_unix=time.time()))
    except Exception as error:failed_stage(directory,started,error);raise


def check_old_metrics(row,source):
    deltas={k:row[k]-source[k] for k in ('kl_sum','js_sum','direct_kl_sum','direct_js_sum')}
    if any(abs(x)>1.e-8 for x in deltas.values()) or any(row[k]!=source[k] for k in ('mass','n_query','n_key')):
        raise ValueError('accepted old regional metric reproduction failed')
    if source['replay_kind']=='r20_saved':
        before,after=source['center'],row['center']
        if any(before[k]!=after[k] for k in ('pck_count','n_point','n_query')) or any(
                abs(before[k]-after[k])>1.e-8 for k in ('distance_sum','distance_squared_sum','mean','rmse')):
            raise ValueError('accepted R20 center metric reproduction failed')
    return dict(regional_sum_differences=deltas,center_checked=source['replay_kind']=='r20_saved')


def evaluate(out):
    directory=out/'evaluation'
    if directory.exists() or (out/'summary.json').exists() or (out/'completion.json').exists():raise ValueError('preserve existing evaluation attempt')
    directory.mkdir();started=time.time();write(directory/'started.json',dict(created_unix=started))
    try:
        lock=verify_lock(out);audit_root=require_forward_audit(out,lock);parts,_=load_parts(lock)
        records=read(out/'forward/records.json');validate_records(records,lock['entries']);old=replay_index(out)
        grouped={}
        for r in records:grouped.setdefault((r['scene'],tuple(r['pair']),r['condition']),[]).append(r)
        rows=[];paired=[];label_sources={};replay_checks=[]
        for entry in lock['entries']:
            # First target byte access in this protocol, after full independent numerical acceptance.
            for path,expected in ((entry['label_path'],entry['label_sha256']),(entry['target_path'],entry['target_sha256'])):
                if sha(path)!=expected:raise ValueError('exact inherited target archive changed')
                label_sources[path]=expected
            with np.load(entry['label_path'],allow_pickle=False) as a:labels={n:a[n] for n in ('gt_xy','gt_visible')}
            with np.load(entry['target_path'],allow_pickle=False) as a:
                targets={n:a[n] for q in QUERIES for n in (q+'_point_xy',q+'_point_visible')}
            for condition in CONDITIONS:
                q=condition['query'];key=(entry['scene'],tuple(entry['pair']),condition['condition']);chosen=grouped[key]
                direct,common=METRICS.target_counts(labels,parts,q,'g8');bank={}
                with np.load(out/chosen[0]['archive_path'],allow_pickle=False) as arrays:
                    for r in chosen:
                        p,w=arrays[r['family']],arrays[r['family']+'__argmax']
                        if array_sha(p)!=r['array_sha256'] or array_sha(w)!=r['argmax_sha256']:raise ValueError('saved probability changed before metrics')
                        center=METRICS.center_metrics(w,parts['g8']['bounds'],targets[q+'_point_xy'],targets[q+'_point_visible'])
                        row=dict(r,**METRICS.region_metrics(p,parts,condition,direct,common),center=center,
                                 label_sha256=entry['label_sha256'],target_sha256=entry['target_sha256'])
                        bank[r['family']]=row;rows.append(row)
                        source=old.get((*key,r['family']))
                        if source:replay_checks.append(dict(scene=entry['scene'],pair=entry['pair'],condition=condition['condition'],family=r['family'],**check_old_metrics(row,source)))
                    for contrast in CONTRASTS:
                        a,b=contrast['from_role'],contrast['to_role']
                        paired.append(METRICS.paired_row(bank[a],bank[b],arrays[a+'__argmax'],arrays[b+'__argmax'],
                            parts['g8']['bounds'],targets[q+'_point_xy'],targets[q+'_point_visible'],contrast))
            print('R21_EVALUATE',entry['scene'],entry['pair_name'],flush=True)
        validate_records(rows,lock['entries'])
        by_role={f:METRICS.summarize([r for r in rows if r['family']==f]) for f in FAMILIES}
        for f in FAMILIES:
            for q in QUERIES:
                if sum(r['center']['n_point'] for r in rows if r['family']==f and r['query']==q)!=VALID_COUNTS[q]:raise ValueError('exact R13 center population changed')
        by_contrast={c['name']:dict(from_role=c['from_role'],to_role=c['to_role'],
            **METRICS.summarize([r for r in paired if r['contrast']==c['name']],paired=True)) for c in CONTRASTS}
        if len(paired)!=1440 or len(replay_checks)!=576:raise ValueError('complete contrast/replay metrics required')
        summary=dict(by_role=by_role,by_contrast=by_contrast,interactions=METRICS.interactions(by_contrast),
            diagnostic_only=True,no_candidate_selection=True,scientific_gate_passed=False,validation_or_fresh_read=False,
            inherited_baselines=True,center_rmse_aggregation='equal-condition/equal-scene means of scene RMSE; not pooled',
            producer_lock_sha256=sha(out/'lock.json'),producer_forward_complete_sha256=sha(out/'forward_complete.json'),
            forward_audit_sha256=sha(audit_root/'forward_verification.json'),independent_audit_pending=True,
            pairs=48,matrices=len(rows),paired_rows=len(paired),started_unix=started,finished_unix=time.time())
        write(directory/'rows.json',rows);write(directory/'paired_rows.json',paired);write(directory/'inherited_metric_checks.json',replay_checks)
        write(out/'summary.json',summary)
        write(directory/'completion.json',dict(passed=True,files=inventory(directory),label_sources=label_sources,
            labels_decoded=True,records=len(rows),paired_rows=len(paired),summary_sha256=sha(out/'summary.json'),finished_unix=time.time()))
        write(out/'completion.json',dict(passed=True,files=inventory(out),producer_lock_sha256=sha(out/'lock.json'),
            scientific_gate_passed=False,diagnostic_only=True,independent_audit_pending=True,finished_unix=time.time()))
    except Exception as error:failed_stage(directory,started,error);raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('freeze','forward','seal','evaluate'))
    parser.add_argument('--out',type=Path,default=DEFAULT_OUT);parser.add_argument('--audit-root',type=Path,default=DEFAULT_AUDIT)
    parser.add_argument('--scene',choices=SCENES);args=parser.parse_args()
    if any(os.environ.get(n)!='4' for n in THREAD_ENV):raise ValueError('launch with OMP/MKL/OPENBLAS_NUM_THREADS=4')
    torch.set_num_threads(4);torch.set_num_interop_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    with threadpool_limits(limits=4):
        out=args.out.resolve()
        if args.stage=='freeze':freeze(out,args.audit_root.resolve())
        elif args.stage=='forward':
            for scene in ([args.scene] if args.scene else SCENES):forward(out,scene)
        elif args.stage=='seal':seal(out)
        else:evaluate(out)


if __name__=='__main__':main()
