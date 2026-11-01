"""Fixed R20 old-TRAIN producer; complete saved forward precedes every label read."""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time
from contextlib import redirect_stdout

import numpy as np  # Required before Torch in this fixed MKL runtime.
import torch
from threadpoolctl import threadpool_info, threadpool_limits

from experiments import round19_center_semantics as HISTORY
from experiments import round20_anchor_fixed_metrics as METRICS
from experiments import round20_anchor_audit_numeric as NUMREF
from models import round20_anchor_pe as MODEL
from models import round17_trace_pe as TRACE

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_OUT=ROOT/'results/round20_anchor_fixed'
DEFAULT_AUDIT=ROOT/'results/round20_anchor_independent_audit'
SCENES=METRICS.SCENES
CONDITIONS=METRICS.CONDITIONS
COUNTS=METRICS.COUNTS
QUERIES=tuple(COUNTS)
CENTER_QUERIES=('g8','g16','g24')
VALID_COUNTS=dict(g8=9521,g16=2463,g24=1096)
FAMILIES=MODEL.FAMILIES
ROLE_CONFIGS=MODEL.ROLE_CONFIGS
OBS_FIELDS=('q_to_k_xy','q_to_k_pixel_cov','q_features','k_features')
ANCHOR_FIELDS=HISTORY.FIELDS
FIELDS=OBS_FIELDS+ANCHOR_FIELDS
THREAD_ENV=HISTORY.THREAD_ENV
RAW_PROBABILITY_BYTES=3074609664
TOTAL_MATRICES=8064
NUMERICAL_POLICY=dict(gaussian_order=32,quartic_order=24,reference_gaussian_orders=[64,96],
    reference_query_rows=8,reference_pairs='first metadata pair in each scene',reference_matrices=504,
    reference_probability_rows=4032,probability_error=1.e-4,reference_convergence=1.e-8,
    same_raw_mass_arithmetic=1.e-12,normalization_error=1.e-12,anchor_error_px=1.e-9,
    moment_normalized_error=1.e-4,content_error=1.e-5,geometry_dtype='float64',anchor_dtype='float64',
    new_covariance_dtype='float64',old_moment_dtype='float32',content_bias_dtype='float32',floor=1.e-30)
SOURCE_ROOTS=('ROUND20_ANCHOR_FIXED_PROTOCOL.md','docs/round20_anchor_preparation_draft.md',
    'docs/round20_anchor_fixed_design_review.md','models/round20_anchor_pe.py',
    'tests/test_round20_anchor_pe.py','experiments/round20_anchor_fixed.py',
    'experiments/round20_anchor_fixed_metrics.py','tests/test_round20_anchor_fixed.py',
    'experiments/round20_anchor_audit.py','experiments/round20_anchor_audit_metrics.py',
    'experiments/round20_anchor_audit_numeric.py','tests/test_round20_anchor_audit.py',
    'tests/test_round20_anchor_audit_metrics.py','tests/test_round20_anchor_audit_numeric.py',
    'experiments/round13_mixture_mechanism.py')
sha,read,write,array_sha=HISTORY.sha,HISTORY.read,HISTORY.write,HISTORY.array_sha
absolute=HISTORY.absolute
inventory,verify_inventory=HISTORY.inventory,HISTORY.verify_inventory


def runtime():
    if any(os.environ.get(n)!='4' for n in THREAD_ENV):raise ValueError('three thread environments must be four')
    if torch.get_num_threads()!=4 or torch.get_num_interop_threads()!=4:
        raise ValueError('fixed Torch intra/inter-op four-thread runtime required')
    if not torch.are_deterministic_algorithms_enabled() or torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
        raise ValueError('deterministic CPU execution and disabled TF32 required')
    pools=threadpool_info()
    if any(p['num_threads']!=4 for p in pools):raise ValueError('actual BLAS/OpenMP pools must use four threads')
    build=io.StringIO()
    with redirect_stdout(build):np.__config__.show()
    return dict(device='cpu',cpu_threads=4,torch_interop_threads=4,python=sys.version,executable=sys.executable,
        platform=platform.platform(),numpy=np.__version__,torch=str(torch.__version__),scipy=importlib.metadata.version('scipy'),
        threadpoolctl=importlib.metadata.version('threadpoolctl'),thread_environment={n:os.environ[n] for n in THREAD_ENV},
        threadpools=pools,numpy_build_sha256=hashlib.sha256(build.getvalue().encode()).hexdigest(),
        torch_build_sha256=hashlib.sha256(torch.__config__.show().encode()).hexdigest(),deterministic=True,tf32=False)


def source_names():
    """Conservative AST closure, including local imports inside frozen tests/references."""
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
            if isinstance(node,ast.Import):names=[v.name for v in node.names]
            elif isinstance(node,ast.ImportFrom):
                if node.level:raise ValueError('relative import needs explicit closure review: '+name)
                if node.module:names=[node.module]+[node.module+'.'+v.name for v in node.names]
            for module in names:
                parts=module.split('.')
                candidates=[Path(*parts).with_suffix('.py'),Path(*parts)/'__init__.py']
                candidates += [Path(*parts[:i])/'__init__.py' for i in range(1,len(parts))]
                pending.extend(str(p) for p in candidates if (ROOT/p).is_file() and str(p) not in found)
    return sorted(found)


def old_control_specs(selection):
    selected=selection['selection'];old=ROOT/'results/round15_regional';stage_a=ROOT/'results/round15_feasibility'
    specs={}
    def add(role,spec,regional=old,center=None):
        specs[role]=dict(family=spec['family'],grid_index=spec['grid_index'],config=spec['config'],
            regional_rows=[str(regional/'evaluation'/s.replace('/','__')/'rows.json') for s in SCENES],
            center_rows=[] if center is None else [str(center/'evaluation'/s.replace('/','__')/'rows.json') for s in SCENES])
    for family,spec in selected['controls'].items():add('regional_'+family,spec)
    for role,spec in selection['legacy_configurations'].items():add('legacy_'+role,spec)
    add('strong_point',selected['strong_point']);add('legacy_point',selection['legacy_configurations']['point'])
    add('strong_quad',selected['strong_quadrature']);add('legacy_quad16',selection['legacy_configurations']['quad16'])
    add('historical_primary',selected['historical_primary'])
    for role,family in (('point_localizer','gaussian_point'),('quartic_localizer','quartic_point')):
        add(role,dict(family=family,**selection['point_localization_selections'][family]),center=stage_a)
    add('legacy_point_localizer',selection['legacy_configurations']['point_pck'],center=stage_a)
    add('old_trace',selected['primary'],ROOT/'results/round17_regional',ROOT/'results/round17_feasibility')
    return specs


def inherited_contract():
    bindings,entries=HISTORY.inherited_contract()
    entries=[dict(e) for e in entries]
    denied={e['target_path'] for e in entries}
    def bind(path,expected=None):
        path=absolute(path).resolve()
        if str(path) in denied or path.name=='labels.npz':raise ValueError('label bytes are forbidden before audit')
        digest=sha(path)
        if expected is not None and digest!=expected:raise ValueError('changed inherited artifact: '+str(path))
        if str(path) in bindings and bindings[str(path)]!=digest:raise ValueError('conflicting inherited binding')
        bindings[str(path)]=digest
        return digest
    acceptance_path=ROOT/'results/round19_center_root_acceptance_20261031.json'
    accepted=read(acceptance_path)
    if (accepted.get('accepted_for_exploratory_interpretation') is not True
            or accepted.get('full_forward_and_final_independent_audits_passed') is not True
            or accepted.get('strict_forward_before_label_chronology_checked') is not True):
        raise ValueError('accepted complete R19 information diagnostic required')
    bind(acceptance_path)
    for path,digest in accepted['files'].items():bind(path,digest)
    r19=ROOT/'results/round19_center_semantics';r19lock=read(r19/'lock.json')
    if r19lock['source_function']!=HISTORY.function_contract():raise ValueError('accepted historical center function changed')
    for path,digest in r19lock['bindings'].items():bind(path,digest)
    for directory in (r19,ROOT/'results/round19_center_semantics_independent_audit'):
        completed=read(directory/'completion.json')
        if completed.get('passed') is not True:raise ValueError('complete R19 evidence required')
        for path,digest in completed['files'].items():bind(directory/path,digest)
        for name,digest in read(directory/'lock.json')['source_hashes'].items():
            bind(ROOT/name,digest)
            bind(directory/'source_snapshot'/name,digest)
    r19records={(r['scene'],tuple(r['pair'])):r for r in read(r19/'forward/records.json')}
    f_lock=read(ROOT/'results/round17_feasibility/lock.json')
    for path,digest in f_lock['baseline_bindings'].items():bind(path,digest)
    regional=ROOT/'results/round17_regional';selection=read(regional/'selection.json')
    bind(regional/'selection.json',(regional/'selection.sha256').read_text().strip())
    bind(regional/'selection.sha256')
    audit_path=absolute(selection['audit_path']);audit=read(audit_path)
    bind(audit_path,selection['audit_sha256'])
    if audit.get('passed') is not True:raise ValueError('accepted regional audit required')
    for name in ('lock','forward_complete','summary','completion'):
        digest=bind(regional/(name+'.json'),selection['regional_'+name+'_sha256'])
        if audit['regional_'+name+'_sha256']!=digest:raise ValueError('regional accepted chain differs')
    if (selection['selection']['primary']['family']!='trace_full'
            or selection['selection']['primary']['config']!=dict(h=.25,temperature=10.)):
        raise ValueError('fixed old primary changed')
    old_controls=old_control_specs(selection)
    for spec in old_controls.values():
        for path in spec['regional_rows']+spec['center_rows']:
            directory=Path(path).parents[2];relative=str(Path(path).relative_to(directory))
            complete=read(directory/'completion.json')
            bind(directory/'completion.json')
            bind(path,complete['files'][relative])
    for entry in entries:
        meta=read(entry['manifest_path']);entry['label_path']=str(Path(entry['path'])/'labels.npz')
        entry['label_sha256']=meta['label_sha256']  # Expected metadata only.
        r=r19records[(entry['scene'],tuple(entry['pair']))]
        entry['r19_archive_path']=str(r19/r['archive_path']);entry['r19_archive_sha256']=r['archive_sha256']
        bind(entry['r19_archive_path'],entry['r19_archive_sha256'])
    partitions_path=regional/'partitions.npz';partitions_sha=bind(partitions_path,read(regional/'lock.json')['partitions_sha256'])
    return bindings,entries,old_controls,str(partitions_path),partitions_sha


def verify_old_rows(lock):
    """Validate complete inherited selections/populations using accepted metadata only."""
    entries={(e['scene'],tuple(e['pair'])):e for e in lock['entries']};cache={};result={}
    for role,spec in lock['old_controls'].items():
        collected={}
        for kind in ('regional','center'):
            rows=[]
            for path in spec[kind+'_rows']:
                if path not in cache:cache[path]=read(path)
                for source in cache[path]:
                    if source['family']!=spec['family'] or source['grid_index']!=spec['grid_index']:continue
                    row=dict(source);e=entries.get((row['scene'],tuple(row['pair'])))
                    if e is None or row['input_sha256']!=e['input_sha256'] or row['config']!=spec['config']:
                        raise ValueError('inherited role/input/config mismatch')
                    if row.get('dtype')!='float64' or not row.get('array_sha256'):
                        raise ValueError('missing inherited probability identity')
                    if kind=='regional':
                        if row['label_sha256']!=e['label_sha256'] or row['replayed_array_sha256']!=row['array_sha256']:
                            raise ValueError('inherited regional target/replay mismatch')
                    else:
                        if row['target_sha256']!=e['target_sha256']:raise ValueError('inherited exact R13 center identity mismatch')
                        row.update(key='g8',condition=row['query']+'_g8')
                    rows.append(row)
            expected={(s,tuple(e['pair']),c['condition']) for s in SCENES for e in lock['entries'] if e['scene']==s
                      for c in CONDITIONS if kind=='regional' or (c['key']=='g8' and c['query'] in CENTER_QUERIES)} if spec[kind+'_rows'] else set()
            got={(r['scene'],tuple(r['pair']),r['condition']) for r in rows}
            if len(got)!=len(rows) or got!=expected:raise ValueError('incomplete inherited selected role: '+role+'/'+kind)
            if kind=='center' and rows:
                for q in CENTER_QUERIES:
                    if sum(r['n_point'] for r in rows if r['query']==q)!=VALID_COUNTS[q]:
                        raise ValueError('old center population changed')
            collected[kind]=rows
        result[role]=collected
    return result


def freeze(out,audit_root=DEFAULT_AUDIT):
    if out.exists():raise ValueError('refusing existing R20 directory')
    bindings,entries,old_controls,partitions_path,partitions_sha=inherited_contract()
    source_hashes={name:sha(ROOT/name) for name in source_names()}
    numerical_references=NUMREF.REFERENCE_POLICY
    current_runtime=runtime()
    if shutil.disk_usage(out.parent).free<RAW_PROBABILITY_BYTES+(512<<20):raise ValueError('insufficient space for complete retained probabilities')
    lock=dict(version=1,created_unix=time.time(),source_hashes=source_hashes,source_function=HISTORY.function_contract(),
        runtime=current_runtime,bindings=bindings,entries=entries,old_controls=old_controls,
        families=list(FAMILIES),role_configs=ROLE_CONFIGS,conditions=list(CONDITIONS),counts=COUNTS,
        expected_valid_counts=VALID_COUNTS,field_allowlist=list(FIELDS),partitions_path=partitions_path,
        partitions_sha256=partitions_sha,audit_root=str(audit_root.resolve()),numerical_policy=NUMERICAL_POLICY,
        numerical_references=numerical_references,pairs=48,unique_query_rows=39888,
        matrices=TOTAL_MATRICES,raw_probability_bytes=RAW_PROBABILITY_BYTES,
        actual_target_hashes_read=False,labels_decoded=False,scientific_gate_passed=False)
    inherited=verify_old_rows(lock)
    out.mkdir(parents=True)
    for name,digest in source_hashes.items():
        target=out/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((ROOT/name).read_bytes())
        if sha(target)!=digest:raise ValueError('source changed while freezing')
    write(out/'old_controls.json',old_controls)
    # Only already accepted evidence; these are not newly measured baseline rows.
    write(out/'inherited_rows.json',inherited)
    lock['old_controls_sha256']=sha(out/'old_controls.json');lock['inherited_rows_sha256']=sha(out/'inherited_rows.json')
    write(out/'lock.json',lock);(out/'lock.sha256').write_text(sha(out/'lock.json')+'\n')


def verify_lock(out):
    lock=read(out/'lock.json')
    if (sha(out/'lock.json')!=(out/'lock.sha256').read_text().strip() or lock['runtime']!=runtime()
            or lock['source_function']!=HISTORY.function_contract() or lock['families']!=list(FAMILIES)
            or lock['role_configs']!=ROLE_CONFIGS or lock['conditions']!=list(CONDITIONS) or lock['counts']!=COUNTS
            or lock['expected_valid_counts']!=VALID_COUNTS or lock['field_allowlist']!=list(FIELDS)
            or lock['numerical_policy']!=NUMERICAL_POLICY or set(lock['source_hashes'])!=set(source_names())
            or lock['numerical_references']!=NUMREF.REFERENCE_POLICY
            or lock['matrices']!=TOTAL_MATRICES or lock['raw_probability_bytes']!=RAW_PROBABILITY_BYTES):
        raise ValueError('R20 immutable contract/runtime changed')
    for name,digest in lock['source_hashes'].items():
        if sha(ROOT/name)!=digest or sha(out/'source_snapshot'/name)!=digest:raise ValueError('joint source changed: '+name)
    denied={e['target_path'] for e in lock['entries']}|{e['label_path'] for e in lock['entries']}
    for path,digest in lock['bindings'].items():
        if path in denied or Path(path).name=='labels.npz':raise ValueError('target entered pre-seal binding')
        if sha(path)!=digest:raise ValueError('inherited binding changed: '+path)
    for name in ('old_controls','inherited_rows'):
        if sha(out/(name+'.json'))!=lock[name+'_sha256']:raise ValueError('inherited evidence file changed')
    if read(out/'old_controls.json')!=lock['old_controls']:raise ValueError('old role declaration changed')
    return lock


def require_audit_freeze(out,lock):
    directory=Path(lock['audit_root']);audit=read(directory/'lock.json')
    if (sha(directory/'lock.json')!=(directory/'lock.sha256').read_text().strip()
            or audit['producer_lock_sha256']!=sha(out/'lock.json') or Path(audit['run']).resolve()!=out.resolve()
            or audit['source_hashes']!=lock['source_hashes']
            or audit['runtime']!=lock['runtime']
            or not lock['created_unix']<=audit['created_unix']<=time.time()):
        raise ValueError('joint independent audit must be frozen before forward')
    for name,digest in audit['source_hashes'].items():
        if sha(directory/'source_snapshot'/name)!=digest:raise ValueError('audit source snapshot changed')
    return directory


def load_parts(lock):
    if sha(lock['partitions_path'])!=lock['partitions_sha256']:raise ValueError('original partitions changed')
    with np.load(lock['partitions_path'],allow_pickle=False) as archive:
        parts={q:{name.split('__')[1]:archive[name] for name in archive.files if name.startswith(q+'__')} for q in QUERIES}
    integer={'bounds','owner','atoms','parent','cell_owner'}
    tensors={q:{name:torch.as_tensor(value,dtype=torch.int64 if name in integer else
                                    torch.float64 if name=='cell_bounds' else torch.float32)
                for name,value in p.items()} for q,p in parts.items()}
    if sum(len(parts[c['query']]['mass'])*len(parts[c['key']]['mass']) for c in CONDITIONS)*48*12*8!=RAW_PROBABILITY_BYTES:
        raise ValueError('native population/byte count changed')
    return parts,tensors


def load_inputs(entry):
    if sha(entry['input_path'])!=entry['input_sha256']:raise ValueError('approved input archive changed')
    with np.load(entry['input_path'],allow_pickle=False) as archive:inputs={f:archive[f] for f in FIELDS}
    for name,shape in (('q_predicted_depth',(144,192)),('q_K',(3,3)),('k_K',(3,3)),('q_to_k_transform',(4,4))):
        if inputs[name].shape!=shape:raise ValueError('legal center input shape changed')
        if name!='q_predicted_depth' and (inputs[name].dtype!=np.float64 or not np.isfinite(inputs[name]).all()):
            raise ValueError('historical cameras must be finite float64')
    obs={f:torch.as_tensor(inputs[f],dtype=torch.float32) for f in OBS_FIELDS}
    return inputs,obs


@torch.no_grad()
def prepare_pair(entry,parts,tensors,predictor=None):
    """Full role-owned preparation; retained support evidence does not alter role cost semantics."""
    inputs,obs=load_inputs(entry);predictor=predictor or HISTORY.extracted_predictor()
    queries={};keys={};retained={}
    with np.load(entry['support_path'],allow_pickle=False) as old, np.load(entry['r19_archive_path'],allow_pickle=False) as r19:
        for q in QUERIES:
            bounds=tensors[q]['bounds'].numpy()
            if not np.array_equal(bounds,old[q+'__bounds']):raise ValueError('accepted support/native bounds changed')
            a,front,fov=predictor({f:inputs[f] for f in ANCHOR_FIELDS},bounds)
            if (a.dtype!=np.float64 or a.shape!=(COUNTS[q],2) or not np.isfinite(a).all()
                    or front.dtype!=np.bool_ or fov.dtype!=np.bool_ or front.shape!=(COUNTS[q],) or fov.shape!=front.shape):
                raise ValueError('invalid historical center outputs')
            if q in CENTER_QUERIES:
                for name,value in (('a_px',a),('a_front',front),('a_fov',fov)):
                    if array_sha(value)!=array_sha(r19[q+'__'+name]):raise ValueError('R19 center/flag identity changed')
            anchor=torch.from_numpy(a/8.)
            queries[q]={family:MODEL.prepare_query(obs,tensors[q],family,anchor) for family in FAMILIES}
            keys[q]={family:MODEL.prepare_key(obs,tensors[q],family) for family in FAMILIES}
            primary=queries[q]['anchor_trace']
            for name,field in (('mean','old_mean'),('cov','original_cov')):
                value=primary[field].numpy()
                if value.dtype!=np.float32 or array_sha(value)!=array_sha(old[q+'__'+name]):
                    raise ValueError('new prepared original moment differs from accepted R17 support')
            for family in FAMILIES:
                if not torch.equal(queries[q][family]['features'],primary['features']):raise ValueError('role query content differs')
                if not torch.equal(keys[q][family]['features'],keys[q]['anchor_trace']['features']):raise ValueError('role key content differs')
            values=dict(bounds_px=bounds,a_px=a,a_front=front,a_fov=fov,mean=primary['old_mean'].numpy(),
                cov=primary['original_cov'].numpy(),a=anchor.numpy(),delta=primary['anchor_delta'].numpy(),
                anchor_cov=primary['tau_cov'].numpy(),q_features=primary['features'].numpy(),
                k_features=keys[q]['anchor_trace']['features'].numpy(),
                tau32=TRACE.trace_temperature(primary['tau_cov'],.25,10.).float().numpy())
            retained.update({q+'__'+name:value for name,value in values.items()})
    return queries,keys,retained


def failed_stage(directory,started,error):
    write(directory/'failure.json',dict(type=type(error).__name__,message=str(error),started_unix=started,finished_unix=time.time()))
    if not (directory/'completion.json').exists():
        write(directory/'completion.json',dict(passed=False,files=inventory(directory),finished_unix=time.time()))


def expected_identities(entries):
    return {(e['scene'],tuple(e['pair']),c['condition'],f) for e in entries for c in CONDITIONS for f in FAMILIES}


def validate_records(records,entries):
    identities={(r['scene'],tuple(r['pair']),r['condition'],r['family']) for r in records}
    if len(identities)!=len(records) or identities!=expected_identities(entries):raise ValueError('incomplete new forward population')
    mapping={(e['scene'],tuple(e['pair'])):e for e in entries};conditions={c['condition']:c for c in CONDITIONS}
    for r in records:
        c=conditions[r['condition']];e=mapping[(r['scene'],tuple(r['pair']))];cfg=ROLE_CONFIGS[r['family']]
        if (r['config']!={k:cfg[k] for k in ('h','temperature')} or any(r[k]!=c[k] for k in ('query','key','scenario'))
                or r['input_sha256']!=e['input_sha256'] or r['shape']!=[COUNTS[c['query']],COUNTS[c['key']]]
                or r['dtype']!='float64' or r['pair_name']!=e['pair_name']):raise ValueError('new forward identity changed')


@torch.no_grad()
def forward(out,scene):
    if scene not in SCENES:raise ValueError('only frozen TRAIN scenes allowed')
    directory=out/'forward'/scene.replace('/','__')
    if directory.exists():raise ValueError('refusing existing forward attempt')
    directory.mkdir(parents=True);started=time.time()
    write(directory/'started.json',dict(created_unix=started,producer_lock_sha256=sha(out/'lock.json'),scene=scene))
    try:
        lock=verify_lock(out);require_audit_freeze(out,lock);parts,tensors=load_parts(lock)
        entries=[e for e in lock['entries'] if e['scene']==scene]
        if len(entries)!=16:raise ValueError('exact sixteen pairs per scene required')
        if shutil.disk_usage(out).free<RAW_PROBABILITY_BYTES//3+(128<<20):raise ValueError('insufficient space for this complete scene')
        records=[];support_records=[];predictor=HISTORY.extracted_predictor()
        for index,entry in enumerate(entries):
            queries,keys,retained=prepare_pair(entry,parts,tensors,predictor)
            folder=directory/'arrays'/entry['pair_name'];folder.mkdir(parents=True)
            support_path=folder/'supports.npz';np.savez(support_path,**retained)
            support_records.append(dict(scene=scene,pair=entry['pair'],pair_name=entry['pair_name'],input_sha256=entry['input_sha256'],
                archive_path=str(support_path.relative_to(out)),archive_sha256=sha(support_path),
                array_sha256={n:array_sha(v) for n,v in retained.items()}))
            for condition in CONDITIONS:
                q,k=condition['query'],condition['key'];arrays={};pending=[]
                for family in FAMILIES:
                    cfg=ROLE_CONFIGS[family];p,diagnostics=MODEL.predict(queries[q][family],keys[k][family],family,cfg['h'],cfg['temperature'])
                    p=p.numpy();METRICS.probability(p,(COUNTS[q],COUNTS[k]));winner=p.argmax(-1).astype(np.int32)
                    arrays[family]=p;arrays[family+'__argmax']=winner
                    pending.append(dict(scene=scene,pair=entry['pair'],pair_name=entry['pair_name'],**condition,
                        family=family,config={n:cfg[n] for n in ('h','temperature')},shape=list(p.shape),dtype=str(p.dtype),
                        array_sha256=array_sha(p),argmax_sha256=array_sha(winner),input_sha256=entry['input_sha256'],numerical=diagnostics))
                archive_path=folder/(condition['condition']+'.npz');np.savez(archive_path,**arrays);digest=sha(archive_path)
                for r in pending:r.update(archive_path=str(archive_path.relative_to(out)),archive_sha256=digest)
                records.extend(pending)
            print('R20_FORWARD',scene,index+1,'/16',flush=True)
        validate_records(records,entries)
        write(directory/'records.json',records);write(directory/'support_records.json',support_records)
        write(directory/'completion.json',dict(passed=True,files=inventory(directory),scene=scene,records=len(records),
            actual_target_hashes_read=False,labels_decoded=False,producer_lock_sha256=sha(out/'lock.json'),
            started_unix=started,finished_unix=time.time()))
    except Exception as error:
        failed_stage(directory,started,error);raise


def seal(out):
    directory=out/'seal'
    if directory.exists() or (out/'forward_complete.json').exists():raise ValueError('refusing existing global seal')
    directory.mkdir();started=time.time();write(directory/'started.json',dict(created_unix=started))
    try:
        lock=verify_lock(out);require_audit_freeze(out,lock);records=[];supports=[]
        for scene in SCENES:
            folder=out/'forward'/scene.replace('/','__');complete=verify_inventory(folder)
            if complete['actual_target_hashes_read'] or complete['labels_decoded'] or complete['producer_lock_sha256']!=sha(out/'lock.json'):
                raise ValueError('pre-seal forward chronology/source mismatch')
            records.extend(read(folder/'records.json'));supports.extend(read(folder/'support_records.json'))
        validate_records(records,lock['entries'])
        expected={(e['scene'],tuple(e['pair'])):e for e in lock['entries']}
        if (len(supports)!=48 or {(r['scene'],tuple(r['pair'])) for r in supports}!=set(expected)):
            raise ValueError('all unique support evidence required')
        for r in supports:
            e=expected[(r['scene'],tuple(r['pair']))]
            if r['input_sha256']!=e['input_sha256'] or r['pair_name']!=e['pair_name']:
                raise ValueError('support identity mismatch')
        by_archive={}
        for r in records:by_archive.setdefault(r['archive_path'],[]).append(r)
        for name,rows in by_archive.items():
            digest=sha(out/name)
            with np.load(out/name,allow_pickle=False) as arrays:
                if set(arrays.files)!={n for f in FAMILIES for n in (f,f+'__argmax')}:raise ValueError('unexpected saved fields')
                for r in rows:
                    p=arrays[r['family']];winner=arrays[r['family']+'__argmax'];METRICS.probability(p,r['shape'])
                    if (digest!=r['archive_sha256'] or array_sha(p)!=r['array_sha256']
                            or winner.dtype!=np.int32 or array_sha(winner)!=r['argmax_sha256']
                            or not np.array_equal(winner,p.argmax(-1))):raise ValueError('saved full probability/winner changed')
        for r in supports:
            if sha(out/r['archive_path'])!=r['archive_sha256']:raise ValueError('saved support archive changed')
            with np.load(out/r['archive_path'],allow_pickle=False) as a:
                if {n:array_sha(a[n]) for n in a.files}!=r['array_sha256']:raise ValueError('saved support array changed')
        write(out/'forward/records.json',records);write(out/'forward/support_records.json',supports)
        write(out/'forward/completion.json',dict(passed=True,files=inventory(out/'forward'),matrices=len(records),
            actual_target_hashes_read=False,labels_decoded=False,finished_unix=time.time()))
        write(out/'forward_complete.json',dict(producer_lock_sha256=sha(out/'lock.json'),
            forward_completion_sha256=sha(out/'forward/completion.json'),forward_records_sha256=sha(out/'forward/records.json'),
            support_records_sha256=sha(out/'forward/support_records.json'),pairs=48,matrices=TOTAL_MATRICES,
            raw_probability_bytes=RAW_PROBABILITY_BYTES,all_probabilities_retained=True,
            actual_target_hashes_read=False,labels_decoded=False,created_unix=time.time()))
        write(directory/'completion.json',dict(passed=True,files=inventory(directory),
            forward_complete_sha256=sha(out/'forward_complete.json'),finished_unix=time.time()))
    except Exception as error:failed_stage(directory,started,error);raise


def require_forward_audit(out,lock):
    audit_root=require_audit_freeze(out,lock);verify_inventory(out/'forward');seal=verify_inventory(out/'seal')
    record=read(out/'forward_complete.json')
    if (record['producer_lock_sha256']!=sha(out/'lock.json') or record['forward_completion_sha256']!=sha(out/'forward/completion.json')
            or record['forward_records_sha256']!=sha(out/'forward/records.json')
            or record['support_records_sha256']!=sha(out/'forward/support_records.json')
            or seal['forward_complete_sha256']!=sha(out/'forward_complete.json')
            or record['actual_target_hashes_read'] or record['labels_decoded']):raise ValueError('changed complete forward seal')
    accepted=read(audit_root/'forward_verification.json');complete=verify_inventory(audit_root/'forward_audit')
    if (accepted.get('passed') is not True or accepted['producer_lock_sha256']!=sha(out/'lock.json')
            or accepted['producer_forward_complete_sha256']!=sha(out/'forward_complete.json')
            or accepted['audit_lock_sha256']!=sha(audit_root/'lock.json')
            or complete['verification_sha256']!=sha(audit_root/'forward_verification.json')
            or not record['created_unix']<=accepted['finished_unix']<=time.time()):
        raise ValueError('passed complete numerical/forward audit required before target bytes')
    return audit_root


def evaluate(out):
    directory=out/'evaluation'
    if directory.exists() or (out/'summary.json').exists() or (out/'completion.json').exists():raise ValueError('preserve existing evaluation attempt')
    directory.mkdir();started=time.time();write(directory/'started.json',dict(created_unix=started))
    try:
        lock=verify_lock(out);audit_root=require_forward_audit(out,lock);parts,_=load_parts(lock)
        records=read(out/'forward/records.json');validate_records(records,lock['entries'])
        grouped={}
        for r in records:grouped.setdefault((r['scene'],tuple(r['pair'])),[]).append(r)
        rows=[];label_sources={}
        for entry in lock['entries']:
            # First actual target bytes are touched only after the complete independent audit above.
            for path,expected in ((entry['label_path'],entry['label_sha256']),(entry['target_path'],entry['target_sha256'])):
                if sha(path)!=expected:raise ValueError('exact old target archive changed')
                label_sources[path]=expected
            with np.load(entry['label_path'],allow_pickle=False) as a:labels={n:a[n] for n in ('gt_xy','gt_visible')}
            with np.load(entry['target_path'],allow_pickle=False) as a:
                center={n:a[n] for q in CENTER_QUERIES for n in (q+'_point_xy',q+'_point_visible')}
            pair_records=grouped[(entry['scene'],tuple(entry['pair']))]
            for condition in CONDITIONS:
                q,k=condition['query'],condition['key'];direct,common=METRICS.target_counts(labels,parts,q,k)
                chosen=[r for r in pair_records if r['condition']==condition['condition']]
                with np.load(out/chosen[0]['archive_path'],allow_pickle=False) as arrays:
                    for r in chosen:
                        p=arrays[r['family']];winner=arrays[r['family']+'__argmax']
                        if array_sha(p)!=r['array_sha256'] or array_sha(winner)!=r['argmax_sha256']:
                            raise ValueError('saved new prediction changed before metrics')
                        metrics=METRICS.region_metrics(p,parts,condition,direct,common)
                        cm=METRICS.center_metrics(winner,parts[k]['bounds'],center[q+'_point_xy'],center[q+'_point_visible']) if q in CENTER_QUERIES else None
                        rows.append(dict(r,**metrics,center=cm,label_sha256=entry['label_sha256'],
                                         target_sha256=entry['target_sha256'] if cm else None))
            print('R20_EVALUATE',entry['scene'],entry['pair_name'],flush=True)
        validate_records(rows,lock['entries']);by_role={}
        for family in FAMILIES:
            selected=[r for r in rows if r['family']==family]
            centers=[dict(scene=r['scene'],pair=r['pair'],query=r['query'],key=r['key'],condition=r['condition'],**r['center']) for r in selected if r['center'] is not None]
            for q in CENTER_QUERIES:
                if sum(r['n_point'] for r in centers if r['condition']==q+'_g8')!=VALID_COUNTS[q]:raise ValueError('new exact center population changed')
            by_role[family]=METRICS.summarize_role(selected,centers)
        inherited=verify_old_rows(lock)
        old_by_role={name:METRICS.summarize_role(v['regional'],v['center']) for name,v in inherited.items()}
        summary=dict(by_role=by_role,old_by_role=old_by_role,screen=METRICS.screen(by_role,old_by_role),
            inherited_baselines=True,new_baselines_optimized_for_a=False,scientific_gate_passed=False,
            validation_or_fresh_read=False,producer_lock_sha256=sha(out/'lock.json'),
            producer_forward_complete_sha256=sha(out/'forward_complete.json'),
            forward_audit_sha256=sha(audit_root/'forward_verification.json'),independent_audit_pending=True,
            pairs=48,matrices=len(rows),started_unix=started,finished_unix=time.time())
        write(directory/'rows.json',rows);write(out/'summary.json',summary)
        write(directory/'completion.json',dict(passed=True,files=inventory(directory),label_sources=label_sources,
            labels_decoded=True,records=len(rows),summary_sha256=sha(out/'summary.json'),finished_unix=time.time()))
        write(out/'completion.json',dict(passed=True,files=inventory(out),producer_lock_sha256=sha(out/'lock.json'),
            scientific_gate_passed=False,independent_audit_pending=True,finished_unix=time.time()))
    except Exception as error:failed_stage(directory,started,error);raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('freeze','forward','seal','evaluate'))
    parser.add_argument('--out',type=Path,default=DEFAULT_OUT);parser.add_argument('--audit-root',type=Path,default=DEFAULT_AUDIT)
    parser.add_argument('--scene',choices=SCENES)
    args=parser.parse_args()
    for name in THREAD_ENV:
        if os.environ.get(name)!='4':raise ValueError('launch with OMP/MKL/OPENBLAS_NUM_THREADS=4')
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
