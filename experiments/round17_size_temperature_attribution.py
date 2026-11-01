"""Unrun TRAIN-only paired attribution: observed trace versus native size.

No configuration search or scientific gate. All probabilities are retained and
globally sealed before old TRAIN labels can be opened. Never changes R17 VAL.
"""
from __future__ import annotations
import argparse
from fractions import Fraction
from math import fsum
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from experiments import round17_regional as R
from models import round17_size_temperature_pe as SIZE

ROOT=R.ROOT
SCENES=R.SCENES
CONDITIONS=R.CONDITIONS
ROLES=('trace_full','size_temperature')
CONFIG=dict(h=.25,temperature=10.)
TOTAL_MATRICES=48*14*2
RAW_PROBABILITY_BYTES=512434944
OWN_SOURCES=('ROUND17_SIZE_TEMPERATURE_PROTOCOL.md','docs/round17_size_temperature_attribution_review.md',
    'models/round17_size_temperature_pe.py','experiments/round17_size_temperature_attribution.py',
    'tests/test_round17_size_temperature_attribution.py',
    'experiments/round17_size_temperature_audit.py','tests/test_round17_size_temperature_audit.py')
sha,read,write,array_sha=R.sha,R.read,R.write,R.array_sha


def no_active_validation_benchmark():
    timing=ROOT/'results/round17_validation/timing'
    if timing.exists() and not (timing/'completion.json').exists():
        raise RuntimeError('do not compete with the frozen validation benchmark')


def accepted_contract(directory):
    """Read only accepted TRAIN metadata and forward commitments, never labels."""
    directory=directory.resolve(); lock=read(directory/'lock.json'); selected=read(directory/'selection.json')
    if (sha(directory/'lock.json')!=(directory/'lock.sha256').read_text().strip()
            or sha(directory/'selection.json')!=(directory/'selection.sha256').read_text().strip()):
        raise ValueError('accepted regional locks changed')
    R.A.verify_files({str(ROOT/n):h for n,h in lock['source_hashes'].items()})
    R.A.verify_files({str(directory/'source_snapshot'/n):h for n,h in lock['source_hashes'].items()})
    audit_path=R.absolute(selected['audit_path']); audit=read(audit_path)
    if sha(audit_path)!=selected['audit_sha256'] or audit['passed'] is not True:
        raise ValueError('accepted independent TRAIN audit required')
    files={str(directory/'selection.json'):sha(directory/'selection.json'),str(audit_path):sha(audit_path)}
    for name in ('lock','forward_complete','summary','completion'):
        digest=sha(directory/(name+'.json')); field='regional_'+name+'_sha256'
        if selected[field]!=digest or audit[field]!=digest:
            raise ValueError('accepted regional provenance mismatch')
        files[str(directory/(name+'.json'))]=digest
    primary=selected['selection']['primary']
    if (primary['family']!='trace_full' or primary['grid_index']!=2 or primary['config']!=CONFIG
            or selected['selection']!=audit['selection'] or selected['train_screen']['passed'] is not True
            or selected['validation_run'] or selected['scientific_gate_passed']):
        raise ValueError('only the fixed accepted TRAIN primary is permitted')
    completed=read(directory/'completion.json')
    for scene in SCENES:
        name='forward/'+scene.replace('/','__')+'/records.json'
        if sha(directory/name)!=completed['files'][name]:
            raise ValueError('old primary commitments changed')
        files[str(directory/name)]=completed['files'][name]
    repair_root=audit_path.parent.parent
    from experiments import round17_regional_audit_path_repair as REPAIR
    repair_lock=REPAIR.verify_repair(repair_root); repair_complete=read(repair_root/'completion.json')
    if (repair_complete['passed'] is not True
            or repair_complete['files'][str(audit_path.relative_to(repair_root))]!=sha(audit_path)
            or audit['path_repair']['repair_lock_sha256']!=sha(repair_root/'lock.json')
            or repair_lock['original_contract']['joint_lock_sha256']!=sha(directory/'lock.json')
            or repair_lock['original_contract']['run']!=str(directory)):
        raise ValueError('accepted regional repair completion required')
    for path in (repair_root/'lock.json',repair_root/'completion.json',audit_path.parent/'completion.json'):
        files[str(path)]=sha(path)
    inherited=dict(lock['source_hashes'],**repair_lock['source_hashes'])
    targets=lock['stage_a']['target_contract']
    target_entries={(e['scene'],tuple(e['pair'])):e for e in targets['entries']}
    entries=[]
    for entry in lock['entries']:
        meta_path=Path(entry['path'])/'manifest.json'
        if sha(meta_path)!=entry['manifest_sha256']:
            raise ValueError('TRAIN manifest changed')
        meta=read(meta_path)
        if meta['split']!='train' or meta['forward_uses_sensor']:
            raise ValueError('only frozen predicted TRAIN observations are allowed')
        target=target_entries[(entry['scene'],tuple(entry['pair']))]
        entries.append(dict(entry,target_path=target['target_path'],target_sha256=target['target_sha256']))
    if len(entries)!=48 or len({(e['scene'],tuple(e['pair'])) for e in entries})!=48:
        raise ValueError('complete old 48-pair TRAIN cohort required')
    return dict(root=str(directory),files=files,source_hashes=inherited,entries=entries,
        selected_configuration=dict(family='trace_full',grid_index=2,**CONFIG),target_contract=targets,
        partitions_sha256=lock['partitions_sha256'])


def sources(prior):
    from experiments import round17_size_temperature_audit as independent
    result=dict(prior['source_hashes'])
    for name in set(OWN_SOURCES).union(independent.SOURCE_NAMES):
        result[name]=sha(ROOT/name)  # Absent independent audit source blocks freeze.
    for name,digest in result.items():
        if sha(ROOT/name)!=digest:
            raise ValueError('source closure changed')
    return result


def freeze(out,regional):
    no_active_validation_benchmark()
    if out.exists():
        raise ValueError('refusing attribution overwrite')
    prior=accepted_contract(regional); source=sources(prior)
    for e in prior['entries']:
        if sha(Path(e['path'])/'inputs.npz')!=e['input_sha256']:
            raise ValueError('TRAIN observation changed')
    parts=R.S.M.partitions()
    raw_bytes=48*2*8*sum(len(parts[c['query']]['mass'])*len(parts[c['key']]['mass']) for c in CONDITIONS)
    if raw_bytes!=RAW_PROBABILITY_BYTES:
        raise ValueError('native population byte count changed')
    out.mkdir(parents=True)
    np.savez_compressed(out/'partitions.npz',**{q+'__'+k:v for q,p in parts.items() for k,v in p.items()})
    if sha(out/'partitions.npz')!=prior['partitions_sha256']:
        # NPZ byte serialization may change timestamps; compare native arrays.
        with np.load(Path(prior['root'])/'partitions.npz',allow_pickle=False) as old:
            for q,p in parts.items():
                for k,v in p.items():np.testing.assert_array_equal(old[q+'__'+k],v)
    for name in source:
        target=out/'source_snapshot'/name; target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((ROOT/name).read_bytes())
    write(out/'lock.json',dict(created_unix=time.time(),source_hashes=source,prior=prior,entries=prior['entries'],
        roles=list(ROLES),config=CONFIG,conditions=list(CONDITIONS),runtime=R.runtime(),
        partitions_sha256=sha(out/'partitions.npz'),matrices=TOTAL_MATRICES,raw_probability_bytes=raw_bytes,
        field_allowlist=list(R.A.FIELDS),size_trace='(width_pixels**2+height_pixels**2)/(64*12)',
        size_temperature_arithmetic='float64 trace and T*(h*h/(h*h+trace)); cast once float32',
        content_arithmetic='tau32*full_GEMM_content32, then add geometry64 and softmax64',
        independent_audit_required=True,all_probabilities_retained=True,labels_decoded=False,
        validation_effects_read=False,changes_current_validation=False,configuration_search=False,
        scientific_gate_defined=False,cost_measured=False))
    (out/'lock.sha256').write_text(sha(out/'lock.json')+'\n')


def verify_lock(out):
    lock=read(out/'lock.json')
    if sha(out/'lock.json')!=(out/'lock.sha256').read_text().strip() or lock['source_hashes']!=sources(lock['prior']):
        raise ValueError('attribution source lock changed')
    R.A.verify_files({str(out/'source_snapshot'/n):h for n,h in lock['source_hashes'].items()})
    R.A.verify_files(lock['prior']['files'])
    if (lock['roles']!=list(ROLES) or lock['config']!=CONFIG or lock['conditions']!=list(CONDITIONS)
            or lock['runtime']!=R.runtime() or sha(out/'partitions.npz')!=lock['partitions_sha256']):
        raise ValueError('fixed attribution policy changed')
    return lock


def require_audit_freeze(out):
    directory=ROOT/'results/round17_size_temperature_independent_audit'
    design=read(directory/'lock.json')
    if (sha(directory/'lock.json')!=(directory/'lock.sha256').read_text().strip()
            or design['producer_lock_sha256']!=sha(out/'lock.json') or Path(design['run']).resolve()!=out.resolve()
            or not read(out/'lock.json')['created_unix']<=design['created_unix']<=time.time()):
        raise ValueError('independent design must precede all attribution forward work')
    R.A.verify_files({str(ROOT/n):h for n,h in design['source_hashes'].items()})
    R.A.verify_files({str(directory/'source_snapshot'/n):h for n,h in design['source_hashes'].items()})


def geometry_content(query,key):
    raw=R.S.M.tiled_cell_mass(query['mean'],query['cov'].double(),key,CONFIG['h'])
    if raw.dtype!=torch.float64 or not bool(torch.isfinite(raw).all()) or bool((raw<0).any()):
        raise ValueError('invalid unchanged Gaussian mass')
    geometry=(raw+1e-30*(key['mass'].double()/64)[None]).log()
    content=query['features']@key['features'].T
    fov=raw.sum(-1)
    numerical=dict(raw_zero_cells=int((raw==0).sum()),raw_zero_fov_rows=int((fov==0).sum()),
        raw_tiny_fov_rows=int((fov<1e-20).sum()),raw_fov_min=float(fov.min()),raw_fov_max=float(fov.max()),
        raw_floor_fov_mass=float(1e-30*(key['mass'].double()/64).sum()))
    return geometry,content,raw,numerical


def role_probability(observations,query_partition,key_partition,role):
    """Future cost hook: own full preparation/integration/tau; no role cache.

    Observation tensor conversion may be shared outside timing. No cost run or
    efficiency claim is implemented in this descriptive protocol.
    """
    if role not in ROLES:raise ValueError('undeclared attribution role')
    query=R.A.prepare_query(observations,query_partition,True)
    key=R.S.M.prepare_key(observations,key_partition)
    geometry,content,_,_=geometry_content(query,key)
    if role=='trace_full':
        return R.P.probabilities(geometry,content,query['cov'],**{'bandwidth':CONFIG['h'],'temperature':CONFIG['temperature']})
    return SIZE.probabilities(geometry,content,query_partition['bounds'],CONFIG['h'],CONFIG['temperature'])


def original_records(lock,scene):
    path=Path(lock['prior']['root'])/'forward'/scene.replace('/','__')/'records.json'
    records=read(path)
    values={(tuple(r['pair']),r['condition']):r for r in records if r['family']=='trace_full' and r['grid_index']==2}
    if len(values)!=16*14:raise ValueError('complete primary commitments missing')
    return values


def identity(row):
    return row['scene'],tuple(row['pair']),row['role'],row['condition']


def population(rows,entries):
    expected={(e['scene'],tuple(e['pair']),role,c['condition']) for e in entries for role in ROLES for c in CONDITIONS}
    if len(rows)!=len(expected) or {identity(r) for r in rows}!=expected:
        raise ValueError('incomplete paired attribution population')
    specs={c['condition']:c for c in CONDITIONS}; bypair={(e['scene'],tuple(e['pair'])):e for e in entries}
    counts=dict(g8=432,g16=108,mixed=243,g24=48)
    for r in rows:
        c=specs[r['condition']]
        if (any(r[k]!=v for k,v in c.items()) or r['config']!=CONFIG or r['dtype']!='float64'
                or r['shape']!=[counts[c['query']],counts[c['key']]]
                or r['input_sha256']!=bypair[r['scene'],tuple(r['pair'])]['input_sha256']):
            raise ValueError('paired source/configuration/shape changed')


@torch.no_grad()
def forward(out,scene):
    no_active_validation_benchmark(); lock=verify_lock(out); require_audit_freeze(out)
    entries=R.scene_entries(lock,scene); directory=out/'forward'/scene.replace('/','__')
    if directory.exists():raise ValueError('refusing forward overwrite')
    if shutil.disk_usage(out).free<RAW_PROBABILITY_BYTES+512*1024**2:raise ValueError('insufficient evidence space')
    directory.mkdir(parents=True); started=time.time()
    write(directory/'started.json',dict(created_unix=started,lock_sha256=sha(out/'lock.json'),runtime=R.runtime()))
    parts=R.read_parts(out); tensors=R.S.M.partition_tensors(parts,'cpu'); previous=original_records(lock,scene)
    records=[]; repeated=[]
    for pair_index,entry in enumerate(entries):
        input_path=Path(entry['path'])/'inputs.npz'
        if sha(input_path)!=entry['input_sha256']:raise ValueError('TRAIN forward input changed')
        inputs=R.A.load_forward_inputs(input_path); queries,keys=R.prepare_pair(inputs,tensors)
        repeated_queries,repeated_keys=R.prepare_pair(inputs,tensors) if pair_index==0 else (None,None)
        dest=directory/'arrays'/Path(entry['path']).name; dest.mkdir(parents=True)
        support={}
        for q,query in queries.items():
            support.update({q+'__'+name:value for name,value in dict(mean=query['mean'].numpy(),cov=query['cov'].numpy(),
                bounds=tensors[q]['bounds'].numpy(),native_trace=SIZE.native_trace(tensors[q]['bounds']).numpy(),
                tau_obs32=R.P.trace_temperature(query['cov'],CONFIG['h'],CONFIG['temperature']).numpy(),
                tau_size32=SIZE.size_temperature(tensors[q]['bounds'],CONFIG['h'],CONFIG['temperature']).numpy()).items()})
        np.savez_compressed(dest/'supports.npz',**support)
        for c in CONDITIONS:
            q,k=c['query'],c['key']; query,key=queries[q],keys[k]
            geometry,content,raw,numerical=geometry_content(query,key)
            probabilities=dict(trace_full=R.P.probabilities(geometry,content,query['cov'],CONFIG['h'],CONFIG['temperature']).numpy(),
                size_temperature=SIZE.probabilities(geometry,content,tensors[q]['bounds'],CONFIG['h'],CONFIG['temperature']).numpy())
            if pair_index==0:
                rg,rc,_,rn=geometry_content(repeated_queries[q],repeated_keys[k])
                rp=dict(trace_full=R.P.probabilities(rg,rc,repeated_queries[q]['cov'],CONFIG['h'],CONFIG['temperature']).numpy(),
                    size_temperature=SIZE.probabilities(rg,rc,tensors[q]['bounds'],CONFIG['h'],CONFIG['temperature']).numpy())
                if rn!=numerical or any(array_sha(rp[role])!=array_sha(probabilities[role]) for role in ROLES):
                    raise ValueError('first metadata pair full-preparation repeat is not exact')
                repeated.extend(dict(condition=c['condition'],role=role,array_sha256=array_sha(rp[role])) for role in ROLES)
            old=previous[tuple(entry['pair']),c['condition']]
            if (array_sha(probabilities['trace_full'])!=old['array_sha256'] or numerical!=old['numerical']
                    or old['input_sha256']!=entry['input_sha256']):
                raise ValueError('fixed primary does not exactly reproduce accepted R17B')
            saved=dict(probabilities)
            for role,p in probabilities.items():
                R.validate_probability(p,(len(query['mean']),len(key['mass'])))
                winner=p.argmax(-1).astype(np.int32); saved[role+'__argmax']=winner
                records.append(dict(scene=scene,pair=entry['pair'],role=role,config=CONFIG,**c,
                    shape=list(p.shape),dtype='float64',array_sha256=array_sha(p),argmax_sha256=array_sha(winner),
                    input_sha256=entry['input_sha256'],primary_regional_array_sha256=old['array_sha256'],numerical=numerical,
                    archive_path=str((dest/(c['condition']+'.npz')).relative_to(out)),archive_field=role))
            if pair_index==0:
                indices=np.linspace(0,len(query['mean'])-1,8,dtype=np.int64)
                observed_tau=R.P.trace_temperature(query['cov'],CONFIG['h'],CONFIG['temperature'])
                size_tau=SIZE.size_temperature(tensors[q]['bounds'],CONFIG['h'],CONFIG['temperature'])
                observed_bias=R.P.content_bias(query['cov'],CONFIG['h'],CONFIG['temperature'],content)
                size_bias=size_tau[:,None]*content
                saved.update(indices8=indices,raw8=raw.numpy()[indices],content8=content.numpy()[indices],
                    tau_obs32_8=observed_tau.numpy()[indices],tau_size32_8=size_tau.numpy()[indices],
                    bias_obs32_8=observed_bias.numpy()[indices],bias_size32_8=size_bias.numpy()[indices])
            np.savez_compressed(dest/(c['condition']+'.npz'),**saved)
        print('SIZE_ATTRIBUTION_FORWARD',scene,pair_index+1,'/16',flush=True)
    population(records,entries); write(directory/'records.json',records)
    write(directory/'determinism.json',dict(pair=entries[0]['pair'],records=repeated,passed=True,
        probability_count=len(repeated),labels_decoded=False))
    R.A.complete_stage(directory,out,'forward',scene,started,record_count=len(records),runtime=R.runtime(),
        labels_decoded=False,primary_exact_matrices=16*14,all_probabilities_retained=True,determinism_matrices=len(repeated))


def forward_chain(out,lock):
    chain={}
    for scene in SCENES:
        directory=out/'forward'/scene.replace('/','__'); value=R.A.verify_stage(directory,out,'forward',scene)
        if (value['record_count']!=16*14*2 or value['labels_decoded'] or value['primary_exact_matrices']!=16*14
                or value['runtime']!=lock['runtime'] or value['determinism_matrices']!=28):raise ValueError('incomplete label-free forward')
        chain[scene]=dict(completion_sha256=sha(directory/'completion.json'),finished_unix=value['finished_unix'])
    return chain


def seal_forward(out):
    no_active_validation_benchmark(); lock=verify_lock(out)
    if (out/'forward_complete.json').exists():raise ValueError('refusing seal overwrite')
    chain=forward_chain(out,lock); rows=[]; total_bytes=0
    for scene in SCENES:
        current=read(out/'forward'/scene.replace('/','__')/'records.json'); population(current,R.scene_entries(lock,scene))
        repeat=read(out/'forward'/scene.replace('/','__')/'determinism.json')
        first=R.scene_entries(lock,scene)[0]['pair']
        expected=[{k:r[k] for k in ('condition','role','array_sha256')} for r in current if r['pair']==first]
        if repeat!=dict(pair=first,records=expected,passed=True,probability_count=28,labels_decoded=False):
            raise ValueError('incomplete deterministic first-pair repeat')
        for r in current:
            with np.load(out/r['archive_path'],allow_pickle=False) as archive:
                p,w=archive[r['archive_field']],archive[r['role']+'__argmax']
                R.validate_probability(p,tuple(r['shape']))
                if array_sha(p)!=r['array_sha256'] or array_sha(w)!=r['argmax_sha256'] or not np.array_equal(w,p.argmax(-1)):
                    raise ValueError('saved native probability/argmax changed')
                total_bytes+=p.nbytes
        rows.extend(current)
    population(rows,lock['entries'])
    if total_bytes!=RAW_PROBABILITY_BYTES:raise ValueError('incomplete raw probability evidence')
    write(out/'forward_complete.json',dict(lock_sha256=sha(out/'lock.json'),forward_chain=chain,sealed_unix=time.time(),
        matrices=len(rows),raw_probability_bytes=total_bytes,labels_decoded=False,all_probabilities_retained=True))


def require_seal(out,lock):
    seal=read(out/'forward_complete.json')
    if (seal['lock_sha256']!=sha(out/'lock.json') or seal['forward_chain']!=forward_chain(out,lock)
            or seal['matrices']!=TOTAL_MATRICES or seal['raw_probability_bytes']!=RAW_PROBABILITY_BYTES
            or seal['labels_decoded'] or seal['sealed_unix']>time.time()):
        raise ValueError('all 48 TRAIN pairs must be sealed before reading any label')
    return seal


def load_labels(entry):
    old=Path(entry['path'])/'labels.npz'; target=Path(entry['target_path'])
    if sha(old)!=entry['label_sha256'] or sha(target)!=entry['target_sha256']:
        raise ValueError('old TRAIN target changed')
    with np.load(old,allow_pickle=False) as archive:
        labels={name:archive[name] for name in ('gt_xy','gt_visible','mixed_point_xy','mixed_point_visible')}
    with np.load(target,allow_pickle=False) as archive:
        # Preserve the exact old Stage A center coordinates, not an allclose
        # R10 variant that could classify a radius-boundary query differently.
        labels.update({q+'_point_'+f:archive[q+'_point_'+f] for q in ('g8','g16','g24') for f in ('xy','visible')})
    return labels


def center_pair_metrics(primary,size,key_bounds,target,visible):
    if visible.dtype!=np.bool_ or target.shape!=(len(primary),2) or not np.isfinite(target[visible]).all():
        raise ValueError('invalid center target population')
    centers=(key_bounds[:,:2]+key_bounds[:,2:]-1)/2
    a,b=primary.argmax(-1),size.argmax(-1)
    ac=np.zeros(len(primary),bool);bc=np.zeros(len(primary),bool)
    ac[visible]=np.square(centers[a][visible]-target[visible]).sum(-1)<=64.
    bc[visible]=np.square(centers[b][visible]-target[visible]).sum(-1)<=64.
    flip=a!=b
    return ac,bc,dict(n_query=len(a),n_point=int(visible.sum()),primary_correct=int(ac.sum()),size_correct=int(bc.sum()),
        flip_all_count=int(flip.sum()),flip_valid_count=int((flip&visible).sum()),
        size_benefit_count=int((~ac&bc&visible).sum()),size_harm_count=int((ac&~bc&visible).sum()))


def evaluate(out,scene):
    no_active_validation_benchmark(); lock=verify_lock(out); seal=require_seal(out,lock)
    directory=out/'evaluation'/scene.replace('/','__')
    if directory.exists():raise ValueError('refusing evaluation overwrite')
    directory.mkdir(parents=True); started=time.time()
    write(directory/'started.json',dict(created_unix=started,forward_complete_sha256=sha(out/'forward_complete.json')))
    entries=R.scene_entries(lock,scene); parts=R.read_parts(out)
    committed=read(out/'forward'/scene.replace('/','__')/'records.json'); population(committed,entries)
    records={identity(r):r for r in committed}; rows=[]; paired=[]; strata=[]
    for entry in entries:
        labels=load_labels(entry)  # First possible target bytes, after global seal.
        for c in CONDITIONS:
            q,k=c['query'],c['key']; identity_prefix=scene,tuple(entry['pair'])
            ref=records[(*identity_prefix,'trace_full',c['condition'])]
            with np.load(out/ref['archive_path'],allow_pickle=False) as archive:
                probs={role:archive[role] for role in ROLES}
            direct,common=R.S.target_counts(labels,parts,q,k)
            visible=labels[q+'_point_visible']; xy=labels[q+'_point_xy']
            ac,bc,pair=center_pair_metrics(probs['trace_full'],probs['size_temperature'],parts[k]['bounds'],xy,visible)
            pairrow=dict(scene=scene,pair=entry['pair'],**c,**pair)
            if pair['size_correct']-pair['primary_correct']!=pair['size_benefit_count']-pair['size_harm_count']:
                raise ValueError('paired PCK harm/benefit conservation failed')
            paired.append(pairrow)
            sizes=parts[q]['bounds'][:,2:]-parts[q]['bounds'][:,:2]
            for width,height in np.unique(sizes,axis=0):
                mask=np.all(sizes==[width,height],axis=-1); valid=mask&visible
                native={}
                for role,p in probs.items():
                    kl,js,mass=R.S.kl_js(p[mask],direct[mask])
                    native[role]=dict(direct_kl_sum=kl,direct_js_sum=js,mass=int(mass))
                strata.append(dict(scene=scene,pair=entry['pair'],**c,width=int(width),height=int(height),
                    n_query=int(mask.sum()),n_point=int(valid.sum()),primary_correct=int(ac[mask].sum()),
                    size_correct=int(bc[mask].sum()),size_benefit_count=int((~ac&bc&valid).sum()),
                    size_harm_count=int((ac&~bc&valid).sum()),
                    flip_all_count=int(((probs['trace_full'].argmax(-1)!=probs['size_temperature'].argmax(-1))&mask).sum()),
                    flip_valid_count=int(((probs['trace_full'].argmax(-1)!=probs['size_temperature'].argmax(-1))&valid).sum()),
                    native=native))
            for role,p in probs.items():
                record=records[(*identity_prefix,role,c['condition'])]
                if array_sha(p)!=record['array_sha256']:raise ValueError('saved probability changed before metrics')
                metric=R.measure(p,parts,c,direct,common)
                rows.append(dict(record,**metric,pck_count=int((ac if role=='trace_full' else bc).sum()),
                    n_point=int(visible.sum()),label_sha256=entry['label_sha256'],target_sha256=entry['target_sha256']))
        print('SIZE_ATTRIBUTION_EVALUATE',scene,entry['pair'],flush=True)
    population(rows,entries)
    write(directory/'rows.json',rows);write(directory/'paired_rows.json',paired);write(directory/'size_strata.json',strata)
    R.A.complete_stage(directory,out,'evaluation',scene,started,record_count=len(rows),runtime=R.runtime(),
        forward_complete_sha256=sha(out/'forward_complete.json'),labels_decoded=True)


def aggregate(rows):
    result={}
    for role in ROLES:
        scenes={}
        for scene in SCENES:
            conditions={}
            for c in CONDITIONS:
                selected=[r for r in rows if r['role']==role and r['scene']==scene and r['condition']==c['condition']]
                mass=sum(r['mass'] for r in selected); n=sum(r['n_point'] for r in selected); correct=sum(r['pck_count'] for r in selected)
                if len(selected)!=16 or mass<=0 or n<=0:raise ValueError('incomplete paired scene/condition population')
                conditions[c['condition']]=dict(**{f:fsum(r[f+'_sum'] for r in selected)/mass for f in ('kl','js','direct_kl','direct_js')},
                    mass=mass,n_point=n,pck_count=correct,pck=float(Fraction(correct,n)))
            scenes[scene]=conditions
        macro={}
        for c in CONDITIONS:
            values=[scenes[s][c['condition']] for s in SCENES]
            pck=sum((Fraction(v['pck_count'],v['n_point']) for v in values),Fraction())/3
            macro[c['condition']]=dict(**{f:fsum(v[f] for v in values)/3 for f in ('kl','js','direct_kl','direct_js')},
                pck=float(pck),pck_exact=dict(numerator=pck.numerator,denominator=pck.denominator))
        result[role]=dict(by_scene_condition=scenes,by_condition=macro,
            common9_kl=fsum(macro[c['condition']]['kl'] for c in CONDITIONS[:9])/9,
            common9_js=fsum(macro[c['condition']]['js'] for c in CONDITIONS[:9])/9,
            direct5_kl=fsum(macro[c['condition']]['kl'] for c in CONDITIONS[9:])/5,
            direct5_js=fsum(macro[c['condition']]['js'] for c in CONDITIONS[9:])/5)
        result[role]['J_descriptive']=.5*(result[role]['common9_kl']+result[role]['direct5_kl'])
    return result


def aggregate_pairs(rows,*,stratified=False):
    """Sum integers inside a scene, then equally average scene rates.

    Native-size groups never receive an invented share of a common-grid KL.
    Empty center populations remain None, not zero-accuracy observations.
    """
    groups={}
    for row in rows:
        name=row['condition']+(f"__{row['width']}x{row['height']}" if stratified else '')
        groups.setdefault(name,[]).append(row)
    additive=('n_query','n_point','primary_correct','size_correct','size_benefit_count',
              'size_harm_count','flip_all_count','flip_valid_count')
    result={}
    for name,group in groups.items():
        scene_values={};differences={}
        for scene in SCENES:
            subset=[r for r in group if r['scene']==scene]
            if len(subset)!=16:raise ValueError('paired/stratified scene cohort incomplete')
            sums={f:sum(r[f] for r in subset) for f in additive}
            if sums['size_correct']-sums['primary_correct']!=sums['size_benefit_count']-sums['size_harm_count']:
                raise ValueError('paired additive conservation changed')
            n=sums['n_point'];nq=sums['n_query']
            diff=Fraction(sums['size_correct']-sums['primary_correct'],n) if n else None
            differences[scene]=diff
            value=dict(sums,primary_pck=sums['primary_correct']/n if n else None,
                size_pck=sums['size_correct']/n if n else None,flip_all_rate=sums['flip_all_count']/nq,
                flip_valid_rate=sums['flip_valid_count']/n if n else None,
                size_minus_primary_pck_pp=float(100*diff) if diff is not None else None,
                pck_difference_exact=dict(numerator=diff.numerator,denominator=diff.denominator) if diff is not None else None)
            if stratified:
                native={}
                for role in ROLES:
                    mass=sum(r['native'][role]['mass'] for r in subset)
                    native[role]=dict(mass=mass,**{f:fsum(r['native'][role][f+'_sum'] for r in subset)/mass if mass else None
                                                   for f in ('direct_kl','direct_js')})
                value['native']=native
            scene_values[scene]=value
        defined=all(d is not None for d in differences.values())
        delta=sum(differences.values(),Fraction())/3 if defined else None
        macro=dict(size_minus_primary_pck_pp=float(100*delta) if defined else None,
            pck_difference_exact=dict(numerator=delta.numerator,denominator=delta.denominator) if defined else None,
            flip_all_rate=fsum(v['flip_all_rate'] for v in scene_values.values())/3,
            flip_valid_rate=fsum(v['flip_valid_rate'] for v in scene_values.values())/3 if defined else None,
            every_scene_has_center_population=defined)
        if stratified:
            macro['native']={role:{f:fsum(v['native'][role][f] for v in scene_values.values())/3
                if all(v['native'][role][f] is not None for v in scene_values.values()) else None
                for f in ('direct_kl','direct_js')} for role in ROLES}
        result[name]=dict(by_scene=scene_values,macro=macro)
    return result


def summarize_supports(rows,expected_queries=39888):
    """Input-only fixed bins, one identity per original query, never per key."""
    groups={}; seen=set()
    for row in rows:
        identity=row['scene'],tuple(row['pair']),row['query'],row['query_index']
        if identity in seen:raise ValueError('duplicate support query; key conditions cannot multiply its weight')
        seen.add(identity)
        group=row['scene'],row['query'],row['width'],row['height']
        if row['tau_size32']<=0:raise ValueError('fixed T10 size temperature must be positive')
        values={name:float(row[name]) for name in ('tau_obs32','tau_size32','observed_trace','native_trace')}
        values['tau_obs_over_tau_size']=values['tau_obs32']/values['tau_size32']
        if not np.isfinite(list(values.values())).all():raise ValueError('nonfinite support description')
        groups.setdefault(group,[]).append(values)
    if len(seen)!=expected_queries:raise ValueError('unique input support population incomplete')
    output=[]
    for (scene,q,width,height),values in sorted(groups.items()):
        statistics={name:dict(zip(('min','p05','median','p95','max'),np.quantile(
            [v[name] for v in values],[0,.05,.5,.95,1],method='linear').tolist())) for name in values[0]}
        output.append(dict(scene=scene,query=q,width=width,height=height,n=len(values),**statistics))
    return dict(unique_queries=len(seen),grouping=['scene','query','width','height'],
        quantile_method='linear',groups=output,input_only=True,key_condition_repetition=False,
        bins_selected_from_effects=False,significance_test=False)


def input_support_statistics(out,lock,parts):
    def unique_rows():
        for entry in lock['entries']:
            path=out/'forward'/entry['scene'].replace('/','__')/'arrays'/Path(entry['path']).name/'supports.npz'
            with np.load(path,allow_pickle=False) as archive:
                for q in ('g8','g16','mixed','g24'):
                    cov,bounds,observed,native,trace=(archive[q+'__'+field] for field in
                        ('cov','bounds','tau_obs32','tau_size32','native_trace'))
                    if (cov.dtype!=np.float32 or observed.dtype!=np.float32 or native.dtype!=np.float32
                            or bounds.dtype!=np.int64 or len(cov)!=len(parts[q]['bounds'])
                            or not np.array_equal(bounds,parts[q]['bounds'])):
                        raise ValueError('saved all-query support schema changed')
                    widths=bounds[:,2:]-bounds[:,:2]
                    observed_trace=np.trace(cov.astype(np.float64),axis1=-2,axis2=-1)
                    for index,(width,height) in enumerate(widths):
                        yield dict(scene=entry['scene'],pair=entry['pair'],query=q,query_index=index,
                            width=int(width),height=int(height),observed_trace=observed_trace[index],
                            native_trace=trace[index],tau_obs32=observed[index],tau_size32=native[index])
    return summarize_supports(unique_rows())


def summarize(out):
    no_active_validation_benchmark(); lock=verify_lock(out); require_seal(out,lock)
    if (out/'summary.json').exists() or (out/'completion.json').exists():raise ValueError('refusing summary overwrite')
    rows=[];paired=[];strata=[]
    for scene in SCENES:
        directory=out/'evaluation'/scene.replace('/','__'); done=R.A.verify_stage(directory,out,'evaluation',scene)
        if done['forward_complete_sha256']!=sha(out/'forward_complete.json'):raise ValueError('label phase seal changed')
        rows.extend(read(directory/'rows.json'));paired.extend(read(directory/'paired_rows.json'));strata.extend(read(directory/'size_strata.json'))
    population(rows,lock['entries']); metrics=aggregate(rows)
    write(out/'summary.json',dict(created_unix=time.time(),lock_sha256=sha(out/'lock.json'),
        forward_complete_sha256=sha(out/'forward_complete.json'),config=CONFIG,roles=list(ROLES),metrics=metrics,
        paired_summary=aggregate_pairs(paired),size_strata_summary=aggregate_pairs(strata,stratified=True),
        support_statistics=input_support_statistics(out,lock,R.read_parts(out)),
        matrices=len(rows),pairs=48,paired_conditions=len(paired),size_stratum_rows=len(strata),
        descriptive_attribution_only=True,scientific_gate_defined=False,independent_audit_pending=True,
        selection_changed=False,current_validation_changed=False,efficiency_claim=False))
    write(out/'completion.json',dict(finished_unix=time.time(),lock_sha256=sha(out/'lock.json'),
        summary_sha256=sha(out/'summary.json'),independent_audit_pending=True,
        files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('freeze','forward','seal-forward','evaluate','summarize'))
    parser.add_argument('--out',type=Path,default=ROOT/'results/round17_size_temperature_attribution')
    parser.add_argument('--regional',type=Path,default=ROOT/'results/round17_regional')
    parser.add_argument('--scene',choices=SCENES)
    args=parser.parse_args()
    if args.stage=='freeze':freeze(args.out,args.regional)
    elif args.stage=='seal-forward':seal_forward(args.out)
    elif args.stage in ('forward','evaluate'):globals()[args.stage](args.out,args.scene)
    else:summarize(args.out)


if __name__=='__main__':main()
