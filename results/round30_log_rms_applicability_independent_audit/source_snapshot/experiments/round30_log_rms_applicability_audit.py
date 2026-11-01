"""R30 input identity, delivery and log arithmetic audit; no probability reference.

Only the four authenticated R28 model fields are decoded and copied. Original
homogeneous geometry evidence is inherited, not reconstructed. Log arithmetic
is copied from the accepted R29 independent implementation without importing
its numerical integration, production model or preparation.
"""
from __future__ import annotations
import argparse
import math
from pathlib import Path
import shutil
import time
import numpy as np
from scipy.special import logsumexp, ndtr
from experiments import round30_log_rms_contract as C

SCOPE=dict(general_numerical_accuracy_established=False,independent_probability_reference=False,
           labels_read=False,efficacy_tested=False,candidate_selected=False,scientific_gate_passed=False,goal_complete=False)
DEFAULT_RUN=C.ROOT/'results/round30_log_rms_applicability'
DEFAULT_OUT=C.ROOT/'results/round30_log_rms_applicability_independent_audit'
PREPARED_NAMES=set(C.OLD_FIELDS)
LSE_SCOPES=('tile_nodes','cell_tiles','fov_tiles','native_total','floor','floor_total','floored_total','content_normalization')


def scope_check(value):
    if any(value.get(k) is not False for k in SCOPE):
        raise ValueError('all seven applicability-only scope flags must be explicit false')


def input_check(old,actual,spec):
    """Byte identity of inherited model inputs; no fresh camera transformation."""
    if set(old)!=set(C.OLD_FIELDS) or set(actual)!=PREPARED_NAMES:
        raise ValueError('exact four-field inherited input boundary required')
    shapes=dict(input__mu=(3,),input__factor=(3,3),input__bounds=(432,4),input__content_bias=(432,))
    for name,shape in shapes.items():
        for value in (old[name],actual[name]):
            if value.dtype!=np.float64 or value.shape!=shape or not np.isfinite(value).all():
                raise ValueError('finite float64 input dtype/shape differs: '+name)
        if C.array_sha(old[name])!=C.array_sha(actual[name]):
            raise ValueError('source input canonical bytes changed: '+name)
    if not np.array_equal(actual['input__bounds'],literal_bounds()) or np.any(actual['input__content_bias']!=0):
        raise ValueError('original native432 boundaries or exact zero bias changed')
    size,index=spec['query_size'],spec['query_index']
    if type(size) is not int or size not in (8,16,24) or type(index) is not int or not 0<=index<(144//size)*(192//size):
        raise ValueError('fixed query identity is outside its original native layout')
    x=(index%(192//size))*size;y=(index//(192//size))*size
    if spec['query_bounds']!=[x,y,x+size,y+size]:raise ValueError('query identity/bounds changed')
    return dict(passed=True,canonical_input_bytes_exact=True,native_bounds_exact=True,zero_content_bias=True,
        original_geometry_provenance_inherited=True,new_geometric_transformation_performed=False,
        real_inputs_read=True,**SCOPE)

def literal_bounds():
    pixel = np.asarray([(x,y,x+8,y+8) for y in range(0,144,8) for x in range(0,192,8)],np.float64)
    return (pixel-.5)/8.

def read_arrays(path, hashes, fields=None,on_decode=None):
    with np.load(path,allow_pickle=False) as archive:
        names=list(hashes) if fields is None else fields
        if fields is None and set(archive.files)!=set(hashes):
            raise ValueError('unexpected prepared/output archive fields')
        result={}
        for name in names:
            result[name]=archive[name]
            if on_decode is not None:on_decode()
    if {name:C.array_sha(value) for name,value in result.items()} != hashes:
        raise ValueError('authenticated array hashes differ')
    return result

def structural_branch(mu,factor):
    rms=math.hypot(float(mu[2]),*map(float,factor[2]))
    if rms==0:return 'zero_depth_rms',True
    if np.all(factor[2]==0):
        return ('nonpositive_constant_depth',True) if mu[2]<=0 else ('positive_constant_depth_gaussian',False)
    return 'variable_depth',False

def verify_producer_lock(run):
    lock=C.read(run/'lock.json')
    scope_check(lock)
    if C.sha(run/'lock.json')!=(run/'lock.sha256').read_text().strip():
        raise ValueError('producer lock bytes changed')
    bindings,specs=C.inherited_evidence()
    expected=dict(version=1,runtime=C.check_runtime(),policy=C.POLICY,gates=C.GATES,
        query_specs_sha256=C.sha(run/'query_specs.json'),case_ids=[s['id'] for s in specs],
        case_count=72,key_count=432,bindings=bindings,real_inputs_read=False,**SCOPE)
    if any(lock.get(k)!=v for k,v in expected.items()) or C.read(run/'query_specs.json')!=specs:
        raise ValueError('producer frozen population/provenance/runtime differs')
    if set(lock['source_hashes'])!=set(C.source_names()):
        raise ValueError('joint local source closure differs')
    for name,digest in lock['source_hashes'].items():
        if C.sha(C.ROOT/name)!=digest or C.sha(run/'source_snapshot'/name)!=digest:
            raise ValueError('current or producer source snapshot changed')
    return lock

def freeze(run,out):
    if out.exists() or any((run/n).exists() for n in ('prepared','prepared_complete.json','forward','started.json','completion.json')):
        raise ValueError('independent freeze must precede every prepared-archive decode')
    lock=verify_producer_lock(run)
    if Path(lock['audit_root']).resolve()!=out.resolve():
        raise ValueError('declared audit destination differs')
    out.mkdir(parents=True)
    for name,digest in lock['source_hashes'].items():
        target=out/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(C.ROOT/name,target)
        if C.sha(target)!=digest:raise ValueError('source changed while freezing independent snapshot')
    C.write(out/'lock.json',dict(created_unix=time.time(),run=str(run.resolve()),producer_root=str(run.resolve()),
        producer_lock_sha256=C.sha(run/'lock.json'),query_specs_sha256=lock['query_specs_sha256'],
        source_hashes=lock['source_hashes'],runtime=lock['runtime'],real_inputs_read=False,**SCOPE))
    (out/'lock.sha256').write_text(C.sha(out/'lock.json')+'\n')

def joint_locks(run,out):
    lock,design=verify_producer_lock(run),C.read(out/'lock.json')
    scope_check(design)
    expected=dict(run=str(run.resolve()),producer_root=str(run.resolve()),producer_lock_sha256=C.sha(run/'lock.json'),
        query_specs_sha256=lock['query_specs_sha256'],source_hashes=lock['source_hashes'],runtime=lock['runtime'],real_inputs_read=False,**SCOPE)
    if (C.sha(out/'lock.json')!=(out/'lock.sha256').read_text().strip()
            or any(design.get(k)!=v for k,v in expected.items())
            or not lock['created_unix']<=design['created_unix']<=time.time()):
        raise ValueError('joint independent lock differs')
    for name,digest in lock['source_hashes'].items():
        if C.sha(out/'source_snapshot'/name)!=digest:raise ValueError('independent source snapshot changed')
    return lock,design

def complete_inventory(directory):
    complete=C.read(directory/'completion.json')
    C.verify_inventory(directory,complete['files'],exclude=('completion.json',))
    return complete

def verify_prepared(run,out):
    lock,design=joint_locks(run,out)
    done=complete_inventory(run/'prepared')
    seal=C.read(run/'prepared_complete.json')
    rows=C.read(run/'prepared/records.json');specs=C.read(run/'query_specs.json')
    expected=dict(producer_lock_sha256=C.sha(run/'lock.json'),audit_lock_sha256=C.sha(out/'lock.json'),
        prepared_records_sha256=C.sha(run/'prepared/records.json'),
        prepared_completion_sha256=C.sha(run/'prepared/completion.json'),
        query_specs_sha256=lock['query_specs_sha256'],case_ids=[s['id'] for s in specs],case_count=72,
        predictions_executed=False,**SCOPE)
    if any(seal.get(k)!=v for k,v in expected.items()) or [r['id'] for r in rows]!=expected['case_ids']:
        raise ValueError('globally sealed ordered72 geometry outcomes required')
    start=C.read(run/'prepared/started.json')
    for stage in (start,seal,done):scope_check(stage)
    if (not design['created_unix']<=seal['started_unix']<=seal['finished_unix']
            or start['started_unix']!=seal['started_unix']
            or start['producer_lock_sha256']!=expected['producer_lock_sha256']
            or start['audit_lock_sha256']!=expected['audit_lock_sha256']
            or done['complete_case_inventory'] is not True
            or done['all_inputs_prepared']!=seal['all_inputs_prepared']
            or seal['all_inputs_prepared']!=all(r['passed'] for r in rows)
            or start['real_inputs_read'] is not False
            or seal['real_inputs_read'] is not any(r['real_inputs_read'] for r in rows)
            or done['real_inputs_read'] is not seal['real_inputs_read']):
        raise ValueError('preparation chronology/status differs')
    names={'started.json','records.json'}|{'arrays/'+s['id']+'.npz' for s in specs}
    if set(done['files'])!=names:raise ValueError('prepared file population differs')
    for row,spec in zip(rows,specs):
        scope_check(row)
        if (any(row.get(k)!=spec[k] for k in ('id','scene','pair','pair_name','query_size','query_index','query_bounds'))
                or row['archive_path']!='prepared/arrays/'+spec['id']+'.npz'
                or C.sha(run/row['archive_path'])!=row['archive_sha256']
                or row['source_archive_path']!=spec['source_archive_path']
                or row['source_archive_sha256']!=spec['source_archive_sha256']
                or type(row['passed']) is not bool or row['status']!=('prepared' if row['passed'] else 'failed')
                or type(row['real_inputs_read']) is not bool or (row['passed'] and not row['real_inputs_read'])
                or not math.isfinite(row['preparation_seconds']) or row['preparation_seconds']<0):
            raise ValueError('prepared identity/status/archive/time differs')
        if row['passed'] and set(row['array_sha256'])!=PREPARED_NAMES:
            raise ValueError('successful prepared hash schema differs')
        if not row['passed'] and not row.get('failure_reason'):
            raise ValueError('failed geometry requires explicit retained reason')
    return lock,seal,rows,specs

def _failure(directory,error,started,real_inputs_read=False):
    C.write(directory/'failure.json',dict(error_type=type(error).__name__,message=str(error),
        started_unix=started,finished_unix=time.time(),real_inputs_read=real_inputs_read,**SCOPE))
    C.write(directory/'completion.json',dict(passed=False,
        files={n:h for n,h in C.inventory(directory).items() if n!='completion.json'},
        finished_unix=time.time(),real_inputs_read=real_inputs_read,**SCOPE))

def input_audit(run,out):
    directory=out/'input_audit'
    if directory.exists() or (out/'input_verification.json').exists():
        raise ValueError('preserve every prior input audit attempt')
    directory.mkdir();started=time.time();C.write(directory/'started.json',dict(started_unix=started,real_inputs_read=False,**SCOPE))
    checks=[];decoded={'value':False}
    def note_decode():decoded['value']=True;row_decoded['value']=True
    try:
        lock,seal,rows,specs=verify_prepared(run,out)
        if started<seal['finished_unix'] or (run/'forward').exists() or (run/'started.json').exists():
            raise ValueError('all prepared inputs must be sealed before input audit and model calls')
        for row,spec in zip(rows,specs):
            row_decoded={'value':False}
            try:
                if not row['passed']:raise ValueError(row['failure_reason'])
                old_hashes={n:spec['array_sha256'][n] for n in C.OLD_FIELDS}
                if C.sha(spec['source_archive_path'])!=spec['source_archive_sha256']:
                    raise ValueError('bound old prepared archive changed')
                old=read_arrays(spec['source_archive_path'],old_hashes,fields=C.OLD_FIELDS,on_decode=note_decode)
                actual=read_arrays(run/row['archive_path'],row['array_sha256'],on_decode=note_decode)
                check=input_check(old,actual,spec)
            except Exception as error:
                check=dict(passed=False,failure_reason=type(error).__name__+': '+str(error),real_inputs_read=row_decoded['value'],**SCOPE)
            checks.append(dict(id=spec['id'],scene=spec['scene'],query_size=spec['query_size'],
                query_index=spec['query_index'],prepared_archive_sha256=row['archive_sha256'],**check))
            C.write(directory/'checks.json',checks)
        joint_locks(run,out)
        passed=bool(len(checks)==72 and all(r['passed'] for r in checks))
        verification=dict(passed=passed,checked_queries=len(checks),producer_lock_sha256=C.sha(run/'lock.json'),
            audit_lock_sha256=C.sha(out/'lock.json'),producer_prepared_complete_sha256=C.sha(run/'prepared_complete.json'),
            producer_prepared_completion_sha256=C.sha(run/'prepared/completion.json'),query_specs_sha256=lock['query_specs_sha256'],
            source_prepared_hashes={s['source_archive_path']:s['source_archive_sha256'] for s in specs},
            all_input_bytes_exact=bool(passed and all(r['canonical_input_bytes_exact'] for r in checks)),
            real_inputs_read=decoded['value'],started_unix=started,finished_unix=time.time(),**SCOPE)
        C.write(out/'input_verification.json',verification)
        C.write(directory/'completion.json',dict(passed=passed,files=C.inventory(directory),
            verification_sha256=C.sha(out/'input_verification.json'),finished_unix=time.time(),real_inputs_read=decoded['value'],**SCOPE))
        if not passed:raise RuntimeError('independent input audit failed; no forward authorized')
        return verification
    except (Exception,KeyboardInterrupt) as error:
        if not (directory/'completion.json').exists():_failure(directory,error,started,decoded['value'])
        raise

def verify_input_audit(run,out):
    lock,prepared,rows,specs=verify_prepared(run,out)
    done=complete_inventory(out/'input_audit');verification=C.read(out/'input_verification.json')
    expected=dict(passed=True,checked_queries=72,producer_lock_sha256=C.sha(run/'lock.json'),
        audit_lock_sha256=C.sha(out/'lock.json'),producer_prepared_complete_sha256=C.sha(run/'prepared_complete.json'),
        producer_prepared_completion_sha256=C.sha(run/'prepared/completion.json'),query_specs_sha256=lock['query_specs_sha256'],
        source_prepared_hashes={s['source_archive_path']:s['source_archive_sha256'] for s in specs},
        all_input_bytes_exact=True,real_inputs_read=True,**SCOPE)
    for stage in (verification,done):scope_check(stage)
    if (done.get('passed') is not True or done['verification_sha256']!=C.sha(out/'input_verification.json')
            or any(verification.get(k)!=v for k,v in expected.items())
            or not prepared['finished_unix']<=verification['started_unix']<=verification['finished_unix']):
        raise ValueError('complete independent input audit required before model outcomes')
    checks=C.read(out/'input_audit/checks.json')
    if ([r['id'] for r in checks]!=[s['id'] for s in specs] or not all(r['passed'] and r['canonical_input_bytes_exact'] and r['real_inputs_read'] for r in checks)
            or done['real_inputs_read'] is not True):
        raise ValueError('all72 input checks must pass')
    return lock,prepared,rows,specs,verification

def verify_forward(run,out):
    lock,prepared,prepared_rows,specs,input_verification=verify_input_audit(run,out)
    forward=run/'forward';done=complete_inventory(forward)
    root_done=complete_inventory(run);seal=C.read(run/'forward_complete.json')
    rows=C.read(forward/'records.json')
    expected=dict(producer_lock_sha256=C.sha(run/'lock.json'),audit_lock_sha256=C.sha(out/'lock.json'),
        producer_prepared_complete_sha256=C.sha(run/'prepared_complete.json'),
        input_verification_sha256=C.sha(out/'input_verification.json'),query_specs_sha256=lock['query_specs_sha256'],
        forward_records_sha256=C.sha(forward/'records.json'),forward_completion_sha256=C.sha(forward/'completion.json'),
        case_ids=[s['id'] for s in specs],case_count=72,all_case_attempts_retained=True,
        all_available_partial_arrays_retained=True,**SCOPE)
    if any(seal.get(k)!=v for k,v in expected.items()) or [r['id'] for r in rows]!=expected['case_ids']:
        raise ValueError('complete ordered72 forward seal required')
    start=C.read(forward/'started.json')
    for stage in (seal,start,done,root_done):scope_check(stage)
    if (start['started_unix']!=seal['started_unix']
            or start['producer_lock_sha256']!=expected['producer_lock_sha256']
            or start['audit_lock_sha256']!=expected['audit_lock_sha256']
            or start['input_verification_sha256']!=expected['input_verification_sha256']
            or not input_verification['finished_unix']<=seal['started_unix']<=seal['phase_finished_unix']<=seal['finished_unix']
            or not math.isfinite(seal['phase_wall_seconds']) or seal['phase_wall_seconds']<0
            or seal['phase_wall_passed']!=(seal['phase_wall_seconds']<=lock['gates']['production_max_phase_wall_seconds'])
            or done['phase_wall_passed']!=seal['phase_wall_passed']
            or root_done['forward_complete_sha256']!=C.sha(run/'forward_complete.json')
            or start['real_inputs_read'] is not False
            or seal['real_inputs_read'] is not any(r['real_inputs_read'] for r in rows)
            or done['real_inputs_read'] is not seal['real_inputs_read'] or root_done['real_inputs_read'] is not seal['real_inputs_read']):
        raise ValueError('forward source/input-audit chronology or phase metadata differs')
    forward_files={'started.json','records.json','call_starts.jsonl','model_results.jsonl'}
    forward_files|={'arrays/'+s['id']+'.npz' for s in specs}
    if set(done['files'])!=forward_files:raise ValueError('forward inventory population differs')
    expected_root={'lock.json','lock.sha256','query_specs.json','prepared_complete.json','forward_complete.json'}
    expected_root|={'source_snapshot/'+n for n in lock['source_hashes']}
    expected_root|={'prepared/'+n for n in C.read(run/'prepared/completion.json')['files']}
    expected_root|={'forward/'+n for n in forward_files}|{'prepared/completion.json','forward/completion.json'}
    if set(root_done['files'])!=expected_root:raise ValueError('producer complete inventory membership differs')
    import json
    starts=[json.loads(line) for line in (forward/'call_starts.jsonl').read_text().splitlines()]
    journal=[json.loads(line) for line in (forward/'model_results.jsonl').read_text().splitlines()]
    if ([s['id'] for s in starts]!=[r['id'] for r in rows if r['attempted']]
            or [r['id'] for r in journal]!=expected['case_ids']
            or seal['attempted_calls']!=len(starts)):
        raise ValueError('actual-call and retained-result journals differ')
    start_map={s['id']:s for s in starts}
    for item in starts+journal:scope_check(item)
    if not all(s['real_inputs_read'] is True for s in starts):raise ValueError('actual model calls must follow real input reads')
    previous=seal['started_unix']
    for row,prep,spec,record in zip(rows,prepared_rows,specs,journal):
        scope_check(row)
        if (any(row.get(k)!=spec[k] for k in ('id','scene','pair','pair_name','query_size','query_index','query_bounds'))
                or row['prepared_archive_path']!=prep['archive_path']
                or row['prepared_archive_sha256']!=prep['archive_sha256']
                or row['archive_path']!='forward/arrays/'+spec['id']+'.npz'
                or C.sha(run/row['archive_path'])!=row['archive_sha256']
                or not previous<=row['started_unix']<=row['finished_unix']<=seal['phase_finished_unix']):
            raise ValueError('fixed forward input/output identity or chronology differs')
        previous=row['finished_unix']
        if row['attempted'] and not row['started_unix']<=start_map[row['id']]['started_unix']<=row['finished_unix']:
            raise ValueError('actual model-start timestamp differs')
        if (record['available_array_sha256']!=row['array_sha256']
                or any(row.get(k)!=v for k,v in record.items() if k!='available_array_sha256')):
            raise ValueError('result work/status/available-array journal changed')
    return lock,seal,rows,prepared_rows,specs

def validate_forward_row(row,bank,prepared,policy):
    scope_check(row)
    if (type(row.get('passed')) is not bool or type(row.get('attempted')) is not bool
            or type(row.get('real_inputs_read')) is not bool
            or row.get('status')!=('complete' if row['passed'] else 'failed')
            or not isinstance(row.get('work'),dict)
            or any(type(v) is not int or v<0 for v in row['work'].values())):
        raise ValueError('explicit coherent call outcome and integer work required')
    if not row['passed'] and (not isinstance(row.get('failure_reason'),str) or not row['failure_reason'].strip()):
        raise ValueError('failed call requires a retained reason')
    for key in ('wall_seconds','call_and_input_load_seconds'):
        if not isinstance(row.get(key),(float,int)) or isinstance(row[key],bool) or not math.isfinite(row[key]) or row[key]<0:
            raise ValueError('finite nonnegative model/call duration required')
    if type(row.get('peak_rss_kib_process')) is not int or row['peak_rss_kib_process']<0:
        raise ValueError('process RSS high-water metadata required')
    inputs={'input__mu','input__factor','input__bounds','input__content_bias'}
    present=set(bank)&inputs
    if present and present!=inputs:raise ValueError('model input copies must be complete')
    if (present or row['attempted']) and not row['real_inputs_read']:raise ValueError('retained inputs or actual call require truthful real-input read flag')
    for name in present:
        if C.array_sha(bank[name])!=C.array_sha(prepared[name]):raise ValueError('model input copy differs from audited geometry')
    success={'log_raw_mass','log_raw_fov_mass','log_floored_mass','log_probability','probability',
             'materialized_raw_indices','materialized_raw_mass','argmax'}
    partial={'partial_log_raw_mass','partial_log_raw_fov_mass','partial_completed_cells','partial_completed_fov',
             'partial_accumulated_tiles','partial_accumulated_fov_tiles'}
    outputs=set(bank)-inputs
    if row['passed']:
        if not row['attempted'] or present!=inputs or outputs!=success:
            raise ValueError('successful call requires four actual inputs and complete outputs')
    elif not outputs<=partial:
        raise ValueError('failed output must contain available partials only')
    if not row['attempted'] and (outputs or any(row['work'].values())):
        raise ValueError('unattempted call must not invent partial outputs or work')
    for name in outputs:
        if name in ('materialized_raw_indices','materialized_raw_mass'):
            dtype=np.int64 if name=='materialized_raw_indices' else np.float64
            if bank[name].dtype!=dtype or bank[name].ndim!=1:raise ValueError('indexed ordinary materialization schema differs')
            continue
        dtype=np.int32 if name=='argmax' else (np.bool_ if name.startswith('partial_completed_') else (np.int64 if name.startswith('partial_accumulated_') else np.float64))
        shape=() if name=='argmax' else ((1,) if name in ('log_raw_fov_mass','partial_log_raw_fov_mass','partial_completed_fov','partial_accumulated_fov_tiles') else (432,))
        if bank[name].dtype!=dtype or bank[name].shape!=shape:
            raise ValueError('model output dtype/shape differs: '+name)
    if row['work'].get('cell_node_evaluations',0)>policy['max_cell_node_evaluations']:
        raise ValueError('retained per-call node count exceeds fixed budget')
    if row['passed'] and int(bank['argmax'])!=int(bank['probability'].argmax()):
        raise ValueError('saved first-index winner differs')
    partial_delivery=partial_check(row,bank,policy) if not row['passed'] and outputs else None
    return dict(delivery_integrity_passed=True,attempted=row['attempted'],model_completed=row['passed'],partial_delivery=partial_delivery,
        producer_real_inputs_read=row['real_inputs_read'])

def audit(run,out):
    if any((out/n).exists() for n in ('audit_started.json','verification.json','summary.json','completion.json')):
        raise ValueError('preserve every prior final audit attempt')
    started=time.time();C.write(out/'audit_started.json',dict(started_unix=started,real_inputs_read=False,**SCOPE))
    checks=[];decoded={'value':False}
    def note_decode():decoded['value']=True;row_decoded['value']=True
    try:
        lock,seal,rows,prepared_rows,specs=verify_forward(run,out)
        if started<seal['finished_unix']:raise ValueError('final audit must follow complete global forward seal')
        for row,prep,spec in zip(rows,prepared_rows,specs):
            row_decoded={'value':False}
            try:
                prepared=read_arrays(run/prep['archive_path'],prep['array_sha256'],on_decode=note_decode)
                bank=read_arrays(run/row['archive_path'],row['array_sha256'],on_decode=note_decode)
                check=validate_forward_row(row,bank,prepared,lock['policy'])
                if row['passed']:
                    algebra=production_check(row,bank,lock['policy'],lock['gates'])
                    check.update(algebraic_output_check=algebra,algebraic_output_check_passed=algebra['passed'])
                else:
                    check.update(algebraic_output_check_passed=False,failure_reason=row['failure_reason'],
                        partial_array_names=sorted(set(bank)-{'input__mu','input__factor','input__bounds','input__content_bias'}))
            except Exception as error:
                check=dict(delivery_integrity_passed=False,algebraic_output_check_passed=False,
                    failure_reason=type(error).__name__+': '+str(error))
            checks.append(dict(id=spec['id'],scene=spec['scene'],query_size=spec['query_size'],query_index=spec['query_index'],**check,real_inputs_read=row_decoded['value'],**SCOPE))
            C.write(out/'checks.json',checks)
        joint_locks(run,out)
        complete=bool(len(checks)==72 and all(r['delivery_integrity_passed'] for r in checks))
        applicability=bool(complete and seal['phase_wall_passed'] and all(r['passed'] for r in rows)
                           and all(r['algebraic_output_check_passed'] for r in checks))
        if (seal['applicability_calls_completed']!=bool(seal['phase_wall_passed'] and all(r['passed'] for r in rows))):
            raise ValueError('producer applicability flag differs from its statuses and original phase budget')
        summary=dict(delivery_integrity_passed=complete,applicability_calls_completed=applicability,
            checked_queries=len(checks),model_completed_calls=sum(r['passed'] for r in rows),
            model_failed_calls=sum(not r['passed'] for r in rows),attempted_calls=sum(r['attempted'] for r in rows),
            calls_not_attempted=sum(not r['attempted'] for r in rows),
            output_audit_failures=sum(not r['delivery_integrity_passed'] or (row['passed'] and not r['algebraic_output_check_passed'])
                                      for r,row in zip(checks,rows)),
            phase_wall_seconds=seal['phase_wall_seconds'],phase_wall_passed=seal['phase_wall_passed'],
            producer_cell_node_evaluations=sum(r['work'].get('cell_node_evaluations',0) for r in rows),
            started_unix=started,finished_unix=time.time(),real_inputs_read=decoded['value'],**SCOPE)
        C.write(out/'summary.json',summary)
        verification=dict(delivery_integrity_passed=complete,applicability_calls_completed=applicability,
            producer_lock_sha256=C.sha(run/'lock.json'),audit_lock_sha256=C.sha(out/'lock.json'),
            producer_prepared_complete_sha256=C.sha(run/'prepared_complete.json'),
            input_verification_sha256=C.sha(out/'input_verification.json'),
            producer_forward_complete_sha256=C.sha(run/'forward_complete.json'),
            producer_records_sha256=C.sha(run/'forward/records.json'),producer_completion_sha256=C.sha(run/'completion.json'),
            summary_sha256=C.sha(out/'summary.json'),checked_queries=len(checks),finished_unix=time.time(),real_inputs_read=decoded['value'],**SCOPE)
        C.write(out/'verification.json',verification)
        C.write(out/'completion.json',dict(files=C.inventory(out),verification_sha256=C.sha(out/'verification.json'),
            delivery_integrity_passed=complete,applicability_calls_completed=applicability,finished_unix=time.time(),real_inputs_read=decoded['value'],**SCOPE))
        return verification
    except (Exception,KeyboardInterrupt) as error:
        _failure(out,error,started,decoded['value'])
        raise

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('freeze','input-audit','audit'))
    parser.add_argument('--run',type=Path,default=DEFAULT_RUN)
    parser.add_argument('--out',type=Path,default=DEFAULT_OUT)
    args=parser.parse_args()
    {'freeze':freeze,'input-audit':input_audit,'audit':audit}[args.command](args.run.resolve(),args.out.resolve())

def probabilities(log_raw, bounds, bias, floor_density):
    """Independent once-only native area floor and bias in log coordinates."""
    log_raw,bounds,bias=map(np.asarray,(log_raw,bounds,bias))
    n=len(bounds)
    if (log_raw.dtype!=np.float64 or log_raw.shape!=(n,) or np.isnan(log_raw).any()
            or np.isposinf(log_raw).any() or bounds.dtype!=np.float64 or bounds.shape!=(n,4)
            or not np.isfinite(bounds).all() or bias.dtype!=np.float64 or bias.shape!=(n,)
            or not np.isfinite(bias).all() or not math.isfinite(floor_density) or floor_density<=0):
        raise ValueError('valid canonical log raw mass, native bounds and finite bias required')
    sides=bounds[:,2:]-bounds[:,:2]
    if np.any(sides<=0):raise ValueError('positive native rectangle sides required')
    floor=np.prod(sides,axis=1)*floor_density
    if not np.isfinite(floor).all() or np.any(floor<=0):
        raise FloatingPointError('original positive finite ordinary area floor product required')
    log_floor=np.log(floor)
    log_floored=np.logaddexp(log_raw,log_floor)
    logits=log_floored+bias;log_z=float(logsumexp(logits))
    if not np.isfinite(log_floored).all() or not math.isfinite(log_z):
        raise FloatingPointError('nonfinite independent log floor or normalizer')
    log_p=logits-log_z;p=np.exp(log_p)
    return dict(log_floored_mass=log_floored,log_probability=log_p,probability=p,
        log_floor_mass=log_floor,log_content_normalizer=log_z)

def finite_or_none(value):
    return float(value) if math.isfinite(float(value)) else None

def exp_diagnostic(value):
    if value==-np.inf:return 0.
    if value>math.log(np.finfo(np.float64).max):return None
    return math.exp(float(value))

def log_absolute_difference(a,b):
    """Log |exp(a)-exp(b)| without ordinary raw materialization."""
    a,b=np.broadcast_arrays(np.asarray(a,np.float64),np.asarray(b,np.float64))
    if np.isnan(a).any() or np.isnan(b).any() or np.isposinf(a).any() or np.isposinf(b).any():
        raise ValueError('canonical log masses may be finite or exact negative infinity only')
    high=np.maximum(a,b);result=np.full(a.shape,-np.inf,np.float64)
    different=a!=b
    result[different]=high[different]+np.log(-np.expm1(-np.abs(a[different]-b[different])))
    return result

def validate_logs(log_raw,structural_zero):
    log_raw=np.asarray(log_raw)
    if log_raw.dtype!=np.float64 or log_raw.ndim!=1 or not len(log_raw):
        raise ValueError('nonempty float64 canonical native log mass required')
    if structural_zero:
        if not np.isneginf(log_raw).all():raise ValueError('structural zero must have exact -inf log mass')
    elif not np.isfinite(log_raw).all():
        raise ValueError('positive nonstructural raw mass must have finite canonical logs')
    return log_raw

def direct_fov_check(log_raw,log_direct,structural_zero,absolute_gate):
    raw=validate_logs(log_raw,structural_zero);direct=np.asarray(log_direct)
    if direct.dtype!=np.float64 or direct.shape!=(1,):raise ValueError('one canonical direct FOV log required')
    validate_logs(direct,structural_zero)
    error=float(log_absolute_difference(float(logsumexp(raw)),float(direct[0])))
    return dict(passed=bool(error<=math.log(absolute_gate)),log_absolute_error=finite_or_none(error),
        absolute_error_materialized_diagnostic=exp_diagnostic(error))

def materialization_check(bank,structural_zero):
    raw=validate_logs(bank['log_raw_mass'],structural_zero)
    with np.errstate(under='ignore',over='ignore'):ordinary=np.exp(raw)
    if not np.isfinite(ordinary).all() or not np.isfinite(bank['materialized_raw_mass']).all():
        raise ValueError('ordinary materialization overflow is a failed admission, never valid packed output')
    indices=np.arange(len(raw),dtype=np.int64) if structural_zero else np.flatnonzero(ordinary>0)
    values=np.zeros(len(raw),np.float64) if structural_zero else ordinary[indices]
    if (bank['materialized_raw_indices'].dtype!=np.int64 or bank['materialized_raw_mass'].dtype!=np.float64
            or not np.array_equal(bank['materialized_raw_indices'],indices)
            or not np.array_equal(bank['materialized_raw_mass'],values)):
        raise ValueError('indexed ordinary materialization must retain exactly exp(lograw)>0, or structural zeros')
    return dict(ordinary_raw_unmaterialized_count=0 if structural_zero else len(raw)-len(indices),
        ordinary_raw_subnormal_count=int(np.count_nonzero((values>0)&(values<np.finfo(np.float64).tiny))))

def local_lse_diagnostic(terms):
    """Reconstruct local-loss metadata for saved log inputs, not density nodes."""
    terms=np.asarray(terms,np.float64)
    if terms.ndim==1:terms=terms[None,:]
    maximum=np.max(terms,axis=-1,keepdims=True);finite=np.isfinite(terms)
    with np.errstate(invalid='ignore',under='ignore'):
        shifted=np.where(finite,terms-maximum,-np.inf);scaled=np.exp(shifted)
    lost=finite&(scaled==0);counts=np.sum(lost,axis=-1);affected=counts>0
    result=dict(sum_count=len(terms),underflow_sum_count=int(affected.sum()),underflow_term_count=int(counts.sum()),
        maximum_log_scaled_omitted_bound=None,maximum_log_absolute_omitted_bound=None,maximum_log_relative_omitted_bound=None)
    if np.any(affected):
        biggest=np.max(np.where(lost,shifted,-np.inf),axis=-1)[affected]
        bound=np.log(counts[affected])+biggest
        retained=np.asarray([math.fsum(map(float,row)) for row in scaled[affected]])
        result.update(maximum_log_scaled_omitted_bound=float(bound.max()),
            maximum_log_absolute_omitted_bound=float((maximum[:,0][affected]+bound).max()),
            maximum_log_relative_omitted_bound=float((bound-np.log(retained)).max()))
    return result

def lse_check(row,bank,zero,native_tiles,fov_tiles,tolerance):
    report=row['lse_diagnostics'];scopes=report['by_scope']
    if (set(scopes)!=set(LSE_SCOPES) or report['exact_arithmetic_local_omission_only'] is not True
            or report['is_pipeline_error_bound'] is not False
            or report['excludes_addition_absorption_density_quadrature_and_roundoff'] is not True):
        raise ValueError('LSE diagnostics must retain local-only scope')
    n=len(bank['log_raw_mass']);counts=dict(tile_nodes=native_tiles+fov_tiles,cell_tiles=native_tiles,fov_tiles=fov_tiles,
        native_total=1,floor=n,floor_total=1,floored_total=1,content_normalization=1)
    sizes=dict(tile_nodes=576,cell_tiles=2,fov_tiles=2,native_total=n,floor=2,floor_total=n,floored_total=n,content_normalization=n)
    fields={'sum_count','underflow_sum_count','underflow_term_count','maximum_log_scaled_omitted_bound',
            'maximum_log_absolute_omitted_bound','maximum_log_relative_omitted_bound'}
    for name,r in scopes.items():
        if set(r)!=fields or r['sum_count']!=counts[name] or any(type(r[k]) is not int or r[k]<0 for k in ('sum_count','underflow_sum_count','underflow_term_count')):
            raise ValueError('local positive sum counts differ')
        if not (0<=r['underflow_sum_count']<=r['sum_count'] and r['underflow_sum_count']<=r['underflow_term_count']<=r['underflow_sum_count']*(sizes[name]-1)):
            raise ValueError('impossible positive-sum underflow counts')
        bounds=[r[k] for k in ('maximum_log_scaled_omitted_bound','maximum_log_absolute_omitted_bound','maximum_log_relative_omitted_bound')]
        if not r['underflow_term_count']:
            if any(v is not None for v in bounds):raise ValueError('no lost terms requires null local bounds')
        elif not all(isinstance(v,(float,int)) and math.isfinite(v) for v in bounds):
            raise ValueError('lost positive terms require finite local log bounds')
        elif bounds[2]>bounds[0]+tolerance or bounds[0]>math.log(max(1,sizes[name]-1))+math.log(np.nextafter(0.,1.))+tolerance:
            raise ValueError('local omitted bound incompatible with exp-underflow event')
    floor=np.prod(bank['input__bounds'][:,2:]-bank['input__bounds'][:,:2],axis=1)*row['floor_density']
    logfloor=np.log(floor)
    saved_inputs=dict(native_total=bank['log_raw_mass'],floor=np.column_stack((bank['log_raw_mass'],logfloor)),
        floor_total=logfloor,floored_total=bank['log_floored_mass'],
        content_normalization=bank['log_floored_mass']+bank['input__content_bias'])
    for name,terms in saved_inputs.items():
        expected=local_lse_diagnostic(terms)
        for key,value in expected.items():
            actual=scopes[name][key]
            if value is None or isinstance(value,int):
                if actual!=value:raise ValueError('saved-input LSE local loss diagnostic differs: '+name+'/'+key)
            elif actual is None or abs(actual-value)>tolerance:
                raise ValueError('saved-input LSE local log bound differs: '+name+'/'+key)
    return dict(passed=True,saved_input_scopes_independently_recomputed=list(saved_inputs),
        node_and_tile_scopes_not_numerically_replayed=['tile_nodes','cell_tiles','fov_tiles'],
        local_omitted_bound_is_pipeline_certificate=False)

def production_check(row,bank,policy,gates):
    if row['passed'] is not True:return dict(passed=False,status='production_failed',failure_reason=row.get('failure_reason'))
    mu,factor,bounds,bias=[bank['input__'+n] for n in ('mu','factor','bounds','content_bias')]
    branch,zero=structural_branch(mu,factor);raw=validate_logs(bank['log_raw_mass'],zero)
    expected=probabilities(raw,bounds,bias,policy['floor_density'])
    p=bank['probability'];logp=bank['log_probability'];lf=bank['log_floored_mass']
    if (not np.isfinite(p).all() or np.any(p<0) or not np.isfinite(logp).all() or not np.isfinite(lf).all()):
        raise ValueError('finite completed log floor, log probability and ordinary probability required')
    rms=math.hypot(float(mu[2]),*map(float,factor[2]));m=None if rms==0 else float(mu[2]/rms)
    sigma=None if rms==0 else math.hypot(*map(float,factor[2]/rms))
    front=0. if zero else (1. if sigma==0 else float(ndtr(m/sigma)))
    fov=np.r_[bounds[:,:2].min(0),bounds[:,2:].max(0)];maxside=policy['h']*policy['maximum_tile_side_in_h']
    native=0 if zero else sum(int(np.prod(np.ceil((box[2:]-box[:2])/maxside).astype(np.int64))) for box in bounds)
    fv=0 if zero else int(np.prod(np.ceil((fov[2:]-fov[:2])/maxside).astype(np.int64)))
    work=dict(cell_node_evaluations=(native+fv)*24**2,native_tiles_completed=native,fov_tiles_completed=fv)
    material=materialization_check(bank,zero);direct=direct_fov_check(raw,bank['log_raw_fov_mass'],zero,gates['direct_fov_additivity_absolute'])
    diag=row['density_diagnostics'];series_keys=('maximum_log_negative_tail_relative_remainder','minimum_scaled_negative_tail_series_lower_bound','maximum_scaled_negative_tail_series_upper_bound')
    nodekeys=('constant_depth_nodes','positive_branch_nodes','negative_branch_nodes','negative_erfcx_nodes','negative_series_nodes')
    if any(type(diag[k]) is not int or diag[k]<0 for k in nodekeys):raise ValueError('density branch counts invalid')
    if diag['negative_series_nodes']:
        rel,lower,upper=[diag[k] for k in series_keys]
        series=all(math.isfinite(v) for v in (rel,lower,upper)) and 0<lower<=upper and rel<=math.log(policy['max_negative_tail_relative_remainder'])
    else:series=all(diag[k] is None for k in series_keys)
    density_range=(diag['maximum_log_density'] is None and diag['minimum_log_density'] is None) if zero else (
        math.isfinite(diag['maximum_log_density']) and math.isfinite(diag['minimum_log_density']) and diag['minimum_log_density']<=diag['maximum_log_density'])
    bound=0. if zero else float(np.prod(bounds[:,2:]-bounds[:,:2],axis=1).sum())/(math.pi*policy['h']**2*math.comb(48,24))
    trunc=row['analytic_truncation_bound'];tol=gates['normalization_absolute']
    logsum=None if zero else float(logsumexp(raw));logfloor=float(logsumexp(expected['log_floor_mass']))
    metadata=bool(row['structural_branch']==branch and row['structural_zero'] is zero and series and density_range
        and np.isclose(row['front_mass'],front,rtol=2e-13,atol=0) and row['rms_depth']==rms
        and row['normalized_depth_mean']==m and row['normalized_depth_std']==sigma
        and row['quadrature_order']==24 and row['maximum_tile_side_in_h']==2. and row['h']==policy['h'] and row['floor_density']==policy['floor_density']
        and row['projection_regularization']=='rms_depth_xy_gaussian' and row['canonical_raw_representation']=='log_mass'
        and row['ordinary_materialization_is_canonical'] is False and row['tail_truncation_used'] is False
        and row['quadrature_error_bound_available'] is False and row['exact_arithmetic_gl_bound_available'] is True
        and row['native_tile_count']==native and row['fov_tile_count']==fv and row['work']==work
        and trunc['exact_arithmetic_only'] is True and trunc['excludes_density_evaluation_and_roundoff'] is True
        and trunc['is_relative_error_certificate'] is False and trunc['raw_native_l1_absolute']==bound and trunc['raw_direct_fov_absolute']==bound
        and diag['constant_depth_nodes']+diag['positive_branch_nodes']+diag['negative_branch_nodes']==work['cell_node_evaluations']
        and diag['negative_erfcx_nodes']+diag['negative_series_nodes']==diag['negative_branch_nodes']
        and (row['log_raw_mass_sum'] is None if zero else (isinstance(row['log_raw_mass_sum'],(float,int)) and abs(row['log_raw_mass_sum']-logsum)<=tol))
        and abs(row['log_total_area_floor']-logfloor)<=tol and abs(row['log_normalizer']-float(logsumexp(lf)))<=tol
        and abs(row['log_content_normalizer']-expected['log_content_normalizer'])<=tol
        and row['ordinary_raw_unmaterialized_count']==material['ordinary_raw_unmaterialized_count']
        and row['ordinary_raw_subnormal_count']==material['ordinary_raw_subnormal_count']
        and row['zero_raw_cells']==(len(raw) if zero else 0) and row['all_raw_mass_zero'] is zero
        and row['area_prior_used'] is bool(zero and np.all(bias==bias[0]))
        and row['raw_mass_below_total_floor'] is bool(zero or logsum<logfloor)
        and row['argmax_index']==int(p.argmax()) and row['argmax_exact_ties']==int(np.count_nonzero(p==p.max()))
        and row['argmax_near_ties_at_1e_12']==int(np.count_nonzero(p.max()-p<=1e-12)))
    local=lse_check(row,bank,zero,native,fv,tol)
    norm=abs(math.fsum(map(float,p))-1.);pe=float(np.max(abs(p-expected['probability'])))
    lfe=float(np.max(abs(lf-expected['log_floored_mass'])));lpe=float(np.max(abs(logp-expected['log_probability'])))
    with np.errstate(under='ignore'):saved_exp_error=float(np.max(abs(np.exp(logp)-p)))
    passed=bool(metadata and direct['passed'] and local['passed'] and work['cell_node_evaluations']<=policy['max_cell_node_evaluations']
        and max(norm,pe,lfe,lpe,saved_exp_error)<=tol)
    return dict(passed=passed,status='checked',metadata_consistent=metadata,structural_branch=branch,structural_zero=zero,
        normalization_error=norm,same_raw_probability_error=pe,log_floor_arithmetic_error=lfe,log_probability_arithmetic_error=lpe,
        saved_probability_exp_error=saved_exp_error,direct_fov=direct,lse_diagnostic_check=local,**material,
        log_raw_mass_sum=logsum,log_total_area_floor=logfloor,raw_below_total_floor=bool(zero or logsum<logfloor),
        all_raw_mass_zero=zero,exact_maximizers=np.flatnonzero(p==p.max()).tolist(),work_expected=work)

def partial_check(row,bank,policy):
    names={n for n in bank if not n.startswith('input__')}
    expected={'partial_log_raw_mass','partial_log_raw_fov_mass','partial_completed_cells','partial_completed_fov',
              'partial_accumulated_tiles','partial_accumulated_fov_tiles'}
    if not names:return dict(passed=True,available_partial_arrays=[])
    if names!=expected:raise ValueError('returned partial log integration requires all masks and tile counts')
    for logkey,countkey,maskkey in (('partial_log_raw_mass','partial_accumulated_tiles','partial_completed_cells'),
                                   ('partial_log_raw_fov_mass','partial_accumulated_fov_tiles','partial_completed_fov')):
        logs,counts,mask=bank[logkey],bank[countkey],bank[maskkey]
        if np.isnan(logs).any() or np.isposinf(logs).any() or np.any(counts<0):raise ValueError('invalid partial log sum/count')
        if not np.array_equal(np.isfinite(logs),counts>0):raise ValueError('partial finite sums must distinguish accumulated from uncomputed cells')
        if np.any(mask&(counts==0)) and not row.get('structural_zero',False):raise ValueError('positive completed cell needs accumulated tiles')
    if (int(bank['partial_accumulated_tiles'].sum())!=row['work']['native_tiles_completed']
            or int(bank['partial_accumulated_fov_tiles'].sum())!=row['work']['fov_tiles_completed']
            or row['work']['cell_node_evaluations']>policy['max_cell_node_evaluations']):
        raise ValueError('partial tile work differs or exceeds fixed budget')
    zero=structural_branch(bank['input__mu'],bank['input__factor'])[1]
    if zero:
        all_masks=np.r_[bank['partial_completed_cells'],bank['partial_completed_fov']]
        if (any(np.any(bank[n]) for n in ('partial_accumulated_tiles','partial_accumulated_fov_tiles'))
                or not (np.all(all_masks) or not np.any(all_masks))):
            raise ValueError('structural-zero partial state must be wholly before or after its exact branch')
    else:
        bounds=bank['input__bounds'];maxside=policy['h']*policy['maximum_tile_side_in_h']
        totals=np.asarray([int(np.prod(np.ceil((box[2:]-box[:2])/maxside))) for box in bounds],np.int64)
        fov=np.r_[bounds[:,:2].min(0),bounds[:,2:].max(0)]
        ftotal=np.asarray([int(np.prod(np.ceil((fov[2:]-fov[:2])/maxside)))],np.int64)
        for counts,masks,expected in ((bank['partial_accumulated_tiles'],bank['partial_completed_cells'],totals),
                (bank['partial_accumulated_fov_tiles'],bank['partial_completed_fov'],ftotal)):
            if np.any(counts>expected) or not np.array_equal(masks,counts==expected):
                raise ValueError('partial completion masks must match independently derived full tile counts')
    return dict(passed=True,available_partial_arrays=sorted(names),
        completed_native_cells=int(bank['partial_completed_cells'].sum()),
        finite_partial_native_cells=int(np.isfinite(bank['partial_log_raw_mass']).sum()),
        incomplete_finite_native_cells=int(np.count_nonzero(np.isfinite(bank['partial_log_raw_mass'])&~bank['partial_completed_cells'])))

if __name__=='__main__':main()
