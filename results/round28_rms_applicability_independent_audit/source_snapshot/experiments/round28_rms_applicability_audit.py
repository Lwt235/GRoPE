"""R28 input, delivery and algebra audit; never a probability reference.

Only eight authenticated R23 prepared fields are decoded. New homogeneous
coordinates are independently reconstructed with scalar math.fsum, including
the unchanged principal-point convention. The saved-output checks reconstruct
area-floor normalization and metadata, not the general projective integral.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import shutil
import time

import numpy as np
from scipy.special import logsumexp, ndtr

from experiments import round28_rms_contract as C

SCOPE = dict(general_numerical_accuracy_established=False,
             independent_probability_reference=False, labels_read=False,
             efficacy_tested=False,candidate_selected=False,
             scientific_gate_passed=False, goal_complete=False)
APPROXIMATE_TIE_TOLERANCE = 1e-12
DEFAULT_RUN=C.ROOT/'results/round28_rms_applicability'
DEFAULT_OUT=C.ROOT/'results/round28_rms_applicability_independent_audit'
PREPARED_NAMES={'r23__'+name for name in C.OLD_FIELDS}|{'mapping_matrix','input__mu','input__factor','input__bounds','input__content_bias'}


def normalized_error(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError('matching finite arrays required for normalized input error')
    return float(np.max(np.abs(actual-expected))/max(1.,float(np.max(np.abs(expected)))))


def _multiply(left, right):
    """Small independent matrix product with scalar fsum reduction."""
    left, right = np.asarray(left), np.asarray(right)
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[0]:
        raise ValueError('matrix product dimensions differ')
    return np.asarray([[math.fsum(float(left[i,k])*float(right[k,j]) for k in range(left.shape[1]))
                        for j in range(right.shape[1])] for i in range(left.shape[0])],np.float64)


def literal_bounds():
    pixel = np.asarray([(x,y,x+8,y+8) for y in range(0,144,8) for x in range(0,192,8)],np.float64)
    return (pixel-.5)/8.


def transformed_inputs(old):
    if set(old) != set(C.OLD_FIELDS):
        raise ValueError('exact eight-field inherited geometry boundary required')
    shapes = dict(input__mu=(3,),input__factor=(3,3),input__bounds=(432,4),
                  input__H=(2,2),input__offset=(2,),target__covariance=(3,3),
                  query_bounds_px=(4,),query_index=())
    for name,shape in shapes.items():
        value = np.asarray(old[name]);dtype = np.int64 if name in ('query_bounds_px','query_index') else np.float64
        if value.dtype != dtype or value.shape != shape or not np.isfinite(value).all():
            raise ValueError('inherited geometry dtype/shape/finite mismatch: '+name)
    if not np.array_equal(old['input__bounds'],literal_bounds()):
        raise ValueError('original 432 native half-pixel boundaries changed')
    h,c = old['input__H'],old['input__offset']
    a = np.asarray([[h[0,0],h[0,1],c[0]], [h[1,0],h[1,1],c[1]], [0.,0.,1.]],np.float64)
    mu = _multiply(a,old['input__mu'][:,None])[:,0]
    factor = _multiply(a,old['input__factor'])
    covariance = _multiply(_multiply(a,old['target__covariance']),a.T)
    return dict(mapping_matrix=a,input__mu=mu,input__factor=factor,
        input__bounds=old['input__bounds'],input__content_bias=np.zeros(432,np.float64)),covariance


def geometry_check(old, actual, spec, tolerance=1e-12):
    expected,covariance = transformed_inputs(old)
    expected_names = {'r23__'+name for name in C.OLD_FIELDS}|set(expected)
    if set(actual) != expected_names:
        raise ValueError('new prepared geometry must contain exactly13 prescribed arrays')
    for name in C.OLD_FIELDS:
        if C.array_sha(actual['r23__'+name]) != C.array_sha(old[name]):
            raise ValueError('retained inherited array identity changed: '+name)
    for name,value in expected.items():
        if actual[name].dtype != np.float64 or actual[name].shape != value.shape or not np.isfinite(actual[name]).all():
            raise ValueError('new homogeneous input dtype/shape mismatch: '+name)
    if (not np.array_equal(actual['mapping_matrix'],expected['mapping_matrix'])
            or not np.array_equal(actual['input__bounds'],expected['input__bounds'])
            or not np.array_equal(actual['input__content_bias'],expected['input__content_bias'])
            or not np.array_equal(actual['input__mu'][2:],old['input__mu'][2:])
            or not np.array_equal(actual['input__factor'][2],old['input__factor'][2])):
        raise ValueError('mapping/principal point, native units, bias or exact depth identity changed')
    if (int(old['query_index']) != spec['query_index']
            or not np.array_equal(old['query_bounds_px'],np.asarray(spec['query_bounds'],np.int64))):
        raise ValueError('inherited fixed query identity changed')
    gram = _multiply(actual['input__factor'],actual['input__factor'].T)
    errors = {name:normalized_error(actual[name],expected[name]) for name in ('input__mu','input__factor')}
    errors['factor_covariance'] = normalized_error(gram,covariance)
    old_rms = math.hypot(float(old['input__mu'][2]),*map(float,old['input__factor'][2]))
    new_rms = math.hypot(float(actual['input__mu'][2]),*map(float,actual['input__factor'][2]))
    if old_rms != new_rms:
        raise ValueError('homogeneous mapping changed depth RMS')
    return dict(passed=bool(max(errors.values())<=tolerance),errors=errors,
        maximum_input_normalized_error=max(errors.values()),depth_rms=old_rms,
        depth_rows_exact=True,principal_offset_unchanged=True,zero_content_bias=True,
        original_geometry_provenance_inherited=True,**SCOPE)


def read_arrays(path, hashes, fields=None):
    with np.load(path,allow_pickle=False) as archive:
        names=list(hashes) if fields is None else fields
        if fields is None and set(archive.files)!=set(hashes):
            raise ValueError('unexpected prepared/output archive fields')
        result={name:archive[name] for name in names}
    if {name:C.array_sha(value) for name,value in result.items()} != hashes:
        raise ValueError('authenticated array hashes differ')
    return result


def probabilities(raw,bounds,bias,floor_density):
    raw,bounds,bias=map(np.asarray,(raw,bounds,bias))
    if (raw.dtype!=np.float64 or raw.shape!=(len(bounds),) or bias.dtype!=np.float64 or bias.shape!=raw.shape
            or not np.isfinite(raw).all() or np.any(raw<0) or not np.isfinite(bias).all()):
        raise ValueError('finite nonnegative saved native mass/bias required')
    floor=np.prod(bounds[:,2:]-bounds[:,:2],axis=1)*floor_density
    if not np.isfinite(floor).all() or np.any(floor<=0):
        raise ValueError('positive single native area floor required')
    floored=raw+floor;logits=np.log(floored)+bias;normalizer=float(logsumexp(logits))
    return np.exp(logits-normalizer),floor,floored,normalizer


def structural_branch(mu,factor):
    rms=math.hypot(float(mu[2]),*map(float,factor[2]))
    if rms==0:return 'zero_depth_rms',True
    if np.all(factor[2]==0):
        return ('nonpositive_constant_depth',True) if mu[2]<=0 else ('positive_constant_depth_gaussian',False)
    return 'variable_depth',False


def verify_producer_lock(run):
    lock=C.read(run/'lock.json')
    if C.sha(run/'lock.json')!=(run/'lock.sha256').read_text().strip():
        raise ValueError('producer lock bytes changed')
    bindings,specs=C.inherited_evidence()
    expected=dict(version=1,runtime=C.check_runtime(),policy=C.POLICY,gates=C.GATES,
        query_specs_sha256=C.sha(run/'query_specs.json'),case_ids=[s['id'] for s in specs],
        case_count=72,key_count=432,bindings=bindings,**SCOPE)
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
        source_hashes=lock['source_hashes'],runtime=lock['runtime'],**SCOPE))
    (out/'lock.sha256').write_text(C.sha(out/'lock.json')+'\n')


def joint_locks(run,out):
    lock,design=verify_producer_lock(run),C.read(out/'lock.json')
    expected=dict(run=str(run.resolve()),producer_root=str(run.resolve()),producer_lock_sha256=C.sha(run/'lock.json'),
        query_specs_sha256=lock['query_specs_sha256'],source_hashes=lock['source_hashes'],runtime=lock['runtime'],**SCOPE)
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
    if (not design['created_unix']<=seal['started_unix']<=seal['finished_unix']
            or start['started_unix']!=seal['started_unix']
            or start['producer_lock_sha256']!=expected['producer_lock_sha256']
            or start['audit_lock_sha256']!=expected['audit_lock_sha256']
            or done['complete_case_inventory'] is not True
            or done['all_inputs_prepared']!=seal['all_inputs_prepared']
            or seal['all_inputs_prepared']!=all(r['passed'] for r in rows)):
        raise ValueError('preparation chronology/status differs')
    names={'started.json','records.json'}|{'arrays/'+s['id']+'.npz' for s in specs}
    if set(done['files'])!=names:raise ValueError('prepared file population differs')
    for row,spec in zip(rows,specs):
        if (any(row.get(k)!=spec[k] for k in ('id','scene','pair','query_size','query_index'))
                or row['archive_path']!='prepared/arrays/'+spec['id']+'.npz'
                or C.sha(run/row['archive_path'])!=row['archive_sha256']
                or row['source_archive_path']!=spec['source_archive_path']
                or row['source_archive_sha256']!=spec['archive_sha256']
                or type(row['passed']) is not bool or row['status']!=('prepared' if row['passed'] else 'failed')
                or not math.isfinite(row['preparation_seconds']) or row['preparation_seconds']<0):
            raise ValueError('prepared identity/status/archive/time differs')
        if row['passed'] and set(row['array_sha256'])!=PREPARED_NAMES:
            raise ValueError('successful prepared hash schema differs')
        if not row['passed'] and not row.get('failure_reason'):
            raise ValueError('failed geometry requires explicit retained reason')
    return lock,seal,rows,specs


def _failure(directory,error,started):
    C.write(directory/'failure.json',dict(error_type=type(error).__name__,message=str(error),
        started_unix=started,finished_unix=time.time(),**SCOPE))
    C.write(directory/'completion.json',dict(passed=False,
        files={n:h for n,h in C.inventory(directory).items() if n!='completion.json'},
        finished_unix=time.time(),**SCOPE))


def input_audit(run,out):
    directory=out/'input_audit'
    if directory.exists() or (out/'input_verification.json').exists():
        raise ValueError('preserve every prior input audit attempt')
    directory.mkdir();started=time.time();C.write(directory/'started.json',dict(started_unix=started,**SCOPE))
    checks=[]
    try:
        lock,seal,rows,specs=verify_prepared(run,out)
        if started<seal['finished_unix'] or (run/'forward').exists() or (run/'started.json').exists():
            raise ValueError('all prepared inputs must be sealed before input audit and model calls')
        for row,spec in zip(rows,specs):
            try:
                if not row['passed']:raise ValueError(row['failure_reason'])
                old_hashes={n:spec['array_sha256'][n] for n in C.OLD_FIELDS}
                if C.sha(spec['source_archive_path'])!=spec['archive_sha256']:
                    raise ValueError('bound old prepared archive changed')
                old=read_arrays(spec['source_archive_path'],old_hashes,fields=C.OLD_FIELDS)
                actual=read_arrays(run/row['archive_path'],row['array_sha256'])
                check=geometry_check(old,actual,spec,lock['gates']['input_normalized_absolute'])
            except Exception as error:
                check=dict(passed=False,failure_reason=type(error).__name__+': '+str(error),errors={},**SCOPE)
            checks.append(dict(id=spec['id'],scene=spec['scene'],query_size=spec['query_size'],
                query_index=spec['query_index'],prepared_archive_sha256=row['archive_sha256'],**check))
            C.write(directory/'checks.json',checks)
        joint_locks(run,out)
        passed=bool(len(checks)==72 and all(r['passed'] for r in checks))
        verification=dict(passed=passed,checked_queries=len(checks),producer_lock_sha256=C.sha(run/'lock.json'),
            audit_lock_sha256=C.sha(out/'lock.json'),producer_prepared_complete_sha256=C.sha(run/'prepared_complete.json'),
            producer_prepared_completion_sha256=C.sha(run/'prepared/completion.json'),query_specs_sha256=lock['query_specs_sha256'],
            source_prepared_hashes={s['source_archive_path']:s['archive_sha256'] for s in specs},
            maximum_normalized_input_error=max((r.get('maximum_input_normalized_error',0.) for r in checks),default=0.),
            started_unix=started,finished_unix=time.time(),**SCOPE)
        C.write(out/'input_verification.json',verification)
        C.write(directory/'completion.json',dict(passed=passed,files=C.inventory(directory),
            verification_sha256=C.sha(out/'input_verification.json'),finished_unix=time.time(),**SCOPE))
        if not passed:raise RuntimeError('independent input audit failed; no forward authorized')
        return verification
    except (Exception,KeyboardInterrupt) as error:
        if not (directory/'completion.json').exists():_failure(directory,error,started)
        raise


def verify_input_audit(run,out):
    lock,prepared,rows,specs=verify_prepared(run,out)
    done=complete_inventory(out/'input_audit');verification=C.read(out/'input_verification.json')
    expected=dict(passed=True,checked_queries=72,producer_lock_sha256=C.sha(run/'lock.json'),
        audit_lock_sha256=C.sha(out/'lock.json'),producer_prepared_complete_sha256=C.sha(run/'prepared_complete.json'),
        producer_prepared_completion_sha256=C.sha(run/'prepared/completion.json'),query_specs_sha256=lock['query_specs_sha256'],
        source_prepared_hashes={s['source_archive_path']:s['archive_sha256'] for s in specs},**SCOPE)
    if (done.get('passed') is not True or done['verification_sha256']!=C.sha(out/'input_verification.json')
            or any(verification.get(k)!=v for k,v in expected.items())
            or not prepared['finished_unix']<=verification['started_unix']<=verification['finished_unix']):
        raise ValueError('complete independent input audit required before model outcomes')
    checks=C.read(out/'input_audit/checks.json')
    if ([r['id'] for r in checks]!=[s['id'] for s in specs] or not all(r['passed'] for r in checks)):
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
    if (start['started_unix']!=seal['started_unix']
            or start['producer_lock_sha256']!=expected['producer_lock_sha256']
            or start['audit_lock_sha256']!=expected['audit_lock_sha256']
            or start['input_verification_sha256']!=expected['input_verification_sha256']
            or not input_verification['finished_unix']<=seal['started_unix']<=seal['phase_finished_unix']<=seal['finished_unix']
            or not math.isfinite(seal['phase_wall_seconds']) or seal['phase_wall_seconds']<0
            or seal['phase_wall_passed']!=(seal['phase_wall_seconds']<=lock['gates']['production_max_phase_wall_seconds'])
            or done['phase_wall_passed']!=seal['phase_wall_passed']
            or root_done['forward_complete_sha256']!=C.sha(run/'forward_complete.json')):
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
    previous=seal['started_unix']
    for row,prep,spec,record in zip(rows,prepared_rows,specs,journal):
        if (any(row.get(k)!=spec[k] for k in ('id','scene','pair','query_size','query_index'))
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
    if (type(row.get('passed')) is not bool or type(row.get('attempted')) is not bool
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
    for name in present:
        if C.array_sha(bank[name])!=C.array_sha(prepared[name]):raise ValueError('model input copy differs from audited geometry')
    success={'raw_mass','probability','floored_mass','argmax'}
    partial={'partial_raw_mass','partial_raw_fov_mass'}
    outputs=set(bank)-inputs
    if row['passed']:
        if not row['attempted'] or present!=inputs or outputs!=success:
            raise ValueError('successful call requires four actual inputs and complete outputs')
    elif not outputs<=partial:
        raise ValueError('failed output must contain available partials only')
    if not row['attempted'] and (outputs or any(row['work'].values())):
        raise ValueError('unattempted call must not invent partial outputs or work')
    for name in outputs:
        dtype,shape=(np.int32,()) if name=='argmax' else (np.float64,(1,) if name=='partial_raw_fov_mass' else (432,))
        if bank[name].dtype!=dtype or bank[name].shape!=shape:
            raise ValueError('model output dtype/shape differs: '+name)
    if row['work'].get('cell_node_evaluations',0)>policy['max_cell_node_evaluations']:
        raise ValueError('retained per-call node count exceeds fixed budget')
    if row['passed'] and int(bank['argmax'])!=int(bank['probability'].argmax()):
        raise ValueError('saved first-index winner differs')
    return dict(delivery_integrity_passed=True,attempted=row['attempted'],model_completed=row['passed'])


def audit(run,out):
    if any((out/n).exists() for n in ('audit_started.json','verification.json','summary.json','completion.json')):
        raise ValueError('preserve every prior final audit attempt')
    started=time.time();C.write(out/'audit_started.json',dict(started_unix=started,**SCOPE))
    checks=[]
    try:
        lock,seal,rows,prepared_rows,specs=verify_forward(run,out)
        if started<seal['finished_unix']:raise ValueError('final audit must follow complete global forward seal')
        for row,prep,spec in zip(rows,prepared_rows,specs):
            try:
                prepared=read_arrays(run/prep['archive_path'],prep['array_sha256'])
                bank=read_arrays(run/row['archive_path'],row['array_sha256'])
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
            checks.append(dict(id=spec['id'],scene=spec['scene'],query_size=spec['query_size'],query_index=spec['query_index'],**check,**SCOPE))
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
            started_unix=started,finished_unix=time.time(),**SCOPE)
        C.write(out/'summary.json',summary)
        verification=dict(delivery_integrity_passed=complete,applicability_calls_completed=applicability,
            producer_lock_sha256=C.sha(run/'lock.json'),audit_lock_sha256=C.sha(out/'lock.json'),
            producer_prepared_complete_sha256=C.sha(run/'prepared_complete.json'),
            input_verification_sha256=C.sha(out/'input_verification.json'),
            producer_forward_complete_sha256=C.sha(run/'forward_complete.json'),
            producer_records_sha256=C.sha(run/'forward/records.json'),producer_completion_sha256=C.sha(run/'completion.json'),
            summary_sha256=C.sha(out/'summary.json'),checked_queries=len(checks),finished_unix=time.time(),**SCOPE)
        C.write(out/'verification.json',verification)
        C.write(out/'completion.json',dict(files=C.inventory(out),verification_sha256=C.sha(out/'verification.json'),
            delivery_integrity_passed=complete,applicability_calls_completed=applicability,finished_unix=time.time(),**SCOPE))
        return verification
    except (Exception,KeyboardInterrupt) as error:
        _failure(out,error,started)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('freeze','input-audit','audit'))
    parser.add_argument('--run',type=Path,default=DEFAULT_RUN)
    parser.add_argument('--out',type=Path,default=DEFAULT_OUT)
    args=parser.parse_args()
    {'freeze':freeze,'input-audit':input_audit,'audit':audit}[args.command](args.run.resolve(),args.out.resolve())


# Copied independent R27 algebra/metadata check, not its numerical reference.
def production_check(row, bank, policy, gates):
    if row['passed'] is not True:
        return dict(passed=False, status='production_failed', failure_reason=row.get('failure_reason'))
    mu, factor, bounds, bias = [bank['input__'+n] for n in ('mu','factor','bounds','content_bias')]
    branch, zero = structural_branch(mu, factor)
    raw, probability = bank['raw_mass'], bank['probability']
    expected, floor, floored, content_normalizer = probabilities(raw,bounds,bias,policy['floor_density'])
    if probability.dtype != np.float64 or probability.shape != raw.shape or not np.isfinite(probability).all() or np.any(probability<0):
        raise ValueError('invalid saved probability')
    fov = np.r_[bounds[:,:2].min(0),bounds[:,2:].max(0)]
    max_side = policy['h']*policy['maximum_tile_side_in_h']
    native_tiles = 0 if zero else sum(int(np.prod(np.ceil((box[2:]-box[:2])/max_side).astype(np.int64))) for box in bounds)
    fov_tiles = 0 if zero else int(np.prod(np.ceil((fov[2:]-fov[:2])/max_side).astype(np.int64)))
    work_expected = dict(cell_node_evaluations=(native_tiles+fov_tiles)*24**2,native_tiles_completed=native_tiles,fov_tiles_completed=fov_tiles)
    raw_sum, floor_sum = math.fsum(map(float,raw)),math.fsum(map(float,floor))
    if not math.isfinite(row['raw_fov_mass']) or row['raw_fov_mass']<0: raise ValueError('invalid direct FOV mass')
    direct_error = abs(row['raw_fov_mass']-raw_sum)
    normalization = abs(math.fsum(map(float,probability))-1.)
    arithmetic = float(np.max(abs(expected-probability)))
    rms = math.hypot(float(mu[2]),*map(float,factor[2]))
    normalized_m = None if rms == 0 else float(mu[2]/rms)
    normalized_s = None if rms == 0 else math.hypot(*map(float,factor[2]/rms))
    bound = 0. if zero else float(np.prod(bounds[:,2:]-bounds[:,:2],axis=1).sum())/(math.pi*policy['h']**2*math.comb(48,24))
    diag=row['density_diagnostics']; trunc=row['analytic_truncation_bound']
    node_keys=('constant_depth_nodes','positive_branch_nodes','negative_branch_nodes','negative_erfcx_nodes','negative_series_nodes')
    if any(type(diag[k]) is not int or diag[k]<0 for k in node_keys):raise ValueError('density branch counts invalid')
    series_keys=('maximum_log_negative_tail_relative_remainder','minimum_scaled_negative_tail_series_lower_bound',
                 'maximum_scaled_negative_tail_series_upper_bound')
    if diag['negative_series_nodes']:
        rel,lower,upper=[diag[k] for k in series_keys]
        series_valid=all(math.isfinite(v) for v in (rel,lower,upper)) and 0<lower<=upper and rel<=math.log(policy['max_negative_tail_relative_remainder'])
    else:series_valid=all(diag[k] is None for k in series_keys)
    front=0. if zero else (1. if normalized_s==0 else float(ndtr(normalized_m/normalized_s)))
    metadata = bool(row['structural_branch']==branch and row['structural_zero'] is zero
        and series_valid and np.isclose(row['front_mass'],front,rtol=2e-13,atol=0)
        and row['rms_depth']==rms and row['normalized_depth_mean']==normalized_m and row['normalized_depth_std']==normalized_s
        and row['quadrature_order']==24 and row['maximum_tile_side_in_h']==2.
        and row['projection_regularization']=='rms_depth_xy_gaussian'
        and row['tail_truncation_used'] is False and row['quadrature_error_bound_available'] is False
        and row['exact_arithmetic_gl_bound_available'] is True
        and row['native_tile_count']==native_tiles and row['fov_tile_count']==fov_tiles and row['work']==work_expected
        and row['h']==policy['h'] and row['floor_density']==policy['floor_density']
        and trunc['exact_arithmetic_only'] is True and trunc['excludes_density_evaluation_and_roundoff'] is True
        and trunc['is_relative_error_certificate'] is False
        and trunc['raw_native_l1_absolute']==bound and trunc['raw_direct_fov_absolute']==bound
        and diag['constant_depth_nodes']+diag['positive_branch_nodes']+diag['negative_branch_nodes']==work_expected['cell_node_evaluations']
        and diag['negative_erfcx_nodes']+diag['negative_series_nodes']==diag['negative_branch_nodes']
        and np.isclose(row['raw_mass_sum'],raw_sum,rtol=1e-13,atol=0)
        and np.isclose(row['total_area_floor'],floor_sum,rtol=1e-13,atol=0)
        and np.isclose(row['normalizer'],math.fsum(map(float,floored)),rtol=1e-13,atol=0)
        and np.isclose(row['log_content_normalizer'],content_normalizer,rtol=0,atol=gates['normalization_absolute'])
        and row['zero_raw_cells']==int(np.count_nonzero(raw==0))
        and row['all_raw_mass_zero'] is bool(np.all(raw==0))
        and row['area_prior_used'] is bool(np.all(raw==0) and np.all(bias==bias[0]))
        and row['raw_mass_below_total_floor'] is bool(np.sum(raw)<np.sum(floor))
        and row['argmax_index']==int(probability.argmax())
        and row['argmax_exact_ties']==int(np.count_nonzero(probability==probability.max()))
        and row['argmax_near_ties_at_1e_12']==int(np.count_nonzero(probability.max()-probability<=1e-12)))
    positive_or_zero = (np.all(raw==0) and row['raw_fov_mass']==0) if zero else (np.all(raw>0) and row['raw_fov_mass']>0)
    passed = bool(metadata and positive_or_zero and normalization<=gates['normalization_absolute']
        and arithmetic<=gates['normalization_absolute'] and np.array_equal(floored,bank['floored_mass'])
        and direct_error<=gates['direct_fov_additivity_absolute']
        and work_expected['cell_node_evaluations']<=policy['max_cell_node_evaluations'])
    return dict(passed=passed,status='checked',normalization_error=normalization,same_raw_probability_error=arithmetic,
        metadata_consistent=metadata,structural_branch=branch,structural_zero=zero,raw_fov_mass=raw_sum,
        direct_fov_mass=row['raw_fov_mass'],direct_fov_additivity_error=direct_error,
        raw_below_total_floor=bool(raw_sum<floor_sum),all_raw_mass_zero=bool(np.all(raw==0)),total_floor=floor_sum,
        exact_maximizers=np.flatnonzero(probability==probability.max()).tolist(),work_expected=work_expected)


if __name__=='__main__':main()
