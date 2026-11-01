"""R38 full-population input preparation and immutable negative-only forward."""
from __future__ import annotations
import argparse
from pathlib import Path
import resource
import time
import traceback
import numpy as np
from experiments import round38_rms_anchor_contract as C
from experiments import round38_rms_anchor_inputs as I
from models import round29_log_rms_projective_pe as MODEL


def save_archive(root,path,arrays):
    if path.exists():raise ValueError('refusing archive overwrite')
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,**arrays)
    return dict(archive_path=str(path.relative_to(root)),archive_sha256=C.sha(path),
                array_sha256={n:C.array_sha(v) for n,v in arrays.items()})


def read_archive(path,digest,fields,hashes=None):
    if C.sha(path)!=digest:raise ValueError('archive bytes changed')
    with np.load(path,allow_pickle=False) as archive:arrays={n:archive[n] for n in fields}
    if hashes is not None and any(C.array_sha(arrays[n])!=hashes[n] for n in fields):
        raise ValueError('canonical input hash differs')
    return arrays


def complete_stage(directory,**metadata):
    if (directory/'completion.json').exists():raise ValueError('refusing completion overwrite')
    C.write(directory/'completion.json',dict(files=C.inventory(directory),**metadata))


def verify_stage(directory):
    result=C.read(directory/'completion.json')
    C.verify_inventory(directory,result['files'],exclude=('completion.json',))
    return result


def prepare(run,audit):
    run,audit=Path(run),Path(audit);lock,_=C.joint_locks(run,audit)
    directory=run/'prepared'
    if directory.exists() or (run/'prepared_complete.json').exists():raise ValueError('retain prior preparation')
    directory.mkdir();started=time.time();clock=time.perf_counter()
    deadline=clock+lock['gates']['preparation_max_phase_wall_seconds']
    C.write(directory/'started.json',dict(started_unix=started,producer_lock_sha256=C.sha(run/'lock.json'),
        audit_lock_sha256=C.sha(audit/'lock.json'),**C.SCOPE))
    specs=C.read(run/'query_specs.json');rows=[];pairs=[];stop=None
    for entry in lock['entries']:
        bank=None;pair_clock=time.perf_counter();decoded=False
        pair=dict(pair_id=entry['pair_id'],**{k:entry[k] for k in ('scene','pair','pair_name')},
            passed=False,attempted=False,source_array_sha256={},cache_array_sha256={},
            source_archive_path=entry['input_path'],source_archive_sha256=entry['input_sha256'],
            cache_archive_path=entry['cache_path'],cache_archive_sha256=entry['cache_sha256'])
        if stop is None:
            try:
                if time.perf_counter()>=deadline:raise TimeoutError('preparation deadline before pair read')
                pair['attempted']=True
                observations=read_archive(entry['input_path'],entry['input_sha256'],C.OBSERVATION_FIELDS)
                decoded=True;pair['source_array_sha256']={n:C.array_sha(v) for n,v in observations.items()}
                cache=read_archive(entry['cache_path'],entry['cache_sha256'],C.CACHE_FIELDS,entry['cache_array_sha256'])
                pair['cache_array_sha256']={n:C.array_sha(v) for n,v in cache.items()}
                bank=I.prepare_inputs(observations,cache)
                if time.perf_counter()>=deadline:raise TimeoutError('preparation deadline after pair construction')
                pair.update(passed=True,status='prepared')
            except Exception as error:
                bank=None;stop=dict(stage='prepare',pair_id=entry['pair_id'],reason=type(error).__name__+': '+str(error))
                pair.update(status='failed',failure_reason=stop['reason'])
        else:pair.update(status='not_attempted',not_attempted_reason='fail_fast_after_first_preparation_failure')
        pair.update(save_archive(run,directory/'pairs'/(entry['pair_id']+'.npz'),bank or {}),
            real_inputs_read=decoded,wall_seconds=time.perf_counter()-pair_clock)
        pairs.append(pair)
        for spec in [s for s in specs if s['pair_id']==entry['pair_id']]:
            row=dict(spec,passed=False,attempted=bool(pair['attempted']),pair_archive_path=pair['archive_path'],
                pair_archive_sha256=pair['archive_sha256'],status=pair['status'])
            values={}
            if bank is not None and stop is None:
                try:
                    if time.perf_counter()>=deadline:raise TimeoutError('preparation deadline before query extraction')
                    values={'input__'+n:v for n,v in I.model_inputs(bank,spec['query_index']).items()}
                    row.update(passed=True,status='prepared')
                except Exception as error:
                    stop=dict(stage='prepare',id=spec['id'],reason=type(error).__name__+': '+str(error));bank=None
                    values={};row.update(status='failed',failure_reason=stop['reason'])
            elif stop is not None:
                row.update(failure_reason=stop['reason'],not_attempted_reason='preparation_failed_or_skipped')
            row.update(save_archive(run,directory/'arrays'/(spec['id']+'.npz'),values));rows.append(row)
        print('R38_PREPARED_PAIR',entry['pair_id'],sum(r['passed'] for r in rows[-48:]),flush=True)
    wall=time.perf_counter()-clock;all_ok=all(r['passed'] for r in rows) and wall<=lock['gates']['preparation_max_phase_wall_seconds']
    C.write(directory/'pair_records.json',pairs);C.write(directory/'records.json',rows)
    complete_stage(directory,started_unix=started,finished_unix=time.time(),wall_seconds=wall,
        complete_case_inventory=len(rows)==C.CASE_COUNT,all_inputs_prepared=all_ok,stop_trigger=stop,
        real_inputs_read=any(p['real_inputs_read'] for p in pairs),**C.SCOPE)
    C.write(run/'prepared_complete.json',dict(prepared_completion_sha256=C.sha(directory/'completion.json'),
        pair_records_sha256=C.sha(directory/'pair_records.json'),records_sha256=C.sha(directory/'records.json'),
        query_specs_sha256=C.sha(run/'query_specs.json'),all_inputs_prepared=all_ok,
        case_count=len(rows),finished_unix=time.time(),**C.SCOPE))


def require_inputs(run,audit):
    lock,_=C.joint_locks(run,audit);seal=C.read(run/'prepared_complete.json')
    prepared=verify_stage(run/'prepared');verification=C.read(audit/'input_verification.json')
    input_completion=verify_stage(audit/'input_audit')
    if (not prepared['all_inputs_prepared'] or not seal['all_inputs_prepared']
        or seal['case_count']!=C.CASE_COUNT or not verification['passed']
        or verification['checked_queries']!=C.CASE_COUNT or verification['checked_pairs']!=C.PAIR_COUNT
        or not input_completion['passed'] or verification['labels_read']
        or verification['prepared_complete_sha256']!=C.sha(run/'prepared_complete.json')
        or verification['producer_lock_sha256']!=C.sha(run/'lock.json')
        or verification['audit_lock_sha256']!=C.sha(audit/'lock.json')
        or input_completion['verification_sha256']!=C.sha(audit/'input_verification.json')
        or seal['prepared_completion_sha256']!=C.sha(run/'prepared/completion.json')
        or seal['records_sha256']!=C.sha(run/'prepared/records.json')
        or seal['pair_records_sha256']!=C.sha(run/'prepared/pair_records.json')):
        raise ValueError('complete independent input audit required before model calls')
    return lock


def forward(run,audit):
    run,audit=Path(run),Path(audit);lock=require_inputs(run,audit);directory=run/'forward'
    if directory.exists() or (run/'forward_complete.json').exists():raise ValueError('retain prior production')
    directory.mkdir();started=time.time();clock=time.perf_counter()
    deadline=clock+lock['gates']['production_max_phase_wall_seconds']
    C.write(directory/'started.json',dict(started_unix=started,producer_lock_sha256=C.sha(run/'lock.json'),
        audit_lock_sha256=C.sha(audit/'lock.json'),input_verification_sha256=C.sha(audit/'input_verification.json'),**C.SCOPE))
    prepared=C.read(run/'prepared/records.json');specs=C.read(run/'query_specs.json')
    if [r['id'] for r in prepared]!=[s['id'] for s in specs]:raise ValueError('preparation schedule differs')
    rows=[];stop=None;charged=0;unknown_work=False
    for index,(spec,prep) in enumerate(zip(specs,prepared)):
        tick=time.perf_counter();row=dict(spec,index=index,passed=False,attempted=False,
            status='not_attempted',input_path=prep['archive_path'],input_sha256=prep['archive_sha256'],
            work={},charged_cell_node_evaluations=0,work_unavailable=False,
            budget_used_before=charged,started_unix=None,finished_unix=None,
            failure_stage=None,not_attempted_reason=None)
        arrays={}
        if stop is not None:
            row['not_attempted_reason']='fail_fast_after_first_failed_scheduled_outcome'
        else:
            stage='precall';row['started_unix']=time.time()
            try:
                if time.perf_counter()>=deadline:raise TimeoutError('production deadline before input read')
                if charged+C.POLICY['max_cell_node_evaluations']>C.GATES['production_max_phase_cell_node_evaluations']:
                    raise RuntimeError('insufficient remaining phase node budget')
                stage='input_delivery'
                bank=read_archive(run/prep['archive_path'],prep['archive_sha256'],C.OLD_FIELDS,prep['array_sha256'])
                stage='precall'
                if time.perf_counter()>=deadline:raise TimeoutError('production deadline after input read')
                stage='model';row['attempted']=True;arrays.update(bank)
                result=MODEL.integrate(*(bank['input__'+n] for n in ('mu','factor','bounds','content_bias')),
                    policy=lock['policy'],deadline=deadline)
                for name,value in result.items():
                    if isinstance(value,np.ndarray):arrays[name]=value
                    else:row[name]=value
                row['model_status']=result['status'];row['status']='ok' if result['passed'] else 'failed'
                row['charged_cell_node_evaluations']=int(result['work']['cell_node_evaluations'])
                if result['passed']:arrays['argmax']=np.asarray(np.argmax(result['probability']),np.int32)
                else:row['failure_stage']='model'
            except Exception as error:
                row.update(passed=False,status='failed',failure_stage=stage,
                    failure_reason=type(error).__name__+': '+str(error))
                if stage=='model':row['work_unavailable']=True;unknown_work=True
            row['finished_unix']=time.time()
            if not row['passed']:
                stop=dict(index=index,id=spec['id'],stage=row['failure_stage'],reason=row['failure_reason'])
        charged+=row['charged_cell_node_evaluations'];row['budget_used_after']=charged
        row.update(save_archive(run,directory/'arrays'/(spec['id']+'.npz'),arrays),
            call_and_output_seconds=time.perf_counter()-tick,
            peak_rss_kib_process=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),**C.SCOPE)
        rows.append(row)
        C.write(directory/'records.json',rows)
        if (index+1)%48==0:print('R38_FORWARD_PAIR',spec['pair_id'],sum(r['passed'] for r in rows[-48:]),flush=True)
    wall=time.perf_counter()-clock
    time_ok=wall<=C.GATES['production_max_phase_wall_seconds']
    node_ok=not unknown_work and charged<=C.GATES['production_max_phase_cell_node_evaluations']
    all_ok=all(r['passed'] for r in rows) and time_ok and node_ok
    complete_stage(directory,started_unix=started,finished_unix=time.time(),wall_seconds=wall,
        complete_case_inventory=len(rows)==C.CASE_COUNT,all_production_completed=all_ok,
        phase_wall_passed=time_ok,phase_node_budget_passed=node_ok,work_unavailable=unknown_work,
        charged_cell_node_evaluations=charged,stop_trigger=stop,**C.SCOPE)
    C.write(run/'forward_complete.json',dict(forward_completion_sha256=C.sha(directory/'completion.json'),
        forward_records_sha256=C.sha(directory/'records.json'),prepared_complete_sha256=C.sha(run/'prepared_complete.json'),
        input_verification_sha256=C.sha(audit/'input_verification.json'),all_production_completed=all_ok,
        case_count=len(rows),actual_calls=sum(r['attempted'] for r in rows),completed_calls=sum(r['passed'] for r in rows),
        phase_wall_seconds=wall,phase_wall_passed=time_ok,phase_node_budget_passed=node_ok,
        charged_cell_node_evaluations=charged,stop_trigger=stop,finished_unix=time.time(),**C.SCOPE))


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('freeze','prepare','forward'))
    p.add_argument('--out',type=Path,default=C.DEFAULT_OUT);p.add_argument('--audit-root',type=Path,default=C.DEFAULT_AUDIT)
    a=p.parse_args();C.configure_runtime()
    if a.stage=='freeze':C.freeze(a.out,'producer',audit_root=a.audit_root)
    else:
        try:globals()[a.stage](a.out,a.audit_root)
        except BaseException as error:
            path=a.out/(a.stage+'_fatal_failure.json')
            if not path.exists():C.write(path,dict(stage=a.stage,error_type=type(error).__name__,message=str(error),
                traceback=traceback.format_exc(),finished_unix=time.time(),complete_case_inventory=False,**C.SCOPE))
            raise


if __name__=='__main__':main()
