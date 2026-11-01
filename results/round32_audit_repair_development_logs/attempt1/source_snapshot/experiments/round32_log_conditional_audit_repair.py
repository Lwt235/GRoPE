"""Post-seal R32 audit delivery repair: serialize only exact NumPy booleans.

No numerical function, integral, gate or frozen source is changed. The original
failed audit and producer trees remain immutable. A fresh directory contains
byte-identical prior lock/input evidence plus an explicit post-seal repair lock.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import shutil
import time
import numpy as np
from experiments import round32_log_conditional_audit as A

C=A.C
DEFAULT_RUN=C.ROOT/'results/round32_log_conditional_preflight'
DEFAULT_ORIGINAL=C.ROOT/'results/round32_log_conditional_independent_audit'
DEFAULT_OUT=C.ROOT/'results/round32_log_conditional_independent_audit_serialization_repair'
REPAIR_SOURCES=('experiments/round32_log_conditional_audit_repair.py',
    'tests/test_round32_log_conditional_audit_repair.py','ROUND32_LOG_CONDITIONAL_AUDIT_REPAIR_PROTOCOL.md')
EXTRA_BINDINGS=('results/round32_log_conditional_execution_20261031.json',
    'results/round32_log_conditional_pipeline_20261031.log',
    'results/round32_log_joint_preflight_20261031.json','results/round32_log_joint_preflight_20261031.log',
    'results/round32_root_controller_development_history_20261031.json')
COPY_FILES=('lock.json','lock.sha256','input_verification.json')
COPY_TREES=('source_snapshot','input_audit')
REPAIR_SCOPE=dict(post_seal_serialization_repair=True,numerical_logic_changed=False,
    numerical_integrations_executed=0,original_failure_retained=True,prior_freeze_replaced=False,
    synthetic_inputs_read=False,independent_synthetic_reference=False,**A.SCOPE)


def numpy_boolean_only(value):
    """JSON default: do not coerce arrays, integers, floats or arbitrary objects."""
    if type(value) is np.bool_:return bool(value)
    raise TypeError('Object of type '+type(value).__name__+' is not JSON serializable')


@contextmanager
def serialized_boolean_writer(out):
    """Change output encoding only; restore the exact frozen writer on all exits."""
    out=Path(out).resolve();original=C.write
    def write(path,value):
        path=Path(path).resolve()
        if not path.is_relative_to(out):raise ValueError('repair writer may only write inside the new repair directory')
        encoded=json.dumps(value,indent=2,sort_keys=True,allow_nan=False,default=numpy_boolean_only)+'\n'
        path.write_text(encoded)
    C.write=write
    try:yield
    finally:C.write=original


def complete_tree(path):
    done=C.read(path/'completion.json');C.verify_inventory(path,done['files'],exclude=('completion.json',))
    return dict(done['files'],**{'completion.json':C.sha(path/'completion.json')})


def verify_original(run,original):
    """All operations before audit are metadata and file-byte hash checks only."""
    producer_files=complete_tree(run);audit_files=complete_tree(original)
    failure=C.read(original/'failure.json');done=C.read(original/'completion.json')
    if (failure.get('error_type')!='TypeError' or failure.get('message')!='Object of type bool_ is not JSON serializable'
            or done.get('delivery_integrity_passed') is not False or done.get('numerical_preflight_passed') is not False):
        raise ValueError('the preserved original NumPy-boolean serialization failure is required')
    lock,seal,rows,_,specs=A.verify_forward(run,original)
    if Path(lock['audit_root']).resolve()!=original:raise ValueError('original producer audit destination differs')
    if not seal['finished_unix']<=failure['started_unix']<=failure['finished_unix']:
        raise ValueError('original audit failure must follow the global reference seal')
    return dict(producer_files=producer_files,original_audit_files=audit_files,
        producer_lock_sha256=C.sha(run/'lock.json'),original_audit_lock_sha256=C.sha(original/'lock.json'),
        original_input_verification_sha256=C.sha(original/'input_verification.json'),
        original_failure_sha256=C.sha(original/'failure.json'),producer_forward_complete_sha256=C.sha(run/'forward_complete.json'),
        frozen_source_hashes=lock['source_hashes'],runtime=lock['runtime'],
        base_case_count=len(specs),scheduled_call_count=len(rows),reference_seal_finished_unix=seal['finished_unix'])


def copied_inventory(original):
    result={name:C.sha(original/name) for name in COPY_FILES}
    for directory in COPY_TREES:
        for path in sorted((original/directory).rglob('*')):
            if path.is_file():result[str(path.relative_to(original))]=C.sha(path)
    return result


def separate_destination(run,original,out):
    for source in (run,original):
        if out==source or out.is_relative_to(source) or source.is_relative_to(out):
            raise ValueError('repair directory must be disjoint from both immutable original trees')


def extra_bindings():
    result={name:C.sha(C.ROOT/name) for name in EXTRA_BINDINGS}
    history=C.read(C.ROOT/'results/round32_root_controller_development_history_20261031.json')
    for name,digest in history['files'].items():
        if C.sha(C.ROOT/name)!=digest:raise ValueError('premature root-controller evidence changed')
        result[name]=digest
    return result


def freeze(run,original,out):
    run,original,out=(Path(p).resolve() for p in (run,original,out));separate_destination(run,original,out)
    if out.exists():raise ValueError('preserve every repair attempt; destination already exists')
    bindings=verify_original(run,original);runtime=C.check_runtime()
    if runtime!=bindings['runtime']:raise ValueError('repair runtime differs from the frozen numerical runtime')
    evidence=copied_inventory(original);extra=extra_bindings();sources={name:C.sha(C.ROOT/name) for name in REPAIR_SOURCES}
    out.mkdir(parents=True)
    for name,digest in evidence.items():
        target=out/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(original/name,target)
        if C.sha(target)!=digest:raise ValueError('original evidence changed during byte-identical copy')
    for name,digest in sources.items():
        target=out/'repair_source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(C.ROOT/name,target)
        if C.sha(target)!=digest:raise ValueError('repair source changed while snapshotting')
    lock=dict(version=1,created_unix=time.time(),producer_root=str(run),original_failed_audit_root=str(original),
        repair_root=str(out),copied_original_evidence=evidence,repair_source_hashes=sources,
        extra_bindings=extra,serialization_change='json.default: exact numpy.bool_ -> Python bool; all other unsupported types rejected',
        output_directory_changed=True,**bindings,**REPAIR_SCOPE)
    C.write(out/'repair_lock.json',lock);(out/'repair_lock.sha256').write_text(C.sha(out/'repair_lock.json')+'\n')
    return lock


def verify_repair(out):
    out=Path(out).resolve();lock=C.read(out/'repair_lock.json')
    run=Path(lock['producer_root']);original=Path(lock['original_failed_audit_root']);separate_destination(run,original,out)
    if (C.sha(out/'repair_lock.json')!=(out/'repair_lock.sha256').read_text().strip()
            or lock['repair_root']!=str(out) or any(lock.get(k)!=v for k,v in REPAIR_SCOPE.items())):
        raise ValueError('post-seal serialization repair lock/scope differs')
    current=verify_original(run,original)
    if any(lock.get(k)!=value for k,value in current.items()) or C.check_runtime()!=lock['runtime']:
        raise ValueError('original immutable evidence or runtime changed')
    if lock['extra_bindings']!=extra_bindings() or lock['copied_original_evidence']!=copied_inventory(original):
        raise ValueError('original failure/controller/input bindings changed')
    if not current['reference_seal_finished_unix']<=lock['created_unix']<=time.time():
        raise ValueError('repair lock must be explicitly post-seal')
    for name,digest in lock['copied_original_evidence'].items():
        if C.sha(out/name)!=digest:raise ValueError('copied original source/input evidence changed')
    if set(lock['repair_source_hashes'])!=set(REPAIR_SOURCES):raise ValueError('repair source membership changed')
    for name,digest in lock['repair_source_hashes'].items():
        if C.sha(C.ROOT/name)!=digest or C.sha(out/'repair_source_snapshot'/name)!=digest:
            raise ValueError('current or snapshotted repair source changed')
    # Exercise the unchanged auditor against the copied identities as well.
    A.verify_forward(run,out)
    return lock


def run_repair(out):
    out=Path(out).resolve();lock=verify_repair(out)
    if any((out/name).exists() for name in ('audit_started.json','completion.json','summary.json','verification.json','repair_run_started.json')):
        raise ValueError('preserve prior repair output; no retry in place')
    C.write(out/'repair_run_started.json',dict(started_unix=time.time(),repair_lock_sha256=C.sha(out/'repair_lock.json'),
        original_failure_sha256=lock['original_failure_sha256'],**REPAIR_SCOPE))
    with serialized_boolean_writer(out):
        result=A.audit(Path(lock['producer_root']),out)
    # Do not append inside the newly sealed output inventory after A.audit.
    verify_repair(out)
    C.verify_inventory(out,C.read(out/'completion.json')['files'],exclude=('completion.json',))
    return dict(verification=result,original_trees_unchanged=True,repair_lock_sha256=C.sha(out/'repair_lock.json'),
        numerical_integrations_executed_by_repair=0)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('freeze','run'))
    parser.add_argument('--run',type=Path,default=DEFAULT_RUN);parser.add_argument('--original',type=Path,default=DEFAULT_ORIGINAL)
    parser.add_argument('--out',type=Path,default=DEFAULT_OUT);args=parser.parse_args()
    if args.command=='freeze':result=freeze(args.run,args.original,args.out)
    else:result=run_repair(args.out)
    print(json.dumps(result,sort_keys=True,allow_nan=False))


if __name__=='__main__':main()
