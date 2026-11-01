"""R33 shared metadata/runtime contract, with no reference or model arithmetic."""
from __future__ import annotations
import ast
from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import numpy as np
import scipy
import scipy.special
from threadpoolctl import threadpool_info
ROOT=Path(__file__).resolve().parents[1]
THREAD_ENV=('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')
from experiments import round32_log_conditional_contract as PREVIOUS
OLD_RUN=ROOT/'results/round30_log_rms_applicability'
OLD_REAL_AUDIT=ROOT/'results/round30_log_rms_applicability_independent_audit'
OLD_AUDIT=ROOT/'results/round29_log_rms_independent_audit' # inherited identity evidence only
OLD_FIELDS=PREVIOUS.OLD_FIELDS
IDENTITY_FIELDS=('id','scene','pair','pair_name','query_size','query_index','query_bounds')
CASE_COUNT=72
CALL_COUNT=144
KEY_COUNT=432
FAIL_FAST='first_failed_scheduled_outcome'
SCOPE=dict(synthetic_inputs_read=False,independent_synthetic_reference=False,labels_read=False,
    general_numerical_accuracy_established=False,efficacy_tested=False,candidate_selected=False,
    scientific_gate_passed=False,goal_complete=False)
POLICY=dict(PREVIOUS.POLICY)
REFERENCE_POLICY=dict(PREVIOUS.REFERENCE_POLICY,max_total_cell_node_evaluations=36000000000,max_phase_wall_seconds=10800.0)
GATES=dict(PREVIOUS.GATES)
BINDINGS=('results/round32_log_conditional_root_acceptance_20261031.json',
 'results/round32_log_conditional_post_result_decision_20261101.json',
 'docs/round33_log_real_reference_design.md','results/round30_log_rms_root_acceptance_20261031.json')
SOURCE_FILES=('ROUND33_LOG_REAL_REFERENCE_PROTOCOL.md','docs/round33_log_real_reference_design.md',
 'experiments/round33_log_real_contract.py','experiments/round33_log_real_preflight.py','experiments/round33_log_real_audit.py',
 'tests/test_round33_log_real_preflight.py','tests/test_round33_log_real_audit.py',
 'experiments/round32_log_conditional_reference.py','models/round29_log_rms_projective_pe.py',
 'docs/round29_log_rms_representation.md','docs/round26_rms_quadrature_derivation.md')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')


def array_sha(value):
    value = np.ascontiguousarray(value)
    header = json.dumps(dict(dtype=str(value.dtype), shape=list(value.shape)),
                        sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(header+b'\n'+value.tobytes()).hexdigest()


def source_names():
    found, pending = set(), list(SOURCE_FILES)
    while pending:
        name = pending.pop()
        if name in found:
            continue
        path = ROOT/name
        if not path.is_file():
            raise ValueError('missing joint source: '+name)
        found.add(name)
        if path.suffix != '.py':
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [n.name for n in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    raise ValueError('relative import requires explicit source review')
                if node.module:
                    names = [node.module]+[node.module+'.'+n.name for n in node.names]
            for module in names:
                parts = module.split('.')
                paths = [Path(*parts).with_suffix('.py'), Path(*parts)/'__init__.py']
                paths += [Path(*parts[:i])/'__init__.py' for i in range(1, len(parts))]
                pending.extend(str(p) for p in paths if (ROOT/p).is_file() and str(p) not in found)
    return sorted(found)


def runtime():
    builds = {}
    for name, function in (('numpy', np.show_config), ('scipy', scipy.show_config)):
        stream = io.StringIO()
        with redirect_stdout(stream):
            function()
        builds[name+'_build_sha256'] = hashlib.sha256(stream.getvalue().encode()).hexdigest()
    return dict(python=sys.version, executable=sys.executable, numpy=np.__version__, scipy=scipy.__version__,
                platform=platform.platform(), threads={n:os.environ.get(n) for n in THREAD_ENV},
                threadpools=threadpool_info(), **builds)


def check_runtime():
    result = runtime()
    if any(result['threads'][k] != '4' for k in THREAD_ENV):
        raise ValueError('the frozen OMP/MKL/OpenBLAS=4 environment is required')
    if any(p['num_threads'] != 4 for p in result['threadpools']):
        raise ValueError('loaded numerical thread pools must use four threads')
    return result


def inventory(directory):
    return {str(p.relative_to(directory)):sha(p) for p in sorted(directory.rglob('*')) if p.is_file()}


def verify_inventory(directory, files, exclude=()):
    actual = {str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file()}
    if actual != set(files) | set(exclude):
        raise ValueError('inventory population differs: '+str(directory))
    for name, digest in files.items():
        if sha(directory/name) != digest:
            raise ValueError('inventory digest differs: '+name)


def inherited_evidence():
    """Authenticate metadata and file bytes only; return all72 R30 source specs."""
    bindings,_,_=PREVIOUS.inherited_evidence()
    bindings.update({str(ROOT/n):sha(ROOT/n) for n in BINDINGS})
    accepted=read(ROOT/BINDINGS[0]);real=read(ROOT/BINDINGS[3])
    if (accepted.get('accepted_for_numerical_interpretation') is not True
            or accepted.get('numerical_preflight_passed') is not True
            or accepted.get('completed_reference_calls')!=46 or accepted.get('comparison_failures')!=0
            or accepted.get('post_seal_serialization_repair') is not True
            or accepted.get('original_formal_pipeline_all_stages_passed') is not False
            or accepted.get('labels_read') is not False or accepted.get('scientific_gate_passed') is not False):
        raise ValueError('accepted repaired R32 numerical pass required')
    prod=ROOT/'results/round32_log_conditional_preflight'
    failed=ROOT/'results/round32_log_conditional_independent_audit'
    repaired=ROOT/accepted['repaired_audit_root']
    for name,digest in dict(accepted['files'],**accepted['source_hashes'],**accepted['repair_source_hashes']).items():
        if sha(ROOT/name)!=digest:raise ValueError('accepted R32 evidence changed: '+name)
        bindings[str(ROOT/name)]=digest
    for name,digest in accepted['source_hashes'].items():
        for directory in (prod,failed,repaired):
            path=directory/'source_snapshot'/name
            if sha(path)!=digest:raise ValueError('original R32 source snapshot changed')
            bindings[str(path)]=digest
    for name,digest in accepted['repair_source_hashes'].items():
        path=repaired/'repair_source_snapshot'/name
        if sha(path)!=digest:raise ValueError('repair source snapshot changed')
        bindings[str(path)]=digest
    for directory in (prod,failed,repaired):
        done=read(directory/'completion.json');verify_inventory(directory,done['files'],exclude=('completion.json',))
        for name,digest in done['files'].items():bindings[str(directory/name)]=digest
        bindings[str(directory/'completion.json')]=sha(directory/'completion.json')
    repairlock=read(repaired/'repair_lock.json')
    if sha(repaired/'repair_lock.json')!=(repaired/'repair_lock.sha256').read_text().strip():raise ValueError('repair lock changed')
    for name,digest in repairlock['extra_bindings'].items():
        if sha(ROOT/name)!=digest:raise ValueError('original failure/controller history changed')
        bindings[str(ROOT/name)]=digest
    for directory,key in ((prod,'producer_files'),(failed,'original_audit_files')):
        verify_inventory(directory,repairlock[key])
    for name,digest in repairlock['copied_original_evidence'].items():
        if sha(failed/name)!=digest or sha(repaired/name)!=digest:raise ValueError('copied original input/source evidence changed')
    if (real.get('accepted_for_applicability_interpretation') is not True
            or real.get('applicability_calls_completed') is not True or real.get('model_completed_calls')!=72
            or real.get('labels_read') is not False or real.get('scientific_gate_passed') is not False):
        raise ValueError('accepted complete R30 applicability/input evidence required')
    decision=read(ROOT/BINDINGS[1]);selected=decision['next_selected_work']
    expected=dict(round='R33',design_path=BINDINGS[2],design_sha256=sha(ROOT/BINDINGS[2]),
        fixed_real_case_count=CASE_COUNT,scheduled_reference_calls=CALL_COUNT,comparison_count=216,reference_orders=[32,48],
        reference_source='experiments/round32_log_conditional_reference.py',
        reference_source_sha256=sha(ROOT/'experiments/round32_log_conditional_reference.py'),
        max_call_cell_node_evaluations=250000000,max_total_cell_node_evaluations=36000000000,
        max_phase_wall_seconds=10800.0,fail_fast=FAIL_FAST,old_real_acceptance=BINDINGS[3],
        old_real_acceptance_sha256=sha(ROOT/BINDINGS[3]),reuse_sealed_R30_inputs_and_production=True,
        producer_rerun_authorized=False,real_prepared_input_arrays_authorized_after_joint_freeze=True,
        labels_authorized=False,efficacy_candidate_selected=False,relax_precision_or_per_call_gates=False)
    if decision['root_acceptance_sha256']!=sha(ROOT/BINDINGS[0]) or any(selected.get(k)!=v for k,v in expected.items()):
        raise ValueError('fixed R33 real-reference selection changed')
    # PREVIOUS already authenticates the complete R30 receipt, current and dual
    # snapshots and full producer/audit inventories. Now join every canonical row.
    oldlock=read(OLD_RUN/'lock.json');oldspecs=read(OLD_RUN/'query_specs.json')
    prepared=read(OLD_RUN/'prepared/records.json');forward=read(OLD_RUN/'forward/records.json')
    checks=read(OLD_REAL_AUDIT/'input_audit/checks.json');gate=read(OLD_REAL_AUDIT/'input_verification.json')
    ids=[row['id'] for row in prepared]
    if (len(ids)!=CASE_COUNT or len(set(ids))!=CASE_COUNT or ids!=oldlock['case_ids']
            or any([row['id'] for row in population]!=ids for population in (oldspecs,forward,checks))
            or not gate['all_input_bytes_exact'] or not all(c['passed'] for c in checks)):
        raise ValueError('complete accepted R30 population/input gate differs')
    specs=[]
    for prep,row,spec,check in zip(prepared,forward,oldspecs,checks):
        if (not prep['passed'] or prep['status']!='prepared' or not row['passed'] or not row['attempted']
                or row['status']!='complete' or set(prep['array_sha256'])!=set(OLD_FIELDS)
                or row['prepared_archive_path']!=prep['archive_path'] or row['prepared_archive_sha256']!=prep['archive_sha256']
                or check['prepared_archive_sha256']!=prep['archive_sha256']
                or any(any(item.get(k)!=prep[k] for k in IDENTITY_FIELDS) for item in (row,spec,check))
                or any(row['array_sha256'].get(k)!=prep['array_sha256'][k] for k in OLD_FIELDS)):
            raise ValueError('accepted R30 prepared/production join changed')
        source=OLD_RUN/prep['archive_path'];production=OLD_RUN/row['archive_path']
        if (source.parent!=OLD_RUN/'prepared/arrays' or production.parent!=OLD_RUN/'forward/arrays'
                or sha(source)!=prep['archive_sha256'] or sha(production)!=row['archive_sha256']):
            raise ValueError('R30 archive identity changed')
        specs.append(dict({k:prep[k] for k in IDENTITY_FIELDS},policy_overrides={},
            source_archive_path=str(source),source_archive_sha256=prep['archive_sha256'],
            array_sha256=prep['array_sha256'],production_archive_path=str(production),
            production_archive_sha256=row['archive_sha256'],production_array_sha256=row['array_sha256'],production_record=row))
    return bindings,specs
