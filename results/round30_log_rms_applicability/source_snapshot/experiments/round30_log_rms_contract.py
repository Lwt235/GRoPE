"""R30 constants and shared provenance/runtime helpers; no numerical model code.

Provenance helpers are intentionally shared. Producer copying and the audit's
independent input identity/output arithmetic checks are independently implemented.
"""
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
POLICY=dict(h=.25,floor_density=1e-30,quadrature_order=24,maximum_tile_side_in_h=2.,
    max_cell_node_evaluations=4_000_000,negative_tail_threshold=10.,negative_tail_series_terms=20,
    max_negative_tail_relative_remainder=1e-10,projection_regularization='rms_depth_xy_gaussian')
GATES=dict(input_normalized_absolute=1e-12,normalization_absolute=2e-12,
    direct_fov_additivity_absolute=1e-8,production_max_phase_wall_seconds=600.)
OLD_RUN=ROOT/'results/round28_rms_applicability'
OLD_AUDIT=ROOT/'results/round28_rms_applicability_independent_audit'
OLD_FIELDS=('input__mu','input__factor','input__bounds','input__content_bias')
BINDINGS=('results/round29_log_rms_root_acceptance_20261031.json',
    'results/round29_log_rms_post_result_decision_20261031.json','docs/round30_log_rms_applicability_design.md',
    'results/round28_rms_root_acceptance_20261031.json')
SOURCE_FILES=('ROUND30_LOG_RMS_APPLICABILITY_PROTOCOL.md','experiments/round30_log_rms_contract.py',
    'experiments/round30_log_rms_applicability.py','experiments/round30_log_rms_applicability_audit.py',
    'models/round29_log_rms_projective_pe.py','docs/round29_log_rms_representation.md',
    'docs/round26_rms_quadrature_derivation.md','tests/test_round29_log_rms_projective_pe.py',
    'tests/test_round30_log_rms_applicability.py','tests/test_round30_log_rms_applicability_audit.py')

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
    """Hash accepted metadata/sealed files; never decode arrays or integrate."""
    bindings={str(ROOT/n):sha(ROOT/n) for n in BINDINGS}
    synthetic=read(ROOT/BINDINGS[0]);old=read(ROOT/BINDINGS[3])
    if (synthetic.get('accepted_for_numerical_interpretation') is not True
            or synthetic.get('numerical_preflight_passed') is not True
            or synthetic.get('scientific_gate_passed') is not False or synthetic.get('labels_read') is not False):
        raise ValueError('accepted R29 fixed synthetic log-numerical pass required')
    if (old.get('accepted_for_applicability_interpretation') is not True
            or old.get('applicability_calls_completed') is not False or old.get('fixed_query_count')!=72
            or old.get('model_completed_calls')!=40 or old.get('labels_read') is not False
            or old.get('independent_input_audit_before_all_forward_calls') is not True):
        raise ValueError('accepted R28 input identity and40/72failure required')
    for receipt,producer,audit in ((synthetic,ROOT/'results/round29_log_rms_preflight',ROOT/'results/round29_log_rms_independent_audit'),
            (old,OLD_RUN,OLD_AUDIT)):
        for name,digest in dict(receipt['files'],**receipt['source_hashes']).items():
            if sha(ROOT/name)!=digest:raise ValueError('accepted predecessor metadata/source changed: '+name)
            bindings[str(ROOT/name)]=digest
        for name,digest in receipt['source_hashes'].items():
            for snapshot in (producer/'source_snapshot'/name,audit/'source_snapshot'/name):
                if sha(snapshot)!=digest:raise ValueError('accepted predecessor source snapshot changed')
                bindings[str(snapshot)]=digest
    decision=read(ROOT/BINDINGS[1]);selection=decision['next_selected_work']
    if (decision['root_acceptance_sha256']!=sha(ROOT/BINDINGS[0]) or selection['round']!='R30'
            or selection['design_path']!=BINDINGS[2] or selection['design_sha256']!=sha(ROOT/BINDINGS[2])
            or selection['old_input_acceptance']!=BINDINGS[3] or selection['old_input_acceptance_sha256']!=sha(ROOT/BINDINGS[3])
            or selection['producer_model']!='models/round29_log_rms_projective_pe.py'
            or selection['producer_model_sha256']!=sha(ROOT/selection['producer_model'])
            or selection['real_prepared_input_arrays_authorized_after_joint_freeze'] is not True
            or selection['labels_authorized'] is not False or selection['efficacy_candidate_selected'] is not False
            or selection['relax_existing_gates'] is not False):
        raise ValueError('fixed R30 applicability-only selection changed')
    for directory in (OLD_RUN,OLD_AUDIT):
        complete=read(directory/'completion.json')
        verify_inventory(directory,complete['files'],exclude=('completion.json',))
        for name,digest in complete['files'].items():bindings[str(directory/name)]=digest
        bindings[str(directory/'completion.json')]=sha(directory/'completion.json')
    prepared=read(OLD_RUN/'prepared_complete.json');done=read(OLD_RUN/'prepared/completion.json')
    verify_inventory(OLD_RUN/'prepared',done['files'],exclude=('completion.json',))
    records=read(OLD_RUN/'prepared/records.json');old_specs=read(OLD_RUN/'query_specs.json')
    checks=read(OLD_AUDIT/'input_audit/checks.json');verification=read(OLD_AUDIT/'input_verification.json')
    input_done=read(OLD_AUDIT/'input_audit/completion.json')
    verify_inventory(OLD_AUDIT/'input_audit',input_done['files'],exclude=('completion.json',))
    ids=[row['id'] for row in records]
    if (len(ids)!=72 or len(set(ids))!=72 or ids!=prepared['case_ids'] or ids!=[row['id'] for row in old_specs]
            or ids!=[row['id'] for row in checks] or not all(row['passed'] for row in checks)
            or not all(row['passed'] and row['status']=='prepared' for row in records)
            or not prepared['all_inputs_prepared'] or prepared['case_count']!=72
            or not done['all_inputs_prepared'] or not done['complete_case_inventory']
            or not verification['passed'] or verification['checked_queries']!=72
            or verification['labels_read'] is not False or verification['independent_probability_reference'] is not False
            or prepared['prepared_records_sha256']!=sha(OLD_RUN/'prepared/records.json')
            or prepared['prepared_completion_sha256']!=sha(OLD_RUN/'prepared/completion.json')
            or prepared['query_specs_sha256']!=sha(OLD_RUN/'query_specs.json')
            or verification['producer_prepared_complete_sha256']!=sha(OLD_RUN/'prepared_complete.json')
            or verification['producer_prepared_completion_sha256']!=sha(OLD_RUN/'prepared/completion.json')
            or input_done['verification_sha256']!=sha(OLD_AUDIT/'input_verification.json')):
        raise ValueError('complete accepted72 R28 input-audit chain differs')
    specs=[]
    for row,spec,check in zip(records,old_specs,checks):
        archive=OLD_RUN/row['archive_path']
        if (sha(archive)!=row['archive_sha256'] or check['prepared_archive_sha256']!=row['archive_sha256']
                or not set(OLD_FIELDS)<=set(row['array_sha256'])
                or any(row[name]!=spec[name] for name in ('id','scene','pair','pair_name','query_size','query_index','query_bounds'))):
            raise ValueError('accepted R28 prepared archive/case identity changed')
        if archive.parent!=OLD_RUN/'prepared/arrays':raise ValueError('unfrozen old archive path')
        specs.append(dict(row,source_archive_path=str(archive),source_archive_sha256=row['archive_sha256']))
        bindings[str(archive)]=row['archive_sha256']
    return bindings,specs
