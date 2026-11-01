"""R28 constants and shared provenance/runtime helpers; no numerical model code.

Provenance helpers are intentionally shared. Producer mapping and the audit's
componentwise geometry/arithmetic checks are independently implemented.
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
OLD_RUN=ROOT/'results/round23_ray_applicability'
OLD_AUDIT=ROOT/'results/round23_ray_applicability_independent_audit'
OLD_FIELDS=('input__mu','input__factor','input__bounds','input__H','input__offset',
    'target__covariance','query_bounds_px','query_index')
BINDINGS=('results/round27_rms_root_acceptance_20261031.json',
    'results/round27_rms_post_result_decision_20261031.json',
    'docs/round28_rms_applicability_design.md','results/round23_ray_root_acceptance_20261031.json')
SOURCE_FILES=('ROUND28_RMS_APPLICABILITY_PROTOCOL.md','experiments/round28_rms_contract.py',
    'experiments/round28_rms_applicability.py','experiments/round28_rms_applicability_audit.py',
    'models/round26_rms_projective_pe.py','docs/round26_rms_quadrature_derivation.md',
    'tests/test_round26_rms_projective_pe.py','tests/test_round28_rms_applicability.py',
    'tests/test_round28_rms_applicability_audit.py')

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
    """Read metadata/hash bytes only. Never decode any old observation archive."""
    bindings={str(ROOT/n):sha(ROOT/n) for n in BINDINGS}
    receipt=read(ROOT/BINDINGS[0]);old_receipt=read(ROOT/BINDINGS[3])
    if (receipt.get('accepted_for_numerical_interpretation') is not True
            or receipt.get('numerical_preflight_passed') is not True
            or receipt.get('scientific_gate_passed') is not False):
        raise ValueError('accepted R27 synthetic numerical pass required')
    for name,digest in dict(receipt['files'],**receipt['source_hashes']).items():
        if sha(ROOT/name)!=digest:raise ValueError('accepted R27 evidence/source changed: '+name)
        bindings[str(ROOT/name)]=digest
    decision=read(ROOT/BINDINGS[1]);selection=decision['next_selected_work']
    if (decision['root_acceptance_sha256']!=sha(ROOT/BINDINGS[0]) or selection['round']!='R28'
            or selection['design_path']!=BINDINGS[2] or selection['design_sha256']!=sha(ROOT/BINDINGS[2])
            or selection['old_input_acceptance']!=BINDINGS[3]
            or selection['old_input_acceptance_sha256']!=sha(ROOT/BINDINGS[3])
            or selection['real_prepared_input_arrays_authorized_after_joint_freeze'] is not True
            or selection['labels_authorized'] is not False or selection['efficacy_candidate_selected'] is not False
            or selection['producer_model_sha256']!=sha(ROOT/selection['producer_model'])):
        raise ValueError('R28 restricted applicability selection changed')
    if (old_receipt.get('accepted_for_applicability_interpretation') is not True
            or old_receipt.get('independent_input_audit_before_all_forward_calls') is not True
            or old_receipt.get('fixed_query_count')!=72 or old_receipt.get('labels_read') is not False
            or old_receipt.get('applicability_calls_completed') is not False):
        raise ValueError('accepted R23 input population and failed applicability required')
    for name,digest in old_receipt['files'].items():
        if sha(ROOT/name)!=digest:raise ValueError('accepted R23 evidence changed: '+name)
        bindings[str(ROOT/name)]=digest
    for directory in (OLD_RUN,OLD_AUDIT):
        complete=read(directory/'completion.json')
        verify_inventory(directory,complete['files'],exclude=('completion.json',))
        for name,digest in complete['files'].items():bindings[str(directory/name)]=digest
        bindings[str(directory/'completion.json')]=sha(directory/'completion.json')
    old_lock=read(OLD_RUN/'lock.json')
    for name,digest in old_lock['source_hashes'].items():
        for location in (ROOT/name,OLD_RUN/'source_snapshot'/name,OLD_AUDIT/'source_snapshot'/name):
            if sha(location)!=digest:raise ValueError('inherited R23 source/snapshot changed')
            bindings[str(location)]=digest
    prepared=read(OLD_RUN/'prepared_complete.json')
    records=read(OLD_RUN/'prepared/records.json');old_specs=read(OLD_RUN/'query_specs.json')
    prepared_completion=read(OLD_RUN/'prepared/completion.json')
    verify_inventory(OLD_RUN/'prepared',prepared_completion['files'],exclude=('completion.json',))
    checks=read(OLD_AUDIT/'input_audit/checks.json');verification=read(OLD_AUDIT/'input_verification.json')
    input_completion=read(OLD_AUDIT/'input_audit/completion.json')
    verify_inventory(OLD_AUDIT/'input_audit',input_completion['files'],exclude=('completion.json',))
    ids=[r['id'] for r in records]
    if (len(ids)!=72 or len(set(ids))!=72 or ids!=prepared['case_ids'] or ids!=[r['id'] for r in old_specs]
            or ids!=[r['id'] for r in checks] or not all(r['passed'] and r['exact_H_offset_bounds'] for r in checks)
            or not all(r['passed'] and r['status']=='prepared' for r in records)
            or not prepared['all_inputs_prepared'] or prepared['case_count']!=72
            or not prepared_completion['all_inputs_prepared'] or not prepared_completion['complete_case_inventory']
            or not verification['passed'] or verification['checked_queries']!=72
            or verification['labels_read'] is not False or verification['independent_probability_reference'] is not False
            or prepared['prepared_records_sha256']!=sha(OLD_RUN/'prepared/records.json')
            or prepared['prepared_completion_sha256']!=sha(OLD_RUN/'prepared/completion.json')
            or prepared['query_specs_sha256']!=sha(OLD_RUN/'query_specs.json')
            or verification['producer_prepared_complete_sha256']!=sha(OLD_RUN/'prepared_complete.json')
            or verification['producer_prepared_completion_sha256']!=sha(OLD_RUN/'prepared/completion.json')
            or input_completion['verification_sha256']!=sha(OLD_AUDIT/'input_verification.json')):
        raise ValueError('complete accepted R23 input-audit chain differs')
    specs=[]
    for row in records:
        archive=OLD_RUN/row['archive_path']
        if sha(archive)!=row['archive_sha256'] or not set(OLD_FIELDS)<=set(row['array_sha256']):
            raise ValueError('fixed R23 prepared archive identity changed')
        if archive.parent!=OLD_RUN/'prepared/arrays':raise ValueError('unfrozen old archive location')
        specs.append(dict(row,source_archive_path=str(archive)))
        bindings[str(archive)]=row['archive_sha256']
    return bindings,specs
