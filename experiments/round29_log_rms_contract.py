"""R29 synthetic-only canonical log-mass contract and shared provenance helpers.

Only metadata/file hashes are read during freeze. No real array decoding,
production density, quadrature or independent reference is implemented here.
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
POLICY = dict(h=.25, floor_density=1e-30, quadrature_order=24,
              maximum_tile_side_in_h=2., max_cell_node_evaluations=4_000_000,
              negative_tail_threshold=10., negative_tail_series_terms=20,
              max_negative_tail_relative_remainder=1e-10,
              projection_regularization='rms_depth_xy_gaussian')
REFERENCE_POLICY = dict(orders=[32,48], outer_relative_tolerances=[1e-11,1e-12],
    outer_absolute_tolerance=0., depth_cutoff=18., inner_x_cutoff=32., standard_x_panel_width=2.,
    outer_quadrature='gk21', outer_norm='max', outer_workers=1,
    maximum_depth_evaluations=32768, maximum_depth_intervals=512,
    outer_geometry_breaks=[-8.,-1.,0.,1.,8.], inner_geometry_breaks=[-8.,-1.,0.,1.,8.],
    narrow_cdf_order=16, narrow_cdf_exponent_span=.25,
    max_call_cell_node_evaluations=250_000_000,
    max_total_cell_node_evaluations=600_000_000, max_phase_wall_seconds=1800.)
GATES = dict(probability_max_absolute=1e-4, raw_fov_relative=1e-6,
    raw_l1_over_reference_fov=1e-6, reference_probability_max_absolute=1e-8,
    reference_raw_relative=1e-8, normalization_absolute=2e-12,
    direct_fov_additivity_absolute=1e-8, integrated_identity_absolute=1e-8,
    near_constant_limit_absolute=1e-8, near_constant_limit_raw_relative=1e-8,
    truncation_raw_l1_relative=1e-10, truncation_probability_absolute=1e-8,
    production_max_phase_wall_seconds=600.)
BINDINGS=('results/round28_rms_root_acceptance_20261031.json',
    'results/round28_rms_post_result_decision_20261031.json','docs/round29_log_rms_design.md',
    'results/round27_rms_root_acceptance_20261031.json')
SOURCE_FILES=('ROUND29_LOG_RMS_PREFLIGHT_PROTOCOL.md','docs/round29_log_rms_representation.md',
    'docs/round26_rms_quadrature_derivation.md','docs/round27_reference_panel_derivation.md',
    'experiments/round29_log_rms_contract.py','experiments/round29_log_rms_fixtures.py',
    'experiments/round29_log_rms_preflight.py','experiments/round29_log_rms_reference.py',
    'experiments/round29_log_rms_audit.py','models/round29_log_rms_projective_pe.py',
    'tests/test_round29_log_rms_projective_pe.py','tests/test_round29_log_rms_preflight.py',
    'tests/test_round29_log_rms_reference.py','tests/test_round29_log_rms_audit.py')

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


def inherited_bindings():
    bindings={str(ROOT/n):sha(ROOT/n) for n in BINDINGS}
    previous=read(ROOT/BINDINGS[0]);synthetic=read(ROOT/BINDINGS[3])
    if (previous.get('accepted_for_applicability_interpretation') is not True
            or previous.get('applicability_calls_completed') is not False
            or previous.get('fixed_query_count')!=72 or previous.get('model_completed_calls')!=40
            or previous.get('scientific_gate_passed') is not False
            or previous.get('labels_read') is not False):
        raise ValueError('accepted R28 applicability failure required')
    if (synthetic.get('accepted_for_numerical_interpretation') is not True
            or synthetic.get('numerical_preflight_passed') is not True
            or synthetic.get('scientific_gate_passed') is not False):
        raise ValueError('accepted unchanged R27 synthetic baseline required')
    for receipt,producer,audit in ((previous,'results/round28_rms_applicability','results/round28_rms_applicability_independent_audit'),
            (synthetic,'results/round27_rms_preflight','results/round27_rms_independent_audit')):
        for name,digest in dict(receipt['files'],**receipt['source_hashes']).items():
            if Path(name).suffix=='.npz':raise ValueError('metadata receipt must not require array access')
            if sha(ROOT/name)!=digest:raise ValueError('accepted predecessor metadata/source changed: '+name)
            bindings[str(ROOT/name)]=digest
        for name,digest in receipt['source_hashes'].items():
            for snapshot in (ROOT/producer/'source_snapshot'/name,ROOT/audit/'source_snapshot'/name):
                if sha(snapshot)!=digest:raise ValueError('accepted predecessor source snapshot changed')
                bindings[str(snapshot)]=digest
    decision=read(ROOT/BINDINGS[1]);selected=decision['next_selected_work']
    if (decision['root_acceptance_sha256']!=sha(ROOT/BINDINGS[0]) or selected['round']!='R29'
            or selected['design_path']!=BINDINGS[2] or selected['design_sha256']!=sha(ROOT/BINDINGS[2])
            or selected['real_input_arrays_authorized'] is not False or selected['labels_authorized'] is not False
            or selected['efficacy_candidate_selected'] is not False or selected['relax_existing_gates'] is not False):
        raise ValueError('R29 synthetic-only selection changed')
    return bindings
