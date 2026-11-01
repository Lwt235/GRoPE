"""R32 shared metadata/runtime contract, with no reference or model arithmetic."""
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
OLD_RUN=ROOT/'results/round29_log_rms_preflight'
OLD_AUDIT=ROOT/'results/round29_log_rms_independent_audit'
OLD_FIELDS=('input__mu','input__factor','input__bounds','input__content_bias')
SCOPE=dict(real_inputs_read=False,labels_read=False,general_numerical_accuracy_established=False,
           efficacy_tested=False,candidate_selected=False,scientific_gate_passed=False,goal_complete=False)
POLICY={'floor_density': 1e-30, 'h': 0.25, 'max_cell_node_evaluations': 4000000, 'max_negative_tail_relative_remainder': 1e-10, 'maximum_tile_side_in_h': 2.0, 'negative_tail_series_terms': 20, 'negative_tail_threshold': 10.0, 'projection_regularization': 'rms_depth_xy_gaussian', 'quadrature_order': 24}
REFERENCE_POLICY={'depth_cutoff': 18.0, 'inner_geometry_breaks': [-8.0, -1.0, 0.0, 1.0, 8.0], 'max_call_cell_node_evaluations': 250000000, 'max_phase_wall_seconds': 1800.0, 'max_total_cell_node_evaluations': 600000000, 'maximum_depth_evaluations': 32768, 'maximum_depth_intervals': 512, 'narrow_cdf_exponent_span': 0.25, 'narrow_cdf_order': 16, 'orders': [32, 48], 'outer_absolute_tolerance': 0.0, 'outer_geometry_breaks': [-8.0, -1.0, 0.0, 1.0, 8.0], 'outer_norm': 'max', 'outer_quadrature': 'gk21', 'outer_relative_tolerances': [1e-11, 1e-12], 'outer_workers': 1, 'standard_x_panel_width': 2.0, 'x_domain': 'complete_finite_cell_interval', 'outer_scaling': 'fixed_initial_interval_midpoint_pilots', 'pilot_reselection_allowed': False}
GATES={'direct_fov_additivity_absolute': 1e-08, 'integrated_identity_absolute': 1e-08, 'near_constant_limit_absolute': 1e-08, 'near_constant_limit_raw_relative': 1e-08, 'normalization_absolute': 2e-12, 'probability_max_absolute': 0.0001, 'production_max_phase_wall_seconds': 600.0, 'raw_fov_relative': 1e-06, 'raw_l1_over_reference_fov': 1e-06, 'reference_probability_max_absolute': 1e-08, 'reference_raw_relative': 1e-08, 'truncation_probability_absolute': 1e-08, 'truncation_raw_l1_relative': 1e-10}
BASE_IDS=('01_point', '02_constant_axis', '03_constant_correlated', '04_zero_depth', '05_negative_constant', '06_radial_central', '07_radial_noncentral', '08_full_central', '09_full_noncentral', '10_negative_mean', '11_near_constant', '12_tiny_front', '13_native432', '14_rank2', '15_constant_axis_bias', '16_radial_central_bias', '17_full_noncentral_bias', '18_tiny_front_bias', '25_wide_constant_axis', '26_wide_constant_correlated', '27_native432_far_cells', '28_subfloat_fov', '29_visible_far_tiles')
IDENTITY_IDS=('19_world_scale', '20_translation', '21_screen_units', '22_axis_rotation', '23_key_permutation', '24_unequal_merge')
BINDINGS=('results/round31_log_conditional_root_acceptance_20261031.json',
    'results/round31_log_conditional_post_result_decision_20261031.json',
    'docs/round32_log_conditional_reference_design.md',
    'results/round31_midpoint_post_failure_diagnostic_20261031/diagnostic.json')
SOURCE_FILES=('ROUND32_LOG_CONDITIONAL_REFERENCE_PROTOCOL.md','docs/round32_log_conditional_reference_design.md',
    'experiments/round32_log_conditional_contract.py','experiments/round32_log_conditional_reference.py',
    'experiments/round32_log_conditional_preflight.py','experiments/round32_log_conditional_audit.py',
    'tests/test_round32_log_conditional_reference.py','tests/test_round32_log_conditional_preflight.py',
    'tests/test_round32_log_conditional_audit.py','docs/round29_log_rms_representation.md',
    'docs/round26_rms_quadrature_derivation.md','models/round29_log_rms_projective_pe.py')

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


from experiments import round31_log_conditional_contract as PREVIOUS
REFERENCE_POLICY = dict(REFERENCE_POLICY, initial_depth_partition='strict_midpoint_greedy_coalescing')

def inherited_evidence():
    """Authenticate all accepted ancestry and the R31 failure, metadata only."""
    bindings,specs,oracles=PREVIOUS.inherited_evidence()
    bindings.update({str(ROOT/n):sha(ROOT/n) for n in BINDINGS})
    receipt=read(ROOT/BINDINGS[0])
    if (receipt.get('accepted_for_numerical_interpretation') is not True
            or receipt.get('numerical_preflight_passed') is not False
            or receipt.get('completed_reference_calls')!=44
            or receipt.get('actual_reference_calls')!=46
            or receipt.get('comparison_failures')!=7
            or receipt.get('labels_read') is not False
            or receipt.get('real_inputs_read') is not False
            or receipt.get('scientific_gate_passed') is not False):
        raise ValueError('accepted R31 complete numerical failure required')
    producer=ROOT/'results/round31_log_conditional_preflight'
    audit=ROOT/'results/round31_log_conditional_independent_audit'
    for name,digest in dict(receipt['files'],**receipt['source_hashes']).items():
        if sha(ROOT/name)!=digest:raise ValueError('R31 accepted bytes changed: '+name)
        bindings[str(ROOT/name)]=digest
    for name,digest in receipt['source_hashes'].items():
        for snapshot in (producer/'source_snapshot'/name,audit/'source_snapshot'/name):
            if sha(snapshot)!=digest:raise ValueError('R31 source snapshot changed')
            bindings[str(snapshot)]=digest
    for directory in (producer,audit):
        done=read(directory/'completion.json')
        verify_inventory(directory,done['files'],exclude=('completion.json',))
        for name,digest in done['files'].items():bindings[str(directory/name)]=digest
        bindings[str(directory/'completion.json')]=sha(directory/'completion.json')
    decision=read(ROOT/BINDINGS[1]);selected=decision['next_selected_work']
    expected=dict(round='R32',design_path=BINDINGS[2],design_sha256=sha(ROOT/BINDINGS[2]),
        synthetic_base_count=23,new_reference_calls=46,reference_orders=[32,48],comparison_count=155,
        reuse_sealed_R29_production_and_reference_outputs=True,producer_rerun_authorized=False,
        real_input_arrays_authorized=False,labels_authorized=False,efficacy_candidate_selected=False,
        relax_existing_gates=False)
    if (decision['root_acceptance_sha256']!=sha(ROOT/BINDINGS[0])
            or decision['diagnostic_path']!=BINDINGS[3]
            or decision['diagnostic_sha256']!=sha(ROOT/BINDINGS[3])
            or any(selected.get(k)!=v for k,v in expected.items())):
        raise ValueError('R32 synthetic-only selection changed')
    return bindings,specs,oracles
