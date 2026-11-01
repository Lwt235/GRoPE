"""R36 shared metadata/runtime contract, with no reference or model arithmetic."""
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
REFERENCE_POLICY={'depth_cutoff': 18.0, 'inner_geometry_breaks': [-8.0, -1.0, 0.0, 1.0, 8.0], 'max_call_cell_node_evaluations': 250000000, 'max_phase_wall_seconds': 1800.0, 'max_total_cell_node_evaluations': 600000000, 'maximum_depth_evaluations': 32768, 'maximum_depth_intervals': 512, 'narrow_cdf_exponent_span': 0.25, 'narrow_cdf_order': 8, 'orders': [32, 48], 'outer_absolute_tolerance': 0.0, 'outer_geometry_breaks': [-8.0, -1.0, 0.0, 1.0, 8.0], 'outer_norm': 'max', 'outer_quadrature': 'gk21', 'outer_relative_tolerances': [1e-11, 1e-12], 'outer_workers': 1, 'standard_x_panel_width': 2.0, 'x_domain': 'complete_finite_cell_interval', 'outer_scaling': 'fixed_initial_interval_midpoint_pilots', 'pilot_reselection_allowed': False}
GATES={'direct_fov_additivity_absolute': 1e-08, 'integrated_identity_absolute': 1e-08, 'near_constant_limit_absolute': 1e-08, 'near_constant_limit_raw_relative': 1e-08, 'normalization_absolute': 2e-12, 'probability_max_absolute': 0.0001, 'production_max_phase_wall_seconds': 600.0, 'raw_fov_relative': 1e-06, 'raw_l1_over_reference_fov': 1e-06, 'reference_probability_max_absolute': 1e-08, 'reference_raw_relative': 1e-08, 'truncation_probability_absolute': 1e-08, 'truncation_raw_l1_relative': 1e-10}
BASE_IDS=('01_point', '02_constant_axis', '03_constant_correlated', '04_zero_depth', '05_negative_constant', '06_radial_central', '07_radial_noncentral', '08_full_central', '09_full_noncentral', '10_negative_mean', '11_near_constant', '12_tiny_front', '13_native432', '14_rank2', '15_constant_axis_bias', '16_radial_central_bias', '17_full_noncentral_bias', '18_tiny_front_bias', '25_wide_constant_axis', '26_wide_constant_correlated', '27_native432_far_cells', '28_subfloat_fov', '29_visible_far_tiles')
IDENTITY_IDS=('19_world_scale', '20_translation', '21_screen_units', '22_axis_rotation', '23_key_permutation', '24_unequal_merge')
BINDINGS=('results/round35_log_real_root_acceptance_20261101.json',
    'results/round35_log_real_post_result_decision_20261101.json',
    'docs/round36_component_round_reference_design.md',
    'docs/round35_reference_roundoff_diagnosis.md')
SOURCE_FILES=('ROUND36_COMPONENT_ROUND_REFERENCE_PROTOCOL.md','docs/round36_component_round_reference_design.md',
    'experiments/round36_component_round_contract.py','experiments/round36_component_round_reference.py',
    'experiments/round36_component_round_preflight.py','experiments/round36_component_round_audit.py',
    'tests/test_round36_component_round_reference.py','tests/test_round36_component_round_preflight.py',
    'tests/test_round36_component_round_audit.py','docs/round29_log_rms_representation.md',
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


from experiments import round35_log_real_contract as PREVIOUS
from experiments import round34_log_narrow_contract as SYNTHETIC
CASE_COUNT=23
CALL_COUNT=46
ORACLE_COUNT=135
COMPARISON_COUNT=339
NARROW_GL8_RELATIVE_REMAINDER_BOUND=1.88e-13
OUTER_ERROR_AGGREGATION='sum_active_scalar_truncation_plus_max_historical_component_roundoff'
OUTER_INTEGRAL_ACCUMULATION='current_leaf_component_fsum'
OUTER_REFINEMENT='single_largest_floored_error_leftmost_tie'
REFERENCE_POLICY=dict(SYNTHETIC.REFERENCE_POLICY,
    outer_error_aggregation=OUTER_ERROR_AGGREGATION,
    outer_integral_accumulation=OUTER_INTEGRAL_ACCUMULATION,
    outer_refinement=OUTER_REFINEMENT)


def inherited_evidence():
    """Metadata/byte authentication only; no input or output array decoding."""
    bindings,_,_=PREVIOUS.inherited_evidence()
    old_bindings,specs,old_oracles=SYNTHETIC.inherited_evidence()
    bindings.update(old_bindings)
    bindings.update({str(ROOT/n):sha(ROOT/n) for n in BINDINGS})
    receipt=read(ROOT/BINDINGS[0])
    expected=dict(accepted_for_numerical_interpretation=True,
        accepted_for_failure_interpretation=True,delivery_integrity_passed=True,
        numerical_preflight_passed=False,fixed_real_case_count=72,
        scheduled_reference_calls=144,actual_reference_calls=22,
        completed_reference_calls=21,executed_failed_calls=1,
        fail_fast_skipped_calls=122,input_delivery_failure_count=0,
        comparison_failures=185,labels_read=False,scientific_gate_passed=False,goal_complete=False)
    if any(receipt.get(k)!=v for k,v in expected.items()):
        raise ValueError('accepted complete R35 rounding failure population required')
    producer=ROOT/'results/round35_log_real_reference'
    audit=ROOT/'results/round35_log_real_reference_independent_audit'
    for name,digest in dict(receipt['files'],**receipt['source_hashes']).items():
        if sha(ROOT/name)!=digest:raise ValueError('R35 evidence changed: '+name)
        bindings[str(ROOT/name)]=digest
    for name,digest in receipt['source_hashes'].items():
        for directory in (producer,audit):
            snapshot=directory/'source_snapshot'/name
            if sha(snapshot)!=digest:raise ValueError('R35 source snapshot changed')
            bindings[str(snapshot)]=digest
    for directory in (producer,audit):
        done=read(directory/'completion.json')
        verify_inventory(directory,done['files'],exclude=('completion.json',))
        bindings.update({str(directory/n):h for n,h in done['files'].items()})
        bindings[str(directory/'completion.json')]=sha(directory/'completion.json')
    decision=read(ROOT/BINDINGS[1]);selected=decision['next_selected_work']
    expected=dict(round='R36',design_path=BINDINGS[2],design_sha256=sha(ROOT/BINDINGS[2]),
        synthetic_base_count=CASE_COUNT,new_reference_calls=CALL_COUNT,
        inherited_oracle_count=ORACLE_COUNT,comparison_count=COMPARISON_COUNT,
        reference_orders=[32,48],narrow_cdf_order=8,narrow_cdf_exponent_span=.25,
        local_exact_arithmetic_relative_bound=NARROW_GL8_RELATIVE_REMAINDER_BOUND,
        outer_error_aggregation=OUTER_ERROR_AGGREGATION,
        outer_integral_accumulation=OUTER_INTEGRAL_ACCUMULATION,outer_refinement=OUTER_REFINEMENT,
        reuse_sealed_R29_production_and_reference_outputs=True,
        reuse_all_accepted_R32_reference_outputs=True,reuse_all_accepted_R34_reference_outputs=True,
        producer_rerun_authorized=False,real_input_arrays_authorized=False,
        labels_authorized=False,efficacy_candidate_selected=False,relax_existing_gates_or_budgets=False)
    if (decision['root_acceptance_sha256']!=sha(ROOT/BINDINGS[0])
            or decision['diagnostic_path']!=BINDINGS[3]
            or decision['diagnostic_sha256']!=sha(ROOT/BINDINGS[3])
            or any(selected.get(k)!=v for k,v in expected.items())):
        raise ValueError('R36 reference-only prospective selection changed')
    diagnostic=ROOT/'results/round35_roundoff_postfailure_diagnostic_20261101'
    delivery=read(diagnostic/'execution_receipt.json')
    verify_inventory(diagnostic,delivery['all_files'],exclude=('execution_receipt.json',))
    result=read(diagnostic/'diagnostic.json');before=read(diagnostic/'before_execution.json')
    if (delivery['exit_code']!=0 or not delivery['exactly_one_reference_integration']
            or delivery['source_before_execution_sha256']!=sha(diagnostic/'before_execution.json')
            or delivery['diagnostic_sha256']!=sha(diagnostic/'diagnostic.json')
            or delivery['completion_sha256']!=sha(diagnostic/'completion.json')
            or delivery['stdout_stderr_sha256']!=sha(diagnostic/'execution.log')
            or decision['diagnostic_receipt_sha256']!=sha(diagnostic/'execution_receipt.json')
            or decision['diagnostic_result_path']!=str((diagnostic/'diagnostic.json').relative_to(ROOT))
            or decision['diagnostic_result_sha256']!=sha(diagnostic/'diagnostic.json')
            or result['fixed_call_id']!='harvard_c3__hv_c3_1__g16__q0030__order48'
            or not result['original_failure_exactly_reproduced']
            or not all(result['all_saved_partial_array_bytes_reproduced'].values())
            or not result['scalar_floor_prevents_success_at_saved_partition']
            or not result['no_new_probability_or_reference_oracle']
            or any(result[k] for k in ('labels_read','efficacy_tested','candidate_selected','scientific_gate_passed','goal_complete'))):
        raise ValueError('accepted observation-only roundoff diagnosis changed')
    for name,digest in before['files'].items():
        if sha(name)!=digest:raise ValueError('diagnostic prior binding changed: '+name)
        bindings[name]=digest
    bindings.update({str(diagnostic/n):h for n,h in delivery['all_files'].items()})
    bindings[str(diagnostic/'execution_receipt.json')]=sha(diagnostic/'execution_receipt.json')
    # PREVIOUS authenticated all R34 accepted evidence and dual source snapshots.
    old_run=ROOT/'results/round34_log_narrow_preflight'
    old_audit=ROOT/'results/round34_log_narrow_independent_audit'
    rows=read(old_run/'forward/records.json');checks=read(old_audit/'checks.json')
    schedule=[(s['id'],o) for s in specs for o in (32,48)]
    if (len(specs)!=CASE_COUNT or [s['id'] for s in specs]!=list(BASE_IDS)
            or len(old_oracles)!=89 or len(rows)!=CALL_COUNT or len(checks)!=CALL_COUNT
            or [(r['id'],r['order']) for r in rows]!=schedule
            or [(c['id'],c['order']) for c in checks]!=schedule):
        raise ValueError('fixed synthetic/oracle populations changed')
    oracles=[dict(o) for o in old_oracles];specmap={s['id']:s for s in specs}
    for row,check in zip(rows,checks):
        if (row['status']!='complete' or not row['attempted'] or not row['passed']
                or not check['delivery_integrity_passed'] or not check['arithmetic_passed']
                or not check['arithmetic']['passed']
                or any(row['array_sha256'].get(k)!=specmap[row['id']]['array_sha256'][k] for k in OLD_FIELDS)):
            raise ValueError('R34 accepted oracle/input identity changed')
        source=old_run/row['archive_path']
        if source.parent!=old_run/'forward/arrays' or sha(source)!=row['archive_sha256']:
            raise ValueError('R34 accepted oracle bytes changed')
        kind='r34_log_narrow'
        oracles.append(dict(id=row['id'],order=row['order'],reference_kind=kind,
            oracle_id=kind+'__order'+str(row['order']),reference_record=row,
            source_archive_path=str(source),source_archive_sha256=row['archive_sha256'],
            array_sha256=row['array_sha256'],source_array_sha256=row['array_sha256'],
            source_fields={n:n for n in ('log_raw_mass','log_raw_fov_mass','probability')}))
    if len(oracles)!=ORACLE_COUNT or len({(o['id'],o['oracle_id']) for o in oracles})!=ORACLE_COUNT:
        raise ValueError('fixed oracle sources/orders missing or aliased')
    return bindings,specs,oracles

