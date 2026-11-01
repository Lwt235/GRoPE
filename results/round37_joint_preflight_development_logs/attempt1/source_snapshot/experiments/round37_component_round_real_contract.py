"""R37 shared metadata/runtime contract, with no reference or model arithmetic."""
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
from experiments import round36_component_round_contract as PREVIOUS
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
BINDINGS=('results/round36_component_round_root_acceptance_20261101.json',
 'results/round36_component_round_post_result_decision_20261101.json',
 'docs/round37_component_round_real_reference_design.md','results/round35_log_real_root_acceptance_20261101.json')
NARROW_GL8_RELATIVE_REMAINDER_BOUND=PREVIOUS.NARROW_GL8_RELATIVE_REMAINDER_BOUND
ORACLE_COUNT=40
PRIMARY_COMPARISON_COUNT=216
OLD_REFERENCE_COMPARISON_COUNT=80
COMPARISON_COUNT=296
SOURCE_FILES=('ROUND37_COMPONENT_ROUND_REAL_REFERENCE_PROTOCOL.md','docs/round37_component_round_real_reference_design.md',
 'experiments/round37_component_round_real_contract.py','experiments/round37_component_round_real_preflight.py','experiments/round37_component_round_real_audit.py',
 'tests/test_round37_component_round_real_preflight.py','tests/test_round37_component_round_real_audit.py',
 'experiments/round36_component_round_reference.py','models/round29_log_rms_projective_pe.py',
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


from experiments import round35_log_real_contract as REAL_INPUTS


def inherited_evidence():
    """Metadata and byte authentication only; never decode model arrays."""
    bindings,_,_=PREVIOUS.inherited_evidence()
    old_bindings,specs,old_oracles=REAL_INPUTS.inherited_evidence()
    bindings.update(old_bindings)
    bindings.update({str(ROOT/n):sha(ROOT/n) for n in BINDINGS})
    receipt=read(ROOT/BINDINGS[0])
    expected=dict(accepted_for_numerical_interpretation=True,delivery_integrity_passed=True,
        numerical_preflight_passed=True,fixed_base_count=23,scheduled_reference_calls=46,
        actual_reference_calls=46,completed_reference_calls=46,inherited_oracle_count=135,
        comparison_failures=0,reference_check_failures=0,real_inputs_read=False,
        labels_read=False,scientific_gate_passed=False,goal_complete=False,
        precision_and_all_resource_gates_unchanged=True,
        outer_error_aggregation=PREVIOUS.OUTER_ERROR_AGGREGATION,
        outer_integral_accumulation=PREVIOUS.OUTER_INTEGRAL_ACCUMULATION,
        outer_refinement=PREVIOUS.OUTER_REFINEMENT)
    if any(receipt.get(k)!=v for k,v in expected.items()):
        raise ValueError('accepted complete R36 synthetic pass required')
    producer=ROOT/'results/round36_component_round_preflight'
    audit=ROOT/'results/round36_component_round_independent_audit'
    for name,digest in dict(receipt['files'],**receipt['source_hashes']).items():
        if sha(ROOT/name)!=digest:raise ValueError('accepted R36 evidence changed: '+name)
        bindings[str(ROOT/name)]=digest
    for name,digest in receipt['source_hashes'].items():
        for directory in (producer,audit):
            snapshot=directory/'source_snapshot'/name
            if sha(snapshot)!=digest:raise ValueError('R36 source snapshot changed')
            bindings[str(snapshot)]=digest
    for directory in (producer,audit):
        done=read(directory/'completion.json')
        verify_inventory(directory,done['files'],exclude=('completion.json',))
        bindings.update({str(directory/n):h for n,h in done['files'].items()})
        bindings[str(directory/'completion.json')]=sha(directory/'completion.json')
    decision=read(ROOT/BINDINGS[1]);selected=decision['next_selected_work']
    expected=dict(round='R37',design_path=BINDINGS[2],design_sha256=sha(ROOT/BINDINGS[2]),
        fixed_real_case_count=72,scheduled_reference_calls=144,inherited_oracle_count=40,
        primary_comparison_count=216,old_reference_comparison_count=80,comparison_count=296,
        reference_orders=[32,48],reference_source='experiments/round36_component_round_reference.py',
        reference_source_sha256=sha(ROOT/'experiments/round36_component_round_reference.py'),
        reference_helper='experiments/round36_component_round_quadrature.py',
        reference_helper_sha256=sha(ROOT/'experiments/round36_component_round_quadrature.py'),
        max_call_cell_node_evaluations=250000000,max_total_cell_node_evaluations=36000000000,
        max_phase_wall_seconds=10800.0,fail_fast=FAIL_FAST,
        reuse_sealed_R30_inputs_and_production=True,reuse_all19_successful_R33_references=True,
        reuse_all21_successful_R35_references=True,producer_rerun_authorized=False,
        real_prepared_input_arrays_authorized_after_joint_freeze=True,labels_authorized=False,
        efficacy_candidate_selected=False,relax_precision_or_budgets=False)
    if decision['root_acceptance_sha256']!=sha(ROOT/BINDINGS[0]) or any(selected.get(k)!=v for k,v in expected.items()):
        raise ValueError('R37 complete real-reference selection changed')
    diagnostic=ROOT/'results/round36_real_initial_cost_lowerbound_20261101'
    delivery=read(diagnostic/'execution_receipt.json')
    diagnostic_prefix=str(diagnostic.relative_to(ROOT))+'/'
    diagnostic_files={n[len(diagnostic_prefix):]:h for n,h in delivery['files'].items() if n.startswith(diagnostic_prefix)}
    verify_inventory(diagnostic,diagnostic_files,exclude=('execution_receipt.json',))
    for name,digest in delivery['files'].items():
        if sha(ROOT/name)!=digest:raise ValueError('diagnostic delivery binding changed: '+name)
        bindings[str(ROOT/name)]=digest
    result=read(diagnostic/'diagnostic.json');lock=read(diagnostic/'lock.json')
    if (not delivery['accepted_for_cost_interpretation'] or not delivery['delivery_integrity_passed']
            or delivery['budget_feasibility']!='unknown' or delivery['source_integrations']!=0
            or decision['diagnostic_receipt_sha256']!=sha(diagnostic/'execution_receipt.json')
            or decision['diagnostic_result_sha256']!=sha(diagnostic/'diagnostic.json')
            or result['conclusion']!='unknown_no_node_counterexample_established'
            or result['case_count']!=72 or result['full_geometry_case_count']!=72
            or result['guaranteed_node_counterexamples']!=[]
            or result['status_counts']!={'initial_lower_bound_only':72}
            or result['reference_integrations']!=0 or result['model_integrations']!=0
            or result['Gaussian_pdf_evaluations']!=0 or result['Gaussian_cdf_evaluations']!=0
            or result['quadrature_weighted_sums']!=0
            or any(result[k] for k in ('labels_read','efficacy_tested','candidate_selected','scientific_gate_passed','goal_complete'))):
        raise ValueError('accepted geometry-only necessary-cost diagnostic changed')
    for field in ('bindings','local_source_hashes'):
        for name,digest in lock[field].items():
            if sha(name)!=digest:raise ValueError('diagnostic frozen binding changed: '+name)
            bindings[name]=digest
    bindings[str(diagnostic/'execution_receipt.json')]=sha(diagnostic/'execution_receipt.json')
    # PREVIOUS independently authenticates all R35 evidence, including failures.
    old_run=ROOT/'results/round35_log_real_reference'
    old_audit=ROOT/'results/round35_log_real_reference_independent_audit'
    rows=read(old_run/'forward/records.json');checks=read(old_audit/'checks.json')
    expected_calls=[s['id']+'__order'+str(order) for s in specs for order in (32,48)]
    if (len(specs)!=CASE_COUNT or len(old_oracles)!=19
            or len(rows)!=CALL_COUNT or len(checks)!=CALL_COUNT
            or [r['call_id'] for r in rows]!=expected_calls
            or [c['call_id'] for c in checks]!=expected_calls):
        raise ValueError('fixed real and inherited oracle population ordering changed')
    success=[(r,c) for r,c in zip(rows,checks) if r['status']=='complete']
    if len(success)!=21 or [r['call_id'] for r,c in success]!=expected_calls[:21]:
        raise ValueError('all21 accepted R35 complete reference identities required')
    specmap={s['id']:s for s in specs};oracles=[dict(o) for o in old_oracles]
    for row,check in success:
        spec=specmap[row['id']]
        if (not row['attempted'] or not row['passed'] or not check['delivery_integrity_passed']
                or not check['arithmetic_passed'] or not check['arithmetic']['passed']
                or any(row[k]!=spec[k] for k in IDENTITY_FIELDS)
                or any(row['array_sha256'].get(k)!=spec['array_sha256'][k] for k in OLD_FIELDS)):
            raise ValueError('old R35 oracle must be complete and independently accepted with exact input identity')
        source=old_run/row['archive_path']
        if source.parent!=old_run/'forward/arrays' or sha(source)!=row['archive_sha256']:
            raise ValueError('old R35 oracle archive changed')
        kind='r35_log_real'
        oracles.append(dict({k:row[k] for k in IDENTITY_FIELDS},order=row['order'],reference_kind=kind,
            oracle_id=kind+'__order'+str(row['order']),reference_record=row,
            source_archive_path=str(source),source_archive_sha256=row['archive_sha256'],
            array_sha256=row['array_sha256'],source_array_sha256=row['array_sha256'],
            source_fields={n:n for n in ('log_raw_mass','log_raw_fov_mass','probability')}))
    if len(oracles)!=ORACLE_COUNT or len({(o['id'],o['oracle_id']) for o in oracles})!=ORACLE_COUNT:
        raise ValueError('duplicate or missing old reference source/order identity')
    return bindings,specs,oracles


