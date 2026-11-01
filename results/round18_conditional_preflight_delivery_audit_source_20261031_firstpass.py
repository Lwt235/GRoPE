"""Post-execution delivery/array audit. Never imports the production model."""
import ast
from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys
import time

import numpy as np
import scipy

ROOT = Path('/home/liwt/GRoPE')
RUN = ROOT/'results/round18_conditional_preflight'
OUT = ROOT/'results/round18_conditional_preflight_delivery_audit_20261031.json'
SOURCE_OUT = ROOT/'results/round18_conditional_preflight_delivery_audit_source_20261031.py'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def diff(a,b):
    return float(np.max(np.abs(np.asarray(a)-np.asarray(b))))


def close(a,b):
    return bool(np.isclose(a,b,rtol=2e-12,atol=1e-300))


started = time.time()
completion = read(RUN/'completion.json')
completion_sha = sha(RUN/'completion.json')
assert started > completion['finished_unix']
lock = read(RUN/'lock.json')
assert sha(RUN/'lock.json') == (RUN/'lock.sha256').read_text().strip()
assert sha(RUN/'lock.json') == 'bad11062d21e085ebc9d6c4b4dfdf3e1427f88bc03c33602dbcfd3700a300c12'
actual_inventory = {str(p.relative_to(RUN)) for p in RUN.rglob('*') if p.is_file()}
assert actual_inventory == set(completion['files']) | {'completion.json'}
for name, checksum in completion['files'].items():
    assert sha(RUN/name) == checksum, name
for name, checksum in lock['source_hashes'].items():
    assert sha(ROOT/name) == checksum, name
    assert sha(RUN/'source_snapshot'/name) == checksum, name
assert len(lock['source_hashes']) == 6
assert sha(RUN/'fixtures.json') == lock['fixture_sha256']
assert sha(ROOT/'results/round17_validation/decision.json') == lock['source_decision_sha256']

# Execute only pure fixture and environment declarations, not module imports,
# the numerical runner, or identities() (which would call production).
runner_source = (RUN/'source_snapshot/experiments/round18_conditional_preflight.py').read_text()
tree = ast.parse(runner_source)
names = {'grid','rotated_covariance','fixtures','environment'}
nodes = [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
assert {n.name for n in nodes} == names
namespace = dict(np=np,scipy=scipy,sys=sys,platform=platform,io=io,
                 redirect_stdout=redirect_stdout,hashlib=hashlib,os=os)
exec(compile(ast.Module(body=nodes,type_ignores=[]),'frozen_descriptor_functions','exec'),namespace)
fixtures = read(RUN/'fixtures.json')
assert fixtures == namespace['fixtures']()
assert namespace['environment']() == lock['environment']
assert len(fixtures) == 25 and len({f['name'] for f in fixtures}) == 25

spec = importlib.util.spec_from_file_location('_r18_frozen_reference_delivery',
    RUN/'source_snapshot/experiments/round18_conditional_reference.py')
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)
summary = read(RUN/'summary.json')
assert [r['name'] for r in summary['rows']] == [f['name'] for f in fixtures]
assert summary['lock_sha256'] == sha(RUN/'lock.json')
identities = read(RUN/'identities.json')
assert identities == summary['identities']
identity_checks = {k:bool(np.isfinite(v) and v <= lock['identity_tolerance'])
                   for k,v in identities['errors'].items()}
identity_passed = all(identity_checks.values()) and all(v is True for v in identities['rejected_inputs'].values())
assert identity_passed == identities['passed']

cases = []
probability_rows_checked = 0
all_saved_metrics_match = True
max_saved_error_difference = 0.
for fixture,row in zip(fixtures,summary['rows']):
    assert read(RUN/(row['name']+'.json')) == row
    path = RUN/row['artifact']
    assert sha(path) == row['sha256']
    with np.load(path,allow_pickle=False) as archive:
        d = {k:archive[k].copy() for k in archive.files}
    assert all(np.isfinite(v).all() for v in d.values())
    p,d96,ref384,ref512 = [d[key] for key in (
        'production64__probability','diagnostic96__probability','reference384__P','reference512__P')]
    probabilities = [p,d96,ref384,ref512]
    norm_errors = [float(abs(v.sum()-1)) for v in probabilities]
    probability_rows_checked += len(probabilities)
    for prefix,order in [('production64',64),('diagnostic96',96),('reference384',384),('reference512',512)]:
        assert int(d[prefix+'__order']) == order
        nodekey = '__quadrature_nodes' if prefix.startswith(('production','diagnostic')) else '__raw_quadrature_nodes'
        assert int(d[prefix+nodekey]) == order**2
        normalizerkey = '__z_geometry' if prefix.startswith(('production','diagnostic')) else '__Zg'
        probabilitykey = '__probability' if prefix.startswith(('production','diagnostic')) else '__P'
        assert close(float(d[prefix+normalizerkey]),float(d[prefix+'__geometry_mass'].sum()))
        assert diff(d[prefix+'__numerator']/d[prefix+normalizerkey],d[prefix+probabilitykey]) <= 2e-15
    areas = np.prod(np.array(fixture['bounds'])[:,2:]-np.array(fixture['bounds'])[:,:2],axis=-1)
    floor_total = float(np.sum(areas*1e-30))
    tail = 2.*math.erfc(14./math.sqrt(2.))
    certified_tail = 2*tail/(floor_total*(1-tail))
    assert close(float(d['reference512__certified_geometry_lower_bound']),floor_total*(1-tail))
    assert close(float(d['reference512__certified_amplified_tail_bound']),certified_tail)
    errors = dict(production_probability=diff(p,ref512),diagnostic96_probability=diff(d96,ref512),
        reference_probability=diff(ref384,ref512),
        reference_z_relative=float(abs(d['reference384__Zg']/d['reference512__Zg']-1)),
        production_z_relative=float(abs(d['production64__z_geometry']/d['reference512__Zg']-1)),
        reference_amplified_tail=float(d['reference512__amplified_tail_bound']),
        certified_amplified_tail=float(certified_tail))
    checks = dict(probabilities_valid=all((v>=0).all() and e<=2e-12 for v,e in zip(probabilities,norm_errors)),
        production_probability=errors['production_probability']<=1e-4,
        reference_probability=errors['reference_probability']<=1e-8,
        reference_z=errors['reference_z_relative']<=1e-8,
        reference_tail=errors['certified_amplified_tail']<=1e-8)
    if fixture['temperature'] == 0:
        errors['production_t0_marginal'] = diff(p,d['marginal__P'])
        errors['reference_t0_marginal'] = diff(ref512,d['marginal__P'])
        checks['production_t0_marginal'] = errors['production_t0_marginal']<=1e-4
        checks['reference_t0_marginal'] = errors['reference_t0_marginal']<=1e-8
    assert checks == row['checks']
    assert all(checks.values()) == row['passed']
    assert set(errors) == set(row['errors'])
    for name,value in errors.items():
        assert close(value,row['errors'][name]), (row['name'],name,value,row['errors'][name])
        max_saved_error_difference = max(max_saved_error_difference,abs(value-row['errors'][name]))
    exponent = np.exp(fixture['temperature']*(np.array(fixture['content'])-max(fixture['content'])))
    old = d['production64__geometry_mass']*exponent
    old /= old.sum()
    assert diff(old,d['production64__old_mix_then_normalize']) <= 2e-15
    assert close(diff(old,p),row['old_vs_conditioned_max_difference'])
    assert close(diff(d['production64__bare_expectation'],p),row['bare_vs_conditioned_max_difference'])

    args = tuple(fixture[k] for k in ('mean','covariance','bounds','content','h','temperature'))
    replay_start = time.time()
    replay = reference.conditional_gl_reference(*args,order=512)
    replay_checks = {}
    for key,value in replay.items():
        archive_key = 'reference512__'+key
        if archive_key in d:
            expected = d[archive_key]
            replay_checks[key] = dict(byte_equal=bool(np.array_equal(np.asarray(value),expected)),
                                     max_absolute_difference=diff(value,expected) if expected.dtype!=bool else float(not np.array_equal(value,expected)))
            assert np.array_equal(np.asarray(value),expected), (row['name'],key)
    marginal_replay = None
    if fixture['temperature'] == 0:
        marginal = reference.gaussian_marginal_reference(fixture['mean'],fixture['covariance'],fixture['bounds'],fixture['h'])
        marginal_replay = {}
        for key,value in marginal.items():
            archive_key = 'marginal__'+key
            if archive_key in d:
                marginal_replay[key] = bool(np.array_equal(np.asarray(value),d[archive_key]))
                assert marginal_replay[key], (row['name'],key)
    correction = None
    if fixture.get('intended_content_location') is not None:
        centers=.5*(np.array(fixture['bounds'])[:,:2]+np.array(fixture['bounds'])[:,2:])
        target=np.array(fixture['intended_content_location'])
        old_reference = d['reference512__geometry_mass']*exponent
        old_reference /= old_reference.sum()
        correction = dict(target=target.tolist(),
            conditioned_mean_distance=float(np.linalg.norm(ref512@centers-target)),
            old_mean_distance=float(np.linalg.norm(old_reference@centers-target)),
            both_use_same_reference512=True,descriptive_only=True)
        assert correction == row['synthetic_content_correction']
    cases.append(dict(name=row['name'],passed=all(checks.values()),checks=checks,errors=errors,
        normalization_errors=norm_errors,geometry_normalizer=float(d['reference512__Zg']),
        diagnostic96_exceeds_production_probability_tolerance=errors['diagnostic96_probability']>1e-4,
        reference512_replay=replay_checks,t0_marginal_replay=marginal_replay,
        reference_replay_seconds=time.time()-replay_start,correction=correction))
    print('AUDIT_REFERENCE_REPLAY',row['name'],'PASS',flush=True)

numerical_pass = all(c['passed'] for c in cases) and identity_passed
assert numerical_pass == summary['synthetic_numerical_passed'] == completion['synthetic_numerical_passed']
expected_action = 'separate_old_train_protocol_required' if numerical_pass else 'stop_order64_preserve_all_numerical_failures'
assert summary['next_action'] == expected_action
assert summary['diagnostic_order_promoted'] is False
for flag in ['scientific_gate_passed','real_inputs_read','fresh_inputs_read','actual_value_run','optimized_gs_run','general_goal_achieved']:
    assert summary[flag] is False
assert sha(RUN/'completion.json') == completion_sha
for name,checksum in completion['files'].items():assert sha(RUN/name)==checksum
for name,checksum in lock['source_hashes'].items():
    assert sha(ROOT/name)==checksum and sha(RUN/'source_snapshot'/name)==checksum
shutil.copy2(__file__,SOURCE_OUT)
result = dict(delivery_audit_passed=True,
    audit_kind='post-execution metadata/array audit and shared-reference replay; not a newly preregistered scientific audit',
    audit_started_unix=started,original_completion_unix=completion['finished_unix'],audit_finished_unix=time.time(),
    registration_before_original_execution=False,production_rerun=False,production_imported=False,
    reference_code_shared_with_original=True,independent_new_reference_implementation=False,
    scope='checks saved synthetic artifacts; frozen GL512 code replay is reproducibility evidence, not independent quadrature correctness',
    input_lock_sha256=sha(RUN/'lock.json'),input_completion_sha256=completion_sha,
    inventory_files_verified=len(completion['files']),current_and_snapshot_sources_verified=6,
    fixed_descriptor_regeneration='AST-only frozen grid/rotated_covariance/fixtures functions; no numerical main or model executed',
    fixed_descriptors_match=True,environment_matches=True,fixtures_checked=len(cases),
    probability_rows_checked=probability_rows_checked,all_saved_checks_and_errors_recomputed=True,
    maximum_saved_error_recalculation_difference=max_saved_error_difference,
    identity_errors=identities['errors'],identity_checks=identity_checks,
    rejected_inputs_record=identities['rejected_inputs'],identity_record_thresholds_passed=identity_passed,
    identity_computation_replayed=False,reference512_cases_replayed=len(cases),
    t0_marginal_cases_replayed=sum(f['temperature']==0 for f in fixtures),
    all_reference_replays_byte_equal=True,
    synthetic_numerical_passed=numerical_pass,scientific_gate_tested=False,scientific_gate_passed=False,
    diagnostic96_promoted=False,next_action=expected_action,
    production_failed_cases=[c['name'] for c in cases if not c['checks']['production_probability']],
    reference_convergence_failed_cases=[c['name'] for c in cases if not c['checks']['reference_probability']],
    production_failures_with_reference_converged=[c['name'] for c in cases if not c['checks']['production_probability'] and c['checks']['reference_probability']],
    both_production_disagreement_and_reference_convergence_failures=[c['name'] for c in cases if not c['checks']['production_probability'] and not c['checks']['reference_probability']],
    cases=cases,audit_source=str(SOURCE_OUT),audit_source_sha256=sha(SOURCE_OUT))
OUT.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print('DELIVERY_AUDIT_COMPLETE',len(cases),'numerical_pass',numerical_pass,'audit_pass',True,flush=True)
