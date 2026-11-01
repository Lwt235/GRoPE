"""Finalize delivery using JSON metadata and byte hashes only; never decode arrays."""
import hashlib
import json
import time
from pathlib import Path

ROOT = Path('/home/liwt/GRoPE')
OUT = ROOT / 'results/round36_component_round_delivery_manifest_20261101.json'
HELPER = ROOT / 'results/round36_delivery_finalize_20261101'
BINDINGS = {}

def read(path):
    return json.loads((ROOT / path).read_text())

def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def add(path, expected=None):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve(strict=True)
    name = path.relative_to(ROOT).as_posix()
    actual = sha(path)
    if expected is not None and actual != expected:
        raise RuntimeError('hash mismatch: ' + name)
    if name in BINDINGS and BINDINGS[name] != actual:
        raise RuntimeError('conflicting binding: ' + name)
    BINDINGS[name] = actual
    return actual

def mapped(mapping, base=''):
    for name, expected in mapping.items():
        add(Path(base) / name, expected)

def complete_tree(directory):
    obj = read(directory + '/completion.json')
    mapped(obj['files'], directory)
    actual = {p.relative_to(ROOT / directory).as_posix() for p in (ROOT / directory).rglob('*') if p.is_file()}
    if actual != set(obj['files']) | {'completion.json'}:
        raise RuntimeError('inventory mismatch: ' + directory)
    add(directory + '/completion.json')
    return len(actual)

if OUT.exists():
    raise FileExistsError(OUT)
receipt_path = 'results/round36_component_round_root_acceptance_20261101.json'
receipt_sha = add(receipt_path, 'e12d0e38dbbc9bf4381239eb319d9da0b7ffbfc9bc931e6deeeb21dbfed80fa8')
receipt = read(receipt_path)
assert receipt['accepted_for_numerical_interpretation'] is True
assert receipt['numerical_preflight_passed'] is True
assert receipt['scientific_gate_passed'] is False
assert (receipt['fixed_base_count'], receipt['actual_reference_calls'], receipt['inherited_oracle_count']) == (23, 46, 135)
assert sum(receipt['comparison_counts'].values()) == 339
mapped(receipt['files'])
mapped(receipt['source_hashes'])
producer_count = complete_tree('results/round36_component_round_preflight')
audit_count = complete_tree('results/round36_component_round_independent_audit')
assert (producer_count, audit_count) == (103, 33)

report_data = 'docs/round36_component_round_reference_report_data'
draft_path = report_data + '/delivery_manifest_draft_v1.json'
draft = read(draft_path)
assert len(draft['ready_file_bindings']) == 79
mapped(draft['ready_file_bindings'])
add(draft_path, '42d9cf708c2c2e95fea8ac9533b7900036e12899398b5ce2e7c9640115f23186')
supp_path = report_data + '/supplemental_completion_v2.json'
supp = read(supp_path)
mapped(supp['files'])
add(supp_path, 'cebd849e24eea1f04bb219312c032b97f6f7b9416338adb9ea44d6fb0b5424b6')
add(supp['report_path'], supp['report_sha256'])
add(supp['previous_completion_path'], supp['previous_completion_sha256'])
report_complete = read(supp['previous_completion_path'])
mapped(report_complete['outputs'], report_data)
assert report_complete['counts'] == {'cases23': 23, 'comparisons339': 339, 'oracles135': 135, 'outer_trace_summaries': 24, 'reference_calls46': 46}
report_files = {p.relative_to(ROOT).as_posix() for p in (ROOT / report_data).rglob('*') if p.is_file()}
assert report_files == set(supp['files']) | {supp_path}

cost_dir = 'results/round36_real_initial_cost_lowerbound_20261101'
cost_path = cost_dir + '/execution_receipt.json'
cost_sha = add(cost_path, '1927b233a5dd740c4e79d08cd52afea1932271c7291ec74dddf21f9303c3b760')
cost = read(cost_path)
assert len(cost['files']) == cost['file_count'] == 99
assert cost['accepted_for_cost_interpretation'] is True
assert cost['all72_initial_geometry_counted'] is True
assert cost['budget_feasibility'] == 'unknown'
assert cost['numerical_preflight_passed'] is False
assert cost['source_integrations'] == 0
mapped(cost['files'])
actual_cost = {p.relative_to(ROOT).as_posix() for p in (ROOT / cost_dir).rglob('*') if p.is_file()}
listed_cost = {p for p in cost['files'] if p.startswith(cost_dir + '/')}
assert actual_cost == listed_cost | {cost_path}
mapped(read(cost_dir + '/completion.json')['files'], cost_dir)
cost_lock = read(cost_dir + '/lock.json')
mapped(cost_lock['local_source_hashes'])
mapped(cost_lock['bindings'])
assert len(cost_lock['bindings']) == cost['binding_count'] == 167

readme_sha = add('README.md')
readme_history = 'results/round36_readme_history_20261101'
readme_before = read(readme_history + '/before_edit.json')
add(readme_history + '/README.before.md', readme_before['sha256'])
for path in (ROOT / readme_history).rglob('*'):
    if path.is_file():
        add(path)

decision_path = 'results/round36_component_round_post_result_decision_20261101.json'
decision_sha = add(decision_path, 'cc446d06523ea72703d8710e7d81b699517beac9c44019ae0e45824231170490')
design_path = 'docs/round37_component_round_real_reference_design.md'
design_sha = add(design_path, 'bdeb0121cc9ac11572a12566594e675705be92b9e2629b7f85f3345a1da2e8d4')
add('ROUND37_COMPONENT_ROUND_REAL_REFERENCE_PROTOCOL.md')
history_dir = 'results/round36_postdecision_draft_history_20261101'
history = read(history_dir + '/note.json')
for original, expected in history['files_before'].items():
    add(Path(history_dir) / Path(original).name, expected)
for path in (ROOT / history_dir).rglob('*'):
    if path.is_file():
        add(path)

for path in HELPER.rglob('*'):
    if path.is_file():
        add(path)
manifest = {
    'version': 1,
    'created_unix': time.time(),
    'status': 'FINAL_DELIVERY',
    'finalized': True,
    'files': dict(sorted(BINDINGS.items())),
    'file_count': len(BINDINGS),
    'root_acceptance_path': receipt_path,
    'root_acceptance_sha256': receipt_sha,
    'accepted_numerical_preflight_passed': True,
    'synthetic_preflight': {'cases': 23, 'actual_reference_calls': 46, 'comparisons': 339, 'old_oracles': 135, 'real_inputs_read': False, 'producer_inventory_files': producer_count, 'audit_inventory_files': audit_count},
    'cost_diagnostic': {'receipt_path': cost_path, 'receipt_sha256': cost_sha, 'bound_receipt_files': 99, 'inherited_bindings_verified': 167, 'real_input_geometry_read_in_diagnostic': True, 'cases': 72, 'budget_feasibility': 'unknown', 'necessary_budget_counterexamples': 0, 'reference_integrations': 0, 'numerical_accuracy_established': False},
    'report_path': supp['report_path'],
    'report_sha256': supp['report_sha256'],
    'report_revision': 2,
    'original_report_completion_path': supp['previous_completion_path'],
    'supplemental_report_inventory_path': supp_path,
    'superseded_delivery_draft_path': draft_path,
    'verified_original_draft_binding_count': 79,
    'readme_sha256_at_delivery': readme_sha,
    'post_result_decision_path': decision_path,
    'post_result_decision_sha256': decision_sha,
    'prospective_design_path': design_path,
    'prospective_design_sha256': design_sha,
    'prospective_round': {'round': 37, 'cases': 72, 'scheduled_reference_calls': 144, 'comparisons': 296, 'historical_success_oracles': 40, 'executed_by_this_delivery': False},
    'pending_bindings': [],
    'manifest_operations': {'json_and_byte_hash_only': True, 'npz_decodes': 0, 'integrator_calls': 0, 'test_runs': 0, 'existing_files_modified': False},
    'labels_read': False,
    'efficacy_tested': False,
    'general_numerical_accuracy_established': False,
    'scientific_gate_passed': False,
    'goal_complete': False,
    'notes': [
        'README hash identifies this delivery version and may change in later rounds.',
        'Original report completion, immutable 79-binding draft, report v1 bytes, and v2 supplemental inventory are retained without replacement.',
        'The cost receipt includes its external report; all 99 file bindings and the full cost directory inventory were verified. This geometry-only lower bound is not a total-cost feasibility or numerical-accuracy pass.',
        'The earlier prospective decision/design wording is preserved. The corrected strict test is E < stored target; the stored target already includes division by 8. No test, freeze or experiment ran under the earlier wording, as recorded by its history.',
        'R35 failure is preserved. R36 is a fixed synthetic reference pass; R37 is a separately selected prospective validation, not an outcome of this delivery.',
        'Development snapshot coverage remains as actually recorded: audit first test captured four direct items, reference/runner captured their then-available sources and missing lists, and final joint/formal closure contained all 19 sources.'
    ]
}
with OUT.open('x') as stream:
    json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
    stream.write('\n')
print(json.dumps({'path': str(OUT.relative_to(ROOT)), 'sha256': sha(OUT), 'file_count': len(BINDINGS), 'verified_draft_bindings': 79, 'verified_cost_files': 99, 'producer_inventory': producer_count, 'audit_inventory': audit_count, 'npz_decodes': 0, 'integrator_calls': 0}, sort_keys=True))
