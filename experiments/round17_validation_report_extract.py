"""External post-acceptance reporting only; never imports an experiment/model.

This file is outside the frozen experiment. It reads completed, accepted JSON
metadata, never probabilities/labels, and preserves failures and all aliases.
It is not a new independent audit, selector, gate, or research-success claim.
"""
from __future__ import annotations

import argparse
import csv
from fractions import Fraction
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def csv_write(path, rows):
    if not rows:
        raise ValueError('refusing an empty report table')
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def gate_records(gates):
    """Preserve hierarchy: alternative cost routes are not mandatory failures."""
    decisions, checks, routes = [], [], []
    def visit(value, path):
        if not isinstance(value, dict):
            return
        if 'passed' in value:
            decisions.append(dict(path=path, passed=value['passed'],
                context={k:v for k,v in value.items() if not isinstance(v, (dict,list)) and k != 'passed'}))
        for key, child in value.items():
            location = path+'.'+key
            if key == 'checks' and isinstance(child, dict):
                checks.extend(dict(path=location+'.'+k, passed=v)
                              for k,v in child.items() if type(v) is bool)
            if key in ('accuracy_route', 'efficiency_route'):
                routes.append(dict(path=location, passed=child, alternative_route=True))
            visit(child, location)
    visit(gates, 'gates')
    return dict(decision_nodes=decisions, failed_decision_nodes=[r for r in decisions if r['passed'] is False],
        mandatory_boolean_checks=checks, failed_boolean_checks=[r for r in checks if r['passed'] is False],
        alternative_cost_routes=routes, original_complete_gate_tree=gates)


def center_fraction(metric, condition, scene=None):
    counts = metric['pck_by_condition'][condition]['counts_by_scene']
    if scene is not None:
        item = counts[scene]
        return Fraction(item['pck_count'], item['n_point'])
    return sum((Fraction(v['pck_count'],v['n_point']) for v in counts.values()),Fraction())/len(counts)


def tables(summary):
    roles = summary['roles']; aliases = roles['aliases']; operators = roles['operators']
    if len(aliases) != 19 or 'matched_constant_temperature' not in aliases or 'matched_isotropic' not in aliases:
        raise ValueError('report requires all nineteen declared aliases')
    detail, macro, contrasts, timing = [], [], [], []
    for scenario, bank in summary['metrics'].items():
        if set(bank) != set(aliases):
            raise ValueError('report would omit an alias')
        primary = bank['primary']
        for alias, metric in bank.items():
            identity = dict(scenario=scenario, alias=alias, operator=aliases[alias], **operators[aliases[alias]])
            macro.append(dict(**identity, **{f:metric[f] for f in ('kl','js','direct_kl','direct_js')},
                all_finite=metric['probability_checks']['all_finite'],
                max_normalization_error=metric['probability_checks']['max_normalization_error']))
            for scene, conditions in metric['detail'].items():
                for condition, values in conditions.items():
                    counts = metric['pck_by_condition'][condition]['counts_by_scene'][scene]
                    detail.append(dict(**identity, scene=scene, condition=condition, **values,
                        **counts, pck=float(center_fraction(metric,condition,scene)),
                        equal_scene_pck=float(center_fraction(metric,condition))))
            if alias == 'primary':
                continue
            for condition in metric['by_condition']:
                for scene in ('equal_scene_macro', *metric['by_scene']):
                    scene_key = None if scene == 'equal_scene_macro' else scene
                    p = primary['by_condition'][condition] if scene_key is None else primary['by_scene_condition'][scene][condition]
                    c = metric['by_condition'][condition] if scene_key is None else metric['by_scene_condition'][scene][condition]
                    delta = center_fraction(primary,condition,scene_key)-center_fraction(metric,condition,scene_key)
                    contrasts.append(dict(scenario=scenario, reference_alias=alias, reference_operator=aliases[alias],
                        scene=scene, condition=condition, primary_kl=p, reference_kl=c,
                        primary_relative_kl_gain=None if c == 0 else 1-p/c,
                        primary_pck_difference_pp=float(delta*100), exact_pck_delta_numerator=delta.numerator,
                        exact_pck_delta_denominator=delta.denominator))
    for alias, operator in aliases.items():
        value = summary['timing'][operator]
        timing.append(dict(alias=alias, operator=operator, **operators[operator],
            mean_seconds=value['mean_seconds'], paired_cells=len(value['cells']),
            primary_time_ratio=summary['timing'][aliases['primary']]['mean_seconds']/value['mean_seconds']))
    return dict(all_alias_scene_conditions=detail, all_alias_scenario_macros=macro,
                primary_against_every_control=contrasts, all_alias_cost=timing)


def accepted_metadata(run):
    """The existence guard precedes opening any new summary or audit score."""
    for name in ('lock.json','lock.sha256','forward_complete.json','summary.json','completion.json','decision.json','decision.sha256'):
        if not (run/name).is_file():
            raise ValueError('completed independent acceptance is required: '+name)
    decision = read(run/'decision.json')
    if digest(run/'decision.json') != (run/'decision.sha256').read_text().strip():
        raise ValueError('decision checksum changed')
    audit_path = Path(decision['audit_path'])
    if not audit_path.is_absolute():
        raise ValueError('accepted independent audit must use an absolute path')
    audit_root = audit_path.parent
    if not (audit_root/'completion.json').is_file():
        raise ValueError('independent audit completion missing')
    complete, audited = read(run/'completion.json'), read(audit_root/'completion.json')
    inputs = {run/name for name in ('lock.json','lock.sha256','forward_complete.json','summary.json','completion.json','decision.json','decision.sha256')}
    inputs.update((audit_path,audit_root/'completion.json',audit_root/'lock.json',audit_root/'lock.sha256'))
    for name in ('lock','forward_complete','summary','completion'):
        if digest(run/(name+'.json')) != decision['validation_'+name+'_sha256']:
            raise ValueError('accepted producer metadata binding changed: '+name)
    if (decision['independent_audit_passed'] is not True or digest(audit_path) != decision['audit_sha256']
            or audited['files'].get(audit_path.name) != digest(audit_path)
            or audited['passed'] is not True or complete['summary_sha256'] != digest(run/'summary.json')
            or complete['files'].get('summary.json') != digest(run/'summary.json')
            or digest(run/'lock.json') != (run/'lock.sha256').read_text().strip()
            or digest(audit_root/'lock.json') != (audit_root/'lock.sha256').read_text().strip()):
        raise ValueError('accepted completion/audit chain changed')
    audit_design = read(audit_root/'lock.json')
    if audit_design['producer_lock_sha256'] != digest(run/'lock.json'):
        raise ValueError('independent audit design belongs to a different producer run')
    summary, verification = read(run/'summary.json'), read(audit_path)
    for name in ('lock','forward_complete','summary','completion'):
        if verification['validation_'+name+'_sha256'] != digest(run/(name+'.json')):
            raise ValueError('independent verification belongs to a different producer run: '+name)
    if (verification['passed'] is not True or verification['independent_audit_lock_sha256'] != digest(audit_root/'lock.json')
            or verification['roles'] != summary['roles'] or decision['roles'] != summary['roles']
            or verification['gates'] != summary['gates'] or decision['gates'] != summary['gates']
            or verification['scientific_gate_passed'] != decision['scientific_gate_passed']
            or decision['scientific_gate_passed'] != summary['gates']['passed']):
        raise ValueError('accepted scientific decision mismatch')
    numeric = {}
    for name, expected in audited['files'].items():
        if name.endswith('_numeric.json'):
            path = audit_root/name
            if digest(path) != expected:
                raise ValueError('bound independent numeric metadata changed')
            numeric[name] = read(path); inputs.add(path)
    if len(numeric) != 2:
        raise ValueError('both fixed-pair numerical reports are required')
    return summary, decision, verification, numeric, inputs


def extract(run, out):
    run, out = run.resolve(), out.resolve()
    if out.exists() or out == run or run in out.parents:
        raise ValueError('external report output must be new and outside the frozen run')
    summary, decision, verification, numeric, inputs = accepted_metadata(run)
    audit_root = Path(decision['audit_path']).parent
    if out == audit_root or audit_root in out.parents:
        raise ValueError('report cannot alter the completed independent audit')
    records = tables(summary)
    out.mkdir(parents=True)
    for name, values in records.items():
        csv_write(out/(name+'.csv'), values)
    write(out/'all_gate_conditions.json', gate_records(summary['gates']))
    write(out/'accepted_decision.json', decision)
    write(out/'independent_verification.json', verification)
    write(out/'all_independent_numerical_checks.json', numeric)
    write(out/'all_paired_timing_cells.json', summary['timing'])
    write(out/'manifest.json', dict(report_only=True, additional_audit=False, probabilities_or_labels_decoded=False,
        actual_value_run=False, optimized_gs_run=False, general_research_goal_achieved=False,
        scientific_gate_passed=decision['scientific_gate_passed'], independent_audit_passed=True,
        source_sha256=digest(Path(__file__)), input_json_sha256={str(p):digest(p) for p in sorted(inputs)},
        output_sha256={p.name:digest(p) for p in sorted(out.iterdir())},
        table_rows={name:len(rows) for name,rows in records.items()},
        scope='Hashes completed metadata only; trusts the accepted audit commitments for large array evidence. All nineteen aliases and every recorded condition are retained. Alternative accuracy/cost routes are reported separately from mandatory checks.'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path('results/round17_validation'))
    parser.add_argument('--out', type=Path, default=Path('results/round17_validation_report_tables'))
    args = parser.parse_args()
    extract(args.run, args.out)


if __name__ == '__main__':
    main()
