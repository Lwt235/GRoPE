"""Post-acceptance delivery tables only; stdlib, no models/arrays/new metrics."""
import csv
import hashlib
import json
from pathlib import Path
import time

root = Path('/home/liwt/GRoPE')
run = root/'results/round21_matched_decomposition'
audit = root/'results/round21_matched_decomposition_independent_audit'
receipt = root/'results/round21_matched_root_acceptance_20261031.json'
out = root/'results/round21_matched_report_tables'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''): h.update(block)
    return h.hexdigest()
def read(p): return json.loads(p.read_text())
r=read(receipt)
assert all(r.get(k) is True for k in ('accepted_for_development_interpretation',
    'full_forward_and_final_independent_audits_passed','strict_forward_before_label_chronology_checked'))
for name,digest in r['files'].items(): assert sha(root/name)==digest,name
assert not out.exists()
summary=read(run/'summary.json')
assert summary['diagnostic_only'] and summary['no_candidate_selection'] and not summary['scientific_gate_passed']
out.mkdir()
(out/'extractor.py').write_bytes(Path(__file__).read_bytes())
counts={}
def flatten(obj,prefix=''):
    result={}
    for k,v in obj.items():
        key=prefix+k
        if isinstance(v,dict):result.update(flatten(v,key+'__'))
        elif isinstance(v,list):result[key]=json.dumps(v,ensure_ascii=False,separators=(',',':'))
        else:result[key]=v
    return result
def table(name,rows):
    rows=[flatten(v) for v in rows]
    columns=list(dict.fromkeys(k for row in rows for k in row))
    with (out/(name+'.csv')).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(rows)
    counts[name]=len(rows)
def levels(items):
    rows=[]
    for name,item in items.items():
        for s,cs in item['by_scene_condition'].items():
            for c,v in cs.items():
                rows.append(dict(name=name,level='scene_condition',scene=s,condition=c,
                    regional=v,center=item['center']['by_scene_condition'][s][c]))
        for level in ('by_condition','by_scene'):
            for key,v in item[level].items():
                rows.append(dict(name=name,level=level,scene=key if level=='by_scene' else None,
                    condition=key if level=='by_condition' else None,
                    regional=v,center=item['center'][level][key]))
        rows.append(dict(name=name,level='macro',scene=None,condition=None,
            regional=item['macro'],center=item['center']['macro']))
    return rows
table('roles',levels(summary['by_role']))
table('contrasts',levels(summary['by_contrast']))
table('interactions',levels(summary['interactions']))
table('regional_center_rows',read(run/'evaluation/rows.json'))
table('paired_rows',read(run/'evaluation/paired_rows.json'))
table('numerical_checks',read(audit/'forward_audit/numerical_checks.json'))
table('support_checks',read(audit/'forward_audit/anchor_checks.json'))
table('historical_replays',read(audit/'forward_audit/historical_replays.json'))
table('historical_metrics',read(audit/'historical_metric_checks.json'))
with (out/'report.json').open('x') as f:
    json.dump(dict(summary=summary,forward_verification=read(audit/'forward_verification.json'),
        final_verification=read(audit/'verification.json'),receipt_sha256=sha(receipt),
        note='Existing accepted evidence only. Repeated role/condition rows are not independent samples. No candidate selection or scientific pass.'),f,indent=2)
assert counts['roles']==128 and counts['contrasts']==160 and counts['interactions']==48
assert counts['regional_center_rows']==1152 and counts['paired_rows']==1440
assert counts['numerical_checks']==72 and counts['support_checks']==48
assert counts['historical_replays']==576 and counts['historical_metrics']==576
completion=dict(passed=True,post_acceptance_delivery_only=True,scientific_gate_passed=False,
    no_candidate_selection=True,created_unix=time.time(),receipt_sha256=sha(receipt),
    controller_sha256=sha(Path(__file__)),table_row_counts=counts,
    files={p.name:sha(p) for p in sorted(out.iterdir())})
with (out/'completion.json').open('x') as f:json.dump(completion,f,indent=2)
print(json.dumps(dict(passed=True,tables=counts,output=str(out)),sort_keys=True))
