"""CPU static evidence index only: ast.parse never imports or executes author code."""
import ast
import datetime
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]
rows = [json.loads(line) for line in (ROOT / 'retrieval_log.jsonl').read_text().splitlines()]
assert len(rows) == 48 and [r['id'] for r in rows] == list(range(1, 49))
facts = json.loads((ROOT / 'mechanism_facts.json').read_text())
index = []
for r in rows:
    item = dict(r)
    if r.get('snapshot'):
        p = ROOT / r['snapshot']
        b = p.read_bytes()
        item['snapshot_sha256'] = hashlib.sha256(b).hexdigest()
        item['snapshot_bytes'] = len(b)
        item['line_count'] = len(b.splitlines())
        if r.get('sha256'):
            assert r['sha256'] == item['snapshot_sha256']
        if r.get('kind') == 'author_source':
            pieces = r['url'].split('/')
            item['repository'] = '/'.join(pieces[3:5])
            item['commit'] = pieces[5]
            item['path'] = '/'.join(pieces[6:])
            item['pinned_browser_url'] = 'https://github.com/' + item['repository'] + '/blob/' + item['commit'] + '/' + item['path']
            item['git_blob_sha1'] = hashlib.sha1(b'blob ' + str(len(b)).encode() + b'\0' + b).hexdigest()
            if p.suffix == '.py':
                tree = ast.parse(b.decode('utf-8'))
                item['static_symbols'] = [dict(name=n.name, line=n.lineno, end_line=n.end_lineno, kind=type(n).__name__) for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    index.append(item)

local_names = ['bound_read1', 'soft_read1', 'learned_projection_pe1', 'native_input_pe1', 'native_render_parameter_pe1', 'native_memory_context_pe1', 'joint_write1', 'surface1', 'geometry_transport1', 'gaussian_scene_memory1', 'source_geometry_control1', 'native_gaussian_pe1', 'coarse_memory_pe1', 'native_render_factorization1', 'native_covariance_role1']
local_paths = [PROJECT / 'results/gaussian_memory' / name / 'report.md' for name in local_names]
local_paths += [PROJECT / 'results/research_related_work/gaussian_parameter_encoding_20261106/notes.md', PROJECT / 'docs/gaussian_memory_related_work_20261111.md', PROJECT / 'results/gaussian_memory/autonomous_research_loop/round_005/analysis.md']
local_index = [dict(path=str(p.relative_to(PROJECT)), sha256=hashlib.sha256(p.read_bytes()).hexdigest(), bytes=p.stat().st_size, use='report-level historical evidence, no prediction/weight read') for p in local_paths]
sources = dict(schema_version=1, experiment='gaussian_pe_route_review1', retrieval_end_utc=rows[-1]['ended_utc'], external_request_count=len(rows), author_source_readme_config_count=sum(r.get('kind') == 'author_source' and r.get('ok', False) for r in rows), bibliography_and_identity=[dict(id=w['id'], **w['bibliography']) for w in facts['works']], retrievals=index, local_history=local_index, limits=['No final OpenReview PDF obtained', 'JSRE formulas are pinned arXiv, publisher metadata only verified', 'JSRE candidate repository is self-identified official, reciprocal author ownership link not verified', 'API commit metadata is not model/backend attestation', 'No downloaded code imported or executed'])
(ROOT / 'sources.json').write_text(json.dumps(sources, ensure_ascii=False, indent=2) + '\n')
stat = os.statvfs(ROOT)
(ROOT / 'resource_completion.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), checks_at_each_http_request=True, paper_completion_checks={w['id']:dict(free_bytes=stat.f_bavail * stat.f_frsize, source_files_complete_or_unknown=True) for w in facts['works']}, project_phase_bytes=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file()), data_disk_new_bytes=0, free_bytes=stat.f_bavail * stat.f_frsize), indent=2) + '\n')
print(json.dumps(dict(external_requests=len(rows), author_files=sources['author_source_readme_config_count'], source_symbols_indexed=sum(len(r.get('static_symbols', [])) for r in index), output='sources.json')))
