"""Small-text identities only; stdlib, no author imports or model inspection."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]
COMMIT = 'fd7e718597cf9963de85c8fffae32a698e8619f5'
tree = json.loads((ROOT / 'repository_tree.json').read_text())
lookup = {e['path']: e for e in tree['tree'] if e['type'] == 'blob'}
old = {e['path']: e for e in json.loads((ROOT / 'source_identities.json').read_text())}
identities = []
for path in sorted((ROOT / 'sources').rglob('*')):
    if not path.is_file():
        continue
    relative = str(path.relative_to(ROOT / 'sources'))
    data = path.read_bytes()
    trimmed = 0
    while hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest() != lookup[relative]['sha']:
        assert data.endswith(b'\n') and trimmed < 2, relative
        data = data[:-1]
        trimmed += 1
    if trimmed:
        path.write_bytes(data)
    identities.append(dict(path=relative, bytes=len(data), git_blob_sha1=lookup[relative]['sha'],
                           SHA256=hashlib.sha256(data).hexdigest(),
                           serialization_extra_newline_trimmed=old.get(relative, {}).get('serialization_extra_newline_trimmed', 0) + trimmed,
                           line_count=len(data.splitlines()),
                           url=f'https://github.com/Rainzor/STAC/blob/{COMMIT}/{relative}'))
    if path.suffix == '.py':
        ast.parse(data.decode(), filename=relative)
assert len(identities) == 16
(ROOT / 'source_identities.json').write_text(json.dumps(identities, indent=2) + '\n')
ledger = json.loads((ROOT / 'retrieval_ledger.json').read_text())
for relative in ['causalvggt/heads/camera_head.py', 'attn-cuda/attn_cuda/__init__.py', 'attn-cuda/csrc/kernels/fwd_kernel.h']:
    ledger['requests'].append(dict(request=len(ledger['requests'])+1, path=relative, status='success'))
ledger.update(explicit_request_count=len(ledger['requests']), unique_author_bodies=len(identities),
              retries=0, failed_explicit_requests=0,
              author_body_bytes=sum(e['bytes'] for e in identities),
              stop_reason='16-body ceiling reached; no further author retrieval',
              source_transport='Connected GitHub fetch tree/file tools; fixed commit only')
assert ledger['explicit_request_count'] == 17
(ROOT / 'retrieval_ledger.json').write_text(json.dumps(ledger, indent=2) + '\n')
local_paths = ['models/streamvggt.py', 'models/aggregator.py', 'layers/attention.py', 'heads/camera_head.py']
local = []
for relative in local_paths:
    path = PROJECT / 'third_party/streamvggt/src/streamvggt' / relative
    data = path.read_bytes()
    ast.parse(data.decode(), filename=str(path))
    local.append(dict(path=str(path), bytes=len(data), SHA256=hashlib.sha256(data).hexdigest()))
(ROOT / 'local_text_identities.json').write_text(json.dumps(local, indent=2) + '\n')
print(json.dumps(dict(author_bodies=len(identities), requests=len(ledger['requests']), author_bytes=ledger['author_body_bytes'],
                      exact_git_blobs=True, local_texts=len(local), author_imports=0, GPU_calls=0)))
