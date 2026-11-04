"""Read only the two small, pinned official TEST text manifests."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import urllib.request

OUT = Path(__file__).resolve().parent
PLAN = json.loads((OUT / 'plan.json').read_text())


def fetch(relative):
    url = ('https://hf-mirror.com/datasets/' + PLAN['repository'] + '/resolve/'
           + PLAN['revision'] + '/' + relative)
    row = dict(path=relative, requested_url=url, started_unix=time.time())
    target = OUT / 'metadata_sources' / relative
    assert not target.exists(), target
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        request = urllib.request.Request(url, headers={'User-Agent': 'GRoPE-metadata-only'})
        with opener.open(request, timeout=25) as response:
            body = response.read(8 * 2**20 + 1)
            if len(body) > 8 * 2**20:
                raise ValueError('text metadata exceeded declared size limit')
            if relative.endswith('.jsonl'):
                parsed = [json.loads(line) for line in body.decode('utf-8').splitlines() if line]
            else:
                parsed = json.loads(body)
            if not isinstance(parsed, list):
                raise ValueError('expected manifest list')
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as stream:
                stream.write(body)
            row.update(ok=True, status=response.status, bytes=len(body), rows=len(parsed),
                       sha256=hashlib.sha256(body).hexdigest(), response_url=response.geturl())
    except Exception as exc:
        row.update(ok=False, error=type(exc).__name__ + ': ' + str(exc))
    row['finished_unix'] = time.time()
    return row


if __name__ == '__main__':
    assert not (OUT / 'transport.json').exists()
    with ThreadPoolExecutor(max_workers=2) as pool:
        records = list(pool.map(fetch, PLAN['files']))
    result = dict(records=records, all_fetched=all(r['ok'] for r in records),
                  plan_sha256=hashlib.sha256((OUT / 'plan.json').read_bytes()).hexdigest(),
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  proxy_used=False, ply_requested=False)
    with (OUT / 'transport.json').open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result, indent=2), flush=True)
