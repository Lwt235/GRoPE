"""Pin official StreamVGGT sources and public weights, without proxy traffic."""
import json
import argparse
import hashlib
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from experiments.gaussian_memory_setup import ROOT, DATA, sha256

COMMIT = '05682bcb05683c4c71522e9cb618fc8ca349063c'
DEST = ROOT / 'third_party/streamvggt'
STORE = DATA / 'streamvggt'
OUT = ROOT / 'results/gaussian_memory/native_history1'


def fetch_sources():
    tree = json.loads((OUT / 'official_tree.json').read_text())
    files = [x for x in tree['tree'] if x['type'] == 'blob' and
             ((x['path'].startswith('src/streamvggt/') and x['path'].endswith('.py')) or
              x['path'] in ('LICENSE', 'README.md', 'requirements.txt'))]
    def one(record):
        dest = DEST / record['path']; dest.parent.mkdir(parents=True, exist_ok=True)
        def valid(data): return hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest() == record['sha']
        if dest.exists() and valid(dest.read_bytes()): return
        session = requests.Session(); session.trust_env = False
        for attempt in range(4):
            try:
                response = session.get(f"https://raw.githubusercontent.com/wzzheng/StreamVGGT/{COMMIT}/{record['path']}", timeout=(20, 40))
                response.raise_for_status(); data = response.content
                assert valid(data), record['path']
                dest.write_bytes(data); return
            except (requests.RequestException, AssertionError):
                if attempt == 3: raise
                time.sleep(2)
    with ThreadPoolExecutor(max_workers=8) as pool: list(pool.map(one, files))
    (DEST / 'PINNED_COMMIT').write_text(COMMIT + '\n')
    print(json.dumps({'official_runtime_files_ready': len(files)}), flush=True)


def range_download(url, path, size, expected):
    if path.exists() and path.stat().st_size == size and sha256(path) == expected: return
    partial = path.with_suffix('.partial'); manifest = path.with_suffix('.ranges.json')
    markers = path.parent / (path.name + '.range_completion'); markers.mkdir(exist_ok=True)
    if manifest.exists():
        state = json.loads(manifest.read_text()); assert state['size'] == size and state['sha256'] == expected
    else:
        prefix = partial.stat().st_size if partial.exists() else 0
        assert prefix < size
        block = 64 << 20
        state = {'size': size, 'sha256': expected, 'contiguous_prefix_bytes': prefix,
                 'ranges': [[a, min(a + block, size) - 1] for a in range(prefix, size, block)], 'connections': 8}
        manifest.write_text(json.dumps(state, indent=2) + '\n')
    fd = os.open(partial, os.O_WRONLY | os.O_CREAT, 0o664); os.ftruncate(fd, size)
    def one(bounds):
        start, end = bounds; marker = markers / f'{start}-{end}.json'
        if marker.exists(): return
        session = requests.Session(); session.trust_env = False
        for attempt in range(5):
            try:
                with session.get(url, headers={'Range': f'bytes={start}-{end}'}, stream=True, timeout=(20, 60)) as response:
                    assert response.status_code == 206 and response.headers['Content-Range'] == f'bytes {start}-{end}/{size}'
                    offset = start
                    for chunk in response.iter_content(1 << 20):
                        assert offset + len(chunk) <= end + 1
                        written = os.pwrite(fd, chunk, offset); assert written == len(chunk); offset += written
                    assert offset == end + 1
                marker.write_text(json.dumps({'start': start, 'end': end, 'complete': True}) + '\n')
                print(json.dumps({'weight_range_complete': [start, end]}), flush=True); return
            except (requests.RequestException, AssertionError, KeyError):
                if attempt == 4: raise
                time.sleep(2 + attempt)
    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            for future in as_completed([pool.submit(one, b) for b in state['ranges']]): future.result()
    finally: os.close(fd)
    assert partial.stat().st_size == size and sha256(partial) == expected, 'final model checksum mismatch'
    partial.replace(path)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--provider', choices=['hf', 'cloud'], default='hf')
    args = parser.parse_args()
    STORE.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
    session = requests.Session(); session.trust_env = False
    pool = ThreadPoolExecutor(max_workers=1)
    source_future = pool.submit(fetch_sources)
    identity = STORE / 'source_identity.json'
    if identity.exists():
        meta = json.loads(identity.read_text())
    else:
        response = session.get('https://hf-mirror.com/api/models/lch01/StreamVGGT?blobs=true', timeout=40)
        response.raise_for_status(); repository = response.json()
        weight = next(x for x in repository['siblings'] if x['rfilename'] == 'model.safetensors')
        meta = {'github_commit': COMMIT, 'hf_repository': 'lch01/StreamVGGT',
                'hf_revision': repository['sha'], 'weight': weight,
                'endpoint': 'https://hf-mirror.com', 'proxy_used': False}
        identity.write_text(json.dumps(meta, indent=2) + '\n')
    if args.provider == 'cloud':
        response = session.get(f"https://hf-mirror.com/api/models/lch01/StreamVGGT/revision/{meta['hf_revision']}?blobs=true", timeout=40)
        response.raise_for_status(); repository = response.json(); assert repository['sha'] == meta['hf_revision']
        weight = next(x for x in repository['siblings'] if x['rfilename'] == 'checkpoints.pth')
        meta = dict(meta, weight=weight, endpoint='https://cloud.tsinghua.edu.cn',
                    fallback_reason='HF mirror and redirected Xet ranges stalled; official author cloud is reachable without proxies.')
        identity = STORE / 'source_identity_cloud.json'; identity.write_text(json.dumps(meta, indent=2) + '\n')
        url = 'https://cloud.tsinghua.edu.cn/d/d6ad8f36fcd541bcb246/files/?p=%2Fcheckpoints.pth&dl=1'
    else:
        weight = meta['weight']
        url = f"https://hf-mirror.com/lch01/StreamVGGT/resolve/{meta['hf_revision']}/{weight['rfilename']}"
    model_path = STORE / weight['rfilename']
    range_download(url, model_path, weight['size'], weight['lfs']['sha256'])
    source_future.result(); pool.shutdown()
    assert (DEST / 'PINNED_COMMIT').read_text().strip() == COMMIT
    metadata = {'complete': True, **meta, 'source_identity_sha256': sha256(identity),
                'source_tree_sha256': sha256(OUT / 'official_tree.json'), 'model_path': str(model_path),
                'model_sha256': sha256(model_path), 'new_training_steps': 0}
    (OUT / 'setup.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps({'setup_complete': True, 'weight_bytes': weight['size'],
                      'weight_sha256': metadata['model_sha256']}), flush=True)


if __name__ == '__main__': main()
