"""Fetch pinned VGGT research dependencies, using the HF mirror without proxies."""
import hashlib
import json
import tarfile
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = Path('/mnt/f70f6709-366e-49a0-861f-497645d98975/users/liwt/data/GRoPE/gaussian_memory')
COMMIT = 'a288dd0f14786c93483e45524328726ab7b1b4ce'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def download(session, url, path, expected_size=None, expected_sha=None):
    if path.exists() and (expected_size is None or path.stat().st_size == expected_size):
        if expected_sha is None or sha256(path) == expected_sha:
            return
    part = path.with_suffix(path.suffix + '.partial')
    for attempt in range(8):
        try:
            offset = part.stat().st_size if part.exists() else 0
            with session.get(url, headers={'Range': f'bytes={offset}-'} if offset else {},
                             stream=True, timeout=(25, 60)) as r:
                r.raise_for_status()
                append = offset > 0 and r.status_code == 206
                count = offset if append else 0
                last = time.monotonic()
                with open(part, 'ab' if append else 'wb') as f:
                    for chunk in r.iter_content(4 << 20):
                        f.write(chunk)
                        count += len(chunk)
                        if time.monotonic() - last > 20:
                            print(json.dumps({'file': path.name, 'bytes': count, 'size': expected_size}), flush=True)
                            last = time.monotonic()
            if expected_size is not None and part.stat().st_size != expected_size:
                raise RuntimeError('incomplete download')
            digest = sha256(part)
            if expected_sha is not None and digest != expected_sha:
                raise RuntimeError('download hash mismatch; preserved partial for inspection')
            part.replace(path)
            print(json.dumps({'downloaded': str(path), 'sha256': digest}), flush=True)
            return
        except (requests.RequestException, RuntimeError) as e:
            # Do not print response URLs: redirects may contain signed credentials.
            print(json.dumps({'file': path.name, 'attempt': attempt + 1,
                              'error_type': type(e).__name__}), flush=True)
            if attempt == 7:
                raise RuntimeError(f'Failed to fetch {path.name}') from None
            time.sleep(min(5 * (attempt + 1), 30))


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.trust_env = False
    meta = session.get('https://hf-mirror.com/api/models/facebook/VGGT-1B?blobs=true', timeout=40)
    meta.raise_for_status()
    meta = meta.json()
    weight = next(x for x in meta['siblings'] if x['rfilename'] == 'model.safetensors')
    identity = {'github_commit': COMMIT, 'hf_revision': meta['sha'], 'weight': weight,
                'endpoint': 'https://hf-mirror.com', 'proxy_used': False}
    (DATA / 'source_identity.json').write_text(json.dumps(identity, indent=2) + '\n')
    print(json.dumps(identity), flush=True)
    archive = DATA / f'vggt-{COMMIT}.tar.gz'
    dest = ROOT / 'third_party' / 'vggt'
    if not dest.exists():
        download(session, f'https://codeload.github.com/facebookresearch/vggt/tar.gz/{COMMIT}', archive)
        with tarfile.open(archive) as tf:
            base = f'vggt-{COMMIT}/'
            members = []
            for member in tf.getmembers():
                if not member.name.startswith(base) or not member.name[len(base):]:
                    continue
                relative = Path(member.name[len(base):])
                if relative.is_absolute() or '..' in relative.parts or member.issym() or member.islnk():
                    raise RuntimeError('unsafe archive member')
                member.name = str(relative)
                members.append(member)
            dest.mkdir()
            tf.extractall(dest, members=members)
        (dest / 'PINNED_COMMIT').write_text(COMMIT + '\n')
    assert (dest / 'PINNED_COMMIT').read_text().strip() == COMMIT
    download(session, f"https://hf-mirror.com/facebook/VGGT-1B/resolve/{meta['sha']}/model.safetensors",
             DATA / 'model.safetensors', weight['size'], weight['lfs']['sha256'])
    print(json.dumps({'ready': True, 'source': str(dest), 'weight': str(DATA / 'model.safetensors')}), flush=True)


if __name__ == '__main__':
    main()
