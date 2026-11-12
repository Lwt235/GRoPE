"""Pinned official AnySplat source and public weights; downloads never use proxies."""
import argparse
import concurrent.futures
import hashlib
import json
import tarfile
import threading
from pathlib import Path

import requests

from experiments.gaussian_memory_setup import ROOT, DATA, download, sha256

OUT = ROOT / 'results/gaussian_memory/gaussian_scene_memory1'
ASSETS = DATA / 'gaussian_scene_assets_v1'
SOURCE = ROOT / 'third_party/anysplat'
COMMIT = '5f5e208a7dd57d52e43ea0d553a95eab526e8775'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def source(session):
    archive = ASSETS / f'anysplat-{COMMIT}.tar.gz'
    if not (SOURCE / 'PINNED_COMMIT').exists():
        download(session, f'https://codeload.github.com/InternRobotics/AnySplat/tar.gz/{COMMIT}', archive)
        SOURCE.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive) as tf:
            for member in tf:
                relative = Path(*Path(member.name).parts[1:])
                if not relative.parts:
                    continue
                if relative.is_absolute() or '..' in relative.parts or member.issym() or member.islnk():
                    raise RuntimeError('Unsafe source archive entry')
                if relative.parts[0] not in ('src', 'config', 'README.md', 'LICENSE', 'requirements.txt'):
                    continue
                member.name = str(relative)
                tf.extract(member, SOURCE)
        (SOURCE / 'PINNED_COMMIT').write_text(COMMIT + '\n')
    assert (SOURCE / 'PINNED_COMMIT').read_text().strip() == COMMIT
    write(OUT / 'setup/anysplat_source.json', dict(
        commit=COMMIT, repository='InternRobotics/AnySplat', proxy_used=False,
        archive_sha256=sha256(archive), files={str(f.relative_to(SOURCE)):sha256(f)
            for f in SOURCE.rglob('*') if f.is_file() and '__pycache__' not in f.parts}))
    print(json.dumps({'source_ready': True, 'commit': COMMIT}), flush=True)


def source_tree():
    """Fetch only runtime text files; check Git object IDs against official tree."""
    manifest = json.loads((OUT / 'setup/official_tree.json').read_text())
    assert manifest['commit'] == COMMIT
    local = threading.local()
    def fetch(item):
        relative = Path(item['path'])
        assert not relative.is_absolute() and '..' not in relative.parts
        path = SOURCE / relative
        def valid(data):
            return hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest() == item['sha']
        if path.exists() and valid(path.read_bytes()):
            return
        if item['size'] == 0:
            content = b''
        else:
            if not hasattr(local, 'session'):
                local.session = requests.Session()
                local.session.trust_env = False
            response = local.session.get(f'https://raw.githubusercontent.com/InternRobotics/AnySplat/{COMMIT}/{relative.as_posix()}', timeout=(15, 45))
            response.raise_for_status()
            content = response.content
        assert len(content) == item['size'] and valid(content), str(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_name(path.name + '.partial')
        part.write_bytes(content)
        part.replace(path)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for i, _ in enumerate(pool.map(fetch, manifest['files']), 1):
            if i % 25 == 0:
                print(json.dumps({'verified_source_files': i, 'total': len(manifest['files'])}), flush=True)
    (SOURCE / 'PINNED_COMMIT').write_text(COMMIT + '\n')
    write(OUT / 'setup/anysplat_source.json', dict(commit=COMMIT, repository='InternRobotics/AnySplat',
        proxy_used=False, official_git_tree='official_tree.json',
        files={item['path']:sha256(SOURCE / item['path']) for item in manifest['files']}))
    print(json.dumps({'source_ready': True, 'commit': COMMIT}), flush=True)


def weights(session, metadata_via_proxy=False):
    endpoint = 'https://hf-mirror.com'
    metadata_session = session
    if metadata_via_proxy:
        # The mirror is unreachable from this host. Only small metadata/config
        # and a redirect header use the existing proxy. Binary transfer below
        # always uses the explicit trust_env=False session.
        endpoint = 'https://huggingface.co'
        metadata_session = requests.Session()
    response = metadata_session.get(endpoint + '/api/models/lhjiang/anysplat?blobs=true', timeout=(15, 45))
    response.raise_for_status()
    metadata = response.json()
    write(OUT / 'setup/anysplat_hf_metadata.json', metadata)
    revision = metadata['sha']
    selected = [x for x in metadata['siblings'] if x['rfilename'] in ('config.json', 'model.safetensors', 'pytorch_model.bin')]
    assert any(x['rfilename'] == 'config.json' for x in selected)
    assert any(x['rfilename'] in ('model.safetensors', 'pytorch_model.bin') for x in selected)
    folder = ASSETS / 'anysplat' / revision
    folder.mkdir(parents=True, exist_ok=True)
    transfer_hosts = {}
    for item in selected:
        name = item['rfilename']
        url = f'{endpoint}/lhjiang/anysplat/resolve/{revision}/{name}'
        if metadata_via_proxy and name == 'config.json':
            config = metadata_session.get(url, timeout=(15, 45))
            config.raise_for_status()
            assert len(config.content) < 100_000
            (folder / name).write_bytes(config.content)
        else:
            if metadata_via_proxy:
                header = metadata_session.get(url, stream=True, allow_redirects=False, timeout=(15, 45))
                assert header.status_code in (301, 302, 303, 307, 308), header.status_code
                url = header.headers['Location']
                header.close()
            from urllib.parse import urlsplit
            transfer_hosts[name] = urlsplit(url).hostname
            download(session, url, folder / name, item.get('size'), item.get('lfs', {}).get('sha256'))
    write(OUT / 'setup/anysplat_weights.json', dict(
        revision=revision, folder=str(folder), endpoint=endpoint, binary_proxy_used=False,
        metadata_proxy_used=metadata_via_proxy, binary_transfer_hosts=transfer_hosts,
        files={x['rfilename']:sha256(folder / x['rfilename']) for x in selected}))
    print(json.dumps({'weights_ready': True, 'revision': revision}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['source', 'source-tree', 'weights'])
    parser.add_argument('--metadata-via-proxy', action='store_true')
    args = parser.parse_args()
    ASSETS.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.trust_env = False
    if args.action == 'source':
        source(session)
    elif args.action == 'source-tree':
        source_tree()
    else:
        weights(session, args.metadata_via_proxy)


if __name__ == '__main__':
    main()
