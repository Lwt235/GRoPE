"""Identity-checked reuse of already-used R51 arrays; no new frames opened."""
from experiments.round51_data import ROOT, OUT as PREVIOUS, read, write, sha, load, pixels

OUT = ROOT/'results/round52_camera_frame/attempt1'


def build():
    destination = OUT/'data/manifest.json'
    if destination.exists():
        raise FileExistsError('retain existing data manifest')
    lock = read(PREVIOUS/'lock.json')
    if sha(PREVIOUS/'lock.json') != (PREVIOUS/'lock.sha256').read_text().strip():
        raise ValueError('R51 lock identity')
    if sha(PREVIOUS/'data/manifest.json') != lock['data_manifest_sha256']:
        raise ValueError('R51 manifest identity')
    manifest = read(PREVIOUS/'data/manifest.json')
    for entry in manifest['entries']:
        if sha(entry['path']) != entry['sha256']:
            raise ValueError('R51 arrays changed')
    write(destination, dict(manifest, reused_from=str(PREVIOUS/'data/manifest.json'),
        reused_manifest_sha256=sha(PREVIOUS/'data/manifest.json'), confirmation_access=False))
    print('DATA reused exact R51 arrays', manifest['counts'], flush=True)


if __name__ == '__main__':
    build()
