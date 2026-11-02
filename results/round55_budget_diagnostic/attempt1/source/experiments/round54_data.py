"""Exact, identity-checked reuse of R52's already-used matching data."""
from experiments.round51_data import ROOT, read, write, sha, load, pixels

PREVIOUS = ROOT/'results/round52_camera_frame/attempt1'
OUT = ROOT/'results/round54_calibrated_plane/attempt1'


def build():
    from experiments import round52_train as previous
    lock=previous.check()
    if not read(PREVIOUS/'completion.json')['round_complete']:
        raise ValueError('finish R52 first')
    destination=OUT/'data/manifest.json'
    if destination.exists():raise FileExistsError('retain data manifest')
    manifest=read(PREVIOUS/'data/manifest.json')
    for e in manifest['entries']:
        if sha(e['path'])!=e['sha256']:raise ValueError('cached data changed')
    write(destination,dict(manifest,reused_from=str(PREVIOUS/'data/manifest.json'),
        reused_manifest_sha256=sha(PREVIOUS/'data/manifest.json'),confirmation_access=False))
    print('DATA',manifest['counts'],'exact R52 arrays; no new frames',flush=True)


if __name__=='__main__':build()
