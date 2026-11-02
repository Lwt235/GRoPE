"""Reuse only the existing R52 development arrays and their identities."""
from experiments.round51_data import ROOT,read,write,sha,load,pixels

PREVIOUS = ROOT/'results/round52_camera_frame/attempt1'
OUT = ROOT/'results/round55_budget_diagnostic/attempt1'


def build():
    from experiments import round52_train as previous
    previous.check()
    if not read(PREVIOUS/'completion.json')['round_complete']:raise ValueError('R52 incomplete')
    destination=OUT/'data/manifest.json'
    if destination.exists():raise FileExistsError('retain data manifest')
    manifest=read(PREVIOUS/'data/manifest.json')
    for e in manifest['entries']:
        if sha(e['path'])!=e['sha256']:raise ValueError('cached data changed')
    write(destination,dict(manifest,reused_from=str(PREVIOUS/'data/manifest.json'),
        reused_manifest_sha256=sha(PREVIOUS/'data/manifest.json'),confirmation_access=False))
    print('DATA',manifest['counts'],'same arrays; no new frames',flush=True)


if __name__=='__main__':build()
