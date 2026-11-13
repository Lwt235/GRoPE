"""Once-only final new-DATA identities; reuse already published prediction SHAs."""
import os,json,time
from pathlib import Path
from experiments import native_feature_field_data as d

def manifest_DATA():
    known={}
    for rec in d.read(d.BASEOUT/'source_targets.json')['records']:known[rec['path']]=rec['sha256']
    for row in ('C','J'):
        root=d.BASEOUT/row
        for p in (root/'completed').glob('*.json'):
            x=d.read(p)
            for k in ('feature','depth'):known[x[k+'_path']]=x[k+'_sha256']
        for p in (root/'states').glob('*.json'):
            x=d.read(p);known[x['path']]=x['sha256']
        for p in (root/'fits').glob('*/status.json'):
            x=d.read(p);known[x['path']]=x['sha256']
        for p in (root/'formation').glob('*.json'):
            x=d.read(p);known[x['field_path']]=x['field_sha256']
    files=[]
    for p in sorted(d.BASECACHE.rglob('*')):
        if p.is_file():files.append(dict(path=str(p),bytes=p.stat().st_size,sha256=known.get(str(p)) or d.sha(p),identity_source='atomic publication identity' if str(p) in known else 'single final hash of new artifact'))
    d.write(d.BASEOUT/'DATA_manifest.json',dict(root=str(d.BASECACHE),files=files,bytes=sum(r['bytes'] for r in files),unchanged_read_only_dependencies=str(d.BASEOUT/'input_manifest.json'),identity_reuse='No repeated bulk hash for already published predictions/states/finals; old weights and data not rehashed'))
    print(dict(DATA_files=len(files),bytes=sum(x['bytes'] for x in files)),flush=True)
if __name__=='__main__':manifest_DATA()
