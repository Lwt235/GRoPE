"""Acquire a fixed sparse subset from the official public SUN3D server.

Selection uses frame indices alone. No image, depth, or method score is inspected.
Native metadata and downloaded files are hashed. Run --metadata first, then --frames.
"""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import re
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/round5_sun3d'
SCENES = {
    'train': ['harvard_c3/hv_c3_1', 'harvard_c5/hv_c5_1', 'harvard_c6/hv_c6_1'],
    'val': ['harvard_c8/hv_c8_3'],
    'test': ['harvard_c11/hv_c11_2', 'hotel_umd/maryland_hotel3',
             'mit_lab_hj/lab_hj_tea_nov_2_2012_scan1_erika',
             'mit_76_studyroom/76-1studyroom2',
             'mit_dorm_next_sj/dorm_next_sj_oct_30_2012_scan1_erika'],
}

def fetch(url, path):
    path = Path(path)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=90) as response, path.with_suffix(path.suffix+'.part').open('wb') as f:
                while chunk := response.read(1024*1024):
                    f.write(chunk)
            path.with_suffix(path.suffix+'.part').replace(path)
            return path
        except Exception:
            if attempt == 3:
                raise
            time.sleep(1+attempt)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def listing(url, path, extension):
    text = fetch(url, path).read_text()
    return sorted(set(re.findall(r'href="([^"/]+\.'+extension+r')"', text)))

def metadata():
    import numpy as np
    rows = []
    for split, names in SCENES.items():
        for name in names:
            d = OUT / name
            base = 'https://sun3d.cs.princeton.edu/data/' + name + '/'
            rgb = listing(base+'image/', d/'image_index.html', 'jpg')
            depth = listing(base+'depth/', d/'depth_index.html', 'png')
            poses = listing(base+'extrinsics/', d/'extrinsics_index.html', 'txt')
            if not rgb or not depth or not poses:
                raise RuntimeError((name,len(rgb),len(depth),poses))
            pose = fetch(base+'extrinsics/'+poses[-1], d/'extrinsics'/poses[-1])
            K = fetch(base+'intrinsics.txt', d/'intrinsics.txt')
            nposes = np.loadtxt(pose).shape[0] // 3
            if nposes != len(rgb):
                raise RuntimeError((name,'pose/image count mismatch',nposes,len(rgb)))
            # SUN3D timestamps are integer microseconds, associated by nearest time.
            stamp = lambda s: int(s.split('-')[1].split('.')[0])
            depth_t = np.array([stamp(s) for s in depth], dtype=np.int64)
            nq = {'train':96, 'val':32, 'test':48}[split]
            queries = np.unique(np.linspace(30,len(rgb)-91,nq).astype(int))
            pairs = [(int(q),int(q+offset)) for q in queries for offset in (15,60)]
            indices = sorted(set(i for pair in pairs for i in pair))
            frames = []
            for i in indices:
                dt = np.abs(depth_t-stamp(rgb[i]))
                j = int(dt.argmin())
                frames.append({'index':i,'rgb':rgb[i],'depth':depth[j],
                               'time_gap_us':int(dt[j])})
            row = {'scene':name,'split':split,'base_url':base,'rgb_count':len(rgb),
                   'depth_count':len(depth),'pose_count':nposes,'pose_file':str(pose.relative_to(OUT)),
                   'pose_sha256':sha(pose),'K_sha256':sha(K),'pairs':pairs,'frames':frames}
            rows.append(row)
            print(name,split,len(rgb),len(frames),len(pairs),'max_sync_us',max(f['time_gap_us'] for f in frames),flush=True)
    path = OUT/'selection.json'
    path.write_text(json.dumps({'selection':'linspace query indices, offsets 15 and 60, nearest depth timestamps',
                               'scene_split_version':1,'scenes':rows},indent=2))
    print('SELECTION',sha(path),flush=True)

def frames():
    manifest = json.loads((OUT/'selection.json').read_text())
    jobs = {}
    for s in manifest['scenes']:
        for f in s['frames']:
            for key, sub in [('rgb','image'),('depth','depth')]:
                path = OUT/s['scene']/sub/f[key]
                jobs[str(path)] = (s['base_url']+sub+'/'+f[key],path)
    def one(job):
        url,path = job
        fetch(url,path)
        return {'path':str(path.relative_to(OUT)),'url':url,'bytes':path.stat().st_size,'sha256':sha(path)}
    records=[]
    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        for record in pool.map(one,jobs.values()):
            records.append(record)
            if len(records)%200==0: print('FILES',len(records),'/',len(jobs),flush=True)
    (OUT/'download_manifest.json').write_text(json.dumps(records,indent=2))
    print('DONE',len(records),sum(r['bytes'] for r in records),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--metadata',action='store_true'); p.add_argument('--frames',action='store_true')
    args=p.parse_args()
    if args.metadata: metadata()
    if args.frames: frames()
