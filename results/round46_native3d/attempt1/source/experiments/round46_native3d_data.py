"""Center-lifted geometry from already-used RGB prediction caches; no new labels."""
from pathlib import Path
import numpy as np
import torch
from experiments.round43_frequency_train import ROOT,sha,read,write,checked
from experiments.round5_train import FIELDS
from models.round46_native3d_pe import TWO_D

CACHE=ROOT/'results/round5_inputs/cache'
MEANS=('qmean','kmean','k_to_q','q_to_k')
COVS=('qcov','kcov','k_to_qcov','q_to_kcov')


def pixels():
    yy,xx=np.mgrid[:18,:24]
    return np.stack((xx.ravel()*8+3.5,yy.ravel()*8+3.5),-1)


def center_points(frame):
    z=np.nan_to_num(frame['predicted_depth'].astype(np.float64),nan=10.,posinf=20.,neginf=.1).clip(.1,20)
    if z.shape!=(144,192):raise ValueError('original predicted-depth resolution required')
    depth=z.reshape(18,8,24,8).transpose(0,2,1,3)[:,:,3:5,3:5].mean((-1,-2)).reshape(-1)
    ray=np.column_stack((pixels(),np.ones(432)))@np.linalg.inv(frame['K']).T
    return ray*depth[:,None]


def project(points,K):
    z=points[:,2].clip(.1,None)
    ray=points.copy();ray[:,2]=z
    homogeneous=ray@K.T
    return (homogeneous[:,:2]/homogeneous[:,2,None]/8).clip(-128,128)


def pair_geometry(q,k):
    qx,kx=center_points(q),center_points(k)
    T=np.linalg.inv(q['c2w'])@k['c2w']
    key_in_query=kx@T[:3,:3].T+T[:3,3]
    invT=np.linalg.inv(T);query_in_key=qx@invT[:3,:3].T+invT[:3,3]
    together=np.concatenate((qx,key_in_query))
    origin=together.mean(0)
    radius=max(float(np.sqrt(np.mean(np.sum((together-origin)**2,-1)))),1e-3)
    qchart=12*(qx-origin)/radius;kchart=12*(key_in_query-origin)/radius
    result=dict(qmean=qchart,kmean=kchart,k_to_q=kchart,q_to_k=qchart,
        qmean_2d=pixels()/8,kmean_2d=pixels()/8,
        k_to_q_2d=project(key_in_query,q['K']),q_to_k_2d=project(query_in_key,k['K']))
    return {name:value.astype('float32') for name,value in result.items()}


def capture_frame_records(entries):
    records=[]
    for e in entries:
        for index in sorted({i for pair in e['pairs'] for i in pair}):
            path=CACHE/e['scene']/f'{index:07d}.npz';meta=path.with_suffix('.json');m=read(meta)
            if m['cache_sha256']!=sha(path) or m['producer']['original_index']!=index:
                raise ValueError('cache identity')
            if m['producer']['source_sha256']!=sha(ROOT/'experiments/round5_cache.py'):
                raise ValueError('cache producer changed')
            records.append(dict(scene=e['scene'],split=e['split'],index=index,path=str(path),
                hashes={str(path):sha(path),str(meta):sha(meta)}))
    return records


def load_frames(lock,scene):
    frames={}
    for record in lock['frames']:
        if record['scene']!=scene:continue
        for p,h in record['hashes'].items():
            if sha(p)!=h:raise ValueError('changed frame cache')
        with np.load(record['path'],allow_pickle=False) as z:
            frames[record['index']]={name:z[name] for name in ('features','predicted_depth','K','c2w')}
    return frames


def build(out,splits):
    lock=checked(out)
    if 'test' in splits:
        if len(read(out/'audit_training/completion.json')['records'])!=60:raise ValueError('seal all models before old-eval geometry export')
    for split in splits:
        folder=out/'data'/split
        if folder.exists():raise ValueError('retain geometry cache')
        folder.mkdir(parents=True)
        entries=[]
        for entry in lock['entries']:
            if entry['split']!=split:continue
            frames=load_frames(lock,entry['scene']);items=[]
            oldpath=Path(entry['folder'])/'inputs.npz'
            if sha(oldpath)!=entry['hashes'][str(oldpath)]:raise ValueError('changed old inputs')
            with np.load(oldpath,allow_pickle=False) as old:
                for j,(qi,ki) in enumerate(entry['pairs']):
                    q,k=frames[qi],frames[ki]
                    if not np.array_equal(q['features'],old['q'][j]) or not np.array_equal(k['features'],old['k'][j]):
                        raise ValueError('visual feature identity changed')
                    items.append(dict(q=q['features'],k=k['features'],**pair_geometry(q,k)))
            path=folder/(entry['scene'].replace('/','__')+'.npz')
            np.savez_compressed(path,**{key:np.stack([item[key] for item in items]) for key in items[0]})
            entries.append(dict(scene=entry['scene'],pairs=entry['pairs'],path=str(path),sha256=sha(path)))
            print('DATA',split,entry['scene'],len(items),flush=True)
        write(folder/'manifest.json',dict(split=split,entries=entries,labels_reused_unchanged=True,features_identical=True))
        (folder/'manifest.sha256').write_text(sha(folder/'manifest.json')+'\n')


def load(split,out,device):
    lock=checked(out);folder=out/'data'/split
    if sha(folder/'manifest.json')!=(folder/'manifest.sha256').read_text().strip():raise ValueError('changed geometry manifest')
    if not read(out/f'data_audit_{split}.json')['passed']:raise ValueError('independent geometry audit required')
    manifest=read(folder/'manifest.json');arrays={};ids=[]
    expected=[e for e in lock['entries'] if e['split']==split]
    if [e['scene'] for e in manifest['entries']]!=[e['scene'] for e in expected]:raise ValueError('scene order')
    for e,original in zip(manifest['entries'],expected):
        if e['pairs']!=original['pairs'] or sha(e['path'])!=e['sha256']:raise ValueError('pair/geometry identity')
        with np.load(e['path'],allow_pickle=False) as z:
            for key in z.files:arrays.setdefault(key,[]).append(z[key])
        labels=Path(original['folder'])/'labels.npz'
        if sha(labels)!=original['hashes'][str(labels)]:raise ValueError('label identity')
        with np.load(labels,allow_pickle=False) as z:
            for key in ('gt_valid','pixcontinuous','key_indices','key_weights'):arrays.setdefault(key,[]).append(z[key])
        ids.extend(dict(scene=e['scene'],pair_id=f'{q}:{k}',q_index=q,k_index=k) for q,k in e['pairs'])
    data={k:torch.from_numpy(np.concatenate(v)).to(device) for k,v in arrays.items()}
    for k in data:
        if k=='gt_valid':data[k]=data[k].bool()
        elif k=='key_indices':data[k]=data[k].long()
        else:data[k]=data[k].float()
    return data,ids


def for_role(data,role):
    result={k:data[k] for k in ('q','k','gt_valid','pixcontinuous','key_indices','key_weights')}
    for mean,cov in zip(MEANS,COVS):
        value=data[mean+'_2d' if role in TWO_D else mean]
        result[mean]=value
        # All current roles use position points, not physical input covariances.
        result[cov]=torch.zeros(*value.shape,value.shape[-1],dtype=value.dtype,device=value.device)
    return result
