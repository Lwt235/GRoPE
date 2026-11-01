"""Frozen observation-only geometry bias for the original Round7 depth pairs.

This module never opens sensor images or labels. Temperatures are metadata for
online learned content, and are deliberately absent from the cached logits.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments.round10_partition_data import dense_observations, OLD_SCENES
from experiments.round6_region_data import read_scene_selection, _load_cache
from experiments.round10_cell_probe import geometry_logits
from models.round10_partition_pe import prepare_support
from models.round11_stratum_pe import prepare_gaussian_strata, gaussian_stratum_logits

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_OUT=ROOT/'results/round12_depth_geometry'
CONFIGS={
    'full4':dict(method='full4',h=.5,temperature=5.),
    'iso4_matched':dict(method='iso4',h=.5,temperature=5.),
    'full1':dict(method='full',h=.25,temperature=0.),
    'quad4':dict(method='quad4',h=.5,temperature=5.),
    'quad16':dict(method='quad16',h=.5,temperature=5.),
    'point':dict(method='point',h=1.,temperature=5.),
    'point_pck':dict(method='point',h=.5,temperature=5.),
    'fixed':dict(method='fixed',h=.5,temperature=5.),
}
SOURCE_NAMES=(
    'ROUND12_DEPTH_DATA_PROTOCOL.md','experiments/round12_depth_geometry.py',
    'tests/test_round12_depth_geometry.py','experiments/round10_partition_data.py',
    'experiments/round6_region_data.py','experiments/round5_rgbd_data.py',
    'experiments/round10_cell_probe.py','experiments/round10_partition_probe.py',
    'models/round10_partition_pe.py','models/round10_cell_integral_fast.py',
    'models/round10_cell_integral.py','models/round11_stratum_pe.py',
    'experiments/round11_result_audit.py','experiments/round10_cell_result_audit.py',
    'experiments/round10_partition_audit.py')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk:=f.read(8*1024*1024):h.update(chunk)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def write_new(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False)


def source_hashes():return {name:sha(ROOT/name) for name in SOURCE_NAMES}


def g8_partition(device):
    """Static native pixel cells; means/covariance are not key Gaussians."""
    y,x=torch.meshgrid(torch.arange(18,device=device),torch.arange(24,device=device),indexing='ij')
    x=x.flatten();y=y.flatten()
    bounds=torch.stack([x*8,y*8,(x+1)*8,(y+1)*8],-1)
    py,px=torch.meshgrid(torch.arange(144,device=device),torch.arange(192,device=device),indexing='ij')
    return dict(bounds=bounds,owner=((py//8)*24+px//8).flatten(),
                parent=(y//2)*12+x//2,mass=torch.full((432,),64.,device=device),
                cell_bounds=(bounds.double()-.5)/8)


def prepare_forward_observation(query_cache,key_cache,device='cpu'):
    dense=dense_observations(query_cache,key_cache)
    return {name:torch.as_tensor(dense[name],device=device,dtype=torch.float32)
            for name in ('q_to_k_xy','q_to_k_pixel_cov')}


def compute_geometry(observation,role,config=None):
    """Fresh support preparation + integral, for production and paired timing.

    Input is dense predicted-pixel xy/pixel covariance in physical pixel units.
    Static partition setup, moment preparation and integral are all inside this
    function. RGB/depth inference, camera projection and transfer are upstream.
    No content feature, value, label, sensor visibility or learned tensor enters.
    """
    if role not in CONFIGS:raise ValueError('unknown frozen role')
    config=CONFIGS[role] if config is None else config
    if config!=CONFIGS[role]:raise ValueError('configuration differs from frozen selection')
    if set(observation)!= {'q_to_k_xy','q_to_k_pixel_cov'}:
        raise ValueError('geometry accepts only predicted xy and pixel covariance')
    xy,cov=observation['q_to_k_xy'],observation['q_to_k_pixel_cov']
    if xy.shape!=(27648,2) or cov.shape!=(27648,2,2):raise ValueError('wrong dense observation shape')
    p=g8_partition(xy.device)
    if config['method'] in ('full4','iso4'):
        query=prepare_gaussian_strata(xy/8,cov/64,p['owner'],p['bounds'],p['parent'])
        return gaussian_stratum_logits(query,p,config['method'],config['h'])
    budget={'quad4':4,'quad16':16}.get(config['method'],1)
    query=prepare_support(xy/8,cov/64,p['owner'],p['bounds'],p['parent'],
                          samples=budget,include_full_moments=budget==1)
    return geometry_logits(query,p,config['method'],config['h'])


def check_pair_join(q_features,k_features,pairs,caches):
    if q_features.shape!=(len(pairs),432,128) or k_features.shape!=q_features.shape:
        raise ValueError('original R7 feature shape changed')
    if len(set(map(tuple,pairs)))!=len(pairs):raise ValueError('duplicate R7 pair')
    for i,(q,k) in enumerate(pairs):
        if not np.array_equal(q_features[i],caches[q]['features']) or not np.array_equal(k_features[i],caches[k]['features']):
            raise ValueError('R7 pair order/features do not match the frozen RGB cache')


def selection_configs():
    result={};identities={}
    for run,roles in [('round10_partition_b',('full1','quad4','quad16','point','point_pck','fixed')),
                      ('round11_stratum',('full4','iso4_matched'))]:
        base=ROOT/'results'/run;selection=read(base/'selection.json');completion=read(base/'completion.json')
        if sha(base/'selection.json')!=(base/'selection.sha256').read_text().strip():raise ValueError('selection changed')
        if sha(base/'lock.json')!=(base/'lock.sha256').read_text().strip():raise ValueError('source lock changed')
        if selection['lock_sha256']!=sha(base/'lock.json') or completion['lock_sha256']!=sha(base/'lock.json'):
            raise ValueError('broken historical selection chain')
        for name in ('selection.json','lock.json'):
            if completion['files'][name]!=sha(base/name):raise ValueError('historical completion mismatch')
        lock=read(base/'lock.json')
        for name,digest in lock['source_hashes'].items():
            if sha(ROOT/name)!=digest:raise ValueError('historical implementation changed')
        for name in ('selection.json','selection.sha256','lock.json','lock.sha256','completion.json'):
            identities[str(base/name)]=sha(base/name)
        for role in roles:result[role]=selection['configurations']['full' if role=='full1' else role]
    if result!=CONFIGS:raise ValueError('selected B/R11 configuration differs from predeclared R12')
    return result,identities


def scene_spec(scene):
    if scene not in OLD_SCENES:raise ValueError('only five old development scenes are allowed')
    brown=scene.startswith('brown_')
    raw=ROOT/('data/round6_sun3d' if brown else 'data/round5_sun3d')
    selection=raw/('selection_v2.json' if brown else 'selection.json')
    cache=ROOT/('results/round6_inputs/cache' if brown else 'results/round5_inputs/cache')/scene
    r7=ROOT/('results/round7_depth_bm4' if brown else 'results/round7_depth_old')/OLD_SCENES[scene]/scene.replace('/','__')
    rows=[r for r in read(selection)['scenes'] if r['scene']==scene]
    if len(rows)!=1:raise ValueError('ambiguous old scene')
    row=rows[0];meta=read(r7/'manifest.json')
    if (row['split']!=OLD_SCENES[scene] or meta['scene']!=scene or meta['split']!=row['split']
            or meta['pairs']!=row['pairs'] or meta['sensor_depth_used_for_forward']):
        raise ValueError('R7 source/selection identity mismatch')
    if meta['selection_sha256']!=sha(selection):raise ValueError('R7 selection changed')
    if sha(r7/'inputs.npz')!=meta['input_sha256']:raise ValueError('R7 observations changed')
    return raw,selection,cache,r7,row,meta


def load_scene_cache(scene):
    raw,selection,cache,r7,row,meta=scene_spec(scene)
    frames=read_scene_selection(raw,row)
    indices=sorted({i for pair in meta['pairs'] for i in pair})
    declared={x['index']:x for x in meta['cache_sources']}
    if len(declared)!=len(meta['cache_sources']) or set(declared)!=set(indices):raise ValueError('R7 cache coverage changed')
    caches={};files={str(selection):sha(selection),str(r7/'manifest.json'):sha(r7/'manifest.json'),
                     str(r7/'inputs.npz'):meta['input_sha256']}
    for path in (raw/row['scene']/'intrinsics.txt',raw/row['pose_file'],
                 raw/row['scene']/'image_index.html',raw/row['scene']/'depth_index.html'):
        files[str(path)]=sha(path)
    for i in indices:
        path=cache/f'{i:07d}.npz';caches[i],sidecar=_load_cache(path,frames[i],row)
        if declared[i]['cache_sha256']!=sidecar['cache_sha256'] or declared[i]['rgb_sha256']!=sidecar['producer']['rgb_sha256']:
            raise ValueError('cache no longer matches original R7')
        for p in (path,path.with_suffix('.json'),frames[i].rgb_path):files[str(p)]=sha(p)
    with np.load(r7/'inputs.npz',allow_pickle=False) as z:
        q,k=z['q'],z['k']
    check_pair_join(q,k,meta['pairs'],caches)
    return caches,meta,files,r7


def freeze(out):
    if out.exists():raise ValueError('refuse overwrite of geometry root')
    configs,identities=selection_configs();scenes={};files=dict(identities)
    for scene,split in OLD_SCENES.items():
        caches,meta,scene_files,r7=load_scene_cache(scene)
        del caches
        expected=192 if split=='train' else 64
        if len(meta['pairs'])!=expected:raise ValueError('wrong old development population')
        scenes[scene]=dict(scene=scene,split=split,pairs=meta['pairs'],r7_root=str(r7),
            r7_manifest_sha256=sha(r7/'manifest.json'),r7_input_sha256=meta['input_sha256'],
            r7_label_sha256=meta['label_sha256'],label_hash_source='copied from original R7 manifest; file unopened')
        files.update(scene_files)
        print('FREEZE',scene,len(meta['pairs']),flush=True)
    for name in ('partitions.npz','completion.json'):
        p=ROOT/'results/round10_partition_data'/name;files[str(p)]=sha(p)
    # Prediction-only replay references: first fixed pair in each old val scene.
    references=[]
    for scene,split in OLD_SCENES.items():
        if split!='val':continue
        slug=scene.replace('/','__')
        for run in ('round10_partition_b','round11_stratum'):
            base=ROOT/'results'/run
            relative=f'evaluation/{slug}/predictions/{slug}/000_0000030_0000045.npz'
            p=base/relative
            if read(base/'completion.json')['files'][relative]!=sha(p):raise ValueError('old prediction identity mismatch')
            files[str(p)]=sha(p);references.append(dict(scene=scene,run=run,path=str(p)))
    out.mkdir(parents=True)
    lock=dict(created_unix=time.time(),stage='before geometry generation',source_hashes=source_hashes(),
        configurations=configs,scenes=scenes,input_files=files,references=references,
        label_files_opened=False,sensor_images_opened=False,fresh_test_read=False,
        pair_count=704,shape_per_pair=[432,432],dtype='float32',geometry_contains_content=False,
        estimated_array_bytes=8*704*432*432*4,
        preflight=dict(pair=[30,45],scenes=[s for s,p in OLD_SCENES.items() if p=='val'],
            independent_query_rows=np.linspace(0,431,12).astype(int).tolist(),
            probability_tolerance=1e-4,cast_logit_tolerance=1e-5))
    write_new(out/'lock.json',lock)
    (out/'lock.sha256').write_text(sha(out/'lock.json'))
    for name in SOURCE_NAMES:
        destination=out/'source_snapshot'/name;destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_bytes((ROOT/name).read_bytes())
    return lock


def verify_lock(out,inputs=True):
    lock=read(out/'lock.json')
    if sha(out/'lock.json')!=(out/'lock.sha256').read_text().strip():raise ValueError('geometry lock changed')
    if source_hashes()!=lock['source_hashes']:raise ValueError('source changed after freeze')
    for name,digest in lock['source_hashes'].items():
        if sha(out/'source_snapshot'/name)!=digest:raise ValueError('source snapshot changed')
    if inputs:
        for name,digest in lock['input_files'].items():
            if sha(name)!=digest:raise ValueError('frozen input changed: '+name)
    return lock


def load_first_observation(out,scene,device='cpu'):
    """Timing API: frozen first R7 pair, no labels or outcome access."""
    lock=verify_lock(Path(out),inputs=False)
    if scene not in lock['scenes']:raise ValueError('unknown frozen scene')
    caches,meta,files,_=load_scene_cache(scene)
    if any(lock['input_files'].get(p)!=v for p,v in files.items()):raise ValueError('source changed')
    q,k=meta['pairs'][0]
    return prepare_forward_observation(caches[q],caches[k],device)


def common_g8(probability):
    p=g8_partition(probability.device)
    result=probability.new_zeros((108,432));result.index_add_(0,p['parent'],probability/4)
    return result


def preflight(out,device):
    lock=verify_lock(out);path=out/'preflight.json'
    if path.exists():raise ValueError('refuse overwrite preflight')
    from experiments import round11_result_audit as REF
    from experiments import round10_cell_result_audit as BREF
    from experiments import round10_partition_audit as AREF
    with np.load(ROOT/'results/round10_partition_data/partitions.npz',allow_pickle=False) as z:parts={k:z[k] for k in z.files}
    rows=[];started=time.time();indices=np.asarray(lock['preflight']['independent_query_rows'])
    for scene in lock['preflight']['scenes']:
        caches,meta,_,_=load_scene_cache(scene);q,k=meta['pairs'][0]
        if [q,k]!=[30,45]:raise ValueError('predeclared first pair changed')
        dense=dense_observations(caches[q],caches[k])
        obs=prepare_forward_observation(caches[q],caches[k],device)
        qf=torch.nn.functional.normalize(torch.as_tensor(caches[q]['features'],device=device),dim=-1)
        kf=torch.nn.functional.normalize(torch.as_tensor(caches[k]['features'],device=device),dim=-1)
        content=qf@kf.T
        key=AREF.reference_key(dense,parts,'g8');bounds=(parts['g8_bounds']-.5)/8
        archives={r['run']:np.load(r['path'],allow_pickle=False) for r in lock['references'] if r['scene']==scene}
        for role,config in CONFIGS.items():
            g=compute_geometry(obs,role,config);cast=g.float()
            p64=(g+config['temperature']*content).softmax(-1)
            p32=(cast+config['temperature']*content).softmax(-1)
            oldrole='full' if role=='full1' else role
            run='round11_stratum' if role in ('full4','iso4_matched') else 'round10_partition_b'
            old=archives[run][oldrole+'__g8_g8']
            saved_error=float(np.abs(common_g8(p64).cpu().numpy()-old).max())
            cast_error=float((cast.double()-g).abs().max())
            probability_cast_error=float((p32.double()-p64).abs().max())
            query=(REF.reference_four_support(dense,parts,'g8') if role in ('full4','iso4_matched')
                   else AREF.reference_support(dense,parts,'g8',samples={'quad4':4,'quad16':16}.get(role,1)))
            query={name:value[indices] for name,value in query.items()}
            ref=(REF.reference_four_logits(query,key,bounds,config['method'],config['h'])
                 if role in ('full4','iso4_matched') else BREF.reference_logits(query,key,bounds,config['method'],config['h']))
            from scipy.special import softmax
            rp=softmax(ref+config['temperature']*content[indices].double().cpu().numpy(),axis=-1)
            reference_error=float(np.abs(p64[indices].cpu().numpy()-rp).max())
            if max(saved_error,probability_cast_error,reference_error)>1e-4 or cast_error>1e-5:
                raise ValueError(f'preflight numerical mismatch {scene} {role}: {saved_error} {cast_error} {probability_cast_error} {reference_error}')
            rows.append(dict(scene=scene,pair=[q,k],role=role,saved_probability_max_error=saved_error,
                cast_logit_max_error=cast_error,cast_probability_max_error=probability_cast_error,
                independent_probability_max_error=reference_error))
        for archive in archives.values():archive.close()
    write_new(path,dict(passed=True,lock_sha256=sha(out/'lock.json'),rows=rows,seconds=time.time()-started,
        label_files_opened=False,sensor_images_opened=False,independent_reference='NumPy literal stratum moments + SciPy CDF positive GL64'))
    print(json.dumps(read(path)),flush=True)


def export_scene(out,scene,device):
    lock=verify_lock(out);check=read(out/'preflight.json')
    if not check['passed'] or check['lock_sha256']!=sha(out/'lock.json'):raise ValueError('preflight missing')
    if scene not in lock['scenes']:raise ValueError('unknown scene')
    info=lock['scenes'][scene];directory=out/info['split']/scene.replace('/','__')
    if directory.exists():raise ValueError('refuse overwrite scene output')
    caches,meta,files,r7=load_scene_cache(scene)
    if any(lock['input_files'].get(p)!=v for p,v in files.items()):raise ValueError('scene source changed')
    directory.mkdir(parents=True);started=time.time();shape=(len(meta['pairs']),432,432)
    arrays={role:np.lib.format.open_memmap(directory/(role+'.npy'),mode='w+',dtype=np.float32,shape=shape) for role in CONFIGS}
    maximum={role:0. for role in CONFIGS}
    for i,(q,k) in enumerate(meta['pairs']):
        obs=prepare_forward_observation(caches[q],caches[k],device)
        for role in CONFIGS:
            g=compute_geometry(obs,role);cast=g.float()
            if not bool(torch.isfinite(cast).all()):raise ValueError('nonfinite geometry bias')
            error=float((cast.double()-g).abs().max());maximum[role]=max(maximum[role],error)
            if error>1e-5:raise ValueError('geometry float32 cast tolerance exceeded')
            arrays[role][i]=cast.cpu().numpy()
        if (i+1)%16==0:print('GEOMETRY',scene,i+1,'/',len(meta['pairs']),round(time.time()-started,1),flush=True)
    for a in arrays.values():a.flush()
    arrays.clear()
    roles={role:{**config,'path':role+'.npy','sha256':sha(directory/(role+'.npy')),
                 'dtype':'float32','shape':list(shape),'cast_logit_max_error':maximum[role]}
           for role,config in CONFIGS.items()}
    write_new(directory/'manifest.json',dict(**info,roles=roles,lock_sha256=sha(out/'lock.json'),
        preflight_sha256=sha(out/'preflight.json'),seconds=time.time()-started,
        label_files_opened=False,sensor_images_opened=False,geometry_contains_content=False,
        feature_pair_join='all q/k features bitwise identical to original R7 in original manifest order'))
    print('COMPLETE',scene,round(time.time()-started,1),flush=True)


def finalize(out):
    lock=verify_lock(out);manifests=[];counts={'train':0,'val':0}
    for scene,info in lock['scenes'].items():
        p=out/info['split']/scene.replace('/','__')/'manifest.json';record=read(p)
        if any(record[k]!=v for k,v in info.items()):raise ValueError('scene identity changed')
        if record['lock_sha256']!=sha(out/'lock.json') or record['preflight_sha256']!=sha(out/'preflight.json'):
            raise ValueError('scene generation chain changed')
        if set(record['roles'])!=set(CONFIGS):raise ValueError('missing geometry role')
        for role,config in CONFIGS.items():
            item=record['roles'][role]
            if any(item[k]!=v for k,v in config.items()):raise ValueError('changed geometry configuration')
            path=p.parent/item['path']
            if item['path']!=role+'.npy' or sha(path)!=item['sha256']:raise ValueError('geometry changed')
            a=np.load(path,mmap_mode='r',allow_pickle=False)
            if a.dtype!=np.float32 or a.shape!=(len(info['pairs']),432,432):raise ValueError('geometry shape changed')
        counts[info['split']]+=len(info['pairs'])
        manifests.append(dict(path=str(p.relative_to(out)),sha256=sha(p),scene=scene,split=info['split']))
    if counts!={'train':576,'val':128}:raise ValueError('incomplete original R7 population')
    write_new(out/'completion.json',dict(lock_sha256=sha(out/'lock.json'),preflight_sha256=sha(out/'preflight.json'),
        manifests=manifests,counts=counts,role_count=8,array_bytes=lock['estimated_array_bytes'],
        label_files_opened=False,sensor_images_opened=False,fresh_test_read=False,
        geometry_contains_content=False,completed_unix=time.time()))


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze','preflight','export-scene','finalize'])
    p.add_argument('--out',type=Path,default=DEFAULT_OUT);p.add_argument('--scene');p.add_argument('--device',default='cpu')
    args=p.parse_args();torch.set_num_threads(4)
    with torch.inference_mode():
        if args.stage=='freeze':freeze(args.out)
        elif args.stage=='preflight':preflight(args.out,args.device)
        elif args.stage=='export-scene':export_scene(args.out,args.scene,args.device)
        else:finalize(args.out)

if __name__=='__main__':main()
