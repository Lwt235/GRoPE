"""Frozen base-region pose confirmation: depth acquisition, cache, existing models."""
import argparse
import concurrent.futures
import os
from pathlib import Path
import time
import urllib.request

import numpy as np
import torch
from PIL import Image

from experiments.round65_common import ROOT, read, write, sha
from experiments.round65_run import configure
from experiments.round58_data import tokenize
from experiments.round60_data import scene_reader, sample_inputs, attach_supervision
from experiments.round68_fixed_region_data import thin_regions, shuffle_indices
from experiments.region_geometry_controls import dependencies
from experiments.region_retokenization import ROLES, SEEDS, make_model, infer, load_arrays
CONDITIONS = ('base',)
from experiments.region_pose_confirmation_metadata import PREP, PLAN, RAW, DEVELOPMENT, checked

OUT = ROOT/'results/region_pose_confirmation/attempt1'
CACHE = RAW.parent/'cache'
PARENT = ROOT/'results/round68_fixed_region/attempt1'
FIELDS = ('features','centers','covariance','world_centers','overlap')


def preserved(path, value):
    path=Path(path)
    assert not path.exists(), 'preserve existing artifact: '+str(path)
    write(path,value)


@torch.no_grad()
def convert(batch, identities, device='cpu'):
    tensors={k:torch.as_tensor(batch[k],device=device) for k in ('points','world','clean')}
    base=tokenize(tensors)
    output={k:base[k].cpu().numpy() for k in FIELDS}
    output['permutation']=shuffle_indices(identities)
    return dict(base=output)


def old_conversion_check(device='cpu'):
    sample=load_arrays(PARENT/'data/eval_base_sample.npz')
    old=load_arrays(PARENT/'data/eval_base.npz')
    actual=convert(sample,old['ids'][:2],device)['base']
    errors={k:float(np.abs(actual[k]-old[k][:2]).max()) for k in FIELDS}
    np.testing.assert_array_equal(actual['overlap'],old['overlap'][:2])
    np.testing.assert_array_equal(actual['permutation'],shuffle_indices(old['ids'][:2]))
    assert max(errors.values())<2e-4,errors
    return dict(passed=True,device=device,errors=errors,pairs=2,
                sample_sha256=sha(PARENT/'data/eval_base_sample.npz'),
                reference_sha256=sha(PARENT/'data/eval_base.npz'))


def freeze():
    cohort=checked()
    assert not OUT.exists()
    metadata=read(PREP/'metadata_accounted.json')
    assert read(PREP/'metadata_review.json')['passed']
    selection=read(RAW/'selection.json')
    assert sha(RAW/'selection.json')==metadata['selection_sha256']
    assert sha(RAW/'metadata_manifest.json')==metadata['manifest_sha256']
    assert not list(RAW.rglob('*.png')) and not list(RAW.rglob('*.jpg'))
    paths=dependencies([Path(__file__), ROOT/'experiments/region_pose_confirmation_score.py',
                        ROOT/'experiments/region_pose_solver.py', ROOT/'experiments/region_pose_analysis.py',
                        ROOT/'experiments/region_pose_utility.py'])
    sources={str(p):sha(p) for p in paths}
    sources.update(cohort['source_sha256'])
    for name in ('cohort_lock.json','scene_selection.json','metadata_accounted.json','metadata_review.json'):
        sources[str(PREP/name)]=sha(PREP/name)
    for name in ('selection.json','metadata_manifest.json'):
        sources[str(RAW/name)]=sha(RAW/name)
    for r in read(RAW/'metadata_manifest.json'):
        path=RAW/r['path'];assert sha(path)==r['sha256'];sources[str(path)]=r['sha256']
    for item in read(PREP/'scene_selection.json')['records']:
        path=RAW/item['place']/'scan_index.html'
        assert sha(path)==item['listing_sha256'];sources[str(path)]=item['listing_sha256']
    sources[str(PARENT/'data/val.npz')]=sha(PARENT/'data/val.npz')
    for item in cohort['models']:
        sources[item['validation_path']]=sha(item['validation_path'])
    planned=[r['scene'] for r in read(PREP/'scene_selection.json')['records']]
    available=[r['scene'] for r in selection['scenes']]
    assert len(cohort['places'])==10 and len(available)>=2
    assert metadata['pairs']==96*len(available)
    preserved(OUT/'evaluation_lock.json',dict(created_unix=time.time(),source_sha256=sources,
        models=cohort['models'],roles=ROLES,seeds=SEEDS,conditions=CONDITIONS,
        planned_scenes=planned,available_scenes=available,unavailable=metadata['failures'],
        complete_cohort=metadata['complete_cohort'],planned_pairs=960,scheduled_pairs=metadata['pairs'],new_fits=0,
        pose_rule=dict(consensus_radius_pair_units=.1,maximum_consensus_refits=5,alignment_rmse_threshold_m=.2,
                       bootstrap_seeds=[94001,94002,94003],bootstrap_replicates=10000),
        old_conversion_check=old_conversion_check(),pixels_access_permitted=True,
        inference='original FP32 batch16, no training/adaptation/selection',goal_complete=False))
    (OUT/'evaluation_lock.sha256').write_text(sha(OUT/'evaluation_lock.json')+'\n')
    print('EVALUATION_FROZEN',sha(OUT/'evaluation_lock.json'),flush=True)


def check():
    lock=read(OUT/'evaluation_lock.json')
    assert sha(OUT/'evaluation_lock.json')==(OUT/'evaluation_lock.sha256').read_text().strip()
    for path,digest in lock['source_sha256'].items():assert sha(path)==digest,path
    return lock


def configure_gpu():
    configure()
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    total=torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(3*2**30/total)


def preflight():
    lock=check();configure_gpu()
    assert not (OUT/'preflight.json').exists()
    conversion=old_conversion_check('cuda')
    validation=load_arrays(PARENT/'data/val.npz')
    records=[]
    for item in lock['models']:
        model=make_model(item['role']).cuda().eval()
        state=torch.load(item['checkpoint_path'],map_location='cpu',weights_only=True)
        assert (state['role'],state['seed'],state['epoch'])==(item['role'],item['seed'],item['selected_epoch'])
        model.load_state_dict(state['state_dict'])
        logits=infer(model,validation)
        previous=load_arrays(item['validation_path'])
        assert np.array_equal(previous['ids'],validation['ids'])
        error=float(np.abs(logits-previous['logits']).max())
        changes=sum(int((a.argmax(-1)!=b.argmax(-1)).sum()) for a,b in
                    ((logits,previous['logits']),(logits.swapaxes(-1,-2),previous['logits'].swapaxes(-1,-2))))
        assert error<2e-4 and changes==0,(item['role'],item['seed'],error,changes)
        records.append(dict(role=item['role'],seed=item['seed'],maximum_logit_error=error,prediction_changes=changes))
        print('PREFLIGHT',item['role'],item['seed'],error,changes,flush=True)
        del model;torch.cuda.empty_cache()
    preserved(OUT/'preflight.json',dict(passed=True,records=records,conversion=conversion,
        completed_unix=time.time(),evaluation_lock_sha256=sha(OUT/'evaluation_lock.json')))


def download():
    lock=check();assert read(OUT/'preflight.json')['passed']
    assert not (OUT/'download_complete.json').exists()
    if not (OUT/'data_access_started.json').exists():
        record=dict(started_unix=time.time(),scenes=lock['available_scenes'],
                    metadata_only_missing=lock['unavailable'],evaluation_lock_sha256=sha(OUT/'evaluation_lock.json'),
                    scope='Frozen new-depth acquisition; these available places now reserved as used.',
                    no_adaptation_or_new_model_selection=True)
        preserved(OUT/'data_access_started.json',record)
        preserved(ROOT/'results/research_data_usage/region_pose_confirmation_access_20261109.json',record)
    jobs={}
    for row in read(RAW/'selection.json')['scenes']:
        for f in row['frames']:
            path=RAW/row['scene']/'depth'/f['depth']
            jobs[str(path)]=(row['base_url']+'depth/'+f['depth'],path)
    def fetch(job):
        url,path=job
        assert url.startswith('https://sun3d.cs.princeton.edu/data/') and url.endswith('.png')
        path.parent.mkdir(parents=True,exist_ok=True)
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        for attempt in range(3):
            try:
                if not path.exists():
                    partial=path.with_suffix('.png.part')
                    with opener.open(url,timeout=35) as response,partial.open('wb') as f:
                        while chunk:=response.read(2**20):f.write(chunk)
                    with Image.open(partial) as image:image.verify()
                    partial.replace(path)
                with Image.open(path) as image:image.verify()
                return dict(path=str(path),url=url,bytes=path.stat().st_size,sha256=sha(path))
            except Exception as error:
                print('DEPTH_RETRY',url,attempt+1,str(error),flush=True)
                if attempt==2:raise
                time.sleep(attempt+1)
    rows=[];failures=[]
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        tasks={pool.submit(fetch,job):job[0] for job in jobs.values()}
        for task in concurrent.futures.as_completed(tasks):
            try:rows.append(task.result())
            except Exception as error:failures.append(dict(url=tasks[task],error=str(error)))
            if (len(rows)+len(failures))%40==0:
                write(OUT/'download_progress.json',dict(completed=len(rows),total=len(jobs),failures=failures,unix=time.time()))
                print('DEPTHS',len(rows),'/',len(jobs),flush=True)
    if failures:
        write(OUT/f'download_failure_{time.time_ns()}.json',dict(failures=failures,completed=len(rows)))
        raise RuntimeError('Preserve failed sources; retry the same requests only.')
    rows.sort(key=lambda r:r['path'])
    preserved(OUT/'download_manifest.json',rows)
    preserved(OUT/'download_complete.json',dict(files=len(rows),bytes=sum(r['bytes'] for r in rows),
        manifest_sha256=sha(OUT/'download_manifest.json'),completed_unix=time.time(),rgb_downloaded=False,proxy_disabled=True))
    print('DOWNLOAD_COMPLETE',len(rows),flush=True)


def raw_lock():
    done=read(OUT/'download_complete.json')
    assert sha(OUT/'download_manifest.json')==done['manifest_sha256']
    downloads={r['path']:r for r in read(OUT/'download_manifest.json')}
    entries=[];hashes={}
    for row in read(RAW/'selection.json')['scenes']:
        k,pose=RAW/row['scene']/'intrinsics.txt',RAW/row['pose_file']
        assert sha(k)==row['K_sha256'] and sha(pose)==row['pose_sha256']
        hashes.update({str(k):row['K_sha256'],str(pose):row['pose_sha256']})
        frames=[]
        for f in row['frames']:
            path=RAW/row['scene']/'depth'/f['depth'];record=downloads[str(path)]
            assert sha(path)==record['sha256']
            frames.append(dict(index=f['index'],path=str(path),sha256=record['sha256']))
        entries.append(dict(scene=row['scene'],pairs=row['pairs'],intrinsics=str(k),poses=str(pose),frames=frames))
    return dict(entries=entries,raw_hashes=hashes)


def prepare():
    lock=check()
    assert not (OUT/'data_manifest.json').exists()
    raw=raw_lock();preserved(OUT/'raw_data_lock.json',raw)
    collected={c:{} for c in CONDITIONS};ids=[];scenes=[];scales=[];excluded=[]
    pending=[];pending_ids=[];start=time.time()
    def flush():
        if not pending:return
        arrays={k:np.stack([p[k] for p in pending]) for k in ('points','world','clean')}
        outputs=convert(arrays,pending_ids,'cuda')
        for c,cache in outputs.items():
            for k,v in cache.items():collected[c].setdefault(k,[]).append(v)
        pending.clear();pending_ids.clear()
    for entry in raw['entries']:
        reader,poses=scene_reader(entry,raw)
        for q,k in entry['pairs']:
            identity=f"{entry['scene']}:{q}:{k}"
            clouds=[reader(q),reader(k)]
            if any(len(c['points'])<512 for c in clouds):
                excluded.append(dict(id=identity,reason='fewer_than_512_observed_points'));continue
            observed=sample_inputs(clouds,identity,0,count=512,augment=False)
            pending.append(attach_supervision(observed,poses[[q,k]]));pending_ids.append(identity)
            ids.append(identity);scenes.append(entry['scene']);scales.append(float(observed['scale']))
            if len(pending)==16:flush()
        reader.cache_clear()
        print('CONSTRUCTED',entry['scene'],'accepted so far',len(ids),flush=True)
    flush();CACHE.mkdir(parents=True,exist_ok=True)
    records={};built={}
    for condition in CONDITIONS:
        cache={k:np.concatenate(v) for k,v in collected[condition].items()}
        cache.update(ids=np.asarray(ids),scenes=np.asarray(scenes),scales=np.asarray(scales))
        assert len(set(ids))==len(ids) and len(ids)+len(excluded)==lock['scheduled_pairs']
        assert all(np.isfinite(v).all() for k,v in cache.items() if k not in ('ids','scenes'))
        path=CACHE/(condition+'.npz');assert not path.exists()
        np.savez_compressed(path,**cache)
        overlap=cache['overlap'];valid=np.stack((overlap.max(-1)>=.3,overlap.max(-2)>=.3),1)
        by_scene={s:dict(pairs=int((cache['scenes']==s).sum()),valid_queries=int(valid[cache['scenes']==s].sum()),
                        total_queries=int(valid[cache['scenes']==s].size)) for s in lock['planned_scenes']}
        records[condition]=dict(path=str(path),sha256=sha(path),pairs=len(ids),valid_queries=int(valid.sum()),
                                total_queries=int(valid.size),by_scene=by_scene)
        built[condition]=cache
    preserved(OUT/'data_manifest.json',dict(caches=records,excluded=excluded,planned_pairs=960,
        scheduled_pairs=lock['scheduled_pairs'],planned_scenes=lock['planned_scenes'],
        available_scenes=lock['available_scenes'],unavailable=lock['unavailable'],
        construction_seconds=time.time()-start,raw_data_lock_sha256=sha(OUT/'raw_data_lock.json'),
        evaluation_lock_sha256=sha(OUT/'evaluation_lock.json'),completed_unix=time.time()))
    print('DATA_COMPLETE',len(ids),flush=True)


def load(condition):
    manifest=read(OUT/'data_manifest.json')
    assert manifest['evaluation_lock_sha256']==sha(OUT/'evaluation_lock.json')
    item=manifest['caches'][condition]
    assert sha(item['path'])==item['sha256']
    return load_arrays(item['path'])


def infer_all():
    lock=check();configure_gpu()
    assert not (OUT/'execution_identity.json').exists()
    identity=dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],
                  started_unix=time.time(),evaluation_lock_sha256=sha(OUT/'evaluation_lock.json'))
    preserved(OUT/'execution_identity.json',identity)
    completed=[]
    def status(stage,active=None,**extra):
        write(OUT/'status.json',dict(identity,stage=stage,active=active,completed=completed,
              observed_unix=time.time(),goal_complete=False,**extra))
    try:
        status('prepare')
        prepare()
        caches={c:load(c) for c in CONDITIONS}
        for item in lock['models']:
            role,seed=item['role'],item['seed']
            model=make_model(role).cuda().eval()
            model.load_state_dict(torch.load(item['checkpoint_path'],map_location='cpu',weights_only=True)['state_dict'])
            for condition in CONDITIONS:
                status('export',dict(role=role,seed=seed,condition=condition))
                torch.cuda.reset_peak_memory_stats();start=time.time()
                logits=infer(model,caches[condition])
                path=OUT/'predictions'/role/str(seed)/(condition+'.npz')
                assert not path.exists();path.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(path,logits=logits,ids=caches[condition]['ids'])
                item_out=dict(role=role,seed=seed,condition=condition,path=str(path),sha256=sha(path),
                    checkpoint_sha256=item['checkpoint_sha256'],data_manifest_sha256=sha(OUT/'data_manifest.json'),
                    evaluation_lock_sha256=sha(OUT/'evaluation_lock.json'),completed_unix=time.time(),
                    seconds=time.time()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated())
                preserved(path.with_suffix('.json'),item_out);completed.append(item_out)
                print('EXPORTED',role,seed,condition,flush=True)
            del model;torch.cuda.empty_cache()
        check();status('inference_complete',prediction_groups=12)
    except BaseException as error:
        status('failed',error=repr(error));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('freeze','preflight','download','infer_all'))
    action=parser.parse_args().action
    if action not in ('preflight','infer_all'):torch.set_num_threads(4)
    globals()[action]()
