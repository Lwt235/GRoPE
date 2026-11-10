"""One prospective new-place test of the already selected F+rotary models."""
import argparse
import http.client
import os
from pathlib import Path
import time
from urllib.parse import urlsplit

import numpy as np
import torch
from PIL import Image

from experiments.round65_common import ROOT, SEEDS, read, write, sha
from experiments.region_geometry_controls import dependencies
from experiments.region_rotary_confirmation_metadata import PREP, RAW, PLAN, checked as check_cohort
from experiments.region_hrope_confirmation import convert, infer, matching_metrics
from experiments.region_frame_robustness import ROLES, model_factory, digest_state, check as check_development
from experiments.region_frame_transform import reframe, SALT
from experiments.region_hrope_control import configure
from experiments.round60_data import scene_reader, sample_inputs, attach_supervision
from experiments.round5_rgbd_data import read_sun3d_intrinsics
import experiments.region_pose_confirmation as raw_helper

OUT = ROOT/'results/region_rotary_confirmation/attempt1'
CACHE = RAW.parent/'camera_matched_cache'
DEV = ROOT/'results/region_frame_robustness/attempt1'
CONDITIONS = ('base', 'reframed')
CONTROLS = ('distance_mlp', 'fixed_hrope', 'geotransformer_rpe', 'predicted_hrope', 'fixed_isotropic5')
COMPARISONS = [('relative5'+suffix, r+suffix) for suffix in ('', '_reframed') for r in CONTROLS]
ACCESS = ROOT/'results/research_data_usage/region_rotary_confirmation_access_20261110.json'


def preserved(path, value):
    assert not Path(path).exists(), str(path)
    write(path, value)


def freeze():
    cohort = check_cohort(); previous = check_development()
    assert not OUT.exists() and not CACHE.exists()
    assert not list(RAW.rglob('*.png')) and not list(RAW.rglob('*.jpg'))
    metadata = read(PREP/'metadata_accounted.json')
    assert sha(RAW/'selection.json') == metadata['selection_sha256']
    assert sha(RAW/'metadata_manifest.json') == metadata['manifest_sha256']
    assert len(metadata['available_scenes']) >= 2
    sources = dict(cohort['source_sha256'])
    for entry in read(RAW/'metadata_manifest.json'):
        path = RAW/entry['path']; assert sha(path) == entry['sha256']
        sources[str(path)] = entry['sha256']
    conversion = ROOT/'results/region_hrope_confirmation/preparation/old_conversion_check.json'
    check_conversion = read(conversion)
    assert check_conversion['passed']
    assert check_conversion['source_sha256'] == sha(ROOT/'experiments/region_hrope_confirmation.py')
    completion = read(DEV/'completion_checks.json')
    assert completion['passed'] and completion['base_replay_prediction_changes'] == 0
    assert completion['original_checkpoint_identities_verified'] == 18
    models = []
    for old in previous['models']:
        model = dict(old, role=old['base_role'])
        replay = DEV/'evaluation'/old['role']/str(old['seed'])/'base.json'
        row = read(replay)
        assert row['checkpoint_sha256'] == model['checkpoint_sha256'] == sha(model['checkpoint_path'])
        assert row['base_replay']['prediction_changes'] == 0 and row['weights_unchanged']
        sources[str(replay)] = sha(replay)
        models.append(model)
    assert {(m['role'],m['seed']) for m in models} == {(r,s) for r in ROLES for s in SEEDS}
    paths = dependencies([Path(__file__), ROOT/'experiments/region_rotary_confirmation_score.py',
        ROOT/'experiments/region_rotary_confirmation_queue.py', ROOT/'experiments/region_rotary_confirmation_report.py'])
    paths += [PLAN, PREP/'cohort_lock.json', PREP/'scene_selection.json', PREP/'metadata_accounted.json',
        RAW/'selection.json', RAW/'metadata_manifest.json', conversion, DEV/'completion_checks.json',
        DEV/'review.json', DEV/'result.json', DEV/'lock.json']
    sources.update({str(p):sha(p) for p in paths})
    preserved(OUT/'evaluation_lock.json', dict(created_unix=time.time(), source_sha256=sources,
        models=models, roles=ROLES, seeds=list(SEEDS), conditions=CONDITIONS,
        planned_places=cohort['places'], planned_scenes=[r['scene'] for r in read(PREP/'scene_selection.json')['records']],
        available_scenes=metadata['available_scenes'], unavailable=metadata['failures'],
        complete_cohort=metadata['complete_cohort'], planned_pairs=1920, scheduled_pairs=metadata['pairs'],
        new_fits=0, comparisons=COMPARISONS, bootstrap_seeds=list(range(101001,101011)),
        change_bootstrap_seeds=list(range(102001,102007)), interaction_bootstrap_seeds=list(range(103001,103006)),
        bootstrap_replicates=10000, consensus_radius_pair_units=.1, maximum_consensus_refits=5,
        alignment_rmse_threshold_m=.2, frame_salt=SALT, pixels_access_permitted=True,
        primary_comparison_indices=[0,1], robustness_comparison_indices=[5,6],
        old_development_both_primary_passed=False, separate_prospective_decision=True,
        reused_conversion_and18_replays=True, fresh_confirmation=True, goal_complete=False))
    (OUT/'evaluation_lock.sha256').write_text(sha(OUT/'evaluation_lock.json')+'\n')
    print('ROTARY_CONFIRMATION_FROZEN', sha(OUT/'evaluation_lock.json'), flush=True)


def check():
    lock = read(OUT/'evaluation_lock.json')
    assert sha(OUT/'evaluation_lock.json') == (OUT/'evaluation_lock.sha256').read_text().strip()
    for path,digest in lock['source_sha256'].items():
        assert sha(path) == digest, path
    return lock


def download():
    lock = check(); assert lock['pixels_access_permitted']
    assert not (OUT/'download_complete.json').exists()
    if not (OUT/'data_access_started.json').exists():
        access = dict(started_unix=time.time(), scenes=lock['available_scenes'],
            reserved_places=lock['planned_places'], metadata_only_missing=lock['unavailable'],
            evaluation_lock_sha256=sha(OUT/'evaluation_lock.json'),
            scope='All20 selected places reserved, including unavailable metadata; no adaptation or new selection.')
        preserved(OUT/'data_access_started.json', access); preserved(ACCESS, access)
    jobs = {}
    for row in read(RAW/'selection.json')['scenes']:
        for frame in row['frames']:
            path = RAW/row['scene']/'depth'/frame['depth']
            jobs[str(path)] = (row['base_url']+'depth/'+frame['depth'], path)
    rows = []
    state = dict(state='downloading', pid=os.getpid(), total=len(jobs), completed=0,
        errors=[], active=None, started_unix=time.time(), proxy_disabled=True,
        evaluation_lock_sha256=sha(OUT/'evaluation_lock.json'))
    def save():
        state['updated_unix'] = time.time(); write(OUT/'download_progress.json', state)
    save()
    for url,path in jobs.values():
        parsed = urlsplit(url)
        assert parsed.scheme == 'https' and parsed.netloc == 'sun3d.cs.princeton.edu' and parsed.path.endswith('.png')
        path.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(6):
            state['active'] = dict(url=url, attempt=attempt+1); save()
            retry_after = 0
            try:
                if not path.exists():
                    conn = http.client.HTTPSConnection(parsed.netloc, timeout=35)
                    try:
                        conn.request('GET', parsed.path, headers={'User-Agent':'GRoPE-research-data/1.0','Connection':'close'})
                        response = conn.getresponse()
                        value = response.getheader('Retry-After')
                        if value and value.isdigit(): retry_after = int(value)
                        if response.status != 200:
                            response.read(); raise RuntimeError('HTTP '+str(response.status))
                        partial = path.with_suffix('.png.part')
                        with partial.open('wb') as stream:
                            while chunk := response.read(2**20): stream.write(chunk)
                        with Image.open(partial) as image: image.verify()
                        partial.replace(path)
                    finally:
                        conn.close()
                with Image.open(path) as image: image.verify()
                rows.append(dict(path=str(path), url=url, bytes=path.stat().st_size, sha256=sha(path)))
                state.update(state='downloading', completed=len(rows), active=None); save()
                if len(rows)%20 == 0: print('ROTARY_DEPTHS',len(rows),'/',len(jobs),flush=True)
                time.sleep(1); break
            except Exception as error:
                failure = dict(url=url, attempt=attempt+1, error=repr(error), unix=time.time())
                state['errors'].append(failure); print('DEPTH_RETRY',failure,flush=True)
                if attempt == 5:
                    state.update(state='source_unavailable', finished_unix=time.time());save()
                    write(OUT/f'download_failure_{time.time_ns()}.json', dict(failure=failure, retained_files=rows))
                    raise
                delay = max((15,30,60,120,240)[attempt],retry_after)
                state.update(state='backoff', next_attempt_unix=time.time()+delay);save()
                while delay>0:
                    seconds=min(30,delay);time.sleep(seconds);delay-=seconds
    rows.sort(key=lambda r:r['path'])
    preserved(OUT/'download_manifest.json', rows)
    preserved(OUT/'download_complete.json', dict(files=len(rows), bytes=sum(r['bytes'] for r in rows),
        manifest_sha256=sha(OUT/'download_manifest.json'), completed_unix=time.time(), rgb_downloaded=False, proxy_disabled=True))
    state.update(state='download_complete', active=None, finished_unix=time.time());save()
    print('ROTARY_DOWNLOAD_COMPLETE',len(rows),flush=True)


def prepare():
    lock = check(); torch.set_num_threads(4)
    assert not (OUT/'data_manifest.json').exists() and not CACHE.exists()
    raw_helper.OUT,raw_helper.RAW = OUT,RAW
    raw = raw_helper.raw_lock();preserved(OUT/'raw_data_lock.json',raw)
    collected,ids,scenes,scales,excluded = {},[],[],[],[]
    pending,pending_ids = [],[];started=time.time()
    def flush():
        if not pending:return
        batch={k:np.stack([r[k] for r in pending]) for k in pending[0]}
        for key,value in convert(batch,pending_ids).items():collected.setdefault(key,[]).append(value)
        pending.clear();pending_ids.clear()
    for entry in raw['entries']:
        reader,poses=scene_reader(entry,raw)
        intrinsic=read_sun3d_intrinsics(entry['intrinsics']);sizes={}
        for frame in entry['frames']:
            with Image.open(frame['path']) as image:sizes[frame['index']]=(image.height,image.width)
        for q,k in entry['pairs']:
            identity=f"{entry['scene']}:{q}:{k}";clouds=[reader(q),reader(k)]
            if any(len(c['points'])<512 for c in clouds):
                excluded.append(dict(id=identity,reason='fewer_than_512_observed_points'));continue
            observation=sample_inputs(clouds,identity,0,count=512,augment=False)
            item=attach_supervision(observation,poses[[q,k]])
            item.update(camera_intrinsics=np.stack((intrinsic,intrinsic)),image_sizes=np.asarray((sizes[q],sizes[k])))
            pending.append(item);pending_ids.append(identity)
            ids.append(identity);scenes.append(entry['scene']);scales.append(float(observation['scale']))
            if len(pending)==16:flush()
        reader.cache_clear();print('ROTARY_CONSTRUCTED',entry['scene'],len(ids),flush=True)
    flush()
    assert len(set(ids))==len(ids) and len(ids)+len(excluded)==lock['scheduled_pairs']
    assert len(set(scenes))>=2
    cache={k:np.concatenate(v) for k,v in collected.items()}
    cache.update(ids=np.asarray(ids),scenes=np.asarray(scenes),scales=np.asarray(scales))
    assert all(np.isfinite(v).all() for k,v in cache.items() if k not in ('ids','scenes'))
    CACHE.mkdir(parents=True)
    path=CACHE/'base.npz';np.savez_compressed(path,**cache)
    changed,r,t=reframe(cache)
    np.savez(OUT/'frames.npz',ids=cache['ids'],rotation=r,translation=t)
    # The known inverse frame change must recover the same observed centers.
    recovered=np.einsum('bvji,bvnj->bvni',r,changed['centers'].astype(float)-t[:,:,None,:])
    roundtrip=float(np.abs(recovered-cache['centers']).max());assert roundtrip<1e-5
    preserved(OUT/'data_manifest.json',dict(cache=dict(path=str(path),sha256=sha(path),pairs=len(ids)),
        excluded=excluded,planned_pairs=lock['planned_pairs'],scheduled_pairs=lock['scheduled_pairs'],
        by_scene={s:dict(pairs=int((cache['scenes']==s).sum())) for s in lock['planned_scenes']},
        planned_scenes=lock['planned_scenes'],available_scenes=lock['available_scenes'],unavailable=lock['unavailable'],
        construction_seconds=time.time()-started,raw_data_lock_sha256=sha(OUT/'raw_data_lock.json'),
        frames_sha256=sha(OUT/'frames.npz'),frame_center_roundtrip_maximum=roundtrip,
        evaluation_lock_sha256=sha(OUT/'evaluation_lock.json'),completed_unix=time.time(),fresh_confirmation=True))
    print('ROTARY_CONFIRMATION_DATA_READY',len(ids),flush=True)


def load():
    manifest=read(OUT/'data_manifest.json')
    assert manifest['evaluation_lock_sha256']==sha(OUT/'evaluation_lock.json')
    item=manifest['cache'];assert sha(item['path'])==item['sha256']
    with np.load(item['path'],allow_pickle=False) as z:return {k:z[k] for k in z.files}


def export(seed):
    lock=check();configure()
    base=load();changed,r,t=reframe(base)
    manifest=read(OUT/'data_manifest.json');assert sha(OUT/'frames.npz')==manifest['frames_sha256']
    with np.load(OUT/'frames.npz',allow_pickle=False) as z:
        np.testing.assert_array_equal(z['ids'],base['ids'])
        np.testing.assert_array_equal(z['rotation'],r);np.testing.assert_array_equal(z['translation'],t)
    for spec in [m for m in lock['models'] if m['seed']==seed]:
        model=model_factory(spec['role']).cuda().eval()
        checkpoint=torch.load(spec['checkpoint_path'],map_location='cpu',weights_only=True)
        assert checkpoint['seed']==seed and checkpoint['epoch']==spec['selected_epoch']
        model.load_state_dict(checkpoint['state_dict']);before=digest_state(model)
        for condition,cache in (('base',base),('reframed',changed)):
            role=spec['role']+('_reframed' if condition=='reframed' else '')
            path=OUT/'evaluation'/role/str(seed)/'base.npz';assert not path.exists()
            logits=infer(model,cache);assert np.isfinite(logits).all() and digest_state(model)==before
            metrics=matching_metrics(logits,base)
            path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,ids=base['ids'],logits=logits)
            preserved(path.with_suffix('.json'),dict(role=role,base_role=spec['role'],condition=condition,
                seed=seed,path=str(path),sha256=sha(path),checkpoint_sha256=spec['checkpoint_sha256'],
                metrics=metrics,weights_unchanged=True,data_manifest_sha256=sha(OUT/'data_manifest.json'),
                evaluation_lock_sha256=sha(OUT/'evaluation_lock.json')))
            print('ROTARY_CONFIRMATION_EXPORTED',role,seed,flush=True)
        del model;torch.cuda.empty_cache()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=('freeze','download','prepare','export'))
    parser.add_argument('--seed',type=int,choices=SEEDS);args=parser.parse_args()
    if args.action=='export':export(args.seed)
    else:globals()[args.action]()
