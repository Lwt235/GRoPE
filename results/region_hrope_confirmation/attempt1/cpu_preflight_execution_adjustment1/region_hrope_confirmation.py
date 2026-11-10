"""Frozen camera-matched confirmation on the already selected unused places."""
import argparse
import os
from pathlib import Path
import time
import numpy as np
import torch
from PIL import Image

from experiments.round65_common import ROOT, SEEDS, read, write, sha
from experiments.round65_run import configure
from experiments.round58_data import tokenize
from experiments.round58_run import metric_torch
from experiments.round60_data import scene_reader, sample_inputs, attach_supervision
from experiments.round5_rgbd_data import read_sun3d_intrinsics, read_sun3d_extrinsics
from experiments.round68_fixed_region_data import shuffle_indices
from experiments.round68_fixed_region_run import query_values, aggregate
from experiments.region_geometry_controls import dependencies
from experiments.region_hrope_data import camera_fields, load_cache as development_cache, SCENE_LOCK
from experiments.region_pose_confirmation_metadata import PREP as META, RAW, checked as check_cohort
from experiments import region_pose_confirmation as old_draft
from models.region_hrope_control import CameraRegionMatcher, ROLES

BASE = ROOT/'results/region_hrope_confirmation'
PREP = BASE/'preparation'
OUT = BASE/'attempt1'
DEV = ROOT/'results/region_hrope_control/attempt1'
PLAN = ROOT/'docs/region_hrope_confirmation_plan.md'
CACHE = RAW.parent/'camera_matched_cache'
FIELDS = ('features','centers','covariance','world_centers','overlap')
INPUTS = ('features','centers','covariance','permutation','intrinsics','image_positions')
COMPARISONS = [('relative5','predicted_hrope'),('relative5','distance_mlp'),
              ('relative5','geotransformer_rpe'),('relative5','fixed_isotropic5'),
              ('predicted_hrope','distance_mlp'),('predicted_hrope','geotransformer_rpe')]


def preserved(path, value):
    assert not Path(path).exists(), 'Preserve '+str(path)
    write(path, value)


@torch.no_grad()
def convert(batch, identities):
    # Keep the CPU tokenizer used by the fully checked old camera reconstruction.
    base = tokenize({k:torch.as_tensor(batch[k]) for k in ('points','world','clean')})
    output = {k:base[k].numpy() for k in FIELDS}
    intrinsic, xy = camera_fields(batch['source_ids'], base['neighbors'].numpy(),
                                  batch['camera_intrinsics'], batch['image_sizes'])
    output.update(permutation=shuffle_indices(identities), intrinsics=intrinsic, image_positions=xy)
    return output


def old_conversion_check():
    torch.set_num_threads(4)
    old = development_cache('val')
    identities = old['ids'][:2].tolist()
    raw = read(SCENE_LOCK)
    rows = []
    for identity in identities:
        scene, q, k = identity.rsplit(':',2)
        q,k = int(q),int(k)
        entry = next(r for r in raw['entries'] if r['scene']==scene)
        reader,poses = scene_reader(entry,raw)
        observation = sample_inputs([reader(q),reader(k)],identity,0,count=512,augment=False)
        item = attach_supervision(observation,poses[[q,k]])
        intrinsic = read_sun3d_intrinsics(entry['intrinsics'])
        frames = {r['index']:r for r in entry['frames']}
        sizes = []
        for index in (q,k):
            with Image.open(frames[index]['path']) as image:
                sizes.append((image.height,image.width))
        item.update(camera_intrinsics=np.stack((intrinsic,intrinsic)),image_sizes=np.asarray(sizes))
        rows.append(item)
        reader.cache_clear()
    batch = {k:np.stack([r[k] for r in rows]) for k in rows[0]}
    actual = convert(batch,identities)
    errors = {k:float(np.abs(actual[k]-old[k][:2]).max()) for k in actual if k!='permutation'}
    for key in ('overlap','permutation','intrinsics','image_positions'):
        np.testing.assert_array_equal(actual[key],old[key][:2])
    assert max(errors.values()) < 2e-4, errors
    return dict(passed=True,source_sha256=sha(__file__),pairs=2,identities=identities,
        maximum_errors=errors,old_camera_manifest_sha256=sha(DEV.parent/'camera_data/manifest.json'),
        new_scene_pixels_accessed=False,completed_unix=time.time())


def freeze():
    assert not OUT.exists() and not CACHE.exists()
    cohort = check_cohort()
    review = read(DEV/'review.json')
    assert review['scientific_review_complete'] and review['result_sha256']==sha(DEV/'result.json')
    assert sha(DEV/'report.md')==review['report_sha256']
    check_old = read(PREP/'old_conversion_check.json')
    assert check_old['passed'] and check_old['source_sha256']==sha(__file__)
    assert not list(RAW.rglob('*.png')) and not list(RAW.rglob('*.jpg'))
    metadata = read(META/'metadata_accounted.json')
    selection = read(RAW/'selection.json')
    assert sha(RAW/'selection.json')==metadata['selection_sha256']
    assert sha(RAW/'metadata_manifest.json')==metadata['manifest_sha256']
    sources = dict(cohort['source_sha256'])
    for record in read(RAW/'metadata_manifest.json'):
        path = RAW/record['path']
        assert sha(path)==record['sha256']
        sources[str(path)] = record['sha256']
    planned = [r['scene'] for r in read(META/'scene_selection.json')['records']]
    available = []
    for row in selection['scenes']:
        k = read_sun3d_intrinsics(RAW/row['scene']/'intrinsics.txt')
        poses = read_sun3d_extrinsics(RAW/row['pose_file'])
        assert np.isfinite(k).all() and abs(np.linalg.det(k))>1e-8
        assert np.isfinite(poses).all() and all(0<=a<len(poses) and 0<=b<len(poses) for a,b in row['pairs'])
        assert len(row['pairs'])==96
        available.append(row['scene'])
    assert len(planned)==10 and len(available)>=2 and metadata['pairs']==96*len(available)
    models = []
    selected = read(DEV/'selected.json')
    assert selected['all15_selected_before_eval'] and len(selected['records'])==15
    for item in selected['records']:
        folder = DEV/'runs'/item['role']/str(item['seed'])
        assert sha(folder/'best.pt')==item['checkpoint_sha256']
        assert sha(folder/'run.json')==item['run_sha256']
        model = dict(item,checkpoint_path=str(folder/'best.pt'),validation_path=str(folder/'validation.npz'))
        models.append(model)
        for name in ('best.pt','validation.npz','run.json'):
            sources[str(folder/name)] = sha(folder/name)
    assert {(m['role'],m['seed']) for m in models}=={(r,s) for r in ROLES for s in SEEDS}
    source_paths = dependencies([Path(__file__),ROOT/'experiments/region_pose_confirmation.py',
        ROOT/'experiments/region_hrope_confirmation_score.py',
        ROOT/'experiments/region_hrope_confirmation_queue.py',ROOT/'experiments/region_hrope_confirmation_report.py'])
    source_paths += [PLAN,PREP/'old_conversion_check.json',META/'cohort_lock.json',META/'scene_selection.json',
        META/'metadata_accounted.json',RAW/'selection.json',RAW/'metadata_manifest.json',
        DEV/'selected.json',DEV/'review.json',DEV/'result.json']
    sources.update({str(p):sha(p) for p in source_paths})
    preserved(OUT/'evaluation_lock.json',dict(created_unix=time.time(),source_sha256=sources,
        models=models,roles=ROLES,seeds=SEEDS,conditions=['base'],input_fields=INPUTS,
        planned_scenes=planned,available_scenes=available,unavailable=metadata['failures'],
        complete_cohort=metadata['complete_cohort'],planned_pairs=960,scheduled_pairs=metadata['pairs'],new_fits=0,
        comparisons=COMPARISONS,bootstrap_seeds=list(range(96001,96007)),bootstrap_replicates=10000,
        consensus_radius_pair_units=.1,maximum_consensus_refits=5,alignment_rmse_threshold_m=.2,
        old_conversion_check=check_old,pixels_access_permitted=True,fresh_confirmation=True,
        hypothesis='The old camera-matched ordering and remaining F utility transfer to unused places.',
        primary='F-HRoPE; symmetric positive, negative, or uncertain interpretation fixed before new pixels.',
        metadata_cohort_preserved=True,goal_complete=False))
    (OUT/'evaluation_lock.sha256').write_text(sha(OUT/'evaluation_lock.json')+'\n')
    print('HROPE_CONFIRMATION_FROZEN',sha(OUT/'evaluation_lock.json'),flush=True)


def check():
    lock = read(OUT/'evaluation_lock.json')
    assert sha(OUT/'evaluation_lock.json')==(OUT/'evaluation_lock.sha256').read_text().strip()
    for path,digest in lock['source_sha256'].items():
        assert sha(path)==digest,path
    return lock


def configure_gpu():
    configure()
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    torch.cuda.set_per_process_memory_fraction(3*2**30/torch.cuda.get_device_properties(0).total_memory)


@torch.no_grad()
def infer(model, cache):
    model.eval()
    rows = []
    for start in range(0,len(cache['ids']),16):
        values = [torch.as_tensor(cache[k][start:start+16],device='cuda') for k in INPUTS]
        rows.append(model(*values).cpu().numpy())
    return np.concatenate(rows)


def preflight():
    lock = check();configure_gpu()
    assert not (OUT/'preflight.json').exists()
    cache = development_cache('val')
    records = []
    for item in lock['models']:
        model = CameraRegionMatcher(item['role']).cuda().eval()
        state = torch.load(item['checkpoint_path'],map_location='cpu',weights_only=True)
        assert (state['role'],state['seed'],state['epoch'])==(item['role'],item['seed'],item['selected_epoch'])
        model.load_state_dict(state['state_dict'])
        logits = infer(model,cache)
        with np.load(item['validation_path'],allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'],cache['ids'])
            error = float(np.abs(logits-z['logits']).max())
            changes = int((logits.argmax(-1)!=z['logits'].argmax(-1)).sum())
            changes += int((logits.argmax(-2)!=z['logits'].argmax(-2)).sum())
        assert error<2e-4 and changes==0,(item['role'],item['seed'],error,changes)
        records.append(dict(role=item['role'],seed=item['seed'],maximum_logit_error=error,prediction_changes=changes))
        del model;torch.cuda.empty_cache()
    preserved(OUT/'preflight.json',dict(passed=True,records=records,completed_unix=time.time(),
        evaluation_lock_sha256=sha(OUT/'evaluation_lock.json'),CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES']))
    print('HROPE_CONFIRMATION_PREFLIGHT_PASSED',len(records),flush=True)


def bind_download():
    old_draft.OUT = OUT
    old_draft.check = check


def download():
    bind_download()
    old_draft.download()


def prepare():
    lock = check();torch.set_num_threads(4)
    assert not (OUT/'data_manifest.json').exists() and not CACHE.exists()
    bind_download()
    raw = old_draft.raw_lock()
    preserved(OUT/'raw_data_lock.json',raw)
    collected,ids,scenes,scales,excluded = {},[],[],[],[]
    pending,pending_ids = [],[]
    started = time.time()
    def flush():
        if not pending:return
        batch = {k:np.stack([r[k] for r in pending]) for k in pending[0]}
        output = convert(batch,pending_ids)
        for k,v in output.items():collected.setdefault(k,[]).append(v)
        pending.clear();pending_ids.clear()
    for entry in raw['entries']:
        reader,poses = scene_reader(entry,raw)
        intrinsic = read_sun3d_intrinsics(entry['intrinsics'])
        sizes = {}
        for frame in entry['frames']:
            with Image.open(frame['path']) as image:sizes[frame['index']] = (image.height,image.width)
        for q,k in entry['pairs']:
            identity = f"{entry['scene']}:{q}:{k}"
            clouds = [reader(q),reader(k)]
            if any(len(c['points'])<512 for c in clouds):
                excluded.append(dict(id=identity,reason='fewer_than_512_observed_points'));continue
            observation = sample_inputs(clouds,identity,0,count=512,augment=False)
            item = attach_supervision(observation,poses[[q,k]])
            item.update(camera_intrinsics=np.stack((intrinsic,intrinsic)),image_sizes=np.asarray((sizes[q],sizes[k])))
            pending.append(item);pending_ids.append(identity)
            ids.append(identity);scenes.append(entry['scene']);scales.append(float(observation['scale']))
            if len(pending)==16:flush()
        reader.cache_clear()
        print('CAMERA_CONFIRMATION_CONSTRUCTED',entry['scene'],len(ids),flush=True)
    flush()
    assert len(set(ids))==len(ids) and len(ids)+len(excluded)==lock['scheduled_pairs']
    assert len(set(scenes))>=2
    cache = {k:np.concatenate(v) for k,v in collected.items()}
    cache.update(ids=np.asarray(ids),scenes=np.asarray(scenes),scales=np.asarray(scales))
    assert all(np.isfinite(v).all() for k,v in cache.items() if k not in ('ids','scenes'))
    CACHE.mkdir(parents=True)
    path = CACHE/'base.npz';np.savez_compressed(path,**cache)
    overlap = cache['overlap']
    valid = np.stack((overlap.max(-1)>=.3,overlap.max(-2)>=.3),1)
    by_scene = {s:dict(pairs=int((cache['scenes']==s).sum()),valid_queries=int(valid[cache['scenes']==s].sum()),
        total_queries=int(valid[cache['scenes']==s].size)) for s in lock['planned_scenes']}
    preserved(OUT/'data_manifest.json',dict(cache=dict(path=str(path),sha256=sha(path),pairs=len(ids)),
        excluded=excluded,planned_pairs=960,scheduled_pairs=lock['scheduled_pairs'],by_scene=by_scene,
        planned_scenes=lock['planned_scenes'],available_scenes=lock['available_scenes'],unavailable=lock['unavailable'],
        construction_seconds=time.time()-started,raw_data_lock_sha256=sha(OUT/'raw_data_lock.json'),
        evaluation_lock_sha256=sha(OUT/'evaluation_lock.json'),completed_unix=time.time(),fresh_confirmation=True))
    print('HROPE_CONFIRMATION_DATA_READY',len(ids),flush=True)


def load():
    manifest = read(OUT/'data_manifest.json')
    assert manifest['evaluation_lock_sha256']==sha(OUT/'evaluation_lock.json')
    item = manifest['cache'];assert sha(item['path'])==item['sha256']
    with np.load(item['path'],allow_pickle=False) as z:return {k:z[k] for k in z.files}


def matching_metrics(logits, cache):
    values = query_values(logits,cache)
    by_scene = {s:aggregate(values,(cache['scenes']==s)[:,None,None]) for s in sorted(set(cache['scenes'].tolist()))}
    selected = [r for r in by_scene.values() if r['valid_queries']]
    macro = {k:float(np.mean([r[k] for r in selected])) if selected else None
             for k in ('region_recall','center_pck','soft_epe','ce')}
    micro = aggregate(values)
    independent = metric_torch(logits,cache['overlap'],cache['world_centers'])
    error = max(abs(micro[k]-independent[k]) for k in ('region_recall','center_pck','soft_epe','ce'))
    assert error<1e-10
    return dict(micro=micro,macro=macro,by_scene=by_scene,independent_metric_error=error)


def infer_all():
    lock = check();configure_gpu()
    assert not (OUT/'inference_status.json').exists()
    cache = load()
    state = dict(state='inference',pid=os.getpid(),started_unix=time.time(),completed=[],active=None,
        evaluation_lock_sha256=sha(OUT/'evaluation_lock.json'),CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'])
    def save():
        state['updated_unix']=time.time();write(OUT/'inference_status.json',state)
    save()
    try:
        for item in lock['models']:
            role,seed = item['role'],item['seed']
            state['active']=dict(role=role,seed=seed);save()
            model = CameraRegionMatcher(role).cuda().eval()
            model.load_state_dict(torch.load(item['checkpoint_path'],map_location='cpu',weights_only=True)['state_dict'])
            logits = infer(model,cache)
            assert np.isfinite(logits).all()
            path = OUT/'predictions'/role/str(seed)/'base.npz'
            assert not path.exists();path.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(path,logits=logits,ids=cache['ids'])
            receipt = dict(role=role,seed=seed,path=str(path),sha256=sha(path),pairs=len(cache['ids']),
                checkpoint_sha256=item['checkpoint_sha256'],metrics=matching_metrics(logits,cache),
                data_manifest_sha256=sha(OUT/'data_manifest.json'),evaluation_lock_sha256=sha(OUT/'evaluation_lock.json'))
            preserved(path.with_suffix('.json'),receipt)
            state['completed'].append(receipt);save()
            del model;torch.cuda.empty_cache()
            print('HROPE_CONFIRMATION_EXPORTED',role,seed,flush=True)
        state.update(state='inference_complete',active=None,finished_unix=time.time());save()
    except BaseException as error:
        state.update(state='failed',error=repr(error));save();raise


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=('old_check','freeze','preflight','download','prepare','infer_all'))
    args=parser.parse_args()
    if args.action=='old_check':
        preserved(PREP/'old_conversion_check.json',old_conversion_check())
        print('OLD_CAMERA_CONVERSION_PASSED',flush=True)
    else:globals()[args.action]()
