"""Fixed unused-scene rigid scoring with independent truth and paired analysis."""
import argparse
import os
import time
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from experiments.round65_common import ROOT, SEEDS, read, write, sha
from experiments.region_hrope_confirmation import OUT, DEV, check as check_evaluation
from experiments.region_pose_solver import estimate, recover_truth, score_transform
from experiments.region_pose_utility import arrays, configure_cuda
from experiments.region_pose_analysis import paired_statistics, finite_or_none, load_estimate


def bind_inputs():
    lock=check_evaluation()
    assert not (OUT/'pose_inputs.json').exists()
    state=read(OUT/'inference_status.json')
    assert state['state']=='inference_complete' and len(state['completed'])==15
    data=read(OUT/'data_manifest.json')
    cache=data['cache']
    assert sha(cache['path'])==cache['sha256']
    expected={(m['role'],m['seed']):m['checkpoint_sha256'] for m in lock['models']}
    predictions=[]
    for row in state['completed']:
        assert row['checkpoint_sha256']==expected[row['role'],row['seed']]
        assert sha(row['path'])==row['sha256']
        predictions.append({k:row[k] for k in ('role','seed','path','sha256','checkpoint_sha256')})
    assert len({(r['role'],r['seed']) for r in predictions})==15
    write(OUT/'pose_inputs.json',dict(created_unix=time.time(),cache=cache,predictions=predictions,
        evaluation_lock_sha256=sha(OUT/'evaluation_lock.json'),selected_sha256=sha(DEV/'selected.json'),
        data_manifest_sha256=sha(OUT/'data_manifest.json'),inference_status_sha256=sha(OUT/'inference_status.json'),
        fresh_confirmation=True))
    print('CONFIRMATION_SCORING_INPUTS_BOUND',flush=True)


def check():
    lock=check_evaluation()
    bound=read(OUT/'pose_inputs.json')
    assert bound['evaluation_lock_sha256']==sha(OUT/'evaluation_lock.json')
    assert bound['data_manifest_sha256']==sha(OUT/'data_manifest.json')
    assert bound['inference_status_sha256']==sha(OUT/'inference_status.json')
    return dict(lock,cache=bound['cache'])


def diagnostics():
    lock=check();configure_cuda()
    folder=OUT/'pose/0'
    assert not folder.exists()
    data=arrays(lock['cache']['path'],('ids','scenes','centers','world_centers','scales'))
    centers=data['centers'].astype(np.float64)
    count=len(centers)
    truth=[recover_truth(centers[i],data['world_centers'][i]) for i in range(count)]
    assert max(t[2] for t in truth)<2e-5
    state=dict(state='estimating',seed=0,pid=os.getpid(),completed=[],active=None,started_unix=time.time(),
        pose_inputs_sha256=sha(OUT/'pose_inputs.json'),CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'])
    def save():
        state['updated_unix']=time.time();write(folder/'status.json',state)
    save()
    gpu=torch.as_tensor(centers,device='cuda')
    for role in ('identity','gt_nearest'):
        records=[];started=time.time();state['active']=dict(role=role,completed_pairs=0);save()
        for index in range(count):
            if role=='identity':
                result=dict(rotation=np.eye(3),translation=np.zeros(3),valid=True,consensus_count=0,
                    truncated_squared_error=None,hypotheses_total=0,distinct_target_hypotheses=0,
                    nondegenerate_hypotheses=0,refits=0)
            else:
                mapped=centers[index,0]@truth[index][0].T+truth[index][1]
                picked=np.square(mapped[:,None]-centers[index,None,1]).sum(-1).argmin(-1)
                result=estimate(gpu[index,0],gpu[index,1],torch.as_tensor(picked,device='cuda',dtype=torch.long),
                    radius=lock['consensus_radius_pair_units'],max_refits=lock['maximum_consensus_refits'])
            score=score_transform(centers[index,0],result['rotation'],result['translation'],truth[index][0],
                truth[index][1],data['scales'][index],valid=result['valid'])
            records.append(dict(**result,**score))
            if (index+1)%32==0:
                state['active']['completed_pairs']=index+1;save()
        path=folder/(role+'.npz')
        payload={k:np.stack([r[k] for r in records]) for k in records[0] if k!='truncated_squared_error'}
        payload['truncated_squared_error']=np.asarray([np.nan if r['truncated_squared_error'] is None else r['truncated_squared_error'] for r in records])
        np.savez(path,ids=data['ids'],scenes=data['scenes'],**payload)
        state['completed'].append(dict(role=role,seed=0,path=str(path),sha256=sha(path),pairs=count,
            seconds=time.time()-started,input_prediction=None,
            interpretation='undeployable diagnostic' if role=='gt_nearest' else 'identity baseline'))
        save();print('CONFIRMATION_DIAGNOSTIC_COMPLETE',role,flush=True)
    state.update(state='estimates_complete',active=None,finished_unix=time.time());save()


def run(seed):
    lock = check()
    bound = read(OUT/'pose_inputs.json')
    assert bound['evaluation_lock_sha256'] == sha(OUT/'evaluation_lock.json')
    assert bound['selected_sha256'] == sha(DEV/'selected.json')
    assert seed in SEEDS
    folder = OUT/'pose'/str(seed)
    assert not folder.exists(), 'Inspect previous seed worker before resuming.'
    configure_cuda()
    assert sha(lock['cache']['path']) == lock['cache']['sha256']
    data = arrays(lock['cache']['path'], ('ids', 'scenes', 'centers', 'world_centers', 'scales'))
    pair_count = lock['cache']['pairs']
    assert len(data['ids']) == pair_count and len(np.unique(data['scenes'])) >= 2
    centers = data['centers'].astype(np.float64)
    truth = [recover_truth(centers[i], data['world_centers'][i]) for i in range(pair_count)]
    assert max(r[2] for r in truth) < 2e-5
    gpu_centers = torch.as_tensor(centers, device='cuda')
    jobs = [p for p in bound['predictions'] if p['seed'] == seed]
    assert len(jobs) == 5
    state = dict(state='estimating', pid=os.getpid(), seed=seed, started_unix=time.time(),
        pose_inputs_sha256=sha(OUT/'pose_inputs.json'), completed=[], active=None,
        CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'], goal_complete=False)

    def save():
        state['updated_unix'] = time.time()
        write(folder/'status.json', state)

    save()
    try:
        for job in jobs:
            assert sha(job['path']) == job['sha256']
            prediction = arrays(job['path'], ('ids', 'logits'))
            np.testing.assert_array_equal(prediction['ids'], data['ids'])
            assert prediction['logits'].shape == (pair_count, 64, 64) and np.isfinite(prediction['logits']).all()
            indices = prediction['logits'].argmax(-1)
            records, started = [], time.time()
            state['active'] = dict(role=job['role'], completed_pairs=0)
            save()
            for index in range(pair_count):
                result = estimate(gpu_centers[index, 0], gpu_centers[index, 1],
                    torch.as_tensor(indices[index], device='cuda', dtype=torch.long),
                    radius=lock['consensus_radius_pair_units'], max_refits=lock['maximum_consensus_refits'])
                score = score_transform(centers[index, 0], result['rotation'], result['translation'],
                    truth[index][0], truth[index][1], data['scales'][index], valid=result['valid'])
                records.append(dict(**result, **score))
                if (index+1) % 32 == 0:
                    state['active']['completed_pairs'] = index+1
                    save()
            path = folder/(job['role']+'.npz')
            assert not path.exists()
            payload = {k: np.stack([r[k] for r in records]) for k in records[0] if k != 'truncated_squared_error'}
            payload['truncated_squared_error'] = np.asarray([
                np.nan if r['truncated_squared_error'] is None else r['truncated_squared_error'] for r in records])
            np.savez(path, ids=data['ids'], scenes=data['scenes'], **payload)
            receipt = dict(role=job['role'], seed=seed, path=str(path), sha256=sha(path),
                pairs=pair_count, seconds=time.time()-started, input_prediction=job,
                interpretation='camera-information-matched learned regional PE')
            state['completed'].append(receipt)
            save()
            print('HROPE_POSE_COMPLETE', job['role'], seed, receipt['seconds'], flush=True)
        state.update(state='estimates_complete', active=None, finished_unix=time.time(),
                     peak_allocated_bytes=torch.cuda.max_memory_allocated())
        save()
    except BaseException as error:
        state.update(state='failed', error_type=type(error).__name__, reason=str(error))
        save()
        raise

def summarize():
    lock = check()
    assert not (OUT/'result.json').exists()
    diagnostic = read(OUT/'pose/0/status.json')
    assert diagnostic['state']=='estimates_complete' and len(diagnostic['completed'])==2
    assert diagnostic['pose_inputs_sha256']==sha(OUT/'pose_inputs.json')
    receipts = list(diagnostic['completed'])
    for seed in SEEDS:
        state = read(OUT/'pose'/str(seed)/'status.json')
        assert state['state'] == 'estimates_complete' and len(state['completed']) == 5
        assert state['pose_inputs_sha256'] == sha(OUT/'pose_inputs.json')
        receipts.extend(state['completed'])
    assert len(receipts) == 17
    data = arrays(lock['cache']['path'], ('ids', 'scenes', 'centers', 'world_centers', 'scales'))
    pair_count = len(data['ids'])
    assert pair_count==lock['cache']['pairs']
    centers, world = data['centers'].astype(np.float64), data['world_centers'].astype(np.float64)
    scene_names = sorted(set(data['scenes'].tolist()))
    # Independent truth composition and score arithmetic, reused from the old review.
    gt = []
    for index in range(pair_count):
        transforms = []
        for view in range(2):
            a, b = centers[index, view], world[index, view]
            rotation, _ = Rotation.align_vectors(b-b.mean(0), a-a.mean(0))
            matrix = np.eye(4)
            matrix[:3, :3] = rotation.as_matrix()
            matrix[:3, 3] = b.mean(0)-rotation.apply(a.mean(0))
            transforms.append(matrix)
        gt.append(np.linalg.solve(transforms[1], transforms[0]))
    gt = np.stack(gt)
    records, accuracy = [], {}
    errors = dict(rmse_m=0., rotation_error_deg=0., centered_translation_error_m=0.)
    count_differences = 0
    for receipt in receipts:
        assert sha(receipt['path']) == receipt['sha256']
        row = load_estimate(receipt['path'])
        np.testing.assert_array_equal(row['ids'], data['ids'])
        np.testing.assert_array_equal(row['scenes'], data['scenes'])
        if receipt['input_prediction']:
            prediction = receipt['input_prediction']
            assert sha(prediction['path']) == prediction['sha256']
            saved = arrays(prediction['path'], ('ids', 'logits'))
            np.testing.assert_array_equal(saved['ids'], data['ids'])
            picked = saved['logits'].argmax(-1)
        elif receipt['role'] == 'gt_nearest':
            mapped = np.einsum('bij,bnj->bni', gt[:, :3, :3], centers[:, 0])+gt[:, None, :3, 3]
            picked = np.square(mapped[:, :, None]-centers[:, None, 1]).sum(-1).argmin(-1)
        else:
            picked = None
        for index in range(pair_count):
            if not row['valid'][index]:
                assert not row['success'][index] and np.isinf(row['rmse_m'][index])
                continue
            matrix = np.eye(4)
            matrix[:3, :3], matrix[:3, 3] = row['rotation'][index], row['translation'][index]
            np.testing.assert_allclose(matrix[:3, :3].T@matrix[:3, :3], np.eye(3), atol=1e-8)
            assert abs(np.linalg.det(matrix[:3, :3])-1) < 1e-8
            points = np.column_stack((centers[index, 0], np.ones(64)))
            residual = points@(matrix-gt[index]).T
            rmse = float(np.linalg.norm(residual[:, :3], 'fro')/8*data['scales'][index])
            angle = float(np.degrees(Rotation.from_matrix(matrix[:3, :3]@gt[index, :3, :3].T).magnitude()))
            translation = float(np.linalg.norm((matrix-gt[index])[:3, 3])*data['scales'][index])
            for key, value in [('rmse_m', rmse), ('rotation_error_deg', angle), ('centered_translation_error_m', translation)]:
                errors[key] = max(errors[key], abs(value-row[key][index]))
            assert bool(rmse < .2) == row['success'][index]
            if picked is not None:
                transformed = (points@matrix.T)[:, :3]
                count = int((np.linalg.norm(transformed-centers[index, 1, picked[index]], axis=1) <= .1).sum())
                count_differences += int(count != row['consensus_count'][index])
        grouped = [float(row['success'][data['scenes'] == s].mean()) for s in scene_names]
        accuracy[receipt['role'], receipt['seed']] = np.asarray(grouped)
        tails = {k: dict(median=finite_or_none(np.median(row[k])),
                        p90=finite_or_none(np.quantile(row[k], .9, method='higher'))) for k in errors}
        records.append(dict(role=receipt['role'], seed=receipt['seed'], pairs=pair_count,
            successful_pairs=int(row['success'].sum()), invalid_estimates=int((~row['valid']).sum()),
            macro_success_rate=float(np.mean(grouped)), micro_success_rate=float(row['success'].mean()),
            by_scene=dict(zip(scene_names, grouped)), all_pair_errors=tails,
            null_error_quantile_means_infinite_failed_estimates=True, receipt=receipt))
    assert errors['rmse_m'] < 1e-6 and errors['rotation_error_deg'] < 1e-5
    assert errors['centered_translation_error_m'] < 1e-6
    assert count_differences == 0
    comparisons = []
    for (left, right), seed in zip(lock['comparisons'], lock['bootstrap_seeds']):
        stats = paired_statistics(np.stack([accuracy[left, s] for s in SEEDS]),
                                  np.stack([accuracy[right, s] for s in SEEDS]), seed=seed)
        signal = stats['mean_gain_pp'] >= 1 and min(stats['paired_seed_gains_pp']) > 0 and stats['intervals_95_pp']['crossed'][0] > 0
        negative = stats['mean_gain_pp'] <= -1 and max(stats['paired_seed_gains_pp']) < 0 and stats['intervals_95_pp']['crossed'][1] < 0
        comparisons.append(dict(left=left, right=right, **stats, numerical_signal=signal,
            negative_signal=negative, direction='left_better' if signal else 'right_better' if negative else 'uncertain'))
    write(OUT/'result.json', dict(completed_unix=time.time(), evaluation_lock_sha256=sha(OUT/'evaluation_lock.json'),
        pose_inputs_sha256=sha(OUT/'pose_inputs.json'),
        scenes=scene_names, records=records, comparisons=comparisons,
        primary_numerical_signal=comparisons[0]['numerical_signal'], primary_direction=comparisons[0]['direction'],
        data_manifest_sha256=sha(OUT/'data_manifest.json'),
        scientific_review_pending=True, fresh_confirmation=True, goal_complete=False))
    write(OUT/'independent_checks.json', dict(completed_unix=time.time(), passed=True,
        result_sha256=sha(OUT/'result.json'), score_maximum_errors=errors,
        primary_success_decisions_changed=0, consensus_count_disagreements=count_differences,
        records=17, pairs_per_record=pair_count, source_sha256=sha(__file__),
        independent_ground_truth='SciPy alignment and homogeneous inverse composition',
        independent_uncertainty='Resample each absolute method accuracy before subtracting.',
        scientific_review_pending=True, goal_complete=False))
    print('HROPE_POSE_RESULT_READY', sha(OUT/'result.json'), flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=('bind_inputs','run','diagnostics','summarize'))
    parser.add_argument('--seed',type=int,choices=SEEDS)
    args=parser.parse_args()
    if args.action=='run':run(args.seed)
    else:globals()[args.action]()
