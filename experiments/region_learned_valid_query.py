"""Truth-valid query pruning of frozen predictions; diagnostic, not deployable."""
import argparse
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from experiments.round65_common import ROOT, SEEDS, read, write, sha
from experiments.region_pose_utility import arrays, configure_cuda
from experiments.region_pose_solver import score_transform
from experiments.region_pose_analysis import paired_statistics, finite_or_none
from experiments.region_supervision_oracle import load_inputs
from experiments.region_supervision_valid_oracle import estimate_subset

OUT = ROOT / 'results/region_learned_valid_query/diagnostic1'
PARENT = ROOT / 'results/region_rotary_relation/attempt1'
ORACLE = ROOT / 'results/region_supervision_oracle/diagnostic1/valid_query_control'
PLAN = ROOT / 'docs/region_learned_valid_query_plan.md'
ROLES = ('relative5', 'distance_mlp', 'geotransformer_rpe', 'fixed_isotropic5',
         'predicted_hrope', 'fixed_hrope')
ORACLE_ROLES = ('valid_gt_nearest', 'valid_overlap_nearest_tie', 'valid_overlap_first_tie')
ERRORS = ('rmse_m', 'rotation_error_deg', 'centered_translation_error_m')


def freeze():
    assert not OUT.exists(), 'Preserve and inspect any existing diagnostic.'
    assert read(PARENT / 'review.json')['scientific_review_complete']
    assert read(ORACLE / 'review.json')['scientific_review_complete']
    parent = read(PARENT / 'result.json')
    original = read(ORACLE / 'lock.json')
    receipts = [next(r['receipt'] for r in parent['records']
        if (r['role'], r['seed']) == (role, seed)) for role in ROLES for seed in SEEDS]
    references = read(ORACLE / 'execution_status.json')['completed']
    assert [r['role'] for r in references] == list(ORACLE_ROLES)
    for receipt in [*receipts, *references]:
        assert sha(receipt['path']) == receipt['sha256']
        if receipt.get('input_prediction'):
            p = receipt['input_prediction']
            assert sha(p['path']) == p['sha256']
    sources = [Path(__file__), PLAN, PARENT / 'result.json', PARENT / 'review.json',
        ORACLE / 'result.json', ORACLE / 'review.json', ORACLE / 'lock.json',
        ORACLE / 'correspondences.npz', ORACLE / 'preparation.json',
        ROOT / 'experiments/region_supervision_valid_oracle.py',
        ROOT / 'experiments/region_supervision_oracle.py', ROOT / 'experiments/region_pose_solver.py',
        ROOT / 'experiments/region_pose_utility.py', ROOT / 'experiments/region_pose_analysis.py',
        ROOT / 'experiments/round65_common.py']
    lock = dict(created_unix=time.time(), source_sha256={str(p):sha(p) for p in sources},
        cache=original['cache'], truth=original['truth'], receipts=receipts, references=references,
        roles=list(ROLES), seeds=list(SEEDS), oracle_roles=list(ORACLE_ROLES),
        pruning_bootstrap_seeds=list(range(99301,99307)),
        relative_comparisons=['distance_mlp','fixed_hrope','geotransformer_rpe','predicted_hrope','fixed_isotropic5'],
        relative_bootstrap_seeds=list(range(99311,99316)),
        oracle_bootstrap_seeds=list(range(99321,99324)), bootstrap_replicates=10000,
        practical_pruning_gain_pp=5., new_fits=0, new_inference=0, new_pose_groups=18,
        deployable=False, fresh_confirmation=False, goal_complete=False)
    for entry in (lock['cache'], lock['truth']):
        assert sha(entry['path']) == entry['sha256']
    data, _ = load_inputs(lock)
    valid = data['overlap'].max(-1) >= .3
    previous = arrays(ORACLE / 'correspondences.npz', ('ids','valid_source_queries'))
    np.testing.assert_array_equal(data['ids'], previous['ids'])
    np.testing.assert_array_equal(valid, previous['valid_source_queries'])
    independent = np.asarray([[any(float(x)>=.3 for x in row) for row in pair]
        for pair in data['overlap']], dtype=bool)
    np.testing.assert_array_equal(valid, independent)
    payload = {}
    for receipt in receipts:
        p = receipt['input_prediction']
        pred = arrays(p['path'], ('ids','logits'))
        np.testing.assert_array_equal(data['ids'], pred['ids'])
        assert pred['logits'].shape == (480,64,64) and np.isfinite(pred['logits']).all()
        picked = pred['logits'].argmax(-1)
        # Verify the first-max tie convention on representative raw rows independently.
        for b in range(4):
            for q in range(64):
                values = pred['logits'][b,q].tolist()
                assert int(picked[b,q]) == values.index(max(values))
        payload[f"{receipt['role']}_{receipt['seed']}"] = picked
    write(OUT / 'lock.json', lock)
    (OUT / 'lock.sha256').write_text(sha(OUT / 'lock.json')+'\n')
    np.savez(OUT / 'correspondences.npz', ids=data['ids'], scenes=data['scenes'],
             valid_source_queries=valid, **payload)
    write(OUT / 'preparation.json', dict(completed_unix=time.time(),
        lock_sha256=sha(OUT / 'lock.json'), correspondences_sha256=sha(OUT / 'correspondences.npz'),
        total_source_queries=int(valid.size), valid_source_queries=int(valid.sum()),
        pairs_with_fewer_than3_queries=int((valid.sum(-1)<3).sum()),
        independent_mask_exact=True, argmax_tie_rule_checked=True,
        predictions_and_target_candidates_unchanged=True))
    print('LEARNED_VALID_QUERY_FROZEN', sha(OUT / 'lock.json'), flush=True)


def check():
    lock = read(OUT / 'lock.json')
    assert sha(OUT / 'lock.json') == (OUT / 'lock.sha256').read_text().strip()
    for path,digest in lock['source_sha256'].items():
        assert sha(path)==digest,path
    for entry in (lock['cache'],lock['truth']):
        assert sha(entry['path'])==entry['sha256']
    prep=read(OUT / 'preparation.json')
    assert prep['lock_sha256']==sha(OUT / 'lock.json')
    assert prep['correspondences_sha256']==sha(OUT / 'correspondences.npz')
    return lock


def run():
    lock=check()
    assert not (OUT / 'execution_status.json').exists(), 'Inspect a previous worker before recovery.'
    configure_cuda()
    data,truth=load_inputs(lock)
    correspondence=arrays(OUT / 'correspondences.npz', ('ids','valid_source_queries',
        *(f'{role}_{seed}' for role in ROLES for seed in SEEDS)))
    np.testing.assert_array_equal(data['ids'],correspondence['ids'])
    mask=correspondence['valid_source_queries']
    centers=torch.as_tensor(data['centers'],device='cuda')
    state=dict(state='preflight',pid=os.getpid(),started_unix=time.time(),active=None,completed=[],
        lock_sha256=sha(OUT / 'lock.json'),CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],goal_complete=False)
    def save():
        state['updated_unix']=time.time();write(OUT / 'execution_status.json',state)
    save()
    try:
        old_indices=arrays(ORACLE / 'correspondences.npz', ('valid_overlap_nearest_tie',))
        reference=next(r for r in lock['references'] if r['role']=='valid_overlap_nearest_tie')
        assert sha(reference['path'])==reference['sha256']
        old=arrays(reference['path'], ('rotation','translation','valid','consensus_count'))
        maximum=0.
        for b in range(2):
            result=estimate_subset(centers[b,0],centers[b,1],
                torch.as_tensor(old_indices['valid_overlap_nearest_tie'][b],device='cuda',dtype=torch.long))
            assert result['valid']==old['valid'][b] and result['consensus_count']==old['consensus_count'][b]
            for key in ('rotation','translation'):
                maximum=max(maximum,float(np.max(np.abs(result[key]-old[key][b]))))
        assert maximum<1e-8
        write(OUT / 'gpu_replay.json', dict(passed=True,maximum_transform_error=maximum,pairs=2,
            CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],device=torch.cuda.get_device_name(0)))
        state['state']='estimating'
        for receipt in lock['receipts']:
            role,seed=receipt['role'],receipt['seed']
            indices=np.where(mask,correspondence[f'{role}_{seed}'],-1)
            started=time.time();records=[]
            state['active']=dict(role=role,seed=seed,completed_pairs=0);save()
            for b in range(480):
                result=estimate_subset(centers[b,0],centers[b,1],
                    torch.as_tensor(indices[b],device='cuda',dtype=torch.long))
                m=int(mask[b].sum())
                assert result['hypotheses_total']==(math.comb(m,3) if m>=3 else 0)
                score=score_transform(data['centers'][b,0],result['rotation'],result['translation'],
                    truth['rotation'][b],truth['translation'][b],data['scales'][b],valid=result['valid'])
                records.append(dict(**result,**score))
                if (b+1)%32==0:
                    state['active']['completed_pairs']=b+1;save()
            payload={k:np.stack([r[k] for r in records]) for k in records[0] if k!='truncated_squared_error'}
            payload['truncated_squared_error']=np.asarray([np.nan if r['truncated_squared_error'] is None
                else r['truncated_squared_error'] for r in records])
            path=OUT / f'{role}_{seed}.npz';assert not path.exists()
            np.savez(path,ids=data['ids'],scenes=data['scenes'],**payload)
            state['completed'].append(dict(role=role,seed=seed,path=str(path),sha256=sha(path),
                seconds=time.time()-started,pairs=480,input_prediction=receipt['input_prediction']))
            save();print('LEARNED_VALID_QUERY_ESTIMATED',role,seed,time.time()-started,flush=True)
        state.update(state='estimates_complete',active=None,finished_unix=time.time(),
            peak_allocated_bytes=torch.cuda.max_memory_allocated());save()
    except BaseException as error:
        state.update(state='failed',error_type=type(error).__name__,reason=str(error));save();raise


def summarize():
    lock=check()
    state=read(OUT / 'execution_status.json')
    assert state['state']=='estimates_complete' and len(state['completed'])==18
    assert not (OUT / 'result.json').exists()
    data,_=load_inputs(lock)
    correspondence=arrays(OUT / 'correspondences.npz', ('valid_source_queries',
        *(f'{role}_{seed}' for role in ROLES for seed in SEEDS)))
    mask=correspondence['valid_source_queries'];counts=mask.sum(-1)
    scenes=sorted(set(data['scenes'].tolist()))
    def scene_accuracy(success):
        return np.asarray([success[data['scenes']==s].mean() for s in scenes])
    truth=[]
    for b in range(480):
        transforms=[]
        for view in range(2):
            a=data['centers'][b,view];w=data['world_centers'][b,view].astype(np.float64)
            rotation,_=Rotation.align_vectors(w-w.mean(0),a-a.mean(0))
            matrix=np.eye(4);matrix[:3,:3]=rotation.as_matrix()
            matrix[:3,3]=w.mean(0)-rotation.apply(a.mean(0));transforms.append(matrix)
        truth.append(np.linalg.solve(transforms[1],transforms[0]))
    truth=np.stack(truth)
    old_rows={};old_accuracy={};new_accuracy={};oracle_accuracy={};records=[]
    for receipt in lock['receipts']:
        assert sha(receipt['path'])==receipt['sha256']
        row=arrays(receipt['path'], ('ids','scenes','success'))
        np.testing.assert_array_equal(row['ids'],data['ids']);np.testing.assert_array_equal(row['scenes'],data['scenes'])
        key=(receipt['role'],receipt['seed']);old_rows[key]=row;old_accuracy[key]=scene_accuracy(row['success'])
    for receipt in lock['references']:
        assert sha(receipt['path'])==receipt['sha256']
        row=arrays(receipt['path'], ('ids','scenes','success'))
        np.testing.assert_array_equal(row['ids'],data['ids']);np.testing.assert_array_equal(row['scenes'],data['scenes'])
        oracle_accuracy[receipt['role']]=scene_accuracy(row['success'])
    errors={key:0. for key in ERRORS};consensus_disagreements=0
    for receipt in state['completed']:
        assert sha(receipt['path'])==receipt['sha256']
        role,seed=receipt['role'],receipt['seed'];key=(role,seed)
        row=arrays(receipt['path'], ('ids','scenes','rotation','translation','valid','success',
            *ERRORS,'consensus_count','hypotheses_total'))
        np.testing.assert_array_equal(row['ids'],data['ids']);np.testing.assert_array_equal(row['scenes'],data['scenes'])
        prediction=receipt['input_prediction'];assert sha(prediction['path'])==prediction['sha256']
        raw=arrays(prediction['path'], ('ids','logits'))
        np.testing.assert_array_equal(raw['ids'],data['ids'])
        np.testing.assert_array_equal(raw['logits'].argmax(-1),correspondence[f'{role}_{seed}'])
        for b in range(480):
            m=int(counts[b]);assert int(row['hypotheses_total'][b])==(math.comb(m,3) if m>=3 else 0)
            if not row['valid'][b]:
                assert not row['success'][b] and all(np.isinf(row[e][b]) for e in ERRORS)
                continue
            assert m>=3
            matrix=np.eye(4);matrix[:3,:3]=row['rotation'][b];matrix[:3,3]=row['translation'][b]
            np.testing.assert_allclose(matrix[:3,:3].T@matrix[:3,:3],np.eye(3),atol=1e-8)
            assert abs(np.linalg.det(matrix[:3,:3])-1)<1e-8
            points=np.column_stack((data['centers'][b,0],np.ones(64)))
            residual=points@(matrix-truth[b]).T
            values=(float(np.linalg.norm(residual[:,:3],'fro')/8*data['scales'][b]),
                float(np.degrees(Rotation.from_matrix(matrix[:3,:3]@truth[b,:3,:3].T).magnitude())),
                float(np.linalg.norm((matrix-truth[b])[:3,3])*data['scales'][b]))
            for name,value in zip(ERRORS,values):errors[name]=max(errors[name],abs(value-row[name][b]))
            assert bool(values[0]<.2)==row['success'][b]
            indices=correspondence[f'{role}_{seed}'][b,mask[b]]
            distances=np.linalg.norm((points@matrix.T)[mask[b],:3]-data['centers'][b,1,indices],axis=-1)
            consensus_disagreements+=int(int((distances<=.1).sum())!=row['consensus_count'][b])
        grouped=scene_accuracy(row['success']);new_accuracy[key]=grouped
        old=old_rows[key]['success'];current=row['success']
        strata={name:dict(pairs=int(select.sum()),old_successful_pairs=int(old[select].sum()),
            new_successful_pairs=int(current[select].sum())) for name,select in
            [('fewer_than3',counts<3),('at_least3',counts>=3)]}
        records.append(dict(role=role,seed=seed,macro_success_rate=float(grouped.mean()),
            old_macro_success_rate=float(old_accuracy[key].mean()),successful_pairs=int(current.sum()),
            old_successful_pairs=int(old.sum()),invalid_estimates=int((~row['valid']).sum()),
            recovered_pairs=int((current&~old).sum()),lost_pairs=int((~current&old).sum()),
            both_success=int((current&old).sum()),both_failure=int((~current&~old).sum()),
            by_scene=dict(zip(scenes,grouped.tolist())),strata=strata,
            error_quantiles={k:dict(median=finite_or_none(np.median(row[k])),
                p90=finite_or_none(np.quantile(row[k],.9,method='higher'))) for k in ERRORS},
            null_error_quantile_means_infinite=True,receipt=receipt))
    assert errors['rmse_m']<1e-6 and errors['rotation_error_deg']<1e-5 and errors['centered_translation_error_m']<1e-6
    assert consensus_disagreements==0
    def stacked(mapping,role):return np.stack([mapping[role,s] for s in SEEDS])
    comparisons=[]
    for role,seed in zip(ROLES,lock['pruning_bootstrap_seeds']):
        stats=paired_statistics(stacked(new_accuracy,role),stacked(old_accuracy,role),seed=seed)
        signal=(stats['mean_gain_pp']>=lock['practical_pruning_gain_pp'] and
            min(stats['paired_seed_gains_pp'])>0 and stats['intervals_95_pp']['crossed'][0]>0)
        comparisons.append(dict(kind='mask_minus_all',left=role,right=role,**stats,
            practical_pruning_signal=signal))
    for other,seed in zip(lock['relative_comparisons'],lock['relative_bootstrap_seeds']):
        comparisons.append(dict(kind='masked_method_comparison',left='relative5',right=other,
            **paired_statistics(stacked(new_accuracy,'relative5'),stacked(new_accuracy,other),seed=seed)))
    for role,seed in zip(ORACLE_ROLES,lock['oracle_bootstrap_seeds']):
        comparisons.append(dict(kind='masked_model_minus_oracle',left='relative5',right=role,
            **paired_statistics(stacked(new_accuracy,'relative5'),np.tile(oracle_accuracy[role],(3,1)),seed=seed)))
    write(OUT / 'result.json',dict(completed_unix=time.time(),lock_sha256=sha(OUT / 'lock.json'),
        scenes=scenes,records=records,comparisons=comparisons,
        oracle_macro_success={k:float(v.mean()) for k,v in oracle_accuracy.items()},
        primary_pruning_signal=comparisons[0]['practical_pruning_signal'],
        pairs_with_fewer_than3_queries=int((counts<3).sum()),new_fits=0,new_inference=0,new_pose_groups=18,
        truth_validity_mask_used=True,deployable=False,fresh_confirmation=False,
        scientific_review_pending=True,goal_complete=False))
    write(OUT / 'independent_checks.json',dict(passed=True,result_sha256=sha(OUT / 'result.json'),
        score_maximum_errors=errors,success_decisions_changed=0,consensus_count_disagreements=consensus_disagreements,
        original_prediction_indices_unchanged=True,all64_source_centers_scored=True,
        identical_source_masks=True,hypotheses_counts_checked=True,
        bootstrap_maximum_error_pp=max(c['independent_draw_max_error'] for c in comparisons),
        pose_groups=18,pairs_per_group=480))
    print('LEARNED_VALID_QUERY_SUMMARY_READY',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('freeze','run','summarize'))
    globals()[parser.parse_args().action]()
