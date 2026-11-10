"""Matched truth-valid query subsets isolate regional-label versus center choices."""
import argparse
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from experiments.round65_common import ROOT, read, write, sha
from experiments.region_pose_utility import arrays, configure_cuda
from experiments.region_pose_solver import estimate, score_transform
from experiments.region_supervision_oracle import load_inputs

PARENT = ROOT / 'results/region_supervision_oracle/diagnostic1'
OUT = PARENT / 'valid_query_control'
PLAN = ROOT / 'docs/region_supervision_oracle_scope_addendum.md'
ROLES = ('valid_gt_nearest', 'valid_overlap_nearest_tie', 'valid_overlap_first_tie')


def estimate_subset(source, target, indices):
    keep = indices >= 0
    if int(keep.sum()) < 3:
        return dict(rotation=np.eye(3), translation=np.zeros(3), valid=False,
            consensus_count=0, truncated_squared_error=None, hypotheses_total=0,
            distinct_target_hypotheses=0, nondegenerate_hypotheses=0, refits=0)
    return estimate(source[keep], target, indices[keep], radius=.1, max_refits=5)


def freeze():
    assert not OUT.exists()
    original = read(PARENT / 'lock.json')
    prep = read(PARENT / 'preparation.json')
    assert prep['correspondences_sha256'] == sha(PARENT / 'correspondences.npz')
    # A known transform with discarded outliers checks the changed query selection.
    source = torch.tensor([[0.,0.,0.], [1.,0.,0.], [0.,1.,0.], [0.,0.,1.], [5.,5.,5.]], dtype=torch.float64)
    rotation = Rotation.from_rotvec([.2, -.3, .1]).as_matrix()
    shift = np.array([.1, .2, -.1])
    target = torch.from_numpy(source.numpy() @ rotation.T + shift)
    target[4] = 100.
    selected = torch.tensor([0,1,2,3,-1])
    result = estimate_subset(source, target, selected)
    assert result['valid'] and result['hypotheses_total'] == 4 and result['consensus_count'] == 4
    np.testing.assert_allclose(result['rotation'], rotation, atol=1e-10)
    np.testing.assert_allclose(result['translation'], shift, atol=1e-10)
    insufficient = estimate_subset(source, target, torch.tensor([0,1,-1,-1,-1]))
    assert not insufficient['valid'] and insufficient['hypotheses_total'] == 0
    sources = [Path(__file__), PLAN, ROOT / 'experiments/region_pose_solver.py',
        ROOT / 'experiments/region_pose_utility.py', ROOT / 'experiments/region_supervision_oracle.py',
        ROOT / 'experiments/round65_common.py', PARENT / 'lock.json', PARENT / 'preparation.json',
        PARENT / 'correspondences.npz', PARENT / 'gpu_replay.json']
    write(OUT / 'lock.json', dict(created_unix=time.time(),
        source_sha256={str(p):sha(p) for p in sources}, cache=original['cache'], truth=original['truth'],
        groups_path=original['groups_path'], roles=list(ROLES),
        comparisons=[[ROLES[1], ROLES[0]], [ROLES[2], ROLES[0]], [ROLES[1], ROLES[2]]],
        bootstrap_seeds=[99201,99202,99203], bootstrap_replicates=10000,
        CPU_subset_checks_passed=True, original_oracle_outcomes_observed_before_addendum=False,
        all480_pairs_retained=True, truth_validity_mask_used=True, deployable=False,
        new_training=False, new_inference=False, fresh_confirmation=False, goal_complete=False))
    (OUT / 'lock.sha256').write_text(sha(OUT / 'lock.json')+'\n')
    with np.load(PARENT / 'correspondences.npz', allow_pickle=False) as z:
        mask = z['valid_region_queries']
        payload = {role:np.where(mask,z[role.removeprefix('valid_')],-1) for role in ROLES}
        np.savez(OUT / 'correspondences.npz', ids=z['ids'], scenes=z['scenes'],
                 valid_source_queries=mask, **payload)
    write(OUT / 'preparation.json', dict(completed_unix=time.time(),
        correspondences_sha256=sha(OUT / 'correspondences.npz'),
        query_counts=mask.sum(-1).tolist(), pairs_with_fewer_than3_queries=int((mask.sum(-1)<3).sum()),
        total_valid_source_queries=int(mask.sum()), lock_sha256=sha(OUT / 'lock.json')))
    print('VALID_QUERY_ORACLE_FROZEN', sha(OUT / 'lock.json'), flush=True)


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
    assert not (OUT / 'execution_status.json').exists(), 'Inspect previous worker before any recovery.'
    configure_cuda()
    data,truth=load_inputs(lock)
    correspondence=arrays(OUT / 'correspondences.npz', ('ids','valid_source_queries',*ROLES))
    np.testing.assert_array_equal(data['ids'],correspondence['ids'])
    centers=torch.as_tensor(data['centers'],device='cuda')
    state=dict(state='estimating',pid=os.getpid(),started_unix=time.time(),active=None,completed=[],
        lock_sha256=sha(OUT / 'lock.json'),CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],goal_complete=False)
    def save():
        state['updated_unix']=time.time();write(OUT / 'execution_status.json',state)
    save()
    try:
        for role in ROLES:
            started=time.time();records=[]
            state['active']=dict(role=role,completed_pairs=0);save()
            for b in range(len(data['ids'])):
                indices=torch.as_tensor(correspondence[role][b],device='cuda',dtype=torch.long)
                result=estimate_subset(centers[b,0],centers[b,1],indices)
                count=int(correspondence['valid_source_queries'][b].sum())
                assert result['hypotheses_total']==(math.comb(count,3) if count>=3 else 0)
                score=score_transform(data['centers'][b,0],result['rotation'],result['translation'],
                    truth['rotation'][b],truth['translation'][b],data['scales'][b],valid=result['valid'])
                records.append(dict(**result,**score))
                if (b+1)%32==0:
                    state['active']['completed_pairs']=b+1;save()
            payload={k:np.stack([r[k] for r in records]) for k in records[0] if k!='truncated_squared_error'}
            payload['truncated_squared_error']=np.asarray([np.nan if r['truncated_squared_error'] is None
                else r['truncated_squared_error'] for r in records])
            path=OUT/(role+'.npz');assert not path.exists()
            np.savez(path,ids=data['ids'],scenes=data['scenes'],**payload)
            state['completed'].append(dict(role=role,path=str(path),sha256=sha(path),seconds=time.time()-started,pairs=480))
            save();print('VALID_QUERY_ORACLE_ESTIMATED',role,time.time()-started,flush=True)
        state.update(state='estimates_complete',active=None,finished_unix=time.time(),
                     peak_allocated_bytes=torch.cuda.max_memory_allocated());save()
    except BaseException as error:
        state.update(state='failed',error_type=type(error).__name__,reason=str(error));save();raise


def finite(value):
    return float(value) if np.isfinite(value) else None


def summarize():
    lock=check();state=read(OUT / 'execution_status.json')
    assert state['state']=='estimates_complete' and len(state['completed'])==3
    assert not (OUT / 'result.json').exists()
    data,_=load_inputs(lock)
    correspondence=arrays(OUT / 'correspondences.npz',('ids','valid_source_queries',*ROLES))
    grouping=arrays(lock['groups_path'],('ids','group'))
    np.testing.assert_array_equal(grouping['ids'],data['ids'])
    scenes=sorted(set(data['scenes'].tolist()));truth=[]
    for b in range(480):
        transforms=[]
        for view in range(2):
            a=data['centers'][b,view];w=data['world_centers'][b,view].astype(np.float64)
            rotation,_=Rotation.align_vectors(w-w.mean(0),a-a.mean(0));matrix=np.eye(4)
            matrix[:3,:3]=rotation.as_matrix();matrix[:3,3]=w.mean(0)-rotation.apply(a.mean(0));transforms.append(matrix)
        truth.append(np.linalg.solve(transforms[1],transforms[0]))
    truth=np.stack(truth);records=[];accuracy={};errors=dict(rmse_m=0.,rotation_error_deg=0.,centered_translation_error_m=0.)
    consensus_differences=0
    for receipt in state['completed']:
        assert sha(receipt['path'])==receipt['sha256']
        row=arrays(receipt['path'],('ids','scenes','rotation','translation','valid','success',*errors,'consensus_count','hypotheses_total'))
        np.testing.assert_array_equal(row['ids'],data['ids']);np.testing.assert_array_equal(row['scenes'],data['scenes'])
        for b in range(480):
            keep=correspondence['valid_source_queries'][b];m=int(keep.sum())
            assert int(row['hypotheses_total'][b])==(math.comb(m,3) if m>=3 else 0)
            if not row['valid'][b]:
                assert not row['success'][b] and np.isinf(row['rmse_m'][b]);continue
            matrix=np.eye(4);matrix[:3,:3]=row['rotation'][b];matrix[:3,3]=row['translation'][b]
            np.testing.assert_allclose(matrix[:3,:3].T@matrix[:3,:3],np.eye(3),atol=1e-8)
            assert abs(np.linalg.det(matrix[:3,:3])-1)<1e-8
            points=np.column_stack((data['centers'][b,0],np.ones(64)))
            residual=points@(matrix-truth[b]).T
            rmse=float(np.linalg.norm(residual[:,:3],'fro')/8*data['scales'][b])
            angle=float(np.degrees(Rotation.from_matrix(matrix[:3,:3]@truth[b,:3,:3].T).magnitude()))
            translation=float(np.linalg.norm((matrix-truth[b])[:3,3])*data['scales'][b])
            for key,value in [('rmse_m',rmse),('rotation_error_deg',angle),('centered_translation_error_m',translation)]:
                errors[key]=max(errors[key],abs(value-row[key][b]))
            assert bool(rmse<.2)==row['success'][b]
            indices=correspondence[receipt['role']][b,keep]
            distances=np.linalg.norm((points@matrix.T)[keep,:3]-data['centers'][b,1,indices],axis=-1)
            consensus_differences+=int(int((distances<=.1).sum())!=row['consensus_count'][b])
        by_scene={s:float(row['success'][data['scenes']==s].mean()) for s in scenes}
        accuracy[receipt['role']]=np.asarray(list(by_scene.values()))
        groups={}
        for g in ('low','high','undefined'):
            mask=grouping['group']==g;available=[s for s in scenes if np.any(mask&(data['scenes']==s))]
            groups[g]=dict(pairs=int(mask.sum()),scene_macro_success=float(np.mean([
                row['success'][mask&(data['scenes']==s)].mean() for s in available])))
        records.append(dict(role=receipt['role'],macro_success=float(np.mean(list(by_scene.values()))),
            successful_pairs=int(row['success'].sum()),invalid_estimates=int((~row['valid']).sum()),by_scene=by_scene,groups=groups,
            error_quantiles={k:dict(median=finite(np.median(row[k])),p90=finite(np.quantile(row[k],.9,method='higher'))) for k in errors},
            null_error_quantile_means_infinite=True,receipt=receipt,deployable=False))
    assert errors['rmse_m']<1e-6 and errors['rotation_error_deg']<1e-5 and errors['centered_translation_error_m']<1e-6
    assert consensus_differences==0
    comparisons=[];bootstrap_error=0.
    for (left,right),seed in zip(lock['comparisons'],lock['bootstrap_seeds']):
        delta=100*(accuracy[left]-accuracy[right]);index=np.random.default_rng(seed).integers(5,size=(10000,5))
        draws=delta[index].mean(-1);alternative=100*(accuracy[left][index].mean(-1)-accuracy[right][index].mean(-1))
        bootstrap_error=max(bootstrap_error,float(np.max(np.abs(draws-alternative))))
        comparisons.append(dict(left=left,right=right,mean_gain_pp=float(delta.mean()),by_scene_gain_pp=delta.tolist(),
            scene_interval_95_pp=np.quantile(draws,[.025,.975]).tolist(),bootstrap_seed=seed,bootstrap_replicates=10000))
    assert bootstrap_error<1e-10
    write(OUT / 'result.json',dict(completed_unix=time.time(),lock_sha256=sha(OUT / 'lock.json'),records=records,
        comparisons=comparisons,scenes=scenes,query_counts=read(OUT / 'preparation.json')['query_counts'],
        pairs_with_fewer_than3_queries=int((correspondence['valid_source_queries'].sum(-1)<3).sum()),
        new_fits=0,new_inference=0,new_pose_groups=3,truth_validity_mask_used=True,
        scientific_review_pending=True,fresh_confirmation=False,goal_complete=False))
    write(OUT / 'independent_checks.json',dict(passed=True,result_sha256=sha(OUT / 'result.json'),
        score_maximum_errors=errors,success_decisions_changed=0,consensus_count_disagreements=0,
        bootstrap_maximum_error_pp=bootstrap_error,pose_groups=3,pairs_per_group=480,
        all64_source_centers_scored=True,identical_valid_query_sets_for_all3_controls=True))
    print('VALID_QUERY_ORACLE_SUMMARY_READY',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('freeze','run','summarize'))
    globals()[parser.parse_args().action]()
