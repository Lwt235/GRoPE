"""Supplementary relative-neighborhood matching diagnostic from saved outputs."""
import argparse
from pathlib import Path
import numpy as np
from experiments import round50_train as P
from experiments.round50_data import OUT, CONDITIONS, sha, write


def plan():
    path=OUT/'neighborhood/plan.json'
    if path.exists():raise FileExistsError('retain plan')
    write(path,dict(source_sha256=sha(__file__),primary_results_read=False,
        scope='additional DEVELOPMENT diagnostic planned before unified model evaluation; not the frozen primary metric',
        formula='mean norm((soft_xy_i-soft_xy_j)-(gt_xy_i-gt_xy_j)) on all visible eligible horizontal/vertical adjacent query-token pairs',
        limitations='Common translation errors cancel. Interpret jointly with center EPE/AUC, never as standalone success. No tuning, labels changed or subset selected by model output.',
        confirmation_access=False))
    (OUT/'neighborhood/source.py').write_bytes(Path(__file__).read_bytes())


def run():
    if P.read(OUT/'neighborhood/plan.json')['source_sha256']!=sha(__file__):raise ValueError('changed source')
    if len(P.read(OUT/'sealed_models.json')['records'])!=48:raise ValueError('seal before analysis')
    grid=np.arange(64).reshape(8,8)
    i=np.r_[grid[:,:-1].ravel(),grid[:-1,:].ravel()]
    j=np.r_[grid[:,1:].ravel(),grid[1:,:].ravel()]
    rows=[];vectors={}
    for c in CONDITIONS:
        with np.load(OUT/'data'/f'dev_{c}.npz',allow_pickle=False) as z:
            target=z['qtarget'].astype(float);valid=z['qvalid']
        eligible=valid[:,i]&valid[:,j];count=eligible.sum(-1)
        if np.any(count==0):raise ValueError('empty scene, disclose instead of silently removing')
        truth=target[:,i]-target[:,j]
        for train in P.REGIMES:
            for inputs in P.REGIMES:
                for role in P.ROLES:
                    for seed in P.SEEDS:
                        p=OUT/'predictions'/train/role/str(seed)/f'{c}_{inputs}.npz'
                        with np.load(p,allow_pickle=False) as z:xy=z['soft_xy'].astype(float)
                        error=np.linalg.norm(xy[:,i]-xy[:,j]-truth,axis=-1)
                        per_scene=(error*eligible).sum(-1)/count
                        vectors[train,inputs,role,seed,c]=per_scene
                        rows.append(dict(train_regime=train,input_regime=inputs,role=role,seed=seed,condition=c,
                            relative_displacement_epe=float(per_scene.mean()),eligible_edges=int(eligible.sum())))
    comparisons=[]
    for regime in P.REGIMES:
        for control in [r for r in P.ROLES if r!='gs_frame']:
            for conds in [('base',),CONDITIONS]:
                diffs=[]
                for seed in P.SEEDS:
                    diffs.append(float(np.mean([vectors[regime,regime,'gs_frame',seed,c]-vectors[regime,regime,control,seed,c] for c in conds])))
                comparisons.append(dict(regime=regime,control=control,conditions=list(conds),
                    error_delta=float(np.mean(diffs)),seed_deltas=diffs,improved_seeds=sum(d<0 for d in diffs)))
    write(OUT/'neighborhood/result.json',dict(rows=rows,comparisons=comparisons,exploratory_not_primary=True))
    print('NEIGHBORHOOD',len(rows),'rows',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['plan','run']);a=p.parse_args()
    plan() if a.action=='plan' else run()
