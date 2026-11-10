"""Label/center relationship on old fit0; no model predictions or tuning."""
import hashlib
import json
import time
from pathlib import Path
import numpy as np

ROOT = Path('/home/liwt/GRoPE')
OUT = Path(__file__).resolve().parent
CAMERA_CACHE = ROOT/'results/region_hrope_control/camera_data/fit0.npz'
DATA_LOCK = ROOT/'results/region_hrope_control/preparation/data_lock.json'
CACHE_RECORD = json.loads(DATA_LOCK.read_text())['caches']['fit0']
CACHE = Path(CACHE_RECORD['path'])


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    plan = json.loads((OUT/'plan.json').read_text())
    assert not (OUT/'result.json').exists()
    assert sha(CACHE) == CACHE_RECORD['sha256']
    with np.load(CACHE, allow_pickle=False) as data:
        ids, scenes = data['ids'], data['scenes']
        overlap, world = data['overlap'].astype(np.float64), data['world_centers'].astype(np.float64)
    assert len(ids) == 576 and overlap.shape == (576,64,64)
    distances = np.linalg.norm(world[:,0,:,None]-world[:,1,None,:],axis=-1)
    rows=[]
    for view in (0,1):
        o = overlap if view==0 else overlap.swapaxes(1,2)
        d = distances if view==0 else distances.swapaxes(1,2)
        a,b=world[:,view],world[:,1-view]
        valid=o.max(-1)>=.3
        top=o.argmax(-1)
        nearest=d.argmin(-1)
        picked_distance=np.take_along_axis(d,top[...,None],-1)[...,0]
        nearest_overlap=np.take_along_axis(o,nearest[...,None],-1)[...,0]
        weights=o*(o>=.1); probabilities=weights/np.maximum(weights.sum(-1,keepdims=True),1e-12)
        far_mass=(probabilities*(d>.1)).sum(-1)
        error=np.linalg.norm(probabilities@b-a,axis=-1)
        for scene in sorted(set(scenes.tolist())):
            selected=valid & (scenes==scene)[:,None]
            rows.append(dict(scene=scene,direction=view,queries=int((scenes==scene).sum())*64,
                valid_queries=int(selected.sum()),available_close_center=int(((d.min(-1)<=.1)&selected).sum()),
                top_overlap_close_center=int(((picked_distance<=.1)&selected).sum()),
                nearest_center_valid_overlap=int(((nearest_overlap>=.3)&selected).sum()),
                target_mass_far_sum=float(far_mass[selected].sum()),
                target_barycenter_error_sum=float(error[selected].sum()),
                overlap_argmax_center_error_median=float(np.median(picked_distance[selected]))))
    keys=('queries','valid_queries','available_close_center','top_overlap_close_center','nearest_center_valid_overlap',
          'target_mass_far_sum','target_barycenter_error_sum')
    totals={k:sum(r[k] for r in rows) for k in keys}
    n=totals['valid_queries']
    summary=dict(valid_query_fraction=n/totals['queries'],
        available_close_center_fraction_of_valid=totals['available_close_center']/n,
        highest_overlap_center_pck=totals['top_overlap_close_center']/n,
        nearest_center_region_recall=totals['nearest_center_valid_overlap']/n,
        mean_overlap_target_mass_beyond_center_threshold=totals['target_mass_far_sum']/n,
        mean_overlap_target_barycenter_error_pair_units=totals['target_barycenter_error_sum']/n)
    result=dict(completed_unix=time.time(),source_sha256=sha(__file__),plan_sha256=sha(OUT/'plan.json'),
        cache=dict(path=str(CACHE),sha256=sha(CACHE),pairs=576),camera_supplement_sha256=sha(CAMERA_CACHE),units='pair-normalized scene units, not metres',
        rows=rows,totals=totals,summary=summary,raw_statistics_only=True,
        no_model_predictions=True,no_change_to_running_comparison=True,fresh_confirmation=False,goal_complete=False)
    (OUT/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':
    main()
