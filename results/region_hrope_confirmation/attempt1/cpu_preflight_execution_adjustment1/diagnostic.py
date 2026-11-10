"""Old validation only: inspect CPU/GPU near-tie consequences before new pixels."""
import time
import numpy as np
import torch
from experiments.region_hrope_confirmation import OUT, check, configure, infer, development_cache, CameraRegionMatcher
from experiments.round58_run import metric_torch
from experiments.round65_common import write, sha

lock=check();configure()
assert not (OUT/'data_access_started.json').exists()
cache=development_cache('val')
rows=[]
for item in lock['models']:
    model=CameraRegionMatcher(item['role']).cpu().eval()
    model.load_state_dict(torch.load(item['checkpoint_path'],map_location='cpu',weights_only=True)['state_dict'])
    logits=infer(model,cache,device='cpu')
    with np.load(item['validation_path'],allow_pickle=False) as z:
        np.testing.assert_array_equal(z['ids'],cache['ids']);previous=z['logits']
    changes=0;max_center=0.;max_margin=0.;correct_changes=0
    for direction in range(2):
        new=logits if direction==0 else logits.swapaxes(-1,-2)
        old=previous if direction==0 else previous.swapaxes(-1,-2)
        truth=cache['overlap'] if direction==0 else cache['overlap'].swapaxes(-1,-2)
        a,b=new.argmax(-1),old.argmax(-1)
        indices=np.arange(len(new))[:,None]
        changes+=int((a!=b).sum())
        centers=cache['centers'][:,1-direction]
        max_center=max(max_center,float(np.linalg.norm(centers[indices,a]-centers[indices,b],axis=-1).max()))
        max_margin=max(max_margin,float(np.abs(np.take_along_axis(old,a[...,None],-1)-np.take_along_axis(old,b[...,None],-1)).max()))
        correct_changes+=int(((np.take_along_axis(truth,a[...,None],-1)>=.3)!=(np.take_along_axis(truth,b[...,None],-1)>=.3)).sum())
    new_metrics=metric_torch(logits,cache['overlap'],cache['world_centers'])
    old_metrics=metric_torch(previous,cache['overlap'],cache['world_centers'])
    metric_differences={k:float(new_metrics[k]-old_metrics[k]) for k in ('region_recall','center_pck','soft_epe','ce')}
    row=dict(role=item['role'],seed=item['seed'],max_logit_error=float(np.abs(logits-previous).max()),
        argmax_changes=changes,maximum_selected_target_center_change=max_center,
        maximum_reference_score_margin=max_margin,overlap_correctness_changes=correct_changes,
        metric_differences=metric_differences)
    rows.append(row);print(row,flush=True)
    del model
write(OUT/'cpu_preflight_execution_adjustment1/diagnostic.json',dict(completed_unix=time.time(),rows=rows,
    source_sha256=sha(__file__),evaluation_lock_sha256=sha(OUT/'evaluation_lock.json'),
    old_validation_only=True,new_pixels_accessed=False))
