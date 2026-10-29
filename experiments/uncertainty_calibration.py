"""Validation-only dustbin calibration after uncertainty scale-shift audit.

All model weights frozen. A single scalar offset is selected by validation CE
for every compared method. New render seeds (900+seed) are used only for test.
This cannot change real-candidate rankings or unmatched AUPRC, only calibration.
"""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from experiments import rendered_data
from experiments.rgb_geometry_probe import RGBGeometry,observation_from_rgb
from experiments.rendered_metrics import metrics
from experiments.rendered_probe import source_snapshot
from models.rendered_attention import RenderedMatcher


@torch.no_grad()
def scores(model,data):
    return torch.cat([model({k:v[start:start+32] for k,v in data.items()}) for start in range(0,len(data['q']),32)])


def main():
    torch.set_num_threads(1)
    out=Path('results/round2_calibration')
    out.mkdir(exist_ok=True)
    hashes=source_snapshot(out)
    grid=[-12.,-8.,-6.,-4.,-3.,-2.,-1.,0.,1.,2.,4.]
    records=[]
    raw={}
    original=rendered_data.patch_observation
    try:
        for seed in [11,22,33,44,55]:
            predictor=RGBGeometry()
            predictor.load_state_dict(torch.load(f'results/round2_rgb_geometry/checkpoints/geometry-{seed}.pt',map_location='cpu',weights_only=True)['state_dict'])
            rendered_data.patch_observation=lambda view,cx,cy,rng,condition:observation_from_rgb(predictor,view,cx,cy,rng,condition)
            validation,_=rendered_data.make_dataset(700+seed,48,'rgb_pred')
            test,_=rendered_data.make_dataset(900+seed,96,'rgb_pred')
            for regime in ['oracle_train','robust_train']:
                for method in ['rope3d','rope_normal','rope_normal_uncertainty']:
                    model=RenderedMatcher(method)
                    model.load_state_dict(torch.load(f'results/round2_rendered/checkpoints/{regime}-{seed}-{method}.pt',map_location='cpu',weights_only=True)['state_dict'])
                    val=scores(model,validation)
                    best=float('inf')
                    choice=None
                    candidates=[]
                    for offset in grid:
                        altered=val.clone()
                        altered[...,-1]+=offset
                        loss=float(F.cross_entropy(altered.flatten(0,1),validation['labels'].flatten()))
                        candidates.append({'offset':offset,'validation_loss':loss})
                        if loss<best: best,choice=loss,offset
                    logits=scores(model,test)
                    before,per_before=metrics(logits.softmax(-1).numpy(),test['labels'].numpy(),test['pixel_error'].numpy())
                    logits[...,-1]+=choice
                    after,per_after=metrics(logits.softmax(-1).numpy(),test['labels'].numpy(),test['pixel_error'].numpy())
                    after['loss']=float(F.cross_entropy(logits.flatten(0,1),test['labels'].flatten()))
                    assert abs(before['unmatched_auprc']-after['unmatched_auprc'])<1e-5
                    raw[f'{regime}/{seed}/{method}/before']=per_before
                    raw[f'{regime}/{seed}/{method}/after']=per_after
                    record={'seed':seed,'regime':regime,'method':method,'offset':choice,
                            'validation_grid':candidates,'before':before,'after':after}
                    records.append(record)
            print('CALIBRATED_FRESH',seed,flush=True)
    finally:
        rendered_data.patch_observation=original
    summary=[]
    for regime in ['oracle_train','robust_train']:
        for method in ['rope3d','rope_normal','rope_normal_uncertainty']:
            selected=[r for r in records if r['regime']==regime and r['method']==method]
            row={'regime':regime,'method':method,'offsets':[r['offset'] for r in selected]}
            for phase in ['before','after']:
                row[phase]={k:float(np.mean([r[phase][k] for r in selected])) for k in selected[0][phase] if all(r[phase][k] is not None for r in selected)}
            summary.append(row)
    (out/'manifest.json').write_text(json.dumps({'source_hashes':hashes,'selection':'validation CE only',
                                                'validation_seed_rule':'700+seed','fresh_test_seed_rule':'900+seed',
                                                'offset_grid':grid,'records':records},indent=2)+'\n')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    np.savez_compressed(out/'per_scene.npz',**raw)
    print('COMPLETE_CALIBRATION',flush=True)


if __name__=='__main__':
    main()
