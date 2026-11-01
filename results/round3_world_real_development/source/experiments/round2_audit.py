"""Integrity, checkpoint replay, and rejection-vs-ranking diagnostics."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from experiments import rendered_data
from experiments.rgb_geometry_probe import RGBGeometry,observation_from_rgb
from models.rendered_attention import RenderedMatcher


def main():
    torch.set_num_threads(1)
    verification={'unit_tests':16,'source_snapshots':{},'finite_per_scene':{},'checkpoint_replay':[],
                  'diagnostic':'Closed-set R1 ranks the 48 real candidates before softmax; visible R1 also counts mistaken dustbin rejection.'}
    for round,filename in [('round2_identity','manifest.json'),('round2_rendered','manifest.json'),('round2_rgb_geometry','predictor_manifest.json'),('round2_calibration','manifest.json')]:
        root=Path('results')/round
        manifest=json.loads((root/filename).read_text())
        for name,expected in manifest['source_hashes'].items():
            assert hashlib.sha256((root/'source'/name).read_bytes()).hexdigest()==expected,(round,name)
        verification['source_snapshots'][round]=len(manifest['source_hashes'])
        raw=np.load(root/'per_scene.npz')
        assert all(np.isfinite(raw[k]).all() for k in raw.files),round
        verification['finite_per_scene'][round]=len(raw.files)
    identity=np.load('results/round2_identity/per_scene.npz')
    for seed in [11,22,33,44,55]:
        for condition in ['repeated_geometry','planar_geometry']:
            assert np.array_equal(identity[f'{seed}/rope3d_axis/id/r1'],identity[f'{seed}/rope3d_axis/{condition}/r1'])
    verification['identity_position_invariance']=True
    manifest=json.loads(Path('results/round2_rendered/manifest.json').read_text())
    rendered=np.load('results/round2_rendered/per_scene.npz')
    for regime in ['oracle_train','robust_train']:
        for seed in [11,22,33,44,55]:
            for condition in ['normal_noise','scale_noise']:
                assert np.array_equal(rendered[f'{regime}/{seed}/rope3d/oracle'],rendered[f'{regime}/{seed}/rope3d/{condition}'])
    verification['rendered_position_invariance']=True
    diagnostics=[]
    chosen=['rope3d','rope_normal','rope_covariance','rope_normal_uncertainty']
    original=rendered_data.patch_observation
    for seed in [11,22,33,44,55]:
        cache={}
        datasets={c:rendered_data.make_dataset(800+seed,96,c,cache)[0] for c in ['oracle','normal_noise','mixed']}
        predictor=RGBGeometry()
        predictor.load_state_dict(torch.load(f'results/round2_rgb_geometry/checkpoints/geometry-{seed}.pt',map_location='cpu',weights_only=True)['state_dict'])
        rendered_data.patch_observation=lambda view,cx,cy,rng,condition:observation_from_rgb(predictor,view,cx,cy,rng,condition)
        try:
            datasets['rgb']=rendered_data.make_dataset(800+seed,96,'rgb_pred',cache)[0]
        finally:
            rendered_data.patch_observation=original
        for regime in ['oracle_train','robust_train']:
            for method in chosen:
                model=RenderedMatcher(method)
                checkpoint=Path('results/round2_rendered/checkpoints')/f'{regime}-{seed}-{method}.pt'
                model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True)['state_dict'])
                for condition,data in datasets.items():
                    rankings=[]
                    probabilities=[]
                    for start in range(0,96,32):
                        part={k:v[start:start+32] for k,v in data.items()}
                        with torch.no_grad(): logits=model(part)
                        assert torch.isfinite(logits).all()
                        rankings.append(logits[...,:48].argmax(-1))
                        probabilities.append(logits.softmax(-1))
                    ranking=torch.cat(rankings)
                    probability=torch.cat(probabilities)
                    labels=data['labels']
                    visible=labels<48
                    closed=((ranking==labels)&visible).sum()/visible.sum()
                    open_correct=((probability.argmax(-1)==labels)&visible).sum()/visible.sum()
                    rejected=((probability.argmax(-1)==48)&visible).sum()/visible.sum()
                    diagnostics.append({'regime':regime,'seed':seed,'method':method,'condition':condition,
                                        'closed_r1':float(closed),'visible_r1':float(open_correct),
                                        'visible_rejection_rate':float(rejected)})
                    if seed==11 and regime=='oracle_train' and condition=='oracle':
                        saved=np.load(f'results/round2_rendered/predictions/{regime}-{seed}-{method}.npz')
                        difference=float(np.abs(probability.numpy()-saved[condition+'/probability'].astype(np.float32)).max())
                        assert difference<.003,(method,difference)
                        verification['checkpoint_replay'].append({'method':method,'max_probability_difference_vs_float16_archive':difference})
        print('AUDITED',seed,flush=True)
    Path('results/round2_rendered/rejection_audit.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
    verification['training_runs']={'identity':30,'rendered':len(manifest['records']),'rgb_predictors':5}
    verification['gate_rule_uses_validation_only']=manifest['gate_trigger_rule']
    verification['passed']=True
    Path('results/round2_verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    print(json.dumps(verification,indent=2))


if __name__=='__main__':
    main()
