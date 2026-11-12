"""Single prospective configuration, frozen after old-fit interface checks."""
import hashlib
import json
from pathlib import Path

from experiments.gaussian_scene_data import OUT, ROOT, input_plan, read
from experiments.gaussian_scene_build import PILOT_CONFIG
from experiments.gaussian_memory_incremental_data import write_json
from experiments.gaussian_memory_setup import sha256


def freeze():
    path=OUT/'plan.json'
    if path.exists(): return read(path)
    assert read(OUT/'native_integration_checks.json')['complete']
    p=input_plan()
    p.update(steps=2000, effective_batch=2, lr=1e-4, weight_decay=.01, gradient_clip=5.,
             source_fit=PILOT_CONFIG, feature_channels=1024, feature_grid=[32,32], slots=2048,
             read_layers=[0,6,12,18], reader_width=256, reader_heads=4, native_first_view_token_index=0,
             source_feature_dtype='float16', loss='0.5*(whole valid raw log MSE + centered log MSE)',
             train_sampling='paired seed permutation of 12 clips x4 queries x2 conditions; batch2; repeat reshuffle',
             main_endpoint='missing ROI centered log MSE, places equal, seeds paired',
             bootstrap=dict(replicates=5000,seed=20261111,units='independent resample place and paired seed'),
             support=dict(evaluation_only=True, rule='query GT surface visible in any source GT depth',
                          depth_tolerance_m=.05,depth_tolerance_fraction=.02,sampling='nearest',grid=[210,210]),
             interventions=dict(kinds=['remove_memory','shuffle_binding','wrong_camera'],model='G final, all3seeds',
                                queries='all12development places x4queries x2conditions',shuffle_seed=20261111,
                                wrong_camera='translate query along its own camera-right by0.5 source units; rotate unchanged',
                                remove_memory='all content invalid, zero residual',
                                shuffle_binding='jointly permute feature and content-valid slots; geometry unchanged'),
             renderer_reference=dict(grid=[210,210],mode='expected depth',coverage_threshold=.0001,
                                     uncovered_fill='source-only unit; report coverage and covered-only scores'),
             official_reference='same four complete sources, cached official streaming prefix then each query independently',
             primary_hypothesis='G scene retrieval can exceed L/T and same-content center P',
             competing_explanations=['common adapter/bias correction','raw source tokens sufficient','center PE sufficient','scene fidelity or receiver limits'],
             development_tuning=False,new_confirmation=False,interface_weights_discarded=True,
             source_state_bytes=4319652,token_state_bytes=4318215,token_slots=2095,
             base_parameters=1054464,center_extra_parameters=192,
             outline=['O2','O3','O4','O5','O6'],choices=['C1','C2','C3','C4','C5','C6'],
             temporary_scope=['static source set with future sources for early query','given cameras','frozen native model and DPT','old development data'],
             decisions='C/P useful without G increment: retain simpler reader; renderer-only or bias-only gain: limited evidence; no L/T net utility: close fixed configuration; no automatic successor')
    write_json(path,p)
    files=['models/gaussian_scene_memory.py','models/gaussian_scene_native.py',
           'experiments/gaussian_scene_data.py','experiments/gaussian_scene_build.py',
           'experiments/gaussian_scene_anysplat.py','experiments/gaussian_scene_bind.py']
    write_json(OUT/'configuration_lock.json',dict(plan_sha256=sha256(path),preflight_sha256=sha256(OUT/'native_integration_checks.json'),
        source_sha256={name:sha256(ROOT/name) for name in files},formal_training_started=False,
        scope='Configuration before any development scoring; later ordinary implementation repairs get versioned records'))
    return p


def schedule(plan, seed):
    import numpy as np
    keys=[(m['id'],q,c) for m in plan['training_catalog'] for q in plan['queries'] for c in plan['conditions']]
    rng=np.random.default_rng(seed); result=[]
    while len(result)<plan['steps']*plan['effective_batch']:
        result.extend(keys[int(i)] for i in rng.permutation(len(keys)))
    return result[:plan['steps']*plan['effective_batch']]


if __name__=='__main__':freeze()
