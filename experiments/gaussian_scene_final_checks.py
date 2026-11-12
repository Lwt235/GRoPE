"""Finite input/export checks for this completed run; no new efficacy thresholds."""
import itertools
from pathlib import Path

import numpy as np
import torch

from experiments.gaussian_scene_data import OUT,ROOT,CACHE,source_path,read,KINDS,SEEDS
from experiments.gaussian_scene_bind import load_tensor
from experiments.gaussian_memory_setup import sha256
from experiments.gaussian_memory_incremental_data import write_json
from models.gaussian_scene_memory import field_bytes


def main():
    torch.set_num_threads(4)
    p=read(OUT/'plan.json');plan_hash=sha256(OUT/'plan.json')
    assert read(OUT/'paired_controls_identity.json')['complete']
    assert read(OUT/'scoring/independent_recomputation.json')['predictions']==2592
    lock=read(OUT/'training_lock.json')
    for path,digest in lock['input_manifest_sha256'].items():assert sha256(path)==digest,path
    weight=read(OUT/'setup/anysplat_weights.json')
    for name,digest in weight['files'].items():assert sha256(Path(weight['folder'])/name)==digest,name
    assert sha256(p['model_path'])==p['model_sha256']
    assert (ROOT/'third_party/anysplat/PINNED_COMMIT').read_text().strip()=='5f5e208a7dd57d52e43ea0d553a95eab526e8775'
    runtime=read(OUT/'setup/anysplat_runtime.json')
    assert runtime['all_retained_weights_loaded'] and runtime['strict_full_state_keys']==1405
    assert runtime['checkpoint_sha256']==weight['files']['model.safetensors']
    for name,digest in p['official_source_sha256'].items():
        assert sha256(ROOT/'third_party/streamvggt'/name)==digest,name
    allowed={'experiments/gaussian_scene_train.py','experiments/gaussian_scene_orchestrate.py',
             'experiments/gaussian_scene_launch.py','experiments/gaussian_scene_score.py',
             'experiments/gaussian_scene_figures.py','experiments/gaussian_scene_postprocess.py'}
    for name,digest in lock['source_sha256'].items():
        if name not in allowed:assert sha256(ROOT/name)==digest,name
    repair=read(OUT/'training_source_change_review.json')
    assert repair['exact_original_reconstruction']
    assert repair['current_source_sha256']==sha256(ROOT/'experiments/gaussian_scene_train.py')
    assert repair['locked_source_sha256']==lock['source_sha256']['experiments/gaussian_scene_train.py']
    fields=[];query_files={}
    for meta in p['training_catalog']+p['catalog']:
        rid=meta['id'];source=load_tensor(source_path(rid));binding=read(OUT/'binding'/f'{rid}.json')
        assert set(source)=={'rgb','c2w','K','source_ids','id','place','split'}
        assert source['source_ids']==p['sources'] and tuple(source['rgb'].shape)==(4,3,448,448)
        assert sha256(source_path(rid))==binding['source_input_sha256']
        init=read(OUT/'source_initialization'/f'{rid}.json')
        assert init['source_payload_sha256']==binding['source_input_sha256']
        assert init['checkpoint_sha256']==weight['files']['model.safetensors']
        assert binding['source_only'] and binding['tensor_save_reload_exact']
        for path,digest in binding['files'].items():assert sha256(path)==digest,path
        field=load_tensor(CACHE/rid/'scene_memory.pt');tokens=load_tensor(CACHE/rid/'token_memory.pt')
        assert tuple(field['means'].shape)==(2048,3) and tuple(field['features'].shape)==(2048,1024)
        assert field['features'].dtype==tokens['features'].dtype==torch.float16
        assert field['source_ids'].tolist()==p['sources']
        assert torch.equal(field['source_c2w'],source['c2w'])
        assert torch.equal(field['source_intrinsic'],source['K'])
        assert field_bytes(field)==p['source_state_bytes'] and field_bytes(tokens)==p['token_state_bytes']
        assert len(tokens['means'])==p['token_slots'] and field_bytes(tokens)<=field_bytes(field)
        cache=read(OUT/'query_cache'/f'{rid}.json')
        assert cache['plan_sha256']==plan_hash
        assert cache['saved_scene_sha256']==binding['files'][str(CACHE/rid/'scene_memory.pt')]
        assert cache['source_state_unchanged'] and cache['four_queries_same_saved_state']
        assert not cache['query_cache_contains_history_prefix']
        query_files.update(cache['files'])
        fields.append(dict(id=rid,source_payload_only=True,scene_tensor_bytes=field_bytes(field),
                           token_tensor_bytes=field_bytes(tokens),content_valid=int(field['has_content'].sum())))
    for path,digest in query_files.items():assert sha256(path)==digest,path
    base=set(itertools.product([m['id'] for m in p['catalog']],p['queries'],p['conditions']))
    exports=[];native=None;empty=[]
    manifests=[OUT/'references/native.json',OUT/'references/renderer.json']
    manifests += [OUT/'evaluation'/f'{kind}_{seed}.json' for seed in SEEDS for kind in KINDS]
    manifests += [OUT/'interventions'/f'G_{seed}.json' for seed in SEEDS]
    for marker in manifests:
        record=read(marker);assert record['complete'] and record['plan_sha256']==plan_hash
        assert sha256(record['prediction_path'])==record['prediction_sha256']
        payload=load_tensor(record['prediction_path']);keys=[tuple(key) for key in payload['keys']]
        if marker==OUT/'references/native.json':
            expected={(*key,name) for key in base for name in ('native_current','official_history')}
            native={key[:3]:value for key,value in zip(keys,payload['depth_metric']) if key[3]=='native_current'}
        elif marker==OUT/'references/renderer.json':expected={(*key,'render_depth') for key in base}
        elif marker.parent.name=='interventions':expected={(*key,name) for key in base for name in p['interventions']['kinds']}
        else:expected=base
        assert len(keys)==len(expected) and set(keys)==expected,marker
        assert tuple(payload['depth_metric'].shape)==(len(expected),210,210)
        assert torch.isfinite(payload['depth_metric']).all() and (payload['depth_metric']>0).all()
        if marker.parent.name=='interventions':
            differences=[float((value-native[key[:3]]).abs().max()) for key,value in zip(keys,payload['depth_metric']) if key[3]=='remove_memory']
            assert len(differences)==96 and max(differences)==0
            empty.append(dict(seed=record['seed'],predictions=96,max_depth_difference=0.,prediction_sha256=record['prediction_sha256']))
        if marker.parent.name in ('evaluation','interventions'):
            seed=record['seed'];kind=record.get('kind','G')
            assert record['checkpoint_sha256']==read(OUT/f'{kind}_{seed}'/'status.json')['checkpoint_sha256']
        exports.append(dict(path=str(marker.relative_to(OUT)),predictions=len(keys),keys_exact=True))
    assert sum(row['predictions'] for row in exports)==2592
    write_json(OUT/'trained_empty_memory_identity_all_seeds.json',dict(
        complete=True,comparisons=empty,predictions=288,max_depth_difference=0.,
        scope='Actual all3seed removed-memory G exports equal the96 unadapted current-image depths exactly; removal also removes the trained adapter.'))
    write_json(OUT/'final_runtime_integrity.json',dict(complete=True,plan_sha256=plan_hash,
        models_actual_weight_hashes_verified=True,native_official_source_hashes_verified=True,
        fixed_scientific_sources_verified=True,documented_training_change='training_source_change_review.json',
        input_manifests_verified=len(lock['input_manifest_sha256']),query_cache_files_verified=len(query_files),
        source_fields=fields,exports=exports,exported_predictions=2592,
        removed_memory_native_identity='trained_empty_memory_identity_all_seeds.json',
        scope='Finite input/source/state/export coverage and implementation checks only; not an efficacy threshold or whole-goal completion.'))
    print(dict(complete=True,source_fields=len(fields),exports=len(exports),predictions=2592,empty_identity_predictions=288),flush=True)


if __name__=='__main__':
    main()
