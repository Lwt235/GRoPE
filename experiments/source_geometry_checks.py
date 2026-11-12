"""Matched input/source/model identities, without an added efficacy gate."""
import hashlib
import numpy as np
import torch
from experiments.source_geometry_data import *

def parent_identity():
    p=parent_plan();verified={}
    for meta in p['training_catalog']+p['catalog']:
        manifest=PARENT_OUT/'inputs'/f'{meta["id"]}.json';record=read(manifest)
        for path,digest in record['files'].items():assert sha256(path)==digest,(path,'changed parent input')
        cache=read(PARENT_OUT/'query_cache'/f'{meta["id"]}.json')
        # Current image patch inputs and query target/camera payloads must remain the same.
        for q in QUERIES:
            for c in CONDITIONS:
                path=str(patch_path(meta['id'],q,c));assert sha256(path)==cache['files'][path]
        init=read(PARENT_OUT/'source_initialization'/f'{meta["id"]}.json')
        assert sha256(PARENT_CACHE/meta['id']/'anysplat_source_init.pt')==init['output_sha256']
        verified[meta['id']]=dict(input_manifest_sha256=sha256(manifest),current_patch_manifest_sha256=sha256(PARENT_OUT/'query_cache'/f'{meta["id"]}.json'))
    assert sha256(p['model_path'])==p['model_sha256']
    for path,digest in p['official_source_sha256'].items():
        assert sha256(ROOT/'third_party/streamvggt'/path)==digest
    result=dict(complete=True,records=verified,actual_native_weight_sha256=p['model_sha256'],
                parent_plan_sha256=sha256(PARENT_OUT/'plan.json'),all24_inputs_and192_current_patch_files_unchanged=True,
                scope='Exact identities inherited from completed parent experiment; not fresh confirmation.')
    write_json(OUT/'parent_input_model_identity.json',result);print('Parent inputs, native weights and official source identity verified.',flush=True)

def field_identity():
    p=plan();records=[]
    for meta in p['training_catalog']+p['catalog']:
        rid=meta['id'];source=load(source_path(rid));assert set(source)=={'rgb','c2w','K','source_ids','id','place','split','depth_pred','depth_sensor','common_valid','unit'}
        source_hash=sha256(source_path(rid));a=load(branch_cache('pred')/rid/'selected_indices.pt')['indices'];b=load(branch_cache('sensor')/rid/'selected_indices.pt')['indices'];assert torch.equal(a,b)
        for mode in MODES:
            r=read(branch_out(mode)/'construction'/f'{rid}.json');assert r['source_input_sha256']==source_hash and r['queries_used'] is False
            field=load(branch_cache(mode)/rid/'scene_memory.pt');binding=read(branch_out(mode)/'binding'/f'{rid}.json')
            assert all(sha256(path)==digest for path,digest in binding['files'].items())
            assert field['features'].shape==(2048,1024) and field['features'].dtype==torch.float16
            assert torch.equal(field['source_c2w'],source['c2w']) and torch.equal(field['source_intrinsic'],source['K'])
            cache=read(branch_out(mode)/'query_cache'/f'{rid}.json');assert cache['source_state_unchanged'] and not cache['query_cache_contains_history_prefix']
            assert all(sha256(path)==digest for path,digest in cache['files'].items())
            records.append(dict(id=rid,mode=mode,scene_sha256=sha256(branch_cache(mode)/rid/'scene_memory.pt'),source_only=True,paired_source_pixels_equal=True,content_valid=int(field['has_content'].sum())))
    lock=read(OUT/'configuration_lock.json')
    assert sha256(OUT/'plan.json')==lock['plan_sha256']
    for path,digest in lock['source_sha256'].items():assert sha256(ROOT/path)==digest,path
    write_json(OUT/'source_state_identity.json',dict(complete=True,records=records,fixed_scientific_sources_unchanged=True))
    print('All48 saved fields, paired pixels and query caches verified.',flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('kind',choices=['parent','fields']);a=p.parse_args()
    parent_identity() if a.kind=='parent' else field_identity()
