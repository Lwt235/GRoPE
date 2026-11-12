"""Finite export/checkpoint/pairing coverage and empty-memory reference identity."""
import numpy as np
import torch
from experiments.source_geometry_data import *
from experiments.source_geometry_training_summary import main as summarize

def main():
    torch.set_num_threads(4);summarize();assert read(OUT/'training_summary.json')['complete']
    lock=read(OUT/'configuration_lock.json')
    assert sha256(OUT/'plan.json')==lock['plan_sha256']
    for name,value in lock['source_sha256'].items():assert sha256(ROOT/name)==value,name
    for mode in MODES:assert sha256(branch_out(mode)/'plan.json')==lock['branch_plan_sha256'][mode]
    for r in read(OUT/'source_state_identity.json')['records']:
        assert sha256(branch_cache(r['mode'])/r['id']/'scene_memory.pt')==r['scene_sha256']
    native=load(read(PARENT_OUT/'references/native.json')['prediction_path'])
    reference={tuple(k[:3]):d for k,d in zip(native['keys'],native['depth_metric']) if k[3]=='native_current'}
    exports=[];empty=[];expected={(m['id'],q,c) for m in plan()['catalog'] for q in QUERIES for c in CONDITIONS}
    for mode in MODES:
        for seed in SEEDS:
            for kind in KINDS:
                folder=branch_out(mode)/f'{kind}_{seed}';status=read(folder/'status.json')
                files=[branch_out(mode)/'evaluation'/f'{kind}_{seed}.json']
                if kind=='G':files.append(branch_out(mode)/'interventions'/f'G_{seed}.json')
                for marker in files:
                    r=read(marker);assert r['plan_sha256']==sha256(branch_out(mode)/'plan.json') and r['checkpoint_sha256']==status['checkpoint_sha256']==sha256(folder/'step_2000.pt')
                    assert sha256(r['prediction_path'])==r['prediction_sha256']
                    x=load(r['prediction_path']);keys={tuple(k) for k in x['keys']}
                    interventions=marker.parent.name=='interventions'
                    full={(*k,i) for k in expected for i in plan()['interventions']['kinds']} if interventions else expected
                    assert keys==full and len(x['keys'])==len(full) and x['depth_metric'].shape==(len(full),210,210)
                    assert torch.isfinite(x['depth_metric']).all() and (x['depth_metric']>0).all()
                    exports.append(dict(mode=mode,kind=kind,seed=seed,manifest=str(marker.relative_to(OUT)),predictions=len(full),keys_exact=True))
                    if interventions:
                        diffs=[float((d-reference[tuple(k[:3])]).abs().max()) for k,d in zip(x['keys'],x['depth_metric']) if k[3]=='remove_memory']
                        maximum=max(diffs);assert maximum==0.,(mode,seed,maximum)
                        empty.append(dict(mode=mode,seed=seed,predictions=len(diffs),max_abs_depth_difference=maximum))
    result=dict(complete=True,source_fields_verified=48,scientific_source_lock_unchanged=True,exports=exports,
                new_predictions=sum(r['predictions'] for r in exports),empty_memory_records=empty,empty_memory_native_identity_predictions=sum(r['predictions'] for r in empty),
                scope='Finite implementation/pairing/export coverage only; no effectiveness claim or new numerical efficacy gate.')
    assert result['new_predictions']==2880 and result['empty_memory_native_identity_predictions']==576
    write_json(OUT/'final_runtime_integrity.json',result);print(dict(complete=True,new_predictions=2880,empty_native_identities=576),flush=True)

if __name__=='__main__':main()
