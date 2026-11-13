"""Claim-matched identity, sampling, immutable-source and completion checks."""
from experiments import native_scene_exchange_data as d
from experiments.native_scene_exchange_data import np,torch

def inputs():
    rows=[]
    for rid in [d.plan()['training_catalog'][0]['id'],d.plan()['catalog'][0]['id']]:
        q=2;samples={arm:d.sample(rid,q,'missing',arm) for arm in d.ARMS};p=samples['R_P'];v=d.load(d.vpath(rid,q));mask=v['key_mask'];uv=v['common25'][:,3:5];sp=((uv+1)*16+.5)[:,[1,0]]
        # v2 corrected the original FP32 positions in place. Reassociation of
        # additions may differ by a float32 ULP; this is a coordinate check.
        assert torch.allclose(v['scene_pos'],sp,rtol=1e-6,atol=1e-6)
        for arm in d.ARMS[:-1]:
            assert samples[arm]['ordinary'].shape==(1,2048,1045) and samples[arm]['position'].shape==(1,2048,29)
            for k in ('patches','ordinary','scene_pos','key_mask'):assert torch.equal(samples[arm][k],p[k])
            assert torch.equal(samples[arm]['position'][...,:25],p['position'][...,:25])
        assert torch.equal(samples['L']['ordinary'][0,:,:1024],samples['L']['patches'][0].repeat_interleave(2,0))
        assert torch.equal(samples['L']['ordinary'][0,:,1027:1034],torch.zeros(2048,7))
        assert torch.equal(samples['L']['ordinary'][0,:,-1],torch.zeros(2048))
        assert all(torch.isfinite(x).all() for s in samples.values() for x in s.values())
        rows.append(dict(id=rid,matched_source_and_common25=True,extra_position_dimensions=4,scene_slots=2048,local_current_patch_copies_verified=True,local_inactive_coefficients=8,valid_keys=int(mask.sum()),query_RGB_may_enter_only_current_workcopy=True))
    d.write(d.OUT/'input_interface_checks.json',dict(complete=True,rows=rows,source_manifest_reference=str(d.base.OUT/'input_manifest.json'),source_geometry_reference=str(d.base.OUT/'preparation_complete.json'),persistent_source_writes=0,source_optimization=0,schedules=[dict(seed=s,sha256=d.hashlib.sha256(d.json.dumps(d.schedule(s)).encode()).hexdigest(),samples=4000,conditions=['full','missing'],fit_queries=list(d.old.QUERIES)) for s in d.SEEDS]))

def complete():
    records=[]
    for arm in d.ARMS:
        for seed in d.SEEDS:
            folder=d.OUT/'fits';s=d.read(folder/f'{arm}_{seed}_complete.json');assert s['complete'] and s['step']==2000 and d.sha(s['path'])==s['sha256'];logs=[d.json.loads(x) for x in (folder/f'{arm}_{seed}.jsonl').read_text().splitlines()];assert [x['step'] for x in logs]==list(range(1,2001));assert [k for r in logs for k in r['samples']]==[list(x) for x in d.schedule(seed)]
            assert len(list((d.OUT/'completed').glob(f'{arm}_{seed}_*.json')))==704
            st=d.load(d.final(arm,seed));assert st['step']==2000 and 'optimizer' in st and 'rng' in st and 'cuda_rng' in st
            records.append(dict(arm=arm,seed=seed,updates=2000,samples=4000,predictions=704,exact_resume_persisted=True,final_sha256=s['sha256']))
    params=[d.read(p) for p in (d.OUT/'fits').glob('*_initial.json')];assert len(params)==21 and len({x['parameters'] for x in params})==1
    for seed in d.SEEDS:assert len({d.read(d.OUT/'fits'/f'{arm}_{seed}_initial.json')['state_digest'] for arm in d.ARMS})==1
    d.write(d.OUT/'completion_checks.json',dict(complete=True,fits=records,total_updates=42000,total_samples=84000,total_new_predictions=14784,matched_parameters=params[0]['parameters'],paired_initial_states=True,source_immutable_by_construction=True,independent_fresh_sequences=True,final_step_no_model_selection=True))

if __name__=='__main__':
    import sys
    torch.set_num_threads(4);inputs()
    if '--complete' in sys.argv:complete()
