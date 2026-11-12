"""Matched updates, actual native changes, unchanged source scenes and scoped costs."""
import math
from experiments.native_receiver_data import *
from experiments import gaussian_scene_training_summary as timing


def main():
    torch.set_num_threads(4);p=plan();records=[];sources=[]
    timing.OUT=OUT;timing.KINDS=KINDS;timing.main()
    assert read(OUT/'training_summary.json')['complete']
    base=torch.load(p['model_path'],map_location='cpu',weights_only=True,mmap=True)
    if 'model' in base:base=base['model']
    expected={n for n in base if n.startswith(PREFIXES)}
    for seed in SEEDS:
        initial_reference=None;order=schedule(p,seed)
        for kind in KINDS:
            folder=OUT/f'{kind}_{seed}';initial=load(folder/'initial.pt');status=read(folder/'status.json')
            frozen=load(FROZEN/f'{kind}_{seed}'/'initial.pt')
            assert initial['fusion_parameters']==787712 and initial['native_parameters']>600000000
            assert initial['schedule_sha256']==frozen['schedule_sha256']
            assert all(torch.equal(v,frozen['model'][n]) for n,v in initial['model'].items())
            if initial_reference is None:initial_reference=initial['model']
            else:assert all(torch.equal(v,initial_reference[n]) for n,v in initial['model'].items())
            logs=[json.loads(l) for l in (folder/'training.jsonl').read_text().splitlines() if l]
            old_logs=[json.loads(l) for l in (FROZEN/f'{kind}_{seed}'/'training.jsonl').read_text().splitlines() if l]
            assert len(logs)==len(old_logs)==2000
            for index,(row,old) in enumerate(zip(logs,old_logs)):
                assert row['step']==index+1 and row['samples']==old['samples']==[list(x) for x in order[2*index:2*index+2]]
                assert math.isfinite(row['loss']) and math.isfinite(row['gradient_norm'])
            path=checkpoint_path(kind,seed);assert sha256(path)==status['checkpoint_sha256']
            state=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
            assert set(state['native'])==expected
            assert all(torch.isfinite(t).all() for t in state['native'].values())
            changes={}
            for group in ['frame_blocks','global_blocks']:
                changes[group]=[float((state['native'][n]-base[n]).norm())
                    for n in [f'aggregator.{group}.{i}.attn.qkv.weight' for i in range(24)]]
                assert all(v>0 for v in changes[group])
            head_keys=[n for n in expected if n.startswith('depth_head.') and state['native'][n].is_floating_point()]
            changed_head=[n for n in head_keys if not torch.equal(state['native'][n],base[n])]
            assert changed_head
            records.append(dict(kind=kind,seed=seed,actual_updates=2000,actual_samples_match_frozen=True,
                fusion_initial_matches_frozen_and_other_arms=True,native_parameters=initial['native_parameters'],
                native_key_set_exact=True,native_qkv_update_norms=changes,changed_depth_head_tensors=len(changed_head),
                final_path=str(path),final_sha256=status['checkpoint_sha256']))
            del state
    for meta in p['training_catalog']+p['catalog']:
        r=read(FROZEN/'features'/f'{meta["id"]}.json')
        assert sha256(r['source_path'])==r['source_sha256']
        assert all(sha256(f)==h for f,h in {**r['files'],**r['inputs']}.items())
        sources.append(dict(id=meta['id'],source_sha256=r['source_sha256'],unchanged=True))
    lock=read(OUT/'configuration_lock.json')
    assert lock['plan_sha256']==sha256(OUT/'plan.json') and lock['pretrained_sha256']==sha256(p['model_path'])
    assert all(sha256(ROOT/n)==h for n,h in lock['source_sha256'].items())
    assert all(sha256(ROOT/'third_party/streamvggt'/n)==h for n,h in p['official_source_sha256'].items())
    write_json(OUT/'matched_checks.json',dict(complete=True,records=records,source_fields=sources,
        scientific_code_identity_unchanged=True,official_source_matches_inherited_frozen=True,
        pretrained_sha256=lock['pretrained_sha256'],
        scope='Correct implementation and matched finite fits; no efficacy or fresh confirmation claim.'))
    cost=read(OUT/'cost.json');assert cost['complete'] and len(cost['queries'])==12
    observations={k:dict(cases=4,retained_samples=20,
        median_ms=float(np.median([t*1000 for r in cost['queries'] if r['kind']==k for t in r['seconds']])),
        range_ms=[float(x) for x in np.array([min(t for r in cost['queries'] if r['kind']==k for t in r['seconds']),max(t for r in cost['queries'] if r['kind']==k for t in r['seconds'])])*1000]) for k in KINDS}
    checkpoint_io=[]
    for seed in SEEDS:
        for kind in KINDS:
            file=OUT/f'{kind}_{seed}'/'checkpoints.jsonl'
            if file.exists():checkpoint_io.extend(json.loads(line) for line in file.read_text().splitlines() if line)
    write_json(OUT/'cost_description.json',dict(complete=True,raw_cost_sha256=sha256(OUT/'cost.json'),observations=observations,
        source_scene_tensor_bytes=p['source_state_bytes'],native_parameters=read(OUT/'preflight.json')['native_parameters'],
        cold_native_load_seconds=cost['cold_native_load_seconds'],
        adapted_weight_loading=cost['adapted_weight_loading'],
        measurement_peak_allocated_GiB=max(r['peak_allocated_GiB'] for r in cost['queries']),
        final_checkpoint_bytes=sum(checkpoint_path(k,s).stat().st_size for s in SEEDS for k in KINDS),
        successful_checkpoint_io_records=len(checkpoint_io),
        successful_checkpoint_io_seconds=sum(row['save_seconds'] for row in checkpoint_io),
        checkpoint_io_scope='Successful snapshot write elapsed; overlaps training-loop elapsed except terminal saves, so do not add as an independent cost. Loading, pauses, interrupted I/O and RAM migration are not included in logged optimization hours.',
        preflight_resume_checkpoint_bytes=read(OUT/'checkpoint_preflight.json')['bytes'],
        scopes={k:cost[k] for k in ['deployment_weights_scope','timing_scope','contention_scope']},
        training_summary_sha256=sha256(OUT/'training_summary.json'),
        inherited_construction=cost['inherited_construction'],inherited_construction_sha256=cost['inherited_construction_sha256']))
    print('Nine complete native updates, matched samples and24immutable source fields verified.',flush=True)


if __name__=='__main__':main()
