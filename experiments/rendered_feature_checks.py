"""Matched initial states, actual samples and unchanged reused source fields."""
import hashlib
import json
import math

from experiments.rendered_feature_data import *
from experiments.gaussian_scene_config import schedule
from experiments import gaussian_scene_training_summary as timing


def main():
    torch.set_num_threads(4);p=plan();records=[];sources=[]
    timing.OUT=OUT;timing.KINDS=KINDS;timing.main()
    assert read(OUT/'training_summary.json')['complete']
    for seed in SEEDS:
        reference=None;order=schedule(p,seed)
        digest=hashlib.sha256(json.dumps(order).encode()).hexdigest()
        for kind in KINDS:
            folder=OUT/f'{kind}_{seed}';initial=load(folder/'initial.pt');state=initial['model']
            assert initial['schedule_sha256']==digest and initial['parameters']==787712
            if reference is None:reference=state
            else:assert all(torch.equal(v,state[k]) for k,v in reference.items())
            assert torch.count_nonzero(state['output.weight'])==0 and torch.count_nonzero(state['output.bias'])==0
            rows=[json.loads(line) for line in (folder/'training.jsonl').read_text().splitlines() if line]
            assert len(rows)==2000 and [r['step'] for r in rows]==list(range(1,2001))
            for r in rows:
                assert r['samples']==[list(x) for x in order[(r['step']-1)*2:r['step']*2]]
                assert math.isfinite(r['loss']) and math.isfinite(r['gradient_norm'])
            final=load(folder/'step_2000.pt')['model']
            assert all(torch.isfinite(x).all() for x in final.values())
            records.append(dict(kind=kind,seed=seed,actual_updates=2000,initial_state_exactly_matched=True,
                actual_samples_paired=True,parameters=787712,
                input_weight_change_norm=float((final['input.weight']-state['input.weight']).norm()),
                output_weight_norm=float(final['output.weight'].norm())))
    for meta in p['training_catalog']+p['catalog']:
        r=read(OUT/'features'/f'{meta["id"]}.json')
        assert sha256(r['source_path'])==r['source_sha256']
        assert all(sha256(f)==h for f,h in r['files'].items())
        assert all(sha256(f)==h for f,h in r['inputs'].items())
        for q in QUERIES:
            d=load(feature_path(meta['id'],q));a=d['valid_indices'];b=d['shuffled_source_indices']
            assert torch.equal(a.sort().values,b.sort().values)
            assert d['render'].shape==d['global'].shape==d['shuffle'].shape==(1024,1024)
        sources.append(dict(id=meta['id'],scene_sha256=r['source_sha256'],unchanged=True))
    lock=read(OUT/'configuration_lock.json')
    assert lock['plan_sha256']==sha256(OUT/'plan.json')
    assert all(sha256(ROOT/f)==h for f,h in lock['source_sha256'].items())
    assert sha256(p['model_path'])==p['model_sha256']
    write_json(OUT/'matched_checks.json',dict(complete=True,records=records,source_fields=sources,
        scientific_code_identity_unchanged=True,native_checkpoint_sha256=p['model_sha256'],
        scope='Matched finite training, saved content and actual inputs; not efficacy or fresh confirmation.'))
    print('Nine matched fits and24unchanged fields checked.',flush=True)


if __name__=='__main__':main()
