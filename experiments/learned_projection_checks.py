"""Matched initialization, actual training samples, and descriptive PE use."""
import hashlib
import json
import math

from experiments.learned_projection_data import *
from experiments.gaussian_scene_config import schedule
from experiments import gaussian_scene_training_summary as timing
from models.gaussian_scene_learned_projection import LearnedProjectionReader


def training_checks():
    p=plan();records=[]
    timing.OUT=OUT;timing.KINDS=KINDS;timing.main()
    assert read(OUT/'training_summary.json')['complete']
    for seed in SEEDS:
        base=load(PARENT_OUT/f'L_{seed}'/'initial.pt')['model']
        old=load(SOURCE_OUT/f'P_{seed}'/'initial.pt')['model'];reference=None
        order=schedule(p,seed);digest=hashlib.sha256(json.dumps(order).encode()).hexdigest()
        for kind in KINDS:
            folder=OUT/f'{kind}_{seed}';initial=load(folder/'initial.pt');state=initial['model']
            assert initial['schedule_sha256']==digest
            assert all(torch.equal(v,state[k]) for k,v in base.items())
            for k,v in old.items():
                assert torch.equal(v,state[k][:,:7] if k=='center_pe.0.weight' else state[k])
            assert torch.count_nonzero(state['center_pe.0.weight'][:,7])==0
            if reference is None:reference=state
            else:assert all(torch.equal(v,state[k]) for k,v in reference.items())
            rows=[json.loads(x) for x in (folder/'training.jsonl').read_text().splitlines() if x]
            assert len(rows)==2000 and [r['step'] for r in rows]==list(range(1,2001))
            for r in rows:
                assert r['samples']==[list(x) for x in order[(r['step']-1)*2:r['step']*2]]
                assert math.isfinite(r['loss']) and math.isfinite(r['gradient_norm'])
            final=load(folder/'step_2000.pt')['model']
            assert all(torch.isfinite(x).all() for x in final.values())
            if kind=='P':assert torch.count_nonzero(final['center_pe.0.weight'][:,7])==0
            records.append(dict(kind=kind,seed=seed,actual_updates=2000,common_original_L_exact=True,
                original_center_initialization_exact=True,all_three_initial_states_identical=True,
                actual_samples_paired=True,extra_column_final_norm=float(final['center_pe.0.weight'][:,7].norm())))
    source_records=[]
    for m in p['training_catalog']+p['catalog']:
        r=read(OUT/'geometry'/f'{m["id"]}.json')
        assert sha256(r['source_path'])==r['source_sha256']
        assert all(sha256(f)==h for f,h in r['files'].items())
        source_records.append(dict(id=m['id'],scene_sha256=r['source_sha256'],unchanged=True))
    lock=read(OUT/'configuration_lock.json')
    assert lock['plan_sha256']==sha256(OUT/'plan.json')
    assert all(sha256(ROOT/f)==h for f,h in lock['source_sha256'].items())
    write_json(OUT/'matched_checks.json',dict(complete=True,records=records,source_fields=source_records,
        scientific_code_identity_unchanged=True,scope='New matched fits and unchanged inherited saved states; no repeated integrator audit'))


@torch.no_grad()
def pe_description():
    """Fixed uniformly sampled pairs; descriptive bias use, not a task ablation."""
    p=plan();readers={};stats={}
    for kind in ('S','G'):
        for seed in SEEDS:
            reader=LearnedProjectionReader();reader.load_state_dict(load(OUT/f'{kind}_{seed}'/'step_2000.pt')['model'])
            readers[kind,seed]=reader.eval().center_pe
            for split in ('training','development'):
                stats[kind,seed,split]=dict(n=0,positive_t=0,sum_delta2=np.zeros(4),sum_base2=np.zeros(4),max_abs_delta=0.)
    indices=torch.from_numpy(np.random.default_rng(20261112).choice(1024*2048,4096,replace=False))
    for split,metas in [('training',p['training_catalog']),('development',p['catalog'])]:
        for meta in metas:
            for q in QUERIES:
                payload=load(geometry_path(meta['id'],q))
                center=payload['center'].reshape(-1,7)[indices].float()
                zero=torch.cat((center,torch.zeros(len(indices),1)),1)
                for kind in ('S','G'):
                    t=payload['sphere' if kind=='S' else 'full'].flatten()[indices].float()
                    x=torch.cat((center,t[:,None]),1)
                    for seed in SEEDS:
                        base=readers[kind,seed](zero);delta=readers[kind,seed](x)-base
                        s=stats[kind,seed,split];s['n']+=len(t);s['positive_t']+=int((t>0).sum())
                        s['sum_delta2']+=delta.square().sum(0).numpy();s['sum_base2']+=base.square().sum(0).numpy()
                        s['max_abs_delta']=max(s['max_abs_delta'],float(delta.abs().max()))
    rows=[]
    for (kind,seed,split),s in stats.items():
        rows.append(dict(kind=kind,seed=seed,split=split,sampled_pairs=s['n'],positive_projection_fraction=s['positive_t']/s['n'],
            projection_bias_delta_rms_per_head=np.sqrt(s['sum_delta2']/s['n']).tolist(),
            center_only_bias_rms_per_head=np.sqrt(s['sum_base2']/s['n']).tolist(),max_abs_projection_bias_delta=s['max_abs_delta']))
    write_json(OUT/'pe_description.json',dict(complete=True,rows=rows,
        rule='Same4096 pair indices from fixed seed20261112 uniform without replacement, each of all24scenes/four queries; same descriptor for full/missing',
        scope='CPU FP32 descriptive learned-bias perturbation with trained weights; not new native predictions, matched training ablation, deployment selector or efficacy evidence'))


def main():
    torch.set_num_threads(4);training_checks();pe_description();print('Matched fits and descriptive PE checks complete.',flush=True)


if __name__=='__main__':main()
