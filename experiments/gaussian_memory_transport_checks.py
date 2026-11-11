"""Bounded checks of paired learning, fixed states and independently loaded reads."""
import json

import numpy as np
import torch

from experiments.gaussian_memory_attention_train import write_json
from experiments.gaussian_memory_setup import sha256
from experiments.gaussian_memory_transport_data import OUT, PREVIOUS, SEEDS, REGIMES, KERNELS, READERS, checked, load_record, load_memory, make_model, model_path, name
from experiments.gaussian_memory_oracle_data import OLD as SURFACE


def main():
    plan=checked(True);torch.set_num_threads(4);runs=[]
    for c in plan['cases']:
        path=model_path(c);logs=[json.loads(s) for s in (path.parent/'training.jsonl').read_text().splitlines()]
        previous=[json.loads(s) for s in (SURFACE/f"local_{c['seed']}"/'training.jsonl').read_text().splitlines()]
        key=lambda r:(r['step'],r['episode'],r['slot'],r['sample_ids'])
        assert len(logs)==2000 and list(map(key,logs))==list(map(key,previous))
        assert np.isfinite([[r['loss'],r['gradient_norm']] for r in logs]).all()
        status=json.loads((path.parent/'status.json').read_text());assert status['complete'];runs.append(status)
        net=make_model(c,plan);assert not hasattr(net,'writer')
        assert sum(p.numel() for p in net.parameters())==plan['parameters'][c['reader']]
    lock=json.loads((OUT/'evaluation_lock.json').read_text())
    for path,digest in lock['models'].items():assert sha256(path)==digest
    states=json.loads((OUT/'memory_identity.json').read_text());assert states['states']==92
    for path,digest in states['sha256'].items():assert sha256(path)==digest
    state_checks=json.loads((OUT/'state_checks.json').read_text())['rows']
    state_error=max(r[k] for r in state_checks for k in ('means_max_difference','covars_max_difference'));assert state_error<2e-5
    training={m['row']['scene'].split('/')[0] for m in plan['catalog'] if m['id'] in plan['training_ids']}
    assert len(training)==12 and not training.intersection(plan['primary_places'])
    scores=json.loads((OUT/'evaluation/scores.json').read_text());previous=json.loads((PREVIOUS/'evaluation/scores.json').read_text())
    assert len(scores['rows'])==46*202 and len(scores['full_query'])==46*42
    old={(r['episode'],r['variant'],r['seed'],r['read_geometry']):r for r in previous['rows'] if r['condition']=='normal' and (r['variant']=='local' or r['seed'] is None)}
    replay=[]
    for r in scores['rows']:
        if r['condition']!='normal' or r['reader'] not in ('large_local','baseline'):continue
        old_reader='local' if r['reader']=='large_local' else 'seed_'+r['kernel']
        ref=old[(r['episode'],old_reader,r['seed'],r['regime'])]
        replay.append(max(abs(r[k]-ref[k]) for k in ('log_mse','hidden_log_mse','visible_log_mse','rmse_m','absrel')))
    assert max(replay)<1e-6
    invariants=[];independent=[];accounting=[]
    for meta in plan['catalog']:
        maps=torch.load(OUT/'evaluation'/(meta['id']+'_maps.pt'),map_location='cpu',weights_only=False)
        for g in REGIMES:
            for k in KERNELS:
                baseline=maps['predictions'][f'{g}_{k}_baseline_normal']
                for seed in SEEDS:
                    key=name(dict(regime=g,kernel=k,reader='prior_residual',seed=seed))
                    diff=float((maps['predictions'][key+'_normal']-maps['predictions'][key+'_shuffled']).abs().max());assert diff==0
                    prior_diff=float((maps['predictions'][key+'_residual_removed']-baseline).abs().max());assert prior_diff<2e-5
                    invariants.append({'episode':meta['id'],'case':key,'shuffle_difference':diff,'prior_replay_difference':prior_diff})
        for key,parts in maps['components'].items():
            diff=float((maps['predictions'][key+'_normal']-parts['geometry']-parts['residual']).abs().max());accounting.append(diff);assert diff<1e-6
        if meta['id'] not in ('fit_00','transfer_08','fidelity_07'):continue
        d=load_record(meta);rec,raw=d['rec'],d['raw'];sample=rec['roles'].index('evaluation');q=int(rec['query'][sample])
        for g in REGIMES:
            memory=load_memory(meta['id'],g)
            for k in KERNELS:
                for reader in READERS:
                    c=dict(regime=g,kernel=k,reader=reader,seed=SEEDS[0]);key=name(c);net=make_model(c,plan).eval()
                    net.load_state_dict(torch.load(model_path(c),map_location='cpu',weights_only=False)['model'])
                    with torch.inference_mode():p=net.read(rec['input_tokens'][sample:sample+1],memory,raw['c2w'][q:q+1],raw['K'][q:q+1],raw['diagnostics']['unit_meters'])[0]
                    difference=float((p-maps['predictions'][key+'_normal'][0]).abs().max());assert difference<2e-4,(c,difference)
                    pred=maps['predictions'][key+'_normal'].numpy().astype('float64');truth=maps['target'].numpy().astype('float64');valid=maps['valid'].numpy()
                    mse=float((((pred-truth)**2*valid).sum(-1)/valid.sum(-1)).mean())
                    row=next(r for r in scores['rows'] if r['episode']==meta['id'] and all(r[z]==c[z] for z in c) and r['condition']=='normal')
                    score_diff=abs(mse-row['log_mse']);assert score_diff<1e-6
                    independent.append({'episode':meta['id'],**c,'CPU_read_max_difference':difference,'numpy_score_difference':score_diff})
    result={'complete':True,'new_runs':runs,'source_sha256':sha256(__file__),'memory_frozen':True,'state_CPU_GPU_max_difference':state_error,
            'training_and_primary_places_disjoint':True,'all_training_samples_match_previous':True,'old_baseline_large_local_replay_max_difference':max(replay),
            'prior_invariants':invariants,'component_accounting_max_difference':max(accounting),'independent_reads_scores':independent}
    write_json(OUT/'checks.json',result)
    print(json.dumps({'checks_complete':True,'fits':len(runs),'old_replay':max(replay),'CPU_read_max_difference':max(r['CPU_read_max_difference'] for r in independent)}))


if __name__=='__main__':main()
