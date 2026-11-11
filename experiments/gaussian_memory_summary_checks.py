"""Bounded independent checks of pairing, frozen outputs and saved-state reads."""
import json

import numpy as np
import torch

from experiments.gaussian_memory_attention_train import write_json
from experiments.gaussian_memory_setup import sha256
from experiments.gaussian_memory_summary_data import OUT,OLD,SEEDS,KINDS,locked,load_record,model,path
from models.gaussian_memory_transport import read_inputs


def main():
    plan=locked();torch.set_num_threads(4)
    old_lock=json.loads((OLD/'evaluation_lock.json').read_text())
    for filename,digest in old_lock['models'].items():assert sha256(filename)==digest,filename
    for filename,digest in plan['input_sha256'].items():assert sha256(filename)==digest,filename
    fit_checks=[]
    for seed in SEEDS:
        directory=path('global',seed).parent;status=json.loads((directory/'status.json').read_text())
        assert status['complete'] and status['steps']==2000
        current=[json.loads(s) for s in (directory/'training.jsonl').read_text().splitlines()]
        previous=[json.loads(s) for s in (path('gaussian',seed).parent/'training.jsonl').read_text().splitlines()]
        assert len(current)==len(previous)==2000
        fields=('step','slot','episode','sample_ids')
        assert all(all(a[k]==b[k] for k in fields) for a,b in zip(current,previous))
        assert all(np.isfinite([r['loss'],r['gradient_norm']]).all() for r in current)
        initial=torch.load(path('global',seed,True),map_location='cpu',weights_only=False)['model']
        old_initial=torch.load(path('gaussian',seed,True),map_location='cpu',weights_only=False)['model']
        assert all(torch.equal(v,old_initial[k]) for k,v in initial.items())
        fit_checks.append({'seed':seed,'steps':2000,'sampling_exact':True,'initialization_exact':True,'all_finite':True})
    cache=json.loads((OUT/'cache_identity.json').read_text());assert len(cache['catalog'])==20
    for filename,digest in cache['sha256'].items():assert sha256(filename)==digest,filename
    new_places={m['row']['scene'].split('/')[0] for m in cache['catalog']}
    assert len(new_places)==20 and not new_places & set(plan['old_places'])
    assert all(set(m['row']['reference']).isdisjoint(m['row']['heldout_query']) for m in cache['catalog'])
    scores=json.loads((OUT/'evaluation/scores.json').read_text())
    assert len(scores['rows'])==66*42 and len(scores['full_query'])==66*15
    assert all(np.isfinite(r['log_mse']) for r in scores['rows']+scores['full_query'])
    old_scores=json.loads((OLD/'evaluation/scores.json').read_text());replay=[]
    for field in ('rows','full_query'):
        previous={}
        for r in old_scores[field]:
            kind=('large_local' if r['reader']=='large_local' else 'local' if r['reader']=='local' else
                  r['kernel'] if r['regime']=='predicted' and r['reader']=='latent' else None)
            if kind is not None:previous[(r['episode'],kind,r['seed'],r['condition'])]=r
        for r in scores[field]:
            if r['split']=='confirmation' or r['kind']=='global':continue
            key=(r['episode'],r['kind'],r['seed'],r['condition']);p=previous[key]
            for metric in ('log_mse','hidden_log_mse','visible_log_mse','rmse_m','absrel'):
                if p[metric] is not None:replay.append(abs(r[metric]-p[metric]))
    assert max(replay)<2e-5
    reads=[];numpy_scores=[];seed=SEEDS[0]
    catalog={m['id']:m for m in plan['catalog']+cache['catalog']}
    with torch.inference_mode():
        for rid in ('fit_00','confirm_00','confirm_15'):
            d=load_record(catalog[rid]);raw,rec=d['raw'],d['rec']
            keep=torch.tensor([r=='evaluation' for r in rec['roles']]);q=rec['query'][keep]
            saved=torch.load(OUT/'evaluation'/(rid+'_maps.pt'),map_location='cpu',weights_only=False)
            x=rec['input_tokens'][keep];camera,K,unit=raw['c2w'][q],raw['K'][q],raw['diagnostics']['unit_meters']
            inputs=read_inputs(d['memory'],camera,K,unit)
            for kind in KINDS:
                net=model(kind,plan).eval();net.load_state_dict(torch.load(path(kind,seed),map_location='cpu',weights_only=False)['model'])
                p=net.read(x,None,camera,K,unit) if kind=='large_local' else net(x,inputs)
                stored=saved['predictions'][f'{kind}_{seed}_normal'];difference=float((p-stored).abs().max());assert difference<2e-5
                reads.append({'episode':rid,'kind':kind,'seed':seed,'max_difference':difference})
                error=stored.numpy().astype('float64')-saved['target'].numpy().astype('float64');valid=saved['valid'].numpy()
                independent=float(np.mean([np.mean(e[v]**2) for e,v in zip(error,valid)]))
                row=next(r for r in scores['rows'] if r['episode']==rid and r['kind']==kind and r['seed']==seed and r['condition']=='normal')
                delta=abs(independent-row['log_mse']);assert delta<2e-6
                numpy_scores.append({'episode':rid,'kind':kind,'seed':seed,'difference':delta})
    permutations=max(r['joint_permutation_max_difference'] for r in scores['invariants'])
    assert permutations<2e-5
    accounting=max(r['accounting_max_diff'] for r in scores['rows']+scores['full_query'] if r['condition']=='normal')
    assert accounting<1e-10
    write_json(OUT/'checks.json',{'complete':True,'source_sha256':sha256(__file__),
        'old39_checkpoint_identities_unchanged':len(old_lock['models']),'new_fit_checks':fit_checks,
        'method_unseen_places':20,'old_new_places_disjoint':True,'reference_query_frames_disjoint':True,
        'old_prediction_metric_replay_max_difference':max(replay),'old_metric_values_checked':len(replay),
        'all_joint_permutations_max_difference':permutations,'decomposition_accounting_max_difference':accounting,
        'independent_CPU_reads':reads,'independent_numpy_scores':numpy_scores,
        'fresh_confirmation_scope':'New to this shared-memory route; depth previously used in old region registration.'})
    print(json.dumps({'checks_complete':True,'old_replay_max_difference':max(replay),'independent_CPU_max_difference':max(r['max_difference'] for r in reads),
                      'independent_numpy_max_difference':max(r['difference'] for r in numpy_scores)}),flush=True)


if __name__=='__main__':main()
