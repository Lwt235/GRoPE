"""Small training-only checks of the matched readout intervention."""
import json

import numpy as np
import torch

from experiments.gaussian_memory_attention_train import write_json
from experiments.gaussian_memory_surface_evaluate import baseline
from experiments.gaussian_memory_transport_data import OUT, REGIMES, KERNELS, READERS, SEEDS, checked, load_record, load_memory, make_model, model_path
from experiments.gaussian_memory_oracle_data import input_seeds
from models.gaussian_memory_transport import read_inputs


def main():
    plan=checked();assert torch.cuda.is_available();torch.set_num_threads(4);rows=[]
    for rid in ('fit_00','coverage_04'):
        meta=next(m for m in plan['catalog'] if m['id']==rid);d=load_record(meta)
        rec,raw=d['rec'],d['raw'];sample=rec['roles'].index('train');q=rec['query'][sample:sample+1]
        x=rec['input_tokens'][sample:sample+1].cuda();camera,K=raw['c2w'][q].cuda(),raw['K'][q].cuda();unit=raw['diagnostics']['unit_meters']
        for regime in REGIMES:
            memory=load_memory(rid,regime,'cuda');data=read_inputs(memory,camera,K,unit)
            for kernel in KERNELS:
                reference=baseline(input_seeds(d,regime),camera,K,unit,'seed_'+kernel,plan['center_variance'])
                for reader in READERS:
                    c=dict(regime=regime,kernel=kernel,reader=reader,seed=SEEDS[0]);net=make_model(c,plan).cuda()
                    net.load_state_dict(torch.load(model_path(c,True),map_location='cuda',weights_only=False)['model'])
                    pred=net(x,data);expected=reference if reader!='latent' else torch.zeros_like(reference)
                    diff=float((pred-expected).abs().max());assert diff<2e-5,(c,diff)
                    removed=net(x,data,'removed');assert torch.equal(removed,torch.zeros_like(removed))
                    empty=read_inputs(dict(memory,opacities=torch.zeros_like(memory['opacities'])),camera,K,unit)
                    assert torch.equal(net(x,empty),torch.zeros_like(pred))
                    if reader=='prior_residual':
                        shuffled=dict(data,key=data['key'].flip(1),value=data['value'].flip(1))
                        assert torch.equal(pred,net(x,shuffled))
                    loss=(pred-d['target']['log_depth_m'][q].cuda()).square().mean();loss.backward()
                    assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
                    key_gradient=float(net.key.weight.grad.norm()) if hasattr(net,'key') else None
                    if reader=='transport':assert key_gradient>0
                    rows.append({**c,'episode':rid,'initial_prior_max_difference':diff,'key_gradient':key_gradient,
                                 'empty_and_removed_finite_zero':True})
    initial_pairs=[]
    for seed in SEEDS:
        latent=next(c for c in plan['cases'] if c['seed']==seed and c['reader']=='latent')
        local=next(c for c in plan['cases'] if c['seed']==seed and c['reader']=='local')
        states={r:torch.load(model_path(c,True),map_location='cpu',weights_only=False)['model'] for r,c in [('latent',latent),('local',local)]}
        for c in plan['cases']:
            if c['seed']!=seed:continue
            family='local' if c['reader'] in ('local','prior_residual') else 'latent'
            state=torch.load(model_path(c,True),map_location='cpu',weights_only=False)['model']
            assert states[family].keys()==state.keys() and all(torch.equal(v,states[family][k]) for k,v in state.items())
            initial_pairs.append(c)
    write_json(OUT/'preflight.json',{'complete':True,'CUDA_device':torch.cuda.get_device_name(0),'training_only_examples':True,
                                   'rows':rows,'identical_family_initializations':initial_pairs,
                                   'max_initial_prior_difference':max(r['initial_prior_max_difference'] for r in rows)})
    print(json.dumps({'preflight_complete':True,'cases':len(rows),'max_prior_diff':max(r['initial_prior_max_difference'] for r in rows)}),flush=True)


if __name__=='__main__':main()
