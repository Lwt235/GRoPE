"""Focused pairing, old-score replay and independent CPU/NumPy score checks."""
import json

import numpy as np
import torch

from experiments.gaussian_memory_setup import ROOT, sha256
from experiments.gaussian_memory_attention_train import write_json
from experiments.gaussian_memory_direct_data import OUT, FID_OUT, SEEDS, VARIANTS, REF_CACHE, model_path, checked_cache, load_record
from models.gaussian_memory_attention import geometry_inputs
from models.gaussian_memory_direct_reference import DirectReferenceAdapter
from models.gaussian_memory_view_token import ViewTokenAdapter


def norm(x):
    return (x-x.mean(-1,keepdims=True))/np.sqrt(x.var(-1,keepdims=True)+1e-5)


def main():
    plan=checked_cache(); torch.set_num_threads(4)
    lock=json.loads((OUT / 'evaluation_lock.json').read_text())
    for p,h in lock['models'].items(): assert sha256(p)==h,p
    runs=[]
    for seed in SEEDS:
        reference=None
        for variant in VARIANTS:
            directory=model_path(variant,seed).parent
            log=[json.loads(x) for x in (directory / 'training.jsonl').read_text().splitlines()]
            assert len(log)==2000 and [r['step'] for r in log]==list(range(1,2001))
            assert all(np.isfinite([r['loss'],r['gradient_norm']]).all() for r in log)
            sequence=[(r['episode'],r['sample_ids']) for r in log]
            if reference is None: reference=sequence
            else: assert sequence==reference
            status=json.loads((directory / 'status.json').read_text()); assert status['complete']
            runs.append(dict(variant=variant,seed=seed,**status))
        old=torch.load(model_path('center',seed,True),map_location='cpu',weights_only=False)['model']
        direct=torch.load(model_path('direct',seed,True),map_location='cpu',weights_only=False)['model']
        for key in old:
            if key not in ('key.weight','value.weight'): assert torch.equal(old[key],direct[key]),key
        assert torch.equal(old['key.weight'][:,-10:],direct['key.weight'][:,-10:])
        for variant in ('wide','direct'):
            initial=torch.load(model_path(variant,seed,True),map_location='cpu',weights_only=False)['model']
            assert torch.count_nonzero(initial['output.weight'])==0 and torch.count_nonzero(initial['output.bias'])==0
    scores=json.loads((OUT / 'evaluation/scores.json').read_text())
    old_rows=[]
    for part in ('development_evaluation','transfer_evaluation'):
        old_rows += [r for r in json.loads((FID_OUT / part / 'scores.json').read_text())['rows'] if r['regime']=='B']
    def key(r):return (r['episode'],r['variant'],r['seed'],r['condition'])
    old_index={key(r):r for r in old_rows}
    replay=max(abs(r[m]-old_index[key(r)][m]) for r in scores['rows'] if key(r) in old_index
               for m in ('hidden_mse','visible_mse','all_patch_mse'))
    assert replay<1e-6
    d=load_record(next(r for r in plan['catalog'] if r['id']=='val_12'))
    rec,raw=d['rec'],d['raw']; q=int(rec['query'][0]); x=rec['input_tokens'][:1]
    teacher=rec['teacher_tokens'][q].float().numpy()
    maps=torch.load(OUT / 'evaluation/val_12_maps.pt',map_location='cpu',weights_only=False)
    independent={}
    with torch.inference_mode():
        for variant in VARIANTS:
            cls=ViewTokenAdapter if variant in ('local','center') else DirectReferenceAdapter
            ckpt=torch.load(model_path(variant,SEEDS[0]),map_location='cpu',weights_only=False)
            net=cls(variant,ckpt['config']['center_variance']).eval(); net.load_state_dict(ckpt['model'])
            memory=d['direct_memory'] if variant=='direct' else d['memory']
            g=geometry_inputs(memory,raw['c2w'][q:q+1],raw['K'][q:q+1])
            prediction=net(x,g).numpy()[0]
            error=((norm(prediction)-norm(teacher))**2).mean(-1)
            delta=float(np.max(np.abs(error-maps['errors'][variant][0].numpy())))
            assert delta<1e-5; independent[variant]=delta
            if variant=='direct': direct_prediction=prediction; direct_error=error
    accounting=max(abs(r['all_patch_mse']-(70*r['hidden_mse']+230*r['visible_mse'])/300) for r in scores['rows'])
    assert accounting<1e-7
    with np.load(OUT / 'standalone.npz') as a:
        assert np.isfinite(a['tokens']).all() and np.isfinite(a['depth_modified']).all()
        live_difference=float(((norm(a['tokens'])-norm(direct_prediction))**2).mean())
        live_error=((norm(a['tokens'])-norm(teacher))**2).mean(-1)
        depth_change=float(np.abs(a['depth_modified']-a['depth_base']).mean())
    identity=json.loads((OUT / 'reference_cache_identity.json').read_text())
    for row in identity['records']:
        saved=torch.load(REF_CACHE / (row['id']+'.pt'),map_location='cpu',weights_only=False)
        assert not saved['query_images_used'] and not saved['teacher_used']
        assert saved['reference_frames']==[0,4,8,12]
        assert saved['memory']['features'].shape==(1200,2048)
        assert row['valid_slots']>=256
    result={'complete':True,'runs':runs,'all_2000_samples_paired':True,'direct_common_initialization_equal':True,
            'new_outputs_initially_zero':True,'B_score_replay_max_absolute_difference':replay,
            'independent_numpy_score_max_differences':independent,'all_patch_accounting_difference':accounting,
            'reference_only_input_boundary_checked':True,'reference_records':42,
            'standalone_vs_cached_input_normalized_mse':live_difference,
            'standalone_hidden_mse':float(live_error[rec['mask'][0].numpy()].mean()),
            'cached_hidden_mse':float(direct_error[rec['mask'][0].numpy()].mean()),
            'existing_depth_head_finite':True,'depth_change_mean_absolute':depth_change,'depth_accuracy_claim':False,
            'auxiliary_source_sha256':{p:sha256(ROOT / p) for p in ['experiments/gaussian_memory_direct_read.py',
              'experiments/gaussian_memory_direct_run.py','experiments/gaussian_memory_direct_cost.py',
              'experiments/gaussian_memory_direct_report.py','experiments/gaussian_memory_direct_checks.py']}}
    write_json(OUT / 'completion_checks.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('runs','auxiliary_source_sha256')}),flush=True)


if __name__=='__main__': main()
