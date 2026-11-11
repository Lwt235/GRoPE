"""Focused checks of loss pairing, old-score replay and independent scores."""
import json

import numpy as np
import torch

from experiments.gaussian_memory_attention_train import write_json
from experiments.gaussian_memory_data import REPORT, CACHE
from experiments.gaussian_memory_setup import ROOT, sha256
from experiments.gaussian_memory_fidelity_data import OUT, OLD_OUT, TOKEN_CACHE, REGIMES, SEEDS, VARIANTS, model_path, assert_lock
from models.gaussian_memory_attention import geometry_inputs
from models.gaussian_memory_view_token import ViewTokenAdapter


def norm(x):
    return (x-x.mean(-1,keepdims=True)) / np.sqrt(x.var(-1,keepdims=True)+1e-5)


def main():
    assert_lock(); torch.set_num_threads(4)
    runs = []
    for seed in SEEDS:
        reference = None
        for regime in REGIMES:
            for variant in VARIANTS:
                path = model_path(regime, variant, seed); directory = path.parent
                log = [json.loads(x) for x in (directory / 'training.jsonl').read_text().splitlines()]
                assert len(log) == 2000 and [r['step'] for r in log] == list(range(1,2001))
                assert all(np.isfinite([r['loss'], r['gradient_norm']]).all() for r in log)
                sequence = [(r['episode'],r['sample_ids']) for r in log]
                if reference is None: reference = sequence
                else: assert sequence == reference
                status = json.loads((directory / 'status.json').read_text()); assert status['complete']
                if regime != 'A':
                    base = torch.load(model_path('A', variant, seed, True), map_location='cpu', weights_only=False)
                    initial = torch.load(model_path(regime, variant, seed, True), map_location='cpu', weights_only=False)
                    assert all(torch.equal(v, initial['model'][k]) for k,v in base['model'].items())
                    if regime == 'C':
                        assert log[0]['clean_identity'] == 0.
                        assert all(abs(r['loss'] - r['all_patch'] - r['clean_identity']) < 1e-7 for r in log)
                    else: assert all(r['loss'] == r['all_patch'] for r in log)
                runs.append(dict(regime=regime, variant=variant, seed=seed, **status))
    old = json.loads((OLD_OUT / 'development_evaluation/scores.json').read_text())
    new = json.loads((OUT / 'development_evaluation/scores.json').read_text())
    def key(r): return (r['episode'],r['variant'],r['seed'],r['condition'])
    old_index = {key(r):r for r in old['rows'] if r['variant'] in VARIANTS}
    replay = max(abs(r[m]-old_index[key(r)][m]) for r in new['rows'] if r['regime']=='A'
                 for m in ('hidden_mse','visible_mse'))
    assert replay < 1e-6
    raw = torch.load(CACHE / 'val_12.pt', map_location='cpu', weights_only=False)
    rec = torch.load(TOKEN_CACHE / 'val_12.pt', map_location='cpu', weights_only=False)
    memory = torch.load(REPORT / 'learned_geometry/memories/val_12.pt', map_location='cpu', weights_only=False)['memory']
    q = int(rec['query'][0]); x = rec['input_tokens'][:1]; y = rec['teacher_tokens'][q:q+1]
    g = geometry_inputs(memory, raw['c2w'][q:q+1], raw['K'][q:q+1])
    maps = torch.load(OUT / 'development_evaluation/val_12_maps.pt', map_location='cpu', weights_only=False)
    independent = {}
    with torch.inference_mode():
        for regime in ('B','C'):
            for variant in VARIANTS:
                ckpt = torch.load(model_path(regime, variant, SEEDS[0]), map_location='cpu', weights_only=False)
                model = ViewTokenAdapter(variant, ckpt['config']['center_variance']).eval()
                model.load_state_dict(ckpt['model'])
                prediction = model(x,g)
                model(y,g)  # An independent complete-input forward cannot change the masked result.
                assert torch.equal(prediction,model(x,g))
                error = ((norm(prediction.numpy()[0])-norm(y.float().numpy()[0]))**2).mean(-1)
                name = regime+'_'+variant
                delta = float(np.max(np.abs(error-maps['errors'][name][0].numpy())))
                assert delta < 1e-5
                independent[name] = delta
                if name == 'C_center': cached_C_prediction = prediction.numpy()[0]
    all_patch_delta = 0.
    for name in ('development_evaluation','transfer_evaluation'):
        result = json.loads((OUT / name / 'scores.json').read_text())
        all_patch_delta = max(all_patch_delta, max(abs(r['all_patch_mse']-(70*r['hidden_mse']+230*r['visible_mse'])/300) for r in result['rows']))
    assert all_patch_delta < 1e-7
    with np.load(OUT / 'standalone_C.npz') as a:
        assert np.isfinite(a['tokens']).all() and np.isfinite(a['depth_modified']).all()
        live_difference = float(((norm(a['tokens'])-norm(cached_C_prediction))**2).mean())
        live_error = ((norm(a['tokens'])-norm(y.float().numpy()[0]))**2).mean(-1)
        depth_change = float(np.abs(a['depth_modified']-a['depth_base']).mean())
    result = {'complete': True, 'runs': runs, 'initial_parameters_equal_to_A': True,
              'all_2000_masked_samples_paired': True, 'C_clean_identity_initial_zero': True,
              'loss_formulas_checked_all_steps': True, 'clean_forward_cannot_modify_masked_prediction': True,
              'A_replay_max_absolute_difference': replay, 'numpy_score_max_differences': independent,
              'all_patch_accounting_max_difference': all_patch_delta,
              'standalone_C_vs_cache_normalized_mse': live_difference,
              'standalone_C_hidden_teacher_mse': float(live_error[rec['mask'][0].numpy()].mean()),
              'standalone_depth_finite': True, 'standalone_depth_change': depth_change,
              'depth_accuracy_claim': False,
              'extra_C_compute_reported': True,
              'auxiliary_source_hashes': {p:sha256(ROOT / p) for p in
                ['experiments/gaussian_memory_fidelity_report.py','experiments/gaussian_memory_fidelity_checks.py',
                 'experiments/gaussian_memory_view_read.py']}}
    write_json(OUT / 'completion_checks.json', result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('runs','auxiliary_source_hashes')}), flush=True)


if __name__ == '__main__': main()
