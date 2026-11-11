"""Proportional final checks, including an independent NumPy score replay."""
import json
import unittest

import numpy as np
import torch

from experiments.gaussian_memory_attention_train import write_json
from experiments.gaussian_memory_data import REPORT, CACHE
from experiments.gaussian_memory_setup import sha256
from experiments.gaussian_memory_view_data import OUT, TOKEN_CACHE, SEEDS, VARIANTS, checked
from models.gaussian_memory_attention import geometry_inputs
from models.gaussian_memory_view_token import ViewTokenAdapter


def norm(x):
    return (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + 1e-5)


def main():
    checked(); torch.set_num_threads(4)
    suite = unittest.defaultTestLoader.loadTestsFromName('tests.test_gaussian_memory_view_token')
    test_result = unittest.TextTestRunner(verbosity=1).run(suite)
    assert test_result.wasSuccessful()
    runs, total_seconds, maximum = [], 0., 0.
    for seed in SEEDS:
        reference = None
        for variant in VARIANTS:
            directory = OUT / f'{variant}_{seed}'
            log = [json.loads(x) for x in (directory / 'training.jsonl').read_text().splitlines()]
            assert len(log) == 2000 and [r['step'] for r in log] == list(range(1, 2001))
            assert all(np.isfinite([r['loss'], r['gradient_norm']]).all() for r in log)
            sequence = [(r['episode'], r['sample_ids']) for r in log]
            if reference is None: reference = sequence
            else: assert sequence == reference
            status = json.loads((directory / 'status.json').read_text()); assert status['complete']
            total_seconds += status['seconds']; maximum = max(maximum, status['peak_GiB'])
            runs.append({'run': directory.name, 'checkpoint_sha256': sha256(directory / 'step_2000.pt')})
    raw = torch.load(CACHE / 'val_12.pt', weights_only=False)
    rec = torch.load(TOKEN_CACHE / 'val_12.pt', weights_only=False); q = int(rec['query'][0])
    memory = torch.load(REPORT / 'learned_geometry/memories/val_12.pt', weights_only=False)['memory']
    ckpt = torch.load(OUT / f'center_{SEEDS[0]}/step_2000.pt', weights_only=False, map_location='cpu')
    net = ViewTokenAdapter('center').eval(); net.load_state_dict(ckpt['model'])
    geometry = geometry_inputs(memory, raw['c2w'][q:q + 1], raw['K'][q:q + 1])
    with torch.inference_mode(): prediction = net(rec['input_tokens'][:1], geometry).numpy()[0]
    teacher = rec['teacher_tokens'][q].float().numpy()
    np_error = ((norm(prediction) - norm(teacher)) ** 2).mean(-1)
    maps = torch.load(OUT / 'development_evaluation/val_12_maps.pt', weights_only=False)
    score_difference = float(np.max(np.abs(np_error - maps['errors']['center'][0].numpy())))
    assert score_difference < 1e-5
    with np.load(OUT / 'standalone/output.npz') as a:
        live_original = a['original_tokens']; live_adapted = a['tokens']
        live_input_difference = float(((norm(live_original) - norm(rec['input_tokens'][0].float().numpy())) ** 2).mean())
        live_adapted_difference = float(((norm(live_adapted) - norm(prediction)) ** 2).mean())
        live_hidden_mse = float(((norm(live_adapted) - norm(teacher)) ** 2).mean(-1)[rec['mask'][0].numpy()].mean())
        depth_change = float(np.abs(a['depth_modified'] - a['depth_base']).mean())
        assert np.isfinite(a['depth_modified']).all() and depth_change > 0
    record = {'unit_tests': test_result.testsRun, 'paired_training_sequence_equal': True,
              'runs': runs, 'training_seconds_total': total_seconds, 'training_peak_GiB': maximum,
              'independent_numpy_score_max_absolute_difference': score_difference,
              'standalone_backbone_vs_half_cache_normalized_mse': live_input_difference,
              'standalone_adapter_vs_cached_input_normalized_mse': live_adapted_difference,
              'standalone_hidden_teacher_mse': live_hidden_mse,
              'cached_hidden_teacher_mse': float(np_error[rec['mask'][0].numpy()].mean()),
              'existing_depth_head_finite': True, 'depth_change_mean_absolute': depth_change,
              'depth_utility_claim': False, 'standalone_input_contains_teacher': False}
    # Include auxiliary source identities without changing the pre-training plan.
    from experiments.gaussian_memory_setup import ROOT
    record['source_hashes'] = {str(ROOT / p): sha256(ROOT / p) for p in
        ['experiments/gaussian_memory_view_checks.py', 'experiments/gaussian_memory_view_read.py',
         'experiments/gaussian_memory_view_report.py', 'tests/test_gaussian_memory_view_token.py']}
    write_json(OUT / 'completion_checks.json', record); print(json.dumps(record), flush=True)


if __name__ == '__main__': main()
