"""Independent checkpoint-to-archive replay after the retrain experiment exits."""
from pathlib import Path
import json
import numpy as np
import torch
from experiments.round5_primitive_retrain import (
    CONDITIONS, SCENES, SEEDS, METHOD, component_dataset, state_error,
    model_probability, real_probability_metrics, sha256, write_json,
)
from models.gaussian_spectral_pe import GaussianSpectralMatcher

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)
root = Path('results/round5_primitive_retrain_raw')
manifest = json.loads((root/'manifest.json').read_text())
summary = json.loads((root/'summary.json').read_text())
protocol = json.loads((root/'frozen_protocol.json').read_text())
expected = {(seed, condition) for seed in SEEDS for condition in CONDITIONS}
assert {(row['seed'], row['condition']) for row in manifest['records']} == expected
assert len(manifest['records']) == 10 and 'elapsed_seconds' in manifest
assert sha256(root/'frozen_protocol.json') == manifest['protocol_sha256'] == summary['protocol_sha256']
for name, checksum in protocol['source_hashes'].items():
    assert sha256(name) == checksum == sha256(root/'source'/name)
for name, checksum in summary['source_data_sha256'].items():
    assert sha256(name) == checksum
models, checkpoint_rows = {}, []
for row in manifest['records']:
    assert sha256(row['checkpoint']) == row['checkpoint_sha256']
    payload = torch.load(row['checkpoint'], map_location='cpu', weights_only=True)
    model = GaussianSpectralMatcher(METHOD, 256, .25).eval()
    model.load_state_dict(payload['state_dict'], strict=True)
    models[row['seed'], row['condition']] = model
    if row['condition'] == 'full':
        previous = row['full_replication']
        assert sha256(previous['old_checkpoint']) == previous['old_checkpoint_sha256']
        old = torch.load(previous['old_checkpoint'], map_location='cpu', weights_only=True)
        assert state_error(payload['state_dict'], old['state_dict']) == {}
    checkpoint_rows.append({'seed': row['seed'], 'condition': row['condition'],
                            'hash_verified': True, 'full_bitwise_old_match': row['condition']=='full'})
replays, source_hashes = [], {}
maximum_normalization_error = 0.
for scene in SCENES:
    source = Path('results/round3_real_support_data_adjacent' if scene in SCENES[:2]
                  else 'results/round3_external_scenes')
    splits = ['test'] if scene in SCENES[:2] else ['train','validation','test']
    for alternate in [False, True]:
        data, _, _ = component_dataset(source, Path('results/round3_cov_decomposition'),
                                      scene, splits, alternate, source_hashes)
        for seed in SEEDS:
            path = root/'probabilities'/f'{seed}-{scene}-{int(alternate)}.npz'
            assert sha256(path) == summary['probability_sha256'][str(path)]
            with np.load(path, allow_pickle=False) as saved:
                for field, key in [('valid','label_valid'), ('soft_labels','soft_labels'), ('key_valid','key_valid')]:
                    np.testing.assert_array_equal(saved[field], data['full'][key].numpy())
                for condition in CONDITIONS:
                    probability = model_probability(models[seed, condition], data[condition])
                    np.testing.assert_array_equal(probability, saved[condition])
                    assert np.isfinite(probability).all() and (probability >= 0).all()
                    maximum_normalization_error = max(maximum_normalization_error,
                        float(np.abs(probability.sum(-1)-1).max()))
                    metrics, _ = real_probability_metrics(probability, data[condition])
                    record = next(row for row in summary['records'] if (row['seed'],row['scene'],row['alternate'],row['condition'])
                                  == (seed,scene,alternate,condition))
                    assert metrics == record['metrics']
            replays.append({'seed':seed,'scene':scene,'alternate':alternate,
                            'both_checkpoint_predictions_bitwise_equal':True,
                            'shared_labels_masks_equal':True,'metrics_exactly_equal':True})
assert len(replays) == 40
report = {'complete':True, 'checkpoints':checkpoint_rows, 'probability_archives':replays,
          'all_80_checkpoint_prediction_arrays_bitwise_equal':True,
          'maximum_float32_probability_row_sum_error':maximum_normalization_error,
          'source_hashes_verified':len(protocol['source_hashes']),
          'data_hashes_verified':len(summary['source_data_sha256']),
          'verifier_sha256':sha256(__file__), 'summary_sha256':sha256(root/'summary.json'),
          'tests':'tests/test_round5_primitive_retrain.py: 3 passed'}
write_json(root/'verification.json',report)
print(json.dumps({key:value for key,value in report.items() if key not in ['checkpoints','probability_archives']},indent=2))
