import numpy as np
import pytest
import torch
import json

from experiments.round12_depth_analysis import (ROLES, development_gate, expected_batch_hash,
                                               raw_metrics, check_close)
from experiments.round7_depth_train import metric_rows, macro


def test_numpy_raw_metrics_match_original_on_missing_and_boundary_targets():
    rng = np.random.default_rng(9)
    target = rng.uniform(.5, 4, (4, 432)).astype('float32')
    prediction = rng.uniform(.5, 5, (4, 432)).astype('float32')
    valid = rng.random(target.shape) > .2; boundary = rng.random(target.shape) > .8
    target[~valid] = np.nan
    ids = [dict(scene='a' if i < 2 else 'b', q_index=i, k_index=i+5, pair_id=f'{i}:{i+5}') for i in range(4)]
    reference = raw_metrics(prediction, target, valid, boundary, ids)
    rows = metric_rows(torch.from_numpy(prediction), dict(target=torch.from_numpy(target),
        valid=torch.from_numpy(valid), boundary=torch.from_numpy(boundary)), ids)
    assert check_close(reference, dict(rows=rows, summary=macro(rows))) < 1e-10
    assert reference['summary']['total_count'] == valid.sum()


def fixture_metrics():
    values = {role: {'AbsRel': np.full((5, 2), .2), 'RMSE': np.full((5, 2), .6)} for role in ROLES}
    values['full4']['AbsRel'][:] = .194
    values['singleview']['AbsRel'][:] = .195
    values['quad16']['AbsRel'][:] = .193
    times = {role: 10. for role in ROLES}; times['full4'] = 4.
    return values, times


def test_gate_preserves_point_and_singleview_utility_with_efficiency_branch():
    values, times = fixture_metrics(); gate = development_gate(values, times)
    assert gate['passed']
    assert gate['point_control'] == 'point'
    assert gate['compression_control'] == 'quad16'
    assert gate['efficiency_branch'] and not gate['accuracy_branch']
    assert gate['versus_singleview']['relative_change'] > -.02


def test_cannot_select_slow_quad_when_full1_is_stronger_and_faster():
    values, times = fixture_metrics()
    values['full1']['AbsRel'][:] = .1929; times['full1'] = 2.
    gate = development_gate(values, times)
    assert gate['compression_control'] == 'full1'
    assert not gate['passed'] and not gate['conditions']['compression_accuracy_or_efficiency']


@pytest.mark.parametrize('condition', ['point', 'singleview', 'scene', 'seed', 'rmse', 'timing'])
def test_each_substantive_guard_can_fail(condition):
    values, times = fixture_metrics()
    if condition == 'point':
        values['point']['AbsRel'][:] = .195
    elif condition == 'singleview':
        values['singleview']['AbsRel'][:] = .19
    elif condition == 'scene':
        values['singleview']['AbsRel'][:, 0] = .193
    elif condition == 'seed':
        values['singleview']['AbsRel'][:2] = .193
    elif condition == 'rmse':
        values['full4']['RMSE'][:] = .62
    elif condition == 'timing':
        times['full4'] = 7.
    assert not development_gate(values, times)['passed']


def test_accuracy_branch_requires_both_gain_and_cost_cap():
    values, times = fixture_metrics(); values['full4']['AbsRel'][:] = .18
    times['full4'] = 12.
    gate = development_gate(values, times)
    assert gate['passed'] and gate['accuracy_branch'] and not gate['efficiency_branch']
    times['full4'] = 13.
    assert not development_gate(values, times)['passed']


def test_batch_digest_is_seed_specific_and_reproducible():
    assert expected_batch_hash(11) == expected_batch_hash(11)
    assert expected_batch_hash(11) != expected_batch_hash(22)


def test_raw_metric_rejects_nonfinite_predictions_even_without_sensor_target():
    with pytest.raises(ValueError):
        raw_metrics(np.array([[np.nan]]), np.array([[0.]]), np.array([[False]]), np.array([[False]]),
                    [dict(scene='a', q_index=1, k_index=2, pair_id='1:2')])


def test_complete_analysis_replays_all_synthetic_runs_and_rejects_wrong_timed_checkpoint(tmp_path):
    """Exercise artifact joins without training or touching research results."""
    from experiments import round12_depth_analysis as audit
    from tests.test_round12_depth_train import stress_bundle
    from experiments.round12_depth_train import stress_authorization
    runs = tmp_path/'runs'; runs.mkdir()
    stress, stress_audit = stress_bundle(tmp_path)
    promotion = stress_authorization(stress, stress_audit)
    original = {'train': [], 'val': []}; identities = []; targets = []; validity = []; boundaries = []
    for split, scenes in (('train', ['t1', 't2', 't3']), ('val', ['a', 'b'])):
        for scene in scenes:
            folder = tmp_path/'data'/scene; folder.mkdir(parents=True)
            count = 192 if split == 'train' else 64
            pairs = [[i, i+5] for i in range(count)]
            (folder/'manifest.json').write_text('{}')
            np.savez_compressed(folder/'inputs.npz', q_depth=np.full((count, 432), 1.3, dtype=np.float32))
            target = np.ones((count, 432), dtype=np.float32); valid = np.ones_like(target, dtype=bool)
            boundary = np.zeros_like(valid); boundary[:, :3] = True
            np.savez_compressed(folder/'labels.npz', target=target, valid=valid, boundary=boundary)
            original[split].append(dict(scene=scene, path=str(folder), pairs=pairs,
                manifest_sha256=audit.sha(folder/'manifest.json'), input_sha256=audit.sha(folder/'inputs.npz'),
                label_sha256=audit.sha(folder/'labels.npz')))
            if split == 'val':
                targets.append(target); validity.append(valid); boundaries.append(boundary)
                identities.extend(dict(scene=scene, q_index=q, k_index=k, pair_id=f'{q}:{k}') for q, k in pairs)
    lock = dict(roles=list(ROLES), seeds=list(audit.SEEDS), source_hashes={}, test_access_authorized=False,
                old_data=original, stress_promotion=promotion,
                geometry={'configurations': {role: {'temperature': 0. if role == 'full1' else 5.}
                                            for role in ROLES if role != 'singleview'}})
    (runs/'lock.json').write_text(json.dumps(lock)); lock_sha = audit.sha(runs/'lock.json')
    (runs/'lock.sha256').write_text(lock_sha)
    target = np.concatenate(targets); valid = np.concatenate(validity); boundary = np.concatenate(boundaries)
    errors = {role: .2 for role in ROLES}; errors.update(full4=.194, singleview=.195, quad16=.193)
    time_rows = []; run_hashes = {}; expected_hashes = {seed: expected_batch_hash(seed) for seed in audit.SEEDS}
    for role in ROLES:
        for seed in audit.SEEDS:
            folder = runs/role/str(seed); folder.mkdir(parents=True)
            pred = np.full_like(target, 1.+errors[role]); arrays = dict(depth=pred, zero_v=pred+.001, reverse_v=pred-.001)
            np.savez_compressed(folder/'val_predictions.npz', **arrays)
            actual = raw_metrics(pred, target, valid, boundary, identities)
            (folder/'validation.json').write_text(json.dumps(actual))
            interventions = {mode: dict(**raw_metrics(arrays[mode], target, valid, boundary, identities),
                mean_absolute_output_change_m=float(np.abs(arrays[mode]-pred).mean())) for mode in ('zero_v', 'reverse_v')}
            (folder/'interventions.json').write_text(json.dumps(interventions)); (folder/'best.pt').write_bytes(b'synthetic unit-test checkpoint')
            gradients = {name: {'present': True, 'finite': True, 'l1': 1.} for name in audit.PARAMETER_NAMES}
            temperature = 0. if role == 'full1' else 5.
            if role == 'full1':
                gradients['qk.weight']['l1'] = 0.
            run = dict(role=role, seed=seed, steps=1000, parameter_count=41345, lock_sha256=lock_sha,
                source_hashes={}, temperature=temperature, initialization_sha256=f'synthetic-seed-{seed}',
                batch_sequence_sha256=expected_hashes[seed], best_step=0,
                best_validation_AbsRel=actual['summary']['metrics']['AbsRel'],
                history=[dict(step=step, validation=actual['summary']) for step in range(0, 1001, 100)],
                training_losses=[.1]*1000, first_step_gradients=gradients, seconds=1.,
                checkpoint_sha256=audit.sha(folder/'best.pt'), prediction_sha256=audit.sha(folder/'val_predictions.npz'),
                validation_sha256=audit.sha(folder/'validation.json'), interventions_sha256=audit.sha(folder/'interventions.json'))
            (folder/'run.json').write_text(json.dumps(run)); run_hashes[str(folder/'run.json')] = audit.sha(folder/'run.json')
            elapsed = 4. if role == 'full4' else 10.
            for scene in ('t1', 't2', 't3', 'a', 'b'):
                time_rows.append(dict(role=role, seed=seed, scene=scene, pair=[0, 5], warmups=3,
                    samples_ms=[elapsed]*15, median_ms=elapsed, checkpoint_sha256=run['checkpoint_sha256']))
    timing = dict(lock_sha256=lock_sha, geometry_recomputed=True, rows=time_rows, run_hashes=run_hashes,
        mean_median_ms={role: 4. if role == 'full4' else 10. for role in ROLES},
        common_dense_projection_cpu=[dict(scene=scene, pair=[0, 5], warmups=1, samples_ms=[2.]*15, median_ms=2.)
                                     for scene in ('t1', 't2', 't3', 'a', 'b')])
    (runs/'timing.json').write_text(json.dumps(timing))
    result = audit.analyze(runs, tmp_path/'analysis')
    assert result['gate']['passed'] and result['independently_recomputed_raw_predictions']
    assert result['valid_count'] == 128*432
    timing['rows'][0]['checkpoint_sha256'] = 'wrong selected checkpoint'
    (runs/'timing.json').write_text(json.dumps(timing))
    with pytest.raises(ValueError, match='checkpoint identity'):
        audit.analyze(runs, tmp_path/'bad_analysis')
