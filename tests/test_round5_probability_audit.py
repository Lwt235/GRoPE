import json

import numpy as np
import pytest

from experiments.round5_probability_audit import audit_runs, compare_metrics, probability_metrics


def sample():
    probability = np.full((4, 432), 1e-5)
    target = np.array([0, 1, 24, 2])
    probability[np.arange(4), target] = 1-431e-5
    return dict(probability=probability, gt_valid=np.array([True, True, True, False]),
        pixcontinuous=np.array([[3.5, 3.5], [11.5, 3.5], [3.5, 11.5], [np.nan, np.nan]]),
        key_indices=np.array([[0, 1, 24, 25], [1, 2, 25, 26], [24, 25, 48, 49], [-1]*4]),
        key_weights=np.array([[1., 0., 0., 0.]]*3+[[0., 0., 0., 0.]]))


def test_known_metrics_and_unknown_nan_rows():
    result = probability_metrics(**sample())
    assert result['n_valid'] == result['pck8_count'] == result['pck4_count'] == 3
    assert result['epe_argmax_sum'] == 0
    assert result['epe_soft_sum'] > 0
    np.testing.assert_allclose(result['ce_sum'], -3*np.log(1-431e-5), rtol=1e-13)
    assert compare_metrics(result, result) == {k:0 for k in ('ce_sum', 'epe_argmax_sum', 'epe_soft_sum')}


def test_swapped_key_columns_are_detected_against_report():
    arrays = sample()
    reported = probability_metrics(**arrays)
    arrays['probability'][:, [0, 1]] = arrays['probability'][:, [1, 0]]
    actual = probability_metrics(**arrays)
    with pytest.raises(ValueError, match='pck4_count'):
        compare_metrics(actual, reported)


def test_wrong_row_major_grid_is_detected():
    arrays = sample()
    reported = probability_metrics(**arrays)
    # A mistaken y/x convention puts the second and third GT coordinates on
    # the opposite axes. Their argmax PCK changes despite unchanged indices.
    arrays['pixcontinuous'] = arrays['pixcontinuous'][:, ::-1]
    with pytest.raises(ValueError, match='pck8_count'):
        compare_metrics(probability_metrics(**arrays), reported)


def test_column_normalization_and_noninteger_validity_are_rejected():
    arrays = sample()
    arrays['probability'] /= arrays['probability'].sum(axis=0)
    with pytest.raises(ValueError, match='sum to one along key'):
        probability_metrics(**arrays)
    arrays = sample()
    arrays['gt_valid'] = arrays['gt_valid'].astype(float)
    with pytest.raises(ValueError, match='Boolean/integer'):
        probability_metrics(**arrays)


def test_zero_positive_target_and_malformed_unknown_labels_fail_explicitly():
    arrays = sample()
    value = arrays['probability'][0, 0]
    arrays['probability'][0, 0] = 0
    arrays['probability'][0, 1] += value
    with pytest.raises(ValueError, match='target probability is zero'):
        probability_metrics(**arrays)
    arrays = sample()
    arrays['key_weights'][-1, 0] = 1
    with pytest.raises(ValueError, match='sentinels'):
        probability_metrics(**arrays)


def test_float_sums_are_compared_per_row_but_counts_exactly():
    actual = probability_metrics(**sample())
    reported = dict(actual, ce_sum=actual['ce_sum']+2.9e-5)
    compare_metrics(actual, reported)
    reported['ce_sum'] += 2e-6
    with pytest.raises(ValueError, match='ce_sum differs per valid row'):
        compare_metrics(actual, reported)
    with pytest.raises(ValueError, match='integer count'):
        compare_metrics(actual, dict(actual, n_valid=3.0))


def test_end_to_end_run_and_sample_index_audit(tmp_path):
    arrays = sample()
    metrics = probability_metrics(**arrays)
    rows = [dict(scene=f'place_{i//3}', pair_id=f'{i}:{i+1}', q_index=i, k_index=i+1,
        method='normalized', seed=11, split='test', **metrics) for i in range(6)]
    run = tmp_path/'normalized_11'
    directory = run/'predictions/test'
    directory.mkdir(parents=True)
    for i in range(6):
        np.savez_compressed(directory/f'pair_{i:04d}.npz', **arrays)
    (run/'test.json').write_text(json.dumps(dict(rows=rows, input_sources=[], sourcehash={}, selected_step=100)))
    result = audit_runs(tmp_path, ['normalized'], [11], expected_scenes=2, expected_query_rows=4)
    assert result['actual_run_count'] == result['expected_run_count'] == 1
    assert result['archives'] == result['unique_sampled_pairs'] == 6
    assert all(len(r['archive_sha256']) == len(r['matched_test_json_sha256']) == 64 for r in result['records'])
    (directory/'pair_0005.npz').rename(directory/'pair_0020.npz')
    with pytest.raises(ValueError, match='misindexed'):
        audit_runs(tmp_path, ['normalized'], [11], expected_scenes=2, expected_query_rows=4)
