"""R19 software-only tests: every numerical array below is synthetic.

Importing production here is permitted solely for synthetic cross-checks;
the independent auditor itself never imports its mathematical implementation.
"""
from copy import deepcopy
from fractions import Fraction
import ast
import json
from pathlib import Path

import numpy as np
import pytest

from experiments import round19_center_semantics as P
from experiments import round19_center_semantics_audit as A


def cameras():
    y, x = np.mgrid[:144, :192]
    return dict(q_predicted_depth=2. + .003*x + .007*y,
        q_K=np.array([[173., 0., 95.], [0., 179., 71.], [0., 0., 1.]], dtype=np.float64),
        k_K=np.array([[181., 0., 96.], [0., 167., 72.], [0., 0., 1.]], dtype=np.float64),
        q_to_k_transform=np.eye(4, dtype=np.float64))


@pytest.mark.parametrize('stride', [8, 16, 24])
def test_independent_solve_projection_matches_historical_ast_on_synthetic_camera(stride):
    inputs = cameras()
    angle = .017
    inputs['q_to_k_transform'][:3, :3] = [[np.cos(angle), 0., np.sin(angle)],
        [0., 1., 0.], [-np.sin(angle), 0., np.cos(angle)]]
    inputs['q_to_k_transform'][:3, 3] = [.03, -.04, .05]
    bounds = A.native_bounds(stride)
    actual = P.extracted_predictor()(inputs, bounds)
    expected = A.center_reference(inputs, bounds)
    assert np.max(np.abs(actual[0] - expected[0])) <= A.COORD_ATOL
    for first, second in zip(actual[1:], expected[1:]):
        np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(A.nearest_reference(expected[0]),
        P.nearest(actual[0], A.key_centers()))


def test_depth_sanitization_bilinear_mean_and_behind_flags_do_not_drop_rows():
    inputs = cameras()
    inputs['q_K'] = inputs['k_K'] = np.eye(3, dtype=np.float64)
    inputs['q_predicted_depth'][:] = 2.
    inputs['q_predicted_depth'][3:5, 3:5] = [[np.nan, np.inf], [-np.inf, -2.]]
    inputs['q_to_k_transform'][0, 3] = 2.
    xy, front, visible = A.center_reference(inputs, A.native_bounds(8))
    assert len(xy) == 432
    assert xy[0, 0] == pytest.approx(3.5 + 2. / 7.55, abs=1e-14)
    assert xy[0, 1] == pytest.approx(3.5, abs=1e-14)
    inputs['q_to_k_transform'][2, 3] = -100.
    xy, front, visible = A.center_reference(inputs, A.native_bounds(8))
    expected = P.extracted_predictor()(inputs, A.native_bounds(8))
    assert len(xy) == 432 and not front.any() and (~visible).any()
    assert np.max(np.abs(xy)) == 1024.
    np.testing.assert_array_equal(xy, expected[0])
    np.testing.assert_array_equal(front, expected[1])


@pytest.mark.parametrize('field', ['q_K', 'k_K', 'q_to_k_transform'])
def test_float32_cameras_are_rejected_instead_of_changing_old_inverse_semantics(field):
    inputs = cameras()
    inputs[field] = inputs[field].astype(np.float32)
    with pytest.raises(ValueError, match='float64'):
        A.center_reference(inputs, A.native_bounds(8))


def test_native_first_minimum_ties_outside_fov_and_invalid_inputs():
    points = np.array([[7.5, 3.5], [7.5, 7.5], [-1024., -1024.], [1024., 1024.]], dtype=np.float64)
    np.testing.assert_array_equal(A.nearest_reference(points), [0, 0, 0, 431])
    np.testing.assert_array_equal(P.nearest(points, A.key_centers()), [0, 0, 0, 431])
    np.testing.assert_array_equal(A.fov(points), [True, True, False, False])
    with pytest.raises(ValueError):
        A.nearest_reference(np.array([[np.nan, 1.]]))
    with pytest.raises(ValueError):
        A.nearest_reference(points.astype(np.float32))


def metric_case(query='g24', valid_count=25, a_correct=12, mu_correct=12):
    n = A.COUNTS[query]
    valid = np.arange(n) < valid_count
    target = np.tile(A.key_centers()[0], (n, 1))
    values = {'key_bounds_px': A.native_bounds(8)}
    for role, correct in (('a', a_correct), ('mu', mu_correct)):
        winner = np.where(np.arange(n) < correct, 0, 100).astype(np.int32)
        values[query + '__nearest_' + role] = winner
        values[query + '__' + role + '_px'] = A.key_centers()[winner].copy()
    values[query + '__primary_argmax'] = np.full(n, 100, dtype=np.int32)
    # Intentionally all outside/behind flags: the valid population must remain.
    for name in A.FLAGS:
        values[query + '__' + name] = np.zeros(n, dtype=bool)
    return values, target, valid


def test_all_valid_pck_boundary_transitions_and_no_predicted_flag_filter():
    values, target, valid = metric_case(valid_count=8, a_correct=8, mu_correct=0)
    target[:8, 0] += 8.
    # Saved a winner remains exactly distance 8; nextafter makes only row 1 fail.
    target[1, 0] = np.nextafter(target[1, 0], np.inf)
    metrics, errors = A.pair_metrics(values, 'g24', target, valid)
    production, prod_errors = P.pair_metrics(values, 'g24', target, valid)
    A.compare(production, metrics)
    assert metrics['n_point'] == 8 and metrics['correct']['a'] == 7
    assert metrics['comparisons']['a_minus_mu']['benefit'] == 7
    assert metrics['comparisons']['a_minus_mu']['harm'] == 0
    assert metrics['flags']['a_front']['false']['n_point'] == 8
    assert sum(metrics['correctness_transitions'].values()) == 8
    assert metrics['correctness_transitions']['primary0_mu0_a1'] == 7
    np.testing.assert_array_equal(errors['a'], prod_errors['a'])
    target[~valid] = np.nan  # Nonexistent labels outside the original mask do not remove valid rows.
    assert A.pair_metrics(values, 'g24', target, valid)[0] == metrics


def synthetic_cohort(gain=2, loss=2):
    rows, errors = [], {}
    for scene in A.SCENES:
        for pair_index in range(16):
            for query in A.QUERIES:
                change = (gain if query == 'g24' else -loss) if pair_index == 0 else 0
                values, target, valid = metric_case(query, a_correct=12 + change)
                metrics, distances = A.pair_metrics(values, query, target, valid)
                pair = [pair_index * 2, pair_index * 2 + 1]
                rows.append(dict(scene=scene, pair=pair, query=query, **metrics))
                errors[(scene, tuple(pair), query)] = distances
    return rows, errors


def test_exact_fraction_boundaries_and_all_three_scene_condition():
    rows, errors = synthetic_cohort()
    summary = A.aggregate(rows, errors)
    assert summary['screen']['passed']
    assert summary['macro']['g24']['deltas']['a_minus_mu'] == A.fraction(Fraction(1, 200))
    assert summary['macro']['g8']['deltas']['a_minus_mu'] == A.fraction(Fraction(-1, 200))
    producer = P.make_summary(rows, errors)
    A.compare({k: producer[k] for k in summary}, summary)
    reduced = deepcopy(rows)
    next(r for r in reduced if r['query'] == 'g24')['correct']['a'] -= 1
    assert not A.aggregate(reduced, errors)['screen']['conditions']['q24_macro_gain_at_least_1_200']
    reduced = deepcopy(rows)
    first = next(r for r in reduced if r['query'] == 'g8')
    first['correct']['a'] -= 1
    assert not A.aggregate(reduced, errors)['screen']['conditions']['q8_loss_at_most_1_200']
    varied = deepcopy(rows)
    for scene, change in zip(A.SCENES, (-2, 1, 1)):
        next(r for r in varied if r['scene'] == scene and r['query'] == 'g24')['correct']['a'] += change
    result = A.aggregate(varied, errors)
    assert result['screen']['conditions']['q24_macro_gain_at_least_1_200']
    assert not result['screen']['conditions']['q24_all_scenes_strict']
    assert not result['screen']['passed']


def test_population_duplicates_and_missing_rows_are_rejected():
    rows, errors = synthetic_cohort()
    with pytest.raises(ValueError):
        A.aggregate(rows[:-1], errors)
    with pytest.raises(ValueError):
        A.aggregate(rows[:-1] + [rows[0]], errors)


def test_scene_median_is_pooled_over_original_valid_rows():
    metrics, error_sets = [], []
    for values in ([0.], [100., 101., 102.]):
        arrays, target, valid = metric_case(valid_count=len(values))
        arrays['g24__a_px'][:len(values)] = target[:len(values)] + np.array(values)[:, None] * [1., 0.]
        row, errors = A.pair_metrics(arrays, 'g24', target, valid)
        metrics.append(row)
        error_sets.append(errors)
    combined = A.combine_metrics(metrics, error_sets)
    assert combined['continuous']['a']['median_error_px'] == 100.5
    assert combined['continuous']['a']['median_error_px'] != np.mean(
        [r['continuous']['a']['median_error_px'] for r in metrics])
    A.compare(P.aggregate(metrics, error_sets), combined)


def test_canonical_hashes_discrete_values_and_fraction_floats_are_exact():
    values = np.arange(12, dtype=np.float64).reshape(6, 2)
    assert A.array_sha(values) == P.array_sha(values)
    assert A.legacy_array_sha(values) == P.legacy_array_sha(values)
    assert A.array_sha(values) != A.array_sha(values.astype(np.float32))
    A.compare({'error_sum_px': 1. + 1e-9}, {'error_sum_px': 1.})
    with pytest.raises(ValueError):
        A.compare({'n_point': 25.}, {'n_point': 25})
    with pytest.raises(ValueError):
        A.compare({'float': np.nextafter(.005, np.inf)}, {'float': .005})
    with pytest.raises(ValueError):
        A.compare({'median_error_px': 1. + 1e-6}, {'median_error_px': 1.})


def test_inventory_detects_added_changed_and_missing_files(tmp_path):
    payload = tmp_path / 'payload.txt'
    payload.write_text('synthetic')
    A.write(tmp_path / 'completion.json', dict(passed=True, files=A.inventory(tmp_path)))
    A.check_inventory(tmp_path)
    payload.write_text('changed')
    with pytest.raises(ValueError):
        A.check_inventory(tmp_path)
    payload.write_text('synthetic')
    (tmp_path / 'extra.txt').write_text('extra')
    with pytest.raises(ValueError):
        A.check_inventory(tmp_path)
    (tmp_path / 'extra.txt').unlink()
    payload.unlink()
    with pytest.raises(ValueError):
        A.check_inventory(tmp_path)


def test_freeze_refuses_existing_forward_before_contract_or_target_access(tmp_path, monkeypatch):
    run, out = tmp_path / 'producer', tmp_path / 'auditor'
    (run / 'forward').mkdir(parents=True)
    monkeypatch.setattr(A, 'check_producer_lock', lambda _: pytest.fail('must reject before reads'))
    with pytest.raises(ValueError, match='precede'):
        A.freeze(run, out)
    assert not out.exists()


def test_final_audit_refuses_missing_forward_completion_before_target_access(tmp_path, monkeypatch):
    run, out = tmp_path / 'producer', tmp_path / 'auditor'
    run.mkdir(); out.mkdir()
    target = tmp_path / 'target.npz'
    target.write_bytes(b'not-to-be-opened')
    monkeypatch.setattr(A, 'check_audit_lock', lambda *_: ({'entries': [{'target_path': str(target)}]}, {}))
    monkeypatch.setattr(A, 'check_seal', lambda *_: ([], {}))
    original = A.sha
    def guarded_sha(path):
        assert Path(path) != target, 'target touched before accepted forward audit'
        return original(path)
    monkeypatch.setattr(A, 'sha', guarded_sha)
    with pytest.raises(FileNotFoundError):
        A.audit(run, out)
    assert A.read(out / 'final_audit/completion.json')['passed'] is False
    assert (out / 'final_audit/failure.json').exists()
    with pytest.raises(ValueError, match='preserve'):
        A.audit(run, out)


def test_auditor_has_no_producer_or_r13_math_imports():
    source = Path(A.__file__).read_text()
    tree = ast.parse(source)
    imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
    for node in imports:
        names = [n.name for n in node.names]
        if isinstance(node, ast.ImportFrom):
            names += [node.module or '']
        assert not any('round13' in name or 'round19_center_semantics' in name for name in names)
