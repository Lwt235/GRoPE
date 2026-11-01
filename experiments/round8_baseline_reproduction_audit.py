"""Compare the ten old-validation baseline reruns with sealed Round7 runs.

Reads only development run metadata, old validation arrays and local weights.
No model, trainer, test label, test prediction or confirmation result is loaded.
"""
from pathlib import Path
import hashlib
import json
import numbers
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (11, 22, 33, 44, 55)
ROLE_MAP = {'pred_pred': 'full', 'singleview': 'singleview'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def numeric_difference(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            raise ValueError('dictionary schema differs')
        return max((numeric_difference(a[k], b[k]) for k in a), default=0.)
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            raise ValueError('list length differs')
        return max((numeric_difference(x, y) for x, y in zip(a, b)), default=0.)
    if isinstance(a, numbers.Real) and isinstance(b, numbers.Real):
        if not np.isfinite([a, b]).all():
            raise ValueError('nonfinite metric')
        return float(abs(a-b))
    if a != b:
        raise ValueError('identity or metadata differs')
    return 0.


def normalized_sources(sources):
    result = []
    for record in sources:
        record = dict(record)
        path = Path(record['path'])
        record['path'] = str((ROOT/path).resolve() if not path.is_absolute() else path.resolve())
        if 'test' in Path(record['path']).parts:
            raise ValueError('test data is forbidden')
        result.append(record)
    return result


def main():
    oldroot = ROOT/'results/round7_depth_development'
    newroot = ROOT/'results/round8_oracle_development'
    out = ROOT/'results/round8_baseline_reproduction_audit'
    if out.exists():
        raise ValueError('refuse overwrite '+str(out))
    torch.set_num_threads(1)
    reports = []
    for newrole, oldrole in ROLE_MAP.items():
        for seed in SEEDS:
            previous, current = oldroot/oldrole/str(seed), newroot/newrole/str(seed)
            a = json.loads((previous/'run.json').read_text())
            b = json.loads((current/'run.json').read_text())
            assert a['role'] == oldrole and b['role'] == newrole
            assert a['seed'] == b['seed'] == seed
            assert a['steps'] == b['steps'] == 1000
            assert a['parameter_count'] == b['parameter_count'] == 41345
            assert a['batch_sequence_sha256'] == b['batch_sequence_sha256']
            for split in ('train', 'val'):
                assert normalized_sources(a['data_sources'][split]) == normalized_sources(b['data_sources'][split])
            for folder, metadata in ((previous, a), (current, b)):
                for file, key in (('best.pt', 'checkpoint_sha256'),
                                  ('val_predictions.npz', 'prediction_sha256'),
                                  ('validation.json', 'validation_sha256')):
                    assert sha(folder/file) == metadata[key]
            with np.load(previous/'val_predictions.npz', allow_pickle=False) as saved:
                old = saved['depth']
            with np.load(current/'val_predictions.npz', allow_pickle=False) as saved:
                new = saved['depth']
            assert old.shape == new.shape == (128, 432)
            depth_error = float(np.max(np.abs(old.astype(float)-new)))
            old_state = torch.load(previous/'best.pt', map_location='cpu', weights_only=True)['state_dict']
            new_state = torch.load(current/'best.pt', map_location='cpu', weights_only=True)['state_dict']
            assert set(old_state) == set(new_state)
            state_error = max(float((old_state[k].double()-new_state[k].double()).abs().max()) for k in old_state)
            state_exact = all(torch.equal(old_state[k], new_state[k]) for k in old_state)
            metric_error = numeric_difference(json.loads((previous/'validation.json').read_text()),
                                              json.loads((current/'validation.json').read_text()))
            history_error = numeric_difference(a['history'], b['history'])
            reports.append(dict(role=newrole, previous_role=oldrole, seed=seed,
                previous_best_step=a['best_step'], current_best_step=b['best_step'],
                best_step_equal=a['best_step'] == b['best_step'],
                depth_bitwise_equal=bool(np.array_equal(old, new)), depth_max_absolute_error_m=depth_error,
                selected_state_bitwise_equal=state_exact, state_max_absolute_error=state_error,
                validation_metric_max_absolute_error=metric_error,
                complete_validation_history_max_absolute_error=history_error,
                matched_initialization_budget_data_and_batches=True,
                previous_prediction_sha256=a['prediction_sha256'], current_prediction_sha256=b['prediction_sha256'],
                previous_checkpoint_sha256=a['checkpoint_sha256'], current_checkpoint_sha256=b['checkpoint_sha256'],
                previous_run_sha256=sha(previous/'run.json'), current_run_sha256=sha(current/'run.json')))
    exact = all(r['best_step_equal'] and r['depth_bitwise_equal'] and r['selected_state_bitwise_equal']
                and r['validation_metric_max_absolute_error'] == 0
                and r['complete_validation_history_max_absolute_error'] == 0 for r in reports)
    result = dict(scope='10 old-validation baseline reruns; no test data or predictions read',
        exact_reproduction=exact, runs=reports, audit_source_sha256=sha(__file__),
        max_depth_error_m=max(r['depth_max_absolute_error_m'] for r in reports),
        max_state_error=max(r['state_max_absolute_error'] for r in reports),
        max_validation_metric_error=max(r['validation_metric_max_absolute_error'] for r in reports),
        max_history_error=max(r['complete_validation_history_max_absolute_error'] for r in reports))
    out.mkdir(parents=True)
    (out/'summary.json').write_text(json.dumps(result, indent=2))
    (out/'source_snapshot.py').write_text(Path(__file__).read_text())
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
