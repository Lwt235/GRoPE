"""Fixed old-validation replay and privileged-field isolation on real pairs."""
from pathlib import Path
import json
import numpy as np
import torch

from experiments.round8_oracle_train import load_development, sources_now
from experiments.round7_depth_data import ROOT, sha
from experiments.round7_depth_train import batch
from models.round7_depth_attention import depth_loss
from models.round8_oracle_attention import OracleDepthAttention


def norm(parameter):
    return 0. if parameter.grad is None else float(parameter.grad.detach().double().norm())


def main():
    out = ROOT/'results/round8_real_forward_audit'
    if out.exists():
        raise ValueError('refuse overwrite '+str(out))
    runs = ROOT/'results/round8_oracle_development'
    lock = json.loads((runs/'lock.json').read_text())
    assert sources_now() == lock['source_hashes']
    torch.set_num_threads(1)
    data, ids, sources, oracle_sources = load_development('val', 'cpu')
    assert sources == lock['data_sources']['val'] and oracle_sources == lock['data_sources']['oracle_val']
    selected = []
    for scene in sorted({r['scene'] for r in ids}):
        indices = [i for i, row in enumerate(ids) if row['scene'] == scene]
        selected.extend((indices[0], indices[-1]))
    d = batch(data, selected)
    valid, eligible = d['valid'], d['oracle_valid']
    assert (valid & eligible).any() and (valid & ~eligible).any()
    expected_route = np.zeros((4, 432, 432), np.float32)
    indices, weights = d['oracle_indices'].numpy(), d['oracle_weights'].numpy()
    for item in range(4):
        for row in range(432):
            np.add.at(expected_route[item, row], indices[item, row], weights[item, row])
    expected_route = torch.from_numpy(expected_route)
    reports = []
    for role in ('gt_route', 'gt_value', 'gt_both'):
        for seed in (11, 22, 33, 44, 55):
            folder = runs/role/str(seed)
            metadata = json.loads((folder/'run.json').read_text())
            assert sha(folder/'best.pt') == metadata['checkpoint_sha256']
            assert sha(folder/'val_predictions.npz') == metadata['prediction_sha256']
            state = torch.load(folder/'best.pt', map_location='cpu', weights_only=True)['state_dict']
            model = OracleDepthAttention(role).eval()
            model.load_state_dict(state)
            original = OracleDepthAttention('pred_pred').eval()
            original.load_state_dict(state)
            with torch.no_grad():
                prediction, attention, _ = model(d, return_attention=True)
                _, predicted_attention, _ = original(d, return_attention=True)
                expected = (torch.where(eligible[..., None], expected_route, predicted_attention)
                            if role in ('gt_route', 'gt_both') else predicted_attention)
                route_error = float((attention-expected).abs().max())
                assert route_error == 0
                with np.load(folder/'val_predictions.npz', allow_pickle=False) as saved:
                    recorded = torch.from_numpy(saved['depth'][selected])
                replay_error = float((prediction-recorded).abs().max())
                assert replay_error < 5e-5
                poisoned = dict(d, target=torch.full_like(d['target'], float('nan')),
                    valid=~d['valid'], boundary=~d['boundary'],
                    k_sensor_fraction=torch.full_like(d['k_sensor_fraction'], float('nan')),
                    oracle_pixel=torch.full_like(d['oracle_pixel'], float('nan')),
                    direct_corresponding_pred_depth=torch.full_like(d['q_depth'], float('nan')),
                    direct_corresponding_sensor_depth=torch.full_like(d['q_depth'], float('nan')))
                if role == 'gt_route':
                    poisoned['k_oracle_value_depth'] = torch.full_like(d['q_depth'], float('nan'))
                if role == 'gt_value':
                    poisoned.update(oracle_indices=torch.full_like(d['oracle_indices'], -999),
                        oracle_weights=torch.full_like(d['oracle_weights'], float('nan')),
                        oracle_valid=~d['oracle_valid'])
                assert torch.equal(prediction, model(poisoned))
                zero_change = float((model(d, intervention='zero_v')-prediction).abs().mean())
                reverse_change = float((model(d, intervention='reverse_v')-prediction).abs().mean())
                assert zero_change > 0 and reverse_change > 0
            gradient = {}
            for subset, mask in (('all_query', valid), ('eligible', valid & eligible),
                                 ('fallback', valid & ~eligible)):
                model.zero_grad(set_to_none=True)
                loss = depth_loss(model(d), d['target'], mask)
                loss.backward()
                norms = dict(qk=norm(model.qk.weight), value=norm(model.value.weight),
                             query=norm(model.query.weight), head_hidden=norm(model.head[0].weight),
                             head_output=norm(model.head[-1].weight))
                assert np.isfinite(list(norms.values())).all()
                assert all(norms[key] > 0 for key in ('value', 'query', 'head_hidden', 'head_output'))
                if role in ('gt_route', 'gt_both') and subset == 'eligible':
                    assert norms['qk'] == 0
                else:
                    assert norms['qk'] > 0
                gradient[subset] = norms
            reports.append(dict(role=role, seed=seed, cpu_replay_max_error_m=replay_error,
                attention_contract_max_error=route_error, unused_field_poisoning_bitwise_invariant=True,
                zero_v_mean_absolute_change_m=zero_change, reverse_v_mean_absolute_change_m=reverse_change,
                gradient_l2_norms=gradient, checkpoint_sha256=metadata['checkpoint_sha256']))
    assert sources_now() == lock['source_hashes']
    out.mkdir(parents=True)
    result = dict(passed=True, scope='15 saved oracle checkpoints on fixed first/last pair of both OLD validation scans',
        pairs=[ids[i] for i in selected], selected_query_valid_count=int(valid.sum()),
        selected_eligible_count=int((valid & eligible).sum()),
        selected_fallback_count=int((valid & ~eligible).sum()), runs=reports,
        max_cpu_replay_error_m=max(r['cpu_replay_max_error_m'] for r in reports),
        max_attention_contract_error=max(r['attention_contract_max_error'] for r in reports),
        audit_source_sha256=sha(__file__), source_lock_sha256=sha(runs/'lock.json'))
    (out/'summary.json').write_text(json.dumps(result, indent=2))
    (out/'source_snapshot.py').write_text(Path(__file__).read_text())
    print(json.dumps(dict(passed=True, runs=len(reports), max_cpu_replay_error_m=result['max_cpu_replay_error_m'],
                         max_attention_contract_error=result['max_attention_contract_error']), indent=2))


if __name__ == '__main__':
    main()
