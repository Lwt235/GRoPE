import numpy as np
import pytest
import torch

from models.gaussian_quadratic_pe import controlled_features
from models.round7_depth_attention import DepthAttention as Original
from models.round12_depth_attention import DepthAttention, ROLES, depth_loss
from experiments.round12_depth_train import parameter_digest


@pytest.fixture
def data():
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(98)
    features = torch.nn.functional.normalize(torch.randn(2, 6, 128, generator=generator), dim=-1)
    keys = torch.nn.functional.normalize(torch.randn(2, 6, 128, generator=generator), dim=-1)
    centers = torch.tensor([[0., 0.], [1., 0.], [2., 0.], [0., 1.], [1., 1.], [2., 1.]]).repeat(2, 1, 1)
    cov = (torch.eye(2)/12).expand(2, 6, 2, 2).clone()
    result = dict(q=features, k=keys, qmean=centers, kmean=centers, qcov=cov, kcov=cov,
                  q_to_k=centers+.07, q_to_kcov=cov*2,
                  q_depth=torch.linspace(1., 4., 12).reshape(2, 6),
                  q_value_depth=torch.linspace(1.1, 4.2, 12).reshape(2, 6),
                  k_value_depth=torch.linspace(1.8, 6., 12).reshape(2, 6))
    qg, kg = controlled_features(result, 'full', .5)
    result['geometry_logits'] = qg @ kg.transpose(-1, -2)
    return result


def test_original_initialization_and_capacity_are_identical(data):
    torch.manual_seed(22); original = Original('full')
    expected = parameter_digest(original)
    for role in ROLES:
        torch.manual_seed(22); model = DepthAttention(role, 5.)
        assert parameter_digest(model) == expected
        assert sum(p.numel() for p in model.parameters()) == 41345


def test_replacing_geometry_only_reproduces_original_forward(data):
    torch.manual_seed(11); original = Original('full')
    torch.manual_seed(11); new = DepthAttention('full4', 5.)
    for intervention in (None, 'zero_v', 'reverse_v'):
        torch.testing.assert_close(new(data, intervention), original(data, intervention), rtol=0, atol=0)
    new(data).sum().backward(); original(data).sum().backward()
    for (_, left), (_, right) in zip(new.named_parameters(), original.named_parameters()):
        torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)


def test_native_singleview_matches_original_and_needs_no_key_fields(data):
    torch.manual_seed(11); original = Original('singleview')
    torch.manual_seed(11); new = DepthAttention('singleview', 5.)
    expected = original(data)
    isolated = {name: data[name] for name in ('q', 'qmean', 'qcov', 'q_depth', 'q_value_depth')}
    torch.testing.assert_close(new(isolated), expected, rtol=0, atol=0)
    poison = dict(isolated, k=float('nan'), geometry_logits=float('nan'), target=float('nan'), valid=False)
    torch.testing.assert_close(new(poison), expected, rtol=0, atol=0)


def test_labels_old_geometry_and_sensor_masks_cannot_change_fusion(data):
    model = DepthAttention('full4', 5.); expected = model(data)
    poisoned = dict(data, target=torch.full((2, 6), float('nan')), valid=torch.zeros(2, 6),
                    boundary=torch.ones(2, 6), q_to_k=float('nan'), qcov=float('nan'), qmean=float('nan'))
    torch.testing.assert_close(model(poisoned), expected, rtol=0, atol=0)


def test_actual_v_head_and_content_paths_have_finite_nonzero_gradients(data):
    torch.manual_seed(11); model = DepthAttention('full4', 5.)
    prediction, attention, values = model(data, return_attention=True)
    target = data['q_depth']*.91; valid = torch.ones_like(target, dtype=torch.bool)
    target[0, 0] = float('nan'); valid[0, 0] = False
    depth_loss(prediction, target, valid).backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name
    assert (prediction-model(data, 'zero_v')).abs().max() > 0
    assert (prediction-model(data, 'reverse_v')).abs().max() > 0
    torch.testing.assert_close(attention.sum(-1), torch.ones_like(target))
    assert values.shape == (2, 6, 64)


def test_frozen_zero_temperature_explicitly_disables_content_gradient(data):
    model = DepthAttention('full1', 0.); prediction = model(data)
    prediction.sum().backward()
    assert model.qk.weight.grad is not None
    assert model.qk.weight.grad.abs().sum() == 0
    assert model.value.weight.grad.abs().sum() > 0
    assert model.head[-1].weight.grad.abs().sum() > 0


def test_all_missing_labels_loss_is_finite_and_zero(data):
    prediction = DepthAttention()(data)
    loss = depth_loss(prediction, torch.full_like(prediction, float('nan')), torch.zeros_like(prediction, dtype=torch.bool))
    assert loss == 0 and torch.isfinite(loss)


@pytest.mark.parametrize('bad', [torch.zeros(2, 6, 5), torch.zeros(2, 6, 6, dtype=torch.float64)])
def test_geometry_contract_rejects_shape_and_precision_mismatch(data, bad):
    with pytest.raises(ValueError):
        DepthAttention()(dict(data, geometry_logits=bad))


def test_geometry_changes_routing_without_mutating_original_data(data):
    model = DepthAttention(); original = data['geometry_logits'].clone()
    first = model(data)
    changed = dict(data, geometry_logits=data['geometry_logits'].flip(-1))
    assert (model(changed)-first).abs().max() > 0
    torch.testing.assert_close(data['geometry_logits'], original, rtol=0, atol=0)
