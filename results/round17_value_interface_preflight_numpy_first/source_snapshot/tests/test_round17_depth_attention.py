"""Pure synthetic tests; no dataset, cache, target archive, or study selection."""
import math

import pytest
import torch
import torch.nn.functional as F

from models.round7_depth_attention import DepthAttention as Original
from models.round12_depth_attention import DepthAttention as Round12
from models.round17_depth_attention import (
    DepthAttention, FUSION_FIELDS, MODES, SINGLEVIEW_FIELDS, TRACE_FIELDS, depth_loss)
from models import round17_trace_pe as TRACE


@pytest.fixture
def data():
    # Six query cells and five cross-source keys also exercise non-square
    # fusion. The singleview control must recover its own six keys.
    generator = torch.Generator().manual_seed(171723)
    q = F.normalize(torch.randn(2, 6, 128, generator=generator), dim=-1)
    k = F.normalize(torch.randn(2, 5, 128, generator=generator), dim=-1)
    qmean = torch.tensor([[0., 0.], [1., 0.], [2., 0.],
                          [0., 1.], [1., 1.], [2., 1.]]).repeat(2, 1, 1)
    kmean = qmean[:, :5] + .08
    qcov = (torch.eye(2) / 12).expand(2, 6, 2, 2).clone()
    kcov = (torch.eye(2) / 12).expand(2, 5, 2, 2).clone()
    factor = torch.randn(2, 6, 2, 2, generator=generator) * .2
    support = factor @ factor.transpose(-1, -2) + torch.eye(2) * .02
    geometry = -.5 * (qmean.double()[:, :, None] - kmean.double()[:, None]).square().sum(-1)
    geometry += torch.randn(2, 6, 5, generator=generator, dtype=torch.float64) * .1
    return dict(q=q, k=k, qmean=qmean, kmean=kmean, qcov=qcov, kcov=kcov,
                q_to_k=qmean + .07, q_to_kcov=support,
                q_depth=torch.linspace(1., 4., 12).reshape(2, 6),
                q_value_depth=torch.linspace(1.1, 4.2, 12).reshape(2, 6),
                k_value_depth=torch.linspace(1.8, 6., 10).reshape(2, 5),
                geometry_logits=geometry, support_covariance=support)


def model(mode='trace', temperature=3.):
    # Arbitrary synthetic constants; not a task configuration or a sweep.
    torch.manual_seed(171724)
    return DepthAttention(mode) if mode == 'singleview' else DepthAttention(
        mode, bandwidth=.625, temperature=temperature)


def exact(left, right):
    torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_initialization_capacity_state_and_rng_match_r7():
    torch.manual_seed(171724)
    original = Original('full')
    expected_rng = torch.get_rng_state().clone()
    for mode in MODES:
        current = model(mode)
        assert sum(p.numel() for p in current.parameters()) == 41345
        assert list(current.state_dict()) == list(original.state_dict())
        for name, value in current.state_dict().items():
            exact(value, original.state_dict()[name])
        exact(torch.get_rng_state(), expected_rng)


@pytest.mark.parametrize('mode', ('trace', 'constant'))
def test_online_content_softmax_and_exact_float32_value_boundary(data, mode, monkeypatch):
    current = model(mode)
    calls = []
    original_probabilities = TRACE.probabilities

    def traced(geometry, content, covariance, bandwidth, temperature):
        calls.append((geometry, content, covariance, bandwidth, temperature))
        return original_probabilities(geometry, content, covariance, bandwidth, temperature)

    monkeypatch.setattr(TRACE, 'probabilities', traced)
    head_inputs = []
    handle = current.head.register_forward_pre_hook(lambda _, args: head_inputs.append(args[0]))
    prediction, attention, values = current(data, return_attention=True)
    handle.remove()
    q = F.normalize(current.qk(data['q']), dim=-1)
    k = F.normalize(current.qk(data['k']), dim=-1)
    content = q @ k.transpose(-1, -2)
    if mode == 'trace':
        assert len(calls) == 1
        assert calls[0][0] is data['geometry_logits']
        assert calls[0][2] is data['support_covariance']
        exact(calls[0][1], content)
        expected = original_probabilities(data['geometry_logits'], content,
                                         data['support_covariance'], .625, 3.)
    else:
        assert not calls
        expected = (data['geometry_logits'] + (content * torch.tensor(3.)).double()).softmax(-1)
    exact(attention, expected)
    assert attention.dtype == torch.float64
    assert prediction.dtype == values.dtype == head_inputs[0].dtype == torch.float32
    exact(head_inputs[0][..., 64:], attention.float() @ values)
    torch.testing.assert_close(attention.sum(-1), torch.ones(2, 6, dtype=torch.float64),
                               rtol=0, atol=1.e-15)


@pytest.mark.parametrize('mode', ('trace', 'constant'))
def test_online_qk_actual_v_gradients_and_interventions(data, mode):
    current = model(mode)
    prediction, attention, values = current(data, return_attention=True)
    target = data['q_depth'] * .91
    valid = torch.ones_like(target, dtype=torch.bool)
    valid[0, 0] = False
    target[0, 0] = float('nan')
    depth_loss(prediction, target, valid).backward()
    for name, parameter in current.named_parameters():
        assert parameter.grad is not None, name
        assert bool(torch.isfinite(parameter.grad).all()), name
        assert bool(parameter.grad.abs().sum() > 0), name
    assert bool((prediction - current(data, 'zero_v')).abs().max() > 0)
    assert bool((prediction - current(data, 'reverse_v')).abs().max() > 0)
    zero, zero_attention, zero_values = current(data, 'zero_v', True)
    reverse, reverse_attention, reverse_values = current(data, 'reverse_v', True)
    exact(zero_attention, attention)
    exact(reverse_attention, attention)
    exact(zero_values, torch.zeros_like(values))
    exact(reverse_values, values.flip(-2))
    with torch.no_grad():
        current.qk.weight[:17].mul_(1.1)
    _, changed_attention, _ = current(data, return_attention=True)
    assert bool((changed_attention - attention).abs().max() > 0)


@pytest.mark.parametrize('mode', ('trace', 'constant'))
def test_zero_temperature_geometry_limit_and_zero_qk_gradient(data, mode):
    current = model(mode, 0.)
    prediction, attention, _ = current(data, return_attention=True)
    exact(attention, data['geometry_logits'].softmax(-1))
    prediction.sum().backward()
    assert current.qk.weight.grad is not None
    exact(current.qk.weight.grad, torch.zeros_like(current.qk.weight))
    assert bool(current.value.weight.grad.abs().sum() > 0)
    assert bool(current.head[-1].weight.grad.abs().sum() > 0)


def test_zero_covariance_trace_equals_constant_outputs_and_gradients(data):
    data = dict(data, support_covariance=torch.zeros_like(data['support_covariance']))
    traced, constant = model('trace'), model('constant')
    for intervention in (None, 'zero_v', 'reverse_v'):
        for left, right in zip(traced(data, intervention, True), constant(data, intervention, True)):
            exact(left, right)
    traced(data).sum().backward()
    constant(data).sum().backward()
    for (_, left), (_, right) in zip(traced.named_parameters(), constant.named_parameters()):
        exact(left.grad, right.grad)


def test_singleview_exact_r7_r12_outputs_and_gradients_without_source(data):
    current = model('singleview')
    torch.manual_seed(171724)
    original = Original('singleview')
    torch.manual_seed(171724)
    round12 = Round12('singleview', 5.)
    own = {name: data[name] for name in SINGLEVIEW_FIELDS}
    poison = dict(own, k=object(), k_value_depth=object(), geometry_logits=object(),
                  support_covariance=object(), q_to_k=object(), kcov=object(),
                  target=object(), valid=object(), boundary=object())
    for intervention in (None, 'zero_v', 'reverse_v'):
        result = current(poison, intervention, True)
        for old in (original(data, intervention, True), round12(own, intervention, True)):
            for left, right in zip(result, old):
                exact(left, right)
        assert all(t.dtype == torch.float32 for t in result)
    current(poison).sum().backward()
    original(data).sum().backward()
    round12(own).sum().backward()
    for parameters in zip(current.parameters(), original.parameters(), round12.parameters()):
        exact(parameters[0].grad, parameters[1].grad)
        exact(parameters[0].grad, parameters[2].grad)


@pytest.mark.parametrize('mode', MODES)
def test_only_allowlisted_fields_are_read_and_input_is_not_mutated(data, mode):
    current = model(mode)
    expected = current(data)
    names = SINGLEVIEW_FIELDS if mode == 'singleview' else TRACE_FIELDS if mode == 'trace' else FUSION_FIELDS
    before = {name: data[name].clone() for name in names}

    class AllowedOnly(dict):
        def __getitem__(self, key):
            assert key in names, key
            return super().__getitem__(key)

        def __iter__(self):
            raise AssertionError('must not enumerate arbitrary input fields')

    minimal = AllowedOnly({name: data[name] for name in names})
    minimal.update(target=object(), valid=object(), boundary=object(),
                   gt_depth=object(), visibility=object(), attention=object(),
                   content=object(), source_sensor_depth=object())
    exact(current(minimal), expected)
    for name in names:
        exact(data[name], before[name])
    if mode == 'constant':
        exact(current(dict(data, support_covariance=object())), expected)


@pytest.mark.parametrize('mode', MODES)
def test_missing_or_invalid_intervention_fails(data, mode):
    current = model(mode)
    with pytest.raises(KeyError):
        current({})
    with pytest.raises(ValueError, match='intervention'):
        current(data, intervention='gt_values')


@pytest.mark.parametrize('name', TRACE_FIELDS)
@pytest.mark.parametrize('failure', ('dtype', 'shape', 'finite', 'device'))
def test_fusion_strict_input_contract(data, name, failure):
    bad = dict(data)
    if failure == 'dtype':
        bad[name] = data[name].float() if name == 'geometry_logits' else data[name].double()
    elif failure == 'shape':
        bad[name] = data[name][0]
    elif failure == 'finite':
        bad[name] = data[name].clone()
        bad[name].view(-1)[0] = float('nan')
    else:
        bad[name] = torch.empty_like(data[name], device='meta')
    with pytest.raises(ValueError):
        model()(bad)


@pytest.mark.parametrize('name', SINGLEVIEW_FIELDS)
@pytest.mark.parametrize('failure', ('dtype', 'shape', 'finite'))
def test_singleview_strict_input_contract(data, name, failure):
    bad = {field: data[field] for field in SINGLEVIEW_FIELDS}
    if failure == 'dtype':
        bad[name] = data[name].double()
    elif failure == 'shape':
        bad[name] = data[name][0]
    else:
        bad[name] = data[name].clone()
        bad[name].view(-1)[0] = float('inf')
    with pytest.raises(ValueError):
        model('singleview')(bad)


@pytest.mark.parametrize('bad_covariance', (
    torch.tensor([[1., .2], [0., 1.]]),
    torch.tensor([[1., 2.], [2., 1.]]),
    torch.tensor([[-1., 0.], [0., -1.]])))
def test_trace_rejects_invalid_covariance_without_repair(data, bad_covariance):
    covariance = bad_covariance.expand(2, 6, 2, 2).clone()
    original = covariance.clone()
    with pytest.raises(ValueError):
        model()(dict(data, support_covariance=covariance))
    exact(covariance, original)


def test_model_precision_and_nonfinite_parameters_are_rejected(data):
    current = model().double()
    with pytest.raises(ValueError, match='parameter'):
        current(data)
    current = model()
    with torch.no_grad():
        current.value.bias[0] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        current(data)


@pytest.mark.parametrize('mode,kwargs', (
    ('full', {}), ('trace', {}), ('constant', {'temperature': 3.}),
    ('trace', {'bandwidth': 0., 'temperature': 3.}),
    ('trace', {'bandwidth': float('inf'), 'temperature': 3.}),
    ('trace', {'bandwidth': 1.e-200, 'temperature': 3.}),
    ('trace', {'bandwidth': 1.e200, 'temperature': 3.}),
    ('trace', {'bandwidth': .625, 'temperature': -1.}),
    ('trace', {'bandwidth': .625, 'temperature': float('nan')}),
    ('trace', {'bandwidth': True, 'temperature': 3.}),
    ('trace', {'bandwidth': .625, 'temperature': torch.tensor(3.)}),
    ('singleview', {'bandwidth': 1.}), ('singleview', {'temperature': 3.})))
def test_constructor_requires_explicit_valid_interface(mode, kwargs):
    with pytest.raises(ValueError):
        DepthAttention(mode, **kwargs)


def test_constant_temperature_float32_overflow_is_rejected(data):
    with pytest.raises(ValueError):
        model('constant', math.ldexp(1., 200))(data)


def test_geometry_changes_routing_and_all_missing_labels_loss_is_zero(data):
    current = model()
    prediction, attention, _ = current(data, return_attention=True)
    altered = dict(data, geometry_logits=data['geometry_logits'].flip(-1))
    changed, changed_attention, _ = current(altered, return_attention=True)
    assert bool((attention - changed_attention).abs().max() > 0)
    assert bool((prediction - changed).abs().max() > 0)
    loss = depth_loss(prediction, torch.full_like(prediction, float('nan')),
                      torch.zeros_like(prediction, dtype=torch.bool))
    assert bool(torch.isfinite(loss)) and float(loss) == 0.
