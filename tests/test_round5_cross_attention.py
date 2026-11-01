"""Focused architecture, symmetry, gradient and input-boundary checks."""
import pytest
import torch

from models.round5_cross_attention import METHODS, Round5CrossAttentionMatcher


def fixture():
    generator = torch.Generator().manual_seed(5147)
    data = {'q': torch.randn(2, 6, 128, generator=generator),
            'k': torch.randn(2, 7, 128, generator=generator)}
    for name, tokens in [('q', 6), ('k', 7), ('q_to_k', 6), ('k_to_q', 7)]:
        key = name+'mean' if name in ('q', 'k') else name
        data[key] = torch.randn(2, tokens, 2, generator=generator)
        factor = torch.randn(2, tokens, 2, 2, generator=generator)*.3
        data[name+'cov'] = factor @ factor.transpose(-1, -2)+torch.eye(2)*.03
    return data


def model(method='normalized'):
    torch.manual_seed(814)
    return Round5CrossAttentionMatcher(method)


def test_methods_have_identical_parameters_initialization_and_shape():
    reference = model()
    for method in METHODS:
        candidate = model(method)
        assert list(candidate.state_dict()) == list(reference.state_dict())
        assert all(torch.equal(v, candidate.state_dict()[k])
                   for k, v in reference.state_dict().items())
        assert sum(p.numel() for p in candidate.parameters()) == sum(p.numel() for p in reference.parameters())
        result = candidate(fixture())
        assert result.shape == (2, 6, 7)
        assert torch.isfinite(result).all()
    assert reference.layers[0].qkv is not reference.layers[1].qkv
    assert torch.count_nonzero(reference.layers[0].output.weight) > 0


def test_each_layer_uses_values_with_nonzero_gradients():
    candidate = model()
    data = fixture()
    result = candidate(data)
    weight = torch.linspace(-1, 1, result.numel()).reshape_as(result)
    (result*weight).sum().backward()
    for layer in candidate.layers:
        assert torch.isfinite(layer.qkv.weight.grad).all()
        assert layer.qkv.weight.grad[192:].norm() > 1e-6
        assert layer.output.weight.grad.norm() > 1e-6
    assert not torch.allclose(candidate(data), candidate(data, zero_values=True))
    candidate.zero_grad(set_to_none=True)
    candidate(data, zero_values=True).square().sum().backward()
    for layer in candidate.layers:
        assert torch.equal(layer.qkv.weight.grad[192:], torch.zeros_like(layer.qkv.weight.grad[192:]))


@pytest.mark.parametrize('field,updates_query', [
    ('qcov', True), ('k_to_qcov', True), ('kcov', False), ('q_to_kcov', False)])
def test_covariance_is_used_in_the_correct_direction(field, updates_query):
    candidate = model()
    data = fixture()
    outputs = []
    handle = candidate.layers[0].register_forward_hook(
        lambda _module, _inputs, output: outputs.append(tuple(x.detach().clone() for x in output)))
    with torch.no_grad():
        candidate(data)
        changed = dict(data)
        changed[field] = data[field]+torch.tensor([[2., .5], [.5, .3]])
        candidate(changed)
    handle.remove()
    affected, unaffected = (0, 1) if updates_query else (1, 0)
    assert (outputs[0][affected]-outputs[1][affected]).abs().max() > 1e-5
    torch.testing.assert_close(outputs[0][unaffected], outputs[1][unaffected], rtol=0, atol=0)


@pytest.mark.parametrize('method', METHODS)
def test_direction_swap_transposes_all_dense_scores(method):
    data = fixture()
    swapped = {name: data[other] for name, other in [
        ('q', 'k'), ('k', 'q'), ('qmean', 'kmean'), ('kmean', 'qmean'),
        ('qcov', 'kcov'), ('kcov', 'qcov'), ('k_to_q', 'q_to_k'),
        ('q_to_k', 'k_to_q'), ('k_to_qcov', 'q_to_kcov'), ('q_to_kcov', 'k_to_qcov')]}
    candidate = model(method)
    torch.testing.assert_close(candidate(data), candidate(swapped).transpose(-1, -2),
        rtol=2e-6, atol=2e-6)


def test_label_and_visibility_fields_are_ignored():
    candidate = model()
    data = fixture()
    expected = candidate(data)
    data.update(labels=torch.full((2, 6), -999), visible=torch.zeros(2, 6, dtype=torch.bool),
        key_mask=torch.zeros(2, 7, dtype=torch.bool), attention_mask=torch.ones(2, 6, 7)*-1e9,
        target_depth=object(), soft_labels=object())
    torch.testing.assert_close(candidate(data), expected, rtol=0, atol=0)


def test_controls_ignore_the_intended_covariance_information():
    data = fixture()
    changed = dict(data)
    for field in ('qcov', 'kcov', 'q_to_kcov', 'k_to_qcov'):
        covariance = data[field]
        rotation = torch.tensor([[0., -1.], [1., 0.]])
        changed[field] = rotation @ covariance @ rotation.T
    for method in ('point', 'fixed', 'isotropic'):
        candidate = model(method)
        torch.testing.assert_close(candidate(data), candidate(changed), rtol=2e-6, atol=2e-6)
    candidate = model()
    assert not torch.allclose(candidate(data), candidate(changed))
    # With V aggregation disabled, geometry has no route to the final head.
    torch.testing.assert_close(candidate(data, zero_values=True),
        candidate(changed, zero_values=True), rtol=0, atol=0)


def test_invalid_method_and_resolution_fail_early():
    with pytest.raises(ValueError, match='positional method'):
        Round5CrossAttentionMatcher('worldrope')
    for resolution in (-.1, float('nan'), float('inf')):
        with pytest.raises(ValueError, match='resolution'):
            Round5CrossAttentionMatcher('normalized', resolution=resolution)
