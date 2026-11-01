"""Independent Gaussian-expectation identities and non-equivalence controls.

The reference uses complex characteristic functions, not the repository's
rotary/expected_phase helpers. No experiment labels or checkpoints are loaded.
"""
import math

import pytest
import torch
import torch.nn.functional as F

from models.gaussian_pe import normalized_rotary


def expected_from_characteristic(content, mean, covariance, frequency):
    """Ordinary-range reference: E[exp(i*w.X)] times complex content."""
    variance = ((covariance @ frequency.T)*frequency.T).sum(dim=-2)
    coefficient = torch.polar(torch.exp(-variance/2), mean @ frequency.T)
    z = torch.view_as_complex(content.reshape(*content.shape[:-1], -1, 2).contiguous())
    return torch.view_as_real(z*coefficient).flatten(-2)


def fixture(dtype=torch.float64):
    generator = torch.Generator().manual_seed(5827)
    content = torch.randn(2, 5, 48, dtype=dtype, generator=generator)
    mean = torch.randn(2, 5, 2, dtype=dtype, generator=generator)
    factor = torch.randn(2, 5, 2, 2, dtype=dtype, generator=generator)*.2
    covariance = factor @ factor.transpose(-1, -2)+torch.eye(2, dtype=dtype)*.03
    frequency = torch.randn(24, 2, dtype=dtype, generator=generator)
    return content, mean, covariance, frequency


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('content_scale', [.125, 1., 9.])
def test_current_matcher_map_is_postnormalized_expected_gaussian(dtype, content_scale):
    raw, mean, covariance, frequency = fixture(dtype)
    content = F.normalize(raw*content_scale, dim=-1)
    actual = normalized_rotary(content, mean, covariance, frequency)
    expected = F.normalize(expected_from_characteristic(raw, mean, covariance, frequency), dim=-1)
    tolerance = 6e-7 if dtype == torch.float32 else 2e-13
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)


def test_raw_content_call_restores_original_norm_instead_of_unit_norm():
    content, mean, covariance, frequency = fixture()
    expected = expected_from_characteristic(content, mean, covariance, frequency)
    actual = normalized_rotary(content, mean, covariance, frequency)
    torch.testing.assert_close(actual,
        content.norm(dim=-1, keepdim=True)*F.normalize(expected, dim=-1),
        rtol=2e-13, atol=2e-13)
    assert not torch.allclose(actual, F.normalize(expected, dim=-1))


def test_postnormalized_expectation_has_matching_content_mean_covariance_gradients():
    values = fixture()
    inputs = [x.detach().requires_grad_() for x in values[:3]]
    frequency = values[3]
    weight = torch.linspace(-.8, 1.2, 48, dtype=torch.float64)
    actual = normalized_rotary(F.normalize(inputs[0], dim=-1), inputs[1], inputs[2], frequency)
    expected = F.normalize(expected_from_characteristic(*inputs, frequency), dim=-1)
    gradients_a = torch.autograd.grad((actual*weight).sum(), inputs, retain_graph=True)
    gradients_b = torch.autograd.grad((expected*weight).sum(), inputs)
    for actual_gradient, expected_gradient in zip(gradients_a, gradients_b):
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=2e-12, atol=2e-12)


def test_probabilistic_query_key_raw_score_uses_sum_covariance():
    q, mq, cq, frequency = fixture()
    k = q.flip(-2)*.7
    mk, ck = mq.flip(-2)+.2, cq.flip(-3)*1.3
    eq = expected_from_characteristic(q, mq, cq, frequency)
    ek = expected_from_characteristic(k, mk, ck, frequency)
    relative_key = expected_from_characteristic(k, mk-mq, cq+ck, frequency)
    torch.testing.assert_close((eq*ek).sum(-1), (q*relative_key).sum(-1),
        rtol=2e-13, atol=2e-13)
    post_score = (F.normalize(eq, dim=-1)*F.normalize(ek, dim=-1)).sum(-1)
    torch.testing.assert_close(post_score,
        (q*relative_key).sum(-1)/(eq.norm(dim=-1)*ek.norm(dim=-1)),
        rtol=2e-13, atol=2e-13)


def test_pre_normalization_does_not_replace_post_normalization():
    # Equal frequency norms make isotropic attenuation a scalar per token.
    # Pre-norm retains different key magnitudes; post-norm removes them.
    content = torch.ones(2, 4, dtype=torch.float64)/2
    mean = torch.zeros(2, 2, dtype=torch.float64)
    frequency = torch.eye(2, dtype=torch.float64)
    covariance = torch.stack([torch.eye(2, dtype=torch.float64)*.02,
        torch.eye(2, dtype=torch.float64)*2.])
    key_pre = expected_from_characteristic(content, mean, covariance, frequency)
    key_post = normalized_rotary(content, mean, covariance, frequency)
    score_pre, score_post = key_pre @ content[0], key_post @ content[0]
    assert score_pre[0]-score_pre[1] > .6
    torch.testing.assert_close(score_post, torch.ones(2, dtype=torch.float64))
    assert not torch.allclose(score_pre.softmax(-1), score_post.softmax(-1))


def test_fixed_finite_query_support_is_not_deterministic_query_rope():
    content = torch.ones(1, 4, dtype=torch.float64)/2
    mean = torch.tensor([[.2, -.3]], dtype=torch.float64)
    frequency = torch.tensor([[1., 0.], [3., 0.]], dtype=torch.float64)
    zero = torch.zeros(2, 2, dtype=torch.float64)
    fixed = torch.eye(2, dtype=torch.float64)*(1/12+.25**2)
    point = expected_from_characteristic(content, mean, zero, frequency)
    torch.testing.assert_close(normalized_rotary(content, mean, zero, frequency), point)
    finite_support = normalized_rotary(content, mean, fixed, frequency)
    assert (finite_support-point).norm() > .2


def test_moment_matching_a_mixture_does_not_preserve_expected_rotary():
    # Same mean/covariance, but the two-point phase mixture can be negative.
    content = torch.ones(1, 4, dtype=torch.float64)/2
    frequency = torch.tensor([[1.], [.5]], dtype=torch.float64)
    covariance = torch.tensor([[.01]], dtype=torch.float64)
    left = expected_from_characteristic(content, torch.tensor([[-math.pi]], dtype=torch.float64),
        covariance, frequency)
    right = expected_from_characteristic(content, torch.tensor([[math.pi]], dtype=torch.float64),
        covariance, frequency)
    mixture = (left+right)/2
    moment_covariance = covariance+math.pi**2
    moment = expected_from_characteristic(content, torch.zeros(1, 1, dtype=torch.float64),
        moment_covariance, frequency)
    assert mixture[0, 0] < -.49
    assert moment[0, 0] > 0
    assert (F.normalize(mixture, dim=-1)-F.normalize(moment, dim=-1)).norm() > 1.


def test_stability_changes_floating_evaluation_not_the_symbolic_operator():
    content = torch.ones(1, 4, dtype=torch.float64)/2
    mean = torch.zeros(1, 2, dtype=torch.float64)
    frequency = torch.tensor([[1., 0.], [2., 0.]], dtype=torch.float64)
    covariance = torch.eye(2, dtype=torch.float64)*2000
    # Naive double exponentials underflow before a subsequent F.normalize.
    naive = F.normalize(expected_from_characteristic(content, mean, covariance, frequency), dim=-1)
    stable = normalized_rotary(content, mean, covariance, frequency)
    assert torch.equal(naive, torch.zeros_like(naive))
    torch.testing.assert_close(stable.norm(dim=-1), torch.ones(1, dtype=torch.float64))
    torch.testing.assert_close(stable[0, :2], torch.ones(2, dtype=torch.float64)/math.sqrt(2))
