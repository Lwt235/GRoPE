"""Independent algebra/domain checks; no optimizer or experiment data are used."""
import copy

import pytest
import torch
import torch.nn.functional as F

from models.gaussian_pe import normalized_rotary
from models.geometry import rotary


def amplitude(covariance, frequency):
    return torch.exp(-torch.einsum('fd,...de,fe->...f',
        frequency, covariance, frequency)/2).repeat_interleave(2, dim=-1)


@pytest.mark.parametrize('resolution', [0., .25, .5])
def test_explicit_filter_ratio_and_affine_conversion(resolution):
    """A direct exponential calculation independently checks stabilized PE."""
    torch.manual_seed(2731)
    frequency = torch.randn(24, 2, dtype=torch.float64)
    encoder = torch.nn.Linear(7, 48).double()
    converted_encoder = copy.deepcopy(encoder)
    reference_cov = torch.eye(2, dtype=torch.float64)*(1/12+resolution**2)
    reference = amplitude(reference_cov, frequency)
    with torch.no_grad():
        converted_encoder.weight.mul_(reference[:, None])
        converted_encoder.bias.mul_(reference)
    inputs = torch.randn(5, 7, dtype=torch.float64)
    h, transformed = encoder(inputs), converted_encoder(inputs)
    torch.testing.assert_close(transformed, h*reference, rtol=1e-14, atol=1e-14)

    # A narrow physical key makes Crel negative definite; another is anisotropic.
    physical = torch.stack([torch.eye(2, dtype=torch.float64)*.001,
        torch.tensor([[.3, .06], [.06, .04]], dtype=torch.float64)])
    relative = physical-torch.eye(2, dtype=torch.float64)/12
    effective = physical+torch.eye(2, dtype=torch.float64)*resolution**2
    torch.testing.assert_close(amplitude(relative, frequency)*reference,
        amplitude(effective, frequency), rtol=1e-14, atol=1e-14)

    mean = torch.randn(5, 2, dtype=torch.float64)
    old_q = normalized_rotary(F.normalize(h, dim=-1), mean, reference_cov, frequency)
    new_q = rotary(F.normalize(transformed, dim=-1), mean, frequency)
    torch.testing.assert_close(old_q, new_q, rtol=2e-13, atol=2e-13)
    for total, ratio in zip(effective, relative):
        # The expected value uses no stabilized/relative PE helper.
        expected = rotary(F.normalize(h*amplitude(total, frequency), dim=-1), mean, frequency)
        actual = normalized_rotary(F.normalize(transformed, dim=-1), mean, ratio, frequency)
        torch.testing.assert_close(actual, expected, rtol=2e-13, atol=2e-13)

    # This is the content that must enter any pre-existing nonlinear dustbin head.
    recovered = F.normalize(transformed/reference, dim=-1)
    torch.testing.assert_close(recovered, F.normalize(h, dim=-1), rtol=1e-14, atol=1e-14)
    assert not torch.allclose(F.normalize(transformed, dim=-1), recovered)


def test_epsilon_branch_prevents_universal_tiny_content_identity():
    """The documented ordinary-norm assumption has a concrete counterexample."""
    h = torch.tensor([[5e-13, 0.]], dtype=torch.float64)
    mean = torch.zeros(1, 2, dtype=torch.float64)
    frequency = torch.tensor([[4., 0.]], dtype=torch.float64)
    covariance = torch.eye(2, dtype=torch.float64)*(1/12+.25**2)
    reference = amplitude(covariance, frequency)
    old = normalized_rotary(F.normalize(h, dim=-1), mean, covariance, frequency)
    converted = rotary(F.normalize(h*reference, dim=-1), mean, frequency)
    torch.testing.assert_close(old.norm(dim=-1), torch.tensor([.5], dtype=torch.float64))
    assert converted.norm() < .2
    assert (old-converted).abs().max() > .3
    # Recovery remains possible for the head; candidate Q/K norms are the issue.
    torch.testing.assert_close(F.normalize(h*reference/reference, dim=-1),
        F.normalize(h, dim=-1), rtol=1e-14, atol=1e-14)
