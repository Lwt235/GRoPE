"""Full-logit checkpoint conversion, including the nonlinear dustbin input."""
import pytest
import torch
from models.gaussian_spectral_pe import GaussianSpectralMatcher
from models.gaussian_relative_pe import RelativeSupportMatcher
from tests.test_normalized_pe import model_fixture


@pytest.mark.parametrize('resolution', [0., .25, .5])
def test_full_logits_survive_fixed_query_conversion(resolution):
    torch.manual_seed(9412)
    original = GaussianSpectralMatcher('gaussian_konly', 16, resolution).double()
    data = {k: v.double() for k, v in model_fixture().items()}
    converted = RelativeSupportMatcher(original)
    # Positional buffers were originally constructed in float32; the common
    # fixed covariance follows the original floating evaluation convention.
    torch.testing.assert_close(converted(data), original(data), rtol=1e-12, atol=1e-12)
    assert sum(p.numel() for p in converted.parameters()) == sum(p.numel() for p in original.parameters())


def test_conversion_does_not_mutate_source_or_allow_variable_query():
    original = GaussianSpectralMatcher('gaussian_konly', 16, .25)
    before = {k: v.clone() for k, v in original.state_dict().items()}
    RelativeSupportMatcher(original)
    assert all(torch.equal(before[k], v) for k, v in original.state_dict().items())
    with pytest.raises(ValueError, match='fixed-query'):
        RelativeSupportMatcher(GaussianSpectralMatcher('gaussian_full', 16, .25))


def test_float32_accuracy_with_negative_relative_covariance_and_rejection_head():
    original = GaussianSpectralMatcher('gaussian_konly', 16, .25)
    converted = RelativeSupportMatcher(original)
    data = model_fixture()
    data['projected_kcov'] = torch.eye(2).expand_as(data['projected_kcov'])*.0001
    torch.testing.assert_close(converted(data), original(data), rtol=2e-6, atol=3e-6)
