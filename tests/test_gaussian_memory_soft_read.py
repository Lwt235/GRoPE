import numpy as np
import torch

from models.gaussian_memory_bound_read import BoundMemoryAdapter, attention_prior
from models.gaussian_memory_soft_read import soft_attention_prior


def projected(weights):
    coverage = weights.sum(-1)
    return {'weights': weights, 'coverage': coverage, 'transmittance': 1 - coverage}


def test_zero_coverage_exactly_matches_parent_content_prior():
    p = projected(torch.zeros(4, 300, 256))
    assert torch.equal(soft_attention_prior(p), attention_prior(p, 'content'))


def test_nonzero_spatial_support_keeps_all_real_slots_accessible():
    weights = torch.zeros(4, 300, 256); weights[..., 7] = .7
    p = projected(weights); bias = soft_attention_prior(p)
    assert torch.isfinite(bias[..., :-1]).all() and torch.isneginf(bias[..., -1]).all()
    assert (bias[..., 7] > bias[..., 8]).all()
    probability = bias.softmax(-1)
    expected = (weights + 1 / 256) / (1 + p['coverage'][..., None])
    torch.testing.assert_close(probability[..., :-1], expected)


def test_uniform_support_and_joint_permutation_in_learned_residual():
    torch.manual_seed(1703)
    adapter = BoundMemoryAdapter().eval()
    with torch.no_grad(): adapter.out.weight.normal_(std=.02)
    tokens = torch.randn(1, 4, 305, 1024); content = torch.randn(256, 32)
    weights = torch.rand(4, 300, 256); weights = weights / weights.sum(-1, keepdim=True) * .6
    p = projected(weights); uniform = projected(torch.full_like(weights, .6 / 256))
    with torch.no_grad():
        a = adapter.content_residual(tokens, {'content': content, 'log_prior': soft_attention_prior(uniform)})
        b = adapter.content_residual(tokens, {'content': content, 'log_prior': attention_prior(uniform, 'content')})
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
        prior = soft_attention_prior(p); permutation = torch.randperm(256)
        columns = torch.cat((permutation, torch.tensor([256])))
        a = adapter.content_residual(tokens, {'content': content, 'log_prior': prior})
        b = adapter.content_residual(tokens, {'content': content[permutation], 'log_prior': prior[..., columns]})
        c = adapter.content_residual(tokens, {'content': content[permutation], 'log_prior': prior})
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
        assert (a - c).abs().max() > 1e-4
    assert sum(p.numel() for p in adapter.parameters()) == 137472
