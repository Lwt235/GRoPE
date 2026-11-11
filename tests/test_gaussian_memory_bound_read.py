import numpy as np
import torch

from models.gaussian_memory_bound_read import BoundMemoryAdapter, attention_prior, projected_slot_weights
from models.gaussian_memory_vggt_adapter import project_memory


def scene(n=8):
    torch.manual_seed(19)
    means = torch.randn(n, 3) * .3; means[:, 2] += 2
    memory = {'means': means, 'covars': torch.eye(3)[None].repeat(n, 1, 1) * .06,
              'opacity': torch.full((n,), .7), 'features': torch.randn(n, 32)}
    cameras = torch.eye(4)[None].repeat(4, 1, 1)
    intrinsic = torch.tensor([[12., 0., 10.5], [0., 12., 7.5], [0., 0., 1.]])[None].repeat(4, 1, 1)
    return memory, cameras, intrinsic


def test_front_to_back_weights_keep_original_slot_identity():
    memory, cameras, intrinsic = scene(2)
    memory['means'] = torch.tensor([[0., 0., 2.], [0., 0., 1.]])
    p = projected_slot_weights(memory, cameras, intrinsic)
    pixel = 7 * 20 + 10
    torch.testing.assert_close(p['weights'][0, pixel], torch.tensor([.21, .7]))
    torch.testing.assert_close(p['transmittance'][0, pixel], torch.tensor(.09))
    torch.testing.assert_close(p['coverage'], project_memory(memory, cameras, intrinsic)['coverage'])
    torch.testing.assert_close(p['coverage'] + p['transmittance'], torch.ones_like(p['coverage']))


def test_empty_support_is_zero_even_after_learning():
    memory, cameras, intrinsic = scene()
    memory['means'][:, 2] = -2
    p = projected_slot_weights(memory, cameras, intrinsic)
    adapter = BoundMemoryAdapter()
    torch.nn.init.normal_(adapter.out.weight, std=.01)
    tokens = torch.randn(1, 4, 305, 1024)
    result = adapter.content_residual(tokens, {'content': memory['features'], 'log_prior': attention_prior(p, 'bound', 8)})
    assert torch.equal(result, tokens)


def test_binding_distinguishes_content_only_from_joint_permutation():
    memory, cameras, intrinsic = scene()
    p = projected_slot_weights(memory, cameras, intrinsic)
    adapter = BoundMemoryAdapter(); torch.nn.init.normal_(adapter.out.weight, std=.01)
    assert sum(x.numel() for x in adapter.parameters()) == 137472
    tokens = torch.randn(1, 4, 305, 1024)
    prior = attention_prior(p, 'bound', 8)
    run = lambda c, b: adapter.content_residual(tokens, {'content': c, 'log_prior': b})
    normal = run(memory['features'], prior)
    perm = torch.tensor([3, 1, 6, 0, 4, 7, 2, 5])
    joint = {k: v[perm] for k, v in memory.items()}
    other = projected_slot_weights(joint, cameras, intrinsic)
    torch.testing.assert_close(normal, run(joint['features'], attention_prior(other, 'bound', 8)), atol=3e-6, rtol=1e-5)
    assert (normal - run(joint['features'], prior)).abs().max() > 1e-4
    covered = attention_prior(p, 'coverage', 8)
    torch.testing.assert_close(run(memory['features'], covered), run(joint['features'], covered), atol=3e-6, rtol=1e-5)


def test_zero_initialization_and_nonzero_learning_gradient():
    memory, cameras, intrinsic = scene()
    p = projected_slot_weights(memory, cameras, intrinsic)
    adapter = BoundMemoryAdapter(); tokens = torch.randn(1, 4, 305, 1024)
    payload = {'content': memory['features'], 'log_prior': attention_prior(p, 'bound', 8)}
    result = adapter.content_residual(tokens, payload)
    assert torch.equal(result, tokens)
    (result * torch.randn_like(result)).mean().backward()
    assert torch.isfinite(adapter.out.weight.grad).all() and adapter.out.weight.grad.norm() > 0
