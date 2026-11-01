"""Independent algebra and supervision audit for compact world-space PE.

These checks establish the implemented operator, not task effectiveness or
novelty. Covariance-gradient checks use strictly positive covariance; the
isometric completion has a square-root derivative singularity at zero.
"""
import math
import unittest

import torch
import torch.nn.functional as F

from experiments.region_pe_probe import soft_loss
from models.gaussian_world_pe import (
    METHODS, GaussianWorldMatcher, isometric_rotary, log_bhattacharyya,
    moment_descriptor, normalized_world_geometry, oblique_world_frequencies,
)


def fixture():
    generator = torch.Generator().manual_seed(412)
    result = {}
    for side, count in [('q', 3), ('k', 7)]:
        result[side] = torch.randn(2, count, 108, generator=generator, dtype=torch.float64)
        result['m'+side] = torch.randn(2, count, 3, generator=generator, dtype=torch.float64)*.2
        factor = torch.randn(2, count, 3, 3, generator=generator, dtype=torch.float64)*.1
        result['c'+side] = factor @ factor.transpose(-1, -2)+torch.eye(3)*.03
    result['key_valid'] = torch.ones(2, 7, dtype=torch.bool)
    result['key_valid'][0, 2] = False
    result['q_valid'] = torch.ones(2, 3, dtype=torch.bool)
    result['soft_labels'] = torch.softmax(torch.randn(2, 3, 7, generator=generator, dtype=torch.float64), -1)
    result['soft_labels'][0, :, 2] = 0
    result['soft_labels'] /= result['soft_labels'].sum(-1, keepdim=True)
    result['label_valid'] = torch.tensor([[True, False, True], [True, True, True]])
    return result


def explicit_rotary(content, coordinate, frequencies):
    pairs = content.reshape(*content.shape[:-1], -1, 2)
    angle = coordinate @ frequencies.T
    a, b = pairs.unbind(-1)
    return torch.stack([a*angle.cos()-b*angle.sin(), a*angle.sin()+b*angle.cos()], -1).flatten(-2)


class WorldPEAudit(unittest.TestCase):
    def test_all_operators_are_query_and_key_permutation_equivariant(self):
        data = fixture()
        query_order = torch.tensor([2, 0, 1])
        key_order = torch.tensor([5, 1, 6, 0, 2, 4, 3])
        permuted = dict(data)
        for field in ['q', 'mq', 'cq', 'q_valid']:
            permuted[field] = data[field][:, query_order]
        for field in ['k', 'mk', 'ck', 'key_valid']:
            permuted[field] = data[field][:, key_order]
        for method in METHODS:
            with self.subTest(method=method):
                torch.manual_seed(19)
                model = GaussianWorldMatcher(method).double()
                original = model(data)
                expected = torch.cat([original[:, query_order, :-1][:, :, key_order],
                                      original[:, query_order, -1:]], -1)
                torch.testing.assert_close(model(permuted), expected, atol=1e-10, rtol=1e-10)

    def test_moment_rope_is_exactly_descriptor_rotary_with_common_content(self):
        data = fixture()
        model = GaussianWorldMatcher('world_moment_rope').double()
        mq, cq, mk, ck, _ = normalized_world_geometry(data)
        q, k = F.normalize(model.encoder(data['q']), dim=-1), F.normalize(model.encoder(data['k']), dim=-1)
        fq = explicit_rotary(q[..., :24], mq, model.freq3[model.center_indices])
        fk = explicit_rotary(k[..., :24], mk, model.freq3[model.center_indices])
        dq = explicit_rotary(q[..., 24:], moment_descriptor(cq), model.moment_freq)
        dk = explicit_rotary(k[..., 24:], moment_descriptor(ck), model.moment_freq)
        expected = model.logtemp.exp()*(fq @ fk.transpose(-1, -2)+dq @ dk.transpose(-1, -2))
        expected.masked_fill_(~data['key_valid'][:, None, :], -1e4)
        torch.testing.assert_close(model(data)[..., :-1], expected, atol=1e-11, rtol=1e-11)
        # The descriptor is a redundant, invertible reparameterization of SPD
        # covariance: six normalized entries plus log trace, one trace constraint.
        descriptor = moment_descriptor(cq)
        trace = descriptor[..., -1].exp()
        reconstructed = torch.zeros_like(cq)
        indices = torch.triu_indices(3, 3)
        weight = torch.where(indices[0] == indices[1], 1., math.sqrt(2)).double()
        entries = descriptor[..., :6]/weight*trace[..., None]
        reconstructed[..., indices[0], indices[1]] = entries
        reconstructed[..., indices[1], indices[0]] = entries
        torch.testing.assert_close(reconstructed, cq)

    def test_isometric_pair_inner_product_matches_derived_two_term_kernel(self):
        data = fixture()
        frequency = oblique_world_frequencies().double()
        generator = torch.Generator().manual_seed(7)
        q = torch.randn(2, 3, 48, generator=generator, dtype=torch.float64)
        k = torch.randn(2, 7, 48, generator=generator, dtype=torch.float64)
        vq = torch.einsum('fd,...de,fe->...f', frequency, data['cq'], frequency)
        vk = torch.einsum('fd,...de,fe->...f', frequency, data['ck'], frequency)
        aq, ak = (-vq/2).exp(), (-vk/2).exp()
        bq, bk = (-torch.expm1(-vq)).sqrt(), (-torch.expm1(-vk)).sqrt()
        rq = explicit_rotary(q, data['mq'], frequency).reshape(2, 3, 24, 2)
        rk = explicit_rotary(k, data['mk'], frequency).reshape(2, 7, 24, 2)
        pairs_q, pairs_k = q.reshape(2, 3, 24, 2), k.reshape(2, 7, 24, 2)
        expected = torch.einsum('bifd,bjfd,bif,bjf->bij', rq, rk, aq, ak)
        expected += torch.einsum('bifd,bjfd,bif,bjf->bij', pairs_q, pairs_k, bq, bk)
        actual = isometric_rotary(q, data['mq'], data['cq'], frequency)
        actual = actual @ isometric_rotary(k, data['mk'], data['ck'], frequency).transpose(-1, -2)
        torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-11)

    def test_positive_isotropic_covariance_has_finite_geometry_gradients(self):
        # No eigenvector differentiation is used in these two candidates.
        for method in ['world_moment_rope', 'world_isometric']:
            with self.subTest(method=method):
                data = fixture()
                data['cq'] = (torch.eye(3, dtype=torch.float64).expand(2, 3, 3, 3)*.08).clone().requires_grad_()
                data['ck'] = (torch.eye(3, dtype=torch.float64).expand(2, 7, 3, 3)*.06).clone().requires_grad_()
                data['mq'].requires_grad_()
                data['mk'].requires_grad_()
                model = GaussianWorldMatcher(method).double()
                loss = soft_loss(model(data), data)
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                for field in ['mq', 'mk', 'cq', 'ck']:
                    self.assertIsNotNone(data[field].grad, field)
                    self.assertTrue(torch.isfinite(data[field].grad).all(), field)

    def test_region_loss_does_not_train_unused_dustbin_parameters(self):
        data = fixture()
        model = GaussianWorldMatcher('world_moment_rope').double()
        soft_loss(model(data), data).backward()
        for name in ['dustbin', 'dustbin_head.weight', 'dustbin_head.bias']:
            gradient = dict(model.named_parameters())[name].grad
            self.assertTrue(gradient is None or bool((gradient == 0).all()), name)
        self.assertGreater(float(model.encoder[0].weight.grad.norm()), 0)
        self.assertGreater(float(model.logtemp.grad.abs()), 0)

    def test_exact_bc_invariance_is_stronger_than_fixed_frequency_rope(self):
        data = fixture()
        a = torch.tensor([[1.2, .3, -.1], [.1, .8, .2], [.2, -.2, 1.4]], dtype=torch.float64)
        shift = torch.tensor([.8, -2., .5], dtype=torch.float64)
        changed = dict(data, mq=data['mq'] @ a.T+shift, mk=data['mk'] @ a.T+shift,
                       cq=a @ data['cq'] @ a.T, ck=a @ data['ck'] @ a.T)
        kernel = log_bhattacharyya(data['mq'], data['cq'], data['mk'], data['ck'])
        affine = log_bhattacharyya(changed['mq'], changed['cq'], changed['mk'], changed['ck'])
        torch.testing.assert_close(kernel, affine, atol=1e-10, rtol=1e-10)
        # A fixed finite set of world directions is not an SO(3)-invariant map.
        rotation, _ = torch.linalg.qr(a)
        rotated = dict(data, mq=data['mq'] @ rotation.T, mk=data['mk'] @ rotation.T,
                       cq=rotation @ data['cq'] @ rotation.T,
                       ck=rotation @ data['ck'] @ rotation.T)
        model = GaussianWorldMatcher('world_moment_rope').double()
        self.assertGreater(float((model(data)-model(rotated)).abs().max()), 1e-4)


if __name__ == '__main__':
    unittest.main()
