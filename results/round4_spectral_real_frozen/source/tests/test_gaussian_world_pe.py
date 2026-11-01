import unittest

import torch
import torch.nn.functional as F

from models.gaussian_pe import bhattacharyya
from models.gaussian_world_pe import (
    METHODS, GaussianWorldMatcher, isometric_rotary, isovolume_isotropic,
    log_bhattacharyya, moment_descriptor, normal_projector,
    normalized_world_geometry, oblique_world_frequencies, plane_isotropic,
    world_unit,
)
from models.geometry import rotary


class GaussianWorldPETest(unittest.TestCase):
    @staticmethod
    def data():
        generator = torch.Generator().manual_seed(18191)
        covariances = []
        for _ in range(2):
            a = torch.randn(2, 5, 3, 3, generator=generator, dtype=torch.float64)*.15
            covariances.append(a @ a.transpose(-1, -2)+torch.diag(torch.tensor([.02, .05, .1], dtype=torch.float64)))
        return {
            'q': torch.rand(2, 5, 108, generator=generator, dtype=torch.float64),
            'k': torch.rand(2, 5, 108, generator=generator, dtype=torch.float64),
            'mq': torch.randn(2, 5, 3, generator=generator, dtype=torch.float64)*.2,
            'mk': torch.randn(2, 5, 3, generator=generator, dtype=torch.float64)*.2,
            'cq': covariances[0], 'ck': covariances[1],
        }

    def test_common_scale_and_invalid_tokens_do_not_define_units(self):
        data = self.data()
        unit = world_unit(data['cq'], data['ck'])
        expected = torch.cat([data['cq'].diagonal(dim1=-2, dim2=-1).sum(-1), data['ck'].diagonal(dim1=-2, dim2=-1).sum(-1)], -1).median(-1).values.sqrt()
        torch.testing.assert_close(unit, expected)
        for scale in [.003, 17., 1000.]:
            scaled = dict(data, mq=data['mq']*scale, mk=data['mk']*scale,
                          cq=data['cq']*scale**2, ck=data['ck']*scale**2)
            original_geometry = normalized_world_geometry(data)
            scaled_geometry = normalized_world_geometry(scaled)
            for a, b in zip(original_geometry[:4], scaled_geometry[:4]):
                torch.testing.assert_close(a, b)
        q_valid = torch.ones(2, 5, dtype=torch.bool)
        q_valid[:, -1] = False
        masked = dict(data, q_valid=q_valid, cq=data['cq'].clone())
        before = normalized_world_geometry(masked)
        masked['cq'][:, -1] = float('nan')
        after = normalized_world_geometry(masked)
        for a, b in zip(before, after):
            torch.testing.assert_close(a, b)

    def test_every_method_scale_invariant_and_ignores_targets(self):
        data = self.data()
        scaled = dict(data, mq=data['mq']*13, mk=data['mk']*13,
                      cq=data['cq']*169, ck=data['ck']*169)
        polluted = dict(data, labels=torch.zeros(2, 5, dtype=torch.long),
                        track_counts=torch.randn(2, 5, 5), label_valid=torch.zeros(2, 5))
        for method in METHODS:
            with self.subTest(method=method):
                torch.manual_seed(25)
                model = GaussianWorldMatcher(method).double()
                output = model(data)
                torch.testing.assert_close(output, model(scaled), atol=2e-7, rtol=2e-7)
                torch.testing.assert_close(output, model(polluted), atol=0, rtol=0)
                self.assertEqual(output.shape, (2, 5, 6))
                self.assertTrue(torch.isfinite(output).all())
                output.log_softmax(-1)[..., 0].sum().backward()
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_isometric_norm_point_limit_and_shared_frequency(self):
        data = self.data()
        frequencies = oblique_world_frequencies().double()
        generator = torch.Generator().manual_seed(28)
        content = torch.randn(2, 5, 48, generator=generator, dtype=torch.float64)
        encoded = isometric_rotary(content, data['mq'], data['cq'], frequencies)
        torch.testing.assert_close(encoded.norm(dim=-1), content.norm(dim=-1))
        point = isometric_rotary(content, data['mq'], torch.zeros_like(data['cq']), frequencies)
        torch.testing.assert_close(point[..., :48], rotary(content, data['mq'], frequencies))
        torch.testing.assert_close(point[..., 48:], torch.zeros_like(content))
        for method in ['world_expected', 'world_isometric', 'world_moment_rope', 'world_normal_rope']:
            torch.testing.assert_close(GaussianWorldMatcher('world_rope').freq3, GaussianWorldMatcher(method).freq3)

    def test_support_controls_preserve_requested_invariants(self):
        covariance = self.data()['cq']
        plane = plane_isotropic(covariance)
        values = torch.linalg.eigvalsh(covariance)
        plane_values = torch.linalg.eigvalsh(plane)
        torch.testing.assert_close(values[..., 0], plane_values[..., 0])
        torch.testing.assert_close(values[..., 1:].prod(-1), plane_values[..., 1:].prod(-1))
        torch.testing.assert_close(plane_values[..., 1], plane_values[..., 2])
        torch.testing.assert_close(normal_projector(covariance), normal_projector(plane))
        spherical = isovolume_isotropic(covariance)
        torch.testing.assert_close(torch.linalg.det(covariance), torch.linalg.det(spherical))
        sphere_values = torch.linalg.eigvalsh(spherical)
        torch.testing.assert_close(sphere_values, sphere_values[..., :1].expand_as(sphere_values))

    def test_density_log_kernel_and_linear_normalization(self):
        data = self.data()
        mq, cq, mk, ck, _ = normalized_world_geometry(data)
        log_kernel = log_bhattacharyya(mq, cq, mk, ck)
        torch.testing.assert_close(log_kernel.exp(), bhattacharyya(mq, cq, mk, ck))
        model = GaussianWorldMatcher('world_density_linear').double()
        raw_q, raw_k = model.encoder(data['q']), model.encoder(data['k'])
        q, k = F.normalize(F.softplus(raw_q), dim=-1), F.normalize(F.softplus(raw_k), dim=-1)
        weights = (q @ k.transpose(-1, -2))*log_kernel.exp()
        actual = model(data)[..., :-1].softmax(-1)
        torch.testing.assert_close(actual, weights/weights.sum(-1, keepdim=True))
        # Changing logtemp changes real-vs-dustbin mass, not the prescribed
        # row-normalized real-candidate kernel.
        with torch.no_grad():
            model.logtemp.fill_(.4)
        torch.testing.assert_close(actual, model(data)[..., :-1].softmax(-1))

    def test_controls_share_original_unit_and_channel_assignment(self):
        data = self.data()
        mq, cq, mk, ck, _ = normalized_world_geometry(data)
        for method, operator in [('world_isometric_plane', plane_isotropic), ('world_isometric_isovolume', isovolume_isotropic)]:
            model = GaussianWorldMatcher(method).double()
            q = F.normalize(model.encoder(data['q']), dim=-1)
            k = F.normalize(model.encoder(data['k']), dim=-1)
            q = isometric_rotary(q, mq, operator(cq), model.freq3)
            k = isometric_rotary(k, mk, operator(ck), model.freq3)
            expected = model.logtemp.exp()*(q @ k.transpose(-1, -2))
            torch.testing.assert_close(model(data)[..., :-1], expected)
        model = GaussianWorldMatcher('world_center_half').double()
        q = F.normalize(model.encoder(data['q']), dim=-1)
        k = F.normalize(model.encoder(data['k']), dim=-1)
        frequency = model.freq3[model.center_indices]
        expected = rotary(q[..., :24], mq, frequency) @ rotary(k[..., :24], mk, frequency).transpose(-1, -2)
        expected += q[..., 24:] @ k[..., 24:].transpose(-1, -2)
        torch.testing.assert_close(model(data)[..., :-1], model.logtemp.exp()*expected)

    def test_moments_smooth_at_isotropy_and_gauge_safe(self):
        covariance = (torch.eye(3, dtype=torch.float64).expand(2, 3, 3)*.2).clone().requires_grad_()
        descriptor = moment_descriptor(covariance)
        descriptor.square().sum().backward()
        self.assertTrue(torch.isfinite(covariance.grad).all())
        a = torch.tensor([[.2, -.3, .1], [.4, .1, .5], [.7, .2, -.4]], dtype=torch.float64)
        rotation, _ = torch.linalg.qr(a)
        scales = torch.diag(torch.tensor([.03, .2, .4], dtype=torch.float64))
        c1 = rotation @ scales @ rotation.T
        sign = torch.diag(torch.tensor([-1., 1., -1.], dtype=torch.float64))
        c2 = (rotation @ sign) @ scales @ (rotation @ sign).T
        torch.testing.assert_close(moment_descriptor(c1), moment_descriptor(c2), atol=0, rtol=0)

    def test_translation_and_invalid_key_mask(self):
        data = self.data()
        shifted = dict(data, mq=data['mq']+torch.tensor([3., -4., 2.]), mk=data['mk']+torch.tensor([3., -4., 2.]))
        valid = torch.ones(2, 5, dtype=torch.bool)
        valid[:, 2] = False
        masked = dict(data, key_valid=valid)
        for method in METHODS:
            model = GaussianWorldMatcher(method).double()
            if method != 'world_absolute':
                torch.testing.assert_close(model(data), model(shifted), atol=1e-6, rtol=1e-6)
            output = model(masked)
            torch.testing.assert_close(output[..., 2], torch.full_like(output[..., 2], -1e4))
            self.assertTrue(torch.isfinite(output).all())


if __name__ == '__main__':
    unittest.main()
