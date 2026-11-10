"""Check the unchanged relation and independently reconstruct rotary attention."""
import math
import unittest
import numpy as np
import torch

from models.round58_support_pe import AttentionBlock, DIM, HEADS, HD
from models.region_hrope_control import CameraRegionMatcher, hrope_layer
from models.region_rotary_relation import ROLES, RotaryRelationMatcher, rotary_relation_layer


class RotaryRelationTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(4)
        torch.manual_seed(823)

    def test_no_relation_recovers_fixed_transport_and_common_gradients(self):
        layer = AttentionBlock().double()
        x = torch.randn(2, 2, 7, DIM, dtype=torch.float64, requires_grad=True)
        matrix = torch.eye(3).expand(2, 2, 7, 3, 3).clone().double()
        matrix[:, 0, :, 0, 1] = .2
        matrix[:, 1, :, 1, 1] = .8
        inverse = torch.linalg.inv(matrix)
        xy = torch.randn(2, 2, 7, 2, dtype=torch.float64)
        for cross in (False, True):
            a = hrope_layer(layer, x, matrix, inverse, xy, cross)
            b = rotary_relation_layer(layer, x, matrix, inverse, xy, cross=cross)
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            params = (x, *layer.parameters())
            ga = torch.autograd.grad(a.square().mean(), params, retain_graph=True)
            gb = torch.autograd.grad(b.square().mean(), params)
            for left, right in zip(ga, gb):
                torch.testing.assert_close(left, right, rtol=0, atol=0)

    def test_identity_transport_restores_each_original_relation_model(self):
        b, v, n = 2, 2, 7
        args = (torch.randn(b, v, n, 11, dtype=torch.float64),
                torch.randn(b, v, n, 3, dtype=torch.float64),
                torch.eye(3, dtype=torch.float64).expand(b, v, n, 3, 3)*.03,
                torch.arange(n).expand(b, v, n),
                torch.eye(3, dtype=torch.float64).expand(b, v, 3, 3),
                torch.zeros(b, v, n, 2, dtype=torch.float64))
        for role in ROLES:
            torch.manual_seed(17); old = CameraRegionMatcher(role).double()
            torch.manual_seed(17); new = RotaryRelationMatcher(role).double()
            self.assertEqual(sum(p.numel() for p in old.parameters()), sum(p.numel() for p in new.parameters()))
            left, right = dict(old.named_parameters()), dict(new.named_parameters())
            self.assertEqual(left.keys(), right.keys())
            for name in left:
                torch.testing.assert_close(left[name], right[name], rtol=0, atol=0)
            torch.testing.assert_close(old(*args), new(*args), rtol=1e-12, atol=1e-12)

    def test_nonzero_relation_with_independent_numpy_block_matrices(self):
        layer = AttentionBlock().double()
        n = 5
        x = torch.randn(1, 2, n, DIM, dtype=torch.float64)
        xy = torch.randn(1, 2, n, 2, dtype=torch.float64)
        matrix = torch.eye(3).expand(1, 2, n, 3, 3).clone().double()
        matrix[..., 0, 1] = .2
        matrix[:, 1, :, 1, 1] = .8
        inverse = torch.linalg.inv(matrix)
        relative = torch.randn(1, 2, n, n, DIM, dtype=torch.float64)
        with torch.no_grad():
            q, k, v = layer.qkv(layer.norm(x)).reshape(1, 2, n, 3, HEADS, HD).permute(3, 0, 1, 4, 2, 5).numpy()
            rel = relative.numpy().reshape(1, 2, n, n, HEADS, HD)
            def block(view, token, which):
                t = np.zeros((HD, HD))
                h = matrix.numpy()[0, view, token]
                geometric = h.T if which == 'q' else np.linalg.inv(h) if which in ('k', 'v') else h
                for i in range(8):
                    t[3*i:3*i+3, 3*i:3*i+3] = geometric
                phase = np.outer(8*xy.numpy()[0, view, token], 10000.**(-2*np.arange(1, 7)/24)).ravel()
                for i, angle in enumerate(phase):
                    angle *= 1 if which == 'out' else -1
                    c, s = np.cos(angle), np.sin(angle)
                    t[24+2*i:26+2*i, 24+2*i:26+2*i] = [[c, -s], [s, c]]
                return t
            for cross in (False, True):
                aggregate = np.empty_like(q)
                for view in range(2):
                    source = 1-view if cross else view
                    for head in range(HEADS):
                        keys = np.stack([block(source, j, 'k')@k[0, source, head, j] for j in range(n)])
                        values = np.stack([block(source, j, 'v')@v[0, source, head, j] for j in range(n)])
                        for i in range(n):
                            score = keys@(block(view, i, 'q')@q[0, view, head, i])
                            if not cross:
                                score += rel[0, view, i, :, head]@q[0, view, head, i]
                            score /= math.sqrt(HD)
                            p = np.exp(score-score.max()); p /= p.sum()
                            aggregate[0, view, head, i] = block(view, i, 'out')@(p@values)
                y = torch.from_numpy(aggregate.transpose(0, 1, 3, 2, 4).reshape(1, 2, n, DIM))
                y = x+layer.output(y); expected = y+layer.ffn(layer.ffn_norm(y))
                actual = rotary_relation_layer(layer, x, matrix, inverse, xy, None if cross else relative, cross)
                torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-11)


if __name__ == '__main__':
    unittest.main()
