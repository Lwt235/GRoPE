import ast
from pathlib import Path
import unittest

import numpy as np
import torch

from models.region_geometry_controls import GeometryControlMatcher, ROLES
from models.round69_compact_relation import CompactMatcher
from experiments.round58_run import shared_digest


class GeometryControlChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_common_initialization_and_disabled_relation(self):
        for seed in (11, 22, 33):
            torch.manual_seed(seed)
            parent = CompactMatcher('relative5').eval()
            for role in ROLES:
                torch.manual_seed(seed)
                model = GeometryControlMatcher(role).eval()
                self.assertEqual(shared_digest(parent), shared_digest(model))
                torch.manual_seed(1400+seed)
                f = torch.randn(1, 2, 7, 11)
                c = torch.randn(1, 2, 7, 3)
                a = torch.randn(1, 2, 7, 3, 3, dtype=torch.float64)
                cov = a@a.transpose(-1, -2)+.1*torch.eye(3, dtype=torch.float64)
                permutation = torch.arange(7).expand(1, 2, 7)
                hook = parent.position.register_forward_hook(lambda m, args, output: torch.zeros_like(output))
                with torch.no_grad():
                    expected = parent(f, c, cov, permutation)
                    actual = model(f, c, cov, permutation, disable_relation=True)
                hook.remove()
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_vendored_definitions_preserve_upstream_source(self):
        root = Path(__file__).resolve().parents[1]/'third_party/region_geotransformer_reference'
        adapted = ast.parse((root/'embedding.py').read_text())
        for filename, name in [('pairwise_distance.py', 'pairwise_distance'),
                               ('positional_embedding.py', 'SinusoidalPositionalEmbedding'),
                               ('geotransformer.py', 'GeometricStructureEmbedding')]:
            original = ast.parse((root/filename).read_text())
            left = next(n for n in original.body if getattr(n, 'name', None) == name)
            right = next(n for n in adapted.body if getattr(n, 'name', None) == name)
            self.assertEqual(ast.dump(left), ast.dump(right))

    def test_geometry_embedding_rigid_and_permutation_consistency(self):
        torch.manual_seed(74001)
        model = GeometryControlMatcher('geotransformer_rpe').double().eval()
        c = torch.randn(2, 2, 9, 3, dtype=torch.float64)
        rotation, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
        permutation = torch.randperm(9)
        with torch.no_grad():
            original = model.position(c)
            moved = model.position(c@rotation+.7)
            reordered = model.position(c[..., permutation, :])
        # Upstream quadratic distance arithmetic may leave tiny self distances.
        torch.testing.assert_close(moved, original, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(reordered, original[..., permutation, :, :][..., permutation, :], atol=2e-6, rtol=2e-6)


if __name__ == '__main__':
    unittest.main()
