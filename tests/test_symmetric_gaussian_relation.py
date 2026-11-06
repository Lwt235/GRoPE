import unittest
import numpy as np
import torch
from torch import nn

from models.round69_compact_relation import CompactMatcher, CompactPosition, descriptors
from models.round73_native_relative5 import CompactNativeClassifier
from models.symmetric_gaussian_relation import (
    SymmetricRegionMatcher, SymmetricNativeClassifier, symmetric_inputs)


class SymmetricRelationTests(unittest.TestCase):
    @staticmethod
    def geometry():
        rng = np.random.default_rng(79201)
        centers = rng.normal(size=(2, 9, 3)) * .2
        matrices = rng.normal(size=(2, 9, 3, 3)) * .12
        covariance = matrices @ matrices.swapaxes(-1, -2) + .002 * np.eye(3)
        return torch.from_numpy(centers), torch.from_numpy(covariance)

    def test_relation_and_rigid_frame(self):
        centers, covariance = self.geometry()
        value = symmetric_inputs(centers, covariance)
        original = descriptors('relative5', centers, covariance)
        expected = .5 * (original + original[..., [0, 2, 1, 4, 3]])
        torch.testing.assert_close(value, expected, atol=0, rtol=0)
        torch.testing.assert_close(value, value.transpose(-3, -2), atol=1e-11, rtol=1e-11)
        rotation = torch.linalg.qr(torch.tensor([[1., 2., 3.], [2., -1., 4.], [3., 1., -1.]], dtype=torch.float64)).Q
        moved = centers @ rotation.T + torch.tensor([.7, -.3, .2], dtype=torch.float64)
        rotated = rotation @ covariance @ rotation.T
        torch.testing.assert_close(value, symmetric_inputs(moved, rotated), atol=1e-10, rtol=1e-10)
        # Actual first-layer inputs have exactly three independent coordinates.
        torch.testing.assert_close(value[..., 1], value[..., 2], atol=0, rtol=0)
        torch.testing.assert_close(value[..., 3], value[..., 4], atol=0, rtol=0)

    def test_initialization_and_rng_match_existing_candidate(self):
        cases = [(CompactMatcher, SymmetricRegionMatcher, 'symmetric3'),
                 (CompactNativeClassifier, SymmetricNativeClassifier, 'symmetric3'),
                 (CompactNativeClassifier, SymmetricNativeClassifier, 'jeffreys_mlp')]
        for baseline, candidate, role in cases:
            with self.subTest(role=role, family=candidate.__name__):
                torch.manual_seed(11)
                reference = baseline('relative5')
                reference_rng = torch.get_rng_state()
                torch.manual_seed(11)
                model = candidate(role)
                self.assertTrue(torch.equal(reference_rng, torch.get_rng_state()))
                self.assertEqual(reference.state_dict().keys(), model.state_dict().keys())
                for name, tensor in reference.state_dict().items():
                    self.assertTrue(torch.equal(tensor, model.state_dict()[name]), name)

    def test_three_column_fold_and_native_gradient(self):
        centers, covariance = self.geometry()
        torch.manual_seed(79202)
        mlp = CompactPosition('relative5').mlp.double()
        first = nn.Linear(3, 96).double()
        with torch.no_grad():
            first.weight.copy_(torch.stack((mlp[0].weight[:, 0],
                mlp[0].weight[:, 1] + mlp[0].weight[:, 2],
                mlp[0].weight[:, 3] + mlp[0].weight[:, 4]), -1))
            first.bias.copy_(mlp[0].bias)
        x = symmetric_inputs(centers, covariance)
        folded = mlp[2](mlp[1](first(x[..., [0, 1, 3]])))
        torch.testing.assert_close(mlp(x), folded, atol=1e-12, rtol=1e-12)
        for role in ('symmetric3', 'jeffreys_mlp'):
            with self.subTest(role=role):
                model = SymmetricNativeClassifier(role).train()
                appearance = torch.rand(2, 9, 4)
                output = model(centers.float(), covariance, appearance)
                self.assertEqual(tuple(output.shape), (2, 60))
                self.assertTrue(torch.isfinite(output).all())
                output.square().mean().backward()
                for name, parameter in model.position.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                self.assertGreater(float(model.position.mlp[0].weight.grad.norm()), 0)


if __name__ == '__main__':
    torch.set_num_threads(4)
    unittest.main()
