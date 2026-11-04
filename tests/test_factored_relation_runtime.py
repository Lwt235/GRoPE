"""Function and full parameter-gradient agreement for a nontrivial reordering."""
import copy
import unittest

import torch
import torch.nn.functional as F
from models.factored_relation_runtime import ProjectedRelation, factored_bias, factorize
from models.round58_support_pe import HEADS, HD
from models.round65_actual_gs import NativeGSClassifier
from models.round73_native_relative5 import CompactNativeClassifier
from models.round68_fixed_region import FixedRegionMatcher
from models.round69_compact_relation import CompactMatcher
from experiments.round65_run import synthetic_inputs


class FactoredRelationTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(4)
        torch.manual_seed(731105)

    def test_bias_and_input_weight_gradients(self):
        base = [torch.randn(2, 2, HEADS, 7, HD, dtype=torch.float64),
                torch.randn(2, 2, 7, 7, 96, dtype=torch.float64),
                torch.randn(HEADS * HD, 96, dtype=torch.float64) / 96**.5]
        cotangent = torch.randn(2, 2, HEADS, 7, 7, dtype=torch.float64)
        outputs, gradients = [], []
        for factored in (False, True):
            q, z, weight = [v.clone().requires_grad_() for v in base]
            result = factored_bias(q, ProjectedRelation(z, weight)) if factored else torch.einsum(
                'bvhid,bvijhd->bvhij', q, F.linear(z, weight).unflatten(-1, (HEADS, HD)))
            outputs.append(result)
            gradients.append(torch.autograd.grad(result, (q, z, weight), cotangent))
        torch.testing.assert_close(outputs[0], outputs[1], atol=1e-10, rtol=1e-10)
        for a, b in zip(*gradients):
            torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-10)

    def test_whole_models_and_all_parameter_gradients(self):
        for task in ('region', 'native'):
            for role in ('distance_mlp', 'direct_mlp', 'gaussian_relation', 'relative5'):
                with self.subTest(task=task, role=role):
                    torch.manual_seed(731105)
                    if task == 'native':
                        original = CompactNativeClassifier(role) if role == 'relative5' else NativeGSClassifier(role)
                        c, s, a = synthetic_inputs(2, 9, 'cpu')
                        data = (c.double(), s.double(), a.double())
                    else:
                        original = CompactMatcher(role) if role == 'relative5' else FixedRegionMatcher(role)
                        c, s, _ = synthetic_inputs(4, 9, 'cpu')
                        data = (torch.randn(2, 2, 9, 11, dtype=torch.float64), c.double().reshape(2, 2, 9, 3),
                                s.double().reshape(2, 2, 9, 3, 3), torch.arange(9).expand(2, 2, 9))
                    original = original.double().train()
                    alternative = factorize(copy.deepcopy(original), task)
                    assert set(original.state_dict()) == set(alternative.state_dict())
                    outputs, gradients = [], []
                    cotangent = None
                    for model in (original, alternative):
                        torch.manual_seed(731106)  # Identical native classifier dropout.
                        value = model(*data)
                        if cotangent is None:
                            generator = torch.Generator().manual_seed(731107)
                            cotangent = torch.randn(value.shape, generator=generator, dtype=value.dtype)
                        (value * cotangent).sum().backward()
                        outputs.append(value.detach())
                        gradients.append({name: parameter.grad for name, parameter in model.named_parameters()})
                    torch.testing.assert_close(outputs[0], outputs[1], atol=1e-9, rtol=1e-9)
                    self.assertEqual(set(gradients[0]), set(gradients[1]))
                    for name in gradients[0]:
                        a, b = gradients[0][name], gradients[1][name]
                        if a is None or b is None:
                            self.assertIs(a, b, name)
                        else:
                            self.assertTrue(bool(torch.isfinite(a).all() and torch.isfinite(b).all()), name)
                            torch.testing.assert_close(a, b, atol=1e-8, rtol=1e-8, msg=name)
                    for name, value in original.state_dict().items():
                        self.assertTrue(torch.equal(value, alternative.state_dict()[name]), name)


if __name__ == '__main__':
    unittest.main()
