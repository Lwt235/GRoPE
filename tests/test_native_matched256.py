"""Input-count controls must preserve selected primitives and the parent stream."""
import unittest
import hashlib

import numpy as np
import torch

from experiments import native_matched256 as exp
from experiments.round65_common import read


class Matched256Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        exp.runner.configure()

    def test_training_matches_explicit_prefix_and_state_keys(self):
        args = exp.runner.synthetic_inputs(1, 512, 'cpu')
        factories = dict(relative5=exp.CompactNativeClassifier, distance_mlp=exp.NativeGSClassifier,
                         fixed_isotropic5=exp.NativeFixedIsotropic)
        for role in exp.ROLES:
            torch.manual_seed(11)
            model = exp.model_for(role).train()
            torch.manual_seed(11)
            plain = factories[role](role).train()
            self.assertEqual(list(model.state_dict()), list(plain.state_dict()))
            with torch.no_grad():
                torch.manual_seed(123)
                observed = model(*args)
                torch.manual_seed(123)
                expected = plain(*(x[:, :256] for x in args))
                torch.testing.assert_close(observed, expected, atol=0, rtol=0)
                model.eval()
                plain.eval()
                torch.testing.assert_close(model(*args), plain(*args), atol=0, rtol=0)

    def test_validation_256_and_explicit_conditions_keep_their_meaning(self):
        class Probe(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(1))
                self.counts = []

            def forward(self, c, s, a):
                self.counts.append((c.shape[1], self.training))
                return torch.zeros(len(c), 60) + self.anchor

        model = Probe()
        bundle = dict(centers=np.zeros((4, 2048, 3), dtype='float32'),
            covariance=np.tile(np.eye(3), (4, 2048, 1, 1)),
            appearance=np.zeros((4, 2048, 4), dtype='float32'), labels=np.arange(4), ids=np.arange(4))
        exp.evaluate(model, bundle)
        exp.evaluate(model, bundle, 'base')
        exp.evaluate(model, bundle, 'sparse256')
        exp.evaluate(model, bundle, 'dense1024')
        self.assertEqual(model.counts, [(256, False), (512, False), (256, False), (1024, False)])
        self.assertEqual(exp.selection_batch(bundle, 0, 4, 'base', 'cpu')[0].shape[1], 256)

    def test_full_epoch_stream_matches_existing_512_runs(self):
        for seed in exp.SEEDS:
            stream = hashlib.sha256(('R65-sampler-v1:' + str(seed)).encode()).hexdigest()
            rng = np.random.default_rng(seed + 65000)
            count, batches = 0, 0
            for index, subset, quaternion, jitter in exp.runner.epoch_batches(rng, 6227):
                stream = exp.runner.stream_update(stream, index, subset, quaternion, jitter)
                count += len(index)
                batches += 1
            self.assertEqual((count, batches), (6227, 390))
            for role, root in exp.PRIOR.items():
                history = read(root / 'runs' / role / str(seed) / 'history.json')
                self.assertEqual(stream, history['training'][0]['stream_sha256'])


if __name__ == '__main__':
    unittest.main()
