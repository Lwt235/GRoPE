import unittest
import numpy as np
import torch

from experiments.native_seed_replication import model_for, ROLES, SEEDS, OLD_SEEDS, VARIANTS
from experiments.native_seed_replication_report import cohort_summaries, replication_signal
from experiments.round65_run import shared_digest, synthetic_inputs
from models.round65_actual_gs import NativeGSClassifier
from models.round73_native_relative5 import CompactNativeClassifier


class NativeReplicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def test_existing_models_and_new_seed_pairing(self):
        for seed in SEEDS:
            common = []
            for role in ROLES:
                torch.manual_seed(seed)
                original = CompactNativeClassifier(role) if role == 'relative5' else NativeGSClassifier(role)
                torch.manual_seed(seed)
                actual = model_for(role)
                self.assertEqual(set(original.state_dict()), set(actual.state_dict()))
                for name, value in original.state_dict().items():
                    self.assertTrue(torch.equal(value, actual.state_dict()[name]), (seed, role, name))
                common.append(shared_digest(actual))
                original.eval(); actual.eval()
                with torch.no_grad():
                    inputs = synthetic_inputs(1, 8, 'cpu')
                    self.assertTrue(torch.equal(original(*inputs), actual(*inputs)))
            self.assertEqual(common[0], common[1], seed)

    @staticmethod
    def fixture(old_gain, new_gain):
        records = []
        for seed in OLD_SEEDS + SEEDS:
            gain = old_gain if seed in OLD_SEEDS else new_gain
            for role in ROLES:
                for variant in VARIANTS:
                    value = .5 + (gain if role == 'relative5' else 0)
                    records.append(dict(role=role, seed=seed, variant=variant,
                                        metrics=dict(class_accuracy=[value]*60)))
        return records

    def test_primary_decision_cannot_be_rescued_by_old_seed_gains(self):
        mixed = cohort_summaries(self.fixture(.2, -.01), replicates=100, seed=75001)
        self.assertGreater(mixed['all8'][0]['mean_gain_pp'], 0)
        self.assertFalse(replication_signal(mixed))
        reversed_groups = cohort_summaries(self.fixture(-.2, .01), replicates=100, seed=75001)
        self.assertLess(reversed_groups['all8'][0]['mean_gain_pp'], 0)
        self.assertTrue(replication_signal(reversed_groups))
        unchanged_new = cohort_summaries(self.fixture(.2, .01), replicates=100, seed=75001)
        self.assertEqual(reversed_groups['new5'], unchanged_new['new5'])


if __name__ == '__main__':
    unittest.main()
