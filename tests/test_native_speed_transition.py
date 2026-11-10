"""Runtime adoption policy and state-preserving model adaptation."""
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

from experiments.native_speed_followup import adapted, paired_summary, choose_implementation
from experiments.round65_common import ROOT, read
from experiments.round65_run import synthetic_inputs
from experiments.native_speed_transition import live
from models.native_shape_attribution import NativeFixedIsotropic


class RuntimeAdoptionTests(unittest.TestCase):
    def test_observer_accepts_stop_but_rejects_pid_reuse(self):
        record = dict(pid=123, start_ticks=987)
        with patch('experiments.native_speed_transition.identity', return_value=dict(record, state='T')):
            self.assertTrue(live(record))
        with patch('experiments.native_speed_transition.identity', return_value=dict(record, state='Z')):
            self.assertFalse(live(record))
        with patch('experiments.native_speed_transition.identity', return_value=dict(pid=123, start_ticks=988, state='R')):
            with self.assertRaises(RuntimeError): live(record)

    def test_existing_measurement_arithmetic(self):
        folder = ROOT/'results/efficiency_sidecar/fullbatch1/shared'
        measurements = read(folder/'timing.json')['measurements']
        for saved in read(folder/'summary.json')['results']:
            rows = [r for r in measurements if r['role'] == saved['role']]
            result = paired_summary(rows, 'baseline', 'candidate')
            self.assertEqual(result['paired_gain_percent'], saved['paired_time_reduction_percent'])
            self.assertEqual(result['median_gain_percent'], saved['median_paired_time_reduction_percent'])
            self.assertEqual(result['positive_blocks'], saved['positive_blocks'])

    def test_prefers_simple_runtime_and_rejects_unstable_candidate(self):
        c = {k: {'speed_supported': True} for k in ('original__geometry_factored',
            'original__geometry_fused', 'geometry_factored__geometry_fused')}
        ok = dict(geometry_factored=True, geometry_fused=True)
        self.assertEqual(choose_implementation(c, ok, ok), 'geometry_fused')
        c['geometry_factored__geometry_fused']['speed_supported'] = False
        self.assertEqual(choose_implementation(c, ok, ok), 'geometry_factored')
        bad = dict(geometry_factored=False, geometry_fused=False)
        self.assertEqual(choose_implementation(c, bad, ok), 'original')
        self.assertEqual(choose_implementation(c, ok, bad), 'original')

    def test_duplicate_drift_is_not_a_speedup(self):
        rows = []
        for block in range(8):
            for label, ms in [('baseline_a', 100), ('baseline_b', 80),
                              ('candidate_a', 94), ('candidate_b', 75)]:
                rows.append(dict(block=block, label=label, wall_ms=ms,
                                 final_loss=1., peak_allocated_bytes=10*2**30))
        result = paired_summary(rows, 'baseline', 'candidate')
        self.assertGreater(result['median_gain_percent'], 3)
        self.assertFalse(result['speed_supported'])

    def test_sphere_adaptation_preserves_rng_parameters_and_calculation(self):
        torch.set_num_threads(1)
        torch.manual_seed(47)
        base = NativeFixedIsotropic()
        data = synthetic_inputs(2, 17, 'cpu')
        target = torch.tensor([1, 13])
        rng = torch.get_rng_state().clone()
        results = []
        for implementation in ('original', 'geometry_factored', 'geometry_fused'):
            model = adapted(base, implementation)
            self.assertTrue(torch.equal(torch.get_rng_state(), rng))
            self.assertEqual(model.state_dict().keys(), base.state_dict().keys())
            for key, value in model.state_dict().items(): self.assertTrue(torch.equal(value, base.state_dict()[key]))
            torch.manual_seed(832)
            output = model(*data)
            F.cross_entropy(output, target).backward()
            grad = torch.cat([p.grad.flatten() for p in model.parameters()])
            results.append((output.detach(), grad.detach()))
            torch.set_rng_state(rng)
        for output, grad in results[1:]:
            torch.testing.assert_close(output, results[0][0], atol=2e-5, rtol=2e-5)
            torch.testing.assert_close(grad, results[0][1], atol=3e-6, rtol=3e-4)
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__':
    unittest.main()
