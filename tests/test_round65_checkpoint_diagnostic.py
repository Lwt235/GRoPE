"""Policy identity and paired interaction fixtures, not a new task gate."""
import unittest
import numpy as np
from experiments.round65_checkpoint_diagnostic import final_state, compare_policies, contrast
from experiments.round65_common import ROLES, SEEDS, VARIANTS


class CheckpointDiagnosticTests(unittest.TestCase):
    def test_final_weights_are_distinct_from_saved_best_weights(self):
        final, best = {'weight':'final'}, {'weight':'best'}
        saved = dict(role='gaussian_relation', seed=11, epoch_completed=120, step=46800,
            lock_sha256='lock', data_sha256='data', stream_sha256='stream', state_dict=final, best_state=best)
        entry = dict(role='gaussian_relation', seed=11, sampling_sha256='stream')
        self.assertIs(final_state(saved, entry, 'lock', 'data'), final)
        for key, value in [('epoch_completed',119), ('step',46799), ('seed',22), ('data_sha256','other')]:
            with self.assertRaises(ValueError): final_state(dict(saved, **{key:value}), entry, 'lock', 'data')

    def test_policy_change_is_not_the_change_in_gaussian_gap(self):
        def stats(value): return dict(class_accuracy=[value]*60, ce=1.)
        baseline = {(r,s,v):stats(.6 if r == 'gaussian_relation' else .5) for r in ROLES for s in SEEDS for v in VARIANTS}
        final = {key:dict(value) for key,value in baseline.items()}
        for seed, change in zip(SEEDS, (.03,-.01,.02)):
            final['gaussian_relation',seed,'base'] = stats(.6+change)
            final['direct_mlp',seed,'base'] = stats(.52)
            final['distance_mlp',seed,'base'] = stats(.65)
        result = compare_policies(baseline, final)
        change = next(r for r in result['policy_changes'] if r['role']=='gaussian_relation' and r['variant']=='base')
        gap = next(r for r in result['final_gaussian_gaps'] if r['control']=='direct_mlp' and r['variant']=='base')
        interaction = next(r for r in result['gaussian_gap_changes'] if r['control']=='direct_mlp' and r['variant']=='base')
        np.testing.assert_allclose(change['paired_pp'], [3,-1,2])
        np.testing.assert_allclose(gap['paired_pp'], [11,7,10])
        np.testing.assert_allclose(interaction['paired_pp'], [1,-3,0], atol=1e-12)
        self.assertTrue(gap['fixed_criterion'])
        negative = next(r for r in result['final_gaussian_gaps'] if r['control']=='distance_mlp' and r['variant']=='base')
        self.assertFalse(negative['fixed_criterion'])
        del final['nope',33,'tilt']
        with self.assertRaises(ValueError): compare_policies(baseline, final)

    def test_incomplete_or_nonfinite_pairing_is_rejected(self):
        for x in (np.zeros((2,60)), np.full((3,60),np.nan)):
            with self.assertRaises(ValueError): contrast(x)


if __name__ == '__main__': unittest.main()
