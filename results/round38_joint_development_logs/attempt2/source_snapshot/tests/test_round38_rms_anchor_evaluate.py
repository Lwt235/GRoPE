import unittest
from fractions import Fraction
from pathlib import Path
import tempfile
from unittest.mock import patch
import numpy as np
from experiments import round38_rms_anchor_evaluate as E


def example(kl=.51,pck=Fraction(179,200)):
    old={r:dict(by_condition={'g24_g8':dict(kl=.5)}) for r in ('strong_quad','legacy_quad16')}
    control=dict(center=dict(by_condition={'g24_g8':dict(pck=E.frac(Fraction(9,10)))}))
    old.update({r:control for r in ('point_localizer','quartic_localizer','legacy_point_localizer')})
    same={r:control for r in ('anchor_point_localizer','anchor_quartic_localizer','anchor_point_legacy')}
    summary=dict(condition='g24_g8',population=dict(pairs=48,queries=2304,valid_centers=1096),
        region=dict(macro=dict(kl=kl)),center=dict(macro=dict(pck=E.frac(pck))))
    return summary,dict(old_by_role=old,by_role=same)


class MetricTests(unittest.TestCase):
    def test_all_eight_equal_boundary_checks_agree_without_promotion(self):
        s,b=example();primary=E.primary_checks(s,b);other=E.independent_checks(s,b)
        self.assertEqual(len(primary['checks']),8);self.assertTrue(primary['passed']);self.assertTrue(other['passed'])
        self.assertFalse(other['eligible_for_promotion']);self.assertFalse(other['independent_probability_reference'])
        self.assertEqual([c['passed'] for c in primary['checks']],[c['passed'] for c in other['checks']])

    def test_exact_fraction_below_bound_fails_six_center_checks(self):
        s,b=example(pck=Fraction(179,200)-Fraction(1,10**12))
        for check in (E.primary_checks(s,b),E.independent_checks(s,b)):
            self.assertFalse(check['passed']);self.assertEqual(sum(not c['passed'] for c in check['checks']),6)

    def test_kl_next_float_above_bound_fails_both_aliases(self):
        s,b=example(kl=float(np.nextafter(.51,np.inf)))
        for check in (E.primary_checks(s,b),E.independent_checks(s,b)):
            self.assertFalse(check['passed']);self.assertEqual(sum(not c['passed'] for c in check['checks']),2)

    def test_gate_failure_precedes_label_function(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)/'run';audit=Path(temp)/'audit'
            with patch.object(E,'require_prelabel',side_effect=ValueError('no complete seal')),patch.object(E,'generate_rows') as labels:
                with self.assertRaisesRegex(ValueError,'seal'):E.run(out,audit)
                labels.assert_not_called();self.assertFalse(out.exists())

    def test_missing_root_receipt_blocks_labels_after_arithmetic_success(self):
        out=Path('/tmp/r38_toy_producer');audit=Path('/tmp/r38_toy_audit')
        v=dict(all_input_and_output_arithmetic_passed=True,can_evaluate_train_triage=True,
            checked_pairs=48,checked_queries=2304,completed_queries=2304,independent_probability_reference=False,labels_read=False,
            producer_lock_sha256='h',audit_lock_sha256='h',producer_forward_complete_sha256='h',input_verification_sha256='h')
        def read(p):
            if p==audit/'completion.json':return dict(files={},all_input_and_output_arithmetic_passed=True,forward_verification_sha256='h')
            if p==audit/'forward_verification.json':return v
            if p==out/'forward_complete.json':return dict(forward_completion_sha256='h',all_production_completed=True,case_count=2304,completed_calls=2304)
            if p==out/'forward/completion.json':return dict(files={})
            raise FileNotFoundError('required explicit root receipt missing')
        with patch.object(E.C,'joint_locks',return_value=({},{})),patch.object(E.C,'read',side_effect=read),patch.object(E.C,'verify_inventory'),patch.object(E.C,'sha',return_value='h'),patch.object(np,'load',side_effect=AssertionError('labels accessed')):
            with self.assertRaisesRegex(FileNotFoundError,'root receipt'):E.require_prelabel(out,audit)

    def test_independent_gate_rejects_population_redefinition(self):
        s,b=example();s['population']['queries']=2303
        with self.assertRaises(ValueError):E.independent_checks(s,b)


if __name__=='__main__':unittest.main()
