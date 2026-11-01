"""Synthetic checks for R20 exact resource screens and full-population metrics."""
from fractions import Fraction
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from experiments import round20_anchor_fixed_metrics as M


class ScreenTests(unittest.TestCase):
    def fixture(self):
        def summary(kl,pck):
            return dict(macro=dict(common9_kl=kl,direct5_kl=kl,J=kl),
                by_condition={c['condition']:dict(kl=kl) for c in M.CONDITIONS},
                center=dict(by_condition={q+'_g8':dict(pck=M.fraction_dict(pck)) for q in ('g8','g16','g24')}))
        new={r:summary(1.,Fraction(9,10)) for r in ('anchor_trace','anchor_point_regional',
            'anchor_quartic_regional','anchor_fixed_regional','anchor_point_localizer',
            'anchor_quartic_localizer','anchor_point_legacy')}
        new['anchor_trace']=summary(.97,Fraction(9,10))
        old={r:summary(1.,Fraction(9,10)) for r in ('strong_point','legacy_point','strong_quad',
            'legacy_quad16','point_localizer','quartic_localizer','legacy_point_localizer')}
        return new,old

    def test_complete_sixty_checks_and_no_scientific_pass(self):
        a,b=self.fixture();s=M.screen(a,b)
        self.assertTrue(s['passed']);self.assertFalse(s['scientific_gate_passed']);self.assertEqual(len(s['checks']),60)
        self.assertEqual({c:sum(x['category']==c for x in s['checks']) for c in
            ('old_macro','old_condition','same_a_macro','old_center','same_a_center')},
            dict(old_macro=8,old_condition=28,same_a_macro=6,old_center=9,same_a_center=9))

    def test_fraction_boundary_has_no_epsilon(self):
        a,b=self.fixture();bound=Fraction(9,10)-Fraction(1,200)
        a['anchor_trace']['center']['by_condition']['g24_g8']['pck']=M.fraction_dict(bound)
        self.assertTrue(M.screen(a,b)['passed'])
        a['anchor_trace']['center']['by_condition']['g24_g8']['pck']=M.fraction_dict(bound-Fraction(1,10**18))
        s=M.screen(a,b);self.assertFalse(s['passed'])
        self.assertEqual(sum(not c['passed'] for c in s['checks']),6)

    def test_per_condition_cannot_be_hidden_by_macro(self):
        a,b=self.fixture();a['anchor_trace']['by_condition']['g24_g24']['kl']=1.02
        self.assertTrue(M.screen(a,b)['passed'])
        a['anchor_trace']['by_condition']['g24_g24']['kl']=np.nextafter(1.02,np.inf)
        self.assertFalse(M.screen(a,b)['passed'])

    def test_each_same_information_region_reference_required(self):
        for reference in ('anchor_point_regional','anchor_quartic_regional','anchor_fixed_regional'):
            a,b=self.fixture();a[reference]['macro']['direct5_kl']=.98
            self.assertFalse(M.screen(a,b)['passed'],reference)

    def test_stable_argmax_center_boundary_and_all_valid_rows(self):
        bounds=np.array([[0,0,1,1],[16,0,17,1]],np.int64)
        p=np.array([[.5,.5],[.1,.9],[1.,0.]],np.float64)
        target=np.array([[8.,0.],[24.,0.],[50.,0.]])
        valid=np.array([True,True,True])
        result=M.center_metrics(p.argmax(-1).astype(np.int32),bounds,target,valid)
        self.assertEqual(result['n_query'],3);self.assertEqual(result['n_point'],3)
        self.assertEqual(result['pck_count'],2)
        self.assertEqual(result['distance_sum'],66.)
        self.assertEqual(result['distance_squared_sum'],2628.)


class ChronologyTests(unittest.TestCase):
    def test_failed_forward_audit_cannot_open_any_array(self):
        from experiments import round20_anchor_fixed as P
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)
            with patch.object(P,'verify_lock',return_value={}), \
                 patch.object(P,'require_forward_audit',side_effect=ValueError('numeric failure')), \
                 patch.object(P.np,'load',side_effect=AssertionError('array accessed')) as load:
                with self.assertRaisesRegex(ValueError,'numeric failure'):P.evaluate(out)
                load.assert_not_called()
            completed=json.loads((out/'evaluation/completion.json').read_text())
            self.assertFalse(completed['passed'])
            self.assertTrue((out/'evaluation/failure.json').is_file())
            self.assertFalse((out/'summary.json').exists())

    def test_existing_failed_attempt_is_preserved(self):
        from experiments import round20_anchor_fixed as P
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp);(out/'evaluation').mkdir()
            marker=out/'evaluation/failure.json';marker.write_text('{"preserve":true}\n')
            with patch.object(P,'verify_lock',side_effect=AssertionError('old attempt accessed')):
                with self.assertRaisesRegex(ValueError,'preserve existing'):P.evaluate(out)
            self.assertEqual(marker.read_text(),'{"preserve":true}\n')


if __name__=='__main__':unittest.main()
