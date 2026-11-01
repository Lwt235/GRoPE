"""Synthetic metadata only; never execute the R20 report extractor on real files."""
import copy
from fractions import Fraction
import json
from pathlib import Path
import tempfile
import unittest

from experiments import round20_anchor_report_extract as R


def fraction(value):
    return dict(numerator=value.numerator,denominator=value.denominator,value=float(value))


def fixture():
    f=fraction(Fraction(1,2));center=dict(pck_count=1,n_point=2,n_query=2,pck=f,
        distance_sum=2.,distance_squared_sum=2.,mean=1.,rmse=1.)
    region=dict(kl=1.,js=.2,direct_kl=1.1,direct_js=.3,mass=100)
    macro=dict(common9_kl=1.,direct5_kl=1.,J=1.)
    value=dict(by_scene_condition={s:{c:copy.deepcopy(region) for c in R.CONDITIONS} for s in R.SCENES},
        by_condition={c:{k:region[k] for k in ('kl','js','direct_kl','direct_js')} for c in R.CONDITIONS},
        by_scene={s:dict(macro) for s in R.SCENES},macro=dict(macro),
        center=dict(by_scene_condition={s:{c:copy.deepcopy(center) for c in R.CENTER_CONDITIONS} for s in R.SCENES},
                    by_condition={c:dict(pck=dict(f),mean=1.,rmse=1.) for c in R.CENTER_CONDITIONS}))
    checks=[]
    for category,count in dict(old_macro=8,old_condition=28,same_a_macro=6,old_center=9,same_a_center=9).items():
        for i in range(count):
            checks.append(dict(category=category,name=category+str(i),metric='pck' if 'center' in category else 'kl',
                reference='synthetic',candidate_value=dict(f) if 'center' in category else 1.,
                reference_value=dict(f) if 'center' in category else 1.,bound=dict(f) if 'center' in category else 1.,passed=True))
    checks[0]['passed']=False
    return dict(by_role={r:copy.deepcopy(value) for r in R.FAMILIES},old_by_role={},
                screen=dict(checks=checks,passed=False)),dict(role_configs={r:dict(h=.25,temperature=10.) for r in R.FAMILIES},old_controls={})


class ReportTests(unittest.TestCase):
    def test_all_conditions_and_false_checks_are_retained(self):
        summary,lock=fixture();tables=R.summary_tables(summary,lock)
        self.assertEqual(len(tables['regional_scene_conditions']),12*14*3)
        self.assertEqual(len(tables['center_scene_conditions']),12*11*3)
        self.assertEqual(len(tables['screen_all_checks']),60)
        self.assertEqual(len(tables['screen_failed_checks']),1)
        self.assertIs(tables['screen_failed_checks'][0]['passed'],False)
        self.assertIs(json.loads(tables['screen_failed_checks'][0]['exact_check_json'])['passed'],False)

    def test_fraction_is_not_replaced_with_rounded_decimal(self):
        value=fraction(Fraction(179,200)-Fraction(1,10**18))
        result=R.fraction_columns('pck',value)
        self.assertEqual(Fraction(result['pck_numerator'],result['pck_denominator']),Fraction(179,200)-Fraction(1,10**18))
        value['numerator']*=2;value['denominator']*=2
        with self.assertRaisesRegex(ValueError,'noncanonical'):R.exact_fraction(value)

    def test_missing_center_condition_is_rejected(self):
        summary,lock=fixture();del summary['by_role']['anchor_trace']['center']['by_condition']['g24_g8']
        with self.assertRaisesRegex(ValueError,'center condition omitted'):R.summary_tables(summary,lock)

    def test_receipt_flags_precede_artifact_reads_and_hashes_must_match(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);receipt=root/'receipt.json';artifact=root/'artifact.json'
            flags={name:True for name in R.REQUIRED_ACCEPTANCE}
            receipt.write_text(json.dumps(dict(flags,accepted_for_development_interpretation=False,files={'absent': '0'*64})))
            with self.assertRaisesRegex(ValueError,'acceptance required'):R.verify_receipt(receipt,root)
            artifact.write_text('{"synthetic":true}\n')
            receipt.write_text(json.dumps(dict(flags,files={'artifact.json':R.digest(artifact)})))
            _,bound=R.verify_receipt(receipt,root);self.assertEqual(len(bound),1)
            artifact.write_text('{"synthetic":false}\n')
            with self.assertRaisesRegex(ValueError,'artifact changed'):R.verify_receipt(receipt,root)


if __name__=='__main__':unittest.main()
