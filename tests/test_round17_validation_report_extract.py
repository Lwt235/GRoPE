"""Small reporting-only tests; standard library, synthetic dictionaries only."""
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from experiments import round17_validation_report_extract as R


def accepted_fixture(root, wrong_field=None, wrong_design=False):
    """Rehash the entire synthetic chain, exposing semantic cross-run mismatch."""
    run, audit = root/'run', root/'audit'
    run.mkdir(); audit.mkdir()
    roles, gates = {'synthetic':'roles'}, {'passed':False}
    R.write(run/'lock.json', {'synthetic':'producer'})
    (run/'lock.sha256').write_text(R.digest(run/'lock.json')+'\n')
    R.write(run/'forward_complete.json', {'synthetic':'seal'})
    R.write(run/'summary.json', {'roles':roles,'gates':gates})
    R.write(run/'completion.json', {'summary_sha256':R.digest(run/'summary.json'),
                                  'files':{'summary.json':R.digest(run/'summary.json')}})
    bindings = {'validation_'+name+'_sha256':R.digest(run/(name+'.json'))
                for name in ('lock','forward_complete','summary','completion')}
    R.write(audit/'lock.json', {'producer_lock_sha256':'0'*64 if wrong_design else bindings['validation_lock_sha256']})
    (audit/'lock.sha256').write_text(R.digest(audit/'lock.json')+'\n')
    verification = dict(bindings, passed=True, roles=roles, gates=gates, scientific_gate_passed=False,
                        independent_audit_lock_sha256=R.digest(audit/'lock.json'))
    if wrong_field is not None:
        verification['validation_'+wrong_field+'_sha256'] = '0'*64
    R.write(audit/'verification.json', verification)
    for scene in ('scene_a','scene_b'):
        R.write(audit/(scene+'_numeric.json'), {'checks':[]})
    R.write(audit/'completion.json', {'passed':True,'files':{p.name:R.digest(p) for p in audit.iterdir()}})
    R.write(run/'decision.json', dict(bindings, audit_path=str(audit/'verification.json'),
        audit_sha256=R.digest(audit/'verification.json'), independent_audit_passed=True,
        roles=roles,gates=gates,scientific_gate_passed=False))
    (run/'decision.sha256').write_text(R.digest(run/'decision.json')+'\n')
    return run


class ReportTests(unittest.TestCase):
    def test_incomplete_run_opens_no_result_metadata(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(R, 'read', side_effect=AssertionError('premature result read')):
                with self.assertRaisesRegex(ValueError, 'completed independent acceptance'):
                    R.accepted_metadata(root)

    def test_false_alternative_route_is_not_mandatory_failure(self):
        gates = dict(passed=False, original=dict(passed=True,
            accuracy_cost=dict(passed=True, accuracy_route=False, efficiency_route=True)),
            scale=dict(passed=False, comparisons={'localizer':dict(passed=False, difference_pp=-1.)}),
            predictions=dict(passed=True, checks={'all_finite':True}),
            general_research_goal_achieved=False)
        result = R.gate_records(gates)
        self.assertEqual({r['path'] for r in result['failed_decision_nodes']},
                         {'gates','gates.scale','gates.scale.comparisons.localizer'})
        self.assertEqual(result['failed_boolean_checks'], [])
        self.assertEqual(result['alternative_cost_routes'][0]['passed'], False)
        self.assertEqual(result['original_complete_gate_tree'], gates)

    def test_every_alias_scene_and_control_is_retained(self):
        aliases = {'primary':'o0', **{f'control{i}':'o1' for i in range(16)},
                   'matched_constant_temperature':'o1','matched_isotropic':'o2'}
        operators = {f'o{i}':dict(family=f'family{i}',h=.25,temperature=10.) for i in range(3)}
        def metric(numerator):
            counts = {'scene_a':dict(pck_count=numerator,n_point=3),
                      'scene_b':dict(pck_count=1,n_point=2)}
            return dict(kl=1.,js=.2,direct_kl=1.1,direct_js=.3,
                detail={s:{'g24_g8':dict(kl=1.,js=.2,direct_kl=1.1,direct_js=.3,mass=10)} for s in counts},
                by_scene={s:1. for s in counts},by_condition={'g24_g8':1.},
                by_scene_condition={s:{'g24_g8':1.} for s in counts},
                pck_by_condition={'g24_g8':dict(counts_by_scene=counts)},
                probability_checks=dict(all_finite=True,max_normalization_error=0.))
        summary = dict(roles=dict(aliases=aliases,operators=operators),
            metrics={'scale':{alias:metric(2 if alias=='primary' else 1) for alias in aliases}},
            timing={op:dict(mean_seconds=1.,cells=[{}]*45) for op in operators})
        table = R.tables(summary)
        self.assertEqual(len(table['all_alias_scene_conditions']),38)
        self.assertEqual(len(table['all_alias_scenario_macros']),19)
        self.assertEqual(len(table['all_alias_cost']),19)
        self.assertEqual(len(table['primary_against_every_control']),54)
        macro = [r for r in table['primary_against_every_control'] if r['scene']=='equal_scene_macro']
        self.assertTrue(all(Fraction(r['exact_pck_delta_numerator'],r['exact_pck_delta_denominator'])==Fraction(1,6) for r in macro))

    def test_output_refuses_existing_or_frozen_run_subdirectory(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for out in (root, root/'reports'):
                with patch.object(R,'accepted_metadata',side_effect=AssertionError('should stop first')):
                    with self.assertRaisesRegex(ValueError,'outside the frozen run'):
                        R.extract(root,out)

    def test_same_run_chain_can_report_an_accepted_scientific_failure(self):
        with TemporaryDirectory() as directory:
            run = accepted_fixture(Path(directory))
            _, decision, verification, numeric, _ = R.accepted_metadata(run)
            self.assertFalse(decision['scientific_gate_passed'])
            self.assertTrue(verification['passed'])
            self.assertEqual(len(numeric), 2)

    def test_cross_run_verification_and_design_rejected_despite_valid_hashes(self):
        for field in ('lock','forward_complete','summary','completion','audit_design'):
            with self.subTest(field=field), TemporaryDirectory() as directory:
                run = accepted_fixture(Path(directory),
                    wrong_field=None if field=='audit_design' else field,
                    wrong_design=field=='audit_design')
                with self.assertRaisesRegex(ValueError, 'different producer run'):
                    R.accepted_metadata(run)


if __name__ == '__main__':
    unittest.main()
