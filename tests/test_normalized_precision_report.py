import unittest
import numpy as np

from experiments import normalized_precision_report as report
from experiments.round65_common import SEEDS, VARIANTS


class NormalizedPrecisionReportTests(unittest.TestCase):
    def test_known_signed_effects_and_statistics_schedule(self):
        levels = dict(relative5=.47, normalized_precision5=.44,
                      mean_precision5=.42, trace_isotropic5=.40, distance_mlp=.35)
        rows = []
        for case in ('region','native512'):
            for condition in (('base','sparse256') if case == 'region' else VARIANTS):
                for role in report.ROLES:
                    for seed in SEEDS:
                        level = levels[role]+seed/10000
                        if case == 'region':
                            scenes = {str(j):dict(valid_queries=10,region_recall=level+j/1000,
                                center_pck=(level+j/1000)/2,hard_epe=1-level-j/1000,
                                soft_epe=1-level-j/1000,ce=1-level-j/1000) for j in range(5)}
                            group = dict(by_scene=scenes,macro={k:np.mean([r[k] for r in scenes.values()]) for k in report.METRICS})
                            rows.append(dict(case=case,condition=condition,role=role,seed=seed,
                                             metrics=dict(group,strata={str(j):group for j in range(4)})))
                        else:
                            accuracy = level+np.linspace(-.01,.01,60)
                            rows.append(dict(case=case,condition=condition,role=role,seed=seed,
                                metrics=dict(mca=float(accuracy.mean()),oa=level,ce=1-level,class_accuracy=accuracy.tolist()),
                                domains={name:dict(mca=level) for name in ('MACGS','ModelNet')}))
        lookup,comparisons = report.comparisons_for(rows)
        self.assertEqual(len(lookup),120)
        self.assertEqual(len(comparisons),80)
        self.assertEqual([r['bootstrap_seed'] for r in comparisons],list(range(87001,87081)))
        for row in comparisons:
            value = 100*(levels[row['left']]-levels[row['right']])
            self.assertAlmostEqual(row['mean_gain_pp'],value,places=10)
            np.testing.assert_allclose(row['paired_seed_gains_pp'],[value]*3,atol=1e-10)
            key = 'crossed_seed_scene_bootstrap_95_pp' if row['case']=='region' else 'crossed_seed_class_bootstrap_95_pp'
            np.testing.assert_allclose(row[key],[value,value],atol=1e-10)
            if row['case']=='region':
                self.assertAlmostEqual(row['mean']['center_pck'],value/2,places=10)
            else:
                np.testing.assert_allclose(row['paired_ce_difference'],[-value/100]*3,atol=1e-10)


if __name__ == '__main__':
    unittest.main()
