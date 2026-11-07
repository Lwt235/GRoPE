"""Known score differences check the complete comparison and seed schedule."""
import unittest
import numpy as np

from experiments import mean_precision_report as report
from experiments.round65_common import SEEDS, VARIANTS


class MeanPrecisionReportTests(unittest.TestCase):
    def test_matched_known_differences_and_bootstrap_schedule(self):
        levels = dict(relative5=.47, mean_precision5=.44, metric_anisotropy5=.42,
                      shape_anisotropy5=.45, trace_isotropic5=.40,
                      distance_mlp=.35, fixed_isotropic5=.36)
        rows = []
        for case in ('region', 'native512'):
            for condition in (('base', 'sparse256') if case == 'region' else VARIANTS):
                for role in report.ROLES:
                    for seed in SEEDS:
                        level = levels[role] + seed / 10000
                        if case == 'region':
                            scenes = {str(j): dict(valid_queries=10, region_recall=level+j/1000,
                                center_pck=(level+j/1000)/2, hard_epe=1-level-j/1000,
                                soft_epe=1-level-j/1000, ce=1-level-j/1000) for j in range(5)}
                            group = dict(by_scene=scenes, macro={key: np.mean([r[key] for r in scenes.values()])
                                                               for key in report.METRICS})
                            metrics = dict(group, strata={str(j): group for j in range(4)})
                            rows.append(dict(case=case, condition=condition, role=role, seed=seed, metrics=metrics))
                        else:
                            accuracy = level + np.linspace(-.01, .01, 60)
                            metrics = dict(mca=float(accuracy.mean()), oa=level, ce=1-level,
                                           class_accuracy=accuracy.tolist())
                            rows.append(dict(case=case, condition=condition, role=role, seed=seed, metrics=metrics,
                                domains={name: dict(mca=level) for name in ('MACGS', 'ModelNet')}))
        lookup, comparisons = report.comparisons_for(rows)
        self.assertEqual(len(lookup), 168)
        self.assertEqual(len(comparisons), 112)
        self.assertEqual([r['bootstrap_seed'] for r in comparisons], list(range(83001, 83113)))
        expected = (3., 4., 9., 8., 2., -1., 12.)
        for row in comparisons:
            value = expected[report.PAIRS.index((row['left'], row['right']))]
            self.assertAlmostEqual(row['mean_gain_pp'], value, places=10)
            np.testing.assert_allclose(row['paired_seed_gains_pp'], [value]*3, atol=1e-10)
            key = 'crossed_seed_scene_bootstrap_95_pp' if row['case'] == 'region' else 'crossed_seed_class_bootstrap_95_pp'
            np.testing.assert_allclose(row[key], [value, value], atol=1e-10)
            if row['case'] == 'region':
                self.assertAlmostEqual(row['mean']['center_pck'], value / 2, places=10)
            else:
                np.testing.assert_allclose(row['paired_ce_difference'], [-value/100]*3, atol=1e-10)


if __name__ == '__main__':
    unittest.main()
