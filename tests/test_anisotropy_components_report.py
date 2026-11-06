"""Known additive and interacting four-cell outcomes guard contrast signs."""
import unittest
import numpy as np

from experiments import anisotropy_components_report as report
from experiments.round65_common import SEEDS, VARIANTS


class FactorialReportTests(unittest.TestCase):
    @staticmethod
    def records(full_increment=0.):
        levels = dict(relative5=.47+full_increment, metric_anisotropy5=.42,
                      shape_anisotropy5=.45, trace_isotropic5=.40,
                      distance_mlp=.35, fixed_isotropic5=.36)
        rows = []
        for case in ('region', 'native512'):
            for condition in (('base', 'sparse256') if case == 'region' else VARIANTS):
                for role in report.ROLES:
                    for seed in SEEDS:
                        level = levels[role] + seed / 10000
                        if case == 'region':
                            scenes = {}
                            for j in range(5):
                                score = level + j / 1000
                                scenes[str(j)] = dict(valid_queries=10, region_recall=score,
                                    center_pck=score/2, hard_epe=1-score, soft_epe=1-score, ce=1-score)
                            group = dict(by_scene=scenes, macro={key: np.mean([s[key] for s in scenes.values()])
                                                               for key in report.METRICS})
                            metrics = dict(group, strata={str(j): group for j in range(4)})
                            rows.append(dict(case=case, condition=condition, role=role, seed=seed, metrics=metrics))
                        else:
                            accuracy = level + np.linspace(-.01, .01, 60)
                            metrics = dict(mca=float(accuracy.mean()), oa=level, ce=1-level,
                                           class_accuracy=accuracy.tolist())
                            rows.append(dict(case=case, condition=condition, role=role, seed=seed, metrics=metrics,
                                domains={name: dict(mca=level) for name in ('MACGS', 'ModelNet')}))
        return rows

    def test_known_additivity_and_positive_interaction(self):
        lookup, comparisons = report.comparisons_for(self.records())
        self.assertEqual(len(lookup), 144)
        self.assertEqual(len(comparisons), 192)
        baseline = report.factorial_interactions(lookup, len(comparisons))
        for row in baseline:
            self.assertAlmostEqual(row['mean_gain_pp'], 0., places=10)
        changed = {(r['case'], r['role'], r['seed'], r['condition']): r for r in self.records(.03)}
        interactions = report.factorial_interactions(changed, len(comparisons))
        self.assertEqual(len(interactions), 16)
        for row in interactions:
            np.testing.assert_allclose(row['paired_seed_gains_pp'], [3., 3., 3.], atol=1e-10)
            key = 'crossed_seed_scene_bootstrap_95_pp' if row['case']=='region' else 'crossed_seed_class_bootstrap_95_pp'
            np.testing.assert_allclose(row[key], [3., 3.], atol=1e-10)
        seeds = [r['bootstrap_seed'] for r in comparisons + interactions]
        self.assertEqual(seeds, list(range(82001, 82001+208)))


if __name__ == '__main__':
    unittest.main()
