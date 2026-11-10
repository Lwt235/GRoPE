import numpy as np
import pytest

from experiments import native_source_diversity_analysis as analysis
from experiments.native_source_diversity_report_recovery import compare_repaired
from experiments.native_source_diversity_review import independent_pair


def test_duplicate_seed_failure_and_repaired_full_comparison_contract():
    labels = np.concatenate([np.repeat(np.arange(60), 3), np.repeat(analysis.CLASSES, 5)])
    old = np.arange(len(labels)) < 180
    sources = np.where(old & (labels < 30), 'MACGS', 'ModelNet')
    rng = np.random.default_rng(71)
    predictions = {(role, seed): np.where(rng.random(len(labels)) < .7, labels, 59)
                   for role in analysis.ROLES for seed in analysis.SEEDS}
    specs = analysis.comparison_specs()
    with pytest.raises(TypeError, match='bootstrap_seed'):
        analysis.compare(predictions, labels, sources, old, specs[0], replicates=129)
    for spec in specs:
        row = compare_repaired(predictions, labels, sources, old, spec, replicates=129)
        for key, value in spec.items():
            assert row[key] == value
        ix = analysis.group_masks(labels, sources, old)[spec['group']]
        a = np.stack([predictions['relative5', seed][ix] for seed in analysis.SEEDS])
        b = np.stack([predictions[spec['right'], seed][ix] for seed in analysis.SEEDS])
        domain = sources[ix] if spec['group'] == 'primary_seven' else np.full(ix.sum(), 'pooled')
        independent = independent_pair(a, b, labels[ix], domain, seed=spec['bootstrap_seed'], replicates=129)
        assert row['count'] == ix.sum()
        assert row['bootstrap_replicates'] == 129
        assert row['mean_gain_pp'] == pytest.approx(independent['mean'], abs=1e-10)
        np.testing.assert_allclose(row['paired_seed_gains_pp'], independent['seeds'], atol=1e-10)
        np.testing.assert_allclose(row['class_source_gains_pp'], independent['cells'], atol=1e-10)
        for key, value in independent['intervals'].items():
            np.testing.assert_allclose(row['intervals_95_pp'][key], value, atol=1e-10)
