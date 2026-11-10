import numpy as np

from experiments.region_pose_analysis import paired_statistics


def test_fixed_scene_bootstrap_preserves_pairing_and_constant_effect():
    baseline = np.array([[.1, .2, .3, .4, .5], [.3, .4, .2, .1, .8], [.7, .8, .4, .3, .6]])
    result = paired_statistics(baseline + .03, baseline, seed=93001, replicates=257)
    np.testing.assert_allclose(result['paired_seed_gains_pp'], [3, 3, 3], atol=1e-12)
    for interval in result['intervals_95_pp'].values():
        np.testing.assert_allclose(interval, [3, 3], atol=1e-12)
    zero = paired_statistics(baseline, baseline, seed=93002, replicates=257)
    assert zero['mean_gain_pp'] == 0
    assert all(interval == [0, 0] for interval in zero['intervals_95_pp'].values())
