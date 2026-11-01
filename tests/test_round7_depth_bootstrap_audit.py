import itertools

import numpy as np
import pytest

from experiments.round7_depth_bootstrap_audit import independent_bootstrap


def test_explicit_crossed_cell_accumulation_matches_enumeration():
    a=np.array([[1.,1.],[1.,1.04]]);b=np.ones((2,2))
    choices=list(itertools.product(range(2),repeat=2))
    seeds=np.array([s for s in choices for p in choices])
    scenes=np.array([p for s in choices for p in choices])
    expected=[np.mean([a[i,j]-b[i,j] for i in s for j in p]) for s,p in zip(seeds,scenes)]
    result=independent_bootstrap(a,b,seeds,scenes)
    assert result['difference']==pytest.approx(.01)
    np.testing.assert_allclose(result['ci95_difference'],np.percentile(expected,[2.5,97.5]),atol=1e-15)


def test_relative_interval_is_ratio_of_macro_errors_and_pairing_is_shared():
    a=np.array([[1.,2.],[4.,8.]]);b=a*2
    sd=np.array([[0,0],[0,1],[1,0],[1,1]])
    pd=sd[::-1]
    result=independent_bootstrap(a,b,sd,pd)
    np.testing.assert_allclose(result['ci95_relative_difference'],[-.5,-.5])
    assert result['improved_seed_count']==2
    assert not independent_bootstrap(np.array([[np.nan]]),np.array([[np.nan]]),np.array([[0]]),np.array([[0]]))['available']
