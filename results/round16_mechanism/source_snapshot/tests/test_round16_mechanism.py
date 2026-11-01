"""Meaningful orchestration guards without opening any research data."""
import ast
from pathlib import Path

import numpy as np
import pytest
from scipy.special import softmax

from experiments import round16_mechanism as P


def test_temperature_identity_preserves_actual_float32_multiplication():
    rng=np.random.default_rng(733)
    geom=rng.normal(size=(9,17))*30
    content=rng.normal(size=geom.shape).astype(np.float32)
    p0=softmax(geom,axis=1)
    for t in (0.,5.,10.):
        expected=softmax(geom+(np.float32(t)*content).astype(np.float64),axis=1)
        np.testing.assert_allclose(P.reconstructed_probability(p0,content,t),expected,rtol=0,atol=1e-13)


def cohort_rows():
    entries=[dict(scene='fixture',pair=[1,2])]
    rows=[dict(scene='fixture',pair=[1,2],family='example',grid_index=i,query=q,config=c)
          for i,c in enumerate(P.A.GRID) for q in P.A.QUERIES]
    return entries,rows


def test_population_rejects_missing_duplicate_and_reselected_configuration():
    entries,rows=cohort_rows();P.population(rows,entries,('example',))
    for bad in (rows[:-1],rows+[rows[0]], [dict(rows[0],config={'h':9,'temperature':5}),*rows[1:]]):
        with pytest.raises(ValueError):P.population(bad,entries,('example',))


def test_relative_upstream_paths_resolve_against_repository_not_filesystem_root():
    assert P.absolute('results/example')==P.ROOT/'results/example'
    assert P.absolute('/tmp/example')==Path('/tmp/example')


def test_joint_lock_includes_independent_implementation_before_any_extraction():
    assert 'experiments/round16_mechanism_audit.py' in P.OWN_SOURCES
    assert 'experiments/round16_mechanism_audit_metrics.py' in P.OWN_SOURCES
    source=Path(P.__file__).read_text();tree=ast.parse(source)
    extract=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='extract')
    calls=[n.func.attr for n in ast.walk(extract) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
    assert 'pair_predictions' not in calls and 'condition_predictions' not in calls
