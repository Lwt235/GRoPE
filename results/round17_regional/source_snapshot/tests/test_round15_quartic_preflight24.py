import pytest
from experiments import round15_quartic_preflight24 as Q


def test_repair_requires24_and_rejects_old_or_other_order():
    Q.require_order24({'quadrature_order':24},'quadrature_order')
    for order in (16,32,48):
        with pytest.raises(ValueError,match='fixed GL24'):
            Q.require_order24({'quadrature_order':order},'quadrature_order')


def test_source_override_is_scoped_and_binds_new_policy(tmp_path,monkeypatch):
    prior=Q.P.SOURCES;seen=[]
    report=tmp_path/'numeric';report.mkdir()
    Q.P.write(report/'verification.json',{'quadrature_order':24})
    def observe(out,path):
        assert path==report
        seen.append(Q.P.SOURCES)
    monkeypatch.setattr(Q.P,'freeze',observe)
    Q.freeze(tmp_path/'cost',report)
    assert seen==[prior+Q.EXTRA_SOURCES]
    assert Q.P.SOURCES==prior


def test_run_rejects_old_lock_or_missing_new_source_before_cost(tmp_path,monkeypatch):
    called=[]
    monkeypatch.setattr(Q.P,'run',lambda *args:called.append(args))
    monkeypatch.setattr(Q.P,'verify',lambda out:dict(order=16,source_hashes=dict.fromkeys(Q.EXTRA_SOURCES)))
    with pytest.raises(ValueError,match='fixed GL24'):Q.run(tmp_path,'cpu')
    monkeypatch.setattr(Q.P,'verify',lambda out:dict(order=24,source_hashes={}))
    with pytest.raises(ValueError,match='provenance'):Q.run(tmp_path,'cpu')
    assert not called
