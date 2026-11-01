import json
from pathlib import Path
import pytest
from experiments import round15_quartic_tiled_numerical_audit as T


def parent_fixture(root):
    directory=root/'frozen';directory.mkdir();source=root/'base.py';source.write_text('fixed source\n')
    (directory/source.name).write_bytes(source.read_bytes())
    report=root/'report';report.mkdir()
    T.A.write(report/'verification.json',dict(passed=True));T.A.write(report/'completion.json',dict(passed=True))
    T.A.write(directory/'lock.json',dict(source_hashes={str(source):T.A.sha(source)},report=str(report),
        report_sha256=T.A.sha(report/'verification.json'),completion_sha256=T.A.sha(report/'completion.json')))
    digest=T.A.sha(directory/'lock.json');(directory/'lock.sha256').write_text(digest+'\n')
    return directory,digest,source


def test_parent_chain_checks_canonical_and_snapshot_sources(tmp_path):
    directory,digest,source=parent_fixture(tmp_path);T.verify_parent(directory,digest)
    source.write_text('changed source\n')
    with pytest.raises(ValueError,match='source changed'):T.verify_parent(directory,digest)


def test_parent_report_change_is_not_silently_rebound(tmp_path):
    directory,digest,source=parent_fixture(tmp_path)
    (tmp_path/'report'/'verification.json').write_text(json.dumps(dict(passed=False)))
    with pytest.raises(ValueError,match='report changed'):T.verify_parent(directory,digest)
