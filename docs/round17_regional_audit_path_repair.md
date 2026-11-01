# R17 regional audit: historical snapshot policy repair

This is a separately timestamped engineering repair after the Stage B source
freeze and execution. It changes no observation, prediction, configuration,
metric, eligibility rule, J selection, numerical tolerance, futility check or
validation requirement. The original producer, independent auditor, source
snapshots, output and failed audit logs remain unchanged.

The frozen R17 `verify_legacy` incorrectly required a flat R12 snapshot for
every item in R12's transitive `source_hashes`. The actual frozen R12 `freeze`
function copied exactly three files:

- `ROUND12_STRESS_PROTOCOL.md`
- `models/round12_stress_pe.py`
- `experiments/round12_stress_probe.py`

It did not copy every Python dependency or every earlier protocol. The first
missing file encountered by the new audit was `ROUND10_PROTOCOL.md`. This is
a verifier assumption error; it does not establish a missing historical
artifact that R12 had promised to save.

The replacement retains complete verification of every original source file
against R12's frozen `source_hashes`. All three promised flat snapshots must
exist and match. Any additional existing source snapshot must also be bound
and match; unbound or ambiguous snapshots fail. Other historically uncopied
sources are explicitly inventoried rather than fabricated or ignored.

`experiments/round17_regional_audit_path_repair.py` temporarily replaces only
the original module's `verify_legacy` in the new audit process, restoring it in
`finally`. The original `_audit`, reference/count routines, metrics, selectors
and gates remain the original function objects. An AST test checks that the
replacement legacy verifier differs only in its snapshot block. A metadata
writer adds the repair lock and source provenance to the new audit's lock and
verification; it does not alter scientific values or decisions.

The new repair namespace has its own `freeze` then `audit` stages. Freeze
decodes only source/lock/failure metadata; it hashes, without JSON-decoding,
the already completed producer summary, completion, seal and retained logs.
The original metadata failure record explicitly says evaluation existed when
the failure was recorded. Its stage label describes the failed precheck, not
the global execution phase at recording time. This repair is not described as
pre-forward or backdated. The original auditor failed in `verify_legacy`
before its own new-regional summary/array/label decoding and before creating
an audit directory; no nonexistent original audit lock is manufactured.

Both root and the repair author report that they have not inspected new
Stage B regional effects. The freeze CLI records this explicit attestation;
it does not pretend a hash can prove a person's access history. The complete
original pipeline/audit logs and failure metadata are hash-bound. Real original
failure/completion artifacts are additionally bound if they exist.

The new audit runs unchanged scientific checks into the repair namespace's
`audit/` subdirectory. Its verification distinguishes audit correctness from
the TRAIN futility result: a correctly computed failed screen remains a stop.
Any unrelated failure propagates and is retained; the repair supplies no new
fallback, tolerance or permission to run old VAL.

After independent review, the intended commands are:

```bash
python -m experiments.round17_regional_audit_path_repair freeze --attest-no-regional-scores-read
python -m experiments.round17_regional_audit_path_repair audit
```

The acceptance artifact, if the complete audit passes, is
`results/round17_regional_audit_path_repair/audit/verification.json`.
This document and repair source/tests must be frozen before that audit runs.
