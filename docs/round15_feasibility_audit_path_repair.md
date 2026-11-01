# R15 independent auditor: path-resolution repair

The original auditor is unchanged. Its first execution failed while hashing
source-input files: it resolved repository-relative metadata paths against `/`
instead of the repository root. The recorded failure occurred before reading
the experiment completion, probability arrays or metrics. The failure record
is a transcription of the original tool output, not a redirected process log.

The original lock is
`9684280daaef94648da3e2670bc5fb479e38258f0c75c3936c07c34a6a0937a1`.
The immutable failure record is
`results/round15_feasibility_independent_audit/path_resolution_failure.json`,
SHA256 `11c08e4489f14e743c7cef3bea2d11949005cd0ab08bbffbcf552b515dfb1405`.

The new `experiments/round15_feasibility_audit_path_repair.py` copies the original
auditor. Its only data-path correction replaces `Path('/')` with `ROOT` for
input and manifest integrity checks. Absolute metadata paths retain their
normal pathlib behavior. No probability, target, cohort, configuration,
threshold, Fraction selection, summary comparison or population check changes.
The full 5,184 probability arrays and center metrics, 216 numerical-preflight
matrices and 108 constraints are rechecked.

The repair freeze binds the original lock, source and source snapshot, the
producer lock and sources, the failure record, this explanation, and the new
repair source and tests. `created_unix` records the actual new freeze time.
`original_design_created_unix` comes from the hash-verified original lock and
is used solely to check that the original design preceded all forward work.
The repair does not claim to have existed before forward execution. Its lock
also records the information already reported to root: producer eligible
indices `[6, 9]`, with detailed scores unopened. These indices are provenance
metadata only and are never used by the repaired selector or assertions.

The repair source is reviewed and frozen before its first result read. Tests
exercise relative and absolute paths, tampering, original-source preservation,
unchanged scientific audit logic, refusal to overwrite, and rejection of
backdated or changed-policy provenance. Any further runtime failure remains
visible and does not justify changing thresholds or skipping rows.
