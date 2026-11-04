# Round 10 B: numerical selection-guard correction

2026-10-30. The original B selector stopped before writing `selection.json`
because it required exact Python-dictionary equality between point and center
calibration summaries. These two native-cell operators are mathematically
identical. The completed GPU calibration summaries differed at floating-point
reduction roundoff (the parent agent measured a maximum of approximately
`2.22e-16`), while point PCK was equal. This is an implementation guard failure,
not a failed research effect gate.

The helper's read-only audit recomputed the combined three-scene summaries and
found a maximum recursive discrepancy of **4.440892098500626e-16**, including
the additional arithmetic used to form macro averages. All 12 h/T comparison
pairs satisfy the proposed `1e-12` tolerance. Eight dedicated repair tests pass.

No frozen B source, protocol, lock, calibration artifact, candidate score or
metric is edited. A separate helper,
`experiments/round10_cell_selection_repair.py`, changes only that recursive
positive-control equivalence guard to **absolute tolerance 1e-12**. Nonnumeric
values and structure must still match exactly; all numeric values must be
finite. Larger discrepancies still fail. The measured maximum discrepancy is
saved with the repaired selection.

The original `min`/`max` selection operations, configuration insertion order,
tie handling, bandwidth/temperature grids, selected baselines and all A/B effect
gates remain unchanged. In particular, the helper does not round, average or
replace point/center calibration scores to force an exact tie. A numerical
near-tie can still resolve exactly as it would in the original selector after
its guard.

The helper provides three explicit stages:

1. `audit`: read-only verification of the original B source/data lock and all
   calibration shards, followed by independent recomputation of the same
   summaries and the guarded discrepancy. This reads no B gate labels.
2. `freeze`: require no existing selection or evaluation directory; verify the
   same original artifacts; write `selection_repair_lock.json` and its digest,
   recording this document, helper source, original B lock and every original
   calibration artifact. This happens before any repaired selection.
3. `select`: verify the extension lock and every original calibration artifact,
   run the unchanged selection operations with the corrected equivalence guard,
   and write the standard selection fields plus correction provenance.

Existing selections, extension locks or evaluation shards are never overwritten.
The frozen B evaluator can read the resulting selection through its existing
selection/hash checks; the correction is visible in the saved selection and
delivery manifest. The evaluation metrics and acceptance thresholds are not
relaxed. The original selector failure should be retained in the final report.
