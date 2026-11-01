# Round 8 oracle diagnostics (old validation only)

Independent NumPy reconstruction from saved predictions; five paired seeds and two already-used validation scans. No confidence interval or generalization claim.

| Role | AbsRel | MAE (m) | RMSE (m) | Boundary AbsRel |
|---|---:|---:|---:|---:|
| pred_pred | 0.161131 | 0.427816 | 0.600949 | 0.156306 |
| gt_route | 0.160680 | 0.427125 | 0.599556 | 0.156091 |
| gt_value | 0.152625 | 0.399095 | 0.573042 | 0.148264 |
| gt_both | 0.151664 | 0.396555 | 0.568250 | 0.147426 |
| singleview | 0.162354 | 0.429408 | 0.604750 | 0.155236 |

Development continuation gate: **FAIL**.
Sensor routing with original predicted V versus singleview: relative AbsRel difference -1.031%; 5/5 seed macros and 2/2 scene means improve.

Negative differences indicate improvement for error metrics; positive delta1 differences indicate improvement.

Factorial differences, coverage partitions, seed/scene matrices, exact population counts and audit records are in `summary.json`.

Oracle inputs are privileged diagnostics, not deployable results or strict upper bounds. Checkpoint selection uses the same development validation, so the factorial contrasts are not strict causal effects.
