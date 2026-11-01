# Round7 coarse depth confirmation

Locked strong baseline: **point**. Primary research gate: **failed**.

All values use valid-query counts within scan, then equal scan and seed weights. Intervals use 20,000 paired crossed scan/seed draws; no pixel resampling or deterministic-baseline seed replication.

| Role | AbsRel | RMSE (m) | MAE (m) | delta1 |
|---|---:|---:|---:|---:|
| full | 0.171546 | 0.589778 | 0.404571 | 0.788351 |
| fixed | 0.171615 | 0.589916 | 0.404727 | 0.788149 |
| isotropic | 0.171645 | 0.590282 | 0.405100 | 0.787536 |
| point | 0.172123 | 0.591970 | 0.406961 | 0.785904 |
| point_pck | 0.171505 | 0.589589 | 0.404324 | 0.788491 |
| singleview | 0.169525 | 0.581457 | 0.394109 | 0.799319 |

Full minus strong baseline AbsRel: -0.000577, 95% CI [-0.0013576341881760502, 0.00037215710108726635]; relative change -0.335%.
Improving seed macros: 5/5. Full minus singleview AbsRel 95% CI: [-0.0032011129510007, 0.0054783968943598865].
RMSE relative-change 95% CI versus baseline: [-0.006863624320998384, -0.00027033390899723386].

Every control, per-scan/seed result, leave-one-scan sensitivity, frozen raw baseline and failed criterion is retained in summary.json. The raw baseline is descriptive. Five place groups do not establish arbitrary institutions or a SOTA native-resolution depth result. The analyzer verifies additive summaries and locked populations/artifacts; separate prediction auditing must verify arrays against sensor labels.

Frozen full-model value interventions (descriptive):
- zero_v: AbsRel 0.172684, mean absolute output change 0.065452 m.
- reverse_v: AbsRel 0.173147, mean absolute output change 0.087049 m.