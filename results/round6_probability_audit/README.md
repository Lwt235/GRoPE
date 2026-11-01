# Independent audit of the sealed Round6 result

The frozen candidate passes its predeclared criteria. Independent NumPy arithmetic reproduces all 384 checked metric/CI quantities exactly, using the original row numerators/denominators and the same 20,000 paired named-scan draws. All five scan groups improve total KL. No candidate, setting, source file or gate was changed after the test.

| comparison: full minus frozen fixed | relative KL change | paired central95% relative CI |
|---|---:|---|
| all four grids | −6.08% | [−8.42%,−4.09%] |
| q8 to k16 | −3.20% | [−4.24%,−2.44%] |
| q16 to k8 | −9.51% | [−13.55%,−5.86%] |

Both same-scale guards pass. Removing any one scan retains a total KL improvement of 5.26%–6.86%. Brown, MIT and home acquisition groups all improve descriptively; no three-group CI is manufactured. Full versus isotropic is exploratory: KL improves 1.49%, with absolute difference CI [−.005347,−.003854]. This is not used to select a new candidate.

Point accuracy does not improve. On q8_k8, full−fixed PCK@8px is −0.00775 percentage points, central95% CI [−0.06364,+0.04626] pp. Full−PCK-optimal point is −0.08940 pp, CI [−0.17605,−0.01331] pp. The latter indicates a small point-accuracy cost; it still passes the predeclared −0.5pp noninferiority guard. Reporting only “noninferior” would conceal this direction.

The numerical subset contains **200 probability archives**: five roles × five scan groups × four grids × two camera pairs. These are 40 grid/pair input conditions, but only **10 distinct camera pairs and five query cameras**, not 200 independent examples. Only these fixed inputs are independently evaluated; the remaining predictions are not replayed. All 480 camera pairs × four grids enter the separate all-row CI arithmetic check.

For the fixed subset:

- Saved float32 probability/label metrics agree to at most 2.05×10⁻⁷ nats per measurement for CE/KL and 1.69×10⁻⁸ for JS.
- Independent direct Gaussian Mahalanobis energy, using the locked RGB inputs and configurations, then float64 stable logsumexp, reproduces KL to 1.07×10⁻⁶ nats per measurement. Maximum probability difference is 4.50×10⁻⁵. Mean soft-EPE discrepancy is at most 1.14×10⁻⁵px.
- PCK8, PCK4 and target-stride PCK counts agree strictly for all 200 archives, both from saved probabilities and from independent energy. No tie allowance was needed. Forty-one point rows have reconstructed top-score gaps within .002, but their aggregate counts remain unchanged.
- No positive GT mass meets a zero/subnormal saved probability in this subset. The auditor explicitly handles this limit rather than replacing impossible CE recovery with a log clamp. In other subsets such probabilities would leave probability-only CE/KL unidentified; energy reconstruction would be a separate check.
- All sealed source hashes, frozen evidence files, role configurations and source metadata still match the root lock. Seven analytic audit tests pass.

The statistical units are five named scan/place groups. Physical-room independence is not guaranteed, bootstrap precision is limited, and inference is deterministic with no training-seed replicas. The measurements concern conditional visible source-image area, supplied reconstruction cameras and RGB-predicted Gaussian support; they do not establish a benefit from optimized 3DGS primitive R/S or new-view rendering.

`summary.json` retains every subset result and tolerance, independent CI reconstruction, source/input/probability hashes and all limits. `verification.json` records strict count checks and the additional sealed-evidence checks. `source_snapshot.py` is the exact independent auditor that produced the result.
