# R25: fixed Student-t4 necessary-condition old TRAIN screen

Prospectively declared 2026-10-31 after accepted R24 synthetic numerical pass. No R25 real input array or label has been opened. The selected work is bound in `results/round24_student_post_result_decision_20261031.json`. This is a cheap falsification stage, not the complete candidate validation or a new confirmation set. No gate is relaxed or replaced.

## Fixed law and representation

Reuse the unchanged R24 Student-t4 producer and conditional-t5 independent reference. One center a, one covariance S=C_exec+h²I, Student scale S/2, ν=4, h=.25. Integrate normalized density over each native key cell, add exactly once 1e−30 times area, then native bias and softmax. No parameter search, mixture, extra network, clipping, covariance repair, tail truncation, or failed-row deletion.

C_archive is accepted R20 anchor covariance Ca=C+(μ−a)(μ−a)ᵀ in reference-8 coordinates. Preserve its exact bytes. Require the original normalized symmetry tolerance 1e−12, and define C_exec=[[Ca00,Ca01],[Ca01,Ca11]] before any new data read. This matches the upper off-diagonal used by old Gaussian production; it is not a claim that old cached matrices were bit-symmetric or averaged. Save both matrices and normalized asymmetry/change. R24 exact symmetry, PSD admission, finite S and strict Cholesky then apply without repair.

Reuse stored float32 trace τ from R20, independently check τ=10*.25²/(.25²+tr Ca) using the original overflow-resistant float64 formula then float32. Compute the entire float32 Q24@K8.T matrix in Torch, then multiply by τ32 in float32 and cast bias to float64. Do not normalize features again or use rowwise/NumPy GEMM. Final P uses unchanged R24 NumPy64 normalization: the equation matches the old rule, but bitwise equivalence with old Torch64 softmax is not asserted. First-index argmax is int32.

## Population, inputs and separation

All 48 original R20 old TRAIN pairs: 16 each harvard_c3/hv_c3_1, harvard_c5/hv_c5_1, harvard_c6/hv_c6_1, in original lock order. Only q24→g8: 48 queries ×432 native keys per pair, total 2304 queries. Never filter by predicted FOV/front flags or labels.

Use the accepted R20 `forward/support_records.json` paths, not the R20 lock's older R17 support_path. Six decoded fields only: g24__a float64[48,2], g24__anchor_cov float64[48,2,2], g24__tau32 float32[48], g24__q_features float32[48,128], g8__k_features float32[432,128], g8__bounds_px int64[432,4]. a/Ca are already in reference-8 units; only bounds convert by (bounds_px−.5)/8. Validate full archive and exact whitelisted array hashes. No original RGB/depth/camera array reopening; preparation provenance is inherited from accepted R20 audits, not freshly independently reconstructed here.

Joint producer, reference, metrics, tests and this protocol source closure are frozen before any actual cache read. Both locks have identical runtime/sources/cohort. Freeze touches metadata JSON and hash strings only; original target/label bytes or NPZ arrays cannot be read. A metadata-only preflight with np.load and integration disabled tests this separation. Producer saves pair inputs, all per-query raw masses, floored masses, P, argmax, statuses and exceptions/available partials; unsuccessful rows have no fabricated P. All 2304 statuses are sealed before independent numerical audit. Label-stage code cannot start until accepted complete prelabel audit.

## Numerical and resource gates

Same R24 GL24 production tiles (edge≤4h), maximum 4,000,000 cell-nodes per call. Separate real-population production phase limit600s includes all input loading/preparation and output sealing up to declared phase end; report finalization separately. Failures and remaining unattempted rows after exhausted budget remain explicit. No unseen inputs after the time limit may create unrecorded attempts.

Independent reference checks every query using GL64 and GL96 conditional-t5 integration (4608 calls), panel side≤4h, total250,000,000 cell-nodes and whole numerical phase1800s. These population budgets differ from R24 synthetic budgets; integrator accuracy and per-call production budgets do not change. Every actual attempt, failure, partial and consumed work is retained. Fixed ordering is pair, query, then64/96. No selective retry.

Required per row: production P max absolute error≤1e−4 versus reference96; raw FOV relative and raw L1/reference FOV≤1e−6; 64/96 P difference≤1e−8 and both raw relative discrepancies≤1e−8; P normalization≤2e−12; native sum versus separately integrated enclosing FOV absolute≤1e−8. Independently reconstruct area floor, native bias, normalization and first-index winners. Positive finite reference masses are required; zero/unrepresentable reference does not count as a pass. Exact and 1e−12 near-ties, zero/tiny raw rows, floor fractions, covariance asymmetry and costs are diagnostics retained for all rows. This is an empirical check of this full old population, not a general-domain quadrature certificate or runtime claim for training.

## Old TRAIN labels and eight necessary gates

Only after complete sealed prediction plus numerical/input audit: original label_path fields gt_xy and gt_visible, target_path fields g24_point_xy and g24_point_visible, and original locked partitions. Verify exact archive hashes at this stage, never before. Retain all2304 queries and independent valid-center masks, total1096 (c3=382,c5=402,c6=312). Region counts use all original visible pixels; no validity-mask merging. Within each scene aggregate divergence sum/visible-pixel mass, then give the three scenes equal weight. PCK uses exact Fraction of each scene's correct/valid counts and equal-scene average. Report KL, JS, direct KL/JS, mean distance/RMSE and each scene; only the inherited gates below decide this stage.

1. q24_g8 regional KL≤1.02×accepted old strong_quad.
2. Same bound versus old legacy_quad16. These are two role checks of the same numerical configuration, not independent baselines.
3–8. q24_g8 PCK≥reference PCK−1/200 versus old point_localizer, quartic_localizer, legacy_point_localizer, and R20 anchor_point_localizer, anchor_quartic_localizer, anchor_point_legacy. Exact rational thresholds; no rounded-percent decisions.

Baseline metrics and role identities are inherited by hash from accepted R20 summary; do not refit controls after viewing new effects. R20 anchor_trace is a matched Gaussian diagnostic, not a ninth gate. Producer and independent metric routes reconstruct counts and reductions separately: identities, integer counts, exact Fractions and gate decisions must match exactly; continuous sums/metrics use relative tolerance1e−10 and absolute tolerance1e−9. All disagreements outside these frozen replay tolerances must be resolved before interpreting efficacy. Scientific_gate_passed and goal_complete remain false even if all eight pass.

## Decisions and remaining requirements

Any numerical failure stops label evaluation for this frozen attempt. Any necessary efficacy gate failure stops this fixed t4 candidate; report all eight and retain all counterexamples. A pass only authorizes separately frozen expansion to all original14 conditions and full strong same-family controls, followed by old robustness/validation, actual online QKV/V task, optimized GS and untouched joint confirmation. No VAL, actual V, GS, fresh scene or new parameter selection is authorized in this round.

R24 does not establish novelty, finite exact QK/RoPE form or a Gaussian blur convolution. C=0 is a Student point kernel; t4 fourth moment diverges. The existing equal-covariance annulus counterexample is retained. The original generality/compactness/usefulness objective remains active regardless of this necessary screen outcome.
