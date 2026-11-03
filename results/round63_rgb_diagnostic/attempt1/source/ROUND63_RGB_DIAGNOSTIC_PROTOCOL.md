# R63 fixed-weight RGB geometry diagnostic

2026-11-02. Implements docs/round63_rgb_diagnostic_plan.md after the completed
negative R62 RGB comparison. No new fitting and no old_eval/confirmation access.
The hypothesis, competing explanations and outcome-dependent decisions are
those in that plan. Parameter substitution is a distribution shift, not proof
of a unique cause or a replacement for matched retraining.

Use the exact48fit/64val pairs and9selected models in preparation/availability.json.
Predicted-full: baseline,observed-center,observed-shape,observed-both,initial-shape,
sphere,and zero-new-relation. Predicted-point and observed-full: baseline only.
Three seeds11/22/33;two splits;54prediction groups. Batch8. Eval mode,all weights
fixed,and both layers receive the same declared intervention. Camera transforms,
RGB features,labels and scoring populations remain R62's. No new mask enters
attention. Second-layer features/predictions may respond to first-layer changes.

Only unmodified predicted-full/point branches capture each layer's local means,
covariances and raw parameter coordinates. Center distance/depth errors are in
metres (R62 coordinates times3). Shape discrepancy is the Euclidean norm of log
eigenvalues of C_observed^(-1/2) C_predicted C_observed^(-1/2). Observed supports
are sensor-derived references,not true GS parameters. Summarize mean,median,
p10/p90,min/max;report raw coordinates and tanh-coordinate saturation above.95.
For raw diagonal coordinates use abs(tanh(x/2));for lower coordinates abs(tanh(x)).
Do not interpret those fixed descriptive cutoffs as method acceptance gates.

Coverage is independently sampled from the same192old raw depth frames using
the original resize/nearest-depth convention. Report all tokens,known native
centers,patches with>=16of64samples,and entirely missing patches separately.
These are pair-token exposure counts,not independent-frame sample sizes.
Native center flags and patch counts are analysis fields only;no metric filtering.

For observed range diagnostics,set M=C-floor^2 I and symmetrize. Add scalar
jitter max(1e-12,1e-12-lambda_min(M)) per token before Cholesky;record it and the
fraction with nonpositive lambda_min. Clip unique Cholesky diagonals to
(.025,.030,.010)*[exp(-2),exp(2)] and lower coordinates to[-.03,.03]. Record
coordinate violations (absolute coordinate tolerance1e-8),relative Frobenius
change,and log-spectrum change after reconstruction plus the original floor.
This coordinate clipping is NOT a nearest-covariance projection. Tiny
near-floor differences are not evidence of a material expressivity bottleneck.

Before execution,check original-branch equivalence and geometry replacement,
zero-relation equivalence,label/coverage exclusion,and simple analytic statistics.
Freeze sources,prepared sample identities,selected models and input hashes.
Use verified idle GPUs and a CUDA probe,then run all fixed forwards. Compare
all original val logits against R62 saved values (maxabs<=1e-3 addresses material
differences),and compare derived CE to R62 selected CE (<=1e-5). Recompute metrics
with the existing independent NumPy summation path;check discrepancy<=1e-5.
These bounds establish implementation consistency,not scientific utility.
Weights must be unchanged before/after. Review metrics,geometry strata and plots
before choosing the next minimal design. Keep R62 unchanged and all negatives.
