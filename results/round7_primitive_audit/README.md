# Real 3DGS primitive scope and mechanism audit

This is an exploratory replay on old Tea/Waldo validation pairs. No Round7 fresh data or old test pair NPZ was opened; shared reconstructed geometry can itself contain old evaluation-photo information. No parameter was fitted and no GPU was used.

- Individual optimized primitive samples test operator equivalence only, without primitive correspondence labels. Actual exported covariance agrees with R diag(s²) Rᵀ within 3.84e-7 relative error.
- Unequal key covariances invalidate an unrestricted exact five-channel claim: row-centered full log-kernel rank is 19/14 at the old 12px blur and 57/45 at one-pixel blur. A common-key-C counterfactual remains exact to 9.72e-14 in probability.
- All 72 old validation camera pairs and both parities enter the independent SfM regional audit. Four empty conditions are recorded explicitly; 140 nonempty pair/parity probability archives remain. All controls share means, masks, targets, mass and the old development bandwidth.
- Strict95 pooled CE is point 1.23445, center-dispersion 1.29912, trace-full 1.34295, full 1.30907; alternate parity has the same ordering. This supplies no positive evidence for internal primitive R/S at the fixed operating point.

`protocol.json` fixes the operating point and scope. `summary.json` stores every sample diagnostic, mask omission, regional aggregate, source/data digest and projection check; `rows.json` stores per-query metrics. `primitives/` contains actual sampled parameters, and `regional_probabilities/` contains predictions/labels/masks for replay. `source/` preserves the executed implementation. The first execution is retained in `../round7_primitive_audit_initial/`; the final run only adds explicit empty-condition accounting, without changing scored results.

See [scope analysis](../../docs/round7_gs_scope.md) for mathematical limits, prior negative evidence and primary paper links. The five-channel polynomial Gaussian attention lift is established mathematics, not a novelty claim.

Reproduce into a new output directory:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m experiments.round7_primitive_audit --out results/round7_primitive_audit_replay
```
