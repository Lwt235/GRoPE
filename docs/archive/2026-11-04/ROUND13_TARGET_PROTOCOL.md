# Round13: training-only regional and center target compatibility by scale

2026-10-30. Motivation is the frozen R12 failure of full4 at q24→g8 center PCK, despite useful direct regional KL. This diagnostic separates properties of the sensor targets from method-specific approximation artifacts. It cannot explain away the failure or modify any existing gate. No candidate, rho/smoothing choice, method configuration, fresh validation/test label, or predicted geometry is evaluated here.

## Scope and freeze

Use all 48 already fixed R10 training pairs, exactly 16 each from Harvard c3/c5/c6. All queries of q8, q16 and q24 are retained, always targeting native g8 cells. There are 48×(432+108+48)=28,224 query records. Before this script reads any old training label or sensor image, freeze this document, script/tests, read-only R11 convex-projection helper, frozen native sensor helpers, exact cohort manifests and source selection/camera/index metadata. Bind expected sensor file hashes without decoding files at freeze. Bind the previous R11 diagnostic lock. No frozen R10/R11/R12 file changes.

The sole non-training result read is the already frozen R12 quad16 direct regional KL summary, recorded at freeze as a **descriptive magnitude reference**. Retain both q24→g8 direct KL and five-condition scale macro direct KL, and exactly .02 times each. These refer to 32 old validation pairs, whereas all new target analysis uses 48 old training pairs. Ratios are therefore cross-cohort magnitudes, not a mathematical common-population bound, tolerance waiver, tuning signal or method selection. Do not rerun quad16, read its validation probabilities/labels, or select a configuration on these results.

## Supervision and populations

Reconstruct each qstride→g8 native hard-cell histogram directly from the same frozen dense independent sensor projections. Every visible query image pixel has mass 1. Query ownership uses actual integer source coordinates and stride; target physical image edges are [-.5,191.5)×[-.5,143.5), native8 bins use floor((xy+.5+1e-9)/8). Source q24 has 48 exact 24×24 regions, with no g16 parent mapping or common-grid approximation.

After freeze, restore original RGB↔sensor timestamp associations and decode original sensor q/k depths. For each q8/q16/q24 region, sample query sensor depth at its native center `(stride*i+(stride-1)/2)` and project with original intrinsics/poses. Use existing independent float64 native projection and .05m+.05×depth consistency. Check reprojected q8/q16 center coordinates/validity/status against original frozen labels; save all three scales' center targets and counts. The q24 target is not a regional mean or a candidate estimate.

For every query retain area-valid, center-valid and joint-valid flags, visible mass and every one-sided/neither-valid population. Compatibility PCK, histogram mass pairs and conditional minimum KL are defined only for joint-valid rows. An invalid center has no center constraint; an empty area histogram has no defined regional KL. Thus in the additional **all-query** and **all-visible-area-mass** averages, their contribution is explicitly zero cost. This is a relaxed compatibility cost convention, not a claim that their localization is correct. Report the exact denominators and conditional metrics separately.

## Fixed diagnostics

An allowed key has native center distance ≤8px from the independently projected valid query center. For each joint-valid query store allowed key identities/count, native first-argmax identity/distance/correctness, whether any tied histogram maximum is allowed, top-tie count, global maximum normalized bin mass, largest allowed bin mass, allowed-set total mass, maximum-minus-best-allowed gap, and allowed/global maximum ratio.

Compute minimum KL(g||p) subject to at least one allowed key being a largest entry, allowing ties. Use the already frozen R11 convex pooling: for an allowed candidate j, pool g_j with descending larger entries until the pool mean exceeds the remaining entries; set the pool to that mean, leave all others unchanged. The allowed candidate with largest original g is optimal by the swap argument. Additionally enumerate all allowed candidates on every real query and assert identical optimum. Synthetic SciPy constrained-optimization, identity grids, zeros/ties and denominator tests are run before freeze.

Strict conflict means the largest allowed original bin count is strictly less than the global largest count; it does not depend on a floating threshold. Native first-argmax conflict can occur with zero minimum KL through a tie. The relaxed lower bound does not guarantee the native first-index decoder chooses an allowed cell.

Report pooled and scene-equal results for all three query scales: deterministic/optimistic regional-mode PCK on joint-valid rows; strict conflict fractions on joint-valid and all queries; query-equal and visible-mass-weighted KL for joint-valid populations; all-query, area-valid-query and all-visible-mass means with unconstrained zero cost; complete validity/mass counts; KL quantiles for all/joint/strict rows; global/allowed bin mass, gap/ratio and allowed-count distributions. q24 conditional and unconditional scene-equal KL values are compared to the frozen .02×quad16 magnitude references, clearly labeled as cross-cohort descriptions.

## Execution and interpretation

CLI stages `freeze`, then `run`; immutable output namespace `results/round13_target_conflict`. Verify all source/cohort hashes at each stage. Run saves one compressed target archive per pair, all 28,224 query diagnostic records, source/label provenance, summary and full artifact completion hashes. No overwrite or silent partial resume. All failures remain explicit.

This is old-training supervision analysis, not new confirmation, a deployable oracle, a PE performance lower/upper bound, proof that a particular approximation can jointly satisfy the tasks, or permission to remove the PCK gate. A large conflict rate can still have tiny KL cost. A low cost does not guarantee a method using only RGB predictions can achieve it. Method-induced multimodal/binned approximation artifacts remain a separate hypothesis; no smoothing parameter or new model is selected by this diagnostic.
