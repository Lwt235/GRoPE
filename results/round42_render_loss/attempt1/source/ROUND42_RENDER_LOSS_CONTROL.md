# R42: matched center-color versus image-rendering supervision

2026-11-01. A bounded old-development control, not a new validated PE. R40/R41
decoupled failed. R41 Waldo shape improves center-color selection loss while
hurting target-image MSE. The pending R41 component renders do not choose the
roles or hyperparameters of this experiment.

Question: under identical multi-source fusion, initialization, view schedule,
capacity and image-based selection, does changing the training objective change
the relative utility of the existing PE? Competing explanation: decoupled has
an unhelpful positional prior even when optimized for the actual output. Stop
this loss-control branch after the fixed comparison; no frequency/temperature
or architecture search within it.

## Fixed factorial and data

Three unchanged R41 models: point,full,decoupled. Two objectives: center,image.
Two R9 scenes, five paired seeds11,22,33,44,55:60 runs. This is not the complete
method/ablation comparison required for promotion. The R41 eleven-role results
and all18 external R9 methods remain references. No untouched data access.

Reuse R41 prepared features, true GS means/R/S covariance, expected-depth pixel
supports, source eligibility, scene units, direct RGB values and fixed renderer.
No new network or geometry/opacity parameter. Every training forward uses ALL
eligible primitives and ALL27648 keys in each legal source view, with query
chunks256 solely for memory. Context indices3,9,15,21 are selection targets;
the other20 are training views. For training target q, exclude q from source
views and query-feature fusion. Average the resulting19 source readouts per
primitive, dividing by its actual eligible-source count. If count0, retain the
original context-trained GS color. This fusion is identical for both objectives.

Center objective: mean RGB squared error between the fused color and target-q
bilinear center RGB, over target-visible primitives with at least one legal
source. Image objective: mean squared error over EVERY RGB element in the
rendered target-q image. Fixed geometry used all24 context images; selection
views are a development proxy, not independent geometry validation. No target
view from the old8-view evaluation enters training or query features.

## Optimization and memory-bounded gradient

100 AdamW updates,lr.001,wd.0001,clip1; Q/K identity and the unchanged R41 shape
scalar initialization. One training target uniformly drawn per update from20
views. All six cells of the factorial share the exact target sequence per seed.
Both objectives evaluate at steps0,20,...100 and choose the earliest strict
minimum of the SAME four-selection-image full-pixel MSE. For selection, source
and query-feature fusion use all20 training views. Final inference uses all24
contexts, exactly as R41. No target-image checkpoint or bandwidth selection.

This differs from R41's single-source batches,500 updates and center-based
selection. Only the WITHIN-R42 factorial isolates the declared objective change;
R41-to-R42 changes cannot be attributed solely to loss choice or equal compute.

For C(theta), the fused primitive colors, first compute C without retaining the
attention graphs. Differentiate the chosen loss through the fixed renderer or
center loss to obtain g=dL/dC. Recompute source attention in chunks and accumulate
sum_i <g_i/count_i, dC_source_i/dtheta>. This is the ordinary chain rule, not an
approximate gradient or a new model. Zero-gradient primitives may be skipped
only in the backward pass. Original fallback colors have zero model derivative.
Keep the shared key graph until its last query chunk; clip/update once after
all source contributions. Finite-precision accumulation need not be bitwise
identical to a monolithic graph.

Before real training: CPU float64 comparison of monolithic and split gradients
for all three roles and both losses, fallback/target-exclusion cases; then one
GPU actual-renderer color-gradient directional check and one training step for
each objective on old context data. The GPU smoke is a functional check, never
score-based selection. TF32off,CPU4,float32 training,16GiB allocation ceiling,
3GPU-hours per scene/seed shard. Use only genuinely idle GPUs; preserve others.

## Evidence and decisions

Save histories, batches, all selected weights, full colors and renders. Preserve
all four R41 target calibration/gate identities and all pixels. Seal60 complete
runs before evaluating target RGB for this experiment. Reuse independent NumPy
score/RGB checks and complete image metric replay. Score warnings require impact
assessment, not automatic scientific failure. No extreme quadrature prerequisite.

Report each objective/role's scene and seed results, their changes in ordering,
and paired differences in decoupled-minus-point/full between objectives. Point
and full are predeclared controls; decoupled is not replaced by a winning cell.
An image-trained candidate is only promising if its gate MSE improves>=2% over
both image-trained controls and the best original simple readout, strictly beats
original GS, improves both scenes and>=4/5 seeds. Even that would not resolve
R40's depth failure, omitted ablations, full applicable geometry/robustness/cost
or untouched joint confirmation. No goal-completion claim from this control.

If image loss improves all roles without closing the decoupled deficit, the
proxy alone does not explain the failure. If decoupled still fails, stop promoting
this raw descriptor-similarity family and use the evidence for a separately
specified compact design. Do not keep extending this control with rescue sweeps.
