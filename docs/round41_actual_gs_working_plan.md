# Next experiment: learned readout on actual context-trained GS

2026-11-01. Working implementation plan, not a frozen experiment or a result.
Historical draft: subsequently implemented and completed as
`ROUND41_ACTUAL_GS_PROTOCOL.md`; results are in `docs/round41_actual_gs_report.md`.
The text below records the plan before execution, not the current job state.
Implement and fix the concrete data/model contract before decoding new task
arrays or reading target RGB. Do not manufacture another static-integrator
prerequisite or expand this into a general numerical certification program.

The R40 depth candidate failed. Its point controls also sometimes improve when
V is reversed, so that residual-depth score is not sufficient evidence of useful
spatial correspondence. The next experiment tests actual color aggregation:
given context-only optimized GS geometry, can a compact learned positional
operator improve source RGB readout and held-view rendering? This may support
or reject a task-specific mechanism; it cannot erase R40's negative result.

## First executable task

Build a context-only data adapter around the existing R9 Teatime and
Waldo-kitchen artifacts. Reuse their 24 context cameras/images, fixed 12,000 GS,
opacity, SH0 renderer and the original full-image target population. Retain
the old 4 calibration/4 gate view identity as reused development, never fresh
confirmation. No target RGB in geometry, features, training or prediction.
R9 cameras came from all-image SfM; keep that calibration limitation explicit.

Actual query anchors are primitive means and query covariance is the saved
R diag(s^2) R^T, without an anchor correction. Work in the inherited context-
camera normalized coordinates and explicitly label their units. Retrain task
parameters in this convention; do not silently transfer SUN3D metres or treat
an SfM camera extent as a known physical metre.

Use frozen local ResNet18 context features, with a fully specified pixel/feature
alignment. Aggregate legal context-center observations into each primitive's
query descriptor. Source keys are dense context-image features, values are
actual RGB, and aggregation is the sole color output: no residual color head
that could hide an ineffective correspondence operator. Full source pixels,
fixed geometry/opacity and the original equal-source-view fusion are preferred
so the old readout baselines remain interpretable.

The key-side 3D support must come from the same context-only GS expected-depth
render for all roles. Specify its pixel footprint/finite-difference convention
and invalid-depth policy before extraction. Verify that key descriptors really
vary across keys; a uniform covariance makes part or all of a shape penalty a
row constant. Shared visibility and fallback must be explicit, with original
GS RGB for primitives that cannot be read. Keep all target pixels in metrics.

Train shared Q/K and any active positional scalars using context observations
only, with source and target context views distinct. Specify exclusion of the
training target view from query feature fusion. Fix a context-only selection
schedule, paired seeds/batches and finite training budget before fitting; these
context views remain old development because the fixed geometry used all24.
Save the selected colors and target renders before opening target RGB.

## Comparisons and decisions

Keep center RoPE, expected/normalized RoPE, the failed R40 decoupled descriptor,
same-capacity center-distance, isotropic and shuffled-shape controls. Rebuild
appropriate learned point/fixed/physical-support controls with the SAME legal
features/depth/visibility; also retain bilinear and original-GS references.
Do not compare different information budgets as evidence for shape. Report
active parameter counts; RGB V is deliberately a direct, parameter-free value.

The implementation should settle the exact roles, steps and selection rule in
one short prospective protocol, then run a small functional CUDA smoke and
the fixed full comparison. Use independent sample scores, full saved-image
metrics and identity checks proportional to the conclusion. No huge quadrature
budget, automatic exact-argmax requirement or additional reviewer hierarchy.

Report full-pixel MSE/PSNR and scene-wise results, fallback coverage, effective
V usage and latency/memory. A favorable color-regression loss is insufficient;
the held-view image comparison must beat strong spatial and original-GS
references before claiming actual utility. Existing final requirements remain
open. If the fixed R40 family fails here too, retain the failure and move on.

## Design hypothesis after the current failures

The useful distinction is between comparing two shape descriptors and using
shape to define a local distance around a center. A compact alternative worth
deriving is a query-centered Gaussian metric score,

    l_ij = content_ij - 0.5 (mu_j-mu_i)^T P_i (mu_j-mu_i),

where positive P_i depends on the true GS covariance and a very small number
of learned regularization/scale parameters. It has a small quadratic dot-
product lift, stays active when a=mu, and does not require an extra network.
This is a known Gaussian/Mahalanobis construction, not a novelty claim, an area
integral, or a proof of solving center/region conflict. Old quadratic controls
make strong comparisons especially important. Do not add it to the experiment
after seeing its results; specify it prospectively as a separate iteration if
the completed task adapter and mechanism evidence justify it.

No new GS arrays, training, colors or renders have yet been produced under this
plan. The immediate next action is to implement the adapter and concrete
context-supervised readout contract, then execute it on an idle GPU.
