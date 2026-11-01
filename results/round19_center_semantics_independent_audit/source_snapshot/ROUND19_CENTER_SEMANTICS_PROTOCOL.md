# R19: source-center information, old-TRAIN diagnostic

2026-10-31. Selected after the accepted R17 scientific failure and R18 fixed
integration-budget failure, before new center predictions or diagnostic scores.
This is a separately declared exploratory screen, not an amendment to either
failed round. It does not select a new Gaussian PE or test a scientific efficacy
gate. The continuous R18 law has not been disproved by its numerical failure.

## Question and fixed population

Does the legitimate projection of the designated source center carry useful
localization information that the projected region mean lost? Compare two
geometric locations on exactly the existing 48 TRAIN pairs: Harvard c3/c5/c6,
16 pairs each. Query partitions are g8/g16/g24; the target key grid is always
g8, with 432 cells. Keep every 432/108/48 query per pair, totaling 28,224 rows.
The cohort and identities come from the R17 feasibility target-contract entries,
not a new directory glob, frame selection, or subset of failures.

Prior exact center-valid counts are 9,521/2,463/1,096 for q8/q16/q24. Evaluation
must reproduce them. They are not row weights across scenes: aggregate integer
correct/valid counts within each scene and scale, then average the three scene
Fractions equally. All rows are saved; only the inherited sensor-center-valid
mask determines which rows have a center target. Never intersect it with a
predicted in-front/FOV mask, area-label validity or a previous method's success.

## Legal forward quantities and inherited evidence

The original area mean mu, query bounds and fixed R17 winner are reused from
the accepted R17 size-attribution forward artifacts described in
`docs/round19_center_semantics_preparation_draft.md`. Bind their actual archive
hashes to the old completion and source/acceptance chain. Their reuse is
inherited evidence, not a new independent recomputation of the original means
or attention probabilities. Only decode required mean/bounds/argmax arrays;
there is no need to open full probability arrays or size-control effects.

New a uses exactly the R13 `predicted_centers` function, extracted from its
source AST with NumPy as its only external mathematical dependency. Freeze
both the whole source-file hash and function-source hash. Inputs are only
`q_predicted_depth,q_K,k_K,q_to_k_transform` and the original bounds. The
existing contract is retained: RGB depth NaN/Inf sanitization, [.1,20] depth
clamp, bilinear sampling at (lo+hi-1)/2, camera transformation, z denominator
lower bound .1, output coordinate clamp [-1024,1024]. In-front and FOV flags
are descriptive; they never alter the output population. These existing
conventions are not newly selected according to scores.

a is in pixels; saved mu is reference-8 and is multiplied by eight exactly
once. For each location, compute squared Euclidean distance to the original
native g8 cell centers and select the first minimum index. This rule also
applies outside the FOV. No GT allowed set, learned point winner or containing
cell is substituted. **Nearest-a and nearest-mu are diagnostics, not the
deployed content-aware point model.** The original R17 h=.25/T=10 argmax is
retained as one fixed descriptive comparator, never a forward anchor.

Forward must save a, mu, a-minus-mu, native nearest indices, fixed R17 argmax,
finite/FOV information, a's projected in-front flag and exact row identities
for all 48 pairs. No in-front status is invented for the reused 2D mean.
No Gaussian is re-centered or integrated; no C_a probabilities, h/T grids,
new model training, old VAL, optimized GS or fresh data are used here.

## Freeze, chronological guards and independent audit

Before any new real prediction, jointly freeze this protocol, producer,
independent auditor, their synthetic audit tests, both design/preparation
notes, R13 function source, exact input/support/winner file bindings, inherited
acceptance/completion/lock bindings and expected target hashes. Source copies
and a lock checksum are mandatory. Runtime is NumPy on CPU with OMP/MKL/
OpenBLAS thread variables one; record versions, build identity and executable.
Software-only synthetic tests may check projection, nearest-cell ties, invalid
inputs, exact Fraction boundaries and population handling before real forward.
No real diagnostic effect may be computed during those tests.

Freeze/forward read only existing expected target hashes in accepted metadata;
do not hash or decode actual target archives yet. Run the complete forward and
seal all predicted arrays and fixed comparator identities globally. Then a
separate independent forward audit must pass before producer evaluation is
allowed to hash or decode the R13 targets. g8/g16/g24 all use that exact target
source, not merely-allclose older coordinate exports. All sources and input
bindings are checked at phase boundaries; started/completed phases are never
silently overwritten, and numerical failures are retained.

The independent auditor must not import producer or R13 mathematical helpers.
Reconstruct bilinear center depth, rays with an independent linear solve,
projection/clamps/flags and native nearest cells for every new row. Require
maximum absolute projected-coordinate discrepancy <=1e-9 pixels, exact flags
and exact nearest indices; do not use a tolerance to change first-index ties.
Verify old mu/bounds/winner archive and array identities rather than claiming
a new independent audit of their historical Gaussian integration. Full source,
file, population, row and global-seal checks apply before label access.

After evaluation, independently recompute all integer counts, transitions,
exact Fractions and screen decisions from the saved predictions and original
labels. Continuous-error sums and squared sums use tolerance
`1e-8 + 1e-12*abs(reference)`; derived mean/median/RMSE use
`1e-10 + 1e-12*abs(reference)` to accommodate reduction roundoff. These
tolerances never apply to counts, Fractions, screen decisions or array hashes,
and continuous errors never determine the screen. Bind producer and audit completions before
root interprets effects. An audit failure and a screen failure remain different
outcomes; neither means the overall research goal is completed.

## One new exploratory progression screen

This screen is newly chosen now, not a historical pre-registered efficacy gate.
The half-percentage-point scale is used as a conservative relevance/protection
criterion, corresponding to the old localization tolerance; it is not a claim
of significance or an exemption from later gates.

Let delta_s = equal-scene PCK8(nearest-a) - PCK8(nearest-mu), using the original
criterion squared pixel distance <=64 and exact rational arithmetic.
Continue to a separately frozen Gaussian-parameter hypothesis only if ALL hold:

1. q24 delta >= +1/200 (at least +0.5 percentage point).
2. q24 nearest-a is strictly better than nearest-mu within EACH of the three
   old TRAIN scenes, not merely in their pooled population.
3. q8 and q16 each have delta >= -1/200.

No epsilon or rounded displays participate. If any condition fails, stop this
anchor-information branch under the current diagnostic; do not switch to a
favorable scene, continuous-error quantile or another center definition. If
the screen passes, it means only that this legal center retained useful old
development information. It does not mean a Gaussian model passed localization,
region, perturbation, efficiency, actual-V, GS or fresh confirmation checks.

Always report all scene/scale correct/valid and paired benefit/harm counts,
exact deltas, all-valid continuous a/mu center-distance mean/median/RMSE,
nearest-index disagreement, and complete correct/incorrect transitions against
the fixed R17 argmax. Predicted front/FOV statuses and inherited center validity
are reported separately. Never trim high errors or use those groups to select
a population or a new parameter. Three reused TRAIN scenes do not supply
independent generalization evidence.

Only a passed screen permits consideration of `mean=a` and
`C_a=C+(mu-a)(mu-a)^T`, the fixed-mean Gaussian second moment. That would be a
new candidate with a new frozen protocol, same-a point/fixed controls, all old
strong controls, all 14 regional conditions, localization, six perturbations
and complete costs. The center information is additional to the old mu/C input
and must be disclosed. Actual learned V, real optimized GS and untouched joint
confirmation remain mandatory; none is authorized by this diagnostic result.
