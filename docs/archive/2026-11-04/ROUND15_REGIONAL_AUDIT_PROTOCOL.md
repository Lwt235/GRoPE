# Round15 Stage B: independent regional audit

2026-10-30. This audit is declared after Stage A's independently verified
feasible indices `[6, 9]`, but before any Stage B forward or regional metric.
It does not change `ROUND15_CANDIDATE_PROTOCOL.md`, its scientific gates,
training cohort, configuration grid, or regional selection objective.

## Evidence and scope

The producer commits all 48 old-training pairs, eight families, twelve global
configurations, and fourteen native conditions before regional evaluation:
64,512 probability hashes. It retains all probabilities for the first
metadata-ordered pair of each of the three training scenes. Other arrays are
not represented as retained. Every evaluation array must match its complete
first-pass hash before that record's labels can affect a metric.

The independent audit will verify the frozen source and input chain, all
forward/evaluation identities and hashes, exact populations, stage chronology,
all per-pair masses, all 96 regional configuration summaries and selections.
It will independently reconstruct native target counts from the same frozen
dense sensor projections, rather than accepting a producer's count cache.
All 4,032 retained first-pair probability matrices will have their native and
common-grid metrics recomputed with a separate NumPy implementation.

This is complete aggregation and provenance coverage plus a declared raw-array
sample; it is not a claim that all 64,512 arrays were independently retained
or independently numerically integrated. Source/selection hashes and explicit
coverage counts must accompany the result.

## Independent metric definition

Use the existing half-pixel native cells, all visible source pixels, and the
original nearest-pixel boundary convention. Native target counts are literal
query/key ownership counts. The nine original partition conditions lift key
mass uniformly to base8 atoms and aggregate source area to common16 queries.
The five q24/key24 conditions use native-cell counts and probabilities.
There is no common16 parent for q24.

KL/JS use the inherited `1e-15` metric logarithm floor. The forward density
floor remains the separate `1e-30 * key_area_reference8` rule. Never substitute
one for the other. Metric numerators use visible source-pixel mass; each scene
and condition then receives equal weight. Compare independent metric sums to
saved values with absolute tolerance `1e-8` and configuration summaries with
absolute tolerance `1e-10`, without using those tolerances to alter selection.

The fixed selection objective is

`J = .5 * equal_scene_mean(equal_9condition_mean(common_KL))
   + .5 * equal_scene_mean(equal_5condition_mean(native_KL))`.

Use the declared h-outer/T-inner grid order for exact ties. Primary selection
minimizes J only over the independently accepted Stage A indices. Record its
unconstrained optimum descriptively. All six Gaussian controls and the
quartic point control select their unconstrained minimum J, regardless of
localization performance. The stronger point family is selected by training
J, with Gaussian first on a tie; quad4 precedes quad16 on a tie. Neither J nor
the Stage A result is a validation acceptance gate.

## Independent probability sample

For each retained first pair, use eight linearly spaced native query indices
for each of g8, g16, mixed and g24. These indices depend only on partition
sizes. Check all fourteen query/key conditions:

- Both quartic families at every one of the twelve h/T configurations.
- The six Gaussian families at h=.5, T=5, plus each family's final Stage B
  selected configuration if different. This deterministic selected-model
  coverage supplements, rather than replaces, the fixed sample.

Support means, covariances, component weights and pooled content come from
literal NumPy pixel slices, independent of the producer's Torch preparation.
Quartic reference integration uses the frozen independent Cartesian
Cholesky implementation from the real-input numerical preflight, at GL64
and GL96 over the actual native rectangles. Require normalized GL64-to-GL96
convergence at most `1e-8` and production-to-GL96 probability error at most
`1e-4`. Gaussian controls use the previously frozen independent R10/R11
Gaussian rectangle integration and full-stratum references, not production
integration helpers. Their final probability tolerance remains `1e-4`.
All sampled queries, including zero/tiny-FOV cases, remain in the check.

If the fixed numerical reference does not converge or any check fails, retain
the failure and stop acceptance. Any numerical/reference repair needs a new
source record and an explicit explanation; no input or h/T subset may be
dropped to obtain a pass.

## Freeze and limits

The independent implementation, pure metric/selection module, tests and this
protocol must be frozen before Stage B forward execution. Absolute paths and
repository-relative paths must be resolved explicitly against the repository,
with a test for both, avoiding the Stage A audit path error. A later repair
must preserve original evidence and record its actual timestamp.

Passing authorizes only the next predeclared old-validation work. It does not
establish a scientific success, fresh-scene generalization, robustness,
actual-V depth utility, or optimized-GS/RGB utility. Those inherited checks
remain outstanding.
