# R20: one fixed source-centered Gaussian, old-TRAIN falsification

2026-10-31. Design declared after accepted R19 results, before any R20
prediction. R19 passed a newly declared information screen; it did not pass
a Gaussian efficacy gate. R17's scientific failure and R18's numerical failure
remain unchanged. This document selects one small fixed operating point for
falsification. Joint implementation/source locks are required before execution.

## Hypothesis and exact operator

Preserve the designated source-center projection when compressing a patch to
one Gaussian. Use the existing R13 `predicted_centers` law, with exactly the
same predicted depth, cameras, interpolation, sanitization and clamps accepted
in R19. No center is selected from a point winner, label or successful subgroup.
For native query bounds, let a be that float64 pixel projection divided by 8.
Let mu,C be the original R17 full1 float32 support moments in reference-8 units.
Promote the original moments to float64 and form

    d = mu.double() - a
    C_a = C.double() + d d^T
    S = C_a + h^2 I
    tau = T h^2 / (h^2 + trace(C_a))
    P_k = softmax_k(log[integral_Bk N(x; a,S) dx + 1e-30 area_k/64]
                    + tau32 * content32)

The **only candidate is anchor_trace at h=.25,T=10**. Keep a and C_a in
float64; do not round them to float32. Reuse the frozen R17 scaled float64
temperature formula, cast tau to float32, multiply content in float32, promote
that product to float64, and use float64 softmax. Gaussian geometry remains
GL32 and native 24px keys retain four 12px integration children, summed before
the single original native-area floor. Pooled features and native partitions
are unchanged. No extra learned parameter, gate, head, mixture or h/T search.

C_a is the old support's second moment about a, not its centered covariance
and not a measurement-confidence estimate. It is the fixed-mean Gaussian
cross-entropy optimum for a distribution with the old first two moments; that
statement neither guarantees PCK nor preservation of regional KL. Mean, spread
and effective content temperature all change relative to R17. The new center
is additional legal information beyond mu,C. For an actual Gaussian whose
designated center already equals its mean, this construction reduces to the
unchanged moments; no new optimized-GS benefit follows from the patch result.

## Fixed controls and complete population

Use exactly the accepted 48 old TRAIN pairs (16 per c3/c5/c6), all 14 original
conditions: the g8/g16/mixed 3x3 common nine and g8->g24, g24->g8, g16->g24,
g24->g16, g24->g24. Counts, native bounds, common aggregation and original
pixel-mass weights remain unchanged. All forward rows, including outside-FOV,
behind-clamp and zero/tiny Gaussian mass, are retained.

The twelve new complete operators are:

| Role/operator | Geometry | h,T | Content law |
|---|---|---|---|
| primary anchor_trace | a,C_a | .25,10 | tau(C_a) |
| mean_only_matched_tau | a,C | .25,10 | exactly primary tau(C_a) |
| anchor_constant | a,C_a | .25,10 | constant T |
| anchor_isotropic_matched | a,tr(C_a)/2 I | .25,10 | exactly primary tau(C_a) |
| same-a Gaussian regional point | a,0 | 1,5 | constant T |
| same-a Gaussian localizer | a,0 | .25,5 | constant T |
| same-a legacy point localizer | a,0 | .5,5 | constant T |
| same-a matched Gaussian point | a,0 | .25,10 | constant T |
| same-a regional/legacy fixed | a,original native size covariance | .5,5 | constant T |
| same-a matched fixed | a,original native size covariance | .25,10 | constant T |
| same-a quartic regional point | a,0 | 1,5 | original quartic constant T |
| same-a quartic localizer | a,0 | .25,5 | original quartic constant T |

Fixed covariance is precisely the old native query area/(64*12) times I.
There was no separately selected historical fixed-PCK model: do not invent it.
Quartic point retains the frozen GL24 tiled fourth-order kernel; it is a
control, not a new candidate. The same-a control settings are inherited from
the old calibration or explicitly matched; they are not newly optimized for a.
Keep separate regional/legacy semantic aliases even when the complete
operator is identical. Attribution roles cannot be promoted after effects.
The mean-only role holds tau(C_a) fixed to isolate rank-one geometric widening;
it is not the complete a,C model with tau(C).

Inherit **all** accepted old regional/legacy Gaussian and quartic point, fixed,
full1, full4, quad4, quad16 controls, the original two selected point localizers
and legacy point localizer, and the fixed R17 candidate. Bind their original
selection, source, completion, forward, label and independent-audit identities.
Use their complete accepted regional rows and the original Stage-A exact
center counts; report this as inherited evidence, not new baseline prediction
or new numerical reintegration. Do not change an old reference or select one
per scene/scale. Matched old R17 attribution evidence can be linked separately.

## Source freeze, forward-only numerical audit and chronological guards

Jointly freeze this protocol, design/preparation notes, new model, producer,
independent auditor, synthetic tests, complete transitive executed local
dependencies, source snapshots, runtime/build identity and all inherited
evidence before real forward. Bind R19 acceptance and its function/input chain.
Use the established CPU-four-thread NumPy-first/Torch runtime; three thread
environment variables, Torch intra/inter-op and actual build identity must be
recorded and checked. No runtime guard bypass or system-library changes.

Synthetic software tests must cover fixed-mean second moments, C_a PSD and
input rejection, a=mu reduction, C_a=0 point reduction (C=0 and a=mu), T=0,
matched-tau equality,
coordinate-unit/translation behavior, original finite-FOV floor/native-child
rules, exact screen boundaries and preservation of all rows. C=0 with a!=mu
instead gives C_a=dd^T and must not be mislabeled a point limit. Tests may
precede freeze and must use no real effects.

Expected old label hashes may be read from accepted metadata at freeze.
Actual regional-label and R13 center-target files must not be hashed or decoded
until all 48 pairs' new forward probabilities are globally sealed and the
independent forward/numerical audit passes. Repeated old labels are development
data; this ordering does not make them fresh or statistically blinded.

Save every new float64 native probability and argmax, identities and canonical
hashes: 48x14x12 = **8,064 matrices**, **3,074,609,664 raw probability bytes**,
plus supports/records. Check free space before starting and preserve all old
evidence. No subset retention or second-pass-storage switch is authorized by
this version. Stages refuse overwrite and preserve any failed attempt.

The independent auditor must reconstruct a with its own float64 linear solve
and bilinear computation for all 39,888 unique pair/query rows (g8/g16/mixed/g24),
with <=1e-9 pixel discrepancy and exact flags. Reused g8/g16/g24 centers must
also match accepted R19 arrays exactly; mixed is newly covered. Bind/recheck
original support and content provenance. Shared-source full replay must match
all 8,064 saved probability/argmax hashes; it is reproducibility, not independent
forward mathematics.

Independent numerical mathematics covers each scene's first metadata pair,
every 14 condition, every 12 operator, eight fixed evenly spaced query rows
and all native keys: **504 matrices / 4,032 probability rows**. Reconstruct
original NumPy moments/content and the new a,C_a separately, using the inherited
R17 normalized moment/content error conventions (<=1e-4 for moments, <=1e-5
for content). Use independent Gaussian GL64/96, including a hybrid integral
from executed production statistics; retain the old quartic independent
reference/convergence contract for the two quartic-point controls. Require
production/reference and hybrid normalized probability errors <=1e-4,
reference convergence <=1e-8, same-raw-mass arithmetic error <=1e-12, exact
prepared tau32/bias32 under the declared arithmetic, and finite/nonnegative
normalization error <=1e-12. Preserve argmax disagreements and all tails.
No increasing integration order, role deletion or tolerance change within a
failed run. Actual reference functions/orders and complete source hashes must
be explicit in the joint lock before the first new forward.

Only a passed complete forward audit permits actual label access. Use exactly
the inherited regional count targets and the R13 g8/g16/g24 independent center
targets, not a merely allclose export. Center-valid totals must remain
9,521/2,463/1,096. Area and center validity stay separate and never depend on
predicted FOV/front, the operator or previous success. Final independent audit
reconstructs all metric sums, integer counts, exact scene Fractions and every
decision from the saved new arrays. Bind producer/audit completions before
root interpretation. KL/JS metric-sum tolerance 1e-8 and aggregate tolerance
1e-10 are comparison tolerances only, never efficacy slack. Counts, hashes,
Fraction PCK and Boolean decisions are exact.

## Fixed development screen before any further investment

This adds a newly declared conservative TRAIN resource screen to the inherited
rules. It is not a retroactive R17 gate or a scientific/VAL pass. Compute
visible-pixel-mass-weighted KL within each scene/condition, then equal-weight
conditions and scenes. Report common9, direct5 and J=.5*(common9+direct5), but
select no parameters. PCK uses native probability argmax with first-index ties,
squared target distance <=64 pixels, original masks and exact scene Fractions.

ALL of the following must hold for the one fixed primary:

1. Keep the eight R17 TRAIN macro rules: common9/direct5 KL improve >=2% versus
   each old strong/legacy regional point; common9 is <=1% worse and direct5
   <=2% worse versus each old strong/legacy quad reference.
2. Keep the nine original center protections: q8/q16/q24 PCK loss <=1/200
   against each original Gaussian/quartic selected localizer and legacy point
   localizer. Exact Fraction arithmetic has no epsilon.
3. Each of all 14 regional conditions has macro KL <=2% worse than each old
   strong/legacy quad reference. This is an additional TRAIN screen now;
   the same kind of protection remains mandatory in old VAL.
4. Common9 and direct5 KL each improve >=2% versus **each** same-a regional
   Gaussian point, quartic point and fixed control. A better center by itself
   is insufficient evidence for retaining full covariance.
5. q8/q16/q24 PCK loss is <=1/200 against each same-a Gaussian/quartic selected
   localizer and same-a legacy point localizer.

Report all scene/condition results, JS, all role PCK/continuous diagnostics,
matched attribution and raw-mass failures, not only these screen inequalities.
Center diagnostics cover every available key condition for the inherited
g8/g16/g24 center-query populations. Mixed queries have regional metrics but
no inherited independent center targets; no new mixed center label is created.
Do not add an unannounced all-scenes-positive TRAIN rule or replace the
equal-scene macro by pooled rows. No speed/efficiency pass is inferred here.

If any condition fails, stop this fixed candidate and preserve every failure;
do not turn immediately to a 12-grid, promote an attribution role, alter a,
or substitute a favorable scale. Further research requires a distinct justified
hypothesis and fresh protocol, not rescue tuning of this fixed experiment.
If all pass and audits are accepted, only then consider separately frozen
equal-budget same-information control calibration and old development
regressions. A broader candidate grid is neither required nor authorized by
this result. Subsequent protocols must preserve the fixed-stage result and
cannot claim these old settings were already best for the new center.

## Remaining obligations

All old-VAL zero/scale, six perturbation, accuracy/cost, complete population
and numerical rules in ROUND17_CANDIDATE_PROTOCOL.md remain mandatory, with
new same-information strong controls added rather than replacing old ones.
Actual learned Q/K/V and equal-capacity head tests, physically defined optimized
GS/RGB comparisons and untouched joint confirmation also remain mandatory.
The original 41,345-parameter, paired-five-seed depth contract and real-GS
controls cannot be waived by a geometric TRAIN result. R8's weak utility even
under its oracle-routing diagnostic and R9's failed RGB improvement are still
relevant limitations, not proven theoretical ceilings. No V training, GS run,
old VAL access or fresh-data access is part of this R20 experiment.
