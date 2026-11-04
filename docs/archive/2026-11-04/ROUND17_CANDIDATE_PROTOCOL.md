# R17 candidate: Gaussian support with trace-conditioned content temperature

Status: design declared before new R17 effects; implementation and stage
locks are still required. This document does not itself authorize VAL, actual
V training, GS experiments or fresh data. R15's fixed quartic candidate stays
stopped and all historical results remain unchanged. R16 was exploratory
diagnosis, not scientific acceptance.

## One fixed change

The sole candidate, `trace_full`, keeps the R15 Gaussian full1 mean, full
query-support covariance C, native key cells, Gaussian GL32 integral,
reference-8 units, content pooling and key-area floor. Its score is

    M_qk = integral over native key cell k of N(x; mu_q, C_q + h^2 I)
    tau_q = T h^2 / (h^2 + trace(C_q))
    score_qk = log(M_qk + 1e-30 * key_area_px/64) + tau_q * content_qk
    P_qk = softmax_k(score_qk)

C is spatial support, including projected pixel footprints and within-region
scatter. It is not estimated epistemic/measurement uncertainty or a confidence
score. The denominator uses the full two-dimensional trace, with coefficient
one; there is no trace/2 alternative, eigenvalue rule, learned coefficient,
shape sweep, anchor replacement, new feature, extra head or new module family.
The h/T grid remains h=(.25,.5,1,2), T=(0,5,10), h-major then T-minor order.

Evaluate tau using the existing float32 C promoted to float64 with the common
positive scale s=max(h²,C00,C11): r=h²/s, d=(C00/s+C11/s), tau=T*r/(r+d).
The two covariance terms are summed before adding r, as specified in the
[numerical preflight contract](ROUND17_PREFLIGHT_PROTOCOL.md). This algebraic
rescaling prevents overflow; it is not clipping or a new parameter. Cast tau
back to float32. Multiply pooled float32 content by tau in float32, promote
the product to float64, add the existing float64 geometry, then softmax.
The new module receives old geometry log mass and does not integrate or add
the floor again. Original native24 Gaussian cells retain the deterministic
four-12px subdivision; sum child masses before the one native area floor.
All finite/symmetry/PSD and input validation that actually execute are part
of the operator and its cost. Their exact tolerances belong to the jointly
frozen source and independent numerical contract, not a hidden effect option.

C=0 must reproduce the old Gaussian point content path; T=0 must reproduce
unmodified full1. There is no promised recovery of the sensor center, exact
query split/merge invariance, low-dimensional exact cell-integral Q/K map,
or general 3DGS/NVS benefit. This changes the coupling of content to a
Gaussian positional support; it does not invent a new probability family.
Related work and limits on novelty claims are recorded in the
[focused literature scope](docs/round17_literature_scope.md).

## Mandatory numerical stage before effects

Follow [ROUND17_PREFLIGHT_PROTOCOL](ROUND17_PREFLIGHT_PROTOCOL.md). Jointly
freeze production model, independent references, runner, tests, both R17
design documents, complete transitive dependencies and upstream evidence
before measured preflight outputs. Synthetic development tests may precede
this lock. Any repair creates a new namespace, keeps the failure, and repeats
the unchanged numerical population; it cannot silently loosen a tolerance.

The frozen synthetic suite covers tau arithmetic and derivatives, point/T0
equivalence, finite/PSD boundaries, rotation and joint coordinate-unit scaling.
The real numerical preflight uses only four forward fields from the first
lexicographic pair of each old TRAIN scene c3/c5/c6. All 14 existing conditions,
12 configurations and eight fixed evenly spaced queries give 504 probability
matrices / 4,032 probability rows and 168 geometry matrices / 1,344 geometry
rows. No label, sensor, VAL or fresh-data decoding is allowed.

Use separately reconstructed NumPy moments/content and independent GL64/96
Gaussian cell integrals, plus the hybrid independent integral using production
support/content. Require normalized probability errors <=1e-4, hybrid errors
<=1e-4, GL64/96 convergence <=1e-8, relative-normalized mean/cov errors <=1e-4,
content error <=1e-5, and finite/nonnegative normalization error <=1e-12.
Retain all zero/tiny-FOV cases and argmax disagreements. Numerical argmax
agreement is not PCK. Both synthetic and real preflight must pass and their
source/output chains must be checked before Stage A can start.

## Stage A: complete center feasibility, then stop or continue

Use exactly the 48 R15 Stage A TRAIN pairs and independent center labels:
16 each from harvard_c3/hv_c3_1, harvard_c5/hv_c5_1 and harvard_c6/hv_c6_1.
New forward covers only trace_full, all 12 configurations, q8/q16/q24→g8:
1,728 probability matrices and 338,688 query/configuration states. Store all
float64 native probabilities (1,170,505,728 uncompressed bytes). No query,
pair, h/T or scale may be removed after observing a result.

Jointly freeze the Stage A producer and complete independent audit before
its first forward. Complete/hash all 48 new forwards before this stage's
center evaluation. These are previously used labels and baselines; this
ordering does not pretend to restore statistical blinding or confirmation.
The four T=0 candidate configurations, every pair and all three query scales,
must match the corresponding frozen R15 Stage B `gaussian_full1` canonical
dtype/shape/bytes hashes: 576 exact array comparisons. A numerical tolerance
cannot substitute for this identity check. The independently referenced
numerical checks remain required even when these hashes match.

Reuse the audited R15 Stage A complete 12-grid Gaussian-point and
quartic-point probabilities/counts and their exact global PCK selections.
Revalidate their whole source/completion/audit chain and pair populations;
do not rerun them merely to change a reference. Keep all three protections:

- selected Gaussian point (old global equal-scene/equal-scale PCK maximum);
- selected quartic point under the same rule;
- legacy Gaussian point h=.5,T=5.

The two selected localizers are currently h=.25,T=5; the audited source of
truth, not this prose, binds their identities. For each candidate configuration,
require PCK@8 at q8, q16 and q24 separately to be no more than 0.5 percentage
points below each reference. Use native argmax and stable first-index ties,
original valid-center masks, integer correct/valid counts, exact Fraction
scene averages, and a tolerance of exactly 1/200 without epsilon slack.
All nine comparisons must pass. Do not replace argmax with the projected mean
or a nearest-center readout. Exact selection ties keep declared grid order.

Stage A outputs only the complete feasible index set and all failed checks.
If it is empty, stop this fixed candidate after independent audit. It selects
no regional winner. A nonempty set only allows the complete Stage B below.

## Stage B: complete regional calibration and fixed primary

Use the same 48 TRAIN pairs, all 12 trace_full configurations (including
ineligible ones), and the full 14 conditions: original 3×3 g8/g16/mixed nine,
then g8→g24, g24→g8, g16→g24, g24→g16, g24→g24. This is 8,064 new metric rows.
Nine conditions use area-only aggregation to fixed common16-query/base8-key
targets; five scale conditions use their native hard-count targets. No GT
information may enter prediction or the common aggregation.

Within each scene/condition use visible source-pixel-mass weighted KL; then
average conditions and scenes equally. Define exactly the existing objective:

    J = .5 * common9_macro_KL + .5 * direct5_macro_KL.

Select the minimum J among Stage A feasible indices, using declared grid
order for exact ties. Report the unconstrained minimum descriptively with
its center failures, but never promote it or select from VAL. No alternative
candidate may replace the selected primary to bypass the stop screen below.

Reuse the independently audited R15 complete eight-family × 12-configuration
regional calibration, baseline selections and all per-pair/condition metrics.
Bind old source, input/label identities, forward commitments, deterministic
replay, completion and independent verification. Gaussian point/fixed/full1/
full4/quad4/quad16 are required Gaussian controls; historical quartic results
remain externally linked, not rerun as a new candidate search. Strong regional
point remains the lower-J of the two old independently calibrated point
families (Gaussian currently wins). Strong quad remains lower-J quad4/16.
The unchanged legacy point h=1,T=5 and quad16 h=.5,T=5 remain separate named
comparisons even if an alias is exactly equivalent.

Full new raw storage would require 3,074,609,664 bytes. If storage requires
the existing two-pass contract, freeze that implementation before the first
forward: every one of the 8,064 matrices receives a canonical hash; all first
passes finish before regional-label evaluation; deterministic second-pass
probabilities must match every hash before metrics. Retain all 14×12 full
query matrices from the three predetermined first pairs (192,163,104 raw
bytes), every metric/commitment and full independent selection/population
audit. Do not claim all probabilities were stored or independently numerically
reintegrated. Every reuse must have the same operator, input, numerical path,
dtype, floor, population and exact array identity; same h/T alone is insufficient.

## New R17 TRAIN futility screen, declared before R17 results

This is a new development resource rule, not a retroactive R15 rule and not
a scientific or VAL pass. Apply it to the already fixed Stage B primary after
all configurations/conditions are complete and independently audited:

| Necessary TRAIN macro evidence | Required against each reference |
|---|---|
| common9 KL vs strong regional point and legacy point | >=2% improvement |
| direct5 KL vs the same two point references | >=2% improvement |
| common9 KL vs strong quad and legacy quad16 | <=1% deterioration |
| direct5 KL vs the same two quad references | <=2% deterioration |

All eight inequalities must hold. Failure stops this selected fixed candidate
before old VAL; it does not prove that no design or other configuration could
work. Do not choose another eligible index after this screen fails. Report
every scene and condition; no extra TRAIN all-three-scenes-positive or
per-condition hard rule is introduced here. No timing claim is inferred.
The screen is a necessary prospect filter; passing it does not establish the
stricter held-fixed development regressions or cost requirements below.

## Old VAL regression remains a separate complete contract

Only after accepted numerical, Stage A and Stage B results, the fixed primary
and the TRAIN screen pass may a separate VAL protocol/source/audit be frozen.
Use the original 32 old VAL pairs (c8/Brown bm4, 16 each); repeated use makes
this development regression, not independent confirmation. Keep all legacy
comparisons, the two selected point localizers and legacy point_pck. Keep the
original zero nine conditions, five scale conditions and six independent
R12 stresses: depth×.9/1.1, target Ry±1 degree and tx±.01m. No combinations,
retuning, population changes or new effect-dependent equivalence classes.

| Check | Complete inherited requirements |
|---|---|
| zero nine-condition common KL | >=2% improvement vs both new strong regional point and legacy point; both VAL scenes improve for each |
| zero accuracy/cost | vs both strong quad and legacy quad16: >=2% lower KL with incremental time<=1.25×, or <=1% worse KL with time<=.5× |
| zero per-condition protection | each of nine common KL conditions <=2% worse than each quad reference |
| zero q8→g8 and q16→g8 PCK | <=.5pp loss against each of three point-localization references; q16 is the predeclared R15 extension, not an original R12 check |
| five-scale direct KL | >=2% improvement vs both strong/legacy point, both VAL scenes improve; every condition <=2% worse than each quad |
| q24→g8 PCK | <=.5pp loss against each of the three point-localization references |
| each of six stresses plus zero | nine-condition macro common KL <=2% worse than each equally perturbed quad; g8→g8 PCK loss <=.5pp vs each equally perturbed localization reference |
| numerical population | all native probabilities finite/normalized, all original targets and query masses retained, complete independent replay |

Stress per-condition values are reported, but the inherited hard KL rule is
the per-state nine-condition macro, not an invented per-condition threshold.
PCK boundaries use exact integer/Fraction arithmetic throughout.

Required attribution controls at the primary h/T are unmodified Gaussian
full1 (constant T) and trace-conditioned isotropic geometry, plus the old
separately calibrated full1 and all other locked controls. The isotropic
geometry uses C_iso=trace(C)/2 I and exactly the primary's tau values computed
from the original full C, avoiding a float32 trace-rounding change. It has no
independent calibration, does not replace primary, and is not a shape sweep.
If T=0 is selected, report that the new coupling is inactive and full1 is
identical; do not attribute an improvement to tau without a matched effect.

Fresh paired CPU timing uses each of five old scenes' first metadata pair,
nine conditions, 3 warmups and 15 measured repeats, four CPU threads and a
frozen rotating role order. Include each role's own preparation, content
pooling/product, native key geometry, Gaussian integral, one floor, new
trace/PSD/symmetry checks, tau arithmetic, softmax and common aggregation.
No cached covariance, tau, logits, free primary statistics, pure-integral
timing or training throughput may substitute. Alias only identical complete
operators/configurations; keep semantic comparisons. Report common prediction/
projection and transfers separately, tensor bytes and actual memory measure;
the gate measures incremental PE, not full vision-system end-to-end latency.

## General effectiveness still requires downstream and unseen evidence

Do not relax the candidate-specific actual-V and optimized-GS requirements
listed in [the inherited inventory](docs/round11_validation_inventory.md)
and [R12 depth contract](ROUND12_DEPTH_PROTOCOL.md). The inventory's original
superseded depth draft is not the current rule. Actual V uses the original
576train/128oldval task, 41,345-parameter equal-capacity heads, paired five
seeds/1,000 steps and fixed loss/selection: >=2% AbsRel benefit over strong
position, strict benefit over singleview, both scenes and >=4/5 seeds improve;
strongest compression accuracy/cost alternative; RMSE worsening<=2%; real
V interventions, gradients and full incremental PE+V/head time. The new tau
must actually multiply learned content there; a frozen correspondence matrix
cannot stand in for learned attention. T0 effective Q/K inactivity is disclosed.

R7/R8 value limitations and R9 actual GS/RGB failure remain. Real optimized
GS parameter access and a physically defined RGB/NVS task require separately
frozen strong point/fixed/isotropic/bilinear and original-GS comparisons,
context-only inputs and the original error/scene conditions. Spatial patch
moments are not evidence of optimized-primitive R/S utility. Only after the
applicable development/value/GS evidence is credible may untouched-place
confirmation be locked, retaining paired-scene intervals, center and known-size
cross-scale protections, noise/cost checks and all disclosed failure cases.
Regional success alone cannot complete the general research objective.
