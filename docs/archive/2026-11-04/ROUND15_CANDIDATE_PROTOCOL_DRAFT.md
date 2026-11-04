# Round15 candidate development protocol — DRAFT, NOT FROZEN

2026-10-30. This is a prospective effects protocol for review. It does **not**
authorize calibration, validation-label evaluation, depth training or fresh
confirmation. The presently authorized work is mathematical/numerical and
label-free cost preflight. Final source, numerical policy, cohort and this
protocol must be frozen before candidate-effect evaluation. R10–R14 decisions
remain unchanged; R12 is failed and R13 OU4 was stopped.

## One fixed change and its scope

The primary is `quartic_full`: the same projected RGB-predicted query pixels,
global mean μ, full covariance C, pooled parent content, actual native key
rectangles and additive density floor as the existing region task, but use

    S = C + h² I
    p(x | μ,S) = exp(-[(x-μ)^T S^-1 (x-μ)]²/(4π)) / (π² sqrt(det S)).

This density has mean μ and covariance S. The exponent is fixed at four;
1/(4π) is fixed by covariance normalization. No exponent search, mixture,
local-frame estimator, center branch, learned confidence or winner projection.
It is an established generalized-Gaussian family used as a new moment closure,
not a novel probability distribution. S=C+h²I is covariance matching, **not**
convolution with Gaussian noise. C=0 produces a quartic point blur, not the old
Gaussian point kernel. Mean μ is a region mean, not the independently projected
source center. No exact query split/merge invariance is claimed.

Integrate over each actual key rectangle before adding parent content:

    probability = softmax(log(M_native_cell + 1e-30*area_reference8) + T*q·k).

The same final probability supplies region metrics, native argmax and any
future V aggregation. Numerical integral, precision, tail policy and derivative
implementation must be accepted by independent preflight and frozen before
this effects protocol is finalized. Finite density floors are retained for all
rows; zero/tiny-FOV rows are reported and never removed.

## Controls and identical search budget

Eight families, each with the same 12 globally shared configurations:
`h ∈ [.25,.5,1,2]`, `T ∈ [0,5,10]`, in this explicit nested order (h outer).

1. quartic_full, fixed primary;
2. quartic_point, same kernel and query μ but C=0;
3. Gaussian point (the existing actual-cell integral with C=0);
4. Gaussian fixed (native query area/(64*12) times I);
5. Gaussian full1 (same μ,C as the primary);
6. Gaussian full4 (four fixed local full-moment strata);
7. Gaussian quad4;
8. Gaussian quad16.

All consume the same complete forward observations and content fields.
Preparation computes only statistics a method actually needs; point must not
pay for unused full covariance. No method receives sensor visibility, source
truth, a different component content bank or extra fitting budget. Existing
frozen B/R11/R12 configurations remain reported as legacy references. Reusing
their previous calculations is allowed only under the exact-contract rules
below; it does not add configurations to the new 12-point grid.

This draft does not add a separately calibrated isotropic quartic contender.
A same-selected-h/T trace-isotropic attribution check may be preregistered
before the final freeze if a claim about anisotropy is intended. It cannot
replace the primary after results are known.

## Old training cohort, fixed objective and center constraints

Use the exact 48 R10 training pairs: 16 each from Harvard c3/c5/c6. Evaluate all
nine g8/g16/mixed query/key conditions and the five q24/key24 scale conditions
from R12. Regional populations and common-grid maps are unchanged. For q24,
use the independently reconstructed old-training sensor centers and provenance
from the frozen R13 target diagnostic; q8/q16 labels must agree with the
original labels. Every declared pair/query remains in its applicable metric
population, with region and center denominators reported separately.

Define a fixed regional calibration objective:

    J = 0.5 * mean_scene(mean_9conditions(common16query→base8key KL))
      + 0.5 * mean_scene(mean_5scaleconditions(direct native-cell KL)).

Within each scene/condition, KL uses total visible source-pixel mass. Families
receive equal weight; do not give the nine-condition family extra weight just
because it has more conditions. This is a declared mixture of two development
risks, not a claim that different target grids have identical difficulty.
Report both terms and every condition separately; J is **selection only** and
never replaces any inherited acceptance gate. No learned task weights,
entropy-dependent normalization or after-the-fact objective adjustment.

First select one localization configuration for each of Gaussian point and
quartic_point by maximizing the equal-scene/equal-stride mean PCK@8 over
q8/q16/q24→g8 on these same 48 pairs; ties use the declared grid order. These
are global configurations, not one per scale. Also retain the original frozen
Gaussian `point_pck` (.5,5) without reselection.

A primary configuration is feasible only if, at **each** of q8/q16/q24→g8,
its equal-scene PCK@8 is no more than .5 percentage points below **each** of:
the new Gaussian localization selection, the new quartic localization
selection, and the original frozen point_pck. Check all three comparisons,
not only the best macro average. The inherited point protection is therefore
never weakened by reselection. Per-scene center values are all reported;
this draft does not invent additional per-scene PCK thresholds.
Construct PCK fractions from integer correct/visible counts. Use exact rational
arithmetic for the equal-scene/equal-stride point selection, grid ties and
the per-scale 1/200 protection boundary; floats are only for reporting.

Among feasible primary configurations choose minimum J, then the declared
grid order. If the feasible set is empty, **stop this unchanged candidate**;
save all results and do not evaluate a newly chosen validation configuration,
change h/T ranges, insert a center decoder or soften the constraints.

For each control, its strong **regional** selection is the unconstrained
minimum J, using the same grid tie rule. Do not discard a strong regional
quad/full1/full4 control merely because it fails center protection. The point
localization selections are separate readouts of the same 12 runs, not extra
search. Save the primary's unconstrained regional optimum descriptively to
show what the center constraint changed; it is not a second validation primary.

Select the strong regional point family between Gaussian point and
quartic_point by their unconstrained minimum training J (tie: Gaussian first),
and the strong quadrature between quad4/quad16 similarly (tie: quad4 first).
The frozen Gaussian point/point_pck and frozen B-selected quadrature are also
kept in validation reporting and inherited protections. There is no
validation-dependent control selection for regional gates.

## Storage and forward-before-supervision contract

Freeze the complete design above **before** executing a cheaper, mandatory
center-feasibility stage. This ordering and its empty-set stopping rule are
declared before any R15 candidate-effect labels are decoded. It changes neither
J nor any acceptance threshold and selects no subset of scenes or queries.

Stage A evaluates only quartic_full, quartic_point and Gaussian point, each at
all 12 configurations, on all 48 training pairs and q8/q16/q24→g8. The original
point_pck (.5,5) is already one of these Gaussian grid configurations. Retain
all final float64 probabilities and canonical hashes: the exact raw array
size is 3×12×48×(432+108+48)×432×8 = 3,511,517,184 bytes (3.27 GiB), before
archive overhead. Compression is optional and never changes probability dtype.
All three scenes' complete forward archives and a global completion manifest
must exist and pass integrity checks **before any Stage A center-target array
is decoded**. Only independently reconstructed sensor point_xy/point_visible
fields are decoded in this stage; regional target counts are not needed.
Numerical acceptance includes the independent synthetic audit and a separately
frozen, label-free old-training-input numerical preflight. Its complete source,
input, output and audit chain must pass before this stage can be frozen.

After all Stage A forwards complete, compute every configuration's PCK with
the same center populations, global point-reference selection and three-scale
feasibility conjunction defined above. Independently audit arrays, populations,
aggregation, selection and all stage hashes before accepting the decision.
An empty feasible set stops this unchanged candidate: do not incur the full
regional calibration, open old-validation candidate labels or scan new h/T.
A nonempty set authorizes **only** the originally declared regional calibration
next. It is not a selected candidate, a regional pass or a scientific promotion.

Stage B then completes all eight families × all 12 configurations × all 14
regional conditions, including infeasible primary configurations for complete
descriptive reporting. Stage A may supply exact matching arrays, but does not
reduce the declared search budget or omit strong controls. Only after Stage B
and its independent audit is the feasible primary selected by the original J.
The streaming contract below applies to Stage B; “before labels” there means
before any **Stage B regional-label evaluation**, since Stage A has already
used its predeclared training-center supervision. The operator remains frozen.

Only roughly 13GB is currently free. Do not delete old frozen evidence or
assume all calibration probabilities fit. Eight families×12 configurations×48
pairs×14 native conditions contain about 3.07 billion probability scalars
(about 24.6GB in float64 before metadata), so the default calibration plan is
streaming, explicitly recorded in the lock.

Before the first Stage B regional calibration-label evaluation, run a forward-only pass of
**all** declared configurations. For each scene/pair/condition/family/h/T save
shape, dtype, canonical array hash, input hashes, numerical diagnostics,
configuration identity and chronology; do not claim these discarded arrays
remain on disk. Pure geometry may be shared across the three T values without
changing their final probability hashes. Save every forward record and a
complete signed-by-hash stage manifest before metric evaluation begins.

Metric evaluation deterministically regenerates each probability from the
same frozen forward-only function and same allowlisted observations. Verify
its canonical array hash against the completed first pass **before** computing
that record's label metrics. The function never receives labels. Preserve all
per-pair sufficient sums, denominators, configuration metrics, selections and
input/output hashes, not merely winning averages. An independent replay must
reconstruct a declared sample of raw arrays plus all metric aggregation and
selection, with full discrepancy disclosure.

Exact replay is a precondition, not an assumption: prior to effects, verify
bitwise repeats on actual allowed training inputs under the frozen dtype,
device/chunk order and deterministic reduction settings. If the implementation
cannot reproduce hashes, pause and settle a reviewed storage/determinism plan
before any effects; do not silently round hashes or switch to approximate
matching after labels have been opened. There is no permission to change the
probability operator because of disk limits.

Before first validation evaluation, freeze the single primary selection and
all control/point-localization selections plus every calibration-stage hash.
For the 32 old validation pairs, preferentially retain **all native primary
probabilities**, including every scale/perturbation condition, with the exact
dtype used for metrics. A complete selected control array may be referenced
from an existing frozen result only when pair, condition, method/configuration,
input/content/geometry, numeric implementation, dtype, floor, probability hash
and label-population identity agree exactly. Configuration names alone are
insufficient. Changed configurations or contracts require recalculation.

Precompute and publish the exact remaining disk requirement before validation.
If controls still need streaming, state that before freeze and apply the same
forward-hash/deterministic-metric/independent-replay contract; all selected
primary arrays stay preserved. These are storage implementation choices,
**not** relaxed scientific or audit gates. No silent overwrites or deletion of
failed attempts. The independent audit receives source snapshots and all
necessary input identities, rather than only a favorable report.

## Old-validation regression gates: unchanged historical minima

Use the same 32 old validation pairs (Harvard c8/Brown bm4, 16 each). They have
been examined repeatedly; call this development regression, never independent
confirmation. No training-configuration selection from these new results.
The primary must pass the conjunction below; each legacy comparison is
recorded separately so a newly weaker baseline cannot erase a prior requirement.

**Nine original partitions:** common-grid macro KL at least2% below the
selected strong point, with both scenes improving, and the original frozen
Gaussian point comparison also passing. Against both the newly selected
strong quadrature and the original frozen B strong quadrature, require either
at least2% lower KL with incremental PE time<=1.25x, or at most1% worse KL with
incremental PE time<=0.5x. Every native partition condition's common KL is no
more than2% worse than each quadrature comparator. g8→g8 PCK loss is at most
.5pp against every locked localization reference, including legacy point_pck.
Also report full1/full4 and fixed with accuracy, center behavior and cost;
retaining the quartic family requires a concrete advantage rather than hiding
a simpler competing closure. No claim of anisotropy benefit from an absent
matched diagnostic.

**Five scale conditions:** equal-scene/equal-condition direct KL at least2%
below the selected strong point and original frozen point; both scenes improve.
Each condition is at most2% worse than both selected and frozen strong
quadrature. q24→g8 PCK loss at most .5pp against all locked localization
references. Retain full1/fixed/quad4/full4 results. If q16 center protection is
also to be a formal validation gate, it must be fixed explicitly before final
freeze; training feasibility at q16 alone does not mean that gate was evaluated.

**Six controlled stresses plus zero:** the unchanged R12 depth×.9/1.1,
target Ry±1° and target tx±.01m states; no combinations or retuning. Every
state retains all nine partitions and the same original labels/populations.
Common KL is at most2% worse than both quadrature references evaluated under
the same perturbation; g8→g8 PCK loss at most .5pp versus all point-localization
references under that perturbation. All probabilities finite/normalized.
Report absolute changes from zero as well as relative protections. Passing
these controlled biases does not prove robustness to realistic error models.

**Cost:** freshly paired timing on the first metadata pair of all five old
development scenes, nine partitions, 3 warmups and15 synchronized repeats.
Time the actual support preparation, key geometry, rectangle integral,
content path that actually executes, softmax and common aggregation, including
any quartic numerical key subdivision. Do not substitute point logits, a cheaper
integral, cached geometry or training throughput. All controls use their
actual selected configurations and required preparation. Exclude only shared
frozen visual/depth prediction and dense projection/transfer as previously;
record common CPU projection and overall data production separately. Report
dtype, full pipeline definition, tensor bytes and allocator peak separately.
This is incremental PE cost, not complete vision-system end-to-end latency.

**Audit:** require independent numerical/gradient, input-order/population,
source/freeze, every saved probability/metric, full selection and gate replay.
Every failed condition remains a failure. A successful final average cannot
hide an omitted condition or missing comparison.

## Promotion still requires actual values and actual GS/RGB evidence

A regional pass does not complete the original research objective or erase
R7/R9 failures. First write a new candidate-specific actual-V protocol using
the original 576train/128oldval samples, paired seeds, optimizer/head/loss and
singleview controls. Preserve the ROUND12_DEPTH_PROTOCOL scientific minima:
at least2% AbsRel benefit over strong position; strict benefit over same-capacity
singleview; both scenes and at least4/5 seeds improve; strongest-compression
accuracy/cost alternative; RMSE deterioration<=2%; actual V interventions,
gradient-path evidence and complete PE+V/head timing. Include the new quartic
point control. T0 makes Q/K content inactive and must be disclosed; do not
quietly reuse the old T5 full4 geometry or claim equal registered parameter
counts imply equal effective capacity. That new protocol must freeze before
training. R12's failed stress promotion remains blocked in its old namespace.

To claim useful actual optimized-GS/RGB/NVS behavior, a separate physically
defined adapter and experiment remain required. Preserve R9's context-only
forward information, same RGB capacity, strong point/fixed/isotropic/bilinear
envelope, at least2% full-pixel MSE improvement in both scenes and comparison
to the same-geometry original GS system. A worse result than original GS is at
most a mechanism finding. A patch moment experiment does not prove internal
optimized primitive R/S utility, and a quadratic/fourteen-channel point identity
does not itself validate RGB rendering or novel-view synthesis.

Only after all applicable development protections and actual-value/GS checks
have credible evidence may a new joint multi-scene confirmation be designed.
Use untouched scenes, frozen global configurations, predeclared paired-scene
uncertainty and the inherited R6/R7/R10–R12 criteria; do not reopen historical
sealed tests to tune this candidate. The draft inventory supplies planning
context, not an already passed confirmation. No current stage authorizes fresh
RGB/depth acquisition or claims the general research goal is achieved.
