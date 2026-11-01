# R17 native-size temperature attribution — unexecuted draft

Status: a separate, descriptive TRAIN attribution study. This document is not an
execution authorization. The new producer, model, independent auditor, tests,
protocol and their inherited source closure must be jointly frozen before any
new forward computation. A separate audit lock must bind that producer lock
before forward starts. The running, frozen R17 old-VAL experiment is unchanged.
No current VAL probabilities, labels or effect summaries may be opened for this
study's design. CPU tests require the parent's confirmed no-timing-overlap
window. Any permitted synthetic test uses one CPU thread and at most 60 seconds;
once the benchmark starts, all such tests pause until it ends.

## Question and fixed operators

Does observed projected support trace contribute beyond the known native query
rectangle size, conditional on the accepted TRAIN primary configuration? The
only two roles are `trace_full` and `size_temperature`, both at **h=0.25, T=10**
(accepted R17 regional grid index 2). There is no sweep, winner selection,
new scientific gate, equivalence test, or permission to replace the primary.

Both roles use the exact original observed mean, full covariance C, features,
Gaussian native-cell integration, area floor and normalization. In particular,
the geometry remains N(mu, C+h²I). The original role uses
`tau_observed = T*h²/(h²+tr(C))`. The sole intervention is

```
trace_native = (width² + height²) / (64*12)
tau_size = T*h² / (h² + trace_native).
```

The integer half-open native bounds determine width and height, independently
of observations or labels. The continuous-pixel footprint covariance is
diag(width²,height²)/(64*12), not the variance of pixel centers alone. Compute
the native trace and ratio in float64, round tau once to float32, multiply the
original float32 content by float32 tau, then cast that product to float64 and
add the unchanged float64 geometry logits. The original observed-tau role keeps
the frozen R17 scaled-ratio evaluation order and its validation checks.

Mixed partitions include 16×8 and 8×16 rectangles. Their trace is 5/12 and tau
is 30/23; replacing width²+height² by 2*area would give trace 1/3 and tau 30/19,
a different intervention. The historical area-based fixed Gaussian baseline
keeps its original definition. This study neither repairs nor renames it.

Nominal square taus are 30/11 (g8), 6/7 (g16), and 0.4 (g24). Their proximity to
the earlier 96-query TRAIN input medians motivates this control only; it does
not establish prediction or metric equivalence. Nominal query-frame size and
observed target-frame support have different meanings, and C is not confidence.

## Cohort, partitions and storage

Use the exact accepted R17 regional 48 old TRAIN pairs, 16 in each of
`harvard_c3/hv_c3_1`, `harvard_c5/hv_c5_1`, `harvard_c6/hv_c6_1`. Bind every
input/manifest and the accepted regional selection, summary, completion and
path-repaired independent audit chain. No cohort or sample is selected by this
attribution's results. The four forward fields remain q_to_k_xy,
q_to_k_pixel_cov, q_features and k_features.

The original nine conditions are the Cartesian product of g8, g16 and mixed
query/key partitions. The five scale conditions, in order, are g8→g24,
g24→g8, g16→g24, g24→g16 and g24→g24. Native query counts are 432, 108, 243
and 48 respectively. All 48×14×2 = **1,344** native float64 probability
matrices are saved, with argmax arrays and canonical hashes. Their raw
probabilities occupy **512,434,944 bytes**, excluding supports and metadata.
All 672 original-role matrices must exactly match the committed accepted R17B
`trace_full__c02` array hashes; a mismatch is an engineering failure, not a
new baseline result. No changed primary is silently accepted.

Each pair retains all four prepared means/covariances, integer native bounds,
native traces and both actual float32 tau vectors. At the first metadata pair
of every scene, retain for all 14 conditions eight fixed linspace query rows
and all native keys: actual raw mass, float32 content and both actual float32
bias arrays. Compute the complete original feature matrix product and native
integrals before taking these rows. Small-row matrix products are not a
substitute for the complete original float32 arithmetic path.

The unchanged historical partition archive may store native bounds as int32
or int64. The existing preparation converts these exact integer bounds into
Torch int64, and the new retained support records save those actual prepared
int64 bounds. The independent native-trace reference accepts int32 and int64
partition coordinates, converts each endpoint to float64 before subtraction,
and rejects floating coordinates. This does not rewrite the original partition
archive or change the model's int64 input contract.

## Stages and source/label discipline

1. Jointly freeze the new source closure, protocol, fixed cohort, partitions,
   old accepted result/audit chain, runtime, intended label identities and
   first-pair/eight-query selection. Snapshot the new source closure. Expected
   old label hashes may be copied from metadata, but label bytes are neither
   hashed nor decoded at this stage. Freeze the independent audit lock next.
2. Run the three scene forwards. Save both roles, all supports/diagnostics and
   exact original-primary commitment matches. Repeat the first pair's forward
   deterministically. No labels enter forward.
3. Seal all three complete scene outputs, hashes and original-primary matches
   into a global forward commitment. Only after this seal may any label bytes
   be hashed or decoded for this study.
4. Evaluate unchanged old TRAIN regional targets and center targets, write all
   rows and descriptive summaries, then seal the result inventory.
5. Run the already-frozen independent audit. Keep a numerical/integrity failure
   as failure; do not filter tiny FOVs, unstable winners, conditions or scenes.

Regional labels are the original R10 dense gt_xy/gt_visible. Native mixed
center targets come from that same old label archive. The g8/g16/g24 center
targets and visibility masks all come from the already audited R13 target
contract, exactly as in R17 Stage A, joined by exact scene/pair/input identity.
Targets use observed sensor geometry and its original visibility rules. They
are not predicted means or area-weighted region centroids. No new sensor data,
validation data or fresh test data are requested.

The 672 exact original-primary matches refer to probability arrays. The R13
contract checked its g8/g16 coordinates against R10 by allclose, not byte
equality; this study retains the exact R13 coordinates rather than substituting
the close R10 version at a PCK radius boundary. Only the 144 old g8-key matrices
at q8/q16/q24 overlap Stage A's center experiment. This study does not add a
separate Stage A PCK replay requirement, and must not claim that all 672 newly
evaluated PCKs replay Stage A or silently switch target sources after results.

## Independent verification

The auditor does not import the new producer or new model. It verifies the
joint source/snapshot/upstream inventories, accepted R17B path-repair chain,
all identities, canonical probability and argmax hashes, complete 1,344-row
population and forward-before-label chronology. It may share the frozen R17B
preparation, float32 content and Gaussian operator for replay; this is declared
shared computation, not independent reconstruction of actual prepared moments.
All 672 original-role arrays must reproduce their original commitments.

For both roles, independently evaluate NumPy float64 tau and its float32
rounding/product. All saved tau vectors and the retained actual bias arrays
must match exactly. Using actual prepared raw mass, independently add the
original area floor once and recompute softmax; require maximum absolute
probability error ≤1e−12. Independently reconstruct support moments/content
from the four allowed inputs for the fixed three first TRAIN pairs, all 14
conditions and eight queries/all keys. Use the frozen independent whole-native
Gaussian GL64/96 reference, plus GL96 on actual prepared support to isolate
integration error. Require reference convergence ≤1e−8 and each production vs
reference/hybrid probability error ≤1e−4. Record every tiny/zero-FOV row and
argmax disagreement, without dropping or redefining its population.

Reconstruct all dense native and common target count tables independently.
Recompute every role's KL/JS, original common16-query→g8-key aggregation, direct
metrics and PCK from the retained native probabilities. First-native-index
argmax wins exact ties; PCK uses squared sensor-center distance ≤64 pixels².
Correctness is restricted to the original visible-center mask. Count integers
must match exactly. Scene PCKs and their equal-scene means/differences use
`Fraction`, with no floating epsilon. Floating metric sums use atol 1e−8;
derived summaries use atol 1e−10. Source/hash/discrete population comparisons
are exact.

## Reporting and interpretation

Report both roles and their paired differences per scene and per condition.
Common9 and direct5 KL/JS retain their original mass-within-scene,
equal-scene/condition macro averaging; the inherited half-common9/half-direct5
J may be shown descriptively, never as a new selection objective. Report exact
PCK fractions, integer denominators, argmax flips on all and visible queries,
and visible harm/benefit transitions. Stratify native direct KL/JS and PCK by
exact width×height, including both rectangular mixed orientations. Do not
assign the nonlinear common-aggregation KL to a native-size stratum.

Describe the input support population once per pair and query, without
repetition across key partitions: 48×(432+108+243+48) = **39,888** unique
queries. The 21 fixed scene×query-partition×width×height groups report n and
linear min/p05/median/p95/max of actual observed tau32, nominal tau32, their
ratio, observed trace and nominal trace. No bins, exclusions, correlations or
significance tests are chosen from effect results.

This is a conditional, post-design TRAIN attribution study. Neither a numerical
audit pass nor either role's metric advantage passes a scientific gate,
establishes equivalence, replaces the selected primary, validates unseen data,
or proves general usefulness of observed covariance. A report must state these
limits and keep the current R17 VAL result/decision separate.

No comparative latency is included in this minimum study. If later authorized
and separately frozen, timing must pair both roles on the five old scenes'
first pairs and nine original conditions, with 3 warmups and 15 repeats. Each
role pays for its own complete support preparation, geometry integration,
content product, tau computation/checks, softmax and common aggregation.
Nominal tau must compute its own bounds trace; shared forward caches cannot
be presented as independent operator speed. Common prediction/projection
stages and excluded upstream work must be described explicitly.
