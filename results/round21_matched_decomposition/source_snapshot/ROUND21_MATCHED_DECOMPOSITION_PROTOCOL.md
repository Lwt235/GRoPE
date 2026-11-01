# R21: fixed matched-content decomposition on old TRAIN

2026-10-31. Declared after the accepted R20 failure and before any R21
forward prediction. R20 passed 58/60 conjunctive resource checks and failed;
its two failed roles refer to the same g24->g8 quad16 configuration. It is
stopped. R17 scientific failure and R18 numerical failure remain unchanged.

This round is a mechanism diagnostic, not a new efficacy candidate, model
selection, rescue calibration or Gaussian-family impossibility test. Its
purpose is to separate changes in designated center, second moment, retained
support shape, and their interaction with a common content bias.

## Population and fixed operators

Use all 48 accepted old TRAIN pairs, 16 per harvard_c3/c5/c6, and exactly
g8->g8, g16->g8, g24->g8. Every query and all 432 native g8 keys are retained.
There are 432/108/48 queries per pair, hence 28,224 unique pair/query rows.
No new data, old VAL, perturbation, training, actual V, GS or fresh access.

The inputs, pixel geometry, within-pixel covariance, pooling, partitions and
float32 moments/content follow accepted R20 and its frozen dependencies.
Let mu32,C32 be its original full1 support, a64 the original R13 designated
center projection divided by eight, and

    d = mu32.double() - a64
    Ca64 = C32.double() + d d^T
    h = .25; T = 10
    tau64 = T h^2 / (h^2 + trace(Ca64))
    b32 = tau64.float() * content32

Use the frozen scaled float64 tau implementation and its float32 multiply.
This same b32 is supplied to all four geometries. Do not recompute tau from a
mixture, another covariance or the chosen geometry. The zero-content arm uses
an exactly zero float32 bias of the same shape.

| Geometry | Center / covariance or components | Zero role | Content role |
|---|---|---|---|
| original full1 | mu32,C32, promoted to float64 | mu_c_zero | mu_c_content |
| changed center only | a64,C32.double() | a_c_zero | a_c_content |
| changed center and second moment | a64,Ca64 | a_ca_zero | a_ca_content |
| original full4 | original four local Gaussian nodes, covariances, weights | full4_zero | full4_content |

Each component includes h^2 I. Geometry is the GL32 integral of its Gaussian
or weighted Gaussian mixture over each native g8 cell, with one floor
1e-30 times native cell area/64 after component summation. Then

    P = softmax64(log(raw_mass64 + native_floor64) + bias32.double()).

All h/T values are fixed; there is no grid, fitting, lambda, mode adjustment,
component correction or selected subgroup. The full4 arm preserves the exact
old implementation. It has the same whole-support first two moments as full1
in real arithmetic, including within-pixel covariance, but uses different
float32 reductions. Save and report the mixture-to-full1 mean/covariance
residuals in float64; do not move nodes or change covariances to force bitwise
moment equality. It is a higher-shape-information diagnostic, not an equally
compact representation or a claim of strictly identical floating statistics.

The two a-centered content roles must reproduce accepted R20
mean_only_matched_tau and anchor_trace, respectively, for these conditions.
The mu/full4 zero roles reproduce the original h=.25/T0 Gaussian paths.
Old full4 complete probabilities were not retained for all pairs: distinguish
new execution from canonical-hash replay of all original commitments, actual
array replay of saved first-pair archives, and comparison with accepted
complete metric rows. All 288 full1/full4 zero matrices must match their old
canonical commitments; 18 of these also have original saved arrays for direct
array comparison (three scenes x three conditions x two roles). Never claim full
direct reuse where archives do not exist. Bind the precise old source,
selection, completion, available archive and metric identities before forward.

## Predeclared comparisons and interpretation

Report ten paired contrasts, with new minus reference consistently signed:

1. For each of four geometries, content minus zero (four contrasts).
2. For each of the two content settings, a_c minus mu_c, a_ca minus a_c,
   and full4 minus mu_c (six contrasts).

Also report each of the three geometry contrasts' content-minus-zero
difference of differences, using the same scene populations and aggregation.
These are descriptive effects, not three newly invented efficacy thresholds.
KL/JS decreases and PCK increases are favorable; continuous error decreases
are favorable. Include all scenes and query scales, unfavorable signs and
zero effects. Do not select a contrast, scale or outcome after observing it.

This is four geometries crossed with two content arms, not a full factorial
experiment on three independent geometric factors. The shape comparison is
at original mu/C; there is no matched full4 at a/Ca. It therefore cannot prove
that higher-order shape explains all of R20's remaining error. Matched bias
controls its numeric input, not its resulting effect after normalization.
Moment residuals remain a finite-precision limitation. No failed finite
optimization or chosen distribution proves that all Gaussians are impossible.

Full4 cannot be promoted from this diagnostic. A later design must state its
own compact encoding and legal information contract and pass all applicable
old gates. Actual optimized GS with a=mu has no anchor correction, and its
existing no-content/T0 task has no temperature effect. This diagnostic does
not resolve that generality limitation or the old actual-V/RGB failures.

## Freeze, storage and forward audit

Jointly freeze this protocol, design review, producer/model/metrics, independent
auditor/metrics/numerics, tests, the full transitive local import closure, source
snapshots, input metadata and inherited accepted-evidence hashes before real
forward. Bind R20 root acceptance, its accepted source/inputs and R19 center
provenance. CPU-four-thread NumPy-first/Torch runtime is unchanged: record and
check build identity, OMP/MKL/OPENBLAS and Torch intra/inter-op settings, TF32
and deterministic flags. Do not bypass runtime guards or change system GLIBC.

Save all 1,152 float64 native P matrices (780,337,152 raw bytes), argmax and
four raw-mass matrices per pair/condition (390,168,576 raw bytes), supports,
content/tau/bias, identities, canonical array hashes and diagnostics. Expected
probability+raw storage is 1,170,505,728 bytes before archives/supports. Check
free space and preserve old evidence. Stages refuse overwrite and retain
failed attempts. Seal all 48 pairs before any actual label hash/decode.

Synthetic tests must cover the eight fixed roles, common-bias identity, zero
content, mixture collapse, within-pixel moments/translation and finite-precision
residual reporting, native-floor placement, preservation of all rows, exact
paired counts and chronology. They do not certify real-data numerical accuracy.

The independent forward audit reconstructs all 28,224 a positions using its
own float64 solve and bilinear interpolation, requiring <=1e-9 pixel error
and exact flags; binds/rechecks original support/content provenance; checks
R20's two accepted-role replays; and checks all 288 old T0 canonical commitments
plus the 18 saved old T0 arrays.
All 1,152 P and 576 raw matrices receive shared-source exact replay; label
this reproducibility, not independent mathematical reference coverage.

Independent Gaussian mathematics is fixed to the first metadata pair in each
scene, all three conditions and eight roles, eight evenly spaced query rows
per matrix, all native keys: 72 matrices / 576 role-query rows, only 72 unique
pair/query identities. Independently reconstruct support moments, full4 local
moments/weights, a/Ca and content in NumPy. Use independent GL64/96 references
and a hybrid integral from executed statistics to distinguish statistic and
integration errors. The references, actual orders, local dependency hashes
and sample indices must be locked before the first forward.

Keep inherited numeric limits: normalized moment error <=1e-4; content error
<=1e-5; full and hybrid P absolute error <=1e-4; reference64/96 convergence
<=1e-8; same-raw arithmetic and normalization error <=1e-12; exact production
tau32/bias32. Preserve argmax changes, original raw-zero/tiny-FOV events and
the full population. A sampled absolute-error pass is not a guarantee of
relative tail accuracy or of full-population integration accuracy. No reference
order increase, tolerance relaxation or arm deletion within a failed run.

## Evaluation and independent acceptance

Expected old label hashes can be read from accepted metadata at freeze.
Only after complete forward sealing and a passed independent forward audit
may the actual regional targets and R13 center-target files be hashed/read.
Old TRAIN remains repeatedly used development data, not a blind test.

Use the unchanged native/common aggregation and visible-pixel mass weights.
For g8->g8, report both common (g16-query/g8-key) and direct/native KL and JS;
for g16/g24->g8 preserve their corresponding original metric semantics.
Report sums/masses, per-scene values and equal-scene averages for each role
and condition. Do not invent a common9/direct5 score for three conditions.

For centers use the unchanged R13 `*_point_visible` masks, including their
original cross-view evaluation contract, and native-key-center first-index
argmax, squared distance <=64px for PCK8; valid totals must be
9,521/2,463/1,096. Report exact integer correct/valid, exact scene and
equal-scene Fractions, mean distance and RMSE in pixels, and fixed paired
benefit/harm/neither/both counts plus paired continuous-error changes.
Continuous distance benefit/harm/equal counts compare the original float64
squared distances, before taking square roots; distance sums and RMSE remain
in pixels. This avoids changing ordering through square-root rounding.
Those transition counts apply to the ten actual two-role contrasts; the three
differences of differences are arithmetic metric contrasts, not invented
four-role transition populations. Do not substitute the R8 depth task's
query-only sensor-valid mask for the R13 center population.
Do not filter using predicted FOV, front-facing status, mass, geometry,
successful history or target-region overlap. Center and regional populations
are distinct. Repeated role rows are not independent queries.

Independent final evaluation recomputes every metric sum, paired contrast,
integer count and exact Fraction from saved predictions and original targets.
Use inherited metric-row tolerance 1e-8 and summary tolerance 1e-10 only for
numeric comparison; exact counts/Fractions/hashes must match. Compare the
available accepted R20 and old full1/full4 T0 metrics under their actual
semantics, preserving any recorded finite-precision discrepancy.

Root accepts complete inventories, source/snapshot/lock chains, numerical
and final completions, and chronological guards before interpreting effects.
There is no scientific pass/fail or replacement candidate selected by this
diagnostic. Stop on integrity/numerical failure; if accepted, preserve every
contrast and use the evidence only to motivate a separately declared design.
All previous scientific, robust-geometry, strong-control, same-information,
cost, learned-Q/K/V, optimized-GS and untouched-confirmation obligations remain.
