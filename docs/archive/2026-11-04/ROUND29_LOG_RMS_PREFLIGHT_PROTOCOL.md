# R29: synthetic validation of canonical log-mass RMS projection

2026-10-31, following the accepted R28 failure and its bound R29 design.
This is a numerical representation experiment, not a new probability law,
efficacy candidate, general accuracy certificate or real-data experiment.
All R26/R28 and earlier scientific failures remain recorded. No real arrays
or labels may be decoded, and no old formal attempt may be overwritten.

## Law and admissibility retained

Copy the immutable R26 RMS projective Gaussian mathematics into a new model.
Preserve input dtype/shape/finite checks, positive ordinary floor-area product,
RMS/depth rank guards, sign-normalized QR, positive-front representability,
h=.25, floor density1e-30, GL24, tile maximum side2h, positive branch,
negative erfcx branch,20-term negative series and its relative remainder1e-10.
No cutoff, density threshold, new covariance repair or additional model.
POLICY is identical to R27, including4,000,000 nodes per production call.
The declared log-density must be finite on every positive-law node; a
nonfinite value is still a failure, with no attempt to assign it zero mass.

Use log-sum-exp of positive weighted density terms and tile masses. Evaluate
native cells and the direct FOV using separate tile integrations. Never set
direct FOV equal to the native sum by construction. Apply one area floor in
log space, then the same bias and normalization. Existing ordinary floor
admission remains: log arithmetic must not quietly admit an input rejected
by the prior positive floor product check.

## Canonical outputs and partial work

Successful NPZ model arrays are:

- log_raw_mass float64[N]; log_raw_fov_mass float64[1].
- log_floored_mass, log_probability and probability float64[N].
- materialized_raw_indices int64[M] and materialized_raw_mass float64[M].

The harness additionally saves exact model inputs and an int32 first-index
argmax. Nonstructural complete canonical raw logs must be finite. Only the
original exact structural-zero cases use -inf raw logs; their ordinary
materializations are all true zeros. For positive mass, materialize only
indices whose ordinary exp is finite and strictly positive. An omitted index
means positive mass not representable in this ordinary output, not zero mass.
Keep subnormal and unmaterialized counts. All numerical raw gates use the
canonical logs; an ordinary subnormal is not a relative-accuracy guarantee.
The old semantic fields all_raw_mass_zero/zero_raw_cells/area_prior_used must
reflect exact structural zeros, never ordinary materialization failures.

JSON log_raw_mass_sum may be null only for structural zero; log normalizers
and total floor are finite. The direct log FOV resides in its NPZ array, so
no nonstandard JSON Infinity encoding is needed. Saved P is checked against
log_probability and the independent once-floor calculation.

Failures retain partial_log_raw_mass[N], partial_log_raw_fov_mass[1],
partial_completed_cells bool[N], partial_completed_fov bool[1],
partial_accumulated_tiles int64[N], partial_accumulated_fov_tiles int64[1],
and actual work/diagnostics. A finite incomplete cell is a partial sum only.
An uncomputed -inf position is not a structural zero. A deadline may prevent
a call; the journals must distinguish that state from an attempted failure.

For each local positive log-sum, count finite terms whose relative exponent
actually becomes zero. In scopes tile_nodes, cell_tiles, fov_tiles,
native_total, floor, floor_total, floored_total and content_normalization retain sum_count,
underflow_sum_count, underflow_term_count and maximum log scaled/absolute/
relative omitted bounds. For N lost terms with maximum shifted exponent d,
log N+d bounds their exact scaled sum; add the local maximum for absolute,
and subtract the retained scaled log sum for relative. These are maximum
single-sum bounds, not their sum, not an end-to-end error certificate, and
exclude absorption of positive terms in addition and all other roundoff.
Null maxima mean no recorded relative-exp loss, not a proof of zero error.

## Fixed population and independent references

Preserve all26 R27 descriptors and their canonical input bytes. Put its20
base cases first, then the following3, then its6 original identities. The
resulting IDs are27_native432_far_cells,28_subfloat_fov,29_visible_far_tiles.
All3 use mu=(12,9,1) for27 and(0,0,1) otherwise,F=0,bias0,h=.25.
Case27 uses native432 bounds(pixel_boundary-.5)/8. Case28 uses the four
Cartesian cells x=[9.75,10,10.25],y=[-.25,0,.25]. Case29 has the one rectangle
[0,-.25,12,.25]. No numerical pilot of this formal population precedes freeze.

For each old20 base, run the immutable R27 independent SVD/conditional
Gaussian reference at orders32/48. Preserve all its failure, raw, direct-FOV,
tail and work semantics; converting valid ordinary reference raw to log is
only a comparison representation. A structural zero converts to -inf only
under the exact branch. A positive reference zero is a failure. Do not use
production density or outputs to repair reference failures.

The3 new exact constant-depth diagonal cases instead use one independent
analytic Gaussian rectangle reference each. Recover the screen mean and
diagonal covariance independently from mu,F and the declared regularizer,
then add log normal-CDF interval probabilities. Enforce exact constant positive
depth and diagonal admission; no approximate diagonal tolerance. Evaluate the
direct FOV separately. Use stable reflected log-CDF differences and no depth
or X cutoff. This is essential for case28 beyond39 standard deviations.
Count CDF evaluations separately; no fictitious quadrature nodes and no
claim of32/48 refinement from repeating an analytic formula. Its floating
evaluation is not an interval or arbitrary-domain precision certificate.

There are29 production statuses,40 old numerical reference statuses and3
analytic reference statuses. Every base comparison, old reference pair,
old6 identity and the old fixed-epsilon near-constant check must pass.
Structural-zero raw-relative exclusions remain only the original exact cases.

## Original numerical gates and budgets

Keep all C27 GATES unchanged: production/reference P max absolute1e-4;
raw-FOV relative and native raw-L1/reference-FOV1e-6; reference-pair P/raw1e-8;
normalization2e-12; native/direct FOV absolute1e-8; integrated identities1e-8;
near-constant P/raw1e-8; reference truncation raw relative1e-10 and P1e-8.
Use log/expm1 identities from the representation note to compare these
same raw quantities. Never replace an absolute raw gate by a log-distance
gate, or waive raw accuracy when the floor dominates P. Retain log errors
when ordinary diagnostics underflow; underflowed summaries cannot drive gates.

Production phase remains600s. Each numerical reference is at most250M nodes;
reference phase at most600M nodes and1800s, including analytic calls and
record writing. Preserve old reference depth/interval limits, GL orders,
standard-X panels, cutoffs and positive tail certificates. Analytic cases
have no cutoff tail; their CDF floating error is separately unproven.
No automatic retry, order increase, budget increase or case deletion.
Record actual attempt journals, available partials, stage times and process
RSS high-water values without calling RSS per-case exclusive memory.

## Freeze, audit and permitted interpretation

Complete distinct pure-toy tests, retaining all failed development attempts.
Jointly freeze full local source closure, fixed descriptors, policy/gates,
runtime and R28 accepted failure/R29 decision/R27 accepted baseline metadata.
Keep current files plus dual snapshots. Prove freeze does not decode NPZ or
call either numerical model/reference with raising sentinels. NumPy-first,
OMP/MKL/OpenBLAS=4, the existing Python runtime, no installation or GPU change.

Formal order: producer freeze, independent audit freeze, production29 seal,
independent43 reference outcomes and all checks, final complete inventories,
root provenance/arithmetic/chronology acceptance and report. Preserve actual
calls even when any earlier case fails. A numerical failure is still a valid
scientific record; delivery integrity and numerical pass are separate flags.

Every stage keeps real_inputs_read=false,labels_read=false,
general_numerical_accuracy_established=false,efficacy_tested=false,
candidate_selected=false,scientific_gate_passed=false,goal_complete=false.
A complete numerical pass only permits selecting a later separately frozen
real applicability screen. It does not satisfy regional/center/robustness,
actual V, optimized GS or untouched confirmation gates.
