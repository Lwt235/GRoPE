# R28: RMS projective Gaussian on the fixed72 real geometry inputs

2026-10-31. This follows the accepted R27 synthetic numerical pass. It is a
fixed real-input applicability and cost screen only. R26 remains failed;
R27 does not establish general numerical accuracy. The R23 implementation's
0/72 completed calls and all earlier scientific failures remain intact.

## Fixed inputs and exact transformation

Reuse all72 previously prepared and independently checked R23 inputs, in the
same order: c3,c5,c6 old TRAIN scenes, pair30/45, query sizes8/16/24, eight
previously fixed indices each. No resampling, patch reconstruction, original
R10 archive decoding, fresh scenes, labels or new anchor/content calibration.
The old input audit validates predicted geometry, not the failed R22/R23
probability implementation. R28 explicitly inherits only that input evidence.

Before array access, authenticate R27 acceptance/selection and R23 acceptance,
full sealed inventories, current/dual-snapshot old sources, prepared records,
query specs, all72 archive hashes and old per-input checks. Provenance and
runtime helpers in `experiments/round28_rms_contract.py` are shared; numerical
mapping and independent geometry/output checks are implemented separately.
Hashing sealed archive bytes during metadata freeze does not decode arrays.

After both new freezes, decode only these fields in each R23 prepared archive:
input__mu[3], input__factor[3,3], input__bounds[432,4], input__H[2,2],
input__offset[2], target__covariance[3,3], query_bounds_px[4], query_index[].
Validate their original canonical dtype/shape/content hashes. The first six
are float64; the two query identity arrays retain their exact original dtype.
Do not access source__* or any observation/label fields from other archives.

Form A=[[H00,H01,c0],[H10,H11,c1],[0,0,1]], c=input__offset, then
mu_h=A mu and F_h=A F. Preserve the entire2x2 H block, including any skew.
Native bounds remain byte-identical and native content bias is exactly zero.
H and c already equal the old camera blocks divided by8; bounds are the old
(pixel_boundary-.5)/8. There is no additional half-pixel subtraction from c.
No depth clamp, projection clip, recentering, covariance symmetrization,
eigenvalue repair, jitter or camera-frame regularizer is introduced.

Each new prepared NPZ contains the8 original arrays named r23__<oldkey>,
mapping_matrix A, and input__mu/input__factor/input__bounds/input__content_bias.
Save source archive identity, all13 canonical array hashes, fixed scene/pair/
query identity and elapsed time. Globally seal all72 prepared outcomes before
the independent input audit, with no production prediction yet executed.

## Independent input gate before prediction

Audit every prepared array against its bound source and case identity. Use
componentwise math.fsum to recompute H*XY+c*Z for the mean and each factor
column; do not invoke producer preparation. Check the factor covariance
against the independent congruence A target_covariance A^T. The normalized
error maxabs(actual-expected)/max(1,maxabs(expected)) must be<=1e-12.
Also check exact lower row/depth preservation, RMS depth agreement, original
native bounds against both their source and the literal144x192/8px layout,
native bias exactly zero, and mapping matrix H/c exactly inherited. No
roundoff tolerance can authorize changing an input or switching its rank.

Seal all72 input checks and write input_verification.json. Producer forward
requires every input check and complete inventory to pass, with both source
freezes and the global prepared seal preceding the input-audit receipt. Any
input failure remains recorded and blocks all dependent model calls. This
input gate does not certify probability quadrature on the new real inputs.

## Immutable model, phase and failure retention

Use exactly `models/round26_rms_projective_pe.py`, SHA256
48b63a51624ddaead32892d46f853cb5cae0b9a88e0e13d29549235b8fa37e9a.
It RMS-normalizes the homogeneous Gaussian, adds h=.25 XY regularization,
projects strict positive Z, integrates native raw mass by GL24/tiles<=2h,
adds one1e-30 area floor, and applies the fixed zero bias. All model policy
constants are unchanged. No actual learned Q/K/V, content-temperature rule,
semantic anchor, covariance counterfactual or alternative model is selected.

Each call has at most4,000,000 density nodes including direct FOV. The fixed
nonstructural432-cell call would require1,990,656 nodes; this static operation
count does not promise runtime or success. All72 forward outcomes share600s,
timed from stage start through final per-case records; post-phase provenance
verification and final inventory sealing are timed separately. Input
preparation/audit costs are recorded separately, not hidden in model timing.
Use the accepted NumPy-first existing runtime, OMP/MKL/OpenBLAS=4, no GPU or
installations. The new runtime is frozen explicitly; old input bytes need
not be reconstructed under their former runtime.

Retain all72 statuses and every actual-call start/result journal. Save all
available partial raw arrays/work on model failure; save input identity even
when the shared deadline prevents a call. Distinguish attempted failures from
unattempted deadline rows. No missing/failing input receives a fabricated P,
zero mass, area prior, new order or retry. The frozen model deliberately
rejects positive-density underflow. Even if the floor could dominate P, a
far cell causing underflow remains an applicability failure of this version.

## Final audit and interpretation

After global forward sealing, independently validate every saved input,
canonical array/archive hash, journal, status, failure reason and work count.
For completed cases, verify finite nonnegative raw/P, single-floor arithmetic,
normalization<=2e-12, raw native sum versus separate FOV absolute error<=1e-8,
model branch/front-mass/tail-series diagnostics, expected tile/node counts,
first-index argmax and exact/near1e-12 ties. Record raw mass below floor, zero
cells, partial work, elapsed time and cumulative-process RSS; do not select
cases based on these diagnostics. Structural-zero handling follows the exact
model branches; underflow must not be relabeled structural zero.

There is no independent probability integration on these real inputs in R28.
The analytic GL bound is exact-arithmetic only and excludes density evaluation
and floating-point effects; do not promote it to a general raw/P certificate.
All72 completed calls, all input/output audits and original budgets are
necessary for applicability_calls_completed=true. Always record
independent_probability_reference=false,
general_numerical_accuracy_established=false, efficacy_tested=false,
candidate_selected=false, labels_read=false, scientific_gate_passed=false,
goal_complete=false. An intact failure delivery can be accepted while the
applicability flag remains false.

Complete independent pure toys, then jointly freeze full local source closure,
runtime, exact72 query descriptors and predecessor bindings. Prove metadata
freeze cannot decode NPZ or invoke the model. Formal order is producer freeze,
audit freeze, prepare72, independent input audit, forward72, final audit,
root inventory/chronology acceptance. Preserve every failed formal attempt.
Real effect/robustness, actual V, optimized GS and untouched joint confirmation
still require later fixed protocols and every original scientific gate.
