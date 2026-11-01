# R15: conditional old-validation regression and paired CPU cost

Draft implementation prepared before Stage B regional results. No execution is
authorized by this document alone. Freeze requires the complete independently
accepted R15 Stage B selection, with the passed Stage A feasibility chain and
all failed numerical history preserved. The independent validation audit must
also be source-frozen before the first validation forward.

The detailed contract is `docs/round15_validation_blueprint.md`; its complete
contents are included in the source lock. Preparation timing is clarified in
`docs/round15_validation_draft_preparation_note.md`. Neither changes Stage B.

## Fixed prediction and populations

Exactly the original 32 validation pairs: Harvard c8 and Brown bm4, 16 each.
They are reused development data, never independent confirmation. No new data,
retuning, checkpoint selection, per-scale bandwidth or replacement primary.
Eighteen semantic aliases are assembled from the accepted selection: primary,
seven regional controls, two point localizers, seven legacy configurations,
and matched trace-isotropic quartic. Strict family/h/T/operator identity permits
only exact alias sharing, in declared first-occurrence order o00, o01, etc.
Strong point and strong quad are references to existing regional aliases.

Matched isotropic is attribution only: separately compute trace(C)/2 I with
the primary h/T and unchanged content/integration. It cannot replace primary
or participate in calibration or reference selection. Trace-only accumulation
is implemented and independently checked before freezing.

Each operator has 68 conditions per pair: seven geometry states times nine
original partitions, plus five scale conditions in zero state. The six biases
retain the exact R12 transforms and units. Zero follows the same reprojected
observation path as R12. Before final freeze, check its float32 forward fields
against the original cached zero fields on the three first training inputs;
if they differ, resolve and document the numerical path before effects.

The loader reads only the ten fields in the implementation map. No sensor
visibility, regional labels or center labels enter prediction. All nonfinite
fallback/clipping and behind-camera rows remain. Quartic native integrals use
explicit fixed GL24, <=1 reference-unit tiling, float64 integration, a single
native-area density floor and the same float32 content path. Gaussian controls
keep their frozen GL32 and native24-to-four12 rectangles. The same probability
supports KL, native argmax PCK and any subsequent V; no separate center decoder.

## Storage, chronology and independent audit

All 32 pairs x 68 conditions x U unique operators are predicted before any
validation label bytes are hashed or decoded. Record every float64 canonical
array hash, shape/config/input identity and numerical diagnostics. Save all
2,176 primary arrays, plus all 136 first-pair arrays per control. Retained raw
bytes are 1,112,516,352 + (U-1)*69,532,272, checked before freeze and forward.
Old evidence is never deleted. Both scenes' first pair is fully repeated with
fresh preparation, requiring bitwise equality for all 68U matrices.

After the global forward seal, primary arrays are read and hash-verified;
controls are exactly regenerated under the same source/runtime before metrics.
Old R12 audited center-label archives may be reused only through the bound
completion/provenance chain. R12's scientific failure remains false. All original
label fields must equal R10; the independent auditor reprojects native sensor
centers, including g8/g16/g24, without using any forward perturbations.

The separate auditor imports no new producer, model or producer-gate module.
It checks all identities, integer populations, retained-array hashes/metrics,
aggregation, gate decisions, timing medians and source chronology. All 224
pair/state geometry records receive independent reconstruction. Its numerical
sample is both first metadata pairs, all 68 conditions, all U unique operators,
eight fixed linspace query indices per partition and all native keys. This is
136U matrices, or 1,088U query rows. Quartic reference GL64/96 convergence must
be <=1e-8; actual probabilities must differ by <=1e-4. Gaussian references use
the already independent high-order integration. Zero/tiny/FOV/floor/argmax
discrepancies are reported without deleting rows or excusing failed tolerances.
Aggregate sums use 1e-8 audit tolerance, summary values 1e-10; decisions and
Fraction-based PCK comparisons must match exactly. Full sampling coverage is
distinguished from all-population provenance/aggregate checks.

## Frozen scientific checks

The blueprint's complete intersection applies. For zero/common9, primary must
beat both strong regional point and legacy point by >=2% macro KL, with both
scenes improving. Against each of strong quad and legacy quad16, separately:
>=2% KL improvement with time <=1.25x, OR KL <=1% worse with time <=0.5x.
Every original condition must be <=2% worse than each quad. Zero g8→g8 PCK
must lose <=0.5pp to each new point localizer and legacy point_pck.

Five-scale direct KL must improve >=2% over each point reference, both scenes
improving; every condition is <=2% worse than each quad. Native g24→g8 PCK
has the same three-reference protection. Each of six perturbations and zero
must have common9 macro KL <=2% worse than both same-state quads and g8→g8
PCK within 0.5pp of all three same-state localizers. Report every condition.
All probabilities must be finite and normalized within 1e-12.

The newly preregistered zero g16→g8 PCK guard uses the same three references
and exact 1/200 boundary; it is explicitly additional to R12. Scene PCK is
formed from integer counts; two scenes are equally weighted using Fraction.
No floating tolerance changes an effect threshold. Report every failed check.

## Complete incremental timing

Five old scenes' first metadata pair x nine original conditions x U operators,
one CPU4-thread process, 3 warmups and 15 samples each, cyclic operator order.
Every timed execution freshly prepares its own necessary query statistics,
key geometry/content, integral/floor/content/softmax and common aggregation.
No family borrows another's moments or geometry. T0 still runs the actual
content path. For each of 45 cells, take the median complete execution time;
then average the 45 medians. Do not add separately computed medians.

Shared predicted RGB/depth and dense projection are outside this incremental
boundary. Report common projection/export wall time separately, not as a full
visual-system cost. Save host load, tensor byte counts and process RSS scope;
static bytes and process high-water RSS are not per-role allocator peaks.
Timing runs alone after prediction/evaluation compute is finished.

## Interpretation and continuation

Independent audit acceptance is distinct from scientific gates. If any gate
fails, preserve the complete run and stop this unchanged validation candidate.
Even a pass does not authorize unmodified R12 training or prove actual V,
optimized GS/RGB or fresh generalization. A simpler control that dominates
quality/cost limits the case for quartic; isotropic matching full limits the
direction-shape attribution. Those observations do not alter the fixed primary.
Only separate frozen follow-up designs can address remaining inherited tasks.
