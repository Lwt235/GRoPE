# R17 old-VAL implementation plan — conditional, not execution authorization

2026-10-30. This is an external implementation note, not part of an already
frozen validation run. It was prepared by reading source and protocols, without
reading R17 Stage B scores. No new VAL code or experiment is executed by this
note. The original R15 validation draft was never frozen or run; it cannot be
launched unchanged for R17.

The controlling design is [ROUND17_CANDIDATE_PROTOCOL.md](../ROUND17_CANDIDATE_PROTOCOL.md).
Preparation remains pending **independently accepted Stage B and its passed
TRAIN screen**. A correct independent audit can accept a failed screen; that
outcome must stop the selected candidate. It cannot authorize validation.

## Minimal versioned changes

Keep every frozen R15/R17 file intact. The existing R15 drafts provide most of
the orchestration and both independently implemented aggregation/gate paths.
Use new R17 files, with these narrow responsibilities:

| New file | Starting interface and necessary change |
|---|---|
| `ROUND17_VALIDATION_PROTOCOL.md` | Freeze the final selected config, 19 semantic aliases, 32-pair/68-condition population, source/data chains, arithmetic, retention, numerical coverage and paired timing; preserve every inherited gate below |
| `models/round17_validation_pe.py` | The separate agent's unrun draft preserves `prepare_query/prepare_key/geometry/predict`; adds `trace_full` and `trace_isotropic_matched`; reuses frozen tau and seven old controls; still requires final source freeze |
| `experiments/round17_validation.py` | Adapt `selected_contract`, role construction and source policy; keep metadata-only target binding, forward/seal/replay, timing and acceptance stages |
| `experiments/round17_validation_metrics.py` | Version the R15 pure module: update alias/family policy and matched controls, retain population, Fraction PCK, macro reductions and scientific inequalities |
| `experiments/round17_validation_audit_metrics.py` | Independently make the same literal role-policy updates; do not import production gate/selector functions |
| `experiments/round17_validation_audit_numeric.py` | Extend the R15 independent geometry/reference interface for trace temperature and separate geometry/tau covariance; retain all fixed VAL samples and stress states |
| `experiments/round17_validation_audit.py` | Adapt upstream acceptance, roles, numerical/source policy and exact tau attribution evidence; preserve independent targets, projection, population, timing and gate checks |

Add corresponding synthetic tests before freezing. There is no need for a new
RGB/depth exporter, new center-label producer, validation search or learned
module. The shared R12/R15 low-level geometry, counts and I/O remain frozen
dependencies. The old draft's fixed `quartic_full` primary, 18-alias checks,
quartic-isotropic implementation and R15-only acceptance string must change;
blindly importing those complete old runners or validators is incorrect.

## Acceptance and role manifest

The new `selected_contract` must validate R17 regional `selection.json` and its
checksum, the regional lock/seal/summary/completion, accepted independent
verification, and any explicitly bound audit-path repair. The repair is an
honestly later engineering patch, not a pre-forward scientific registration.
Bind its lock/source/snapshot/completion provenance without changing the
original regional source lock. Require exact agreement of selection and
`train_screen` among summary, verification and acceptance artifact, with
`train_screen.passed=true` and `next_action=prepare_old_validation_protocol`.
Keep `scientific_gate_passed=false` at this prerequisite stage. Also bind the
accepted Stage A, inherited localization selections, numerical preflight and
same-prepared arithmetic audit. Do not independently reselect any reference.

Retain the existing manifest shape:

```text
roles = {
  aliases: {semantic_alias: oNN},
  operators: {oNN: {family, h, temperature}},
  primary: "primary", strong_point: regional_alias, strong_quad: regional_alias
}
```

The proposed fixed alias order is primary; the seven existing
`regional_{quartic_point,gaussian_point,gaussian_fixed,gaussian_full1,
gaussian_full4,gaussian_quad4,gaussian_quad16}`; the two `localizer_*` aliases;
the seven `legacy_*` aliases; `matched_constant_temperature`; and
`matched_isotropic`: **19 semantic aliases**. Primary uses `trace_full`.
Matched constant temperature uses `gaussian_full1` at primary h/T. Matched
isotropic uses `trace_isotropic_matched` at primary h/T.
Both attribution roles are evaluated completely, have no calibration and
cannot replace primary or enter strong-control selection.

The three localization references remain the accepted global Gaussian-point
selection, accepted global quartic-point selection, and legacy Gaussian
point_pck h=.5/T=5. Regional comparisons remain the accepted strong regional
point (chosen previously across the two point families) and legacy Gaussian
point h=1/T=5; quadrature comparisons remain the accepted strong quad and
legacy quad16 h=.5/T=5. Keep both semantic comparisons even when their
operator/configuration happens to coincide. The old separately optimized
full1 remains distinct from the newly matched constant-temperature role.

Canonical IDs follow first occurrence of identical family/h/T and the same
complete numerical contract. Determine U from the sealed manifest; update the
old 18-alias bounds rather than assuming its former count. A conservative
structural bound is 9<=U<=19. Do not merge trace and constant-T families from
an observed equality, including at T=0: their executed checks/cost paths can
differ. If T=0 is the selected primary, check the expected probability equality
to matched full1 and disclose inactive coupling, with no tau-effect claim.

## Forward and attribution arithmetic

Reuse the ten-field loader and `models.round12_stress_pe.perturbed_observations`:
the four projected geometry/content fields plus q_points3, q_pixel_cov3,
q_to_k_transform, k_K, q_predicted_depth and k_predicted_depth. No sensor or
supervision field enters prediction. Preserve zero reprojection, depth×.9/1.1,
target Ry±1 degree and target tx±.01m exactly, including fallback/clipping and
all behind-camera or floor-dominated rows. Five scale conditions use zero only.
Repeat the three-first-TRAIN-pair zero/cache float32 equality preflight before
VAL freeze; an old allclose check alone cannot justify hash reuse.

Primary uses the existing full query moments, native Gaussian GL32 mass,
native24→four12 numerical children and one native-area floor. Then use the
frozen `models.round17_trace_pe.probabilities(geometry64,content32,C32,h,T)`:
double common-scale tau arithmetic, rounded tau32×content32, then double
addition/softmax. The zero/point/T0 identities and all PSD/symmetry checks stay.
Do not approximate the new rule with scalar T, detach content, add another
floor or change integration policy. The old quartic-point controls retain
their fixed GL24 native-key tiling; they are still required localization and
regional controls even though the new primary is Gaussian.

For matched isotropic, independently prepare original full C for that role.
Keep it as the draft adapter's `tau_cov` and construct geometry covariance C_iso=tr(C)/2 I.
Compute tau from the original C, using the identical frozen arithmetic. Passing
C_iso to the tau function is not an acceptable substitute: float32 trace
rounding can change tau. The R15 trace-only quartic-isotropic preparation
cannot simply be renamed. A minimal implementation retains original C32 and
forms C_iso in double from those same diagonal values; each execution pays
for its own statistics and validation checks. Never borrow primary's cached
C/tau during timing. Forward metadata and synthetic/numerical tests should
verify primary/iso effective-tau equality for every paired query/state/config.
Keep geometry covariance and tau covariance explicit in reference interfaces.

Matched constant T follows the unchanged Gaussian full1 content path with
the same h/T. This separates the temperature effect from geometry. It does
not use the trace rule merely because the preparation adapter is shared.

## Population, storage and immutable labels

Use exactly 16 old pairs each from `harvard_c8/hv_c8_3` and
`brown_bm_4/brown_bm_4`. These are reused development data. Each operator has
7 states×9 original partitions +5 zero-scale conditions =68 conditions:
**2176U records**, with the unchanged native token counts 432/108/243/48.
One pair has one sensor-visible regional population across operators/states;
center populations are fixed by pair and query partition, independent of
keys and perturbations. Do not filter by visibility in forward preparation.

Keep all 2176 primary float64 matrices and the two fixed first-pair samples
for each other unique operator (136 matrices each). Raw retained bytes remain
`1,112,516,352 + (U-1)*69,532,272`; retained array count is
`2176 + 136*(U-1)`. Recompute storage from literal shapes and U; leave old
evidence intact. All 2176U canonical hashes and numerical diagnostics are
committed before any VAL label/sensor bytes are hashed or decoded. Both
first pairs repeat all 68U matrices with fresh preparation and exact hashes.

The existing R15 `old_target_contract` is reusable in intent: metadata-only
R10 manifest/expected hash plus R12 label completion/provenance and independent
audit bindings. R12's scientific failure remains false. After the global seal,
`load_labels` verifies every original R10 field, not just the selected label
subset, and accepts only the three additional q24 fields in the R12 extension.
The independent audit reprojects native sensor centers for g8/g16/mixed/g24:
26,592 centers total. q24 is not a projected region mean. Reuse
`check_sensor_labels`, `center_reference`, `decode_sensor` and camera/source
hash bindings; forward perturbations must never alter these target centers.

Preserve row identity `(scene,pair,operator,scenario,condition)` and current
sum/count fields. Final population checks require all 32 pairs; per-scene
evaluation checks only its exact 16-pair shard. KL is visible-mass weighted
within scene/condition, then equally weighted over conditions and scenes.
PCK uses integer correct/valid counts with exact Fraction scene means and
the unchanged 1/200 boundary. No validation J selection is performed.

## Gates: reuse the existing mathematics, change only role bindings

The R15 production `gates` and its independent counterpart already express
the required intersection. Version their role validators and preserve:

| Evidence | Requirement |
|---|---|
| zero common9 vs each strong/legacy regional point | >=2% macro KL improvement; both VAL scenes strictly improve for each reference |
| zero vs each strong/legacy quad | >=2% KL gain with time<=1.25×, or KL<=1% worse with time<=.5× |
| each original zero partition | KL<=2% worse than each quad |
| zero g8→g8 and g16→g8 | PCK loss<=.5pp to each of the three localization references |
| five direct scale conditions | >=2% macro KL gain over each point reference; both scenes improve; every condition<=2% worse than each quad |
| zero g24→g8 | PCK loss<=.5pp to each of the three localization references |
| each of six stress states plus zero | common9 macro KL<=2% worse than each same-state quad and g8→g8 PCK loss<=.5pp to each same-state localizer |
| all predictions | finite, nonnegative, normalized within 1e-12; exact population and replay |

The q16 guard is the predeclared R15 extension, not an original R12 gate.
Stress per-condition KL is reported; the inherited stress hard rule is its
nine-condition macro. Do not add a new per-condition stress rule or replace
existing requirements with the weaker TRAIN screen. Matched attribution
results are reported without post-hoc promotion or changing a failed gate.

## Independent numerical coverage and fair cost

Keep independent reconstruction of **224 pair/state geometries**, **448
native/common target-table pairs**, complete row/alias/metric aggregation and
all retained arrays. Fixed numerical samples are both first VAL pairs, all
68 conditions and all U operators: **136U matrices / 1088U query rows**, eight
linspace queries and every native key. They need their own references: the
old first-TRAIN-pair preflight is not a substitute for VAL/stress numerics.

Reuse the R15 independent `reference_observations`, reference banks, literal
bounds, common maps and control integrations; add the independent R17 trace
arithmetic. For trace/iso, use Gaussian GL64/96 references plus a hybrid using
actual float32 support/content, with the original C for tau even when geometry
is isotropic. Preserve <=1e-4 actual/reference and hybrid error, <=1e-8
reference convergence, and <=1e-12 normalized probabilities. Include exact
tau32/bias32 evidence for the same prepared inputs. The quartic-point control
still needs its independent quartic reference. Record all tiny/FOV/floor and
argmax differences; none excuses a failed numerical tolerance.

The old audit independently measures all retained primary arrays and control
samples; other control rows have complete provenance/count/aggregate checks
and exact producer replay. For the R17 complete-replay contract, the proposed
adaptation extends the audit with the Stage B shared-operator replay pattern
for unsaved controls and independently measures **all 2176U matrices**.
That replay is explicitly not independent forward mathematics; independent
integration remains the fixed 136U sample. Make this scope explicit in the
new freeze rather than calling sampled integration an all-matrix proof.

Timing retains five old scenes' first metadata pairs×nine conditions×U:
45U cells, three warmups and 15 samples each, one CPU4-thread process with a
frozen rotating order. Require both full evaluation chains to finish first.
Each measured execution freshly prepares that role's moments, pooled query
and key content, key geometry, integral/floor, tau and its finite/PSD/symmetry
checks where applicable, content bias, softmax and common projection. The
full1 and iso controls pay for their own actual required computation. Use no
shared role caches, no precomputed tau/logits, no raw-integral shortcut and no
different configuration's historical timing. T0 executes its actual frozen
path. Take one full-time median per cell and then the mean of 45 medians.

Keep all timings positive/finite and source/config/scene/pair-bound. Preserve
common prediction/projection/export times, host load, tensor-byte and process
RSS scope separately. The gate measures incremental PE, not the full visual
pipeline. The model adapter must serve both forward and timing so timing
cannot silently omit newly added validation checks or original-C preparation.

## Freeze and stopping boundary

Before any VAL forward: accept the complete R17 prerequisite chain and screen;
finalize/tests all new producer/reference/metrics sources; lock role IDs,
precision/zero-path evidence, metadata-only cohorts, storage, cost and audit
coverage; freeze the independent audit and bind its lock. No conditional
primary changes, h/T adjustment, per-scale selection or tau variant follows
the first VAL outcome. Any failed scientific gate stops this unchanged
candidate; any audit/numerical failure preserves its evidence before repair.

Even a complete old-VAL pass does not satisfy actual learned V, optimized GS
RGB/NVS or untouched-scene confirmation. Those separately frozen inherited
requirements, the earlier failures, and the unfinished general research goal
remain explicit. In particular, a later actual-V implementation must apply
tau to learned content, not substitute a fixed correspondence matrix.
