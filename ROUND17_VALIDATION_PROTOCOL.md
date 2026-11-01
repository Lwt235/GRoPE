# R17: fixed old-validation regression, complete replay and paired cost

2026-10-30. Declared after accepted R17 TRAIN calibration, before any new R17
VAL prediction or effect. This is reused development validation, not independent
confirmation. The controlling candidate remains
[ROUND17_CANDIDATE_PROTOCOL.md](ROUND17_CANDIDATE_PROTOCOL.md). Its original
scientific rules, historical failures, actual-V/GS requirements and untouched
confirmation boundary remain unchanged.

## Prerequisites and one immutable primary

Bind the complete accepted numerical preflight, arithmetic audit, Stage A,
Stage B source/input/label/completion/selection chains and accepted independent
audits. Stage B selection SHA256 is
`eadfd5071378c7239676ce58a0364ec2cb44c80ea3f198be800819691a700054`.
Its primary is **trace_full, h=.25, T=10, grid index 2**. The TRAIN screen is
passed; audit correctness alone would not permit this stage. No validation
selection, per-scene or per-scale parameters, fallback winner or replacement
primary is permitted.

The accepted regional audit includes an explicitly later historical-path
repair. Bind its lock `f8f1618ef075c82de31236c2d658635f89e0a4bb02b688ad974b372f60e15df9`,
source snapshots, original failure logs, nested verification and both completion
chains. R12 actually copied only three specified snapshots; all original source
hashes are still checked. This engineering repair was not a pre-forward freeze.
It changed no operator, population, numeric tolerance, metric or effect gate.

The final producer and independent auditor, all transitive local dependencies,
tests and this protocol are frozen before the first new VAL forward. A separate
independent-audit lock must name the exact producer lock. Any engineering
repair preserves the failed run and has a new explicit source/timestamp record.
No numerical tolerance can be relaxed in response to observed effects.

## Roles, populations and numerical paths

Use 19 semantic aliases in this fixed order: primary; seven regional controls
(quartic_point, gaussian_point, gaussian_fixed, gaussian_full1, gaussian_full4,
gaussian_quad4, gaussian_quad16); Gaussian and quartic point localizers; seven
legacy roles (point, point_pck, fixed, full1, full4, quad4, quad16); matched
constant temperature; matched isotropic. Exact complete family/h/T identity
permits first-occurrence alias sharing. Family identity and executed checks
remain part of the identity even if particular probabilities coincide.

The accepted manifest yields 15 unique operators:

| ID | Family | h | T |
|---|---|---:|---:|
| o00 | trace_full | .25 | 10 |
| o01 | quartic_point | 1 | 5 |
| o02 | gaussian_point | 1 | 5 |
| o03 | gaussian_fixed | .5 | 5 |
| o04 | gaussian_full1 | .25 | 0 |
| o05 | gaussian_full4 | .25 | 0 |
| o06 | gaussian_quad4 | .5 | 0 |
| o07 | gaussian_quad16 | .5 | 5 |
| o08 | gaussian_point | .25 | 5 |
| o09 | quartic_point | .25 | 5 |
| o10 | gaussian_point | .5 | 5 |
| o11 | gaussian_full4 | .5 | 5 |
| o12 | gaussian_quad4 | .5 | 5 |
| o13 | gaussian_full1 | .25 | 10 |
| o14 | trace_isotropic_matched | .25 | 10 |

Strong regional point and legacy point both refer to o02; strong quad and
legacy quad16 both refer to o07. Keep each named comparison. The three point
localizers are o08, o09 and o10. o04 is the old independently calibrated full1;
o13 is the separate matched constant-T attribution. o14 independently computes
the original full C32, retains it for tau, and uses double tr(C)/2 I only for
Gaussian geometry. Its tau must equal primary's tau bitwise for identical
query/state inputs. Neither attribution control is calibrated or promoted.

Use exactly 32 original VAL pairs, 16 each from Harvard c8 and Brown bm4.
Every operator has 7 states × 9 original partition conditions plus 5 zero-state
24px scale conditions, hence 68 conditions per pair, **32,640 total matrices**.
States are zero, depth×.9/1.1, target Ry±1 degree and tx±.01m with unchanged
R12 transforms, fallback/clipping and units. No stress combinations. Native
counts g8/g16/mixed/g24 remain 432/108/243/48; all forward rows are retained.

Forward reads only the ten declared predicted geometry/content/camera fields.
Sensors, visibility labels and histogram/count targets cannot enter prediction.
The zero reprojected float32 fields must exactly equal cached fields on the
three original first TRAIN pairs before final freeze. Record all four field
hashes and old double reprojection errors, not merely an allclose assertion.

Primary's full Gaussian GL32 geometry and native24 four-12px subdivisions are
unchanged. Add the original native area floor exactly once. Tau is the frozen
R17 rescaled double calculation from original C32, cast to float32, multiplied
by content32 in float32, then added to geometry64 before softmax64. All actual
finite/symmetry/PSD checks remain in the operator and its time. Gaussian
controls keep their original path; quartic-point controls keep GL24 tiled
native-cell integration. No extra key covariance, clipping, new head or readout.

## Storage, execution ordering and full audit

Commit all 32,640 canonical probability hashes and native argmax hashes before
any VAL label/sensor bytes are hashed or decoded. Save all 2,176 primary
matrices and 136 matrices for each other operator from the fixed first pair
of both scenes: **4,080 retained arrays, 2,085,968,160 raw bytes**. Both first
pairs repeat every 68×15 matrix with fresh preparation and bitwise hashes.
Check disk capacity; preserve old evidence. Hash-seal both scenes globally
before evaluation. Read/hash primary arrays and regenerate every control
bitwise before computing its metrics.

Bind immutable R10 labels and R12 extended center-label archives by their
existing audited provenance. Check equality of all original label fields;
the only extension is the three q24 center fields. Independent sensor centers
remain unperturbed and are reprojected independently: 26,592 centers across
g8/g16/mixed/g24, not projected region means. All 224 pair/state forward
geometries and 448 unique native/common count-table pairs are independently
reconstructed. Regional mass and center-valid populations must not change
with an operator, state or key partition.

The independent audit checks the full source/chronology/population, every
retained array and **all 32,640 matrices** through bitwise shared-production
replay and independently computed targets, KL/JS and integer PCK. It does not
claim that shared replay is an independent forward implementation. It reduces
both saved and independently recomputed rows and compares all summaries,
decisions and exact Fraction PCK gates. Per-row metric-sum tolerance is 1e-8;
aggregate comparison tolerance is 1e-10. Tolerance does not modify effect gates.

Independent numerical integration covers both first VAL pairs, all 68
conditions, 15 operators, eight fixed evenly spaced query indices, and all
native keys: **2,040 matrices / 16,320 query rows**. It separately reconstructs
NumPy support/content and geometry. Trace/full-isotropic use Gaussian GL64/96
and an independent hybrid with actual prepared support/content. Require
actual/reference and hybrid probability error ≤1e-4, reference convergence
≤1e-8, same-raw-mass probability arithmetic error ≤1e-12, exact prepared
tau32/bias32, and normalized finite nonnegative probabilities within 1e-12.
Old Gaussian/quartic controls retain their independent integration references.
All zero/tiny FOV and argmax discrepancies are recorded without removing cases.

The prepared actual support/raw/tau/bias values for these checks are recomputed
from the frozen producer source and bound to saved full-probability hashes;
they are not represented as extra tensors saved by the original forward.
Independent NumPy math stays separate. Full primary/isotropic tau attribution
is also checked across all pair/state/query groups, not just numerical samples.

## Unchanged scientific intersection

Within scene/condition, use visible-pixel-mass-weighted KL; then equally average
conditions and scenes. Native probability argmax with first-index tie breaking
serves PCK@8. PCK uses integer counts and exact equal-scene Fraction means with
the 1/200 boundary, no epsilon. Invalid centers retain their historical masks.

| Check | Requirement |
|---|---|
| zero common9 vs each strong/legacy regional point | ≥2% macro KL improvement; both scenes strictly improve for each |
| zero vs each strong/legacy quad | ≥2% KL gain with time≤1.25× OR KL≤1% worse with time≤.5× |
| each of nine zero conditions | KL≤2% worse than each quad |
| zero g8→g8 and g16→g8 PCK | loss≤.5pp against each of three localizers |
| five-scale direct KL vs each regional point | ≥2% macro gain; both scenes improve |
| each of five scale conditions | KL≤2% worse than each quad |
| zero g24→g8 PCK | loss≤.5pp against each of three localizers |
| each of six stresses plus zero | common9 macro KL≤2% worse than each same-state quad; g8→g8 PCK loss≤.5pp vs each same-state localizer |
| every matrix | finite, nonnegative, normalized and complete population |

q16 protection is the predeclared R15 extension, not an original R12 rule.
Stress per-condition metrics are reported; the inherited stress hard rule is
the nine-condition macro. Preserve all failures and do not add or remove gates
after viewing the result.

## Complete incremental cost and remaining research

Time five old scenes' first metadata pair × nine conditions × 15 operators:
675 cells, 3 warmups and 15 measured repeats each, one CPU4-thread process with
fixed cyclic role order. Wait until both evaluation shards finish. Each timed
call freshly prepares its own moments, query/key pooled content, key geometry,
integrals/floor, tau and all executed checks, content bias, softmax and common
projection. No borrowed primary C/tau, cached logits or pure-integral shortcut.
The gate uses mean of the 45 complete-execution cell medians for each operator.

Report shared projection/export time, host load and whole-process high-water
RSS separately. The existing prepared_tensor_bytes field sums named tensor
field sizes and may count aliased storage twice (notably C/tau_cov); it is not
an allocated-memory measurement. Process RSS is not a per-role allocator peak.
These are incremental PE times, not complete visual-system or training times.

Accept audit correctness separately from the scientific intersection. Any
scientific failure stops this unchanged candidate without reselecting from the
12 TRAIN configurations or tuning on VAL. A complete pass permits preparation
of separately frozen actual learned-V and optimized-GS/RGB experiments; it does
not bypass the failed old R12 training gate, satisfy the general research goal,
or authorize fresh confirmation. Those remaining requirements are explicit in
the candidate protocol and the downstream implementation note.
