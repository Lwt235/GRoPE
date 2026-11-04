# R35: complete real-input verification with the accepted GL8 reference

2026-11-01. Selected after the accepted R34 synthetic PASS: all46 new
reference calls and all247 comparisons passed. The new reference changes
only narrow normal GL16 to GL8, with its local exact-arithmetic bound and
unchanged accuracy/resource gates. The R33 real preflight failure remains
retained:19 complete,1 resource failure,124 fail-fast skips. No scientific
candidate has passed the earlier efficacy and generality requirements.

Reuse all72 accepted R30 prepared inputs in the same R33 order, including
all formerly failed/skipped cases. Copy exactly the four float64 arrays
mu[3],factor[3,3],bounds[432,4],content_bias[432]. Preserve the seven query
identity fields, original homogeneous mapping and pixel-boundary convention,
zero bias,h=.25 and floor1e-30. Do not reconstruct geometry, choose an anchor
or content rule, rerun production, decode labels or access fresh scenes.

Use unchanged `experiments/round34_log_narrow_reference.py` at orders32/48,
144 calls. Keep all R33 real-phase budgets:250M nodes/call,36B nodes/phase,
10800s/phase,32768 depth evaluations/call,512 outer intervals/call. Keep
full finite X, original GL32/48 X orders, span.25 narrow GL8, SVD, coalescing,
fixed pilot scales, depth cutoff18, outer GK21/tolerances and all other rules.
The local GL8 bound excludes floating-point and adaptive-outer error.

Keep R33 fail-fast on the first failed scheduled outcome, including reference
exception/resource/gate failure, input-decode failure, or pre-call phase
deadline. Preserve the trigger plus all144 outcome positions, including
unattempted suffix rows with explicit cause, no input read/call start,
zero actual work and empty archives. IO failure is a delivery failure,
not a measured probability counterexample. Do not stop retrospectively on
post-seal comparisons, retry failed cases or increase budgets after results.

Retain all original216 comparisons:72 new32/new48 and144 sealed R30
production/new-reference. Add exactly38 strict regression comparisons
against the19 complete, accepted R33 reference outputs. They are precisely
the first19 R33 scheduled calls: nine inputs with both old orders and the
tenth with only old32. Each old output is compared with both new orders.
Bind all old input/output hashes and seven query identity fields to the
fixed72 specs. Require completed+attempted+passed plus independent delivery
and arithmetic acceptance. Never use the failed old48 partial or any skipped
archive as an oracle. Oracle keys include source identity and order.

The full comparison denominator is254, reported as original216 plus added38.
Reference comparisons, including the added38, use P/raw-FOV/raw-L1 relative
1e-8. Production comparisons retain P1e-4 and raw relative1e-6. Normalization,
direct-FOV, truncation, all input/arithmetic/resource checks and explicitly
inherited six identities/finite-epsilon evidence remain required. Missing
outputs produce unavailable failed comparison rows, never zeros or omissions.
The known R33 prefix is development regression material, not a new independent
population or untouched confirmation. Both comparison groups must pass.

Jointly freeze complete producer/auditor source closure, runtime and all
metadata-only query/oracle specs before decoding any new input arrays.
The independent exact72-input audit must seal before the first reference
call. Decode old R30 production and the19 old R33 reference outputs only
after the global144-outcome seal. The auditor imports no integrator/model;
it recomputes saved arithmetic and GL8 local-bound/work checks independently.
Preserve development attempts and source bytes before execution, then accept
formal PASS or coherent FAIL only after delivery stages and root replay.

Only all144 complete, all254 comparisons and every other required gate can
pass this fixed real numerical preflight. No result here establishes PE
efficacy or general parameter-domain accuracy. Full2304-query old-TRAIN
screening, center/region and perturbation regressions, actual V and GS tasks,
and a later untouched-data confirmation remain mandatory before the research
goal can be achieved. The compact 3D anchor ideas remain unselected here.
