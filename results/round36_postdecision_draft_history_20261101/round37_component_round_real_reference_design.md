# R37: complete real-input verification with component-wise roundoff accounting

2026-11-01. Selected after R36's accepted fixed synthetic PASS: all46
references and all339 comparisons passed. R35 remains a real numerical
FAIL (21 complete,1 rounding failure,122 skipped); its failed partial is
not an oracle. The subsequent prospectively frozen, geometry-only all72
initial-cost diagnostic found no necessary node-budget counterexample.
Its result is unknown as to actual feasibility and accuracy, not a PASS.

Reuse all72 original accepted R30 prepared inputs, in the unchanged R35
order. Copy the four original float64 arrays mu[3],factor[3,3],bounds[432,4],
content_bias[432] byte-for-byte and retain all seven query identity fields.
Keep the homogeneous mapping, pixel boundaries, zero bias,h=.25,floor1e-30.
Do not reconstruct inputs, choose anchor/content rules, rerun production,
read labels or access fresh scenes.

Use unchanged `experiments/round36_component_round_reference.py` and its
unchanged component-round quadrature helper at orders32/48:144 calls.
Keep the real-phase budgets of R33/R35:250M inner nodes/call,36B/phase,
10800s/phase,32768 depth evaluations/call,512 active outer intervals/call.
Full finite X, width2 panels, SVD, narrow GL8/span.25, coalescing, fixed
pilots, depth cutoff18, GK21 and epsabs0/epsrel1e-11,1e-12 are unchanged.
Keep R36 current-leaf component fsum, active scalar truncation plus maximum
historical component-roundoff, strict E<target/8, initial stopping check,
largest floored-error/leftmost-tie refinement and complete trace retention.
No accuracy/resource gate is relaxed; these estimates are not rigorous
floating-point or whole-pipeline accuracy certificates.

Keep fail-fast on the first failed scheduled outcome: reference exception,
resource/gate failure, input-decode failure or pre-call phase deadline.
Retain the trigger and all144 scheduled positions. Unattempted suffixes
have explicit cause, no input read/call start, zero actual work and empty
archives. IO failures are delivery failures, not probability counterexamples.
Do not retry failed inputs, change budgets or stop retrospectively based
on post-seal comparisons.

All original216 comparisons remain:72 new32/new48 and144 sealed R30
production/new-reference. Retain the19 accepted R33 complete references
and add all21 accepted R35 complete references, including their duplicate
query/order identities under distinct source keys. The40 old outputs yield
80 strict reference comparisons (each with both new orders), for296 total.
R33 contributes its first19 scheduled successes; R35 contributes its first21.
Bind complete+attempted+passed and independent delivery/arithmetic acceptance,
the four input hashes, seven identity fields and every old output hash.
Do not use either round's failed partial or skipped outcomes as an oracle.
Missing results yield unavailable failed comparisons, never omitted rows.

All reference comparisons use P/raw-FOV/raw-L1 relative1e-8; production
uses P1e-4 and raw relative1e-6. Normalization,direct-FOV,tail bounds,input,
arithmetic,resource,GL8 and R36 trace checks remain required. Six old
identities and finite-epsilon evidence are explicitly inherited, not rerun.
These old development regressions are not independent new populations.

Jointly freeze complete producer/auditor source closure,runtime,query and
oracle specifications before decoding new prepared arrays. Seal the exact
72-input independent audit before the first integration. Decode sealed
production and all40 old reference outputs only after the global144-outcome
seal. The auditor imports no model/integrator and independently recomputes
saved arithmetic and R36 trace, including all historical roundoff entries,
active leaves, refinement/stop ordering, attempted callbacks and partials.
Preserve all development attempts and their prior source snapshots. Root
acceptance follows complete delivery and saved-arithmetic replay.

Numerical PASS requires all144 complete,all296 comparisons and every other
required gate. It does not establish PE efficacy, general parameter-domain
accuracy or repair the preserved R35 result. Full2304-query old-TRAIN,
center/region and perturbation checks, actual V and GS tasks, and later
untouched-data confirmation remain mandatory for the scientific goal.
