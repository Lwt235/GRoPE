# R31 synthetic independent-reference protocol

2026-10-31. Implements docs/round31_log_conditional_reference_design.md.
Freeze joint sources/runtime and accepted predecessor identities before any
new synthetic NPZ decode. No real arrays, labels, efficacy or new model call.

Authenticate R30 root acceptance and its post-result selection, plus the R29
accepted numerical receipt, current and dual source snapshots, all producer
and audit inventories, all29 production records and all43 accepted old
reference records. Use exactly the23 base IDs in R29 BASE_IDS order. A source
descriptor binds the original producer archive and canonical four input
hashes, original production result, both old order32/48 oracles for20 bases
or the single analytic oracle for the three additional bases. No descriptor
or input byte may change. All six original identities remain inherited.

New runner and independent auditor freeze separately. After both freezes,
runner copies exactly the four synthetic input fields into23 prepared
archives and seals all preparation outcomes. The independent input stage
checks both old and prepared dtype/shape/content bytes against the original
arrays and descriptors. Seal all23 input checks before any reference call.
Runner then loads only these four prepared fields, executing base order and
order32 then48 for each base.
One shared600M node/1800s budget covers all46 outcomes, loads and record writes.
Per-call250M inner nodes,32768 depth evaluations and512 intervals include pilot
work. Budgets and numerical rules are as selected in the design; no retry.

Persist every scheduled call, start/result journal, attempted flag, resource
counts, failure reason and all available partial arrays. An unattempted call
after deadline is not a numerical evaluation. Successful positive-law raw
logs must all be finite; only exact structural-zero branches use -inf. Save
native log raw, direct-FOV log raw, once-floored log mass, log P and P, plus
variable-depth pilot scales and returned scaled integrals. Preserve pilot
counts, scaled quad_vec error, outer convergence, depth tail bound and local
exp-loss diagnostics. Structural-zero results must carry explicit flags.
Do not replace unresolved positive values with zero or floor-only outputs.

After global reference sealing, the independent auditor imports no reference
integrator or production model. It verifies complete ordered46-call inventory,
all source/input/output hashes and budget arithmetic, and recomputes saved
log-mass normalization, scale restoration, normalization and direct-FOV gates.
It now decodes the accepted old producer/oracle outputs for comparisons.
For each23 base, compare new32 vs new48 and EACH new reference vs BOTH old
accepted references (one analytic oracle for the additional three). Apply
unchanged strict reference P<=1e-8, raw-FOV relative and raw-L1/FOV<=1e-8.
Compare unchanged old production vs each new reference at P<=1e-4 and raw
relative/L1<=1e-6. Exact structural zeros alone have old raw-relative exemption.
Use log/expm1 errors for raw quantities; floored P never waives raw gates.
Normalization<=2e-12, direct-FOV absolute<=1e-8; depth-tail/raw<=1e-10 and
biased probability tail<=1e-8. Store all comparisons even if one fails.

An accepted numerical preflight requires all46 actual calls complete, all
input/delivery/arithmetic/comparison/budget gates pass, and all inherited
identities/source evidence remain valid. Integrity can be accepted for a
coherent failure. Always real_inputs_read=false, labels_read=false,
general_numerical_accuracy_established=false, efficacy_tested=false,
candidate_selected=false, scientific_gate_passed=false, goal_complete=false.
Track synthetic_inputs_read and independent_synthetic_reference explicitly;
do not call this an independent real probability reference. No oldTRAIN
effect, oldVAL, actual V, optimized GS or untouched confirmation is unlocked.
Report all development failures and preserve the first formal attempt.
