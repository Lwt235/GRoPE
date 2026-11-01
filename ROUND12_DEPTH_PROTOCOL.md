# Round 12 depth: actual V with the fixed four-Gaussian regional bias

Old development only. This protocol and all source/data identities must be
locked before any training. Training is authorized only after the separate
Round12 scale/perturbation gate passes. Implementation and synthetic tests do
not constitute a training result. No historical test observations, labels or
predictions are inputs, and no new confirmation data are opened.

## Single change and populations

Reuse Round7's exact 41,345-parameter one-attention/V/query-residual head,
initialization, input RGB features, RGB-predicted depth, value transformation,
query-center targets, valid-query mask and boundary convention. Replace only
the cross-view geometry logits. The shared identity-initialized 128x128 Q/K
projection remains learned; precomputed geometry contains no content score.
No gate, extra network, sensor source value, learned covariance, or extra block.

Training is Harvard c3/c5/c6, 192 pairs each (576 total); validation is Harvard
c8 and Brown bm4, 64 pairs each (128 total), loaded from the original Round7
development archives. All query-sensor-valid centers are included, even when
cross-view correspondence is absent, occluded or outside the source view.
No GT depth alignment or source-visibility loss mask. Cameras remain supplied
RGB-D reconstruction metadata; this is a conditional posed experiment.

Eight fusion roles: full4 (fixed primary), iso4_matched, full1, quad4, quad16,
point, point_pck, fixed. All inherit their frozen R10B/R11 h/T. Iso4_matched
uses full4 h/T. Geometry producer supplies float32 [N,432,432] pure positional
logits, using full4's fixed four strata or the corresponding control, real key
rectangle integrals, identical floor and every predicted pixel. Each geometry
manifest joins the exact original Round7 pair identities, RGB features and
input/label archive hashes; the producer never reads sensor labels.

A ninth learned role, singleview, uses query RGB features and query predicted
patch depths as keys/values. Its native square-grid geometry is the original
Round7 quadratic full operator with h=.5,T=5. No cross-source cache field is
required or allowed to influence this control. All 41,345 parameters are active.
Also report the original frozen query depth, without artificial seed replication.

## Matched training and selection

Five paired seeds 11,22,33,44,55. For every role: 1,000 steps, batch8, AdamW
lr=.001, weight_decay=.0001, gradient norm clip1, log-depth L1 over valid query
centers. The seed defines original NumPy default_rng integer batches with
replacement, shared across roles; save initialization and batch-sequence hashes.
No geometry/content h/T search and no role-specific budget extension.

Evaluate at steps 0,100,...,1000; select the earliest strictly minimum equal-
scene validation AbsRel. Preserve every validation summary and training loss.
Only one current role's geometry is loaded at a time. Geometry is fixed and may
be cached; learned Q/K/content, attention and V/head run during training.
Singleview receives the identical optimizer, seed and evaluation schedule.

Save the selected checkpoint, raw validation depth, per-pair additive metric
sums, full validation history, source/data/checkpoint/prediction hashes and
device/time/memory records. At the selected checkpoint evaluate zero-V and
reversed-key-V interventions; save their full raw depths and metric sums.
Record first-step per-parameter gradient norms and finite status, including
Q/K, V and head. T=0, if ever selected, is explicitly allowed to give zero Q/K
gradient and must not be presented as learned content use; current frozen
temperatures are read and reported, not silently overridden. Interventions
are diagnostics, never new contenders or checkpoint-selection criteria.

## Development gate, not confirmation

AbsRel uses valid-query sums within each scene/seed and equal scene/seed means.
Report MAE, RMSE (square root within scene), log RMSE, delta1, boundary AbsRel
and MAE, every scene/seed and raw baseline. Counts must match for all methods.
Strong position control is the lowest five-seed validation AbsRel among point,
point_pck, fixed, with this order breaking exact ties. Separately choose the
strongest compression control among full1,quad4,quad16 by five-seed validation
AbsRel, with that order breaking exact ties. A fast full1 can therefore defeat
the candidate; the comparator cannot be chosen because it runs slowly.
All controls remain in the report, including iso4_matched.

Full4 passes only if all of the following hold:

1. AbsRel is at least 2% lower than strong position control and strictly lower
   than the independently trained, same-capacity singleview.
2. Each comparison improves both validation-scene seed means and at least
   four of five seed scene-means.
3. Against the selected compression control, either AbsRel is at least 2%
   lower with incremental PE+actual V/head time <=1.25x, OR AbsRel is no more
   than 1% worse with that time <=0.5x. Report both branches. RMSE is no more
   than 2% worse than either the strong position or compression control.
4. Complete paired runs, source/data integrity, raw-prediction metric replay,
   valid/missing-depth behavior, initialization/batch parity and actual V/head
   paths pass audit. Every failed gate and intervention is retained.

Two repeatedly used validation scenes do not warrant population confidence
intervals. There is no bootstrap significance claim here. A pass only permits
planning a new locked multi-scene confirmation jointly with regional/scale/
localization/cost requirements; it does not erase the Round7 failed independent
depth confirmation or Round9 actual-GS RGB failure. A failure forbids promoting
this unchanged candidate on the strength of regional KL alone.

This is an explicit **pre-training revision** of the exploratory inventory's
draft. Requiring >2% task accuracy over quad16 and singleview was stricter than
the original R7 utility gate and ignored the candidate's declared compression
tradeoff. Before any Round12 training or depth score, the contract retains the
original strong-position utility threshold, requires actual singleview gain,
and adds the same accuracy/cost alternatives as R11 for the strongest selected
compression control. It does not change any historical result or R11 gate.

Timing is freshly paired on the first original pair of all five development
scenes and all five selected checkpoints, for all nine roles. Three warmups
and fifteen synchronized repeats per role/scene/seed; take medians then equal
scene/seed means. Start with identical already projected predicted pixel
positions/covariances, recompute support moments/strata, key rectangles and
integrals, cast to the actual float32 geometry cache precision, then execute
learned Q/K, softmax, V aggregation and the selected depth head. Singleview
recomputes its original native quadratic geometry and head. Do not substitute
cached geometry, operator-only timings or training-throughput ratios. Shared
RGB/depth predictors, dense pixel projection and transfers are outside this
incremental gate; record common CPU dense projection wall time on those same
five pairs and geometry-export total time separately. This is not whole vision
system end-to-end latency. Iso4_matched performance and time are reported but
it cannot replace the fixed primary after results are known.

## Integrity and execution boundaries

Freeze this protocol, model/trainer/analyzer and imported original helper
sources, geometry producer protocol/lock/completion/scene manifests, all role
geometry hashes, and original train/val input/label hashes. Source snapshots
are immutable and directories refuse overwrite. Depth freeze requires a passed
stress summary, source lock, completion-file chain and an independent audit
whose passed flag binds that exact stress lock/summary/completion SHA256.
The depth lock records these files and digests; training and timing reject
failed, absent or changed promotion evidence. Each run rechecks the lock
and used data; independent processes may train different roles/seeds without
sharing mutable output files. Analysis requires exactly nine roles times five
seeds and reconstructs all selected/intervention metrics from saved depths and
the original validation labels. Fresh test access is never authorized by this
development lock. GPU allocation is managed by the parent task before launch.
