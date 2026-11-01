# Current research state

Updated 2026-11-01 after the user resumed the existing research goal following
the review of rendering-task drift.

Objective: a compact, generally useful position encoding derived from 3DGS
parameterization, supported by matched strong controls, actual tasks and
untouched confirmation. No candidate has met the complete objective.

Latest scope clarification: R41/R42 rendering readout/loss results are task-specific
evidence. R43/R44 test PE inside the full two-block Transformer, but still only
in projected 2D coordinates. Rendering is optional downstream evidence; native
3D and cross-task PE utility remain unproven. The unfrozen R45 draft records
this boundary and the need to distinguish generic Gaussian/RBF effects from
benefits of learned orientation/scale. No historical result or criterion changed.

The user's route review exposed an unsupported dependency: success of an
untrained static region kernel had become a prerequisite to learning/GS tasks.
Early decoupled-shape synthetic gains are not established real-data success:
R2's shared-random-identity control substantially weakened that interpretation.
The R38 rank-one anchor correction vanishes for a true GS with a=mu, while the
underlying shape covariance remains nonzero. R13 documents region/center target
conflict but does not prove the finite tolerated empirical gates impossible.

Completed R40 experiment: `ROUND40_DECOUPLED_GPU_PROTOCOL.md`. Center3D RoPE plus one
scalar-weighted seven-dimensional CENTRAL covariance descriptor distance, with
center/expected/normalized, same-capacity center-distance, isotropic/shuffled
and all six R7 controls. Both views use the same predicted geometry. The primary
is fixed `decoupled`, not whichever ablation wins. 576/128 old TRAIN/VAL pairs,
13 roles x five paired seeds,1000steps. No fresh data authorization.

Evidence directory: `results/round40_decoupled_gpu/attempt1`. All65 GPU training
runs finished, both shards exited0. Independent full depth metric/identity
replay completed under `audit_impact/`: decoupled AbsRel .164532846 versus best
old point .161017277 (+2.1833%), center_rbf .161791281 (+1.6945%) and singleview
.162353858 (+1.3421%). Zero of five seeds beat any of those three controls.
Turning off shape at the fixed primary checkpoints slightly improves AbsRel.
No primary promotion and no result-driven frequency/temperature rescue.

The original audit stopped at a 2.1193e-5 P error versus2e-5 tolerance for a
point_pck sample. Retained it and used a separately recorded impact version;
only point_pck seeds11/44 warned, no winner changes, sampled depth change at
most1.399e-6m. Primary and all three decisive controls had no warning. See
`audit_impact/numeric_impact.json`; no retraining or prediction replacement.

All60 selected cross-view checkpoints completed old q24->k8 diagnosis at
`results/round40_region_diagnostic/attempt2`, with full metric replay. First
attempt's return-unpacking error and source/log were retained; corrected only
that interface. Primary KL2.11244/PCK44.579% versus full .544402/82.777%.
On773 target-consistent queries, pooled PCK47.684% versus full88.202%; failure
cannot mainly be blamed on center/region conflict. Full2304-query population
and1096center labels preserved. No source numerical warnings in region audit.

An additional warning about interpretation: point depth AbsRel improves when
V order is reversed (.1610173->.1598165). Do not equate lower residual-depth
error or nonzero gradients with correct spatial correspondence.

All R40 job handles are terminal. Complete report:
`docs/round40_gpu_research_report.md`. A matched short synthetic train benchmark
found CPU4 34.23ms/step versus GPU10.90ms/step (3.14x), excluding data/validation.
This is not the final cost gate or a general acceleration claim.

R41 actual-GS experiment IS NOW COMPLETED. Protocol:
`ROUND41_ACTUAL_GS_PROTOCOL.md`, evidence `results/round41_actual_gs/attempt1`.
Implemented direct RGB attention (no residual color head), actual R/S covariance,
context-only ResNet features/GS depth supports, same visibility/fallback as R9.
Scene-normalized units explicitly differ from metres; all Q/K retrained.
Eleven roles x five seeds x two scenes =110 runs,500steps, all predictions sealed
before target RGB. GPU4/5 shards exited0 in300.76/256.59s; peak837.875MiB.
Preflight4checks,CUDA smoke,110 independent score/RGB samples and full1328 frame
metric rows passed. Max P error8.93e-7,RGB5.04e-7,MSE replay9.72e-17; no warning.
All110 selected Q/K changed. Both main training handles14330/80742 are TERMINAL;
audit79756 and metrics92979 are also TERMINAL. Do not restart them.

R41 primary FAILED again: gate MSE .024874800 versus learned point .022275296
(+11.6699%), original simple point_h.25 .022241047 (+11.8419%) and original GS
.022219161 (+11.9520%),0/5 winning seeds for each comparison; both scenes worse.
Shape-off .024185743 improves2.7701%; reversed V .046824345 worsens88.2401%.
Learned full .022274579 is virtually tied with point, not meaningful success.
Complete report: `docs/round41_actual_gs_report.md`. R41 sources are frozen;
do not change them when implementing a successor.

R41 component diagnosis IS NOW COMPLETE. Queue91902 exited0 on GPU4; do not
restart it. Frozen ten-checkpoint, no-training scale-only / shape-only gate MSE:
.025186092 / .024093993. Shape-only improves3.13895% relative full but remains
8.16464% worse than learned point. Complete record:
`docs/round41_component_diagnosis.md`, `results/round41_shape_components/summary.json`.
Across481160 source observations (seed repetitions, not new scenes), full/off/
scale-only/shape-only mass on source-specific top-trace-decile keys is
.10625/.07394/.13373/.05347, and RMS projected-center displacement is
47.12/47.33/47.54/47.17px. Trace bins are not true uncertainty/edges; RMS is
outlier-sensitive. Raw scale similarity hurts these fixed models, but removing
it is insufficient. Max independent probability error9.63e-8, zero winner
changes. GPU full-context replay exactly matches saved selection MSE.

CPU context attempt2 also complete: full/off/scale-only/shape-only means
.02539809/.02502613/.02634465/.02451141. Waldo full shape helps center loss
(.0297972 vs off.0304115) but hurts image MSE (.0313877 vs off.0309577).
First CPU attempt stopped at too-tight1e-7 threshold; source/failure retained.
Maximum CPU/GPU MSE difference3.39e-7,72.5x below smallest shape-only/full effect;
retained numeric note, not a scientific failure. Both CPU handles terminal.

R42 matched objective control IS COMPLETE. Frozen protocol
`ROUND42_RENDER_LOSS_CONTROL.md`; full report `docs/round42_render_loss_report.md`.
Evidence `results/round42_render_loss/attempt2`:60runs,10seed shards, all training
process exits0. Three unchanged R41 models point/full/decoupled, center/image
losses,2oldscenes,5optimization seeds,100updates. Same multi-source fusion,
initialization, target schedule and four-context-image selection within R42.
Only within-round factorial isolates loss, not R41-to-R42 differences.

Attempt1 failed on Waldo view1 (one visible GS, zero jointly supported center).
Own Teatime process stopped; all sources/logs/three outputs retained, no metrics.
Attempt2 samples COMMON supported training targets for BOTH losses:
20Teatime/19Waldo. Original source pool, selection views, all8 evaluation views
and all pixels retained. Control is conditional on common supervision support.
Old handles25099/37946 terminal; freeze/preflight57578 and smoke95730 exited0.
CPU4 tests passed; GPU renderer directional derivative relative error1.72e-6.

Launcher43045 was stopped intentionally with exit143 AFTER nine launched shards
had exited0; pending Waldo44 was waiting on GPU0 occupied by other users.
Only that unstarted shard moved to idle GPU1; completion handle99399 exited0.
`launcher_reassignment.json` and `launcher_completion.json` preserve the chain.
All R42 training handles are TERMINAL, do not restart them. No live queue remains.
Audit8262 and metrics67324 exited0. All60 selected Q/K updated. Max independent
P/RGB errors8.80e-7/4.71e-7,zero winner changes/no warnings. All1088 frame metrics
replayed, max MSE difference9.72e-17. Peak allocated848.88MiB; sum per-run
training/selection wall time4373.40s, not exclusive GPU timing or elapsed time.

Gate means (point/full/decoupled):
center .022277209/.022275362/.024880107;
image  .022289946/.022193594/.024134251.
Image loss improves primary2.9978% vs center (5/5 paired seeds), but primary
still loses to image-point8.2742%,image-full8.7442%,old simple point8.5122%,
original GS8.6191%;0/5 winning seeds against EACH, both scenes worse.
Image zero_shape .023703017 improves1.7868%; reverseV .047161387 worsens95.4127%.
Image-vs-center closes29.14%/25.50% of primary-minus-point/full MSE gaps; this is
not a causal decomposition of R13 region/center conflict or all historical error.
Full improves only.4323% vs image-point, opposite scene directions, no promotion.
All10 image-primary best checkpoints are at100; convergence not established,
shape strength remains.953-.978. This is failure at the matched fixed budget,
not a theorem of impossibility under every budget.

DECISION: stop promoting/rescuing raw covariance-descriptor similarity. The
supervision mismatch contributes but does not remove the deficit; neither more
frequency/temperature sweeps nor renamed ablation winners are authorized as a
success claim. Keep the full scientific objective active. Fresh data untouched.

R43 COMPLETED TRANSFORMER SCREEN (negative; bounded diagnosis also complete): distinguish
using physical rendering covariance directly as matching support from using the
GS PARAMETERIZATION to construct a learnable PE. R42 tests the former's concrete
candidate, not the latter in general. Review learned frequency/basis alternatives
against direct priors before implementation. A grouped Gaussian frequency family
would need controls for fixed frequencies, isotropic/diagonal covariance, ordinary
learned affine position projection and free learned frequencies; its factorization
may just reparameterize existing learned RoPE. Do not claim novelty or positivity
from this sketch. Physical R/S utility and learned frequency R/S utility are
different claims, neither can silently substitute for the other. Any successor
must retain actual task tests and map applicability of prior geometry/robustness/
cost checks explicitly. No weakening final success criteria to fit a new branch.

The user explicitly restored active status after the pause. During the pause,
the root agent had mistakenly finished one R43 draft write, acknowledged it,
then stopped. That draft is untested, not evidence and not a frozen decision.
The inspected implementation passed four CPU preflight tests and all13 CUDA
toy roles (max independent P error2.071e-7, zero winner changes). Protocol,
sources and old data identities are frozen at
`results/round43_frequency/attempt1/lock.json`; its SHA256 is
83112dc07d8d521eddd409125a07c4662674959810d7236f3391f064e5f484a8.
This is a frequency-parameterization hypothesis with a fixed
two-block Transformer, not another RGB readout/renderer optimization. R5's
negative Transformer result remains valid for its particular formula. Physical
input-GS R/S utility and GS-style learned frequency parameterization are distinct
claims. Rendering is an optional downstream transfer test; it does not gate
all PE research. The R43 protocol records this clarification BEFORE results.
Initially all eight GPUs were occupied. After other jobs finished, a live query
verified GPUs1..5 empty (1..4MiB, no compute process). Five seed shards were then
launched:11/GPU1/handle90524,22/GPU2/65497,33/GPU3/59297,44/GPU4/72430,
55/GPU5/43086. Each executes13 matched roles,1000steps. Details and UUIDs are in
`execution_handles.json`; logs are `seed_*.log` in the attempt. No other process
was stopped. Re-poll these exact handles before concluding/restarting work.
All five training handles are now TERMINAL exit0, all65 runs completed.
Training audit handle77046 also exited0 and sealed all65 checkpoints. Maximum
independent sampled P error3.721e-7, no warnings or winner changes; matched
initialization, batches and earliest minimum-CE selection verified. All65
selected shared weights and all30 learned-frequency banks changed. Peak allocated
805.68MiB; summed per-run training/selection wall time3815.66s, not a throughput
benchmark. Source identities and the frozen experiment have not changed.
Old-evaluation exports launched AFTER this seal, one seed/GPU each:
11/99272,22/37154,33/4627,44/97316,55/52073; all now TERMINAL exit0.
Full metrics audit72668 exited0. All432 keys,110189 valid queries/model,480pairs
and primary/zero-V outputs retained. All62400 frame metric rows replayed; zero
PCK4/PCK8/valid-count differences. Max frame CE sum difference1.826e-4 and
argmax EPE sum difference.001359px do not change the PCK conclusion.

R43 main PCK8=.919973726 versus point=.923507882 (-.353416pp;1/5winning seeds,
0/5winning scenes). Against fixed_gaussian -.025840pp(2/5seeds),gaussian_iso
+.001762pp(2/5),group_affine -.033039pp(1/5),free_frequency +.003232pp(3/5).
All seven new frequency roles are within.0363pp of each other and below point.
Primary CE1.592955 versus point1.583625 (+.5892%). No promotion even without
appealing to the frozen1pp criterion; there is no positive point-control effect.
Old_eval zero-V primary PCK8=.617890309, supporting V-path use only.

One post-result bounded diagnostic was separately frozen in
`ROUND43_FREQUENCY_DIAGNOSTIC.md`, evidence `results/round43_frequency/diagnostic1`.
Only64old VAL pairs,20selected models (four roles x5seeds),50forward conditions,
no training: normal/zero_phase plus initial_bank for the two learned-bank roles.
Freeze33110 and run15696 both TERMINAL exit0. Full logits-to-NumPy metric
computation, normal replay PCK count differences0,maxCE frame sum diff5.724e-5.
Primary VAL normal/zero_phase/initial_bank PCK8:
.939920424/.485941645/.939219401. Free-frequency:
.940337249/.487286851/.940147783. The network clearly uses positional phases,
but restoring initial frequencies changes PCK8 only.0701pp/.0189pp at the fixed
weights. This is not a retrained ablation or an optimization-causality theorem.

Full report `docs/round43_frequency_report.md`; terminal handles recorded in
`results/round43_frequency/attempt1/execution_completion.json`. NO R43 job or
queue remains live. Do not restart any of the completed handles. Untouched
confirmation cohort remains unopened; the full research goal stays ACTIVE.

NEXT: retire promotion/tuning of this fixed global-frequency constraint. Review
the distinct, more direct hypothesis of learned SPATIAL Gaussian basis functions
as per-token positional vectors inside the same Transformer. Centers/covariances
would control local spatial responses. Compare ordinary RBF, learnable Fourier/
MLP and Gaussian Point Encoder prior work; no novelty or improvement presumed.
R44 is now COMPLETE and negative against the strongest point RoPE. Its local
advantages against additive-vector controls do not establish the full goal.

R44 protocol `ROUND44_SPATIAL_PE_PROTOCOL.md`, model
`models/round44_spatial_pe.py`, runner/auditor `experiments/round44_spatial_*.py`.
The preceding goal turn was concrete progress: completed and audited R43's65
training runs and50 diagnostic conditions, then retired its fixed candidate.
R44 tests spatial_full:36 learned Gaussian SPATIAL basis responses, one36x48
linear mixture, then normalize(h+p(x)) inside the unchanged two-block Transformer.
It is an application of known RBF/Gaussian Point Encoder bases, not a new basis
theorem. Learned precision is Cholesky SPD; covariance has GS R/S form. There
is no physical input-token covariance, renderer, RGB head or RoPE hybrid.

Eight new roles x5seeds=40new runs; six unchanged R5 roles'30 R43 runs/outputs
are reused with full identity hashes. New/old backbone initialization and1000
sampled batches must match; all new roles share projection initialization.
Added parameters full1908,diag1872,iso1836,fixed1728,fixed-centers1836,
Fourier-fixed1728,Fourier-learned1782,coordinate-MLP1836. A fixed native-grid RMS
normalizer matches initial basis energy. All Gaussian ablations start identically.
40 new checkpoints must be sealed before their old_eval export. Old datasets,
budget, selection and full query populations remain the same; fresh untouched.

Prospective utility criteria remain>=1pp versus strongest R5/Fourier/MLP controls,
>=4/5seeds and>=4/5scenes positive,CE<=2%worse. Component attribution is separate:
full versus fixed/iso/diag/fixed-centers is reported continuously, not forced to
independently earn an entire-method1pp margin. Any simplification requires a new
prospective experiment, never a relabeled ablation winner. This distinction was
recorded before R44 results; R43's frozen failure is unchanged.

Freeze/preflight handle40722 TERMINAL exit0. Five CPU tests passed, including
nontrivial independent NumPy formulas, all parameter counts/common initialization,
shape gradients at isotropy, input scope and 2D/3D exact Gaussian responses under
JOINT coordinate/basis transforms. This is not task robustness or3D task success.
Evidence `results/round44_spatial/attempt1`; lock SHA256
955125d409e88f410cad3c55542b9683c81a792355fcea65fcec4dc1427c5368.

GPU query initially found other processes on all eight devices. The bounded
controller `experiments/round44_queue.py` subsequently ran and is now TERMINAL
exit0 (tool handle55351; historical host PID121242). CUDA smoke exited0: all8 roles passed,
max sampled P error2.504e-7,zero winner changes. After GPUs freed, all five
training shards launched: seed11/GPU2/PID12779,22/GPU3/PID17975,
33/GPU1/PID19480,44/GPU4/PID19481,55/GPU5/PID22427. These are CHILD host PIDs,
not separate tool handles; the controller tool handle was55351. All40 runs,
training audit, allfive exports and full metrics are now complete. All13 child
exits0,active list empty,parent exit0 directly observed. Queue source, GPU UUIDs,
actual exits and logs are retained at `queue_status.json`/`queue.log`. No R44
queue/child remains active; do not restart or stop these historical PIDs.

R44 main PCK8=.898325531 versus point=.923507882 (-2.518235pp),0/5winning
seeds and0/5winning scenes; CE1.714961 versus1.583625 (+8.2934%). It exceeds
the additive Fourier-fixed/Fourier-learned/MLP controls by6.48896/5.22749/
1.64427pp,each5/5seeds and5/5scenes. Those local comparisons pass, but the
strong point comparison does not: no overall promotion. Component differences
full-minus-fixed/iso/diag/fixed-centers are +.010685/-.048260/+.036948/+.003293pp;
no stable evidence that learning full centers/ellipsoids drives the gains.
The equal-parameter isotropic role also exceeds the MLP, so a tiny parameter
count difference does not explain all observed local advantage.

All40 selected shared/projector/basis parameters updated.29/40 best checkpoints
are at1000steps: convergence is not established. New sampled probability error
max4.312e-7,zero winner changes/no warnings. All67200 frame metric rows replayed,
zero PCK4/PCK8/valid-count discrepancies. Peak664.12MiB; summed per-run
training/selection wall1716.81s, not a matched throughput benchmark.
Full report: `docs/round44_spatial_report.md`. Full goal remains ACTIVE and
unmet; the unused confirmation cohort is untouched.

The next-design review in `docs/round45_phase_warp_draft.md` has now become
the frozen R45 experiment (live status below). It uses u(x)=x+sum v_g*Gaussian_g(x), then original RoPE at u(x),
with v=0 recovering the strong original point baseline exactly. This tests
local Gaussian coordinate maps in a unitary PE instead of R44's additive
normalized vectors, not a width/budget sweep or a renamed ablation winner.
Required strong controls: ordinary affine/free-frequency and MLP/Fourier maps,
all sharing point initialization, plus Gaussian shape/center simplifications.
Review direct prior work before freezing. First-step zero geometry gradients
are expected at v=0; verify nonzero v gradient and subsequent geometry gradient.
No task result yet supports this candidate, and it does not waive3D/cross-task, robustness,
cost or untouched confirmation requirements.

Training-time theoretical scope review is in `docs/round43_scope_note.md`.
A no-data/no-checkpoint 2D/3D counterexample
(`results/round43_frequency/frame_diagnostic.json`) shows that rotating only
frequency moments and rebuilding a canonical six-point Cholesky template is
not the same as rotating all finite frequencies. Analytic Gaussian and explicit
frequency transforms agree at~4e-16, whereas the reconstructed mean complex
phase differs by .770/.857 in the fixed examples despite correct first/second
moments. This limits automatic frame-equivariance claims; it is not a task PCK
effect or a new pre-training gate. No frozen source, template or method changed.

Related-work note `docs/round41_related_work_note.md` now includes 3D Gaussian
Point Encoders (2026), Elliptical Attention(2024), Learnable Fourier Features(2021),
and HARoPE(2026). Existing `models/gaussian_quadratic_pe.py` already implements
five-channel Gaussian attention; converting it to nine channels is not itself a
new design. Review bounded/finite-frequency limitations before more GPU runs.

Static region gates do not block these development mechanism experiments.
Actual Transformer task, full applicable scale/robustness/cost, attribution of
the proposed parameterization and untouched joint confirmation remain required
for the final claim. If claiming physical input-GS R/S utility, establish that
separately. Changes to
final acceptance standards must be explicit and prospective, not silent.

The old `round40_half_ray_pe.py` and its numeric auditor are unexecuted reserves;
they are not the active R40 definition. R38/R39 failures remain unchanged. Do not
edit README or other historical bound artifacts just to update current status.

R45 is now FROZEN and RUNNING: `ROUND45_PHASE_WARP_PROTOCOL.md`,
`models/round45_phase_warp.py`, `experiments/round45_phase_warp_{train,audit}.py`.
This goal turn implements the preceding scope clarification rather than
repeating a status-only plan. The scientific question is Gaussian coordinate
maps inside the complete Transformer, with exact original-point initialization.
Eight new roles x5seeds=40 runs, plus30 identity-checked R43 legacy controls.
Added parameters full252/iso180/fixed72/fixed-centers180/Fourier126/MLP180/
affine4/free-frequency48. Same data,1000updates,paired initialization/batches,
selection and primary criteria as specified prospectively. No renderer or fresh
cohort access. The 2D screen does not prove3D or cross-task effectiveness.

Formula-level literature review found that HARoPE v1 acts on head-dimensional
Q/K CONTENT (R_x A_h q), not a2x2 coordinate warp. Our affine control is NOT its
reproduction. Gaussian/RBF bases and identity initialization already have
precedents; no new basis theorem or absence-of-prior-work claim is made.

Attempt1 retained: preflight59612 TERMINAL exit1, five of six tests passed,
one auditor role-dispatch typo caused KeyError before any training/real array
decoding. Corrected only dispatch and output directory; source/lock/failure
record retained. Attempt2 preflight62083 TERMINAL exit0, all6 tests passed,
including exact full-network identity for all40 role/seed combinations, nonzero
independent phase formulas, unit norms, mapped relative phase, gradient flow,
affine-frequency equivalence and input scope. CUDA smoke also exited0.
Active evidence `results/round45_phase_warp/attempt2`, lock SHA256
f1276ec5e145fbcda1f19dba5979329e1c619a0db9c57ae6e74d50cc61c5426c.
Bounded queue tool handle54656, host PID11281, source and actual child exits in
queue_source.py/queue_status.json/queue.log. Re-poll THIS handle before any
conclusion/restart. All further stages are ordered by the same controller:
training -> checkpoint seal -> old_eval export -> full metric replay.
The full goal remains ACTIVE and unmet; no R45 task outcome is asserted yet.
