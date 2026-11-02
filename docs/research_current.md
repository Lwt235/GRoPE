# Current research state

Latest 2026-11-02: user resumed the full research goal; ACTIVE and UNACHIEVED.
R61 is COMPLETE, including all fits, metrics, actual visual review and Chinese
interpretation. Read docs/round61_normal_relation_report.md and its completion
record. Previous experimental goal turn was PROGRESS (R61 terminal evaluation);
the intervening discussion was planning only. Current turn is PROGRESS (R61
closure and preparation of the explicitly chosen RGB bridge). Do not restart
any R61 or earlier controller. No R62 experiment has been launched yet.

R61 root: results/round61_normal_relations/attempt1; 32 unchanged sources; lock
 de370b0d994d7d045b26eeb9d21fc38b4a524c411623337ba141130b2cb02027.
Controller92910/PID6062 exited0 at ab71da; watcher10962 exited0 at2d3b33.
All20children exit0. Six new source fits and six new real adaptations;15source
and15target models reused.48new independently recomputed prediction groups plus
120exact reused groups.12new selections precede all new evaluation exports.
Five implementation tests passed;metric maxerror1.78e-15;both report1 PNGs viewed.

R61 decision: retain the exact compact11->96->96 relation MLP. Real-adapted
region improvements over normal+spectra are+1.334/+2.572pp(base/sparse),3/3seeds
and5/5scene means positive. Versus axial:+.791/+1.174pp,only2/3seeds each.
Synthetic low-overlap vsnormal/axial:+1.457/+.947pp,3/3region seeds positive.
Synthetic sparse vsaxial:region-.419pp;center-.466pp withall3seeds worse.
All three full/normal/axial source-direct real averages remain belowdistanceMLP.
No universal triaxial necessity,source-transfer,actualGS or fresh-confirmation claim.
Target models start from own source weights; sparse eligibility populations change.

NEXT: user explicitly chose to reconnect the candidate to RGB before prioritizing
actualGS object classification. Test identical RGB matching task/features/cameras:
direct learned PE,RGB-predicted Gaussian relations,and matched PRoPE/calibrated
HRoPE controls. Observed geometry is PRIVILEGED diagnostic input,not a fair RGB
competitor. Preserve known camera relations equally in the relevant comparisons.
If observed works but predicted fails,investigate prediction/interface adaptation;
do not infer RGB prediction quality from cross-task scores. Do not expand local
point-cloud attribution indefinitely. Fix hypothesis,competing explanations,input
contracts,controls,budget and decisions before launching suitable idle GPUs.
Full original geometry,robustness,generality,actual-task/primitive and untouched
confirmation requirements remain. R55/R57/R58/R59/R60 negatives remain authoritative.
The following entries are HISTORICAL and do not supersede this latest status.

R60 completed background (authoritative outcomes retained):
Latest2026-11-02:R60 is COMPLETE; the full research goal remains ACTIVE and
UNACHIEVED. Read docs/round60_real_scene_report.md and the completion record in
results/round60_real_scene/attempt1. Current turn was PROGRESS:all30 target fits,
120evaluations,independent metrics,actual visual review and scientific interpretation
completed. PRoPE branches(3)/(4) were re-read;their latest messages do not undo
the completed R54/R55 comparison. Do not restart any completed controller.

R60 lock:fb24317caebd29419e117b6f248cd3b1e7eb3d7aab53ab070569ff2873ea0639;
24 scientific sources unchanged. Controller69037/hostPID10878 exited0 at46fc26;
report watcher4685 exited0 at14eb25. All38 child jobs exited0. Four implementation
tests and bounded pose/input/stream/selection checks passed. Independent metric
maxerror1.33e-15. Both report1 PNGs actually viewed during this review.

Same10 R59 roles x seeds11/22/33.30 exact source models plus30 NEW target fits;
60epochs,1080updates each. Target starts from each role's OWN source checkpoint;
initial target backbones differ across roles. Compare complete source-plus-target
pipelines,not a target-only PE reset. New target fitting/selection cumulative
1323.42seconds on shared host,not a controlled speed benchmark.
Old native SUN3D3/1/5scene split,576/64/480pairs. Sensor depth is PUBLIC INPUT
for this task;pose only supplies labels/scoring. No RGB-only/optimized3DGS/full
registration claim. Pose-independent centering/shared RMS scale;64regions/32NN.

## R60 evidence and immediate next step

Both source transfer and fixed target adaptation were planned before outcomes.
All30 target selections precede all exports.120groups,all5/5scenes evaluable.
Scene-macro regionR@1 by source-base/source-sparse/adapted-base/adapted-sparse:
candidate35.629/40.852/46.280/48.782;
distanceMLP36.742/42.545/44.469/46.700;
descriptorMLP33.384/39.290/42.583/45.853;
angleDescriptor33.511/39.348/42.737/46.022;
matrixMLP33.215/39.799/40.329/45.002.

Source candidate versus distanceMLP:-1.113pp base,-1.693pp sparse;all3sparse
region/softEPE/CE comparisons worse. Retain this transfer limitation.
Adapted base versus distanceMLP:+1.810pp,paired+4.172/-.254/+1.512;5/5scene means
positive. Center+1.135pp,2/3seeds;softEPE-.020704/CE-.068383,3/3seeds and5/5scenes.
Versus descriptor/angleDescriptor/matrix region+3.696/+3.542/+5.951pp;all4metrics
3/3seeds and5/5scenes better. Adapted sparse versus distanceMLP:+2.081pp,paired
+2.359/+.907/+2.978;5/5scene means better,center+.451pp. Sparse region versus all
three descriptor/matrix controls also3/3seeds and5/5scenes better;other metrics
have some scene/seed exceptions documented in the report.

Eligible queries:base20422/61440=33.239%,25/480zero-validpairs;
sparse8811/61440=14.341%,117/480zero-validpairs. No native-input exclusions.
Sparse changes supports/query populations;higher absolute sparse recall does
not establish fixed-query robustness. Units are normalized scene units,not metres
or pixels. Old scenes and reused candidate models are not fresh confirmation.

DECISION:retain exact11->96->96 invariantMLP as promising DEVELOPMENT candidate;
no general transfer/uniqueGaussian/novelty/global success claim. Current controls
do not isolate known normal-PPF relations. Read docs/round60_relation_literature_note.md
and docs/round61_normal_relation_plan.md. RoITr already uses normal point-pair
features;the current center-triplet angle control is not that control.

Next plan adds only two same-backbone controls:distance+spectrum+unsigned normal
PPF,and original spectrum with axisymmetrized directional covariance relations.
Six new source fits plus six fixed real-target adaptations;reuse existing exact
references. Keep source direct transfer separate from adaptation. No new module
stack,no additional pose input,no post-result budget rescue. R61 is NOT yet
implemented/frozen/launched. Fix concrete sources before launching on idle GPUs.
Original cross-task/actual-primitive/generalization/robustness/fresh-confirmation
objective and R55/R57/R58/R59 negatives remain. No reserved data access.

R59 completed background (authoritative findings retained):
R59 is COMPLETE. Read docs/round59_relation_attribution_report.md
and results/round59_relation_attribution/attempt1/completion.json. The full
research goal is ACTIVE and UNACHIEVED. Previous goal turn was PROGRESS (report
watcher/execution metadata and live queue); this turn is PROGRESS (remaining
fits,30 selections,120 main+30 diagnostic groups,independent metrics,actual
visual review and scientific interpretation). Do not restart completed jobs.

Controller40218/hostPID56148 exited0 atf5d1f6;report watcher42158 exited0 atbb31a1.
All18children exited0.15 frozen scientific sources unchanged;lock:
260319452a918ebbbd1ffff96df63e0a5c54eb9cf24fb773691168bead9706cf.
Both PNGs actually viewed in tool call containing74d7be. New fitting/selection
cumulative7590.89seconds; shared-host timing is not a controlled speed benchmark.

## R59 evidence and next scope

Primary is EXACT R58 invariant_mlp,adaptively selected AFTER development results.
Reuse18 R58 fits/72predictions;12NEW fits are distance_mlp,descriptor_mlp,
angle_descriptor_mlp,matrix_mlp x11/22/33. Same7caches,60epochs/14880updates,
backbone,common initialization and actual stream.78NEW prediction sets include
30base-reframe diagnostics. No official test/reserved confirmation access.
All30 selected before new predictions. Independent metric maxerror1.33e-15.

Base regionR@1/centerPCK/softEPE/CE means:
candidate38.499/22.504/.316488/3.352861;
distanceMLP35.078/19.996/.330808/3.443439;
descriptorMLP37.108/21.437/.323537/3.387721;
angleDescriptorMLP37.247/21.358/.323636/3.380668;
matrixMLP34.278/19.368/.332969/3.464639.
Candidate improves base region/center/softEPE for ALL3seeds against ALL4new
controls. Versus angle region+1.252pp,paired+2.767/+.585/+.404;object-conditional
CI[+.721,+1.813]pp. Versus descriptor+1.391pp,CI[+.870,+1.944]. CE versus angle
only2/3better. Old data/reused candidate and conditional object intervals are
not fresh confirmation or an estimate of training-seed uncertainty.

Low-overlap versus descriptor/angle region+2.680/+2.374pp,all3seeds positive;
center/softEPE alsoall3;angleCE only2/3. Noise versus descriptor/angle means
+.958/+1.130pp butonly2/3regionbetter;seed33-.595/-.024pp. Center3/3better,
softEPE/CE mixed. Sparse versus descriptor/angle region+.366/+.198pp,intervals
crosszero;angleonly1/3wins,centermean-.012pp. No stable sparse strong-control
advantage. Candidate versus pure distance/full matrix improves region andcenter
onall4conditions x3seeds. Keep these distinctions;no blanket robustness claim.

Fixed reframe: candidate region/center unchanged all3,maximum probability1.34e-6,
but446A-to-B winner-index changes overALLqueries including ineligible ones.
Do not claim every winner unchanged. MatrixmaxP.200,62306winnerchanges;mean
region+.043pp/center-.112pp/CE+.00110. Declared frame sensitivity is behavior,
not an implementation error;its causal contribution to baseline gap unisolated.
Angle controls have some local probability changes(.057/.113max) despite bounded
val checks;region mean changes+.002/-.006pp. Retain the full diagnostic,do not
convert small aggregate effects into new numerical gates or retroactively fix
the frozen comparison. No confirmed cause of those local changes is claimed.

DECISION:promising DEVELOPMENT candidate,not a generally validated method and
not promoted to untouched confirmation. A single11->96->96 map of Gaussian
invariant relations has useful base/low-overlap evidence beyond learned distance,
direct local descriptors,angles and raw complete matrices. No new module stack.
Sparse uncertainty remains. R58 scalar-form failure and R57 classification
limitations are retained. This is observed-support parameterization,not optimized
3DGS or a recovered texture-transport map; no novelty claim.

Next keep this exact structure and prepare one real-scene native3D correspondence
development experiment with matched strong controls and explicit density scope.
Read docs/round60_real_scene_plan.md. Metadata/file availability is now checked:
9 old SUN3D scenes,576fit/64val/480old_eval pairs,1605required sensor depth files
present. Saved results/round60_real_scene_preparation/availability.json;no new
numeric arrays/model outputs read. The plan declares sensor-depth point-cloud
input,a pose-independent shared normalization,and unchanged R59 candidate/all9
controls. Both fixed source transfer and60epoch target adaptation are planned
BEFORE any target outcome;512/256density conditions and per-scene reporting.
Pose belongs only in labels/scoring,not model input. Implement and freeze concrete
sources/data/configuration before launch. No next source lock/job exists yet.
No post-result budget/temperature/width rescue.
Base/low-overlap evidence justifies this development step without requiring all
old conditions to be wins. Generalization,cross-task evidence and untouched
confirmation remain necessary before the full research objective can be achieved.

R58 completed background (authoritative outcomes retained):
R58 is COMPLETE. Read docs/round58_observed_support_report.md
and results/round58_observed_support/attempt1/completion.json. The full research
goal remains ACTIVE and UNACHIEVED. The R58 primary is NOT promoted. Previous
goal turn was PROGRESS (repeatability diagnostic and completed fits); this turn
is PROGRESS (remaining fits,18 selections,72 evaluations,independent metrics,
actual visual review,scientific interpretation and completion record).

All R58 jobs are terminal. Controller93987/hostPID17782 exited0 atc39775; report
watcher73520 exited0 atc8766a. All26children exited0. Completion record created
at1bd93e. Do not restart any R58 worker/controller/watcher or historical R57 jobs.
Eight frozen scientific sources unchanged;lock:
0f843169bf465bb3202084cfd4f7f56d5846cba6ed34958152be3b128d4f993a.

## R58 evidence and decision

Observed Gaussian supports from32 input neighbors at64tokens/view,not latent
covariance predictions or optimized splats. Disjoint raw point-ID sampling,
independent partial views/noise/SE(3),no ground-truth pose in forward. Two
self/cross Transformer pairs;Gaussian sqrt(2J) sinusoidal relative embedding
inside self-attention only. Cross/final matching use content. Region labels
come from actual point-set overlap. No overlap-detection/full-registration claim.

Six roles x seeds11/22/33,18new fits.60epochs/14880updates each; common backbone,
shared initialization,actual order and validation-CE selection. Old official-
train-only ModelNet7910fit/965val/965old_dev_eval. All18 selections precede new
exports. Official test and reserved scenes remain untouched. Old objects do not
become fresh through a new task. Source/config:ROUND58_OBSERVED_SUPPORT_PROTOCOL.md.

Base means (regionR@1%,centerPCK%,softEPE,CE):
fullGaussian34.458/19.374/.331958/3.465510;
Gaussian-distance34.274/19.221/.331812/3.470909;
Euclidean35.135/19.970/.330620/3.450592;
distance+angle34.951/19.808/.332029/3.455356;
invariantMLP38.499/22.504/.316488/3.352861;
NoPE17.683/8.789/.416250/3.940711.
Full versus MLP region-4.041pp,paired-4.762/-3.926/-3.435;center-3.129pp.
Object-bootstrap region interval[-4.614,-3.450]pp is conditional on these three
fits,not seed uncertainty/independent confirmation. Full versus Euclidean-.677pp
and versus distance+angle-.493pp,all3seeds lower. Full versus metric-only+.184pp,
only1/3seeds better. No stable trace-component benefit.

Full/MLP regionR@1 by base,noise02,sparse256,low_overlap:
34.458/38.499;32.343/36.546;35.065/39.140;33.052/35.692.
Across all4conditions x3seeds,full loses to MLP in region,center,softEPE andCE.
Sparse full versus Euclidean/angle-2.161/-2.539pp,all3negative. Preserve the
limited positive:low-overlap full versus Euclidean+.365pp/angle+.563pp,all3
region signs positive,but object intervals cross zero,center signs mix,and
full remains below MLP and below metric-only. No cherry-picked promotion.

The MLP control has positive evidence worth following:versus both Euclidean and
angle,all4conditions x3seeds improve regionR@1,centerPCK andCE. This control
ALREADY consumes Gaussian-related invariant statistics,so its success is not
proof that Gaussian parameters are useless. Nor does it isolate their benefit:
nonlinear distance encoding or extra pairwise local-feature comparison could
explain gains. The original primary has failed;retain that fact when choosing
the successful development control for follow-up.

Effective query fractions across conditions63.49/62.99/45.71/48.61%;zero-valid
pairs58/59/161/58 of965. Sparse absolute recall increases do not prove robustness
on an unchanged population. All comparisons within each condition are matched.
Validation oracle diagnostic:4.407%eligible queries cannot both maximize overlap
and satisfy.1center threshold,even allowing all overlap ties,although another
available center satisfies it. This limited conflict does not explain away the
simultaneous region/center improvements of MLP. Base nearest-centerPCK bound82.34%.

A prespecified old-val repeatability diagnostic completed atfe9d19:
107710 relations,893analyzable objects. Mean symmetric relative changes:
Euclidean.094429,Gaussian metric.179075,full.174191. On99.44%of objects Gaussian
forms change more. This mixes support,center,sampling and nonlinear sensitivity;
not isolated covariance noise or a task-performance bound. Full is slightly
less variable than metric-only,so trace is not uniquely implicated. No new gate.
See docs/round58_development_diagnostics.md and results/.../diagnostics/repeatability.
One missing-bracket parse error occurred before data access;fixed/source-retained
in diagnostics/preparation_failure1. Main frozen experiment was unaffected.

Checks:18validation CE replays<=8.89e-16;72metric groups<=8.89e-16;actual direct
Gaussian scalar3.55e-15;trained rigid probability<=8.20e-7,zero winner changes;
permutation logit<=1.15e-5. Same-author independent arithmetic,not second-lab
replication. Both PNGs viewed in tool call containing341ffc. New fit/selection
cumulative10077.99seconds,not controlled speed benchmark. Full selected45/39/33,
MLP45/57/45;no global convergence claim,no budget/frequency/temperature rescue.

## Next authorized research scope

Stop this scalar Jeffreys/sinusoidal candidate;do not open fresh confirmation.
Continue from the successful R58 invariantMLP as a DEVELOPMENT discovery. It is
one two-layer map of11Gaussian-related invariants,not an added module stack.
Question:does its benefit require directional Gaussian relations,or are generic
learned distance and local-content comparisons sufficient?

R58 completion authorized one matched attribution comparison on the same task/data/backbone/
budget/seeds,exactly reuse R58 references,add distance-MLP,distance+local-descriptor
MLP,and distance/angle+local-descriptor MLP controls. Explicitly state features,
parameter counts,initialization,how angle neighbors are aggregated,and decisions
before launch. Common local descriptors are the same11input features all R58
models already received;geometry/pose labels must never enter forward. Reuse is
not a new independent fit or fresh confirmation. R59 above now implements that
scope,including a complete matrix-input control. If generic controls match,do not claim Gaussian-specific
utility. A reliable directional-shape gain would motivate separately frozen
real-scene and cross-task validation before untouched confirmation.

The full original objective remains:compact general PE with reliable strong-
control gains,appropriate geometry/robustness/generalization/actual-task evidence,
and untouched confirmation. R57 classification negatives remain part of evidence;
a local learned-control win cannot establish the full objective or novelty.

R57 completed background (its outcome remains authoritative):
R57 is COMPLETE. Read docs/round57_gaussian_pair_report.md
and results/round57_gaussian_pair/attempt1/completion.json. The full research
goal remains ACTIVE and UNACHIEVED; the candidate is NOT promoted. This turn
made PROGRESS: all remaining fits, selection checks, exports, metrics, reports,
visual inspection and the experiment-completion record finished.

All R57 jobs are terminal. Controller77259 exited0 at a2dd48; report watcher87784
exited0 at2eb30b; completion62192 exited0 at1416fa. All15children exited0.
Do not restart these sessions, completed workers or historical controllers.

## R57 result and interpretation

Nine new120epoch/29760update fits: pair_full, pair_distance, pair_center x
seeds11/22/33. Same R48 ModelNet official-train-only7910fit/965val/965old_dev_eval,
backbone, initial shared weights, actual sampling/augmentation stream and
validation-CE selection.39 exact R56/R48 references at the SAME seeds reused;
not39new independent runs. All48 selections preceded new dev_eval exports.
29 frozen scientific sources unchanged; lock:
3e1f37a49b82a07d48a5ac0432b9c803d84bf4563ba2b928aa94a5ec0507de0a.
Official ModelNet test and the reserved cross-scene confirmation data remain unused.

Full Gaussian-pair OA89.292%,mCA84.486%,CE.434493. Same-capacity distance:
89.223/83.683/.445731; center-only88.877/84.019/.433895; MLP89.154/84.457/.437119;
free quadratic joint89.154/84.192/.434653; R56 single-query joint89.223/83.749/.431164.
Full versus distance OA+.069pp, paired -1.658/+1.036/+.829; mCA/CE also2wins1loss.
Full versus MLP OA+.138pp, paired -.415/+1.554/-.725; mCA+.029pp, CE only1/3better.
Versus free quadratic OA+.138pp,1/3wins. These are not stable strong-control gains.
Full versus center-only OA+.415pp,mCA+.467pp and versus fixed distance OA+.484pp,
mCA+.902pp:all3seeds positive. Preserve this limited component evidence.

Prespecified positives: jitter02 versus MLP+.553pp,all3seeds;1024points versus
adaptive distance+.794pp,all3seeds;Y90 versus R56 joint+.933pp,all3seeds.
Limits: jitter02 remains-.173pp versus free quadratic;256points is-.484pp versus
center-only,all3seeds worse,and also below MLP/free quadratic. Tilted rotation
still catastrophically low at6.425%; no whole-network SO3/affine-invariance claim.
Removing only the trace compatibility term at fixed full checkpoints changes
OA-.242pp on average (two losses/one gain);CE increases in all3seeds. This is
component dependence,not recovered physical covariance or unique task utility.
The adaptive-distance term still uses shape during this intervention.

All48 formula samples:maxP6.93e-6,new9max7.35e-7,winner changes0,warnings0.
Saved validationCE replay<=1.33e-8;336metric correct counts and3diagnostic correct
counts exact,CE errors small. Same-author independent arithmetic,not a second
lab/full-network rewrite.39 reference prediction copies exact. Both result PNGs
viewed. New fitting/selection cumulative24645.52sec; not a formal speed benchmark.
The19 positional Q/K channels use FP64 geometric products; no fused-kernel claim.
Full selections100/95/100;distance100/115/120;center80/105/120. No convergence claim.

## Scope recorded at R57 completion (now concretized by R58 above)

Do not promote this classification candidate or open fresh confirmation. Do not
rescue it by temperature/frequency/budget scans. The conditional ordinary-shape
similarity reserve in docs/round57_attribution_note.md remains UNEXECUTED; its
stable-utility trigger was not met. Original scientific objective is unchanged.

Next make one native3D local-correspondence development experiment concrete:
state whether a Gaussian is learned comparison geometry or observed region
support, how its center/shape transforms, what relationship is being predicted,
and which competing explanation each matched control removes. Use the same
information/backbone for all controls, including appropriate generic and geometric
strong controls; preserve center-localization and region-correspondence measures.
Known-frame matching and unknown-pose registration are distinct tasks. For the
latter, ground-truth relative pose belongs only in supervision/scoring; it cannot
be used to pre-align forward inputs. Specify data/configuration/decisions before
training. At R57 completion no next-round candidate or budget was frozen; the
R58 state at the top of this document supersedes that execution status.

A classification tie does not prohibit a separately motivated geometry diagnostic;
a local-task pass does not erase the classification and rotation limitations.
General utility,appropriate geometry/robustness,cross-task evidence and untouched
confirmation are still required. Do not substitute one convenient task for them.

PRoPE branches(3)/(4) were read. Their same-camera calibrated comparison and
bounded budget diagnosis are already completed in R54/R55:plane and PRoPE are
near ties at5000steps,while full Gaussian lacks stable extra utility over point/
affine. For future calibrated multi-view studies preserve known camera operators
and feature inputs across PRoPE,plane and Gaussian variants. R57 latent Gaussian
similarity is not a texture-transport transform. Its mathematical construction
alone does not establish usefulness or novelty.

The attribution note also records the R46/R5/R41/R57 input semantics and checked
GeoTransformer/REGTR primary methods as task/control references. They have NOT
been implemented or reproduced. The following entries are historical background;
latest execution state and next scope are given above.

The following R56 findings remain authoritative background and negative evidence.

R56 completion state (its jobs are terminal; R57 jobs are listed above): read
[the R56 report](/home/liwt/GRoPE/docs/round56_native_joint_report.md) and
results/round56_native_joint/attempt1/completion.json. R52--R55 also complete.
The full research goal remains ACTIVE and UNACHIEVED. All R56 jobs are terminal.
Do not restart any completed queue or historical PID.
This goal turn made concrete progress: all nine fits,39 selected-model checks,
39 exports/273 condition groups,12 diagnostics,analysis and visual review finished.
Previous goal turn was a verified wait on the original live35361/18034 handles.
No confirmation dataset or official ModelNet test was accessed.

## R56 outcome and decision

Same old ModelNet official-train-only split7910fit/965val/965dev_eval; same
120epochs/29760updates,backbone,initial shared weights,sampling/augmentation,
optimizer and validation-CE selection. Nine new fits (three joint roles x seeds
11/22/33),30 exact R48 references reused at the SAME seeds. Do not mix these
three-seed means with the old five-seed summary or count reused fits as new data.
All39 selections preceded new evaluation. Sources26 unchanged at lock
269ca1e9530ea4707fc0bc2850b40ea1b5a232e3b0b4883ccca7b1c9534898c4.

Base OA:joint_conditional89.223%,joint_fixed88.808%,joint_free_quadratic89.154%,
conditional-alone88.325%,diagonal-alone89.223%,centerRoPE88.774%,MLP+RoPE89.154%.
Joint full vs Gaussian-alone +.898pp,3/3 seeds; CE improves in all3. This supports
center-phase complementarity. Free quadratic/fixed distance also improve when
center phase is added; the effect is not unique to Gaussian parameterization.
Full joint vs free joint only+.069pp,1/3 seeds,mCA-.444pp; versus MLP only+.069pp,
2/3 seeds,mCA-.709pp. Mean CE slightly better but only2/3 seeds in each comparison.
These strong-control results are near ties,not a Gaussian-specific gain.
Y90 and jitter02 OA are .898pp/.760pp lower than free joint,ALL3 seeds worse.
Tilted rotation remains catastrophic (full joint5.803%); no SO3 robustness claim.
Point reversal changes no class winner in any39models; tiny logit differences
(max1.93e-5) do not change the outcome.

Fixed-checkpoint shape interventions on full joint decrease OA21.071pp for
trace-preserving isotropization and53.610pp for fixed Z90 covariance rotation.
The models DO depend on shape parameters; lack of unique utility is not evidence
that the predictor is ignored. These interventions shift learned attention and
are not proof of recovered physical covariance. Each layer retains its mean
predicted from its CURRENT input; early changes can alter later features/means.
They do not clamp all baseline-trajectory centers. All12 diagnostics retained.

DECISION:no promotion,no fresh confirmation,no diagonal/isotropic joint sweep,
no budget/temperature/frequency rescue. Preserve positive component effects and
negative strong-control/robustness results. Full objective remains intact.

## R57 rationale, recorded at R56 completion

Current R56 Gaussian defines a query-conditioned region over point keys. Consider
whether Gaussian parameters of BOTH tokens should define their relative relation.
The existing compact_gaussian_dual_pe_reserve_20261101.md derives Gaussian
Jeffreys similarity with19position Q/K channels. Its width/additivity
counterexamples remain valid for use as probability mass; they do not reject
all learned token-similarity uses. Neither closed form nor invariance establishes
actual task usefulness. Do not restart its old static-integrator gates.

Before training, formulate one prospective comparison:full Gaussian-pair score,
same adaptive-center distance with the covariance-compatibility term removed,
and learned centers with fixed covariance; retain RoPE/MLP/free-quadratic strong
controls. The same-capacity removal control separates shape compatibility from
generic adaptive distance; the center-only control separates moving both centers.
Use one positional relation,not extra branches/losses/rendering. Explicitly assess
whether this is a meaningful structural change rather than another parameter
scan. The prospective R57 plan is now implemented and frozen as recorded above;
no R57 task results are available yet.
R56 does not prove this next hypothesis. Old development evidence first;
strong controls,appropriate robustness/cross-task and fresh confirmation still
required for any eventual generally useful compact scheme.

## Execution evidence and scope

Controller35361 TERMINAL exit0(fb7c88),all15children exit0. Report watcher18034
TERMINAL exit0(f7032b). Completion47178 TERMINAL exit0(80467b). Both result PNGs
actually viewed. Four bounded tests,30 reference-function checks and real-input
gradient checks passed before fitting. All actual sample streams equal originals.
39 independent first-layer formula samples:maxP6.93e-6,winner changes0,warnings0;
saved validation CE replay<=1.33e-8. All273 metric correct counts exact,CE error
<=1.13e-7. Same-author independent arithmetic,not another lab/full-network rewrite.
New fitting/selection cost16492.57sec with checkpoint overhead; shared-host
wall time is not a formal throughput benchmark. Common338508params; full/free
PE1746,full model340254. Some selected models at120epochs; no convergence proof.
Two transient host NVIDIA cleanup/NVML waits resolved without restarts,kills,
driver edits or source changes; retained in operational_delay1/2.json. The first
report-watcher sandbox invocation could not see hostPID5539; its preserved
watch_attempt0_failure.json predates the successful unchanged host invocation.

## Literature and interpretation retained

PRoPE branch(3),01a199fa-0796-7a40-9ba5-c084c53d3891,and branch(4),
01a19a69-896a-7423-8fff-3f54c97288d0,were read. Camera-relative structure is not
our invention; covariance shape alone does not specify internal texture transport.
PRoPE https://arxiv.org/html/2507.10496v2#S3.SS4 ;
HRoPE https://arxiv.org/html/2610.28284v1#S3.SS3 .
R56 weights spatial preference without attenuating unit-amplitude rotary content;
it does not perform homography alignment or measure real GS support covariance.

Riva et al., https://arxiv.org/html/2409.13291v1 ,sections4/5.2,combine separate
Gaussian and RoPE heads; clean/noise tradeoffs prevent a general robustness
inference. Publisher:https://doi.org/10.2312/stag.20241345 . No reproduction.
MoPE https://arxiv.org/html/2606.01258v1 ,definition1/section3.1,uses Gaussian
amplitude around an absolute origin and discusses its origin prior; this differs
from R56's conditional relative quadratic. No novelty claim for our combination.
He et al., https://arxiv.org/html/2601.02339v1 ,sections3.1.1/3.1.2,use measured
Gaussian covariance in regularized graph metrics,spectral shape descriptors and
center PE. That multi-component pipeline does not validate R56's learned region.
Point-GN https://arxiv.org/html/2412.03056v1 ,section3.2,uses fixed axiswise
Gaussian coordinate features in a non-parametric classifier,not full anisotropic
Gaussian Transformer PE. No paper scores are compared with our different split.

The following R55/R54/R53/R52 status entries precede R56.

Updated 2026-11-02: R55,R54,R53 and R52 are COMPLETE.
Full goal remains active and unachieved. No owned research job remains live.
Do not restart these completed rounds. Next scope:native3D center/region encoding.
R51 real visual matching COMPLETE, including the explicitly requested HRoPE
versus MLP comparison. The user explicitly resumed after the PRoPE discussion
and approved the previously described next plan through the active goal.
Full goal remains unachieved. The earlier pause entries below are historical.

Previous goal work made progress:16 R53 fits, matched references, bounded checks,
and the PRoPE/calibrated-plane comparison gap are retained. All old interrupted
processes were authoritatively terminal before resumption. R53 resume14324 is
now TERMINAL exit0(4f37e3):18 new fits,12 reused references,60 exports,30 checked
models complete. Analysis/plot/reference27272 TERMINAL exit0(bf4ada). No R53 job
is live; do not restart. Interrupted history is archived, no completed run replaced.
Read docs/round53_ray_frame_report.md and completion.json. SoftEPE:PRoPE4.2069,
ray-point4.0813,ray-iso4.1042,ray-full4.0611,ray-affine4.0635,hybrid4.1675.
Full vs PRoPE improves3.464%,3/3seeds/5scene means (two scenes nearly tied).
Full vs point only-.0201px,2/3seed and2/5scene wins; versus affine-.00236px.
No unique Gaussian benefit/promotion. All30 checks/60 metric replays passed
bounded impact review; warnings0, no winners changed. Unused confirmation remains untouched.

R54 sources: ROUND54_CALIBRATED_PLANE_PROTOCOL.md, models/round54_calibrated_plane.py,
tests/test_round54_calibrated_plane.py. Three bounded tests passed45786 exit0
(b0a308), including official PRoPE arithmetic with explicit chart/phase adapters.
This is a matched implementation, not an official training-pipeline reproduction.
Two new plane/offset roles x3seeds;21 identity-checked R52 references reused.
Nine real-input checks and21 reference identities verified88928 exit0(cf29ac).
Lock f0c936b2c16b80305c1e6ee7ecefe7997f80d51e41d6654d1a13d4bda869f94a.
GPU queue86598 is TERMINAL exit0(8105a9), all7 children exited0; CPU checker30710
is TERMINAL exit0(f0e9c3). Analysis/plot/reference84445 TERMINAL exit0(a28002).
All21 scientific sources,27 selected model identities and54 prediction exports
verified;42 reused reference exports equal R52 exactly. Read
docs/round54_calibrated_plane_report.md and completion.json. SoftEPE:P4.2069,
offset4.2781,plane4.5467,iso4.3951,GS4.4012,affine4.4123. Plane is worse than P
by8.079%,3/3 seeds and5/5 scene means; GS is worse by4.619%,also3/3 and5/5.
Filling the known-camera comparison gap did not reveal local-shape utility.
This does not establish PRoPE's universal superiority over HRoPE; the pipeline
and task differ from the original paper. All27 operator/CPU-validation checks
and54 metric replays passed bounded impact review,warnings0,winner changes0.
New6 fits/selection cost387.54sec. Source/data/model comparisons remain frozen.

R55 is COMPLETE under ROUND55_BUDGET_DIAGNOSTIC_PROTOCOL.md. Read
 docs/round55_budget_diagnostic_report.md and
 results/round55_budget_diagnostic/attempt1/completion.json.
Previous goal turn made progress:implemented/froze the fixed budget intervention
and launched GPU fits. This turn verified the live handles,completed24 fits/48
exports/24 selected-model checks,analyzed results,and closed the experiment.
No fresh confirmation accessed. The full research objective remains unachieved.

One matched extension only:eight unchanged R52/R53/R54 roles x3 paired seeds,
5000steps. The original1000-step init,batches,schedule,selected weights/functions
and validation histories reproduced exactly for all24 models. Subsequent4000
updates used constantlr.0002 with continuous optimizer state. All24 selections
preceded evaluation;shared initialization and complete sampling hashes matched.
34 scientific sources unchanged at lock
6bb56fc73932ef32b058c08612381008247f06759c84a3dd194d1701b1a84e7c.
All selected steps were2600..4900;none at5000,without claiming global convergence.
No5000-step NoPE/MLP run was performed;their old1000-step scores are not a fair
ranking comparator for this diagnostic. These are correlated development runs.

SoftEPE/PCK8%:P3.8636/87.3356,offset3.9148/87.0646,plane3.8789/87.3371,
forwardGS3.9185/87.4501,ray-point3.7065/88.0617,ray-iso3.7482/87.8231,
ray-full3.7344/88.0644,ray-affine3.6669/88.3119. Ray-full vs point changes
from-.0201px at1000 steps to+.0279px at5000;1/3 seed wins and0/5 scene wins.
PCK8+.00272pp is a tie. Full vs affine+.06748px,1/3 seed wins,0/5 scenes;
soft-tail+.12694pp is worse inall3 seeds. Full vs iso retains+.2413pp PCK8
(3/3seeds,4/5scenes),but point/affine prevent a unique-shape claim.
Full vs P retains-.1292px/+ .7288pp,3/3seeds,4/5scenes;point/affine also gain.
Do not assign the cross-route gain solely to Gaussian shape or inverse frames.

Plane vs P gap shrinks from+.3399px to+.0153px:the large short-budget penalty
mostly disappears;these are now near ties,not universal PRoPE superiority.
Plane beats offset by.0360px/+.2725pp PCK8 in3/3seeds,but not P consistently.
ForwardGS vs P gap shrinks to+.0549px,still0/3 softEPE seed wins.
RGB-permuted PCK8 rises with longer training:P75.605%,ray-point74.399%,
ray-full74.413%,affine75.149%. Full still loses13.652pp under permutation;
local appearance matters,but the test cannot distinguish stronger geometry,
retained image-level features or development-scene statistical memory.

Queue64824 TERMINAL exit0(d1c8f4),all16children exit0;CPU checker98385 TERMINAL
exit0(879491). All24 operator checks:maxP5.28e-8,maxV2.76e-6,winner changes0,
full CPU valCE error<=1.51e-7. All48 metric replays:PCK/tails exact,EPE<=2.22e-9,
warnings0. Same-author independent arithmetic,not another lab/full-network rewrite.
New training/selection cost7676.82sec,including checkpoint/prefix-check overhead.
Initial analysis87058 exited1(7cf399) due to stale MLP scene index;failed source
and record retained in postprocessing/attempt0. Corrected analysis91461 exit0
(63a9f6),final plot96297 exit0(1dbc45);both plots visually inspected. No science
source,selection,prediction or training changed. Completion72758 exit0.

DECISION:end tuning rescue of this local-frame route;do not scan more budgets,
features or matrix factorizations to manufacture a winner. Preserve its positive
center/generic-frame signals and negative full-shape result. No candidate promoted.
Next structural hypothesis:in native3D,retain center RoPE on content and express
conditional Gaussian region preference as extra positional Q/K channels in the
same dot product. No physical covariance attenuation of the center phase.
Use the R48 native point-cloud setting to remove camera/predicted-depth dependence;
compare each part alone,joint full/simpler/fixed-distance/free-quadratic forms,and
RoPE/MLP strong controls with identical data/backbone/budget. Only exact unchanged
references may be reused. R48's severe tilted-rotation failure remains a limitation.
The unexecuted R49 draft is background,not permission to resume its older projected
matching setup,60-fit plan or1pp guide. A new native3D protocol and identity checks
must precede training. No next-round code/protocol has yet been frozen or launched.

Objective: a compact, generally useful position encoding derived from 3DGS
parameterization, supported by matched strong controls, actual tasks and
untouched confirmation. No candidate has met the complete objective.

2026-11-02 archival PAUSED planning update (superseded by completed R54 above):
user requested reading the PRoPE discussion
in the branch titled "检索3DGS与RoPE结合进展 (3)" (thread
01a199fa-0796-7a40-9ba5-c084c53d3891). Read its latest two exchanges and checked
PRoPE arxiv2507.10496v2, official prope/torch.py (blob
1c450ccc05bca91a6b9ddbd19971437896ef83b5), and HRoPE arxiv2610.28284v1
sections3.3/4.2.2. No research job was resumed.

R52 camera_only is a PRoPE-style adaptation, not a new project contribution or
an exact reproduction of the official pipeline. Current HRoPE predicts its
R/t/plane transform; unlike camera_only, it does not explicitly insert the known
camera extrinsics into that transform. Identical available metadata does not
isolate the incremental benefit of local surface geometry. The missing matched
control is calibrated HRoPE: H=P F, F=[R_n,z*n;0,1], with known P retained.
The paper reports this P*F comparison separately from predicted3x3 HRoPE.
Common F cancellation is explicitly its reduction to PRoPE; R52's common-L
cancellation limits the whitening interpretation, not a proof of implementation
failure or a sufficient reason to prefer inverse frames.

The priority then proposed for resumption was to align/document the PRoPE camera chart,
units, rotary conventions and Q/K/V/output transport; fill the calibrated-plane
control under the same inputs, predictor feature source and training budget.
Compare P, P*plane (plus offset-only control), P*Gaussian (isotropic/full), and
P*affine; retain NoPE/MLP references. Reuse old checkpoints only if scientific
configuration and functions are unchanged. First question: can local plane or
Gaussian geometry add utility beyond the same known camera operator? HRoPE
winning is a mechanism reference, not a universal prerequisite or new gate.
Use paired seeds/scenes, continuous localization and tails; geometry strata must
be fixed independently of model errors. If needed, permit one bounded matched
budget/feature-source diagnosis rather than indefinite tuning. R53 partial fits
remain an independent frozen candidate for eventual original-protocol closure;
do not splice new controls into its frozen sources or let it dictate the route.
Native3D/cross-task utility and untouched confirmation remain required for the
general claim. This paragraph is a plan, not an executed experiment or approval
to resume the paused goal.

Latest scope clarification: R41/R42 rendering readout/loss results are task-specific
evidence and cannot establish general Transformer PE utility. R43--R45 test PE
inside the full two-block Transformer in projected 2D coordinates; R46/R47
test native 3D coordinates with matched anchor-geometry controls and no renderer.
The main question is whether the GS-form positional parameterization improves
the same Transformer against strong PE controls at a matched training budget.
Rendering is optional downstream evidence, never a prerequisite or substitute.
Cross-view matching is also one task, not proof of generality. Distinguish
generic Gaussian/RBF effects, learned center/orientation/scale benefits, and
effects of changed input geometry. Learning a PE covariance does not require
using a physical rendering covariance. Native 3D utility and cross-task utility
are not established as general Gaussian advantages; R46/R47's completed evidence
is recorded below. No historical result or criterion changed.

R48 is COMPLETE and NEGATIVE for its fixed primary development claim. Native
point-cloud classification OA:conditional88.622%,RoPE88.518%,MLP88.891%,free
quadratic88.642%; only the comparison with NoPE gives>=1pp and5/5seed wins.
No unique full-Gaussian benefit or promotion. The severe shared tilted-rotation
failure was checked with a bounded independent input/replay path; see
docs/round48_native_point_report.md. Controller24005 is TERMINAL exit0; do not
restart R48 training/export. All official-test/confirmation data remain unused.
R49 joint center-RoPE/Gaussian-prior code/protocol/tests exist as UNEXECUTED
drafts. They were not frozen, unit-tested or trained. Historical statements
below that say no R49 files exist predate those drafts and are superseded.

R50 local-frame mechanism experiment COMPLETE, ROUND50_LOCAL_FRAME_PROTOCOL.md.
The user discussed HRoPE's surface-deformation explanation and resumed the goal.
Test identical-input compact GS-frame, HRoPE operator adaptation, MLP, affine,
RoPE, isotropic/identity frames and Gaussian-bias controls inside the same
Transformer. Procedural multi-plane correspondence is a bounded development
test, not a renderer objective, real-data result or fresh confirmation. Both
exact and identically corrupted geometry are supplied to all methods; corruption
is a synthetic proxy, not a trained predictor. Full goal ACTIVE/unachieved.

R50 four bounded CPU tests passed: physical homography/differential labels,
independent nonzero matrix/transport path, shared initialization/label exclusion/
permutation, and shape gauge/finite-difference gradients. Generated fixed data
512fit/64val/128paired scenes for each of4 development conditions. Sources frozen
at lock SHA1c86fb2db11f38963441e23c3e8b162f6b986fdedb8a63cc8bf3922e45f7807a.
All48 trainings/384 evaluation groups complete; controller75996 TERMINAL exit0
(d95300), all6train+3eval children exit0. No owned live research jobs remain.
Postcheck49758,neighborhood30940,plot30171,zoom diagnostic21607 all exit0.
GPU devices are not exposed inside the sandbox; use host execution for CUDA,
observe current occupancy rather than reserving fixed device numbers, and never
modify drivers or interfere with another user's processes.

Read docs/round50_local_frame_report.md before continuing. R50 oracle AUC
four-condition means:GS57.311%,MLP59.561%,affine57.494%,HRoPE59.343%,
isotropic57.309%,identity56.811%,RoPE58.982%. Exact values in summary.json.
No unique full-GS advantage; zoom is a material weakness. Fine localization and
the39-scene/166-query severe-deformation subset do show local frame advantages,
but these are not general GS success. Base PCK8 is~100%; grid AUC ceiling77.998%
versusMLP77.510% makes the proposed+1pp base guide unattainable. Preserve the
old guide, explicitly stop treating it as a valid scientific rejection rule.

Fixed-checkpoint diagnostics changed the next decision: RGB token permutation
hardly changes metrics, while changing phase units/removing rotary V transport
damages fixed models. Exact centers+cameras let the task be solved geometrically;
R50 cannot independently verify appearance alignment. No trained operator bug:
all48 independent formula samples maxP3.28e-7,maxV1.57e-6,zero winners changed;
CPU validationCE max1.05e-7. GS axes did differentiate (mean ratio1.074-1.398),
so spherical-init zero orientation derivative is not proof of training failure.

R51 is COMPLETE: ROUND51_VISUAL_ALIGNMENT_PROTOCOL.md.
Ten methods x three seeds; 1000 updates each. Original SUN3D576fit/64val/480old
evaluation pairs; same RGB ResNet features/native positions/cameras for all,
without per-token3D centers/depth/normals/covariances. Full GS remains primary;
requested HRoPE/MLP comparison is explicitly reported. Two HRoPE forms: inherited
R50 adaptation and paper-form hard bounds/rotary convention. Official repository
still says Coming soon; these are our operator implementations, not the official
LVSM/VGGT pipeline or its reported benchmark. Shared backbone/features/frequencies
and QK normalization are declared adaptations. Full goal unachieved.

R51 lock812c939ec3d824f5e17da5bf1ec08147f08a782d40e2d6f332ed591beecac0df,
results/round51_visual_alignment/attempt1. All30 fits/60 exports complete; queue
9626 TERMINAL exit0 (9ee31b), all10child tasks exit0 on initially-only-idle GPU5.
CPU postcheck9616 TERMINAL exit0 (9fad4e); analysis/plot12476 exit0 (96081c).
No owned live research jobs. Do not restart these or any terminal R50 jobs.
Three bounded unit tests, all10real-input CPU checks and GPU smokes passed.
All30 selected operator samples: maxP1.63e-7,maxV9.01e-7,zero winners changed;
CPU validationCE max1.50e-7; all60 metrics replayed, largest EPE difference8.53e-9.
Shared initial states and actual batches matched; all learned PE tensors changed.
No fresh confirmation accessed; full scientific goal remains unachieved.

Read docs/round51_visual_alignment_report.md. PCK8 / softEPE: MLP79.331%/6.192,
HRoPE adaptation80.758%/5.993, paper-form79.996%/6.239, fullGS80.292%/6.107,
affine80.225%/6.108, iso80.210%/6.128, identity80.270%/6.079,
NoPE80.564%/5.845. HRoPE adaptation beats MLP in3/3seeds and5/5scene means
on both metrics (+1.427pp PCK8,-3.218%softEPE), with small extra gains over
identity; paper-form is unstable, and NoPE has better mean continuous error.
FullGS is near-tied with generic/simple controls, not a unique shape success.
RGB permutation reduces PCK8 by~63--72pp: genuine visual dependence in this
limited sense, not proof of physical plane/covariance recovery. HRoPE's gain is
mostly at15frame gap; at60frames softEPE10.253 versusMLP10.235,NoPE9.714.
All focal ratios1; no zoom test. Three seeds/five reused rooms are developmental.

R52 is COMPLETE under ROUND52_CAMERA_FRAME_PROTOCOL.md. The earlier
docs/round52_camera_frame_plan.md draft is superseded by that executable protocol.
P=diag(K,1) T^-1 retains camera depth in a homogeneous4D lift, with a fixed3m
unit; A=P G with canonical Gaussian root G. It is not an inverse2D projection.
Eight roles x three seeds: NoPE,MLP,smoothHRoPE,localGS(A=G),camera-only,
camera-isotropic,camera-fullGS(primary),camera-affine. The localGS control
isolates adding P; R51 used A=G^-1, so comparison to R51 alone would confound
the convention. Same R51 cached inputs/labels, common initialization/batches,
1000updates; no unused confirmation opened. Three bounded tests and all eight
real-input CPU gradient checks passed. Lock:
a92cb1b3d86db8e9a49bd7b78c136a64cfef8258fea9f09759c591ac256fe5ed.
Evidence results/round52_camera_frame/attempt1. All24 fits/48 exports complete;
controller16683 TERMINAL exit0(ba832b), all10child tasks exit0; CPU postcheck5902
TERMINAL exit0(1929cd), analysis/plot89375 exit0(a233fa), reference replay51439
exit0(17e7df). Do not restart these terminal jobs. No owned live research jobs.
Training used initially-only-idle GPU5; eval used later-idle GPU3/5. Future GPU
selection must inspect current occupancy. All24 selections sealed before export.
All24 independent NumPy operator samples: maxP1.63e-7,maxV2.23e-6,zero winner
changes; CPU validationCE error<=1.39e-7. All48 metric groups replayed; EPE error
<=7.58e-9px, PCK/tails exact, warnings0. All PE tensors changed. Nine repeated
NoPE/MLP/HRoPE checkpoints equal R51 tensor-for-tensor; not new independent trials.

Read docs/round52_camera_frame_report.md. PCK8/softEPE: camera_only87.841%/4.2069,
cameraGS87.457%/4.4012, cameraIso87.298%/4.3951, cameraAffine87.218%/4.4123,
localGS80.352%/6.0824, NoPE80.564%/5.8445, MLP79.331%/6.1923,
HRoPE80.758%/5.9931. Camera-only beats NoPE by7.277pp and28.021%softEPE,
with3/3seeds and5/5scenes positive. FullGS is worse than camera-only in3/3seeds
and all5scene softEPE means (+4.619%); iso/affine differences are small ties.
Do not rename camera-only as a Gaussian success. Camera benefit extends to60gap:
softEPE7.009 versusNoPE9.714,GS7.334. RGB permutation drops camera-only32.439pp,
GS36.244pp; visual dependence plus a strong camera prior, not physical-GS recovery.
No unused confirmation accessed; full goal remains unachieved.
Data-free algebra witness13930 TERMINAL exit0 (55ea94) records that common L
cancels from G_i G_j^-1 (max1.67e-16), whereas inverse-frame translation retains
whitening. Read docs/round52_operator_scope_note.md; this limits physical claims,
does not alter the frozen trial and is not a proven cause of task performance.
R51 already shows the task/metrics can register a
limited HRoPE adaptation benefit, but not difficult-geometry superiority or GS
generality. The SPD-frame pure-rotation limitation is a mathematical observation
(docs/round51_operator_scope_note.md), not an established cause of task errors.
Do not discard NoPE, inflate tiny ties into failures, or return to static repair.

R53 archival execution notes (completion above supersedes the pause state):
docs/round53_ray_frame_draft.md records the design origin; the executable
ROUND53_RAY_FRAME_PROTOCOL.md is now FROZEN and GPU training has started.
Three bounded tests passed again (48127 exit0,839afb); ten real-input checks
and twelve immutable reference-function/initialization/batch checks passed
(38056 exit0,058a77). Lock c19be86d01aae2f0928299da65b9414bca8ec40e5d4c322e6e9dbe9a31c46b5c.
User subsequently PAUSED the goal. Queue51887 TERMINAL exit143(4c5f73) and CPU
postcheck36170 TERMINAL exit143(dcd3c7) were intentionally stopped with SIGTERM,
along with owned training child65399. All three process exits were verified.
At that pause16/18 new fits and12 references were saved, without complete evaluation.
The subsequent authorized closure is recorded above. Preserve the interruption
history and do not restart completed jobs.
See results/round53_ray_frame/attempt1/user_pause.json; queue_status_before_pause.json
preserves the interrupted queue state. This is user interruption, not model failure.
Only currently idle GPUs0/1/3/5/7 were used at launch; future launches recheck.
A=(C G)^-1,mu=d*camera_ray,6D
Cholesky covariance+one depth; full-head4D transport primary, half-head native
rotary hybrid as a fixed control. This retains shared-scale whitening of center
differences and uses one main operator, but no utility is established. Include
the newly strong frozen R52 camera projection operator, not just NoPE/MLP/HRoPE.
The model factory has10roles:18new fits (pose-only,ray-point,ray-iso,
ray-full,ray-affine,ray-hybrid)x3seeds and12 explicitly reused reference models
from R52. Original training locks and records are retained; reuse is not new
confirmation. Scientific inputs/comparisons and bounded postcheck are frozen.
Preserve unused confirmation and the native3D/cross-task goal. The completion
report above supersedes these historical execution details and the former pause.

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
R45 is now complete with no stable gain (details below), and it does not waive3D/cross-task, robustness,
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

R45 is now COMPLETE: `ROUND45_PHASE_WARP_PROTOCOL.md`,
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
Completed evidence `results/round45_phase_warp/attempt2`, lock SHA256
f1276ec5e145fbcda1f19dba5979329e1c619a0db9c57ae6e74d50cc61c5426c.
Bounded queue tool handle54656 TERMINAL exit0, historical host PID11281.
All13 child exits0: smoke,5train,training-audit,5export,metrics. Active list is
empty; source and actual exits remain in queue_source.py/queue_status.json/
queue.log. Full40new/30reused model sealing and old_eval replay are complete.
Do NOT restart any of these historical handles/PIDs. CUDA toy max P error
1.589e-7 and all8 initial full-network differences0. Main independent sampled
P error3.224e-7,0winner changes/no warnings. Full67200 frame rows replayed,
zero PCK4/PCK8/valid-count discrepancies; maxCE frame-sum difference1.826e-4.
All40 selected shared and active positional parameters changed. Best-step
counts700:2,800:13,900:12,1000:13; no convergence claim. Peak677.77MiB,
summed training/selection1886.26s, not a throughput benchmark.

R45 primary PCK8=.923574821 vs point=.923507882 (+.006694pp),2/5winning seeds,
3/5winning scenes; CE+0.1990%,PCK4slightly lower. Paired point deltas in pp:
+.151354,-.005151,-.134141,+.070238,-.048831. The lack of stable useful gain
does not hinge on the1pp criterion or numerical auditing. Full-minus-Fourier/
MLP/affine/free-frequency differences +.041519/+.044458/+.042344/+.021440pp.
All new roles are within.04446pp. Components full-minus-fixed/iso/fixed-centers
are +.034240/+.031838/+.026562pp,4/5positive seeds each and5/4/4positive rooms;
report these small positive tendencies without substituting them for utility.
Primary zero-V PCK8=.596677892 versus normal.923574821.

Before old_eval exports, a bounded old-VAL phase-reset diagnostic was frozen
in results/round45_phase_warp/diagnostic1/plan.json, source
experiments/round45_phase_diagnostic.py. Freeze handle8735 and run30200 both
TERMINAL exit0. Six roles x5seeds x2conditions=60forward conditions on64VAL
pairs,3840frame rows. No training/tuning/new data. Normal replay CE and count
differences0. Restoring original point phases at fixed content weights gives
primary PCK8 .942610837 versus normal.943065555: learned-phase effect+.045472pp,
3/5positive seeds. Corresponding fixed/Fourier/MLP/affine/free effects in pp:
+.064418/+.159151/+.405457/+.164835/-.009473. Fixed-weight dependence is not a
retrained ablation, independent confirmation or evidence of baseline superiority.
NO R45 job remains active. Terminal record execution_completion.json and full
report docs/round45_phase_warp_report.md are authoritative. Fresh cohort untouched.

The review in docs/round46_native3d_design.md has become frozen R46 (status
below). Stop
promotion/tuning of R45's 2D candidate. Cover native3D coordinates inside a full
Transformer with matched point/non-Gaussian controls; do not substitute a
rendering task. Source inspection found R5 mean3 is a patch centroid whereas
native2D positions are exact token centers. A native3D experiment must control
center-lifting geometry separately (e.g. same center geometry with2D and3D
point controls); the code difference does not prove a cause of old failures.
No new arrays opened for this review. It also does not establish a second task,
general geometry/robustness/cost or untouched confirmation. Full goal ACTIVE/unmet.

R46 COMPLETE: ROUND46_NATIVE3D_PROTOCOL.md,
models/round46_native3d_pe.py, experiments/round46_native3d_{data,train,audit}.py.
The preceding turn made concrete progress: R45's40new trainings and60diagnostic
conditions completed and rejected its promotion; no R45 process remains live.
R46 tests native3D positions in the same complete Transformer, with center2d
and gaussian2d roles on the SAME newly center-lifted geometry. These isolate
the changed position anchor from the PE effects. The old point2D result remains
as a historical geometry control, not a falsely matched new-input baseline.

Center rays use existing RGB-predicted depth, sanitized/clamped before averaging
the four half-pixel-center samples. All864 points are expressed in the query
camera frame, centered and isotropically RMS-normalized, then multiplied by12.
Both attention directions share those3D coordinates. Input-only normalization
and no covariance/label mask in forward. Known cameras remain supplied metadata.
Same visual feature bytes, old labels,432keys and query population. 1605old
scene/frame records frozen; no unused cohort access. The3D PE's covariance is
a learned field parameter, not physical optimized-GS R/S.

55new trainings=11roles x5seeds plus5 identity-checked historical point controls.
Primary64Gaussian centers,full3D precision,zero64x3 coordinate mixture then
original-form unitary RoPE;768added parameters. Iso448/fixed192/fixed-centers576/
Fourier320/MLP448/affine9/free-frequency72/point3d0/center2d0/gaussian2d252.
All3D roles start exactly at a common3D point function;2D controls start at the
common2D point function. Shared neural initialization and1000sampled batches
match the prior baseline identities. Same1000updates/batch8/selection budget.
Primary criteria and continuous component reporting are prospective in protocol;
full3D/cross-task robustness/cost and untouched confirmation are still outstanding.

Freeze+preflight tool25108 TERMINAL exit0, all6tests passed. Frozen lock SHA256
45cdfe3e5d81792573a2c87964e78067a013e136bd1bf8ad539236ae1c382a69.
Current evidence results/round46_native3d/attempt1. Bounded controller tool
handle82629 is now TERMINAL exit0; exact historical host PID and launch identities
are in execution_handles.json. Do not restart it. It ran data construction/independent
geometry audit -> CUDA smoke ->5seed training shards -> checkpoint seal ->
old_eval geometry construction/audit ->5exports -> full-query metric replay.
Actual state/exits are in queue_status.json/queue.log and execution_completion.json.
All stages completed; scientific results are recorded below. Full goal ACTIVE/unmet.

R46 update: data child61213 exited0 and all576train/64val pairs' coordinate
arrays passed independent geometry reconstruction; maximum coordinate error
3.815e-6, visual input bytes identical. All11 CUDA smoke roles passed, maxP
1.681e-7,zero winner changes and exact same-dimensional point initialization.
Training launched on verified idle GPUs1..5: seeds11/22/33/44/55, child host
PIDs108770/108771/108772/108788/108789, respectively. These are historical controller
children, not tool handles; all have exited0. Parent handle82629 is also terminal.
Training launch records and UUIDs also saved in execution_handles.json.

R46 terminal result: all55new trainings+5historical controls sealed and exported,
all480old_eval pairs' new geometry passed, all57600frame metric rows replayed.
PCK4/PCK8/valid count discrepancies0; maximum CE frame-sum difference1.612e-4.
New sampled NumPy P error max3.669e-7,0winner changes/no warnings. All55 shared
and existing active positional parameters updated. Best steps500:7,600:6,700:6,
800:15,900:14,1000:7; no convergence claim. Peak774.835MiB, summed per-run
training/selection2417.218s, not a formal throughput comparison. Controller
handle82629 directly returned exit0;15children all exit0,active[]; no R46 jobs live.

Primary gaussian3d PCK8=.879149725 vs point3d .851576467 (+2.757326pp),5/5
positive seeds/scenes and CE-7.4948%. However mlp3d=.882505583: primary loses
.335586pp,0/5winning seeds,1/5winning scenes. Gaussian-minus-Fourier is only
+.060967pp,3/5seeds and2/5scenes. Thus the nonlinear-coordinate-map benefit
is real on old data but is not a Gaussian-specific advantage. Strong same-input
center2d=.924121336, so primary is4.497161pp lower,0/5seeds/scenes. No promotion;
this does not depend on the1pp margin or a numerical audit veto.

Center2d-minus-legacy anchor change +.061345pp,4/5seeds3/5scenes;
gaussian2d-minus-center2d +.006720pp,3/5seeds3/5scenes. Point3d-minus-center2d
-7.254487pp combines dimension/normalization/frequency configuration, not pure
dimension. Full-minus-fixed3d/iso3d/fixed-centers3d +.196522/-.076409/+.116501pp;
no stable full-ellipsoid advantage over isotropy. The center-lifting change does
not supply evidence that anchor/region conflict explains all prior failures.
Primary zero-V PCK8=.651677441; V-path dependency is not PE-shape attribution.

Report docs/round46_native3d_report.md. Stop promotion of this frozen global
Gaussian warp; no R47 candidate is frozen or launched. Subsequent design must
address what Gaussian centers/orientations/scales add beyond ordinary nonlinear
position maps inside a complete Transformer. Rendering remains optional; matching
is one task, not generality. Full geometry/robustness/cost, a second task and unused
joint confirmation remain outstanding. Fresh cohort untouched; goal ACTIVE/unmet.

R47 COMPLETE: ROUND47_CONDITIONAL_GAUSSIAN_PROTOCOL.md,
models/round47_conditional_gaussian.py and experiments/round47_conditional_*.py.
The preceding goal turn was progress: R46 completed all55 new trainings/old_eval,
independent metric replay and a terminal report, rejecting its promotion. No R46
process is live. R47 changes global Gaussian warps to query-conditioned Gaussian
positional channels inside the complete Transformer. One shared linear predictor
from the current normalized token outputs head-specific relative means and SPD
precisions; an exact quadratic Q/K lift uses5/9 positional channels in2D/3D.
No renderer, physical input covariance, rotary hybrid or final geometric shortcut.

This has direct Gaussian-attention/PE precedents (EMNLP2018 localness, ICLR2020
appendixE generalized quadratic PE, ICCV2021 SMCA); no novelty theorem is claimed.
The new question is whether conditional Gaussian constraints improve on ordinary
dynamic quadratic PE at the same capacity and initial first derivative. Six
families in2D/3D (conditional/diagonal/isotropic/static/fixed/free-quadratic) plus
NoPE give65new runs. Reuse30 fully checked R46 point/warp/MLP/Fourier controls.
Primary conditional3d remains fixed; secondary2D cannot replace a failing primary.
Same old data bytes, shared initialization,1000updates/batches and selection.
Separate full-vs-component attribution remains continuous; no new static gates.

Freeze/preflight handle73320 TERMINAL exit0, all6CPU tests passed, including65
paired initial full-network Gaussian functions, independent nonzero pair energies,
SPD/shape gradients, free-quadratic first-derivative match, input scope and
translation/permutation plus joint coordinate/precision identities. These are
implementation checks, not end-to-end generalization. Evidence
results/round47_conditional/attempt1, lock SHA256
0f1c841699d8f3f6b9df3f606ee4782c6aac6ebbe0e45bc1fe4d2886a438ab0a.
Controller tool handle13639 is TERMINAL exit0; historical parent/child identities
are in execution_handles.json and queue_status.json. Do not restart historical PIDs.
All controller stages completed: CUDA smoke ->5seed training shards ->seal ->
5exports ->metrics. Results below. No unused cohort access; full goal ACTIVE/unmet.

R47 CUDA smoke completed exit0 for all13roles: maxP1.563e-7,0winner changes,
all initial full-network fixed-Gaussian/NoPE reference differences0. Five training
children launched on verified idle GPUs1..5: seeds11/22/33/44/55,
PIDs40608/40609/40610/40612/40613. Controller host PID40250/tool13639 is the
historical handle; the children are not separate tool sessions. All65new runs
and the controller have now exited0; do not restart historical PIDs.

Bounded old-VAL input-geometry diagnosis was separately frozen before execution:
experiments/round47_geometry_diagnostic.py, evidence
results/round47_conditional/geometry_diagnostic1. Handle89911 TERMINAL exit0.
All64oldVAL pairs/10556valid queries/432keys, no fitting or R47 changes. Untrained
nearest-key PCK8: projected2d93.4255%,native3d55.3050%,native3d_XY54.8030%.
Dropping Z does not recover the loss. Label-weighted predicted-anchor residual
Z variance fraction87.70%,covariance condition25.70; these are not true3D errors.
Single old VAL scene, not general confirmation or a trained-model comparison.
docs/round47_scope_note.md records exact scope, direct Gaussian-attention priors,
and the algebraic fact that centered3D coordinates do not retain a fixed camera
origin's perspective projection. Same-input3D comparisons remain valid and all
frozen R47 rules remain unchanged. Native point-cloud classification is a possible
second task; only official metadata/literature reviewed, no data downloaded/read.

R47 terminal result: all65new+30reused models sealed/exported, all91200 metric
frame rows replayed with0 PCK4/PCK8/valid-count discrepancies; maxCE frame-sum
difference2.693e-4. New independent sampled P error3.309e-6,0winner changes,
no warnings. All65shared weights changed, all50models with learned position
parameters changed every positional tensor. Best steps400:1,500:1,600:3,700:6,
800:6,900:19,1000:29; no convergence claim. Peak655.376MiB, summed per-run
training/selection2597.096s. Tool13639 directly returned exit0,13children all0,
active[]. No main R47 job remains live; execution_completion.json is authoritative.

Primary conditional3d PCK8=.834798111 versus rope3d .851576467 (-1.677836pp),
0/5positive seeds/scenes; versus mlp3d .882505583 (-4.770747pp),also0/5 each.
No promotion. Full-minus-fixed/static/isotropic/diagonal3d +7.659334/+3.948869/
+3.437183/+.734369pp; all5/5seeds,scenes5/5 except diagonal4/5. Full-minus-free
quadratic3d +.744784pp,5/5seeds4/5scenes. Conditional ellipsoid structure has a
local effect, but does not beat the stronger methods. The failure is not a
floating-point audit veto or solely the1pp utility criterion.

Prespecified secondary conditional2d PCK8=.923258078 vs rope2d .924121336:
-.086326pp,2/5seeds2/5scenes,CE+.7915%. PCK4 improves+.850468pp,4/5seeds5/5
scenes; EPE improves-.168147px(-3.5837%),5/5seeds5/5scenes,with larger effect in
Harvard. These recorded metrics show a tradeoff, not blanket failure or a renamed
successful primary. Full2D beats free_quadratic2d by2.290998pp,5/5seeds/scenes.
Versus isotropic/diagonal2d effects only+.045777/+.022268pp. Full/free comparison
changes SPD plus parameter coupling; it does not isolate SPD as the sole cause.

Prospective synthetic CUDA cost probe source experiments/round47_cost_probe.py,
results/round47_conditional/cost_probe1. Freeze9095 and run44250 TERMINAL exit0,
GPU1 verified idle. Batch4,N432/1024,six selected models,five warmups+20rotating
repeats per condition. Conditional2d inference14.277/14.324ms vs rope2d9.581/
9.682ms (about1.48--1.49x); forward/backward38.744/38.596ms vs31.297/30.653ms.
N1024 inference incremental allocated244.5MiB vs90.5MiB; forward/backward
505.32vs468.38MiB. Small parameter count is not free runtime/memory. Shared-host
probe, not production throughput/task-length generalization. No probe jobs live.

Full report docs/round47_conditional_gaussian_report.md. Retire promotion of this
frozen primary; keep its genuine component and secondary fine-localization
signals. Next research direction: independent native3D Transformer point-cloud
classification, provisionally ModelNet40, with official train-only development
and official test reserved; no R48 spec frozen or data downloaded yet. Do not
keep patching matching frequencies/widths or let another task erase matching
limitations. Cost evidence is recorded above; applicable geometry/robustness,
cross-task utility and unused joint confirmation remain unmet. All existing
unused confirmation scenes untouched, goal ACTIVE.

R48 STARTED: ROUND48_NATIVE_POINT_PE_PROTOCOL.md and
models/round48_point_transformer.py. The user's concern about rendering-task
drift remains controlling: GS center/scale/orientation parameterization must
be tested inside actual Transformer positional channels against matched strong
controls. R41/R42 rendering results never establish that claim. R43--R47 returned
to that question; their negative results are preserved, not erased by a new task.

This turn made concrete progress: pinned public ModelNet40 HDF5 acquisition,
new paired four-block self-attention classifier, data/train/audit/controller
implementations, frozen prospective protocol and six passing synthetic tests.
The unchanged R47 conditional3D Gaussian positional form is primary; diagonal,
isotropic, static, fixed, equal-capacity free quadratic, RoPE, Fourier/MLP warp
and NoPE give ten roles x five seeds. XYZ content and all non-PE weights/training
streams are shared. NoPE still receives XYZ. No renderer or task-specific loss.
This global scalar-attention model is not a canonical Point Transformer replica.

Acquisition tool40422 TERMINAL exit0 (chunk28a56c), archive435335111bytes,
SHA256 f01b8189281fae5790e39deb9f3eca86e446b771bdc665c6ad05f28d039b20e7,
HF revision d5dc795541800feeb7a4b3bd3142729a0d2adf7a. The original Stanford
server timed out; pinned mirror is referenced by NVlabs research code. Published
mirror hash verified, no independent original-server digest. Acquisition opened
only ZIP metadata. Main freeze occurred before HDF5 decoding. Preparation opened
only five official TRAIN HDF5 files; official TEST members remain unopened.
Stratified declared split:7910 fit/965 val/965 dev_eval, all40classes, no byte-
identical clouds across partitions. All selected models must be sealed before
model dev_eval; its raw arrays were automatically partitioned during preparation.
Neither this development split nor five seeds constitutes fresh joint confirmation.

Main lock af09505eb0db1ea8becbe50da9857298b898f619637772a9737b63ea5a0d546f,
19 frozen sources; evidence results/round48_modelnet/attempt1. Freeze/preflight
tool46527 TERMINAL exit0 (chunk3bd625), six CPU tests passed. CUDA smoke child9316
and data child9549 exited0. All ten CUDA roles passed direct NumPy positional
checks: max probability error2.0628e-7,zero winner changes. Synthetic batch32,
N512 training-step medians:RoPE35.524ms,conditional61.894ms (5warmups/10repeats),
not task utility or production throughput. No scientific setting changed afterward.

Controller tool handle24005 is LIVE, parent hostPID9006; do not restart it.
Its actual child state is in queue_status.json and tool output; initial handles
are saved in execution_handles.json. Currently four seed shards were launched:
seed11/PID9725/GPU1,seed22/PID9726/GPU2,seed33/PID9727/GPU4,
seed44/PID9728/GPU5;seed55 pending. GPU3 was occupied by another job and skipped.
These child PIDs are not separate tool handles. Each role runs120epochs,batch32,
512points,common AdamW/warmup/cosine/augmentation/selection; eight GPU-hours per
seed shard maximum. Controller will seal50models,export prescribed development
variants and replay metrics. Read the LIVE handle rather than starting duplicates.
No R48 utility conclusion yet; all existing confirmation scenes and official
ModelNet test remain unused. Full research goal ACTIVE/unmet.

R48 continuation: controller24005 re-polled live (tool returned the same active
session, not a restart). Four seed shards are progressing through the first of
ten roles each, approximately epoch45/120 at the latest observation; seed55 is
still pending. No role-level comparison is valid from these unmatched partial
seed results. Main frozen sources/configuration remain unchanged.

Bounded literature/source check completed while training. At author commit
3e08aa732a81033f1c07c0b0533c334fe448e0d3, GaussianMamba3D replaces the patch
encoder, but retains a separate3->128->trans_dim MLP positional encoder
(Mamba3D/models/Mamba3D.py:590,604,717). Therefore its Gaussian encoder results
are not a direct test of Gaussian Transformer PE. The stored-mean covariance
gradient hook (PointNet/encoder.py:1058) also cannot simply be called the natural
gradient of our shared conditional predictor. Sources and limits are recorded
in results/round48_modelnet/preconditioning_scope1/result.json.

A data-free analytic counterexample was fixed and executed in
experiments/round48_preconditioning_scope.py (exit0,chunk8120bf). For3 scalar
Gaussian means sharing2 weights, pullback Fisher J^T F J=[[3,2],[2,6]], weight
loss gradient[3,1]. The true natural-gradient direction is[8/7,-3/14], whereas
preconditioning outputs before backprop gives[2,.75], not even collinear. At
equal exact KL1e-4, linear loss decreases are.0253546vs.0206474. This only
rejects an unjustified optimizer equivalence; it does not show that the output
hook is useless, explain R47/R48 failures, or require a new optimization gate.
No task data read, no R48 change, no new candidate or result promotion.

R48 progress: the first4/50 trainings are COMPLETE:conditional/11,diagonal/22,
isotropic/33,static/44. Each executed120epochs/29760updates; selected epochs
115/105/95/60 respectively under the frozen minimum-val-CE rule. All shared
weights and positional parameter tensors changed. Per-run training/selection
times1945--1982seconds, peak allocated1.490--1.509GB (decimal bytes), not a formal
inference benchmark. A first-completion check (tool5464 TERMINAL exit0,chunkd64b80)
verified output hashes, budget/selection and frozen19sources, then replayed the
selected validation CE and sampled first-layer positional probabilities. Maximum
probability difference1.9632e-6,zero winner changes; max val CE difference9.30e-9.
Snapshot:results/round48_modelnet/attempt1/early_training_check.json. This covers
four runs only, does not seal the full experiment or access dev_eval, and does
not rank methods across different seeds. The full planned audit is still pending.

Controller24005 remains LIVE with the same four training child PIDs; it has
automatically started diagonal/11,isotropic/22,static/33,fixed/44. Seed55 is
still pending. No R48 source/configuration change, no fresh confirmation access,
no method utility conclusion. Continue the existing handle rather than relaunch.

R48 reporting preparation COMPLETE while training continues: new standalone
experiments/round48_report.py renders all10roles, all5seed differences, all7
prespecified conditions, parameter counts and training/selection cost. Whiskers
are seed ranges, not confidence intervals; robustness plots use each method's
own base and accompany absolute OA tables. The reporter preserves the frozen
decision and requires all50sealed models plus terminal successful controller
state. It is not imported by the training pipeline and does not change any of
the19frozen scientific sources or the candidate/selection rules.

Synthetic-only rendering smoke10349 exited0 (chunkc18dd8), plots inspected;
small reporting refinements preserved that source/output and smoke21787 exited0
(chunk988fe1). results/round48_modelnet/reporting_smoke1 and reporting_smoke2
contain ARTIFICIAL FIXTURES, NOT task results; never cite their numbers as
experimental evidence. A direct incomplete-run guard check (tool25116) rejected
production reporting while queue24005 is live, before reading task metrics or
prediction arrays; report1 does not exist. When the main controller completes,
run the reporter with the existing Python and MPLCONFIGDIR=/tmp/grope-round48-matplotlib,
then inspect real plots and write the scientific interpretation. Goal ACTIVE.

R48 live progress: fifth complete run fixed/44 executed120epochs/29760updates,
selected epoch85, shared weights changed, zero learned PE parameters by design;
training/selection1305.58seconds. Total5/50 complete. Controller24005 was directly
polled live (chunkcbe4d0); seed44 now trains free_quadratic, and other shards
continue their second roles. Same four owned GPUs/PIDs,seed55 pending. No new
audit gate or model comparison from selection data; dev_eval and confirmation
remain unopened by the evaluator. Continue existing controller.

R48 data-identity check COMPLETE: experiments/round48_source_identity.py and
results/round48_modelnet/source_identity1. Prospective metadata-only plan,
exit0(chunk34b65e),0.54seconds: all9840 original mesh filename-stem IDs unique;
fit7910/val965/dev_eval965 have zero cross-partition shared IDs. All40class
counts match the declared manifest in every partition. Only five official-train
ID-list JSONs and shape_names.txt were decoded; no HDF5/prediction/test members
opened. This checks source-object names, not geometric near-duplicates with
different names, and does not change the split or add a model-promotion gate.

R48 progress:8/50 trainings complete. Four paired seed groups are now available:
11 conditional/diagonal,22 diagonal/isotropic,33 isotropic/static,44 static/fixed.
Actual initial shared-weight digests and full sampling/augmentation stream
digests agree within every group, and differ across seeds. All completed runs
used120epochs/29760updates; all19frozen scientific sources unchanged. Metadata
snapshot pairing_snapshot1.json (exit0,chunkc2656e); no predictions or dev_eval
read and no cross-method ranking. Full50-run audit remains pending.

GPU3 became genuinely idle; controller24005 observed two free samples and
automatically launched seed55,hostPID104142,UUID
GPU-bed502d0-6c7a-c66d-42b6-8aa08676eec7. The launch was directly returned by
the existing live handle (chunkc0b694); fixed/55 has started training. All5seed
shards are now active on GPUs1/2/4/5/3, pending[]. Other four child PIDs remain
9725/9726/9727/9728. Latest parent poll chunk1cf300 confirms live; do not restart.
Seeds11/22/33 now train isotropic/static/fixed,seed44 free_quadratic. Full goal
ACTIVE/unmet; no task utility claim or fresh confirmation access.

R48 verified-wait continuation: controller24005 directly returned LIVE again
(chunk87650a). Now10/50 trainings are complete; new free_quadratic/44 and
fixed/55 both executed120epochs/29760updates, with selected epochs110/70.
Their shared weights changed; free_quadratic/44 positional tensors changed.
All five owned seed shards remain active. Seeds11/22/33 continue isotropic/
static/fixed; seed44 now trains RoPE,seed55 free_quadratic. No restart, changed
scientific setting, development-evaluation ranking, or confirmation access.
Continue the existing controller through all50 runs and its planned audit/export;
the research objective remains ACTIVE and unachieved.

R48 interpretation clarification COMPLETE, before dev_eval: data-free
experiments/round48_control_scope.py, results/round48_modelnet/control_scope1,
tool57502 TERMINAL exit0(chunk6feaff). The equal-capacity free quadratic control
has precision diagonal in(-5/16,7/16) and linear coefficients in(-1/4,1/4).
Finite Gaussian raw outputs realize P=exp(2)/16 I=.461816I, outside the free
diagonal bound; free outputs realize P=-I/16, impossible for Gaussian SPD.
At a precision feasible for both, P=e/16 I, Gaussian mean(2,0,0) gives
b=(e/8,0,0)=.339785 on x, beyond the free linear range. Thus the bounded
coefficient families are not nested. The comparison changes nonlinear maps
and attainable ranges as well as SPD and center-precision coupling; parameter
count/initial-tangent matching does not isolate SPD. No trained coefficients,
task data or checkpoints were inspected. This does not show a bound is active
in practice, diagnose R47/R48 utility, change R48, or create an additional gate.
All19frozen sources unchanged. The standalone reporter now states this limit;
its calculation/plotting/selection code and earlier synthetic smoke artifacts
are unchanged. Controller24005 remains LIVE;11/50 trainings complete, seed33
has moved from fixed to free_quadratic. Full objective ACTIVE/unachieved.

R48 verified wait: live controller24005 polled repeatedly, latest chunk12187d;
14/50 trainings now complete. Additional fixed/33,isotropic/11,static/22,rope/44
all ran120epochs/29760updates and changed shared weights (metadata read chunk
cb1e90); selected epochs85/55/65/90. Current five live shards train static/11,
fixed/22,free_quadratic/33,fourier/44,free_quadratic/55. Original owned PIDs and
parent9006 unchanged. No utility ranking or additional gate; all50-model audit
and dev_eval remain pending, confirmation remains unused. Continue handle24005.

R48 verified-wait continuation: same controller24005 still LIVE (chunke1f441),
16/50 trainings complete. New free_quadratic/33 and /55 completed120epochs/
29760updates, selected epochs120/90, shared and all positional tensors changed
(metadata chunkbb01e6). A selected final epoch is not a convergence claim.
Current roles:static/11,fixed/22,rope/33,fourier/44,rope/55; five shards continue.
No model/configuration changes this turn, no dev_eval exports yet, no additional
audit or confirmation access. Keep waiting on24005; objective ACTIVE/unmet.

R48 verified wait: controller24005 directly returned LIVE (chunk3d5586), now
20/50 trainings complete. New fixed/22,fourier/44,static/11,rope/55 all completed
120epochs/29760updates, selected epochs65/85/75/90; all shared weights changed,
and learned positional tensors changed where present (metadata chunke6a0b1).
All19frozen sources still match. Current roles:fixed/11,free_quadratic/22,
rope/33,mlp/44,fourier/55. Five shards active, no dev_eval exports, no setting
change or fresh confirmation access. Full audit/evaluation and goal remain unmet.

R48 coverage review COMPLETE during training: source/report hashes and findings
in results/round48_modelnet/coverage_review1.json (chunkbe3302). R40 decoupled
AbsRel .164532846 versus point .161017277 is +2.1833% WORSE, not an improvement.
R40 already combined center3D RoPE with physical shape-descriptor distance, and
its center_rbf role combined RoPE with a learned scalar center-distance penalty.
That residual-depth model and its reverse-V anomaly do not establish full
Transformer utility. R44 replaced RoPE by normalize(content+Gaussian vector);
R45/R46 warped RoPE coordinates; R47/R48 Gaussian roles use unrotated content
plus conditional quadratic channels. These inspected roles do not directly test
unwarped center RoPE together with the conditional Gaussian spatial prior.

This leaves an UNIMPLEMENTED candidate question: tau*<R(x_i)q_i,R(x_j)k_j>
-.5*(delta-m_i)^T P_i*(delta-m_i), with the existing quadratic positional lift.
It could test complementary phase/content interaction and adaptive locality,
against each part alone, fixed/isotropic priors, free quadratic plus RoPE and
retained strong coordinate-map controls. A coverage gap is not positive evidence
or novelty; R40's simpler combination and all negatives remain. This is not a
frozen next round, new gate, or license to modify R48. Decide whether it is worth
testing only after the full current result. No new task arrays/checkpoints or
confirmation were read for the review; existing aggregate reports were read.

R48 live controller24005 confirmed again (chunk7e4838);26/50 trainings complete.
Additional rope/33,mlp/44,free_quadratic/22,fixed/11,fourier/55,fourier/33 all
ran120epochs/29760updates; selected epochs120/85/70/75/100/85. Shared weights and
learned PE tensors changed where present (metadata chunks3d531b/3ba163).
All19frozen sources still match. Current roles:free_quadratic/11,rope/22,mlp/33,
nope/44,mlp/55. Same five shards active, no dev_eval exports or confirmation
access. Full audit/evaluation and full research objective remain unfinished.

Focused primary-source literature review COMPLETE during R48 training:
results/round48_modelnet/literature_scope1.json (chunkf057bb). Four inspected
sources: BAM arXiv2505.22842v4, MoPE2606.01258v1, HeatTok2608.22485v1, and
Localized Gaussians for Point Clouds Correspondence2409.13291v2/STAG2024.
BAM already uses a generalized-Gaussian relative-index log-prior. MoPE's actual
experiment uses absolute-origin Gaussian-envelope positional vectors; Section7
explicitly leaves its proposed Morlet-RoPE validation for future work. Its small
single-seed EGA combination is not evidence for our conditional3D joint score.
HeatTok directly encodes Gaussian center/scale/orientation using blended phase
coefficients; our algebra C^2+S^2=1-2a(1-a)(1-cos(u-v)) shows this blend is not
generally unit modulus. This is not a measurement of its trained attenuation or
a rejection of its task results. The STAG paper mixes Gaussian-only geometry
heads with RoPE dot-product heads, and reports substantial noisy-data weakness;
it does not test the same-head conditional anisotropic joint prior here.
Do not claim novelty from our repository coverage gap, import these task claims
as our evidence, or conflate amplitude modulation with additive log-priors. The
review is bounded, not exhaustive; no external implementation was reproduced,
no new gate was added, and no task arrays/checkpoints/confirmation were read.
All19R48frozen sources remain unchanged. Finish R48 before choosing a new round.

R48 COMPLETE: controller24005 TERMINAL exit0 (chunkeb4af0), all50trainings and
all50exports finished; queue state complete/active[]. All14child stages exited0.
All training/export PIDs, including final train55/104142 and export55/76953,
are terminal. Observation cells1619/1623/1626/1630/1633/1636/1639/1642/1646 are
also terminal. Do not restart any old job. Report39867 exited0(chunkee8538),
plots inspected; results/round48_modelnet/attempt1/report1/report.md and
new docs/round48_native_point_report.md contain results/interpretation.

All50models ran120epochs/29760updates. Within-seed shared initialization and
actual sampling/augmentation digests matched; frozen selection rule held.
Shared weights changed50/50; all positional tensors changed in35/35learned-PE
models. Max sampled P error6.92553e-6,zero winner changes/zero warnings; max
validation CE replay error1.32635e-8. Full dev_eval replay50models*965objects*
7conditions gives350metric rows,zero count discrepancies,max CE error1.15086e-7.
Permutation max logits1.56164e-5,zero class flips. Independent formula paths,
not an independently authored complete network/lab reproduction. All19frozen
sources still match. Sum training+selection77257.650seconds; peak1615664640B;
shared-host timings, not a throughput benchmark. Two models selected epoch120,
so no universal convergence claim. Officialtest and old confirmation untouched.

R48 primary conditional OA88.6217617/MCA83.6360874/CE.443558689.
Means OA:diagonal89.0777202,isotropic88.5388601,static87.9378238,fixed87.9585492,
free_quadratic88.6424870,rope88.5181347,fourier88.4352332,mlp88.8911917,
nope87.4196891 (OA/MCA in percent). Conditional OA differences in pp:
vsrope+.103626943(3/5seeds),fourier+.186528497(2/5plus2ties),
mlp-.269430052(1/5),free-.020725389(1/5plus1tie),nope+1.202072539(5/5).
Vsdiagonal-.455958549(2/5),isotropic+.082901554(3/5),static+.683937824(3/5),
fixed+.663212435(4/5). No primary promotion or renamed component winner.
Near ties are not material failures, but no stable strong-control advantage
emerges even without treating the1pp development margin as a binary theorem.
Conditional mean train+selection32.78min vsRoPE19.28/MLP22.26. Generality unmet;
R47's same-3D-input matching negative remains and cannot be erased by this task.

R48 robustness:conditional OA Y90=88.725389,jitter02=88.476684,
points256=87.958549,points1024=89.139896,permuted=base. All roles collapse under
the prescribed Rz45@Rx30 condition to OA4.4974--8.0000%;conditional6.549223.
OnlyY-axis rotations were trained; no SO3 invariance claim. A large shared
failure warranted one bounded check, not a new gate or an angle/temperature
sweep. New experiments/round48_rotation_check.py and results/round48_modelnet/
rotation_check1 saved plan/source before checking task arrays or replaying models.
Tool64394 TERMINAL exit0(chunke96756),4.29seconds. All965olddev inputs match an
independent coordinate-wise rotation within1.06130e-7; orthogonality5.45877e-8,
det.9999999924, inverse error1.31837e-7, checked distance error1.63651e-7.
Up-axis tilt52.23875deg. Identity-hash-selected32objects,seed11,conditional/
rope/nope CPU replay matches saved base/SO3 predictions within6.91414e-6 logits,
zero class flips; inverse rotation restores base predictions in this sample.
No observed transform/replay defect; retain common orientation failure without
claiming its causal training explanation. No retraining,new task estimate,
fresh data or original-result replacement. Diagnostic checked the frozen inputs
and19scientific sources again; all unchanged.

NEXT: decide/freeze a bounded joint-phase/locality experiment on existing
matching development data, keeping the full objective and original negatives.
The exact question is whether center RoPE's content-relative phase and a learned
Gaussian spatial preference complement each other inside one Q/K score. The
existing quadratic lift supplies9extra channels in3D. It is a hypothesis, not
an implemented/proven method or novelty claim. Compare both parts alone,
RoPE+fixed/compact Gaussian priors,RoPE+equal-capacity free quadratic and the
strong MLP/Fourier/NoPE controls with matched budgets/initialization/data.
The full-Gaussian/free comparison includes nonlinear/range differences, not
isolated SPD. A gain shared by a fixed prior is not evidence for adaptive GS
shape. If there is a stable matched result, validate the same formula on native
point clouds and then untouched confirmation; do not relabel old dev as fresh.
No R49 protocol/model/training exists yet. No new module/loss/renderer is needed
for this proposed score change. Goal ACTIVE/unachieved, no active R48 jobs.
