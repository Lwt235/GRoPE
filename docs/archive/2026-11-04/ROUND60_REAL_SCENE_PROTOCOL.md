# R60: unchanged Gaussian-relation PE on real observed point clouds

2026-11-02. Implements docs/round60_real_scene_plan.md before any new target
model outcome. R59 base/low-overlap positives motivate this development study;
its sparse uncertainty and all historical negatives remain. No general utility,
novelty, optimized-3DGS or fresh-confirmation claim.

H: the same learned Gaussian invariant relations help real local correspondence
beyond distance, local descriptors, triplet angles and complete raw matrices.
Alternatives: gains depend on synthetic CAD views; ordinary controls suffice on
real sampling/noise; source weights require target adaptation. Direct transfer
AND a single fixed adaptation budget are specified now. If adapted results lack
strong-control gains, do not claim real-task benefit. If only adaptation helps,
do not claim direct transfer. A positive local result still needs cross-task and
untouched confirmation; no automatic global promotion.

## Fixed inputs and information boundary

Exact R46 old pair IDs:576fit/64val/480old_eval across3/1/5 SUN3D scenes. See the
plan's explicit scene list. Capture selection/download manifests, native depth,
intrinsics and pose identities before data generation. No new downloads, R6
reserved scenes or official ModelNet test. All target observations are OLD
development data even though this task differs from the old RGB matching task.

Decode native sensor depth with the existing SUN3D reader; use finite positive
samples only and zero-based native intrinsics. This is a DEPTH point-cloud task,
not an RGB-only system. No appearance input is added. Source-frame IDs within a
pair have distinct depth files; association timing remains the dataset's recorded
nearest-depth association, not a new synchronization estimate.

Independently center both complete valid camera point clouds. Divide BOTH by
sqrt(mean of their two centered mean squared radii),floored at1e-6 only for
numerical safety. This normalization receives no pose. Sample512points without
replacement by hash('R60-native-observed-support-v1:pairID:realization'). Sparse
uses the same first256; do not change subsequent random draws. Fewer than512
native valid points excludes the WHOLE pair for every method/condition, with IDs
and reasons reported. Zero-overlap pairs otherwise remain.

Use the original R58 tokenizer:64FPS anchors,32nearest observed points,mean and
covariance+.02^2 I,11local features,point adjacency radius.05,query maxoverlap>=.3,
soft targets proportional to overlap*1[overlap>=.1],center PCK<=.1. Units are
normalized scene units,not metres. Pose transforms sampled ORIGINAL sensor
points to world scoring coordinates,subtracting the first camera translation
and dividing by the same scale. Sensor noise remains in these scoring points;
they are not noiseless surface truth. Inputs are never aligned by ground truth.
The model API still accepts only(features,centers,covariance).

Training has two fixed independent point realizations,alternated by epoch,with
independent uniform SO(3) and[-.5,.5]^3 translations in normalized local frames.
Validation and old_eval keep native camera axes after centering/scaling. Base
and sparse reuse IDs,scale and sampling order; supports and eligible populations
can change,so absolute cross-condition scores are not fixed-query robustness.

## Models, fitting and exports

EXACT original R58/R59 model factories. Candidate invariant_mlp11->96->96,
GELU,10368PE parameters; all9 R59 controls retained. Seeds11/22/33. Source
checkpoints are the30 exact R59 selections. All source roles used the same
source-training protocol; no new source fit. Target initial backbone weights
DIFFER across roles because source training already occurred. Record each source
checkpoint and actual starting state; do not assert common target initialization.

Direct transfer updates no weights. Adapt ALL30 models with60epochs,batch32,
AdamW lr1e-4->1e-6,3epoch warmup/cosine,wd.01,clip1,validation every3including0,
strict minimum bidirectional validationCE/earlier tie.1080updates if all576fit
pairs are retained; otherwise report the input-rule-derived count. Original R58
fit function is reused via a scoped adapter for root/config/factory/cache/scope;
loss,schedule form,optimizer,sampling RNG(seed+58000),and selection logic are
unchanged. No target budget/temperature/frequency/width search.

Freeze all30 target selections BEFORE generating/exporting old_eval results,
including direct-transfer results. Export both phases on base512 and sparse256:
120groups total(10roles x3seeds x2phases x2conditions). All IDs and masks matched
within a condition. Record both global query-weighted and per-scene metrics;
primary aggregation is the5-scene macro regionR@1. Report centerPCK,softEPE,CE,
coverage,per-scene and per-seed paired differences. Nearby frames are correlated;
do not manufacture confidence from a large query count. Any intervals must be
explicitly conditional and respect the small number of scenes/models.

## Proportionate checks and execution

Synthetic checks for depth backprojection,pose-independent inputs,sparse nesting,
normalization/world geometry and scoped adapter restoration. Actual bounded
checks recompute sampled support/overlap independently; replay source identities
and finite CUDA gradients/updates on old fit inputs. No repeated all-formula PE
audit or full-query exact-invariance gate. Keep R59 angle edge behavior as recorded.

Postfit check30 selections,actual streams/source states/cache identities,stored
validationCE via independent NumPy and bounded checkpoint replay. Recompute all
120 exported metric groups with NumPy,including scene aggregation. This is the
same author's separate arithmetic,not an independent laboratory replication.

Use CPU4 and live-confirmed idle GPUs,two observations5seconds apart,max6workers.
CUDA probe precedes launch. Preserve others' jobs and system GLIBC. Ordinary
pre-run fixes retain failed source versions; no new scientific round for a typo.
