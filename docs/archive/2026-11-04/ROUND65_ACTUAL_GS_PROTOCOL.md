# R65: native optimized Gaussian parameters as Transformer PE

Status: fixed scientific specification; execution requires the validated lock
at results/round65_actual_gs/attempt1/lock.json. This document alone does not
establish that a fit has run. Freeze source/configuration identities after implementation checks and before
decoding training vertex arrays or performing task-model training. Public file
headers, training identities and byte metadata are preparation evidence only.

## Question and competing explanations

Does the existing 11-dimensional Gaussian relation PE improve classification
when optimized 3DGS centers and covariances are directly available? R62/R64
failed against strong RGB controls. This experiment removes RGB parameter
prediction from the pipeline; it does not cancel those failures or establish
that prediction was their unique cause.

Competing explanations are that the attributes themselves suffice through an
ordinary MLP, that center relationships suffice, or that any gain is specific
to one source domain or sampling density. Retain the exact R58/R61 relation
formula and a single 11->96->96 MLP. No new rendering, auxiliary reconstruction
loss, pretrained backbone, teacher, feature pyramid or learned Gaussian head.

## Data and protection

Use all 7,757 official TRAIN entries of zxAi/GSModel60, revision
e6407400f768fc7c0d8f8d1b2ed112739b38e9dc: 3,246 MACGS-derived and 4,511
ModelNet_Splats-derived objects, all 60 classes. The latter identities have
all been checked against ShapeSplat's published ModelNet training list.
Published LFS identities and file sizes are recorded in train_catalog.json.
Per the user's directory instruction, raw bytes are stored under
/mnt/f70f6709-366e-49a0-861f-497645d98975/users/liwt/data/GSModel60;
preprocessed arrays go under the same data directory's
GRoPE/round65_actual_gs_cache. Repository results retain the manifest link.
All 7,757 byte identities are distinct. Metadata provides no independent
physical-instance grouping for nonidentical MACGS reconstructions; record
that limitation rather than equating file independence with physical identity.

The complete selected TRAIN population is development data, declared before
vertex decoding. None of it will later be relabelled fresh confirmation.
Keep GSModel60 TEST, original ModelNet TEST and the five reserved SUN3D scenes
unread. This is not a published-test benchmark reproduction. ModelNet objects
may overlap the old point-cloud development population; do not claim new
independent confirmation simply because their representation changed.

Within each class, sort file paths by SHA256('R65-gs-train-split-v1:'+path).
Let n=floor(class_count/10). First n validation, next n development evaluation,
all remaining fitting. No label-dependent exclusion or post-result resampling.
Select all checkpoints using fit/validation only before evaluating dev_eval.

Decode the original binary PLY centers, log scales, wxyz quaternions, opacity
logits and three DC SH coefficients. Normalize the quaternion, reconstruct its
rotation, and form physical covariance R diag(exp(2*log_scale)) R^T. Center the
whole object by its vertex mean and divide positions and scales by the same
RMS radius. Add 1e-8 I in these normalized units to covariance, uniformly for
all methods, and report its relative contribution. No learned or tuned shape
normalization. Keep covariance in FP64. Reject malformed/nonfinite source
records with a versioned failure; never silently drop an object.

For every object retain 2,048 native Gaussians selected uniformly, using a
deterministic per-path hash seed, without replacement unless fewer exist.
Do not aggregate them into synthetic patches. Store input identities and
sampling indices. Higher-order SH is not used by any method. DC appearance
and sigmoid opacity are common content, not geometry uncertainty or visibility
labels. No numerical vertex arrays have been read as of this plan's creation.

## Matched models

Common content per token: normalized center (3), symmetric vectorization of
half log(Sigma/.04) (6), sigmoid opacity mapped to [-1,1] (1), and DC SH times
0.28209479177387814 (3). Every role receives these same 13 channels. Therefore
the center controls still have access to shape through the content branch.
Common stem 13->96->96; four self-attention blocks of width96, two heads,
FFN ratio2; final LayerNorm, concatenated mean/max pooling, 192->128->60
classifier with dropout.3. No hierarchy or grouping module.

Five roles, each seeds 11/22/33, trained from scratch:

- gaussian_relation (primary): exact SupportPosition('invariant_mlp') from
  R58. Eleven invariant center/shape relations, 11->96->96, shared over layers.
- direct_mlp (primary strong control): relative displacement/.2 (3), its
  norm/.2 (1), and the same six log-covariance channels from each endpoint
  (12); 16->96->96. These descriptors retain all relative center/covariance
  information used by the Gaussian construction. No camera/calibration input
  difference is present in this experiment.
- distance_mlp: center distance/.2, 1->96->96, same attention insertion.
- center_rope: the fixed spherical R48 frequency schedule on 12*center.
- nope: no explicit attention PE; all common geometric content remains.

Relation insertion is (q dot k + q dot r_ij)/sqrt(48), with ordinary raw Q/K.
RoPE rotates Q/K before the dot product. Common parameters are initialized
identically within each seed before role-specific parameter draws. Record
actual parameter counts; do not add unused parameters to fake exact capacity.
The direct MLP has slightly more PE parameters than the primary.

## Training and selection

Fixed 120epochs, batch16, 512 sampled tokens per object; deterministic paired
object order, token selection and augmentation for all roles of a seed.
AdamW learning rate .001, weight decay .01; five-epoch linear warmup then
cosine to .00001. Label smoothing .1, gradient clipping1. Same full SO(3)
rotation augmentation, generated from normalized Gaussian quaternions, for
centers and covariances together. Center jitter std.01 clipped at .05; shapes
remain unchanged for this augmentation. Unlike old Y-only classification
training, SO(3) augmentation is fixed prospectively for both source domains;
rotation evaluations do not represent unseen rotation-distribution transfer.

Evaluate ordinary (unsmoothed) validation CE every five epochs including0;
earliest strict minimum selects the checkpoint. No per-role temperature,
training duration or learning-rate tuning after results. Preserve full curves.
Use idle GPUs, verified by live occupancy and CUDA access. Do not interfere
with other users. Record peak memory and shared-host elapsed time; these are
not controlled efficiency benchmarks.

## Evaluation, interpretation and decisions

Primary: mean-class accuracy on the internal development evaluation split.
Also report overall accuracy and CE, all three paired seeds, per-class and
two source-domain summaries. Fixed conditions: base512, sparse256, dense1024,
center jitter std.02 clipped.1, fixed tilt Rz(pi/4)Rx(pi/6), and permutation.
All variants use the same objects; change covariance with rigid rotations.
Numerical permutation/geometry checks are implementation evidence, not a
substitute for task robustness. Tilt probes model behavior after SO(3)
training, not previously unseen rotation support.

A useful development lead requires beating direct_mlp and both center PE
controls on mean-class accuracy across all three paired seeds, a mean gain
of at least .5 percentage points over each, and a positive lower endpoint of
a paired class bootstrap 95% interval (10,000 replicates, seed65001, averaging
paired seeds within class before resampling). Also compare with NoPE. Report
all comparisons and intervals without promoting a secondary metric to primary.
Failure against a strong control is preserved even if a weaker ablation loses.
The practical threshold is for this bounded development decision, not a claim
of general effectiveness or a universal prerequisite for unrelated tasks.

If Gaussian relations only beat center controls, shape participation may help
but their special structure is not supported over an ordinary MLP. If benefits
are limited to one source or density, retain that scope. If a stable development
lead appears, assess remaining historical transfer/robustness failures and
choose an unused confirmation plan before opening confirmation data. No outcome
of R65 alone completes the global objective or erases the RGB failures.

## Checks proportional to the claim

Before fits: synthetic quaternion/covariance reconstruction against an
independent implementation, exact native-to-model normalization checks, shared
input and initialization tests, finite forward/backward, token permutation,
and Gaussian relation rigid-motion invariance. Confirm all train/val/dev_eval
identities disjoint and raw bytes correct. No old static integration gates.
After fits: verify complete budgets and paired actual streams, replay selected
validation results, independently recompute classification metrics and compare
the relation operator with NumPy. Check source/config/checkpoint identities.
Inspect actual curves and comparison plots. Ordinary implementation fixes are
versioned failure records; they are not new scientific rounds.

## Primary sources and scope

- https://huggingface.co/datasets/zxAi/GSModel60 : public release and split.
- https://github.com/PKU-OV3-LAB/GAPromptPlus/blob/main/object_level/datasets/GSModel60Dataset.py : released attributes and loader.
- https://github.com/qimaqi/ShapeSplat-Gaussian_MAE/blob/main/datasets/modelnet_split/modelnet40_train.txt : original training identities.
- https://github.com/graphdeco-inria/gaussian-splatting/blob/main/utils/general_utils.py : wxyz rotation convention.
- https://arxiv.org/html/2609.19716v1 : GSModel60 reconstruction-derived task motivation.

These sources establish provenance and conventions, not efficacy of our PE.
No GAPrompt++ backbone or published accuracy is claimed as reproduced.
