# R53: ray-anchored inverse Gaussian frames as one positional operator

2026-11-02. Prospective old-development experiment. The preceding goal turn
made progress: R52 completed with a strong camera-only result and no unique
Gaussian benefit. Preserve that result and its immutable sources. The full
general/native-3D/cross-task/confirmation objective remains ACTIVE and unachieved.

## Hypothesis and competing explanations

A Gaussian can define a local coordinate frame: its center is on the observed
token ray, while its shape defines a local metric. Relative inverse frames may
make these degrees of freedom useful to a Transformer, beyond known cameras or
ray centers alone. This differs from R52's A=P G, whose common shape cancels.
The algebra motivates the test; it has not established the cause of R52 errors.

Alternatives: known camera projection already suffices; only ray centers help;
isotropic shape suffices; unconstrained affine transport explains the benefit;
or removing native 2D rotary channels is the relevant change. There is no
auxiliary depth predictor, renderer, extra loss or matching head. A shape head
receiving gradients alone is not mechanism evidence or task utility.

PRIMARY ray_gaussian uses a single 4D relative frame on all48 channels/head.
Strong references: unchanged R52 NoPE, MLP, smooth HRoPE adaptation, and its
best camera projection operator (renamed projection_only for clarity).
New controls: pose_only, ray_point, ray_iso, ray_affine, ray_gaussian_hybrid.
The hybrid applies the same Gaussian frame to24 channels and native2D rotation
to24. It is fixed as a control, not a result-selected replacement primary.

Primary must provide stable and meaningful utility relative to the strong
projection control, and shape attribution must survive ray-point/isotropic/
affine comparisons. Inspect paired seeds, scene consistency, fine/coarse errors,
tails, difficulty and cost together. Do not insist every tiny component gain
exceeds an arbitrary threshold. Ties support simplification; improvements solely
from the ray center or camera cannot be labeled Gaussian-shape advantages.
Negative results are retained, not rescued by scanning temperatures/widths/depth.
If supported, progress to native3D/another actual task and unused confirmation;
this one camera-conditioned test alone never completes the general PE goal.

## Coordinates and exact fixed parameterization

C_i is camera-to-query-reference in fixed3m length units. The query is identity;
key is inv(qc2w)*kc2w. K=diag(fx/192,fy/144,1) is the principal-point-centered
image chart, and positions are (pixel-principalpoint)/8. A ray is
K^-1 [(8/192)*position_x,(8/144)*position_y,1]^T.

    d_i = .1 + softplus(raw_depth + inverse_softplus(.9))
    mu_i = d_i * ray_i
    G_i = [L_i, mu_i; 0,1],   Sigma_i = L_i L_i^T
    F_i = C_i G_i,           A_i = F_i^-1
    score_ij = T * q_i^T A_i A_j^-1 k_j
    transported_value_i = A_i sum_j attention_ij A_j^-1 v_j

Queries/keys are normalized before transport, as in all prior matched controls.
A repeats over twelve4D blocks/head. V/output use the corresponding inverse/
forward transport. There are no rotary frequencies in the primary. Token
positions/cameras remain in the shared content input for every role, including
NoPE. Same two96D/2-head bidirectional blocks and48D normalized matching head.

One shared96->7 field predicts depth plus six Cholesky parameters: diagonal
exp(.5*tanh(raw[1:4])), strict lower .5*tanh(raw[4:7]) in row-major lower-triangle
order. Initially d=1(3m),L=I. Depth is positive but not upper-clipped. The shape
is a learned metric, not a measured physical rendering covariance or a calibrated
uncertainty posterior. Cholesky and GS rotation/axis scales describe SPD shapes,
but their parameterizations/optimization differ; do not claim this isolates that
change from ray anchoring, camera order or channel allocation.

ray_point has L=I and one learned depth (97parameters); ray_iso has exp(.5tanh)
scalar L (194parameters); full/hybrid have679parameters; ray_affine predicts
depth and a general3x3 generator expm(.5*tanh(raw)) (970parameters). These are
information/budget controls, not equal parameter counts. Full/iso/point/affine
have the same initial frames and full-head function; hybrid's function differs
because it explicitly retains rotary channels. pose_only uses G=I and A=C^-1,
zero PE parameters. It is distinct from projection_only's A=diag(K,1) C^-1 and
half-head native rotation. Keep both because R52 proved projection_only strong.

For a shared camera and L, relative translation is L^-1(mu_j-mu_i). Changing
the world reference C_i'=W C_i leaves F_i^-1 F_j unchanged. These are operator
identities, not proof of physical recovery, neural equivariance or perspective
surface rectification. This is a local-coordinate PE, not a new Gaussian kernel.

## Exact information and matched reuse

Reuse the hashed R52 arrays:576fit/64val/480old_eval pairs,432tokens,128D frozen
ResNet features, native positions and cameras. No model input includes token
depth, normals, actualGS covariance or labels. Camera poses are RGBD-derived,
so the setting is not fullyRGB-only. No unused scene or official test is opened.

Seeds11/22/33;1000updates,batch8,AdamW lr.001/wd.0001,cosine1->.2,clip1,
FP32/noTF32; earliest minimum validationCE at0,100,...1000. Shared initial
weights and actual batch RNG seed+50000 match R52. Same q->k bilinear labels,
no invented reverse supervision. A fixed budget does not demonstrate convergence.

18NEW fits: six new roles x three seeds. REUSE12 sealed R52 references:
nope->nope,mlp->mlp,hrope->hrope,projection_only->camera_only. Verify original
locks/config/data/checkpoints/run records, shared initialization and actual
batch hashes; compare old/new selected functions on the first fit pair. Preserve
original run records and origin lock hashes; copied records explicitly say reused.
R52 already demonstrated unchanged controls reproduce R51 tensor-for-tensor.
Reuse saves repeated training and adds no independent confirmations or seeds.

Seal all30 new/reused selections before exporting R53 old_eval. Export all30
again through the declared model factory; any reference replay differences are
reported. No checkpoint changes after selection. Same fixed RGB permutation
seed510000+globalpairindex per view/pair, preserving cameras, positions and labels.

## Measures, implementation checks and execution

Primary continuous softEPE; report CE,PCK4/PCK8,AUC0..8,argmaxEPE,>16px tails,
scene p95, paired seed/scene effects, fixed15/60-frame-gap slices and costs.
Three seeds/five reused rooms are development evidence, not independent query
trials or fresh confirmation. No zoom claim from constant-focal data.

Three bounded CPU tests already cover independent ray/matrix/full-head transport,
world-reference identity, nonzero finite differences, all six shape gradients at
identity, shared initialization, reference equality, labels and token permutation.
Before freezing, validate real-input finite gradients and reuse identities.
GPU smoke precedes training. Only currently idle GPUs observed twice5s apart;
one worker/device,CPU4, no modification of libc/drivers or other users' jobs.

Postcheck separately reconstructs one first-layer full-operator sample per
selected model with NumPy/SciPy, replays complete validationCE onCPU, and replays
all60 saved metric groups. Numeric warning levels5e-5(probability)/5e-6(CE) locate
issues; judge winner/feature/metric impact rather than rejecting tiny deviations.
This same-author arithmetic is not another laboratory or a full-network rewrite.
Ordinary pre-run fixes retain versioned failure records; they do not create new
scientific rounds or retroactively change outcomes. No static-integral gates.
