# R54: matched calibrated-plane comparison under the same PRoPE camera operator

2026-11-02. Development experiment authorized when the user resumed the goal
after the PRoPE branch review. This fills a missing control; a plane success
does not establish the Gaussian PE objective. R53 is closed separately under
its existing lock, without changing this comparison based on its outcomes.

## Question, alternatives and decisions

Fixed new primary contrast: camera_plane minus camera_only on continuous
localization error. Gaussian candidate remains the unchanged R52 camera_gs;
compare it explicitly with camera_plane and all simpler/general controls.
Hypothesis: once the known camera projection P is kept in every geometry role,
local plane orientation/offset can improve matching beyond P alone.
Competing explanations: any benefit is offset-only; generic learned coordinate
flexibility explains it; or this feature/data/budget setting cannot use learned
local geometry. Neither the old predicted-HRoPE ranking nor common-frame
cancellation establishes which explanation is correct.

Roles: nope,mlp,hrope,camera_only,camera_offset,camera_plane,camera_iso,camera_gs,
camera_affine. Only camera_offset/camera_plane are new:6 fits,3 paired seeds.
Reuse21 R52 references with origin locks/checkpoints/config/batches preserved
and exact old/new selected-function checks. They add no independent trials.
All27 selected models are sealed before exporting54 baseline/RGB-shuffled groups.

If calibrated planes improve over P but full Gaussian does not, reconsider the
Gaussian representation's geometric role. If Gaussian gains survive the plane,
isotropic and affine comparisons, proceed to native3D/cross-task development
and untouched confirmation. If neither helps, one prospectively fixed matched
budget/feature-source diagnosis is permitted; no indefinite rescue sweep.
HRoPE winning is not a mandatory gate. Tiny ties favor simplicity and are not
numerical failures. Full goal is not achieved by this one task or development data.

## Shared operator and declared adaptations

PRoPE source: https://arxiv.org/html/2507.10496v2 and official
https://github.com/liruilong940607/prope/blob/main/prope/torch.py,
Git blob1c450ccc05bca91a6b9ddbd19971437896ef83b5. The exact MIT-licensed source is
snapshotted as third_party/round54_prope_torch.py for one bounded transport test.
The official camera/Q/K/V/output arithmetic is compared after explicit
coordinate/phase/channel adapters; this is not the official trained pipeline.

Keep R52's principal-point-centered chart K=diag(fx/192,fy/144,1), common3m
length unit, half48D-head4D camera blocks and half native2D rotation. R52 uses
interleaved positive query/key/value phase and inverse output phase, frequencies
logspace(-1,.6,6) and (pixel-principalpoint)/8 positions. Official code's default
image-midpoint chart, patch-grid phases/frequencies and split channel layout
are declared differences, shared across this comparison rather than tuned.
Same two96D/2head bidirectional blocks, pre-transport Q/K normalization, content
input geometry and matching head. NoPE retains the same positional/camera inputs.

For each layer A_i=P_i F_i. Query transforms by A_i^T, key/value by A_i^-1,
and output by A_i. Geometry score is q_i^T P_i F_i F_j^-1 P_j^-1 k_j.
All methods share native2D rotary half. Known P never has to be predicted.
Calibrated-plane form follows HRoPE Eq.7, sections3.3/4.2.2:
https://arxiv.org/html/2610.28284v1

    F=[R_n,z*n;0,1], R_n*e3=n, |n|=1.

Use minimal rotation R_n and a two-coordinate stereographic normal chart:
n=(2a,2b,1-a*a-b*b)/(1+a*a+b*b), quaternion (1,-b,a,0)/sqrt(1+a*a+b*b).
This covers all finite-chart normals except the antipode at infinity; it has
no finite-coordinate singularity. A zero-initialized96->3 linear head predicts
a,b,raw_z, with z=tanh(raw_z). Bound z to the same +/-3m range as R52 means.
There are291 PE parameters. Offset-only has n=e3,R_n=I,96->1,97 parameters.
Both have exactly P's initial function; orientation gradients are nonzero at
initialization for nonconstant token features. No extra geometric supervision.
Common F cancels exactly, reducing to PRoPE, as intended in the cited paper.
This is a learned plane-frame prior, not evidence of recovered physical planes,
exact2D patch warping or full neural/world-reference equivariance.

Unchanged R52 Gaussian/iso/affine models have1164/388/1164 PE parameters; means,
scales and rotation parameterization stay frozen. Capacities differ and are
reported; an unrestricted affine control distinguishes Gaussian specificity.
Predicted3x3 HRoPE remains an explicitly different reference, not a matched
known-camera ablation. Changing that implementation is outside this round.

## Data, training, measures and bounded checks

Exact hashed R52 arrays:576fit/64val/480old_eval pairs from5 already-used rooms.
Same frozen128D ResNet features,432tokens, native positions and RGBD-derived
cameras; no token depth/normals/actualGS covariances or labels enter model input.
No unused scene or ModelNet40 official test is accessed. Seeds11/22/33,
1000steps,batch8,AdamW lr.001/wd.0001,cosine1->.2,clip1,FP32/noTF32, actual
batch RNG seed+50000. Earliest minimum validationCE every100steps selects
checkpoints. Fixed budget is not proof of convergence. No additional tuning.

Primary softEPE, plus CE,PCK4/8,AUC0..8,argmaxEPE,>16px tails,p95,paired seeds,
scene means, fixed15/60-frame gaps, parameter/time/memory cost. RGB permutation
uses the same fixed seed510000+pairindex as R52. Frame gap is a difficulty proxy;
all focal ratios1 means no zoom claim. No error-selected subsets or per-query
independence/significance claims. New geometric strata need a separate prospective
definition; they are not silently added after seeing these outcomes.

Three bounded tests: independent nonzero plane/matrix/transport with common-F
reduction; shared init/input exclusion/permutation/finite-difference gradients;
official PRoPE transport under the declared adapters. Check finite real-input
gradients before freeze and CUDA smoke before training. Queue uses devices
observed idle twice5s apart, one worker/GPU, CPU4, preserving others' jobs.
Independent postcheck covers all27 selected first-layer samples, full CPU valCE
and all54 stored metric groups. NumPy/SciPy path is independent arithmetic by
the same author, not an independent lab/full-network rewrite. Warnings5e-5(P)
and5e-6(CE) trigger impact inspection, not automatic scientific rejection.
Ordinary failures retain versioned sources/logs; no new round for a pre-run fix.
