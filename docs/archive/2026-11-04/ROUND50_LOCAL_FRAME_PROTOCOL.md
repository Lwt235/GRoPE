# R50: controlled local-frame mechanism experiment

2026-11-01. The user resumed the goal after discussing HRoPE. R49's joint
RoPE/Gaussian-distance drafts remain unexecuted and are not overwritten.
This is a bounded DEVELOPMENT experiment, not confirmation or a paper replication.

## Question and competing explanations

Can a compact learned Gaussian local frame (center, orientation, scale) help
the same Transformer match neighborhoods under projective deformation, compared
with ordinary RoPE, an information-matched MLP and a general affine transform?
Alternatives: any learned transform suffices; center/ordinary positional cues
already solve the task; gains come from value transport/channel allocation;
the supplied geometry is too noisy; this controlled task is insufficiently
representative. No synthetic success establishes real-world/general utility.

## Data and task

Procedural textured scenes have four finite slanted planar panels and an infinite
background plane. Textures are continuous random Fourier fields, shared in
world coordinates, with NO per-primitive identity embeddings or Gaussian label
kernel. Cameras and ray/plane intersection determine RGB, observed surface
geometry, visibility and exact cross-view correspondences. Image generation is
only a data source: no image reconstruction/rendering objective is optimized.

Images are 64x64, token patches 8x8, 64 tokens/view. Inputs to EVERY model are
the same flattened RGB patches and 28 geometric numbers: observed center XYZ,
normal, six symmetric covariance components from the local patch differential,
camera rotation/translation, focal lengths and native pixel position. No surface
ID, visibility, correspondence, label or target condition enters the model.
All geometry is in the first camera frame. Common preprocessing is fixed in
source, without evaluation statistics. The predictor may learn latent frames;
their parameters are NOT claimed to recover unique physical geometry.

Two training regimes use identical RGB/labels: exact geometry and geometrically
corrupted inputs. Corruption is a synthetic estimator-error proxy, NOT an actual
learned geometry estimator. Every role receives exactly the same corrupted
arrays. Centers, normals and covariance are perturbed using an independent RNG;
cameras remain calibrated. This separates structure from extra information.

Fixed generated scenes: 512 fit, 64 validation, 128 development scenes for each
of base, slant, zoom, wide-baseline. Development conditions share scene seeds
to allow paired comparisons; they are not 512 independent scenes. Training
slant<=25deg, camera baseline .2-.6, focal ratio .85-1.15. Stress conditions
increase slant to45-65deg, focal ratio to1.6-2, or baseline to1-1.4. All samples
are retained; visibility/center-grid eligibility is reported, not tuned.
Labels are bilinear interpolation weights at the true projected location.
Both directions supervise training. Report query-to-key matching.

## Models

Common 96D, two-head, two-block bidirectional Transformer; identical trainable
RGB/geometry input projection, residuals, FFNs and cosine matching head.
Only the positional operator/predictor changes. Final scores contain no extra
geometry shortcut. Three paired initialization/sampling seeds:11,22,33.

- rope: conventional Q/K 2D RoPE on all48 head channels; V unchanged.
- mlp: shared96->16->2 MLP predicts bounded coordinate offsets before RoPE.
- hrope: learned13-parameter R+t*kappa*n^T, multiplied by normalized intrinsics;
  24 geometric channels plus24 native2D RoPE channels; Q/K/V/output transforms
  follow HRoPE Appendix A.3/A.4. No VGGT or LVSM; call this operator adaptation.
- gs_frame (fixed primary): G=[R diag(exp(s)) R^T,mu;0,1], A=G^-1. The canonical
  SPD square root depends on covariance, not the sign/permutation of its axes.
  Apply A^T to Q,
  A^-1 to K/V and A to output, in4D blocks across24 channels; other24 retain
  native2D RoPE with corresponding V/output transport. One12-scalar predictor
  (6D rotation,3 log-scales,3 center) shared across layers/directions.
- gs_iso: same construction with one log-scale and3 center scalars;4 outputs.
- affine: general orientation-preserving affine frame using exp(B),translation;
  12 predicted scalars. Same transport/channel layout as gs_frame; both start
  at identity but parameterization/derivatives/ranges differ. No isolated causal
  claim about SPD or Gaussianity from this comparison alone.
- frame_identity: fixed identity geometric half; same remaining rotary blocks
  and V/output transport. Checks layout/transport versus conventional RoPE.
- gaussian_bias: conditional Gaussian quadratic distance prior in native2D
  image coordinates with unrotated Q/K, as an additive-prior comparator.

All share the same input information and training schedule; predictor parameter
counts differ and will be reported. Affine and full GS have equal output count.
Rotations are irrelevant at the initial spherical GS shape; they acquire
gradients when scales differ. Check non-spherical gradients and trained fields.
Both GS forms start at identity. This is not an SO(3)-equivariant network claim.
No method gets special pretrained features, loss or extra optimization steps.
Native2D RoPE alone does not explicitly know cross-view correspondence; the
shared geometry-containing content stem can learn it in ALL roles.

## Fixed budget and interpretation

8 roles x3 seeds x2 geometry regimes =48 fits. AdamW lr1e-3,wd1e-4, batch16,
1000 updates, cosine1->.2, gradientclip1. Validation CE at0/every100; choose
earliest minimum. Shared parameters and actual batch stream are paired.
All checkpoint choices are sealed before development evaluation. Every selected
model is evaluated on all4 camera conditions with exact AND corrupted geometry.
No angle/temperature/width sweep after inspecting results.

Primary metric: PCK curve AUC over thresholds0..8 pixels (equivalently mean
clip(1-error/8,0,1)), plus EPE/PCK4/PCK8/CE and tail error. Report per-scene
paired changes and seed spread. Also report deformation strata of the true
local warp's singular-value ratio (<1.3,1.3-1.8,>=1.8), without conditioning
on model errors. Stratum thresholds are fixed here, not fitted to winners.
The exact/exact regime is the primary mechanism comparison; corrupted/corrupted
is a parallel feasibility comparison. Cross-regime tests measure degradation.
All three are reported; success in one is never silently substituted for another.
RoPE axis frequencies are fixed logspace(-1,.6), with12 pairs per axis for the
full48 channels and6 per axis in the24-channel blocks. This is an adaptation,
not HRoPE's original LVSM frequency/backbone/training recipe.

A useful preliminary signal requires GS beating BOTH MLP and affine on base
and the equal-condition mean (3/3 paired seeds, >=1pp AUC), without a mean
PCK8 regression >1pp. These are development interpretation guides, not a
mathematical theorem or general-purpose prerequisite. Preserve continuous
differences even if guides are not met. Component-specific gains must exceed
gs_iso/frame_identity to support full shape. Report HRoPE separately; do not
rename it as GS success. Good oracle results but bad corrupted results motivate
estimator investigation; HRoPE gains without GS gains motivate redesigning GS;
neither gaining leaves mechanism/task suitability unresolved, not disproven.

## Correctness and execution

Before training: bounded tests for true homography correspondences, local
differentials, independent block-matrix attention/transport, inverse/gradient
and no label access. One CUDA smoke per role and timed small steps. Freeze
source/config/data hashes, run on host GPUs idle in two observations, CPU4 per
worker. Preserve prior results/other users' jobs. After training: parameter and
batch identity checks, recompute metrics from saved compact errors/log-losses,
and a sampled independent operator path. Numerical discrepancies are judged by
their effect; no costly static quadrature prerequisite.

Source: https://arxiv.org/html/2610.28284v1 sections3.2,4.2 andA.3-A.4.
HRoPE's structured relative transformation is inspiration, not evidence that
our Gaussian frame works. Existing R40-R48 negatives and unused real-task
confirmation sets remain intact. Any successful next step needs real input
geometry and real task validation before cross-task/fresh confirmation.
