# R51: real visual matching and a direct HRoPE/MLP comparison

2026-11-02. Prospective development experiment, following the user's resumed
research authorization and explicit request for HRoPE versus MLP. R50 stays
frozen. This tests Transformer PE, with no renderer or rendering objective.

Hypothesis: a compact GS covariance frame helps learn cross-view visual
correspondence from RGB features, positions and cameras. Competing explanations:
generic feature-dependent coordinates or affine transforms suffice; rotary
value transport accounts for apparent gains; complete anisotropic GS shape is
unnecessary; or this task/budget does not benefit from local-frame operators.
An explicitly requested secondary question is whether HRoPE beats the MLP on
these same metrics. Its success would motivate surface-aware structure, not
establish the GS candidate. Its failure would be local to this adaptation,
information regime and limited budget, not a refutation of the paper.

## Fixed information and task

Reuse R46's source identities: 576 fit pairs, 64 validation pairs, 480 previously
used evaluation pairs in five rooms. The latter are DEVELOPMENT, despite their
historical directory name `test`. No untouched confirmation/official test is
opened. All methods receive the same frozen 128D ResNet18 RGB features, native
image positions, calibrated intrinsics and relative camera extrinsics. The
camera poses come from RGBD reconstruction, so do not call this fully RGB-only.
Cached depth, normals, mean3/cov3 and projected correspondences are never model
inputs. Only features/K/c2w numerical fields are loaded from frame caches.
Original features are verified equal to the old pair inputs; labels stay exact.
Original q->k bilinear supervision is retained, with no invented reverse labels.

Two 96D/2-head bidirectional cross-attention blocks, pre-LN/residual/FFN, followed
by the same 48D normalized cosine matching head. No final geometric shortcut.
The shared input MLP is (128+18)->96->96. RGB features are multiplied by sqrt128;
camera metadata: c2w rotation9, origin/3 (3), log(fx/W),log(fy/H), principal
point(cx/W,cy/H), centered native image positions/96 (2). Query camera frame
is the pair's reference; no data-dependent scale normalization. Intrinsics are
expressed in a principal-point-centered normalized image chart, diag(fx/192,
fy/144,1). Rotary positions are (pixel-principalpoint)/8. The common metadata
retains the principal point. W192,H144;432 candidate tokens and no attention mask.

## Models and HRoPE fidelity

Ten retrained roles: RoPE, MLP coordinate warp, R50 HRoPE adaptation,
paper-form HRoPE, full GS frame (PRIMARY), isotropic GS, free affine frame,
identity geometric frame, conditional Gaussian bias, NoPE. Preserve R50 operator
sources; NoPE disables explicit rotation but receives the same positional and
camera information. Common weights and actual sampled batches match within seed.
The MLP coordinate warp is 96->16->2 (1586 PE parameters); HRoPE uses 96->13
(1261), full GS 96->12 (1164). This is information/budget matching, not exact
parameter equality. The affine comparison matches GS parameter count.

Primary paper https://arxiv.org/html/2610.28284v1, Sections3.2/A.3/A.4/A.5.
Public repository https://github.com/Inception3D/HRoPE was checked 2026-11-02:
README says Coming soon; no runnable official implementation is available.
Both HRoPE implementations use H=K(R+kappa*t*n^T), rotation6D, unit normal,
Sherman-Morrison inverse, half-head geometry, and H^T Q / H^-1 K,V / H output.
The inherited adaptation uses .5*tanh bounds and positive Q/K/V rotary phase.
The separately requested paper-form role uses hard clipping t,kappa to +/-.5,
denominator>=.1, matrix entries+/-5; Hcal=diag(H,Phi) gives negative Q/K/V
and positive output phase. Both initialize R=I,t=0,n=e3,kappa=.25.

These are local implementations of the paper's operator, NOT its official
LVSM/VGGT pipeline. Shared deviations: smaller two-block matching backbone,
frozen ResNet features+camera-conditioned learned parameter head reused across
layers, normalized pre-transform Q/K with learned temperature, fixed common
logspace(-1,.6) rotary frequency band, centered token-stride positions, normalized
intrinsics, matching CE and limited training. Paper-form vs inherited adaptation
changes clipping and rotary convention together, so cannot isolate either cause.
No direct original-paper benchmark claim is permitted.

## Budget, evaluation and decisions

Seeds11/22/33;1000updates,batch8,AdamW lr.001/weight_decay.0001,cosine1->.2,
gradient clip1, full FP32/noTF32. Validate q->k CE initially/every100updates;
select earliest minimum. Seed batch RNG seed+50000. Seal all30 selections before
evaluation. Use currently idle GPUs, checking occupancy twice5s apart and CUDA;
one worker/device, CPU4. Do not interrupt other processes.

Primary metric soft EPE (pixels, lower); report CE, argmax EPE, AUC0..8, PCK4,
PCK8, error>16px tails and per-scene p95. Scene-equal aggregation and paired
seed/scene differences, not per-query independence claims. Report15/60-frame-gap
strata as prespecified descriptive difficulty slices. Derive argmax grid ceilings
from the existing labels, never require an impossible fixed+1pp gain. Broad
usefulness needs stable gains versus MLP/affine/HRoPE without material coarse or
tail regressions. GS-vs-isotropic/identity checks constrain shape attribution.
Three seeds/five reused rooms are developmental, not final generality.

For every selected model, evaluate baseline and a single fixed RGB-token
permutation per view/pair (seed510000+globalpairindex). Preserve positions,
cameras and labels. If permuting RGB again has little effect, investigate the
information structure before promotion. This diagnoses dependence, not proof
that learned geometry is physically correct. No tuning of permutations or
temperatures to rescue results. No checkpoint changes after selection.

Bounded verification: camera/chart/world-frame conventions, independent
nonzero HRoPE matrix/transport test, shared initialization and input exclusion;
GPU finite gradient smoke; replay stored metrics; independently reconstruct
one full operator sample per selected model and replay validation CE on CPU.
Judge numerical differences by winner/output/metric impact. Ordinary pre-run
fixes retain a failure record and do not start a scientific round. A positive
result would justify further seed/second-task/unused-data tests; a negative
result must be reported and determine the next mechanism test.
