# R52: a single camera x Gaussian local-frame PE

2026-11-02. Prospective old-development experiment. Full research goal ACTIVE
and unachieved. The preceding explanatory response made no new experimental
progress; this turn implements and tests the next discriminating comparison.
R50/R51 sources and results remain frozen. No renderer or static-integral gate.

## Hypothesis, alternatives and decisions

Primary: an explicit known camera mapping lets a compact Gaussian region frame
provide useful additional structure inside Transformer attention. Alternatives:
camera alone suffices; isotropic shape or generic affine flexibility suffices;
the existing common positional/camera features already suffice; or the fixed
rotary/value transport is an unhelpful constraint. Mathematical SPD limitations
are motivation, not evidence for a causal explanation of earlier task errors.

Eight roles, all retrained: NoPE, MLP, inherited smooth HRoPE adaptation,
local_gs (A=G), camera_only (A=P), camera_iso, camera_gs (PRIMARY), camera_affine.
local_gs isolates adding P without also changing the direction of G. R51 used
A=G^-1, so comparing R52 directly to R51 GS alone would confound that convention.
NoPE retains the common positions/cameras; it is not a geometry-blind baseline.
The HRoPE role remains our adaptation, not an official paper replication.

Compare primary against every strong control, especially NoPE, camera_only,
camera_iso and parameter-matched camera_affine. Stable, practically meaningful
benefit of full shape would justify a second task and then unused confirmation.
Camera-only success cannot be called a Gaussian advantage; ties with isotropic
support simplification. Negative results are retained. Three seeds/five reused
rooms cannot establish generality, and no universal numeric pass threshold is
invented. Do not promote a method with gains on one chosen metric but material
coarse/tail or difficult-slice regressions without stating the tradeoff.

## Coordinate derivation and operator

Reuse exactly R51 RGB features, positions, camera metadata and labels. T_i is
camera-to-pair-reference; its origin is divided by the fixed 3m unit. Intrinsics
in the centered image chart are K_i=diag(fx/192,fy/144,1). Lift K to diag(K,1).

    P_i = diag(K_i,1) T_i^-1
    G_i = [ L_i  mu_i ; 0  1 ],   L_i = R_i diag(exp(ell_i)) R_i^T
    A_i = P_i G_i,     A_i^-1 = G_i^-1 P_i^-1
    geometric score = q_i^T A_i A_j^-1 k_j
    value transport = A_i sum_j attention_ij A_j^-1 v_j

P maps (x/3m,y/3m,z/3m,1) to centered homogeneous image coordinates retaining
depth. Dividing the first two coordinates by the third gives centered pixels
normalized by W,H. This invertible 4D lift is NOT an invertible depth-free 2D
projection and supplies no token depth. G is a learned latent Gaussian frame
in the common reference chart, not a measured/recovered physical 3DGS.

The primary reuses R51's 96->12 field: rotation6D, ell=.5*tanh(raw scale),
mu=tanh(raw center). Isotropic uses 96->4; affine uses 96->12 with matrix_exp
of a bounded general 3x3 generator. Initial G=I for all. Known-camera-only has
no PE parameters. Canonical SPD roots retain Gaussian axis sign/permutation
invariance. Under a common rigid world-coordinate change W=[U,c;0,1],
G' = W G diag(U^T,1), P'=P W^-1: the common right factor cancels in A_i A_j^-1.
This is an operator identity; learned field equivariance is NOT implied.

Keep the inherited half-head split: 24 geometry dimensions (six 4D blocks) and
24 rotary dimensions; positive Q/K/V phase, negative output phase. Native 2D
phase and value transport are held fixed, not separately tuned in this round.
Full 48D ordinary RoPE/MLP and unchanged smooth HRoPE are inherited controls.
Same two 96D, two-head cross-attention blocks and matching head; common initial
weights and actual batches identical within seed; no extra loss/branch/head.

## Data, training and evaluation

Reuse hashed R51 arrays: 576 fit pairs,64 validation,480 old evaluation across
five rooms. Native positions,128D frozen ResNet RGB,K,c2w for everyone. Never
input token depth, normals, physical covariance, labels or masks to attention.
Camera poses originate RGBD reconstruction. All untouched scenes stay unopened.

Seeds11/22/33;1000updates,batch8,AdamW lr.001,wd.0001,cosine1->.2,clip1,
FP32/noTF32; validate CE initially/every100, select earliest minimum. All24
selections sealed before old-eval export. Same seed+50000 sample schedule as R51.
Use idle GPUs verified twice5s apart and CUDA smoke; never interfere with others.

Primary soft EPE; also CE,PCK4,PCK8,AUC0..8,argmax EPE,>16px tails,scene p95,
paired seeds/scenes, fixed15/60frame-gap slices. Report cost and parameter counts.
No per-query independence/significance claim or automatic win from tiny ties.
Same fixed RGB-token permutation diagnostic, seed510000+globalpairindex, no
retraining or label/camera/position change. No postselection temperature tuning.

Bounded implementation checks before freeze: camera physical projection/inverse
and world-frame identity; independent nonzero NumPy full operator/transport and
gradient; common initialization, control equivalence, label exclusion/token
permutation, Gaussian gauge. Real-input finite CPU/GPU gradients. After training:
independent first-pair/first-layer operator per selected checkpoint, CPU validation
replay, independent saved-metric replay. Same-author independent arithmetic is
not a second laboratory or independent full-network implementation. Numerical
differences are interpreted by probability winners, features and metric impact.
Ordinary pre-run fixes retain failure/version records, not new scientific rounds.
