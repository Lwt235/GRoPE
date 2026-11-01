# R39: fixed dual-side 3D Gaussian overlap development screen

Declared 2026-11-01 before R39 input decode, numerical tests, forward or label
access. Selected after the accepted R38 negative result and saved-output
diagnostic. This is a new compact candidate, not a repair or re-run of R38.
The original gates indexed by `docs/research_gate_map_20261101.md` all remain.

## Hypothesis and exact definition

Use both views' RGB-predicted depth to place query/key supports in one 3D
camera frame. Unlike R38, there is no perspective density integration. Unlike
old 2D R6/R9, key depth and key shape vary. This extra information is available
to all new controls and must not be attributed to covariance alone.

For each source image, clean predicted depth exactly as R10/R13: NaN=10,
+Inf=20, -Inf=.1, clip [.1,20] metres. Intrinsics are the supplied standard
pinhole matrices. Every pixel has x=z K^-1[u,v,1] and within-pixel covariance
z² B Bᵀ/12, B=K^-1[:, :2]. Every native patch has one Gaussian, centered at
the patch-center ray with bilinear cleaned depth. Its covariance is
Q=mean(within-pixel covariance)+mean[(x-a)(x-a)ᵀ]. Query mean/covariance are
rigidly transformed to the key camera. No front/FOV/depth/rank exclusion.
Covariance Q is a second moment around the specified center, not the exact
covariance of the original pixel mixture. No exact partition-additivity claim.

One shared, positive base metric is Γ=γI, γ=4 ell². ell² is the median over
both complete images' z² tr(B Bᵀ)/2. The multiplier 4=(.25*8)² inherits the
old 2-pixel reference blur scale; this is a new 3D isotropic modeling choice,
not an exact conversion of screen blur. No bandwidth search. γ is in m².
The prepared kernel is rigid-frame invariant and similarity equivariant when
geometry and γ transform consistently. Fixed metre-valued cleanup thresholds
do not independently imply preprocessing invariance under arbitrary unit changes.

Token total mass m=number of source image pixels/64, so q24 has m=9, k8 m=1.
This is image-area mass, not peak opacity or calibrated visible volume. Set
S=Γ+Qq+Qk, δ=aq-ak, and use the known Gaussian RBF kernel-mean overlap:

    log K = log(mq mk) + (3 log γ - log det S)/2 - δᵀ S^-1 δ/2.

One additive floor K_floor=K+mq mk*1e-30 is applied in log space. Add the
unchanged R20 whole-matrix float32 content/tau product, then one row softmax,
first-index native argmax. Content is not rescaled by the new 3D covariance.
Query mass cancels in the final probability. No additional mode/center readout.

Three fixed roles: `full` uses both full Q, `point` sets both Q=0,
`isotropic` replaces each Q with trace(Q)I/3. They share anchors, γ, masses,
content, candidate keys and all input information. Only `full` is selected as
primary. Point/isotropic are descriptive ablations, not a tuned strong-control
envelope or an alternative selected after results. Same-3D strong quadrature,
fixed support and other original controls remain required for positive claims.

This closed-form kernel is known; see the cited derivation in
`docs/gaussian_overlap_measure_pe_reserve_20261101.md`. No novel kernel, finite
exact Q/K-channel factorization, opacity/occlusion equivalence, actual-GS
success, or general effectiveness is asserted. Actual-GS use would require a
separately frozen key-geometry source and same-information controls; reverting
to common 2D pixel keys would reproduce the old R9 special case.
The repository also already contains 2D overlap and a different 3D
Bhattacharyya kernel in `models/gaussian_measure_pe.py` and
`models/gaussian_world_pe.py`; R39 is not the first dual-side Gaussian PE here.
R24--R38 integral-to-rectangle additivity/reference tests certify those stated
integral laws, not this different single-Gaussian-per-key operator. R39 does
not inherit their passes or claim to satisfy their cell-integral identities.
All original empirical partition, center, robustness and task gates remain;
exact refinement invariance of refitted key Gaussians is specifically absent.

## Population, verification and ordering

Reuse exactly the accepted R38/R20 48 TRAIN pairs in saved order, 16 per old
scene; q24=48 and k8=432, total 2304 primary queries. Read only old predicted
input archives and R20 support cache fields before root numerical acceptance.
No VAL, fresh data, actual V or GS reads. Input fields are the two predicted
depth rasters, q_K, k_K, q_to_k_transform, and archived q_points3/q_pixel_cov3
for reconstruction checks. Cache fields are exactly R38's six content fields.
Labels and their expected hashes may appear in existing JSON metadata, but
label bytes are not hashed or opened until all predictions are sealed and the
root accepts complete independent numerical verification.

Before any execution, snapshot joint source closure and protocol; retain every
failed attempt with exit status. Synthetic tests cover independent preparation,
Dirac/PSD inputs, explicit Gaussian expectation quadrature, rigid/similarity
transform, extreme log-space/floor behavior, and all three roles. Production
uses centered scatter and Cholesky; a separately authored auditor reconstructs
inputs and uses eigen decomposition. Each full 48x432 matrix of all three
roles is independently verified. Original cached q pixel geometry must agree
with depth reconstruction to normalized error 1e-10. Prepared geometry error
<=1e-10, score/log-P normalized error<=1e-9, P absolute error<=1e-10,
row normalization<=2e-12, first-index argmax exact, finite outputs and PSD
supports (negative eigenvalue tolerance 1e-12 relative to max(1,norm)) required.
Both implementations symmetrize S as (S+Sᵀ)/2 only after checking support
symmetry to relative 1e-12, to remove floating-point antisymmetry. No hidden
jitter, clipping eigenvalues, retries or numerical salvage.

Four CPU threads, deterministic Torch, TF32 off, NumPy imported before Torch,
accepted Python runtime. Fixed 1200s each for production, independent numeric
audit, and each metric route; at most 512 MiB retained primary/ablation arrays.
Fail before labels if source/input integrity, numerical, population or resource
checks fail. Outputs/reads/failures are retained; logging goes outside sealed
stage directories. Freeze is metadata/byte hashing only; synthetic preflight
precedes actual archive decoding. Runtime here is descriptive, not the original
full-operator cost benchmark.

## Original eight necessary checks and decision

After root prelabel acceptance only, open original gt_xy/gt_visible and
g24_point_xy/g24_point_visible. Use both existing independently authored metric
routes, original g8/g16/g24 partitions, original per-scene mass weighting and
equal-scene macro. Center population remains 382/402/312=1096, PCK@8 with
native first winner, exact fractions. Continuous metrics agree rel1e-10/abs1e-9,
identities/counts/gate decisions agree exactly.

Primary KL<=1.02*strong_quad and <=1.02*legacy_quad16 (same configuration alias);
PCK>=reference-.005 for all six original localizers, unchanged. Report all three
roles, per-scene KL/PCK/RMSE, but do not promote an ablation instead of `full`.
Any primary failure stops this fixed design. Eight passes only permit broader
development on these exact parameters; all other TRAIN/VAL/stress/cost/actual
V/GS/fresh confirmation gates remain. Scientific_gate_passed=False and
goal_complete=False for every R39 screen outcome. No result-driven parameter
rescue, omitted query, alternate denominator, or subset success claim.
