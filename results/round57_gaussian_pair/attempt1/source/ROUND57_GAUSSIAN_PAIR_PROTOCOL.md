# R57 — two Gaussian parameter sets define one positional relation

Prospective development experiment, 2026-11-02. The complete goal remains a
compact generally useful GS-form Transformer PE, with strong-control utility,
appropriate geometry/robustness/cross-task evidence and untouched confirmation.
R56 is complete and not promoted. No result of R57 exists at protocol creation.

## Hypothesis and why the change is structural

R56's query-conditioned Gaussian is used by the network but does not outperform
MLP/free quadratic reliably. Its key is a point coordinate, not a Gaussian.
R57 makes the positional relation depend on BOTH tokens' centers and shapes.
Keep center RoPE on content and replace the positional score with one Gaussian
Jeffreys similarity. No new branch, loss, feature input or trainable mixture.

Each token predicts m_i and P_i=Sigma_i^-1 from its current normalized feature,
using exactly the R47 bounded mean/Cholesky parameterization (initial Sigma=16I).
Its latent center is c_i=x_i+m_i. Define delta=c_i-c_j and

    b_full(i,j) = -1/4 [delta^T(P_i+P_j)delta
                        + tr(P_i Sigma_j + P_j Sigma_i) - 6].

This is -1/2 [KL(G_i||G_j)+KL(G_j||G_i)]. The score is symmetric, nonpositive
and zero for identical Gaussians. A common invertible affine change of the
Gaussian parameters leaves this geometric score unchanged. This property is
also shared by its adaptive-distance component; it is NOT a claim that the
learned predictor, center RoPE or the whole network is affine/SO3 equivariant.
Sigma is a learned comparison parameter, not measured GS covariance or verified
localization uncertainty. Similarity does not determine internal texture warp.

With svec preserving the Frobenius inner product, theta=[P c,-svec(P)/2] and
eta=[c,svec(Sigma+c c^T)], Jeffreys=(theta_i-theta_j)^T(eta_i-eta_j).
Q=[theta/sqrt(2),eta/sqrt(2),1] and K=[eta/sqrt(2),theta/sqrt(2),-theta^Teta/2]
give b_full up to the query constant theta_i^Teta_i/2. Thus19 positional Q/K
channels suffice in3D. The implementation subtracts the irrelevant query
constant before adding the FP64 geometric product to FP32 content scores;
there is no fused-kernel speed claim. Covariance is evaluated from its Cholesky
factor without eigenvectors. This classical divergence/lift is not new theory.

Competing explanations: moving both centers is sufficient; adaptive distance
without shape compatibility is sufficient; generic learned position functions
are sufficient; shape compatibility over-restricts useful token interactions;
apparent gains are development/initialization variation.

## Fixed comparison and information control

- New roles: pair_full (primary); pair_distance (omit only the trace term);
  pair_center (learn both centers, hold P=I/16 fixed).
- Full and distance share the identical1746-parameter predictor, parameter
  ranges and initialization. They differ by exactly one covariance-compatibility
  term; initial functions and first derivatives agree. Center uses582 PE params.
  For distance, eta's second moment excludes Sigma, using the same19-channel
  lift. Center uses that distance form at fixed P. All start from the same
  fixed-distance positional softmax, including the R56 joint-fixed reference
  mathematically; do not demand bitwise equality across distinct arithmetic.
- Three paired seeds11/22/33:9new fits. Reuse the39 already selected R56/R48
  reference models and their R56 exports, with exact model/source/data/config/
  prediction identities. These are not39new independent repeats. Do not mix
  three/five-seed means. Reference factories and functions remain unchanged.
- Same global4-layer96D2-head Transformer and XYZ inputs to all roles; same
  R48 CONFIG,120epochs/29760updates, batch32,512points,Y-rotation,jitter,
  AdamW,warmup/cosine,label smoothing,clip and minimum-validation-CE selection
  every5epochs including0 (earliest exact tie). Same initialization, actual
  sampling/augmentation hashes, per-update dropout seeds and resumable optimizer.
- Same official-train-only cached ModelNet split7910fit/965val/965old_dev_eval.
  Do not reopen the archive, official test, or home_ac/hotel_barcelona/hotel_graz/
  mit_1_242/harvard_computer_lab. No new data acquisition or confirmation.

## Checks and evaluation

Before training, bounded independent NumPy direct-KL vs19-channel tests cover
nonzero shapes, self/symmetry/common-affine properties at fixed parameters,
equal-covariance reduction, full/distance initialization and gradient identity,
finite differences and optimizer continuation. Real fit/val preflight verifies
reference identities and actual new-model gradients. Freeze protocol, transitive
sources, cache manifest and reference identities before CUDA smoke and fits.
All48 selected models are sealed before new old_dev_eval export. Sampled checks
store query AND key features so the independent calculation covers both shapes.
Numerical warnings are interpreted by output/conclusion impact, not universal
integrator gates. Identity/substantive implementation errors require correction.

Primary utility:base OA versus pair_distance,pair_center,joint_conditional,
joint_free_quadratic,joint_fixed,MLP+RoPE and centerRoPE. Report all16 roles,
mCA/CE,paired seed differences and all seven old R48 conditions:base,y90,tilt,
jitter02,256/1024points,point reversal. Preserve the severe tilted-rotation
limitation; no augmentation/budget change to chase it in this comparison.

One prespecified diagnostic:remove the covariance-compatibility term from each
selected pair_full checkpoint at inference on base old_dev_eval,all3seeds,
all4layers/heads. No retraining/tuning. Earlier interventions can alter later
features/centers. Sensitivity alone does not prove useful physical shape; the
separately trained same-capacity distance control is the stronger comparison.

Use only GPUs idle in two live observations5seconds apart, at most6 workers,
CPU4 each, actual CUDA smoke before fitting; preserve other users' jobs.
Save model/optimizer/RNG/stream/best state every5epochs;8hour operational worker
limit,16GiB allocation ceiling. Avoid querying NVML when no GPU job is pending;
an observation timeout is not proof of process termination. Ordinary pre-run
fixes retain versioned failure evidence rather than creating a new round.

## Decisions

Small inconsistent differences are ties; no arbitrary1pp rule. A useful full
candidate must improve strong controls, especially equal-capacity pair_distance,
with compatible mCA/CE and no hidden substantive robustness loss. Only then
consider a prospectively frozen simplification/cross-task study and unused
confirmation. If distance/center/generic controls match it, retain that outcome
without calling it Gaussian-shape success. Do not rescue a failed full variant
by repeated widths/temperatures/learning-rates/budgets or additional modules.
Nine development fits and algebraic correctness cannot complete the full goal.

Related foundations: [Gaussian embeddings](https://arxiv.org/abs/1412.6623),
[exponential families and Bregman divergences](https://jmlr.org/papers/v6/banerjee05b.html),
and docs/compact_gaussian_dual_pe_reserve_20261101.md. Its exact counterexamples
to probability-mass additivity remain true; R57 does not reinterpret similarity
as mass or claim to repair those failures. R56 report remains negative.
