# R45: Gaussian coordinate maps inside Transformer RoPE

2026-11-01. This prospective specification is frozen with its source/data
manifest before toy tests or new training. R43/R44 failures remain unchanged.

Hypothesis: a small spatial Gaussian field can adapt rotary phases locally,
improving a full Transformer over point RoPE and ordinary learned position maps.
Alternative explanations: any nonlinear map helps, global frequency adjustment
is sufficient, or Gaussian geometry is redundant. All new models initially
produce exactly the original point-RoPE function, so changing initial frequency
coverage cannot explain a difference among them. No pretrained weights used.

The object is the PE inside the Transformer. Correspondence is a measurement,
not a definition of general PE success. This screen remains a 2D instance;
native 3D, cross-task, robustness, cost and untouched confirmation are outstanding.
Physical GS R/S usefulness is a separate claim: these are learned field parameters.
No renderer, new content/value branch, decoder change or static quadrature gate.

## Positional operator and fixed roles

Use the unchanged R5 two-block bidirectional Transformer, including Q/K/V,
residuals, FFNs, matching head and shared original 24 two-dimensional frequencies.
The backbone has 166611 parameters. For native position x, let z=(x-c)/e,
c=(11.9375,8.9375), e=(12,9). Define

    phi_g(z) = exp(-.5 (z-m_g)^T P_g (z-m_g))
    d(x) = e * ((phi(z)/r0) @ V),    u(x) = x + d(x)
    PE_j(h,x) = R(omega_j^T u(x)) h_j.

36 Gaussian centers start on the R44 6x6 grid; initial standard deviation1/3
in normalized coordinates. P=L L^T with positive exponential diagonals and
free lower entry, as R44. Its inverse has the GS rotation/axis-scale form.
r0 is the fixed initial RMS basis energy over the432 native grid centers,
without labels. Reuse R44 basis code/initialization without changing that source.
V has shape36x2, starts exactly zero, and is the only output mixture. The old
36x48 parameter is replaced, not retained as inactive capacity. All heads,
layers and directions share the map. Rotation preserves every content-pair norm.
Far from a finite Gaussian field with finite scales, d(x) tends to zero.
Nonlinear u does NOT guarantee raw-coordinate translation/rotation invariance,
injectivity, or that geometric positions have been corrected. It is a PE chart.

Eight new roles and added active parameter counts:

| role | definition | count |
| --- | --- | ---: |
| warp_full (primary) | learned centers/full precision/V |252|
| warp_iso | learned centers/isotropic precision/V |180|
| warp_fixed | fixed Gaussian basis, learned V |72|
| warp_fixed_centers | fixed centers, learned full precision/V |180|
| fourier_warp | R44 learned Fourier36 basis, zero36x2 V |126|
| mlp_warp | R44 GELU36 coordinate basis, zero36x2 V |180|
| affine_warp | d=e*(A z), A=zero2x2 |4|
| free_frequency | original omega_j + norm(omega_j)*zero2-vector offset |48|

Fourier and MLP use the same fixed initial RMS convention as Gaussian maps.
No bias is added to the output map: a common constant shift would cancel in
shared Q/K rotary scores. Affine also has no independently learned translation.
Free frequencies are a local mechanism control, not a full RoPE-ViT reproduction.
An affine map is expressible by suitable free frequencies in relative scores.
Shapes of active parameter tensors differ honestly; no dummy capacity padding.

V=0 intentionally makes basis gradients zero on step1. Verify nonzero V
gradient, then basis gradients after a nonzero V update. Record actual first-step
gradients and selected-parameter movement without declaring this expected zero
an implementation failure. All complete network outputs must initially match
point RoPE exactly for all five paired seeds on toy inputs.

## Fixed execution and records

Reuse the exact R5 caches already used in R43/R44:576 train pairs in c3/c5/c6,
64 val in c8,480 OLD evaluation pairs in c11/hotel_umd/mit_lab_hj/
mit_76_studyroom/mit_dorm_next_sj. Full scene/pair/source/data identities are
copied from the verified R43 lock. The unused home_ac/hotel_barcelona/hotel_graz/
mit_1_242/harvard_computer_lab cohort is not accessed or renamed.
All432 keys and all110189 old-evaluation valid queries/model are retained.
Predicted geometry, fixed128D features, known cameras and label visibility
population are unchanged. The forward method uses only the original ten fields;
physical covariance is accepted by the interface but unused by these new roles.

40 new runs=8roles x seeds11,22,33,44,55. Reuse the six unchanged R5 roles'
30 R43 sealed runs/predictions only after complete artifact identity checks.
New backbone initialization and1000 sampled batches must match those controls.
Same batch8, AdamW lr.001/wd.0001, clip1, cosine multiplier1 to.2, validation
at0,100,...1000 and earliest strict minimum scene-macro CE selection.
No method-specific learning rate, duration, field-size or bandwidth selection.
This equal-budget comparison does not establish convergence or optimization
over all possible settings. All40 selected checkpoints are sealed before their
old-evaluation export. Record primary and zero-V per-query sufficient statistics,
full metric rows, histories, first gradients, parameter movement, timing/memory.

Use truly idle GPUs with current occupancy checks, CPU4, TF32off, deterministic
algorithms, <=16GiB allocated and <=3GPUhours/seed. A bounded queue records
actual child exits and never stops another user's process. No live queue restart
based only on observation timeout. Checkpoint and data bytes remain unchanged.

Correctness: exact complete-network identity initialization; independent nonzero
Gaussian/Fourier/MLP/affine/free-frequency phase formulas; pair norms, relative
phase in mapped coordinates, finite differences/gradient flow and input scope.
The general Gaussian response's 2D/3D joint-frame algebra is already checked in
R44; no rebranding that as 3D task success. CUDA toy smoke for every new role.
Independently reconstruct first-layer sampled probabilities with NumPy complex
rotation, eight query rows/all keys/two heads of the first validation pair.
P error>5e-5 is an impact warning, not an automatic tiny-number veto. Full-query
metric replay uses saved argmax/softxy/log-partition/label logits; it is not a
second independent complete network. Author independence is not claimed.

## Decisions

Primary is warp_full throughout. Compare to each decisive control: strongest
legacy by PCK8, fourier_warp, mlp_warp, affine_warp, free_frequency.
Promising development: >=1pp scene-macro PCK8, >=4/5 positive paired seeds,
>=4/5 positive old rooms and CE worsening<=2% for every comparison.
Report all metrics and smaller effects, without claiming significance from
repeatedly used rooms. Component comparisons with fixed/iso/fixed-centers are
continuous attribution evidence, not each forced to earn a whole-method1pp.
Simplification needs a prospective comparison; do not relabel the best ablation.

Failure ends promotion of this frozen candidate, not all Gaussian PE research.
Do not tune against old_eval or open fresh data to rescue it. A pass justifies
native3D and another Transformer task, perturbation/scale/cost assessment and
only later unused joint confirmation. This round cannot complete the full goal.

## Prior work and scope checked before training

- [3D Gaussian Point Encoders §3](https://arxiv.org/html/2511.04797v1)
  already uses learned means, Cholesky precision and linear Gaussian mixtures.
  It evaluates PointNet/Mamba3D encoders and reports optimization difficulties.
  Our Gaussian basis is not novel; natural-gradient/distillation systems are
  not reproduced, and a fixed-budget negative result cannot refute them.
- [Learnable Radial Basis Positional Embedding, AAAI2023](https://ojs.aaai.org/index.php/AAAI/article/view/25307)
  already studies learnable radial position embeddings for coordinate-MLPs,
  with graph-Laplacian regularization. No novelty claim for using an RBF as PE.
- [HARoPE v1 §3.3](https://arxiv.org/html/2510.10489v1)
  uses a head-dimensional SVD transform on query/key CONTENT before rotation,
  q'=R_x A_h q, k'=R_y A_h k. This is not a2x2 coordinate warp; our affine
  control is not a reproduction of HARoPE. This formula-level reading refines
  the earlier abstract-only related-work note. Identity initialization is also
  established practice, not a novelty claim here.

The bounded search does not establish absence of a prior identical nonlinear
Gaussian-to-phase design. The proposed combination is an empirical hypothesis,
not a new RBF/rotation theorem or an asserted first-of-its-kind method.
