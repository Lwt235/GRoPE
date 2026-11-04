# R46: native 3D positional maps with center-geometry controls

2026-11-01. Freeze sources, old pair identities and RGB-prediction frame-cache
hashes before new array decoding/tests/training. R45 remains a negative utility
result: +.006694pp versus point,2/5 winning seeds. It is not silently promoted.

Hypothesis: GS-form learned spatial centers/full ellipsoids are useful in native
3D positional coordinates inside a full Transformer. Alternatives: dimensional
representation/coordinate normalization or changing patch centroids to pixel
centers explains gains; ordinary learned maps suffice; the Gaussian freedom is
redundant. This extends the tested domain, not a rescue bandwidth/LR sweep.
Even a positive3D result cannot erase R45's2D limitation or establish cross-task
generality. No renderer or static integration prerequisite is introduced.

## Inputs and the center/region control

Use exactly the already-used R5/R43 pair populations:576 train,64 val,480
old_eval pairs in the same nine rooms. No new frames, input features or labels.
Frame caches contain RGB-predicted metric depth,128D features, resized K and
known reconstruction c2w. Those poses are supplied metadata, not RGB-only pose
predictions. Preserve hashes for all1605 scene/frame records, old pair input/
label identities and prior model outputs. The unused confirmation cohort is
not accessed. All432 keys, original valid labels and110189 old-eval queries
per model remain; no depth/visibility-based token removal or attention mask.

For each token center pixel (8i+3.5,8j+3.5), sanitize predicted depth
(NaN->10,+inf->20,-inf->.1), clamp [.1,20] meters, then average the four center
pixels. This is bilinear sampling at the half-pixel center. Lift that single
center ray with K inverse. This is distinct from R5's uniform64-point patch
centroid. Sensor depth is never used for forward geometry.

For each ordered pair use the query camera as common3D frame: q center points
and transformed key center points, T=inverse(c2w_q)*c2w_k. On their concatenation
of864 points, compute origin=mean and r=max(RMS Euclidean radius,1e-3m). All3D
roles receive x=12*(point-origin)/r for BOTH attention directions. This fixed
isotropic normalization uses all input points, no labels, score selection or
learned calibration. It preserves a common frame and removes global origin/
unit dependence away from depth/radius clamps. It does not guarantee invariance
to reference-camera swaps, physical view changes or arbitrary raw rotations of
the learned field. Those task robustness questions remain outstanding.

New center2d/gaussian2d roles receive the same center-lifted geometry projected
into each attention's image frame, with original token-stride units. Native
positions remain exact grid centers. Cross projection uses z=max(z,.1), then
clips each2D coordinate to[-128,128], matching the prior finite convention.
Input covariance interface placeholders are zero and unused by every new role;
learned field covariances are separate PE parameters, not physical GS support.
Visual feature bytes must exactly equal every original pair's q/k arrays.
Original label bytes are reused directly, without re-export or changed masks.

Build train/val after freeze, compare ALL new coordinate arrays to independent
pinhole/world-camera formulas, then train. Build old_eval only after all selected
models are sealed, audit its geometry, then export predictions. No fresh data.

## Complete Transformer, positional forms and controls

Keep R5's two-block bidirectional Transformer,96D internal channels, two heads,
Q/K/V, residuals, FFNs and matching head. Backbone166611 parameters is identical
across methods/seeds; batch identities also match the existing R43 point runs.

Nine3D roles use24 unit sphere directions with fixed Fibonacci construction:
z_j=1-2(j+.5)/24, theta_j=j*pi*(3-sqrt5), direction=(sqrt(1-z²)cos(theta),
sqrt(1-z²)sin(theta),z). Magnitudes logspace(-1,.6,24). This fixed dimensionless
frequency range and coordinate factor12 are chosen before data/scoring, not a
tuned physical length scale or exact isotropic spherical quadrature claim.

Primary gaussian3d:

    z=x/12
    phi_g(z)=exp(-.5*(z-m_g)^T P_g*(z-m_g)), P_g=L_g L_g^T
    u(x)=x+12*(phi(z)/r0)@V
    PE_j(h,x)=R(omega_j^T u(x))*h_j.

64 centers on4x4x4 normalized grid {- .75,- .25,.25,.75}; initial spatial
standard deviation.5. Parameterize L=2*(diag(exp(logdiag))+strict-lower),
so SPD inverse precision has a GS rotation/axis-scale representation. Three
off-diagonal parameters/Gaussian avoid ambiguous eigenvectors. V is64x3 and
zero initialized. r0 is fixed initial RMS basis-vector energy on an8x8x8
uniform cube grid; no data-dependent basis normalization. All heads/layers/
directions share one field. No new content/value/head/shape branch.

| role | positional parameters | added count |
| --- | --- | ---: |
| gaussian3d (primary) | learned center/full precision/V |768|
| iso3d | learned center/isotropic precision/V |448|
| fixed3d | fixed Gaussian basis, learned V |192|
| fixed_centers3d | fixed centers, learned full precision/V |576|
| fourier3d |32 learned3D frequencies/phases ->64 trig bases -> zero V |320|
| mlp3d |3->64 GELU basis with learned weights/bias -> zero V |448|
| affine3d | u=x+A*x, zero3x3 A, no cancelling translation parameter |9|
| free3d | w_j=w0_j+norm(w0_j)*zero3-vector offset |72|
| point3d | fixed3D frequencies |0|
| center2d | original2D point RoPE on center-lifted geometry |0|
| gaussian2d | exact frozen R45 warp_full on center-lifted geometry |252|

Fourier directions use the same Fibonacci construction with32 directions and
frequency magnitudes. At z=x/12 its angle is z@(12*w0+12*delta_w)+phase.
MLP base weights are N(0,1/3), with zero learned offsets/bias. Both use the
same cube initial-energy convention and zero64x3 output. No inactive padding.
All3D new models initially exactly match point3d; the two2D roles initially
match original point2D with the same geometry. No claim that2D and3D functions
initially coincide. Rotations preserve content-pair norms. At zero V, first
geometry gradients are intentionally zero; verify gradient flow after V updates.

55 new runs=11roles x5seeds. Reuse ONLY the five unchanged R43 point runs,
renamed legacy_point2d in the table, with exact artifact hashes. They preserve
old centroid-based cross projection and are historical geometry controls, not
geometry-matched3D baselines. No reuse of a model with changed inputs as if it
were a separately trained matched control.

## Execution, proportional audit and decisions

Seeds11,22,33,44,55;1000updates,batch8,AdamW lr.001/wd.0001,clip1, cosine
multiplier1->.2. Same RNG seed+50000 sampled batches, matching shared neural
initialization. Validate0,100,...1000, select earliest strict minimum scene-macro
CE. Seal55new+5historical records before old_eval construction/export.
No role-specific LR, frequency, width, duration or temperature scan. A fixed
budget comparison does not prove convergence or impossibility under all budgets.

CPU4,TF32off,deterministic algorithms,<=16GiB allocated and<=3GPUhours/seed.
Launch only on freshly checked idle GPUs; preserve others' processes. Capture
actual child exits. Record full primary/zero-V outputs and metric sufficient
statistics, sampled independent phase probabilities, selected parameter changes,
history, timings and memory. Treat numerical warnings by their effect on the
claim, never inflate an irrelevant tiny discrepancy into a scientific failure.

CPU preflight: all-role/seed exact initial backbone+point function, parameter
counts, nonzero independent3D phase formulas/norms, all three lower-triangle
shape directions, gradient/finite difference, center vs patch-centroid definition,
independent pair geometry, common world-gauge/unit transformations away from
clamps, input scope and affine-frequency equivalence. CUDA smoke all11 roles.
Data auditor compares all coordinates with separate pinhole/world-camera
formulas (float32 tolerance3e-5 absolute+3e-7 relative), and exact q/k feature
identity. Phase audit checks first layer, eight query rows/all432 keys/two heads
for every selected model against NumPy complex rotation; P>5e-5 warns. Full
query metrics replay from argmax/softxy/log-partition/label logits. These are
independent formulas, not another author or a second complete network.

Primary gaussian3d must exceed EACH of legacy_point2d,center2d,point3d,
gaussian2d,fourier3d,mlp3d,affine3d,free3d for promising development: >=1pp
scene-macro PCK8,>=4/5 positive seeds,>=4/5 positive old rooms,CE<=2%worse.
Report all smaller effects without renaming a winning ablation. Fixed/iso/
fixed-centers differences are continuous component attribution, not each
required to gain an entire-method1pp. Old rooms are reused, not fresh inference.

Report center2d-minus-legacy to assess the center-lifting geometry change;
gaussian2d-minus-center2d and gaussian3d-minus-point3d isolate their respective
PE effects. point3d-minus-center2d includes dimensional representation,
normalization and fixed frequency schedule, not dimension alone. The two
Gaussian fields also differ in basis count/capacity; do not overclaim a pure
dimension-only interaction. A pass justifies further robustness/cost and a
second Transformer task before unused joint confirmation. A fail ends promotion
of this frozen candidate, without erasing previous failures or trying fresh data.

Gaussian/RBF bases and learned positional maps have known precedents reviewed
in R44/R45, including [3D Gaussian Point Encoders](https://arxiv.org/html/2511.04797v1).
This is a native3D PE application hypothesis, not a new Gaussian/rotation theorem,
a full reproduction of that point encoder, or proof that physical GS R/S helps.
