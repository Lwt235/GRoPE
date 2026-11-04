# R47: query-conditioned Gaussian positional channels

2026-11-01. R46 is complete: its global Gaussian warp beats native3D point
RoPE but loses to the ordinary MLP map. No candidate is promoted. This round
changes the scientific mechanism, not R46 bandwidth, width or training time.
Freeze this specification, sources and reused identities before new training.

## Hypothesis, prior work and alternatives

Predict a query-specific positional center/ellipsoid from the current token
state, so attention can adapt its spatial region to content. The hypothesis is
that a Gaussian precision constraint gives useful, transferable structure beyond
an equally sized unconstrained dynamic quadratic PE. Alternatives: fixed local
attention suffices; an input-independent ellipse suffices; only scalar bandwidth
matters; a generic quadratic predictor is as good; the chosen geometry is limiting.

This is an application experiment, not a new Gaussian-attention theorem.
[Yang et al., EMNLP2018](https://aclanthology.org/D18-1475/) already predict
Gaussian center/window from hidden representations.
[Cordonnier et al., ICLR2020, appendix E](https://arxiv.org/pdf/1911.03584)
already use learned non-isotropic Gaussian quadratic PE and report no accuracy
gain from non-isotropy in their experiment.
[SMCA, ICCV2021, section3.2](https://openaccess.thecvf.com/content/ICCV2021/papers/Gao_Fast_Convergence_of_DETR_With_Spatially_Modulated_Co-Attention_ICCV_2021_paper.pdf)
predicts query/head-dependent2D centers and axis scales. This experiment is not
a full replication of those systems or a claim that dynamic Gaussians are new.
Repository R6 already has Gaussian bias in a complete Transformer, but from
physical input covariance with one learned bandwidth and an additional FINAL
matching bias. Here the covariance is predicted from content and final matching
has no geometry. Neither physical rendering covariance nor a renderer is used.

## One positional operator

Keep R5's two bidirectional96D/2-head blocks, Q/K/V, residuals, FFNs and matching
head,166611 shared parameters. Replace rotary PE inside attention with:

    delta_ij = x_j - x_i
    score_ij = T * normalized(q_i)^T normalized(k_j)
               - .5 * (delta_ij - m_i)^T P_i (delta_ij - m_i).

One zero-initialized linear predictor from LayerNorm(token),96 dimensions,
outputs each head's d center offsets and d(d+1)/2 precision parameters.
It is shared across both layers and directions. No extra depth, value, matching
or shape-similarity branch. Predict mean m=4*tanh(raw_mean/4), shape
t=3*tanh(raw_shape/3); L=diag(exp(t_diag))+strict_lower(t_off), P=L*L^T/16.
Initial width is4 in the already-frozen coordinate units, same for both domains,
chosen before scoring. This is not a physical standard deviation in metres.

Let b=P*m. Remove the query-only constants, including Gaussian log determinant
and any query opacity: they cancel in softmax. With a common coordinate origin,
the positional score has an EXACT dot-product lift:
Qpos=[P*x_i+b, -.5*P_aa, -P_ab(a<b)], Kpos=[x_j,x_ja^2,x_ja*x_jb].
Use matching upper-triangle order: five channels for2D, nine for3D. Subtract
the mean query coordinate from both positions for numerical conditioning only.
Compute this small positional product in float64, add in the existing float32
score dtype, and leave attention temperature/content normalization unchanged.
There is no additional rotary operator. The quadratic identity is known algebra.

This gives a relative positional operator. Under a JOINT change of coordinates,
centers and precision, the Gaussian energy is invariant; a trained predictor
does not automatically rotate its outputs when raw coordinates rotate. Such
end-to-end robustness remains an empirical requirement, not a claimed symmetry.

## Fixed comparisons and inputs

For BOTH d=2 and d=3 train these six roles, plus one NoPE model (13x5=65 runs):

| family | learned positional parameters2D /3D | meaning |
| --- | ---: | --- |
| conditional |970 /1746|query-dependent center/full precision, primary3D|
| diagonal |776 /1164|query-dependent center/axis scales|
| isotropic |582 /776|query-dependent center/one scale|
| static |10 /18|head-dependent constant center/full precision|
| fixed |0 /0|zero offset, fixed width4|
| free_quadratic |970 /1746|equal-capacity unconstrained quadratic predictor|
| nope |0|content attention only, trained once|

The free-quadratic predictor uses exactly the full model's output layout and
same bounded raw transforms. Define U=diag(t_diag)+strict_lower(t_off),
P=(I+U+U^T)/16 and b=m/16, with no SPD constraint. This matches the full Gaussian's
initial score and first derivatives. Thus extra conditional capacity and initial
optimization scale are controlled; negative precision directions are permitted
and must not be secretly clipped. This control is not called a Gaussian.

All new models have paired shared initialization. All six positional families
start at exactly the same fixed isotropic function within each coordinate domain;
NoPE deliberately differs. They need not initially equal rotary baselines.
All active position parameters can receive first-step gradients; no padding.

Reuse R46's center-lifted2D/3D data byte-for-byte, already independently audited:
576train,64val,480old_eval pairs; original features/labels,432keys and110189
valid old_eval queries. Same predicted depth and supplied camera metadata. No
new geometry construction or raw/fresh frame access. Preserve R46 data manifests,
array hashes, audit identities and original label hashes. NoPE ignores positions.
No covariance, label or visibility mask enters the positional predictor.

Reuse30 sealed R46 models with complete output identities: center2d->rope2d,
point3d->rope3d, gaussian2d->warp2d, gaussian3d->warp3d, mlp3d and fourier3d.
These are matched unchanged baselines; no relabeling of changed-input evaluation.

## Training, checks and decisions

Seeds11/22/33/44/55; same1000updates/batch8/AdamW(.001,.0001),clip1,cosine1->.2,
same sampled batches and shared initialization as R46. Select earliest strict
minimum val scene-macro CE at0,100,...1000. No role-specific tuning or added
training budget. Seal all65 models before new old_eval exports. Retain outcomes
for all95 new/reused models. A fixed budget is not a convergence proof.

Primary remains conditional3d, not the best member after scoring. Compare against
rope2d,warp2d,rope3d,warp3d,mlp3d,fourier3d and free_quadratic3d. Promising
development requires>=1pp PCK8,>=4/5positive paired seeds,>=4/5positive old rooms
and CE<=2%worse for EACH comparison, as in the preceding screen. Report continuous
effects of fixed/static/isotropic/diagonal controls without requiring every
component to earn1pp. Prespecified conditional2d comparisons show whether effects
depend on coordinate representation; a positive2D result cannot replace a failed
primary. Do not rename an ablation winner into the primary or treat old rooms as
fresh evidence. Positive generic quadratic/locality effects alone do not establish
the Gaussian hypothesis. A pass still needs cost/robustness, another Transformer
task, and unused joint confirmation. A fail ends promotion of this frozen design.

CPU tests: independent direct Gaussian/quadratic energy vs lifted probabilities
in2D/3D at nonzero parameters; shared initialization/parameter counts; SPD/full
shape gradients and finite difference; matched free-quadratic first derivatives;
translation/permutation and joint geometric-transform identities; input scope.
CUDA smoke tests all13 roles before training. Check selected first-layer score
samples (eight query rows/allkeys/twoheads) against separate NumPy prediction and
DIRECT pair energies. P error>5e-5 warns, assessed by impact rather than an
automatic scientific veto. Recompute every final query metric from saved outputs.
These are independent formulas by the same author, not a second full network.

Use verified idle GPUs, CPU4, deterministic algorithms,TF32off, <=16GiB allocation
and<=3GPUhours/seed. Record real parent/child exits. Do not touch other users' jobs.
Preserve fixed-width and anisotropy failures; no static integral accuracy or CPU
quadrature-speed gate is introduced.
