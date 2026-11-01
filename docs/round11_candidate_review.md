# Conditional Round 11 proposal: retain local covariance in four existing strata

2026-10-30. This is a design review, not an experiment result. It uses the parent
agent's reported Round 10 A outcomes and the already implemented operators. No
new task labels, confirmation inputs or candidate scores were read or run for
this review. All frozen Round 10 sources and outputs remain unchanged.

## What the present evidence does and does not justify

The parent reported that A improves over the strong point control by about
19.18%, but is about 1.25% worse than quad16 overall. Its q16→g8 condition is
about 3.91% worse, whereas q8→g8 is about 2.76% better. This is consistent with
a loss from compressing larger query support into one Gaussian. It does not
prove that mechanism: geometry errors, conditioning on visibility, and the
shape of native cells can also contribute. The already declared B experiment
isolates native key integration. A similar coarse-query pattern after B would
make the query-summary hypothesis more specific.

The smallest justified further design is not a new mixture network. Existing
quad4 and quad16 already form positive mixtures of Gaussian nodes, because
they retain within-pixel covariance. The proposed change is one additional
term in the covariance of each existing quad4 node: the projected positions'
scatter **within that stratum**. Keep every mean, weight, parent content
feature, value, key integration rule and h/T calibration budget unchanged.

## One formula and one attribution change

Partition each real query cell into its existing deterministic 2×2 equal-area
strata. For stratum r, let projected pixel positions be x_p and within-pixel
covariances be C_p. Use

\[
 a_r=|S_r|,\qquad
 \mu_r=\frac1{a_r}\sum_{p\in S_r}x_p,\qquad
 \Sigma_r=\frac1{a_r}\sum_{p\in S_r}
  \{C_p+(x_p-\mu_r)(x_p-\mu_r)^T\}.
\]

The parent query measure and native-key score are

\[
 p_q(x)=\sum_{r=1}^{4}\frac{a_r}{A_q}\mathcal N(x;\mu_r,\Sigma_r),
 \qquad
 w_{qk}=\sum_{r=1}^{4}\frac{a_r}{A_q}
  \int_{R_k}\mathcal N(x;\mu_r,\Sigma_r+h^2I)dx,
\]
\[
  s_{qk}=T\langle f_q,f_k\rangle+
    \log\{w_{qk}+10^{-30}A_k/64\}.
\]

Only the existing parent content dot product is used. Giving each component a
new content head or learned weight would destroy the intended attribution.
All strata participate, including those with poor predicted geometry; sensor
visibility never chooses their weights or removes them. The component number
is fixed at four before evaluating scores, rather than selected after trying
many mixture sizes.

The law of total covariance gives an exact useful check:

\[
 \mu=\sum_r\alpha_r\mu_r,\qquad
 \Sigma=\sum_r\alpha_r[\Sigma_r+(\mu_r-\mu)(\mu_r-\mu)^T].
\]

These are the original one-Gaussian query moments. Thus four full-moment
Gaussians retain the same global first two moments while preserving four local
centers; they do not obtain a better global covariance from new information.
Quad4 uses the same centers and weights but omits the within-stratum scatter.
That is the cleanest matched comparison.

This representation uses Gaussian parameters to define a positional measure.
It is a known positive Gaussian-mixture integral and remains a pairwise relative
PE for unequal key cells. It is not a new probability kernel, an exact finite
five-channel Q/K map, or evidence about optimized rendering primitives' R/S.
The first probe should preserve A/B's projected-pixel moment construction;
changing simultaneously to projected 3D moments would confound perspective
linearization with the covariance closure. A later claim about reusable 3D
parameters would require that separate check.

## A diagnostic before another task-score search

On a fixed metadata-defined old-training subset only, define a dense forward
reference using **every** projected predicted-geometry pixel and its known
within-pixel covariance, with the same blur and native key integral. This is
not sensor truth and must never count as a research effect result. It provides
an approximation reference under identical upstream geometry.

Measure three differences, without labels or choosing an advantageous subset:

1. One Gaussian versus the dense forward reference: total global moment-closure
   error at each query scale.
2. Four full-moment Gaussians versus the same dense reference: residual
   non-Gaussian structure within each stratum.
3. Quad4 versus four full-moment Gaussians, and quad16 versus the same dense
   reference: the cost and benefit of retaining within-stratum covariance.

Use the same normalization/floor, report every declared scale/partition, and
audit the total-moment identity. Dense-reference approximation error can
support or refute the hypothesized mechanism, but a reduction does not imply
better sensor-truth correspondence: the upstream prediction itself can be
wrong. No label-driven covariance clipping, stratum selection or mixture-count
search should be added to force the diagnostic positive.

If the four-node closure does not reduce the large-query approximation loss,
or loses its expected representation/compute advantage, stop this candidate
before a new broad calibration run. It would then be adding components without
evidence for the intended mechanism.

## Minimum controls and effect test

Use the same actual-cell operator on all roles:

- One full Gaussian, preserving the B reference.
- Four nodes with within-pixel covariance only (existing quad4).
- Four nodes with within-stratum full covariance (the sole new candidate).
- Four nodes with the same per-node trace but isotropic covariance.
- Sixteen nodes with within-pixel covariance only (existing quad16).
- The existing strong center/point and size-aware controls, including the
  separately selected point-PCK control.

An optional fixed 90° covariance rotation uses the candidate's h/T without
extra calibration; it is a mechanism check. Native key integration, parent
content, feature pooling, raw pixel population, and numerical floor are shared.
Preparation time must include computing all four local covariances. Storage
and scores must be counted honestly: this is four Gaussians per token, not
one Gaussian at the same exact cost. Compare the complete incremental PE
latency against optimized quad16, not against a deliberately slow control.

Keep the previous research gates unchanged: point improvement, both-scene
direction, the declared accuracy-or-speed tradeoff against strong quadrature,
partition protection and point-PCK protection. Passing the forward numerical
diagnostic is not an exemption from any of them. Since A and B have already
used these development scenes, a positive development result needs fresh
locked confirmation; it cannot be described as a new independent test.

If this single justified extension fails, do not keep increasing the number of
components until a reused gate passes. Retain the narrow existing result and
state that the more general objective is still unachieved.

## Two intrinsic limits to diagnose honestly

**First two moments cannot characterize arbitrary support.** In one dimension,
equal mass at `−1,+1` and masses `1/4,1/2,1/4` at `−sqrt(2),0,+sqrt(2)` both have
mean zero and variance one. They give different probabilities to many cells.
Adding the same small continuous noise to both preserves that counterexample.
Any deterministic score based only on those shared first two moments must give
the same output to both, so it cannot recover both exact coverage measures.
Four local Gaussians reduce this compression ambiguity for some supports but
do not remove the general finite-summary limitation.

**Area-distribution argmax and center-point correspondence can conflict.** A
query's central pixel can lie on a thin foreground structure while most of its
area belongs to a different background cell. For example, if the true regional
distribution over two distant cells is `(0.8,0.2)` but the center point belongs
to the second cell, exact region matching predicts the first as argmax. Forcing
the second cell to be an argmax requires at least

\[
 D_{KL}((.8,.2)\Vert(.5,.5))
   = .8\log 1.6+.2\log .4\approx .1927
\]

nats of regional KL in this two-cell example. This establishes that universal
zero-area-error plus perfect center PCK is impossible for one shared argmax.
It does **not** establish that the actual finite improvement/protection gates
are inconsistent or that their observed failure is unavoidable.

A future evaluation-only diagnostic can compute PCK of the independent sensor
area distribution's own argmax against the existing sensor center target,
retaining the same validity population and deterministic tie rule. Report the
frequency and margin of conflict. This diagnostic must remain outside model
inputs and must not be used to silently waive the predeclared PCK guard. If the
targets turn out to demand incompatible behavior in practice, that is a task
definition finding to explain, not permission to relabel a failed design.

Even a successful four-Gaussian regional encoder would not by itself resolve
Round 8's source-value bottleneck or validate optimized GS internal covariance
in Round 9. The broader user objective requires those distinct empirical claims
to be checked on their own appropriate inputs and strong controls.
