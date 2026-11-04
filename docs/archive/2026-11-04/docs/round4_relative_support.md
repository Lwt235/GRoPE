# Fixed-query Gaussian PE as relative-support filtering

This is a checkpoint reparameterization of `gaussian_konly`. It removes the dynamic query filter call while retaining 48-dimensional Q/K and the same learned parameter count. It neither proves that fixed query covariance is sufficient for the task nor improves the mathematical prediction of a converted checkpoint. Choosing `gaussian_konly` over variable-query `gaussian_full` remains an empirical decision.

## Identity on the ordinary normalization domain

Let the shared content encoder produce `h = Wz + b` at its last linear layer, and let `N(h)=h/||h||`. For frequency block `f`, define

\[
D(C)_f=\exp[-\tfrac12\omega_f^T C\omega_f]I_2,
\qquad R(\mu)_f=R_{\omega_f^T\mu}.
\]

The filter is scalar within each two-channel rotary block, so `D(C)` commutes with every `R(μ)`. With positive finite filters and nonzero content, the implemented normalized Gaussian response is

\[
\Phi(h,\mu,C)=R(\mu)N(D(C)h).
\]

The preliminary content normalization cancels inside the second normalization. This expression assumes the preliminary content has unit norm; the implementation's epsilon branch is discussed below.

For the fixed-query model,

\[
C_0=(1/12+\delta^2)I,\qquad C_{k,\mathrm{eff}}=C_k+\delta^2I.
\]

Set `D0=D(C0)`, `h'=D0h`, and convert the encoder's last layer by `W'=D0W, b'=D0b`. Then

\[
Q'=R(\mu_q)N(h'_q)=Q,
\]

\[
K'=R(\mu_k)N\left(D(C_k-I/12)h'_k\right)
 =R(\mu_k)N\left(D(C_{k,\mathrm{eff}})h_k\right)=K.
\]

This follows from the exact elementwise identity

\[
D(C_k-I/12)D_0=D(C_k+\delta^2I).
\]

Consequently all candidate logits `exp(logtemp) Q Kᵀ` agree in real arithmetic on this domain. The temperature clamp is unchanged. Query and key means still use the existing common query-image coordinate system; this algebra does not remove camera projection, correspondence geometry, or fitted query means.

## The dustbin must use the recovered content

The old rejection logit is `dustbin + head(N(hq))`. Applying the same head directly to `N(h'q)` would alter the model, because `D0` generally changes the content direction. Preserve it as

\[
\ell_\varnothing=\mathrm{dustbin}
 +\mathrm{head}\left(N(D_0^{-1}h'_q)\right).
\]

`RelativeSupportMatcher` stores the inverse diagonal reference and performs this recovery. Thus the identity includes all 49 logits, softmax normalization, and unmatched probability, rather than only candidate rankings.

## What relative support means here

`Crel=Ck-I/12` can have negative eigenvalues. Its exponential response is the ratio of two positive Gaussian filters; it is not the characteristic function of a Gaussian with covariance `Crel`. In particular, amplitudes greater than one are permitted, and positive-definiteness as a distributional kernel is not claimed. A smaller key footprint relative to the fixed reference amplifies the corresponding transformed content frequencies before normalization.

For physical `Ck` positive semidefinite, `Crel ≥ -I/12`. Thus each relative amplitude obeys

\[
0<D(C_{\rm rel})_f\leq\exp(\|\omega_f\|^2/24).
\]

For the current frequencies, whose largest norm is approximately `3.9811`, the maximum relative gain is approximately `1.9355`. At `δ=.25`, the smallest reference amplitude is approximately `0.3148` and the largest inverse reference is approximately `3.18`. These bounds concern the frozen frequency range and resolution; arbitrary frequency magnitudes or resolutions could make the conversion ill-conditioned or underflow its weights.

Although the explicit relative covariance no longer contains `δ`, the converted weights and inverse-reference buffer depend on it. Changing `δ` after conversion does not preserve the original function. The common blur was absorbed into the encoder; it was not proven irrelevant.

More generally, a shared static diagonal spectral filter adds no ordinary-domain function-class capacity when it can be absorbed into a freely adjustable final affine encoder. For example, the common isotropic floor `δ²I` alone can be absorbed even for variable-query `gaussian_full`, while retaining both token-dependent physical covariance filters and recovering the original dustbin content with its inverse. This observation concerns expressiveness, not the fitted checkpoint or training bias: initialization, regularization, optimization and finite precision still depend on the parameterization. Token-dependent covariance filtering cannot in general be absorbed into one static shared encoder. No additional implementation or trained variant is needed to establish this algebraic point.

## Floating-point and tiny-content limitations

`F.normalize` uses `h / max(||h||, eps)` with default `eps=1e-12`. When an encoder norm is below epsilon, its output has norm below one. The original `normalized_rotary` preserves this input norm, whereas the converted plain rotary preserves the potentially different norm of `F.normalize(D0h)`. Thus the current conversion is **not an exact identity for all tiny finite encoder outputs**, even in double precision. Zero content maps to zero in both implementations, but this does not extend the identity to a neighborhood of zero.

A simple counterexample sets the original encoder output to `5e-13` in the highest-frequency channel and zero elsewhere, with physical key covariance `I/12`. Both original Q/K norms are `0.5`; the converted norms are approximately `0.1574` because that channel is scaled by the reference amplitude. The candidate logits differ materially. The inverse-reference dustbin recovery can still agree in that example, so checking only dustbin logits would miss the problem. This is a real limitation, not an accuracy tolerance to hide.

On ordinary content, row-wise checkpoint multiplication, inverse recovery, and changed evaluation order introduce rounding differences. Plain query rotary uses double intermediates consistently with the reference, but neither bitwise identity nor exact equality of near-tie decisions follows. Before adopting a converted checkpoint, measure the minimum original and converted raw encoder norms, maximum errors of every logit and probability, and any prediction/metric changes on the intended data. The current unit tests check all logits in float64 at resolutions `0`, `.25`, `.5`, and float32 with negative relative covariance, including the nonlinear dustbin input. They do not establish a universal tiny-content identity.

## Scope of the simplification

- The fixed covariance must be shared by every query and compatible with the frozen rotary frequencies. Token-dependent fitted query covariance cannot be absorbed into one shared last linear layer.
- The shared encoder must end in the affine output being reparameterized. Conversion occurs after its nonlinear hidden layers; it is not an insertion before GELU.
- The Gaussian exponential admits the covariance difference because its log-amplitude is linear in covariance. The resolvent `1/(1+v/2)` does not obey the same covariance subtraction identity.
- The invertible row scaling gives an equivalent ordinary-domain function class. Reinitializing or continuing training with Adam, weight decay, or the original optimizer state need not follow the same trajectory. No training-path equivalence is claimed.
- This construction is an algebraic simplification of the existing normalized spectral response. It supplies no new evidence of positional-encoding novelty or covariance usefulness. Those claims must come from the frozen experiments and appropriate fixed/trace/area/scale controls.

The practical benefit to test is full-matcher inference cost, including dustbin recovery, not only the removed query-filter primitive. A useful audit converts trained checkpoints without fitting anything, checks the domain and complete outputs, and measures repeated warmed timings at the same batch/token sizes. Any CPU timing result is specific to that implementation and hardware; it does not establish GPU-kernel speed or a gain over the variable-query model.
