# Unselected reference alternative: full-Gaussian whitening and positive radial quadrature

2026-11-01. Theory and frozen-source review only after the accepted R33
per-call work-budget failure. No real/formal input arrays, labels or new
integrations were read or run. This is not an implementation, reference
selection, numerical acceptance or efficacy decision. R33's work failure
does not establish a probability mismatch or invalidate its conditional law.

## Formula and independent preparation

Use the same RMS law, not an additional screen convolution. For RMS depth
`s>0`, let `m=μ/s`, `G=F/s` and form the rectangular augmented factor

\[
D=[G,\;he_x,\;he_y],\qquad V=DD^T.
\]

When the original depth row is nonzero, `V` is positive definite in exact
arithmetic. A full-factor QR of `Dᵀ`, with diagonal signs normalized, yields
`V=LLᵀ` without forming the covariance Gram matrix. This differs from the
producer's latent-depth decomposition, conditional 2D factor and algebraic
δ expression. A separately implemented SVD factorization is another possible
route, but neither makes a badly conditioned input automatically accurate.

At screen position `y`, put

\[
r=(y_x,y_y,1)^T,\quad w=L^{-1}r,\quad b=L^{-1}m,\quad
\rho=\|w\|,\quad u=w/\rho,\quad t=u^Tb.
\]

The change of variables `X=zr`, with Jacobian `z²` and strict `z>0`, gives

\[
g(y)=\frac{e^{-d^2/2}}{2\pi\det L\,\rho^3}
       [(1+t^2)\Phi(t)+t\phi(t)],
\qquad d^2=\|b-tu\|^2.
\]

This is the same raw positive-depth subdensity, with total mass
`Φ(μz/||Fz||)`, not a density divided by front probability. Its derivation
is the earlier radial identity; no novelty claim is made.

For the residual, a complete Householder QR of the scaled vector `u`
produces an orthonormal frame `(u,Q⊥)`; use `d²=||Q⊥ᵀb||²`, with the first
QR column's sign accounted for in `t`. Do not subtract `||b||²−t²`.
Equivalently, a suitably scaled QR of the two-column matrix `[w,b]` gives
the perpendicular component. Orthogonal transforms reduce cancellation but
do not certify tiny residuals: errors of order machine precision times
`||b||` can dominate an almost-parallel residual. Orthogonality and solve
residual diagnostics are necessary evidence, not forward-error bounds.

All determinant/norm factors and masses belong in the log domain. For
`t≥0`, the two bracket terms are nonnegative and can be combined by log-sum;
`log(1+t²)` can use `2log hypot(1,t)`. Extreme intermediate exponents still
need declared range/failure rules; a finite mathematical density does not
ensure this parameterization is representable in float64.

## Negative t: an independent positive integral

For `t=−q<0`, avoid the erfcx subtraction and the producer's fixed 20-term
series. The exact identity is

\[
K(q)=\int_0^\infty z^2e^{-qz-z^2/2}\,dz,\qquad
\log g=-\tfrac12\|b\|^2+\log K(q)
 -\tfrac32\log(2\pi)-\log\det L-3\log\rho.
\]

Computing `||b||²` directly avoids reconstructing it as `d²+q²`. Scaling
the positive scalar integral by `λ=max(1,q)` gives

\[
K(q)=\lambda^{-3}\int_0^\infty
 u^2\exp[-(q/\lambda)u-u^2/(2\lambda^2)]\,du.
\]

This also works as `q→0`; using `λ=q` there would collapse the transformed
integrand and introduce an artificial `q⁻³` singularity. Generalized
Gauss–Laguerre with weight `u²e⁻ᵘ` has a useful bounded multiplier:

\[
\begin{cases}
\exp[-u^2/(2q^2)]\le1,&q\ge1,\\
\exp[(1-q)u-u^2/2]\le e^{1/2},&0\le q\le1.
\end{cases}
\]

Positive weights and log-sums avoid subtractive negative-tail cancellation.
Nodes/weights can be precomputed and scalar arguments batched. However,
two Laguerre orders agreeing is convergence evidence, not a certified
relative-error bound; small-q convergence and all branch transitions must
be validated independently. The exact identity `K(0)=√(π/2)` is a useful
separate check. No Laguerre order or branch accuracy policy is selected here.

A finite-interval alternative has explicit scalar truncation bounds. For
the *scaled* integral above, truncation after `U≥1` leaves at most

\[
\begin{cases}
e^{-U}(U^2+2U+2),&q\ge1,\\
Ue^{-U^2/2}+\sqrt{\pi/2}\,\operatorname{erfc}(U/\sqrt2),&0\le q\le1.
\end{cases}
\]

In both branches its full integral is at least `e⁻³ᐟ²/3`, by integrating
over `[0,1]`. Thus these tails can be converted to a uniform relative
truncation bound, distinct from finite-interval quadrature and floating-point
errors. A log-coordinate integral has exponent
`3v−q exp(v)−exp(2v)/2`; centering it at its analytically known maximum can
also control scale, but requires two tail truncations and extra rules.
It is less obviously economical for millions of scalar evaluations.

## Structural and range limits

- Preserve exact depth structure. `s=0` and deterministic nonpositive depth
  are structural raw zero. Exact deterministic positive depth uses an
  independent 2D Gaussian rectangle reference; a full 3D inverse does not
  exist. A tiny nonzero depth variance is not this branch.
- RMS normalization removes arbitrary world scale but not conditioning.
  As depth variance tends to zero, full 3D whitening can involve enormous
  nearly parallel vectors while the screen law remains benign. Do not use
  a pseudoinverse, rank threshold, jitter or the production δ routine to
  silently rescue it.
- Retain the existing positive-front representability guard. A finite
  full-rank Gaussian has positive density at every finite screen position;
  a nonfinite positive log-density is failure, not structural zero.
- Keep canonical log raw masses, direct-FOV mass and actual partial outputs.
  Ordinary materialization is optional and indexed; underflow is not a
  substitute for raw relative-error checks. Add the area floor once, then
  the sealed bias, without first conditioning on front mass.

## Cost and independence of a screen GL reference

Fixed GL32/48 over the existing physical tiles of side at most `2h` removes
the multiplication by hundreds of initial depth intervals. For the current
24×18 grid of unit-area keys with `h=.25`, native and direct-FOV integration
together contain 3,456 tiles: approximately 3.54 million screen nodes at
order 32 and 7.96 million at order 48 per query. These are static work
counts, not timing measurements.

The scalar K rule adds another factor **only on negative-t nodes**, but that
fraction is not known here. For illustration, 48 K nodes at every GL48
screen node would exceed 382 million scalar-integrand evaluations. Thus
this route is not automatically within the old 250-million budget. Report
screen-density nodes, K nodes and linear-algebra work separately; changing
the meaning of a charged work unit must not be described as unchanged cost.
Batched evaluation can remove Python overhead, not the mathematical work.

The density evaluation would be independent in algebra, preparation and
negative-tail computation. The screen tensor-GL family and possibly tile
boundaries would be shared with production. This is useful independent
agreement evidence, but weaker against a shared spatial-quadrature error
than the conditional-depth route. In particular, aligned native and FOV
tile sets can contain exactly the same nodes; their additivity check then
mostly checks bookkeeping rather than a second integration method.

The existing RMS derivative proof gives the exact-arithmetic absolute bound

\[
\sum_B|E_B|\le
\frac{\mathrm{FOVarea}}{\pi h^2\binom{2n}{n}}
\left(\frac{\mathrm{tileSide}}{2h}\right)^{2n}.
\]

It applies at `n=32,48` too, provided the density and quadrature are exact.
It helps explain why the RMS law differs from an unsmoothed narrow-ray
integrand, but does not cover whitening, K error or tiny-FOV relative error.
The unchanged raw-FOV, raw-L1, probability and reference-refinement gates
remain necessary. Agreement with accepted conditional references on the
unchanged synthetic population would be important cross-method evidence;
it would not certify all real inputs or permit reading efficacy labels.

**Assessment:** mathematically feasible and potentially much cheaper in
positive-branch-dominated cases, with clear independent scalar checks.
Its conditioning and negative-branch work are genuine blockers to assuming
it is the next efficient full-population reference. A separately frozen
numerical preflight would be required before replacing the conditional
reference. R33's budget-only failure supplies no reason to weaken its
accuracy gates or to promote an efficacy candidate.
