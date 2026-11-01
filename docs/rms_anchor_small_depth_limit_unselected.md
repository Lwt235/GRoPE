# Unselected review: 3D anchor second moments and the small-depth limit

2026-11-01. Mathematics and existing source/document review only. No new
candidate is selected; no input arrays, labels, R33 intermediate outputs or
numerical integrations were read or run. This extends the earlier
[anchor-construction memo](rms_projective_anchor_constructions_unselected.md),
without changing any frozen protocol or source.

## Exact relation to R20, and the required limit

Work in homogeneous screen/depth coordinates, after the same legal camera
map. Let the original support have mean `m` and covariance `Σ`, and let the
predicted 3D anchor be `a=(b,z₀)`, with `z₀>0`. Define

\[
u=b/z_0,\qquad d=m-a,\qquad Q=\Sigma+dd^T
 =\begin{bmatrix}A&c\\c^T&v\end{bmatrix}.
\]

The proposed fixed-mean Gaussian is `N(a,Q)`. Its depth variance is
`v=Σzz+(mz−z₀)²`: small original depth variance alone does **not** establish
the limit; the anchor/mean depth gap must also vanish relative to `z₀`.

If `v=0`, positive semidefiniteness gives `c=0`, and the RMS law is exactly

\[
Y\sim N_2\left(u,\;A/z_0^2+h^2I\right).
\]

If the original support also has this same constant positive depth, its
screen moments satisfy `μ₂=mxy/z₀`, `C₂=Σxy,xy/z₀²`. Hence

\[
A/z_0^2=C_2+(\mu_2-u)(\mu_2-u)^T=C_{a,2}.
\]

This is exactly the R20 **T0 geometric law**, followed by the same native
integration and floor. The statement assumes the same underlying support
and within-pixel covariance mapping; it is not a claim of bitwise equality
between the old float32 moment reductions and a new 3D path. It also requires
the strict projected anchor `u` to equal the R20 anchor: R20's target-depth
clamp and screen clipping invalidate that identity on affected rows. No
content-temperature rule follows from this limit.

More generally, with `Jₐ=[I₂,−u]/z₀`,

\[
J_aQJ_a^T=E\{[J_a(U-a)][J_a(U-a)]^T\}.
\]

It matches the second moment of the **linearized** original projection about
the anchor. It need not equal the second moment of its exact projection.
Retaining depth correlations before projection adds information relative to
a 2D closure, but a single 3D Gaussian still cannot recover arbitrary
higher-order patch shape discarded by `m,Σ`.

## What the RMS term adds, and why global moment language is unsafe

Write a sample from the anchored Gaussian as `(b+ξ,z₀+ζ)`, and set
`W=ξ−uζ`, `k=Cov(W,ζ)=c−uv`. With independent screen-numerator noise
`η~N₂(0,h²(z₀²+v)I)`, the exact projected law is

\[
Y=u+\frac{W+\eta}{z_0+\zeta},\qquad z_0+\zeta>0.
\]

For every `v>0`, the Gaussian depth density is positive at zero. Since the
independent added numerator noise has positive variance,

\[
E[\|Y-u\|^2;Z>0]\ \geq\
2h^2(z_0^2+v)\int_0^\infty z^{-2}p_Z(z)\,dz=\infty.
\]

The first absolute moment of each screen coordinate also diverges, by the
same argument with `1/z` and the minimum absolute first moment of a Gaussian
over its mean. Thus an ordinary whole-plane projected mean/covariance is
not available even for arbitrarily small nonzero `v`; principal-value
cancellation is not an expectation. Positive-depth conditioning does not
remove the divergence. Finite-FOV cell probabilities are nevertheless
well-defined, and can converge to the constant-depth Gaussian as `v→0`.
This is a singular limit of moments, not a failure of native-cell masses.

For a *local asymptotic* description, restrict depth to
`|ζ|≤κz₀`, for any fixed `0<κ<1`, and let `ε²=v/z₀²→0`. This mathematical
restriction is only for explaining the expansion; it is **not** a proposed
model truncation. Its omitted Gaussian probability is exponentially small
in `1/ε²`, while the untrimmed moments still diverge. On this restricted law,
the added-noise variance has expansion

\[
h^2(z_0^2+v)E[(z_0+\zeta)^{-2}]
 =h^2[1+4\epsilon^2+O(\epsilon^4)].
\]

Relative to fixed screen blur, the leading local extra term is
`4h²v/z₀²`: one part comes from using `E[Z²]` in the numerator noise, and
three from the random denominator. The local geometric mean displacement
starts at `−k/z₀²`, with next term `−3kv/z₀⁴`. Its order depends on the
cross covariance: it is second order when all centered 3D perturbations
scale together, but small depth variance alone allows `k=O(√v)`.
These are neither exact global moments nor finite-FOV-conditioned moments.

## An exact center counterexample, including actual GS

Let `z₀>0`, `v>0`, `α>0`, and define a rank-one Gaussian by

\[
Z\sim N(z_0,v),\qquad X=\alpha(Z-z_0),\qquad Y=0.
\]

Its mean is already the anchor `a=(0,0,z₀)`, so the 3D anchor correction is
the identity. Let `t²=h²(z₀²+v)` be the RMS numerator-noise variance. The raw
screen density is, up to a positive constant,

\[
g(x,y)=\int_0^\infty z^2
 \exp\!\left[-\frac{(zx-\alpha(z-z_0))^2+(zy)^2}{2t^2}
             -\frac{(z-z_0)^2}{2v}\right]dz.
\]

At the projected anchor, define `w=(v⁻¹+α²/t²)⁻¹>0`. Differentiation and one
integration by parts give

\[
\partial_x g(0,0)\ \propto\
\frac{\alpha}{t^2}\int_0^\infty z^3(z-z_0)e^{-(z-z_0)^2/(2w)}dz
=\frac{3\alpha w}{t^2}\int_0^\infty z^2e^{-(z-z_0)^2/(2w)}dz>0.
\]

Therefore the anchor is not even a stationary point of this smooth density.
Positive-depth normalization or a constant area-floor density does not fix
that derivative. This does not predict failure on the current native grid,
but disproves a general center-mode guarantee, including the actual-GS case
`a=m`. Its mode shift and the local mean shift can even have opposite signs;
using one to claim the other is unsafe.

Another limitation is radial depth mismatch. When both the support and
anchor lie on one camera ray, their unsmoothed screen positions coincide.
Nevertheless, the rank-one term can increase `v` by `(mz−z₀)²`, changing
front probability and the RMS screen scale mixture although the screen
anchor has not moved. The earlier zero-mean radial Cauchy counterexample
therefore remains relevant. A fixed-mean cross-entropy optimum in 3D is not
an optimum for finite-FOV screen KL or the old region labels.

## Consequence for the next scientific decision

The construction remains compact: one legal predicted center, one second
moment and the already defined RMS projection. Its constant-common-depth
limit provides a useful consistency check. It is not a mechanism that
jointly enforces regional quality and center ranking. Any later selection
must treat the anchor's additional depth information and the resulting
depth/tail changes as part of the hypothesis, and must test the same output
against both old center and region requirements. R20's successful center
guards cannot be inherited from the limit. Actual GS gains also cannot be
attributed to an anchor correction that is the identity when `a=m`.
