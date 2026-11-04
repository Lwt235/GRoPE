# R26: an exact-arithmetic quadrature bound

2026-10-31. Independent mathematical derivation for the proposed RMS-depth law. No formal fixture, real input, label, or efficacy calculation was executed. The numerical constants below are elementary arithmetic on the bound. This document does not certify the floating-point density evaluator.

Let the original homogeneous Gaussian be \((U,Z)\), and let \(s^2=E Z^2>0\). Add independent \(\eta\sim N_2(0,h^2s^2I)\) to U before strict-positive-depth projection. Conditional on the original U and Z=z>0, the screen density is a translated isotropic Gaussian of standard deviation \(hs/z\). The unconditioned density g is the mixture of these densities, retaining the indicator Z>0. If s=0, the raw measure is identically zero and the bounds hold trivially.

For a multi-index of total order k, the conditional Gaussian derivative is bounded by a constant times \((z/(hs))^{k+2}\). All moments of Z/s exist, so dominated differentiation justifies these spatial derivatives even though z has no finite upper bound. The mean and variance of W=Z/s satisfy \(m_W^2+\sigma_W^2=1\).

For the even one-coordinate derivative of order 2n, Fourier inversion of the standard two-dimensional Gaussian gives

\[
\sup_y|\partial_x^{2n}\phi_2(y)|
\le (2\pi)^{-2}\int_{\mathbb R^2}|\omega_x|^{2n}e^{-\|\omega\|^2/2}\,d\omega
=\frac{(2n-1)!!}{2\pi}.
\]

Also \(E|W|^{2n+2}\le(2n+1)!!\). One way to see this is to expand the even moment of m+σG, with G standard normal. Replacing the deterministic m by an independent mG′ preserves the second moment and increases every nonzero even-power term: the extra Gaussian even moments are at least one. The resulting mG′+σG is standard normal. Consequently,

\[
M_{2n}:=\sup_y|\partial_x^{2n}g(y)|
\le\frac{(2n-1)!!(2n+1)!!}{2\pi h^{2n+2}},
\]

with the same bound for the y derivative. In particular, \(g(y)\le1/(2\pi h^2)\). These are bounds on the unconditioned raw density, not on its finite-FOV normalization.

For an n-point Gauss–Legendre rule over an interval of length ℓ, the exact-arithmetic integration remainder obeys

\[
|I-Q_n|\le A_n\ell^{2n+1}\sup|f^{(2n)}|,
\qquad A_n=\frac{(n!)^4}{(2n+1)((2n)!)^3}.
\]

On a rectangular tile with sides ℓx,ℓy, decompose the tensor error as
\((I_x-Q_x)I_y+Q_x(I_y-Q_y)\). Gauss–Legendre weights are positive and sum to the interval length, so no mixed derivative is needed:

\[
|I_{xy}-Q_xQ_y|\le A_n M_{2n}\,\mathrm{area}\,(\ell_x^{2n}+\ell_y^{2n}).
\]

If both sides are at most L, the factorials cancel to give the simple bound

\[
\boxed{|I_{xy}-Q_xQ_y|\le
\frac{\mathrm{area}}{\pi h^2\binom{2n}{n}}
\left(\frac{L}{2h}\right)^{2n}.}
\]

Adding tile bounds gives the same formula with total area. For separately tiled native cells, it bounds the sum of absolute cell-mass errors. It also bounds the absolute error of a separate integral over their enclosing rectangle. Their numerical difference is bounded by the sum of the two bounds. Native cells must actually tile that rectangle; overlaps or omitted regions invalidate the comparison.

For GL24, \(\binom{48}{24}=32,247,603,683,100\). At h=.25 and total area432:

| Maximum tile side | Bound on total absolute raw cell error |
| --- | ---: |
| 4h | 19204.199143659105 |
| 2h | 6.822702098808657e−11 |
| h | 2.4239107072818402e−25 |

The 4h bound is uninformative; it does not prove that a particular 4h calculation is inaccurate. The 2h bound is small in absolute terms, but by itself only guarantees raw relative error1e−6 if the true FOV mass is at least about6.83e−5. Very small FOV masses still require independent relative-error evidence. Reducing the tile side to h does not remove this logical limitation.

The bound assumes exact input parameters, exact RMS normalization, exact density evaluations, exact quadrature nodes/weights and exact arithmetic. It excludes QR/linear-solve rounding, erfcx cancellation, negative-tail series evaluation rounding, underflow/overflow, node/weight approximation and summation error. A negative-tail series remainder bound covers only that mathematical series truncation; it is not a replacement for these missing error terms. Neither bound certifies normalized probabilities after floor/content, nor gives a training/backpropagation cost guarantee.

The proposed density formulas are consistent with this law. In normalized conditional coordinates, set a=(u1,u2,m), b=(v1,v2,1). Then

\[
\delta=\frac{\|a\times b\|^2}{\sigma^2D}
=\frac{\|B^{-1}(my-\mu_{xy})\|^2+
[(\sigma\det(y,\mu_{xy})+\det(\beta,my-\mu_{xy}))/\det B]^2}{D}.
\]

The positive-t formula follows by multiplying J₂(t) by σ² before evaluation; the negative-t formula uses J₂(−q)=φ(q)K(q). Density denominators require positive-diagonal Cholesky/QR-normalized B and L, or absolute determinants for arbitrary factors. Screen-unit equivariance with scalar h applies to uniform scaling and orthogonal screen rotations; a general non-isometric affine transformation requires transforming the smoothing covariance as a matrix.

The radial counterexample remains: for X=Zr with zero-mean Gaussian depth, this new law gives a half-mass two-dimensional Cauchy kernel of scale h. The original fixed screen blur gives a half-mass Gaussian kernel. The quadrature bound neither removes that modeling difference nor establishes regional, center, V, GS or confirmation performance.
