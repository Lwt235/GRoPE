# A compact Gaussian positional map under confirmation

The frozen fresh-camera confirmation and the direct dense-support checks now
support this compact map for conditional regional matching with available
reconstructed geometry. See [the third-round report](../round3_report.md).
Its mathematical ingredients are Gaussian characteristic functions, rotary
position encoding and per-token Q/K normalization. We do not claim new
Gaussian integration mathematics or an official reproduction of
WorldRoPE/HRoPE.

The relevant prior ingredients are [mip-NeRF's integrated positional
encoding](https://jonbarron.info/mipnerf/), [WorldRoPE's probabilistic rotary
integration](https://arxiv.org/html/2610.16337v1), and [Query-Key
Normalization](https://arxiv.org/abs/2010.04245). The current research claim
must come from the support semantics, explicit normalization placement and
controlled evidence, rather than presenting these ingredients as inventions.

## From 3DGS parameters to a position distribution

Write a Gaussian as

\[
G=(\mu,R,s),\qquad \Sigma=R\operatorname{diag}(s^2)R^\top.
\]

Its center determines phase; its rotation and scales determine directional
frequency filtering. Project a reconstructed support into the query camera:
\(m=\pi(T\mu)\), \(C=J R_c\Sigma R_c^\top J^\top\).
The implementation uses patch units and a common, development-fixed
resolution covariance \(\gamma^2 I\), with \(\gamma=0.25\).

The current regional experiment aggregates visible GS primitives before
projection. The moment covariance is
\(\sum_j w_j[\Sigma_j+(\mu_j-\bar\mu)(\mu_j-\bar\mu)^\top]\).
This is a Gaussian parameterization of region support. It must not be
described as evidence that the optimized *individual* GS rotation/scale is
useful: the earlier decomposition found that center dispersion accounts for
the exact-overlap benefit, and adding primitive internal covariance hurts
that kernel on all four scenes.

## One map, unchanged head dimension

For a 48-dimensional content vector, combine adjacent coordinates into 24
complex values \(z_\ell\). Use fixed oblique frequencies \(\omega_\ell\).
For \(X\sim\mathcal N(m,C)\),

\[
\mathbb E[e^{i\omega_\ell^\top X}]
=e^{-\frac12\omega_\ell^\top C\omega_\ell}
 e^{i\omega_\ell^\top m}.
\]

The candidate is simply the normalized expected rotary map:

\[
\boxed{\quad
\operatorname{PE}_G(z)_\ell
=\|z\|_2\,
\frac{e^{-\frac12\omega_\ell^\top C\omega_\ell}
e^{i\omega_\ell^\top m}z_\ell}
{\sqrt{\sum_r e^{-\omega_r^\top C\omega_r}|z_r|^2}}
\quad}
\]

Use the ordinary Q/K dot product afterwards. The encoder, learned scalar
temperature and matching head are shared with every control. This map adds
no learned parameters, extra channels, pairwise shape bias or geometric
gate. Broad supports favor low frequencies without shrinking the whole
token norm. Normalizing only before expected rotation does not have this
property.

## Exact scope of the mathematical claims

- Nonzero content retains its norm; zero content maps to zero.
- At zero covariance and zero resolution blur, the map is ordinary RoPE.
- A common translation of both projected means leaves their dot product
  unchanged, because normalization does not depend on the means.
- The map is nonlinear in content. It is not an orthogonal or unitary
  operator and does not preserve the inner product of arbitrary contents
  sharing the same Gaussian.
- The numerator uses summed covariance, but its separate content-dependent
  normalization denominators prevent replacing it by an arbitrary function
  of \(C_q+C_k\) alone.
- Support size describes spatial extent. It is not automatically calibrated
  uncertainty about an estimated center. No robustness to erroneous geometry
  follows from this formula.
- Projection makes key encodings query-camera dependent. A single encoded
  key cannot be reused unchanged across different query-camera frames.

## Numerical implementation

`models/gaussian_pe.py:normalized_rotary` uses float64 intermediate values and
content-aware logarithmic scaling. This avoids underflow when the least
attenuated frequency happens to have zero content. Independent tests compare
ordinary values and gradients against direct float64 expected rotation and
normalization, as well as sparse-content and large-covariance regressions.
Pathological derivatives such as \(e^{1500}\) exceed float64 representation;
the implementation does not promise finite gradients for every finite input.
Training checks gradients for finiteness rather than silently clipping NaNs.

## What the fixed confirmation tests

The protocol is `results/round3_normalized_confirmation/frozen_protocol.json`.
Five seeds fit only Teatime train photographs; Teatime validation CE selects
checkpoints. Raw RGB and a shared frozen ResNet18 representation are both
predeclared. The previously unexported camera pairs are not used for tuning.
The fixed/trace/area/shuffle interventions change **key** covariance; fitted
query geometry remains common. Projected-center RBF receives the same query
and key means. Results must distinguish general support, anisotropy and
overall matching quality. Existing sparse SfM labels support conditional
regional correspondence, not dense visibility, rejection or novel-view
reconstruction claims.
