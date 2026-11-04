# Round 5: Gaussian rotary operator and baseline audit

Audit date: 2026-10-30. No model or training code was changed by this audit. The current `normalized_rotary` is exactly Gaussian expected rotary encoding followed by content-wise L2 normalization and restoration of the input norm, on the ordinary nonzero domain. After the current matcher's preliminary Q/K normalization, it is simply **post-normalized expected Gaussian RoPE**. A separately named implementation of that composition is a duplicate baseline, not another method to train against.

The Gaussian integration itself is established prior art. The present operator alone does not establish a new 3DGS-to-PE contribution. Any defensible contribution must identify a useful input-support representation/estimator, a carefully delimited normalization intervention, or a demonstrated downstream benefit, while acknowledging the existing constructions.

## Primary sources and the official-code limit

The inspected primary sources are:

- [mip-NeRF, v3, §3.1, Eqs.10–16](https://arxiv.org/html/2103.13415v3): Gaussian integration attenuates Fourier sine/cosine features by directional variance. Its IPE is a coordinate feature supplied to a radiance-field MLP, rather than a Q/K attention implementation.
- [Google's mip-NeRF implementation](https://github.com/google/mipnerf/blob/84c969e0a623edd183b75693aed72a7e7c22902d/internal/mip.py): `expected_sin` and `integrated_pos_enc` implement that expectation. The latter constructs the frequency-basis projected variance, including a full-covariance path. Inspected commit: `84c969e0a623edd183b75693aed72a7e7c22902d`.
- [WorldRoPE, v1, Eqs.8–10; Appendix B.1; Appendix A.7](https://arxiv.org/html/2610.16337v1): the paper specifies Gaussian/mixture expected rotary blocks, deterministic image-coordinate queries, and a probabilistic-query extension. It also specifies transformed values and inverse-rotated outputs. Its supports come from layer-wise depth hypotheses, depth spread, and patch footprint.
- The [official WorldRoPE repository](https://github.com/mjmjeong/WorldRoPE/tree/80014af658fc0eaee9b0605a363aade2ccdd7516) contains only its README at inspected commit `80014af658fc0eaee9b0605a363aade2ccdd7516`; the [README](https://github.com/mjmjeong/WorldRoPE/blob/80014af658fc0eaee9b0605a363aade2ccdd7516/README.md) states that implementation and checkpoints are unreleased. It is linked by the [authors' project page](https://cv.snu.ac.kr/research/WorldRoPE/). Consequently this audit cannot verify hidden implementation choices or claim an official-code reproduction.
- [QKNorm, Henry et al., 2020](https://arxiv.org/abs/2010.04245): per-head Q/K L2 normalization and a learned attention scale are existing ingredients. That paper does not establish the behavior of normalization around a Gaussian rotary expectation; its placement must be specified explicitly here.

WorldRoPE's displayed Eq.10 uses projected queries/keys divided by `sqrt(d)` without an explicit post-expectation L2 normalization. This is a statement about the paper formula, not a claim that an unreleased implementation has been inspected for every possible normalization. No reported performance of the local two-block matcher should be labeled full WorldRoPE performance.

## Exact operator identity

Write a real content vector `h ∈ R^(2F)` as complex blocks `z_f=h_(2f)+i h_(2f+1)`. For `X~N(m,C)`, its Gaussian characteristic function gives

\[
\chi_f(m,C)=\mathbb E[e^{i\omega_f^T X}]
 =\exp(-\tfrac12\omega_f^TC\omega_f)e^{i\omega_f^Tm}.
\]

Let `A_G=E[R(X)]`. Multiplication of each content block by `χ_f` is precisely `A_G h`. In real coordinates,

\[
A_G=\operatorname{blockdiag}_f\{a_fR(\theta_f)\},
\quad a_f=e^{-\omega_f^TC\omega_f/2},\quad\theta_f=\omega_f^Tm.
\]

This uses the same sine/cosine expectations as IPE. Packaging those coefficients as content rotations changes how they enter a network; it does not introduce a different Gaussian integral.

For `N(x)=x/||x||`, the current implementation is

\[
T_G(h)=\|h\|N(A_Gh).
\]

The repository's stabilized log-domain calculation evaluates this expression without first materializing small amplitudes. The current `GaussianPEMatcher` first sets `c=N(h)`, so

\[
T_G(N(h))=N(A_Gh).
\]

Therefore a baseline implemented as `post_L2(expected_Gaussian_RoPE(raw_Q))` gives the same Q as `normalized_rotary(pre_L2(raw_Q), ...)`, under the same geometry, frequency basis and normalization domain. This holds separately for K, so it preserves the entire candidate score matrix under the same temperature. It also gives the same attention output if V, masks, softmax, output projection and all other processing are identical.

This identity covers arbitrary finite SPD covariance in the ordinary domain, not only diagonal covariance, axis-aligned frequencies or deterministic queries. The single-Gaussian covariance parameterization `R diag(s²) Rᵀ` is one way to supply `C`; the operator sees `C`, not an additional rotation/scale mechanism.

## What normalization placement changes

Pre-normalization gives `A_G N(h)`, whose norm generally depends on content and covariance. Post-normalization gives `N(A_G h)`, whose norm is one. For deterministic RoPE the two commute because rotations are orthogonal; an expected rotary matrix generally contracts frequency blocks differently, so they do not commute.

For example, equal-norm frequencies and isotropic covariance yield `A_G=aR(m)` for each token. Pre-normalization retains the token's scalar attenuation `a`, whereas post-normalization cancels it completely. With unequal frequency norms or anisotropic covariance, relative block amplitudes remain after post-normalization. This is frequency redistribution, not a guarantee of calibrated uncertainty or correct matching.

At fixed contents and geometry, the normalized score is

\[
s_{ij}=\tau\,
\frac{h_i^T A_i^TA_jh_j}
{\|A_ih_i\|\,\|A_jh_j\|}.
\]

A scalar global temperature cannot generally undo the per-token denominators. Thus expected Gaussian RoPE with only pre-QK normalization is a meaningful distinct control. A test with two keys of different isotropic covariance explicitly changes its attention probabilities, while postnorm makes their otherwise equal contents/means score equally.

`normalized_rotary(raw_Q)` is different again: it restores `||raw_Q||`. In a new Transformer, either normalize projected Q/K per head before invoking it, as the old matcher does, or make the norm-restoration convention explicit in every control. Normalizing across a flattened concatenation of all heads is not the same as per-head QKNorm. Use the same learned scale and initial scale convention for the matched operator ablations; otherwise `/sqrt(d)` versus learned-temperature changes become an additional factor.

The exact mathematical map is nonlinear in content and is not an orthogonal matrix. It is also not `E[N(R(X)h)]`: every sampled deterministic rotation preserves the content norm, so placing normalization inside that expectation recovers the unnormalized expectation on unit content, rather than normalization of the expectation.

## Deterministic versus probabilistic queries

Taking `Cq=0` and the query's native pixel center gives a deterministic query rotary map. Taking `Cq=I/12+δ²I` gives a fixed finite-support query, not a deterministic one. A fitted query covariance and a fitted query mean introduce further differences. In particular, all methods using a fitted query mean may benefit from geometry even if their covariance is fixed.

For independent Gaussian query/key locations expressed in one frame, direct algebra yields

\[
(\mathbb E R(X_q))^T(\mathbb E R(X_k))
 =\mathbb E R(X_k-X_q),
\]

with relative mean `mk−mq` and covariance `Cq+Ck`. This is also explicitly treated by WorldRoPE's probabilistic-query extension. Post-normalization adds the separate content-dependent denominators above; they cannot generally be recovered from summed covariance alone. If query/key location errors are correlated, their relative covariance instead includes the cross-covariance terms, so independent marginal expectations should not be advertised as a correlated joint posterior calculation.

The optional round-4 fixed-query checkpoint conversion can absorb a constant filter into the last affine encoder layer, as derived in [the relative-support note](round4_relative_support.md). That does not make arbitrary token-varying query support equivalent to a deterministic query model, or prove that dropping fitted query geometry is harmless.

## Gaussian mixtures and covariance estimation remain real differences

For a mixture, the exact complex coefficient is

\[
\chi_f=\sum_m\alpha_m
 e^{-\omega_f^TC_m\omega_f/2}e^{i\omega_f^Tm_m}.
\]

This is still a compact scaled rotation per frequency block, but its phase need not be `ωfᵀ` times one shared mean. Matching the mixture's first two moments to one Gaussian generally changes its characteristic function. A symmetric pair centered at `±π` with frequency `1` has negative expected cosine, while its zero-mean moment Gaussian has positive expected cosine. Postnormalizing does not fix that difference. Thus the current one-Gaussian moment representation should not be claimed to reproduce the multiple-hypothesis operator.

The meaningful representation differences are:

| Item | Current regional Gaussian-support experiments | What remains to establish in round 5 |
|---|---|---|
| Source of geometry | Available scene reconstruction and selected GS primitives | Support estimated solely from declared input views; no target/test reconstruction leakage |
| Per-token distribution | One moment Gaussian after region aggregation | Whether one Gaussian is adequate for real multi-surface regions |
| Covariance content | Center scatter plus mean primitive covariance, with a common floor | Incremental value of primitive internal R/S beyond center scatter, after retraining |
| Meaning of covariance | Effective spatial support | Whether any estimator also predicts calibrated location error; this is not automatic |
| Query geometry | Native or fitted means/covariance, depending on the adapter | Explicit shared policy; query covariance ablations must not silently change its mean |
| Geometry evolution | Frozen input geometry in the matcher | Whether a downstream architecture needs or benefits from layer-wise refinement |

These are differences in data, representation and model integration. None converts the same `N(A_Gh)` formula into a new independent operator. The earlier primitive-covariance deletion result does not currently support a strong claim that optimized individual GS R/S information is necessary.

## Minimal nonduplicated baseline set

For the first decisive two-block cross-attention experiment, four trained conditions suffice to separate the immediate questions:

| Condition | Q/K map after common per-head pre-L2 | Geometry policy | Main comparison |
|---|---|---|---|
| Projected point | `R(m)c` | Same estimated means; zero covariance | Is support useful beyond projected positions? |
| Expected Gaussian | `A_G c` | Same full covariance and floor as the candidate | Does post-normalization help? |
| Normalized Gaussian | `N(A_G c)` | Same full covariance and floor | Current candidate; also the expected-plus-postnorm baseline |
| Normalized center-only | `N(A_center c)` | Same means, weights/masks and floor; remove primitive internal covariance | Is any gain specific to the supplied primitive shape? |

All conditions share features, Q/K/V projections, head count and dimensions, frequency vectors/coordinate units, attention masks, temperature, residual/FFN structure, output/matching heads, optimizer budget, seed pairing and checkpoint selection. V/output positional processing must be identical for these controlled comparisons. Adding V and an output head makes this a downstream attention experiment; it does not reproduce WorldRoPE's value/output transformations or full task architecture.

If the new input representation has no separate primitive internal covariance, center-only is identical to its full covariance and must not be trained twice. Substitute a same-trace isotropic covariance control to test anisotropy, and describe the experiment as testing that input representation rather than pre-existing optimized GS R/S.

A literal deterministic-query/single-Gaussian-key Eq.10-inspired condition is optional if that query policy is itself under study. Label it as an adapted paper-formula control, since keeping its raw Q/K and `/sqrt(d)` convention would differ from the matched pre-QKNorm ablation. A separate multiple-hypothesis predictor or full WorldRoPE comparison needs a separately specified model; neither is required to pretend that the algebraically identical postnorm baseline is distinct.

Do not add separate training runs for “current normalized Gaussian”, “expected Gaussian plus post QKNorm”, and “mip-style attenuation plus the same rotary/postnorm”: with identical inputs they are aliases. Verify them numerically and list the identity. Conversely, changing means, covariance estimator, mixture representation, normalization domain or V/output policy is a distinct experiment and must be named as such.

## Independent numerical checks and limits

`tests/test_round5_operator_identity.py` contains 13 passing CPU tests. Its reference uses complex Gaussian characteristic functions and does not call the repository's `rotary` or `expected_phase` helpers. It checks ordinary float32/float64 output identities, the norm-restoration distinction, content/mean/covariance gradients, the summed-covariance score identity, pre/postnorm non-equivalence, finite versus deterministic query support, mixture moment mismatch and naive exponential underflow.

The identity with a literal `F.normalize` composition assumes norms outside its epsilon branch and representable intermediate expectations. `F.normalize` maps extremely small nonzero vectors to norms below one, while the stabilized implementation can retain the symbolic normalized direction even when a naive exponential underflows. These numerical-domain qualifications do not establish a different mathematical operator. Use the same stable implementation when treating postnormalized expectation as the duplicate control; do not manufacture a numerical baseline failure by computing an underflowed numerator first.

Audited local source SHA256 values:

```text
models/gaussian_pe.py  1fb7bbfbc1631abac28fcff4e304f2eea521f578229e29d603dfd98edd9104d4
models/geometry.py     173dbb8a58c00a41fabe5918cf4cc5422809e829aa492a8601747dbc4c4f8e20
```

The audit establishes an implementation identity and a fair baseline boundary. It is not an exhaustive novelty search or proof that no useful application-specific contribution remains. Current scientific priority is independent downstream evidence with input-only geometry and honest primitive-shape attribution, rather than renaming the same expected-and-normalized rotary map.
