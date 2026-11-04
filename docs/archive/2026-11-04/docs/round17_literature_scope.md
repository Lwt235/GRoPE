# R17 literature and novelty boundary

Checked official sources on 2026-10-30, before new R17 experimental effects.
This is a focused prior-art check, not an exhaustive novelty search.

Quadratic Gaussian positional attention already has explicit precedent in
[Cordonnier, Loukas and Jaggi, ICLR 2020](https://arxiv.org/abs/1911.03584v2)
and its [authors' implementation](https://github.com/epfml/attention-cnn).
The earlier five-dimensional Gaussian score in this repository must not be
described as a new invention merely because its covariance is obtained from
3D support. R17 keeps the existing native Gaussian integral rather than
claiming a new exact finite-dimensional factorization of its cell log mass.

[Bayesian Attention Mechanism, v4 / ICLR 2026](https://arxiv.org/abs/2505.22842v4)
already frames positional encodings as probabilistic priors and uses a
generalized Gaussian prior. Thus adding a log positional prior to content,
or changing a Gaussian radial exponent, is not by itself a novelty claim.
Its language-model context experiments are not evidence for our geometric
regional correspondence, actual-value depth or GS tasks.

[Calibration Attention, v2](https://arxiv.org/html/2508.08547v2)
predicts an instance temperature from the CLS representation and trains it
through a calibration objective. It is relevant precedent for conditioning
temperature on representation, but it is not the same operation as our
fixed per-query support-trace multiplier. In particular, its classification
reliability motivation does not make our support covariance a confidence or
noise estimate. The paper remains marked UnderReview on its arXiv abstract.

[Linear Log-Normal Attention, v4](https://arxiv.org/abs/2311.13541v4)
studies attention concentration and distribution. This is related context
for separating attention sharpness from task accuracy; we do not infer our
specific trace formula, its validity, or an experimental gain from that work.

Our current hypothesis is deliberately narrower: under an existing Gaussian
support bias, set tau=T h²/(h²+tr C) to bound a continuous content-induced
mode response while preserving the point limit. The direct numerator and
denominator derive from the response inequality in the accompanying theory
note; they were selected before R17 results, not by searching these papers'
reported scores. Whether this adjustment improves discrete native attention
remains an experimental question. We make no claim of being first to use
adaptive temperatures, Gaussian PE, probabilistic PE, or geometry-conditioned
attention.
