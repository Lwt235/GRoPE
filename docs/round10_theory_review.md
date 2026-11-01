# Round 10: independent support/partition and integration review

2026-10-30. This review uses public prior work, frozen Round 6–9 reports and
source code, and synthetic operator inputs. It does not read sealed scene
inputs, labels, predictions, or new confirmation results. Passing the tests
below verifies an integration implementation; it is not evidence of task
improvement.

## Research question and limits

The defensible question is whether a Gaussian is an economical summary of the
actual spatial support represented by a token. Covariance is not automatically
uncertainty, and optimized rendering-splat covariance is not automatically
token support. Round 6 supports a scoped regional correspondence benefit;
Rounds 7–9 do not establish downstream usefulness or an independent value of
optimized primitive R/S. A partition probe can test a missing core premise,
but cannot make the earlier downstream gates pass by definition.

Gaussian region integration and positional features are established by
[mip-NeRF](https://arxiv.org/html/2103.13415v3). Gaussian token parameterization
is established by [GPSToken](https://arxiv.org/html/2509.01109), and learned
multi-view Gaussian supports plus integrated rotary embeddings are particularly
close in [WorldRoPE](https://arxiv.org/html/2610.16337v1). A new network predicting
the same parameters would therefore need a more specific contribution.
[Continuum Attention](https://www.jmlr.org/papers/v26/24-0879.html) explicitly
treats implemented attention as a discretization of an integral operator.
Neither Gaussian integration nor mass-weighted attention is a new theorem here.

## Primary contract recommendations

1. Use fixed, old development observations. Real partition changes recompute
   support, mass, content and values from that same observation. Copying a
   Gaussian and splitting its mass is only an algebra check.
2. Hold query partition fixed for the first key-partition test. The existing
   region truth conditions on visible source mass. Changing query partitions
   changes those conditional populations: averaging children by native area
   does not equal averaging by visible mass. Ground-truth visible masses must
   not become forward weights.
3. Use independently projected sensor-pixel counts as task truth. A second
   dense projection using the same predicted geometry is useful only to
   separate support approximation error from upstream geometry error.
4. Compare methods within the same partition and valid-query population.
   Raw KLs over different category spaces are not a partition-invariance
   statistic. Aggregate to a common declared coarse partition, or lift to a
   common fine field, before testing consistency. Report the information loss
   of coarse token values rather than attributing all of it to PE.
5. Strong controls must retain identical inputs. Multipoint representatives
   come from the same predicted geometry, never sensor truth. Include all
   extraction, projection, covariance formation and integration costs when
   asserting an efficiency advantage. Query-sampling and Gaussian-summary
   controls should use the same key integration rule where the goal is to
   isolate query approximation.
6. For mixed key covariances, the full pairwise Gaussian-overlap operator is
   required; the existing five-channel exact lift does not apply. Key mass is
   native cell area, not determinant-inferred projected area. Same-trace
   isotropic and orientation controls preserve mass and means.
7. A geometric partition identity does not imply full attention invariance
   after real merging: pooled content changes `exp(content_score)`, and pooled
   values discard variation. Exact identities require unchanged score/value
   fields and additive cell mass. EWA-equivalent formulas are implementation
   checks, not independent accuracy competitors.

## One lean, conditional iteration

The full overlap approximation replaces a known native key indicator with an
area-scaled Gaussian. This is avoidable when keys are axis-aligned cells:

\[
  w_{qk}=\int_{R_k}\mathcal N(x;\mu_q,C_q+h^2I)\,dx,
  \qquad b_{qk}=\log w_{qk}.
\]

This retains one query Gaussian and no new learned module. The cell mass
already includes area, so adding `log area` again is wrong. Normalizing by the
sum over a rectangular FOV is conditioning on that supplied FOV. With unchanged
content, splitting a key into disjoint children conserves its total weight.
The operation is a known integrated relative position bias; it is not a new
Gaussian kernel, a finite exact Q/K map, or a solution to poor input depth.

This is a justified iteration only if failures are substantially associated
with Gaussian approximation of keys or key partition changes. If the main loss
instead comes from non-Gaussian query support, projection linearization, or
incorrect predicted geometry, exact key integration cannot remove that loss.
Do not infer that adding mixtures will necessarily help.

A simple analytical check shows the issue without task labels. A standard
normal query over `[-4,4]` gives exactly half the mass to `[-4,0]`. Gaussian key
surrogates also give one half with two equal cells. Keeping the left cell and
splitting only the right half into `[0,1],[1,2],[2,4]` changes the normalized
Gaussian-overlap left mass to **0.4695406431**. Exact native-cell integration
retains **0.5**. This demonstrates a discretization defect, not an empirical
performance advantage. Even coarse numerical quadrature can have a similar
defect, so its convergence must be checked rather than assumed.

## Implemented numerical operator

[`models/round10_cell_integral.py`](../models/round10_cell_integral.py)
implements positive conditional Gaussian integration. It avoids subtracting
four nearly equal bivariate CDF corners. Standardized x intervals are split at
normal-density scales and at conditional-CDF transitions; each of the twelve
segments uses 24 Gauss–Legendre nodes. `erfc` differences avoid subtracting
near-one univariate CDF values. The implementation retains autograd through
means, covariance, bandwidth and bounds, and uses float64 in memory-bounded
query/key chunks.

Input rectangles use physical pixel bounds: pixel-index interval `[a,b)` is
`[a-.5,b-.5)` in the original integer-center coordinates. Divide bounds by 8
when means use the existing reference-stride units. Bounds, means, bandwidth
and covariance must all share consistent units.

The operator returns raw masses without a floor, mass multiplier, score
temperature or FOV normalization. A separate normalization helper refuses
all-zero rows unless the caller explicitly supplies a floor. Such a floor
perturbs partition additivity and belongs in the predeclared common numerical
policy, with affected-row counts.

The standardized x integral is truncated to `[-12,12]`, omitting less than
`4e-33` absolute mass. This absolute bound does **not** bound relative error
after conditioning on an extremely unlikely FOV. Very narrow, far-off-image
queries can have an all-zero row. A caller must report and predeclare a fallback
for them; it must not silently discard them or call the method an unrestricted
log-domain tail integrator. Covariances that are numerically singular in
float64 are rejected. No arbitrary ill-conditioning guarantee is claimed.

## Numerical evidence only

`/home/liwt/miniColmap/.venv/bin/python -m pytest -q tests/test_round10_cell_integral.py`
passed **22 tests in 2.63 s**. They cover independent SciPy bivariate Gaussian
rectangle probabilities, correlations through `±0.99999999`, analytical
separable cases, positive and correlated tails, genuine rectangle subdivision,
unit changes, batch/chunk identity, gradients and input rejection.

An additional fixed-seed synthetic check used 128 Gaussian queries, covariance
eigenvalues from `1e-6` to `256`, 24 rectangles with extents 1/2/4/8, bandwidth
0.25, and means over a deliberately larger area than the nominal image:

- 3,072 query/cell masses; maximum absolute discrepancy from SciPy `mvnun`:
  **4.48e-16**.
- Default order 24 versus order 48: maximum absolute discrepancy **2.33e-15**.
- Relative discrepancy against SciPy for reference masses above `1e-12`:
  **4.88e-5**; tiny-probability CDF subtraction in the independent reference is
  itself cancellation-limited, hence the separate analytical tail tests.
- 20 of the 128 deliberately broad-position query rows had zero mass over all
  sampled rectangles. This is not a valid FOV partition benchmark and not a
  task failure rate; it exposes why callers must handle zero mass explicitly.
- One CPU call for 432 queries × 432 cells, four threads, took **1.35 s** after
  a small warmup. This is a diagnostic timing, not a controlled speed claim.

### Faster implementation for the actual Round 10 cell domain

At the parent agent's request, a separate
[`round10_cell_integral_fast.py`](../models/round10_cell_integral_fast.py)
specializes the same operator to native cell side lengths at most 2 reference
units and bandwidth at least 0.25. Adding `h² I` to a PSD covariance bounds the
conditional standard deviation below by `h`; thus a cell spans at most eight
conditional deviations. One physical-coordinate GL32 rule suffices in the
audited domain. It has no global 12σ truncation. Actual floating-point
underflow of extreme masses still requires a declared common numerical policy.
No-grad diagonal covariance uses the exact product of two normal interval
probabilities, ensuring strong point/isotropic controls are implemented
efficiently. When covariance gradients are required, the quadrature path
preserves the off-diagonal derivative even at zero correlation.

The combined generic/fast suite passed **38 tests in 2.56 s**. The forward-only
[`numerical audit`](../results/round10_cell_integral_audit/verification.json)
then loaded six geometry arrays from the lexicographically first old training
pair only, without RGB, labels, sensor files, or validation/test inputs. For
both projected 3D moments and moments of projected pixels, all 3×3 combinations
of g8/g16/mixed partitions were checked at the smallest bandwidth 0.25:

- **1,226,178** query/cell masses; maximum GL32-versus-robust absolute error
  **8.88e-16**.
- Maximum normalized probability error on rows with FOV mass above `1e-30`:
  **6.66e-16**.
- The same check for GL16 raw mass had maximum error **1.24e-14**; the default
  remains GL32 rather than selecting a lower order from task scores.
- **378** independent SciPy comparisons, selected by fixed query ordinals,
  covariance trace extremes and closest cell centers using forward geometry
  alone: maximum absolute error **3.33e-16**.

This is accuracy verification for the integration code, not task evaluation.
The exact file and source hashes, comparison rows and diagnostic CPU timings
are retained in that audit. The audit script refuses to overwrite its output.

## Continuation and stopping

Freeze the task population, primary metrics, controls, calibration budget and
numeric fallbacks before evaluation. A positive continuation signal needs a
material accuracy or measured compute benefit over strong sampling controls,
not only a gain over a center point. A promising development result then needs
fresh-scene confirmation and the earlier integrity/strong-control process.
Historical confirmation scenes cannot become fresh evidence when reused for
iteration. Repeated model selection also needs a declared finite candidate
budget or new locked confirmation evidence.

If exact cell integration is numerically correct but does not improve the
independent task reference over strong controls, its partition identity alone
does not rescue the general PE claim. If neither the Gaussian summary nor its
one justified integration correction passes, retain the scoped Round 6 result
and report that the broader objective remains unachieved. User authorization
to persist does not justify weakening gates or relabeling a failed result.
