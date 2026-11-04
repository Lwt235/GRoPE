# R18 conditional normalization: synthetic falsification only

2026-10-31, written after the accepted R17 scientific failure and before this
prototype's numerical results. This is a new development hypothesis, not a
continuation or repair of R17. No real TRAIN/VAL/GS/fresh input or labels are
authorized by this protocol. Historical failures and all scientific gates stay.

## One selected law

For X~N(mu,C), native disjoint rectangles B_j and bounded content c_j, define

    M_j(X) = integral_Bj N(y;X,h²I)dy + epsilon |B_j|
    A(X) = sum_j M_j(X),  Z_T(X) = sum_j M_j(X) exp(T c_j)
    pi_0(dX) = N(mu,C)(dX) A(X) / E[A]
    P_j = E_pi0[M_j(X) exp(T c_j) / Z_T(X)].

C denotes spatial support, not uncertainty. The geometry-only conditioning on
the key field of view is part of the selected law. It uses neither sensor
visibility nor truth nor a point winner, and discards no query. The alternative
unconditioned E_N[softmax] is a diagnostic control, not an eligible fallback.
At T=0 the A/Z cancellation recovers the convolved full Gaussian cell masses
plus one original area floor, normalized on the same FOV. At C=0 it is the
ordinary point operator. Fixed content on a partition can be merged by summing
probabilities; re-pooling content is a different operation.

In the whole plane without floor and with continuous linear content g^T y,
the output is N(mu+T h²g,C+h²I). The old normalization after marginalization
instead shifts by T(C+h²I)g. This is a restricted analytic mechanism claim,
not a native argmax, PCK, robustness or real-task guarantee. R10 already
documented conditional averaging versus unnormalized score averaging; the
normalization principle is not a new discovery. The current latent Gaussian,
geometry conditioning and integration implementation require their own checks.

## Frozen numerical budget and fixtures

All arithmetic in this synthetic prototype is float64. Point/full limiting
identities refer to this arithmetic and the continuous laws; they do not claim
bitwise reproduction of historical float32 content products. The production
approximation is a tensor 64-by-64 standard-normal Gauss-Hermite rule; order96
is reported as a diagnostic and cannot replace order64 after seeing results.
The inner isotropic rectangle masses use separable normal-CDF differences,
mirroring positive intervals to avoid right-tail subtraction of rounded ones.
There is no GL32 inner full-covariance integration or extra key covariance.

Independent reference code uses tensor Gauss-Legendre orders384 and512 on
[-14,14]^2 in standard-normal coordinates, own erfc interval arithmetic and
independent implementation. Every case reports raw integrals, E[A], reference
convergence and amplified normal truncation bounds. The numerical-denominator
estimate is descriptive. The checked bound instead uses the analytic lower
bound `E[A 1_inside] >= total_floor*(1-tail)` and is
`2*tail/[total_floor*(1-tail)]` for maximum absolute component error; this
certifies truncation alone, not quadrature error. A second independent
one-dimensional adaptive integral computes convolved Gaussian rectangle mass
at T0. Same SciPy elementary functions are disclosed shared dependencies;
reference formulas, quadrature nodes and accumulation code are independent.

The runner source is the exact fixture specification. It contains eleven fixed
Gaussians on a 5-by-4 unit-cell FOV: zero covariance, nominal8/16/24 covariance,
rotated anisotropy, rank1, wide support, boundary, outside, tiny-visible and
floor-dominated cases. Each is tested at T0 and its fixed nonzero temperature,
for 22 probability rows. Three further fixed nonzero-content rows contain a
sharp peak, repeated peaks, and a deliberately informative content-correction
location, giving 25 rows in total, all retained. The last case descriptively
reports distance of the probability mean from its declared synthetic intended
location for both orders of normalization; this is not a selection gate.
Base content is a deterministic bounded trigonometric function of cell centers;
the three alternatives are fully specified in source, with no random selection.
Additional fixed tests cover constant content, key permutation, translation,
joint unit change (including inverse-square floor density), fixed-content
fine/coarse key merging, and whole-plane linear-content mean/covariance.
Invalid nonfinite/asymmetric/negative-covariance input is rejected, never
silently repaired. Rank-deficient PSD inputs are allowed.

## Decisions, reporting and chronology

Before executing the synthetic tests or cases, freeze this protocol, model,
runner, independent reference, mathematical reviews and their source snapshots;
save exact fixture JSON and its hash plus Python/NumPy/SciPy environment.
Execution fixes OMP/MKL/OpenBLAS thread variables to one and records numerical
build hashes; synthetic timings are descriptive, not paired scientific costs.
Sources and fixture hashes are checked before and after execution. Save all
probabilities, raw integrals, timings and failures outside previous results.
There is no deletion, automatic retry, adaptive case replacement, or numerical
order promotion. Engineering failure requires a separately documented repair.

Every case must meet: finite nonnegative P, row normalization error<=2e-12;
production64 vs reference512 max absolute P error<=1e-4; reference384 vs512
max absolute P error<=1e-8 and relative E[A] error<=1e-8; conservative truncated
tail bound<=1e-8. At T0, reference vs independent convolved mass P<=1e-8 and
production vs that mass<=1e-4. All fixed identities have absolute error<=2e-12
except the independently integrated T0 checks just specified. Order96 error
is descriptive only. Report every failed numerical condition and raw value.

Passing means only that this fixed integration budget passed these synthetic
checks. It does not prove general quadrature convergence, numerical correctness
on real supports, a PE cost advantage or any scientific gate. A numerical
failure stops order64 from entering TRAIN under this protocol. Subsequent
mathematical/numerical revision must be separately justified and frozen, not
declared a retrospective pass. If all checks pass, real old-TRAIN feasibility,
same-information point/fixed/isotropic/conditional-quadrature controls and full
scientific validation need a new frozen protocol before execution. Actual V,
optimized GS and untouched confirmation remain mandatory and unstarted.

The projected-source-center alternative remains unselected and unexecuted.
