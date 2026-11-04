# R24: fixed Student-t4 numerical preflight, synthetic inputs only

2026-10-31. Selected after the accepted R23 real-input applicability failure.
The ray implementation completed 0/72 calls; 22 underflow, 25 inner-budget,
25 shared-deadline failures are retained. Its distribution definition was not
scientifically falsified, but continuing to optimize it is not a prerequisite
for testing a different compact parameter-to-PE family. R24 makes no claim of
empirical superiority, new mathematics or complete research success.

## Hypothesis and exact law

Use one center a and one covariance S=C+h²I, with fixed nu=4 and h=.25.
The Student scale matrix is S/2, not S. For q=(y-a)^T S^-1(y-a),

    p(y) = [pi sqrt(det S)]^-1 (1+q/2)^-3.
    M(B) = integral_B p(y) dy.
    f(B) = 1e-30 area(B).
    P(B) = softmax_B[log(M(B)+f(B)) + b(B)].

The full-plane covariance is S; its center density is twice the same-S
Gaussian's. It has a heavier tail, divergent fourth moment, and no universal
regional advantage. The R15 fixed-t3 reserve and the same-covariance annulus
counterexample in `docs/student_t4_route_comparison_unselected_draft.md`
remain valid limitations. Nu4 is a fixed rational density choice, not a
hyperparameter selected on new results. There is no nu/h/T search.

Adding h²I defines covariance here; it is not convolution with Gaussian
blur. C=0 gives a Student-t4 point kernel. This differs from R22's exact
positive-depth Gaussian pushforward, and neither its probability references
nor its numerical pass can be inherited. There is no new mixture component,
network, learned gate, center repair, winner projection or finite-dimensional
exact Q/K/RoPE representation claim. A future scientific test would require
same-center/same-C Gaussian and t4 point controls plus all old strong controls.
Nothing in R24 authorizes that future test or reading real arrays or labels.

Inputs are finite float64 a[2], exactly symmetric C[2,2], finite positive
rectangles partitioning a rectangular FOV, and finite native b[N]. The fixed
fixtures symmetrize their analytically specified C when creating input bytes.
The model does not symmetrize, clip eigenvalues, add jitter or infer rank.
Allow nominal PSD roundoff only when lambda_min(C) >=
-1e-12*max(1,max(abs(C))); report that residual. S itself must pass strict
Cholesky and is the covariance defining the actual finite-input law. This
admission tolerance does not replace small eigenvalues or change input C.
There is no submass conditioning or positive-depth factor in this new law.

## Frozen synthetic population and production

`experiments/round24_student_contract.py:fixtures()` defines all input bytes.
There are 18 base geometries, each at zero content and at one fixed native
content array, giving 36 cases. Bounds are 16 unit cells over [-2,2]^2 except
two geometries with 432 native unit cells over [0,24]x[0,18]. Geometries include
zero/rank-one C, three patch scales, rotated/thin/large/correlated C, FOV
boundary/outside means, a far mean whose raw mass is below the area floor,
and an anchor second-moment matrix. No case uses observations or labels.

For nonzero content, use native cell centers k and

    similarity(k)=sin(.71*kx+.23)*cos(.37*ky-.19),
    tau=10*h²/(h²+tr C), b=tau*similarity.

This deterministic bounded bias only checks floor/integration/softmax order.
It is not a learned Q/K test, content evaluation or empirical temperature
selection. Keep all cases, including extreme/raw-tiny ones.

Six additional zero-content identities are fixed: translation; screen-unit
rescaling by3.7 with C,h,floor-density transformed consistently; reverse key
permutation; unequal native-cell merging into four rectangles; axis swapping;
and an exact zero-C repeated-input check. The last is implementation
consistency, not an independent numerical reference or a finite point-limit
sequence. Thus production has42 calls. Independent references cover all36
base cases, at both prescribed orders, with no identity-reference substitution.

The sole production rule is Cartesian tensor GL24 on physical subrectangles
with each side <=4h. Tile a cell regularly using ceil(side/(4h)); evaluate
the normalized Student density before summing, with float64 Cholesky solves.
Do not replace integration by center scoring. Integrate the enclosing FOV
separately under the same rule for an additivity check. Add the area floor
once after native integration, then apply b with stable softmax. No hidden
mass clipping, early far-tail zero, adaptive order, alternate density or retry.
Cap each call at4,000,000 density cell-node evaluations, including the
separately integrated FOV. Report the actual work, raw/FOV masses, floors,
normalization, argmax/ties, raw zeros/tiny mass, time and cumulative-process
RSS. Whole42-call phase120s includes per-case output records; final source
verification/inventory sealing is separately timed. Retain available partial
arrays and precise reasons for failure; no failed input is replaced.

## Independent reference and gates

The independent reference must not call production density, tiling,
normalization or integrator. Its different one-dimensional formula is:
for sx=sqrt(Sxx), u=(x-ax)/sx, D=Syy-Syx²/Sxx,

    p_x(x)=3/(4 sqrt(2) sx)*(1+u²/2)^(-5/2),
    Y|x ~ t5(location=ay+Syx/Sxx*(x-ax), scale²=D*(2+u²)/5).

Integrate p_x times the conditional t5 CDF interval over x, using physical
panels <=4h and fixed GL64/96. Evaluate a stable tail interval; do not use
naive subtraction when it loses available precision. D must remain positive;
no repair or pseudo-inverse. SciPy base CDF functions and NumPy linear algebra
are disclosed shared dependencies, not code-path independence violations.
The full reference phase is capped at120s and2,000,000 x cell-node evaluations
across all72 reference calls, including direct FOV work. No replacement order
or retries. Two-order agreement is a convergence check, not a universal proof.

Keep the existing numerical gate values: production vs highest-order
reference P max absolute <=1e-4; raw FOV relative and raw-cell L1 divided by
reference FOV <=1e-6; reference64/96 P max absolute and raw relative <=1e-8.
For these nonempty FOVs and h>0 the mathematical raw mass is positive. A zero
or nonfinite reference is unresolved/failure, not an excuse to drop raw
relative comparisons. Independently validate raw/floor/bias arithmetic,
finite/nonnegative arrays, normalization<=2e-12, separately integrated FOV
additivity<=1e-8, identities<=1e-8, runtime/work caps and complete population.
Apply the floor before b: normalizing raw then adding the same floor changes
the law, especially below-floor cases. A P-only match cannot certify tails.
No explicit integration-domain truncation is used here; do not borrow R22's
Gaussian tail certificate for the Student distribution. GL differences are
numerical evidence, not a rigorous quadrature/roundoff bound.

## Freeze, audit and interpretation

Complete separate toy tests, then jointly freeze the exact contract/protocol,
full local imported source closure, runtime and prior decision bindings before
any formal fixture is evaluated. Producer freeze then independent audit freeze
must precede every formal call. No actual observation archive is opened at any
stage. Produce and seal all42 outcomes before the independent72 reference
calls. Independent audit and root verify every inventory/hash and chronology.
Preserve every failed formal attempt and all earlier scientific failures.

Use the existing NumPy/SciPy environment, OMP/MKL/OpenBLAS=4, no GPU jobs or
installations. Any positive result is only `numerical_preflight_passed` for
this fixed synthetic population. Always `scientific_gate_passed=false`,
`real_inputs_read=false`, `labels_read=false`, `goal_complete=false`.
The original regional, center, perturbation, actual V, optimized GS and
untouched joint-confirmation gates remain mandatory and unchanged.
