# R27: RMS-depth law with independent standard-X panels

2026-10-31. Selected after the accepted R26 numerical preflight failure:
7 reference calls across4 base IDs violated native/direct-FOV additivity.
All native comparisons passed, but that cannot waive the failed FOV checks.
R26 remains failed and all its sources/artifacts are immutable. This new
protocol changes only independent-reference X segmentation and adds two
prospective wide-FOV regressions. No real observations or labels are read.

The production model is exactly `models/round26_rms_projective_pe.py`, SHA256
48b63a51624ddaead32892d46f853cb5cae0b9a88e0e13d29549235b8fa37e9a.
The design and preserved failure are bound by the R26 post-result decision.

## One-Gaussian law and limits

Inputs are finite float64 mu[3], factor[3,3], native bounds[N,4] and bias[N].
The homogeneous screen/depth Gaussian has covariance factor@factor.T. Define
s=hypot(mu_z, *factor_z)=sqrt(E Z²). For s>0 divide mu and factor by s,
add independent XY Gaussian covariance h²I, then perspective-project at
strict Z>0. Raw M(B) is the unconditioned front probability in B. Set
f(B)=1e-30 area(B), and P=softmax(log(M+f)+bias). Fix h=.25 reference-8 units.
There is one Gaussian, no mixture, new learned module or finite-exact RoPE
claim. Real-data anchor and content-temperature rules remain unselected.

If s=0, or if factor_z is exactly zero and mu_z<=0, raw is structurally zero.
Positive deterministic depth gives the exact projected Gaussian plus h²I.
Nonzero factor_z must remain stochastic, including the near-constant fixture;
normalization underflow cannot silently replace it by deterministic depth.
No rank tolerance, covariance repair, eig clipping, pseudoinverse or jitter.
Native positive rectangles must partition their rectangular enclosing FOV.

This differs from R22/R23's fixed screen convolution: conditional added screen
variance is h² E[Z²]/z². A zero-mean radial Gaussian yields a half-mass Cauchy
kernel, although its original perspective support is a point. Preserve this
modeling counterexample. Scalar h permits uniform screen scaling and rotations,
not arbitrary shear unless the regularizing matrix also transforms.

## Fixed population, producer and work

`experiments/round27_rms_fixtures.py` defines 20 base cases and 6 identities.
The original18 R26 bases and6 identities are retained with identical input
bytes and IDs. Two new constant-depth wide-FOV cases are appended to the base
list before the unchanged6 identities, as prescribed by the bound R27 design.
Fourteen base geometries cover positive point, independent/correlated constant
depth, exact-zero depth, negative constant depth, central/noncentral rank1
radial, central/noncentral full rank, negative mean, near-constant depth,
strictly-positive subfloor raw, native432, and rank2. Four repeats add the fixed
bias .6 sin(.7 cx-.2 cy) to cases 2,6,9,12. This bias tests arithmetic only.
Default bounds are 16 unit cells over [-2,2]². Structural zeros use unequal4
from x=[-2,-.5,2],y=[-2,1,2]; zero-depth has bias, negative-constant has none.
Native432 uses the 144x192 grid with 8px cells and (pixel_boundary-.5)/8.

The tiny case mu=(0,0,-12), factor=diag(0,0,1) has front mass Phi(-12), below
the total floor; an explicit positive event proves representable raw mass.
It is not eligible for structural-zero exemption. The near-constant case
factor_z=(0,0,2^-36) is compared to its matching constant-depth geometry.

Six identities are positive world scaling by7; screen translation(.75,-.5);
uniform screen scaling3.7 with h and floor density transformed; 90-degree
rotation; native432 reverse permutation; and unequal merging of each row's
first three cells versus the last. The first four use biased base17 and carry
bias with the cells. Merge uses unbiased base9. All26 attempts are retained.

Producer preparation uses QR nullspaces and a sign-normalized QR-derived
conditional XY factor with hI included, avoiding Schur subtraction. The
analytic positive-depth density and stable positive/negative-t branches are
specified in `docs/round26_rms_projective_design_draft.md`. For q=-t>=10 use the fixed20-term
alternating K(q) series; the next term bounds truncation, and the reported
positive interval must have relative remainder<=1e-10. Moderate negative q
uses erfcx. This series bound excludes floating-point error.

Integrate density with tensor GL24 on tiles whose sides are at most2h.
Integrate the FOV separately. Cap each call at4M actual density nodes,
including direct FOV. Cap the entire26-case producer phase at600s, including
per-case output writes. Save native raw, floored mass, P, direct FOV, branches,
work, argmax/ties, zero/tiny raw, times and process RSS. Failed calls retain
partial arrays and consumed work, with no replacement or higher-order retry.

The exact-arithmetic GL bound is total_area/(pi h² binom(48,24)) for these
tiles; its proof is in `docs/round26_rms_quadrature_derivation.md`. It excludes
density evaluation, factorization, node/weight and summation rounding, and
does not certify tiny-FOV relative error or normalized P.

The two additional formal inputs are fixed before R27 execution: base25 has
mu=(0,0,1), factor=0, zero bias and four cells from x=y=[-4,0,4]. Base26 has
mu=(.21,-.13,1.7), factor=[[.15,0,0],[.36,.05,0],[0,0,0]], cells from
x=[-4,-.3,4],y=[-4,1.2,4], and the same fixed native bias. They test broad
independent and correlated constant-depth FOVs. They are synthetic development
regressions, not untouched scientific confirmation or a new efficacy role.

## Independent reference

Independently normalize RMS and use SVD of the depth factor for a latent
nullspace. Given standard depth zeta, XY has constant conditional Gaussian
covariance and Z=m+sigma*zeta. Integrate the conditional rectangle Z*B at
positive Z, avoiding division by Z. Do not call the producer's preparation,
density or integration code. Construct conditional XY residuals without
subtracting nearly equal covariance entries. Shared NumPy/SciPy libraries
are disclosed dependencies, not independent implementations.

Use depth zeta in [-18,18] and standardized inner X in [-32,32], with explicit
analytic omitted-mass bounds. The Gaussian rectangle uses inner GL32/48;
outer quad_vec GK21 has epsabs=0, epsrel=1e-11/1e-12, norm=max, workers=1.
Outer breakpoints are roots of each native/FOV boundary minus conditional
location at 0, +/-1, +/-8 marginal standard deviations. Inner breakpoints are
conditional Y boundaries at 0, +/-1, +/-8 residual standard deviations.
Only interior finite roots are used, with exact duplicate removal.
In addition, include every interior point from the fixed global standard-X
grid {-32,-30,...,30,32} in each clipped X interval before taking this union.
Thus every X panel has width at most2, including independent XY. This is the
only reference algorithm change. Its pure-Gaussian exact-arithmetic bound
and general-case limits are in `docs/round27_reference_panel_derivation.md`.
Keep all prior toy tests and add independent analytic wide-Gaussian toys;
no R26 formal integration is rerun while developing this reference.
Narrow normal intervals use positive GL16 from midpoint/halfwidth when
halfwidth*(abs(midpoint)+halfwidth)<=.25; other intervals use reflected
log-CDF/expm1 arithmetic. No cancellation-prone direct tiny CDF subtraction.

The reference may scale the depth integrand by the analytic front probability
for numerical stability, but must multiply it back once and report the raw
unconditioned mass. Exact structural-zero branches remain analytic; a true
positive zero or underflowed reference is unresolved and fails.

Each order has at most32768 depth evaluations,512 adaptive depth intervals,
and250M elementary inner nodes. All40 reference calls share600M elementary
inner nodes and1800s. Charge nodes before each batch; GL16 PDF nodes count16,
not one CDF operation. A shared remaining budget prevents overshoot. Record
adaptive errors as estimates, separately from analytic truncation bounds.
Do not increase cutoffs, quadrature orders or budgets after seeing results.
These caps were set during pre-freeze complexity review: native432 can require
roughly220 initial depth breakpoints, so even one32-node inner panel per cell
can already require64M inner nodes. This is operation-count planning, not a
formal fixture trial or a performance measurement. Batch evaluation of all
rectangles preserves the same rules and charges every elementary node.

## Gates and provenance

For positive-mass cases require producer/high-reference P max error<=1e-4,
raw FOV relative error and native raw L1/reference FOV<=1e-6. Require reference
32/48 P and raw differences<=1e-8. Analytic omitted-native-mass bound divided
by reference FOV must be<=1e-10, and its bias-aware normalized-P bound<=1e-8.
Conservative per-cell tail bounds may be summed. Neither floor dominance nor
two matching zero references can waive raw-relative gates.

Only base4/base5 are structurally zero. Require every native raw and direct
FOV raw exactly zero and independently reconstruct P proportional to
area*exp(bias); raw-relative metrics are explicitly not defined in that branch.
All cases must satisfy finite nonnegative arrays, single-floor/native-bias
arithmetic, normalization<=2e-12, direct-FOV additivity<=1e-8 and fixed budgets.
Six identities require raw/floored/P absolute errors<=1e-8. Near-constant
versus constant requires P error and raw L1/constant FOV<=1e-8 plus stochastic
branch evidence. Save first-index argmax and exact/near1e-12 ties; argmax
changes do not replace probability gates or introduce an undeclared tie gate.

Pass pure toy/unit tests first. Then jointly freeze full local source closure,
runtime, exact fixtures and R26 acceptance/decision and R27 design bindings, producer
before independent audit. Prove freeze paths do not load arrays or integrate.
All26 production attempts must be globally sealed before any40 references.
Independently audit all hashes, inventories, chronology, arrays and outcomes.
Use NumPy-first existing runtime with OMP/MKL/OpenBLAS=4; no installation/GPU.

Any positive result is only a fixed-population numerical preflight. Always
scientific_gate_passed=false, goal_complete=false, real_inputs_read=false,
labels_read=false. Real applicability and all previous regional/center,
perturbation, actual V, optimized GS, strong-control and untouched confirmation
requirements remain pending. Failed attempts are evidence, never discarded.
