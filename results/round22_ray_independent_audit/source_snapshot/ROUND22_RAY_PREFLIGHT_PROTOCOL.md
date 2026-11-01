# R22: bounded numerical falsification of one 3D Gaussian pushforward

2026-10-31, selected after accepted R21 development diagnostics. This is a
synthetic numerical study, not an efficacy candidate or a repair of an old
scientific failure. No TRAIN/VAL/GS/fresh arrays or labels are inputs. R21 does
not establish that one 3D Gaussian can recover the shape retained by four
local Gaussians. All previous scientific requirements remain unchanged.

## Selected measure and input contract

Use camera-frame `X = mu + L xi`, `xi ~ N(0,I_r)`, with explicit finite
float64 `mu[3]` and `L[3,r]`, r=0..3. The factor is the input representation;
there is no covariance inversion, added thickness, jitter, learned module,
eigenvalue clipping, opacity or visibility oracle. Let U=(Xx,Xy), Z=Xz,
`pi(X)=H U/Z+c` only on the strict event Z>0. With h=.25 in reference-8
image units, independent image Gaussian blur and native rectangular cells B,

    M(B) = E[1{Z>0} Pr(pi(X)+N(0,h²I) in B | X)],
    P(B) = [M(B)+1e-30 area(B)] / [sum_B M(B)+1e-30 sum_B area(B)].

There is no content term (T=0). Sum raw cell masses before applying the single
area floor. In particular, do not normalize by Pr(Z>0) before applying this
fixed floor. Zero forward mass produces an explicitly reported area prior.
Z>0 is not an occlusion or sensor-visibility test. No pre-blur FOV culling is
allowed. The API assumes camera-frame factors; this study does not test an
external world-to-camera preprocessing implementation.

For v=||L_z||²>0, Gaussian conditioning gives

    beta = Sigma_Uz/v,
    U|z ~ N(mu_U + beta(z-mu_z), D),
    D = (L_U N)(L_U N)^T,  N an orthonormal basis of null(L_z),
    Y|z ~ N(H[mu_U+beta(z-mu_z)]/z+c, H D H^T/z²+h²I).

Integrate that rectangle probability over the original positive-depth
Gaussian density. Do not insert another z² Jacobian in this conditional
formula. v=0 is an exact branch: one screen Gaussian when mu_z>0 and zero
raw mass otherwise. Small nonzero v is not thresholded to zero. Invalid input
fails; no effect-dependent parameter repair. This measure has no claimed
finite exact Q/K feature map, rotary group property, differentiable training
implementation, actual-V benefit, GS benefit or new mathematical family.

## Fixed production rule and budget

The exact numerical constants are `POLICY` in
`experiments/round22_ray_contract.py`. Use deterministic scalar/vector
adaptive integration, not a grid selected by observed errors. For varying
depth, integrate standard-normal t on `[max(-14,-mu_z/sqrt(v)),14]` using
`quad_vec`, GK21, max norm, workers=1, epsabs=1e-12, epsrel=1e-10 and at most
512 subintervals. The integrand may be scaled by the analytic p+=Pr(Z>0) to
avoid losing very small positive mass; multiply back by p+ before the floor.
That arithmetic scaling is not a different positive-depth-conditioned law.

Initial depth breakpoints include t=-8,-4,0,4,8 and all positive roots where
either coordinate of the conditional projected mean crosses a cell boundary
or that boundary plus/minus 8h. Keep only roots inside the integration range.
At most 256 breakpoints and 32,768 depth-integrand evaluations per call are
allowed. These geometry-aware splits address the documented narrow-depth
rank1 counterexample; they do not prove uniform convergence.

Diagonal screen covariance uses stable mirrored normal-CDF differences.
Correlated screen covariance uses its own deterministic one-dimensional
conditional-Gaussian rectangle integral, epsabs=1e-13, epsrel=1e-11, limit128;
the implementation declares its geometric split rule in the frozen source.
At most 4,000,000 inner integrand evaluations per call are allowed. Exhausting
any budget, nonfinite/negative outputs, a failed quadrature status or an
unhandled branch is a numerical failure. No retry at higher orders/budgets.
Total wall time for the 15 base plus 8 identity production calls is capped
at 120 seconds, from the start of case processing through the last case JSON
write, including input assembly and per-case output bookkeeping. Unit tests,
final source verification/inventory/sealing and later independent references
are excluded; record final sealing time separately. Timings are hardware-dependent preflight
resource checks, not an end-to-end or scientific efficiency comparison.

The analytic omitted-front-depth mass is computed from normal tails and is
at most 2 Phi(-14). It bounds truncation only. Adaptive integration error
estimates, inner error estimates and successful library status must be saved
but are not rigorous quadrature error bounds. Numerical accuracy is separately
checked against independent formulas below.

## Frozen cases and independent references

`fixtures()` is the exact float64 specification of fifteen base inputs.
All use H=I, c=0, a 4x4 grid of unit cells on [-2,2]² and h=.25:

| Cases | Inputs and purpose | Independent reference |
| --- | --- | --- |
| Four rank0 points | (.2,-.3,1), (.2,-.3,-1), camera origin, (3,0,1) | Analytic Gaussian interval products or exact zero |
| Two constant-depth rank2 planes | mu=(.2,-.3,2); diagonal/correlated L in source | Independent 1D Gaussian rectangle integration |
| Three radial rank1 lines | X=(.2,-.3,1) Z, Z~N(m,1), m=2,-2,-12 | Phi(m) times analytic point-blur mass; includes raw mass below floor |
| Two offset rank1 lines | Z~N(1,1), Ux=beta(Z-1.3)+.2*1.3, Uy=0, beta=32/512 | Projected coordinate x, z=alpha/(x-beta), Jacobian abs(alpha)/(x-beta)² |
| Rank2 plane through camera | X=(xi1,0,xi2), independent standard normals | Positive-half Cauchy CDF in x, Gaussian blur in y |
| Rank3 isotropic at camera | mu=0,L=I | Solid-angle rectangle mass from density 1/[4pi(1+x²+y²)^(3/2)], then independent blur integral |
| Rank3 correlated at camera | mu=0; L=[[1,0,0],[.4,.7,0],[.3,-.2,1]] | Solid angle of each rectangular ray cone after applying inverse L in latent isotropic coordinates, then independent blur integral |
| One canonical tilted rank2 plane | X=(S,V,1+.5S), S~N(0,1), V~N(0,.25), independent | Projected u: s=u/(1-.5u), z=1/(1-.5u), ds/du=1/(1-.5u)², then analytic conditional y blur |

All reference rules use the two predeclared orders64 and96; neither is an
eligible replacement production rule. Point/radial branches are analytic.
Constant-depth correlated mass uses independent fixed Gauss-Legendre
integration. Camera-plane Cauchy and both fullrank solid-angle formulas use
Gauss-Hermite blur integration (1D and 2D respectively). Offset/tilted cases
use fixed Gauss-Legendre panels in projected x over FOV expanded by 14h,
with panel width at most h/2 and support/pole boundaries explicitly split.
No more than 128 panels per call are allowed. The per-cell omitted-blur
mass is bounded by 2 Phi(-14) p+; the FOV bound must account for the union
of cells conservatively. These are truncation bounds, not GL error bounds.
Shared NumPy/SciPy elementary functions are disclosed dependencies; reference
code must not call production conditioning, integration or normalization.

Across both orders on all fifteen base cases, the reference cap is
2,000,000 cell-node evaluations and 120 seconds wall time. Save each order's
raw arrays, call counts, time, tail bounds and any failure. Neither agreement
between two orders nor a small library error estimate proves a global error
bound; no such claim is made. The reference covers these canonical cases,
not arbitrary nonzero-mean fullrank/tilted-plane parameters or the whole rank0..3 domain.
The reference timer covers initial producer verification, reference calls and
array writes, all numerical/identity checks and the final source check;
final summary/inventory/sealing writes are excluded and timed separately.
The correlated zero-mean case was added before any formal forward to cover
the varying-depth correlated inner-integration branch; no budget or tolerance
was increased. Its independent cone reference maps each corner r=(x,y,1) to
a=inv(L)r/||inv(L)r||. The four directions lie in the same open hemisphere
with normal L^T e_z. Split the convex quadrilateral into two triangles; each
has solid angle 2 atan2(abs(det[a,b,c]),1+a.b+b.c+c.a). Divide their sum by
4pi before the GH blur average. This is a change of coordinates in an
isotropic latent Gaussian, not an approximation or an extra z² factor.
The triangle formula is classical ([van Oosterom and Strackee, 1983](https://pubmed.ncbi.nlm.nih.gov/6832789/)); its unit-vector expression is also given in
[Eqs.8–9 of Nikkhoo and Walter, 2015](https://academic.oup.com/gji/article/201/2/1119/572006).
The cone adaptation here is our analytic reference derivation, not a novelty claim.

`identity_fixtures()` separately prescribes eight production comparisons:
world positive scale3.7, origin isotropic factor scale1e-6, screen units3.7
(including h and inverse-square floor-density changes), screen translation,
key permutation, unequal-area merging into four cells, an orthogonal factor
rotation, and a fixed forward point-limit check with L=1e-6 I. Compare saved
production arrays with the corresponding base production arrays. Only the
fifteen base cases have independent high-order references. The point-limit
comparison is a sanity check at one epsilon, not an independent integration
of its finite fullrank input or proof of an entire limiting sequence.
The origin-scale invariant law must not be confused with the exact rank0
camera-origin input, whose strict positive-depth mass is zero.

## Frozen acceptance and complete reporting

For every base case, require finite nonnegative masses/probabilities and P
normalization error<=2e-12. Against reference96, production max absolute P
error must be<=1e-4, relative FOV mass error<=1e-6, and sum absolute cell-mass
error divided by reference FOV mass<=1e-6. Reference64 vs96 must satisfy
the corresponding three errors<=1e-8. Exact-zero reference cases require
exact-zero raw production mass, not division by an arbitrary tiny denominator.
Do not treat a failed/nonconverged reference as a passing production result.

Save all per-cell relative errors as diagnostics, including tiny cells;
they have no separate relative-error gate beyond the FOV-scaled L1 gate.
This explicitly does not certify relative accuracy of every negligible cell.
The m=-12 radial case must pass raw FOV and L1-relative checks even when
the final P is dominated by the area floor. All raw mass, p+, raw FOV sum,
floor mass, fallback/tiny flags, argmax changes, time, work counters and peak
process RSS remain visible; no cases/rows are dropped.
The production rule also independently integrates the enclosing FOV rectangle.
Its difference from the sum of native cell masses must be<=1e-8 in absolute
mass. The main FOV-relative gate and normalization use the sum of native
cells. Reference work counts native cells only; it has no extra FOV integral.

For each method with omitted raw mass bounded by T, require the conservative
floor-denominator amplification `2*T/total_floor <= 1e-8`. This uses a lower
bound on the denominator rather than an estimated mass and certifies only
the omitted-domain contribution. Reference cell-tail bounds must be combined
into a valid FOV T before using this test. For identity comparisons require
max absolute raw/P differences<=1e-8, with prescribed permutation/merging;
the fixed point-limit check uses the same 1e-8 bound. Algebraic normalization
still uses2e-12. These integrated comparisons do not require quadrature to
preserve identities bitwise. Record approximate/actual ties; no universal
argmax stability requirement is inferred from a small P error.

The conjunction of all base, reference, identity, library-status and resource
checks determines numerical preflight status. A failure stops this fixed
implementation from real-data experiments. A pass means only these fixed
synthetic checks passed; it does not pass any old scientific requirement.
No case replacement, h/T search, budget promotion or waived failure is allowed.

## Chronology and next work

Development unit tests may use separate toy inputs before freezing; they may
inspect the case specification but must not run the fifteen base or eight
identity forward calls before the lock. Before any formal numerical case,
freeze this protocol, exact base/identity JSON, all ten declared source files
plus their discovered local import closure,
source snapshots and runtime/build hashes. Bind the accepted R21 result and
post-result decision. NumPy precedes numerical imports; OMP/MKL/OpenBLAS=4.
No installation, system GLIBC changes or GPU jobs are needed.

Producer writes all available case arrays and metadata without overwriting a
started run, seals a complete inventory including failures, then the separate
reference/audit process verifies locks and all arrays before computing its
own comparisons. Audit outputs are also inventoried. Root checks current and
snapshot source hashes, case inventory, chronology, runtime, gates and every
output hash before accepting the interpretation. Source or engineering errors
are recorded with their chronology; formal runs are never silently replaced.

If this preflight passes, a separate old-development protocol must establish
same-information comparisons: Jacobian and pushforward operators sharing
the same legal 3D Gaussian, alongside historical 2D and full-pixel controls.
The 3D Gaussian center is not automatically the independently designated
source center. No raw-moment equivalence or R20 localization inheritance is
assumed. Full region/center/scale/stress, actual V, optimized GS and untouched
joint confirmation remain required. Old repeated TRAIN/VAL cannot become a
fresh confirmation set. The research goal remains active whatever this
numerical study returns.
