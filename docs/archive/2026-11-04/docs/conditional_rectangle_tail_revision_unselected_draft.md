# Unselected conditional-rectangle tail revision

2026-10-31. Analytic preparation only, while the fixed R23 run remains under
audit. No real input/output array was read, no model was run, and no frozen
R22/R23 source was changed. The reported first failures motivate examining
one arithmetic limitation; they do not establish the full R23 outcome. This
revision is **not selected** before that outcome is sealed and accepted.

## A narrowly defined change to the same measure

The current correlated screen-Gaussian rectangle calculation uses

    R(B|z) = integral_[l,u] phi(t) w_B(t,z) dt,       0 <= w_B <= 1,
    l=(x_left-m_x(z))/s_x(z), u=(x_right-m_x(z))/s_x(z).

Here the screen covariance already includes h²I, and w_B is its conditional
y interval probability. The frozen implementation scales the integral by
phi(t_closest), where t_closest is the integration point nearest zero. A
remote finite cell can have a strictly positive mathematical tail but an underflowed
scale; the implementation correctly fails rather than silently accepting it.

The proposed numerical approximation integrates only
`[max(l,-14), min(u,14)]`. Test an empty intersection before forming the scale,
and return zero integral plus a declared omitted-tail certificate for it.
For a nonempty interval, the new closest point has |t_closest|<=14; its scale
is at least phi(14), approximately 1.10e-43, so this particular scale
underflow cannot occur. The scaled density exponent remains nonpositive and
bounded on that interval. Keep the existing conditional-y computation,
geometry breakpoints filtered into the retained interval, tolerances and
budgets. There is no h/T search, change to the 3D Gaussian, jitter, rank
threshold or new PE component. The diagonal stable-CDF path can stay intact.

This omits a quantified part of the *blurred conditional distribution*; it
is not a pre-blur FOV visibility test. The exact target remains the original
positive-depth pushforward. The approximation must not be described as exact
tail evaluation or a change to the support of that target distribution.

## Certificate, including the floor

Write delta=2 Phi(-14), approximately 1.56e-44. For each correlated rectangle,

    0 <= R(B|z)-R_trunc(B|z) <= delta.

Integrating with the original positive-depth density gives a per-cell raw
error at most p_plus*delta. A simple conservative native-cell L1 bound for N
cells is `T_inner=N*delta`; `N*p_plus*delta` is also valid when the analytic
positive-depth mass is safely represented. Use the unscaled bound if the
numeric representation of p_plus would otherwise understate a certificate.
With the old depth-domain certificate T_depth, the total omitted-domain
bound is `T=T_depth+T_inner`. This sum does not include quadrature/roundoff.
For exact constant depth, T_depth=0 and the front indicator is 0 or 1.

The N multiplier is deliberately conservative. At a fixed z all native cells
use the same standardized x variable, so disjoint cells actually admit a
common-event union bound delta. Such a tighter implementation is unnecessary
to meet the existing tail gate; it must not be assumed for separately
implemented or differently standardized operations without proving it.

For raw masses m_i, nonnegative omitted masses e_i and fixed floors f_i,
let F=sum_i f_i>0. With sum_i e_i<=T and the same single floor on both sides,
normalization gives

    ||P_exact - P_trunc||_1 <= 2T/F,
    max_i |P_exact_i-P_trunc_i| <= 2T/F.

Thus retain the existing amplification gate 2T/F<=1e-8, now using both tail
terms. For the R22 16 unit cells, F=16e-30 and T<=17delta, giving roughly
3.32e-14. For R23's 432 unit-area reference-8 cells, F=432e-30 and
T<=433delta, giving roughly 3.13e-14. These are analytic arithmetic estimates,
not new model observations. Screen-unit transformations must preserve F by
the existing inverse-area floor-density transformation.

The enclosing-FOV integral has its own per-rectangle inner bound delta; do
not count it as an extra native cell in P's denominator. Retain its separate
mass and tail record. Exact truncated integration remains additive because
the same x-tail event is removed from every cell; finite quadrature still
requires the old sum-cells versus FOV check. Report the conservative N-cell
bound even when it is looser than this structural observation.

## Optional shared-x vector integration: a cost question, not a tail proof

The 432 native keys have 24 x intervals and 18 y intervals. At a fixed depth,
the 18 cells sharing an x interval can be integrated as one vector with
components `phi(t)*w_j(t,z)`. The x scale and conditional y mean are common;
conditional Gaussian CDF differences remain separate components. Use the
union of their existing y-boundary roots and root-plus/minus-8-conditional-sd
breakpoints, together with the same standard-normal core breakpoints, all
filtered into the clipped x interval. This is a rearrangement of the same
18 integrals. Retain a separately integrated enclosing FOV; summing vector
components is not an independent FOV check. For a general cell list, grouping
must use identical declared x intervals, not an approximate coordinate merge.

Such grouping can reduce Python callback and repeated scalar arithmetic
overhead. It does **not** automatically reduce the old work count: a callback
evaluating 18 cell integrands must consume 18 inner cell-node evaluations,
not one. Keep the 4,000,000 cap in those comparable units and separately
report vector callbacks, component evaluations and scalar FOV work. A shared
mesh can require at least the hardest component's nodes for all 18 outputs,
so it can increase total charged work or fail a fixed budget sooner. Keep
the existing per-integral subinterval cap for the shared mesh and retain all
failures; do not divide counts by vector width to claim a speedup.

A direct substitution of max-norm `quad_vec` with the old relative tolerance
also needs care: its relative tolerance is based on a vector norm, so a large
component can loosen the allowed estimate for a small component. One possible
predeclared conservative rule is to request max-norm absolute error at most
the old inner epsabs (with relative stopping disabled), then verify its error
estimate against each component's original abs/rel threshold. This is at
least as strict on the estimates, may increase cost, and remains an estimate
rather than a proof. Tolerances apply to the same scaled integrals as before;
multiply error estimates by the shared marginal scale and original depth
mass before reporting raw errors. No failed group is retried through another
kernel. Tail bounds, raw-relative canonical gates and total budgets cannot
be replaced by successful vector-library status.

This optional engineering direction is distinct from the justified tail
bound. Neither it nor the tail clipping has been implemented or selected.
The two early reported 4M-work failures make an automatic cost-success claim
particularly unwarranted; the complete accepted R23 status must decide what
merits a separately frozen revision.

## What this does not establish

The floor certificate is an **absolute probability** statement. An entire
remote FOV can have positive mass below the omitted-tail bound, so returning
zero can have raw relative error one while easily passing the P-tail gate.
No universal raw-FOV-relative guarantee follows. All original R22 raw-FOV
and raw-L1/reference-FOV gates at 1e-6, both reference orders and their 1e-8
convergence gates, exact-zero cases, identities and resource limits must
remain unchanged. In particular, the m=-12 radial below-floor case must
still pass its raw comparisons. Leaving its diagonal path unchanged is a
reason it may be unaffected, not a substitute for rerunning the frozen suite
under a separately frozen revision. Per-cell relative errors and newly zero
cells remain visible; zero after truncation is not exact zero front mass.

The clipping resolves only one failure mechanism. Conditional-y tails can
still lose relative precision; near-singular correlations, large projected
factors, endpoint arithmetic, outer missed narrow peaks, integration status,
or fixed work/wall budgets can still fail. The old adaptive error estimates
are not rigorous bounds, and adding an analytic truncation term does not
turn them into one. Canonical references still do not validate arbitrary
nonzero-mean real-input 3D Gaussians.

Skipping remote cells can reduce work, but 432 rectangles plus a direct FOV
integral at every outer node can remain expensive. Clipping does not remove
the nested cost on cells intersecting [-14,14], increase any cap, justify
early stopping on a favorable subset, or establish scientific efficiency.
Useful future records would include clipped/empty interval counts, separate
depth/inner/FOV tail terms, zero raw cells, existing work counters, partial
arrays and complete failure chronology. A selected revision would first
need independent toy checks for correlated remote rectangles and all old
canonical gates, then a separately frozen applicability decision. No such
selection or execution is made here.
