# R27: fixed standard-X panels in the independent reference

2026-10-31. This is a prospective reference revision after the accepted R26
numerical failure. No R27 formal fixture, real input or label was executed to
choose the rule. R26 sources and sealed results remain unchanged and FAIL.
This document does not select an efficacy candidate or claim general numerical
correctness of the projective law.

R26's independent rectangle integrator split standardized X at conditional-Y
CDF transition roots. It did not independently limit the interval over which
the standard-normal marginal density was integrated. When the conditional-Y
coefficient is zero, there are no such roots at all. A wide enclosing FOV can
therefore receive one GL32/48 rule across a large part of [-32,32], while its
smaller native cells receive much narrower rules. A nonzero Y coefficient also
does not ensure that its roots cover the marginal Gaussian peak.

The sealed R26 outcomes expose this gap. Reference direct-FOV additivity failed
for seven orders: central radial GL32/48 (about 1.024e-4 / 3.919e-7), noncentral
radial GL32 (9.857e-6), native432 GL32/48 (7.822e-5 / 7.861e-7), and the biased
central-radial repeat. The outer adaptive error estimates were only about
4e-14 to 5e-13: they estimated integration error of an already inaccurate inner
integrand. Native-cell comparisons agreeing with production and with each
other did not waive the mandatory separately integrated FOV check.

R27 makes one reference change. Every clipped standardized-X interval includes
the internal points of the fixed global grid

`-32, -30, ..., -2, 0, 2, ..., 30, 32`.

These points are united with the existing conditional-Y roots and the interval
endpoints, with exact duplicate removal. Each resulting panel is at most two
standard deviations wide, including when the Y coefficient is zero. Both the
scalar toy path and the batched ragged-panel path use this rule. The RMS/SVD
construction, depth and X cutoffs, GL32/48 orders, outer GK21 rules, narrow GL16
formula, all gates, and the production model remain unchanged. New panel work
is charged to the same fixed resource budgets.

For the pure Gaussian marginal this has an exact-arithmetic justification.
Write phi(u)=exp(-u^2/2)/sqrt(2*pi). Fourier inversion gives

\[
\sup_u |\phi^{(2n)}(u)|\le
\frac{(2n-1)!!}{\sqrt{2\pi}}.
\]

The n-point Gauss-Legendre error on an interval of length ell is at most

\[
A_n\,\ell^{2n+1}\sup|f^{(2n)}|,\qquad
A_n=\frac{(n!)^4}{(2n+1)((2n)!)^3}.
\]

For disjoint panels contained in [-32,32], with maximum length two, summing
the errors therefore gives

\[
E_n\le
64 A_n 2^{2n}\frac{(2n-1)!!}{\sqrt{2\pi}}
=\frac{64\,2^n(n!)^3}
{(2n+1)((2n)!)^2\sqrt{2\pi}}.
\]

The constants are approximately 1.90905e-63 for n=32 and 1.44126e-103 for n=48.
These are bounds for exact nodes, weights, density values and arithmetic,
not floating-point certificates.

When the conditional-Y coefficient is zero, its interval probability Q(z)
does not depend on standardized X. The inner integrand is phi(u)*Q(z), with
0<=Q(z)<=1, so the same bound applies. Positive outer Gaussian weights have
total mass at most one; integrating these inner error bounds does not enlarge
them. This addresses the missing marginal-peak resolution mechanism in the
radial cases, in exact arithmetic.

It is not a uniform proof for every correlated Gaussian rectangle. With a
nonzero Y coefficient the integrand is phi(u)*Q(u); Q's derivatives can become
large as the conditional slope relative to the residual standard deviation
grows. The retained Y roots help locate these transitions, but the pure-phi
bound does not bound this full product, adaptive outer errors, SVD rounding,
CDF evaluation, or underflow. An absolute bound alone also gives no uniform
raw relative guarantee for arbitrarily small FOV mass. All independent
two-order, raw/P, truncation, direct-FOV, identity and resource checks remain
mandatory, with no changed thresholds or selective retry.

The new independent regression tests use positive constant depth and an
axis-aligned Gaussian. Its exact native and direct-FOV probabilities are
products of normal-CDF differences. The standardized FOVs are
[-24,24] x [-1.25,1.25] and [-31.7,30.2] x [-1.25,1.25], with unequal native
partitions and h=.29, distinct from the formal fixtures. Each GL order is
checked against this analytic oracle and native/FOV additivity. The asymmetric
case prevents a symmetric interval from being the only wide-domain check.
The prior 26 toy checks are retained; their shared-budget expectation changes
only because the new fixed grid splits the native/FOV toy at zero and charges
128 rather than 64 X nodes before its deliberate narrow-Y budget failure.

These tests and this rule are new, explicitly post-R26-failure diagnostics.
They cannot be represented as successful R26 checks or as untouched scientific
confirmation. R27 must receive its own joint freeze and full numerical run.
