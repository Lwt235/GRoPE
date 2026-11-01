# R27 preparation: preserve the RMS law, repair the independent X panels

2026-10-31. Prospective numerical revision after the accepted R26 failure.
No R27 formal fixture, real array or label has been evaluated. The R26
production model, law, h, floor, bias and its failed formal attempt remain
unchanged. This document selects a new synthetic numerical preflight only.

R26 completed24 production calls, but7 independent reference calls spanning
four base IDs violated native-sum versus separately integrated-FOV additivity.
The largest gap was1.0239028297359676e-4 against the unchanged1e-8 gate.
All native producer/reference and reference32/48 comparisons passed, which
does not waive the failed FOV checks. R26 is numerically unresolved, not passed.

The reference's inner integral contains the standard Gaussian density phi(X)
times a conditional Y interval probability. Its breakpoints covered changes
in that Y probability but omitted the X density's own variation. With zero
conditional XY coefficient there were no interior Y breakpoints at all;
even with correlation, broad direct-FOV X panels could remain. Outer adaptive
convergence only estimates the error of the already approximated inner
integral. Its tiny error estimates did not certify the whole reference.

For R27, add all interior points from the fixed global standard-X grid
{-32,-30,...,30,32} to every clipped inner-X interval, then union with the
unchanged conditional-Y roots and remove exact duplicates. Every panel is
at most2 standard-X units, including when XY are independent. Keep GL32/48,
outer GK21 tolerances, cutoffs, narrow-CDF GL16, budgets and all error gates.
Use the exact same frozen R26 producer. This is a reference integration
revision, not a change to the model or a new favorable implementation role.

The ordinary Gaussian derivative bound and GL remainder justify very small
exact-arithmetic absolute error for the independent-XY wide Gaussian case.
They do not prove arbitrary correlated conditional rectangles or tiny-FOV
relative accuracy. Preserve every original convergence, raw, P, truncation,
direct-FOV, identity and resource requirement. No claim of a general domain
certificate or an efficacy gain follows from a future numerical pass.

Repeat all18 R26 base inputs and6 identities without changing their bytes or
IDs. Add two prospective wide-FOV constant-depth base cases, both h=.25:

1. ID25_wide_constant_axis: mu=(0,0,1), factor=0, zero bias;
   four quadrants from x=y=[-4,0,4]. This has analytic independent Gaussian
   cell masses and a direct FOV spanning32 standardized X units.
2. ID26_wide_constant_correlated: mu=(.21,-.13,1.7),
   factor=[[.15,0,0],[.36,.05,0],[0,0,0]];
   four rectangles from x=[-4,-.3,4],y=[-4,1.2,4], with the same prescribed
   bias .6 sin(.7 cx-.2 cy). This checks the broad correlated case.

The ordered population is the original18 bases, these2 new bases, then the
original6 identities:26 producer calls and40 reference calls. The new wide
cases fit the unchanged4M producer-node cap by construction. Before formal
execution, add distinct analytic wide Gaussian unit tests (including an
asymmetric standard-X interval), freeze the full new source closure/runtime
and both snapshots, and prove metadata freeze reads no numerical archives.

No real-input anchor or content rule is selected. The possible shear and
3D-anchor second-moment constructions remain unselected mathematical ideas;
old region/center, robustness, actual V, optimized GS and untouched joint
confirmation still determine scientific success.
