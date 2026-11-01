# R29 log representation and unchanged error quantities

2026-10-31. This note describes an algebraic representation, not a new density
or a general floating-point certificate. The R26 exact-arithmetic quadrature
bound and R27 independent-reference limitations continue to apply.

For finite node log density l_i, positive GL weight w_i and positive tile
Jacobian J, log tile mass is log J+LSE(l_i+log w_i). A cell is LSE of its tile
masses. The same operation computes a separately integrated direct FOV.
With native floor f_j=area_j*1e-30 and bias b_j, log floored mass is
logaddexp(L_j,log f_j), and log P_j is this value plus b_j minus the LSE of
all logits. Thus there remains exactly one floor and one native normalization.
The original ordinary floor product must still be positive and finite.

Exact structural zero is log0=-inf with an explicit structural state. An
ordinary positive mass that cannot be materialized has a finite canonical
log and no ordinary packed output at that index. Unfinished integration has
its own completed masks and tile counts. Neither kind is a structural zero.

For finite a,b, define D(a,b)=log|exp(a)-exp(b)|. If a=b the difference is
exactly zero, represented by -inf. Otherwise

    D(a,b)=max(a,b)+log(-expm1(-abs(a-b))).

If exactly one log is -inf, D is the other; if both are -inf, D=-inf.
This avoids subtracting two tiny ordinary values. Given producer logs Lp
and reference logs Lr, with Sp=LSE(Lp),Sr=LSE(Lr):

- FOV relative error is abs(expm1(Sp-Sr)), or equivalently exp(D(Sp,Sr)-Sr).
- Native raw-L1/reference-FOV is exp(LSE(D(Lp_j,Lr_j))-Sr).
- Native/direct FOV absolute error is exp(D(Sp,Ldirect)).

Compare canonical log errors to log(original threshold) before optional
ordinary diagnostics. A large log error fails without first overflowing an
exponential. The original all-structural-zero exclusions apply unchanged;
there is no arbitrary tiny-denominator exemption. Absolute raw identity
errors use D and the original absolute threshold, including merge identities
where parent masses are combined by LSE. P is still compared in ordinary
probability units as declared by the old gates.

For a log-sum local maximum m, exponentiating each finite l_i-m can lose
terms. If N terms round to zero and d is their largest shifted log, their
exact scaled sum is at most N*exp(d). Store logN+d; its absolute counterpart
adds m, and its ratio to the retained sum subtracts the retained scaled log
sum. These are local omitted-term bounds relative to computed log inputs.
They exclude errors in those inputs, exp itself on retained terms, positive
terms absorbed by addition, LSE output rounding and all quadrature error.
Maximum bounds over calls are not cumulative pipeline bounds. In particular,
very large negative finite logs can have poor relative accuracy through
their own float64 spacing. Finiteness is never a substitute for reference
comparison.

The old20 independent references retain their SVD/conditional integration,
cutoff-tail bounds and two-order comparisons. The3 additional exact positive
constant-depth diagonal cases instead use independent Gaussian CDF interval
products without cutoffs. For a standardized interval[a,b], reflect positive
intervals to the negative side and use logPhi(b)+log(-expm1(logPhi(a)-logPhi(b))).
Enforce valid strict interval ordering and finite log probability; do not
replace a failed CDF difference by zero. Independence gives addition of X/Y
log interval probabilities. This is an analytic special-case oracle with
floating CDF evaluation, not an arbitrary correlated-projective reference.
