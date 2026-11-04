# R29: preserve the RMS Gaussian law using canonical log mass

2026-10-31, selected after accepted R28 applicability failure. R28 completed
40/72 calls;32 failed at the combined positive-density underflow/overflow
guard. Its metadata does not distinguish the precise offending floating-point
value. No R28 retry or reclassification is authorized. No labels were read.

The next experiment changes numerical representation only. Keep the R26 RMS
Gaussian, exact depth-rank admission, QR construction, positive-front guard,
h=.25, floor density1e-30, negative-tail formula/20-term remainder gate,
GL24, tiles<=2h, independent direct-FOV integration and original work/time
budgets. There is no anchor, learned content rule, mixture or extra network.
Return the existing finite log-density before its premature ordinary exp;
use positive-weight log-sum-exp for tiles and native accumulation. Finite
log-density alone is not evidence that the integration is accurate.

Canonical raw output is log mass, with exact structural zero explicitly
separated from unrepresentable positive ordinary mass and unfinished partial
work. Do not expose an underflowed0 as canonical raw. Keep indexed ordinary
materializations only where valid; preserve canonical logs everywhere else.
The once-only area floor is logaddexp(log_raw,log_floor), followed by the
same bias and normalization. Structural-zero flags retain only the two exact
R26 branches. Nonfinite log geometry, lost depth rank or failed front admission
remain failures. No tail cutoff, clipping, retry or density threshold is added.

Log-sum-exp itself can lose relatively tiny terms in exp(log_term-max).
Record these occurrences and an explicit log upper bound from their count
and largest shifted exponent, with the scope limited to that local positive
sum in exact arithmetic. Do not label that bound a total floating-point,
quadrature or general relative-error certificate. Ordinary materialization
can be inaccurate in the subnormal range; numerical gates use canonical logs.

Before reading any real arrays, freeze and run all26 R27 synthetic descriptors
unchanged, plus three predetermined constant-depth diagonal Gaussian cases:

1. Native432 bounds (pixel_boundary-.5)/8;mu=(12,9,1),F=0,bias0.
2. Four cells from x=[9.75,10,10.25],y=[-.25,0,.25];mu=(0,0,1),F=0,bias0.
3. One cell [0,-.25,12,.25];mu=(0,0,1),F=0,bias0.

These separately exercise many far cells, an entire FOV below ordinary
float64 range, and a visible cell containing far nodes/tiles. They are not
selected from R28 outputs. Existing26 descriptors retain their input bytes;
production output bytes may change because reduction order changes.

For the20 old base cases, use the unchanged R27 independent SVD/conditional
Gaussian reference at orders32/48 and convert its positive raw results to
logs for comparison. Preserve its strict failures and all tail/work checks.
The three new diagonal constant-depth cases have a separate independent
analytic reference: normal-CDF interval products evaluated as log differences,
using their actual homogeneous mean/covariance and h. Direct FOV is computed
separately. This analytic branch is restricted to those declared exact
constant-depth, diagonal cases; it cannot be used as a general reference or
counted as an order-refinement experiment. Do not import production density.

Keep old P, raw-FOV-relative, raw-L1/FOV, reference-pair, direct-FOV,
normalization, identity, near-constant, tail and budget thresholds unchanged.
Use log/expm1 differences to express the identical raw quantities when an
ordinary value is unrepresentable. A floor-dominated P cannot waive raw gates.
Retain all6 old identities and the finite-epsilon near-constant comparison.
Each production call retains4Mnodes, production phase600s; existing independent
reference budgets remain250M/call,600M/phase and1800s. Count analytic CDF
evaluations separately rather than fabricating quadrature nodes.

Joint toy checks, full source/runtime freeze and metadata-only sentinel checks
precede formal synthetic execution. Independently audit complete inventories,
log and ordinary materialization semantics, chronological journals and every
unchanged numerical gate. Preserve all failed attempts. A numerical pass
permits a later separately frozen real applicability screen only. It does
not authorize labels, anchor/content selection, V/GS/fresh confirmation or
scientific success. Every earlier failed scientific gate remains mandatory.
