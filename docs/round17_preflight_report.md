# R17 trace-temperature numerical preflight report

The frozen synthetic preflight, first-TRAIN-pair numerical preflight, and
supplementary arithmetic audit all passed their declared numerical checks.
This establishes implementation agreement over the tested domain. It does
**not** establish improved localization, regional KL, speed, generalization,
or passage of a scientific gate. No regional or center labels, VAL inputs, or
test inputs were read in these preflights.

This is an external report written after the numerical results. It does not
modify any frozen protocol, source, threshold, case, or result. The numerical
contract is [ROUND17_PREFLIGHT_PROTOCOL.md](../ROUND17_PREFLIGHT_PROTOCOL.md);
the supplement is [round17_arithmetic_addendum.md](round17_arithmetic_addendum.md).

The candidate retains the original native Gaussian cell integral and uses
`tau = T h² / (h² + tr C)`, where `C` uses reference-8 coordinates. Production
geometry and softmax use float64; prepared support/content use float32. The
overflow-safe temperature computation follows the locked order
`s=max(h²,C00,C11); r=h²/s; c=C00/s+C11/s; tau=T*(r/(r+c))` in float64,
then `tau32 * content32` is evaluated in float32 before adding the float64
geometry. The numerical reference is independently implemented in NumPy/SciPy.

## Coverage and results

| Check | Declared coverage | Maximum errors | Result |
|---|---|---|---|
| Synthetic integration and arithmetic | 19 fixed query cases, four native key partitions, four h values, three T values: 48 matrices / 912 probability rows | Probability `6.9588518974983415e-12`; GL64→96 reference convergence `2.55351295663786e-15`; same-mass probability `2.6645352591003757e-15` | PASS |
| First old TRAIN pair in each of three scenes | 42 support sets; 168 geometry matrices / 1,344 geometry rows; 504 probability matrices / 4,032 probability rows | End-to-end probability `5.1419837626132114e-6`; same-prepared-support integration-only probability `9.992007221626409e-16`; reference convergence `1.1934897514720433e-15` | PASS |
| Saved TRAIN arithmetic supplement | All 42 support and 168 geometry archives, 504 matrices / 4,032 rows | Same-mass probability `5.551115123125783e-16`; tau32 exact; bias32 exact | PASS |

The probability error limit remains `1e-4`, reference-convergence limit `1e-8`,
and same-mass probability limit `1e-12`. No tolerance was widened. The
production integral uses the frozen GL32 implementation and native 24-pixel
cells split into 12-pixel integration tiles. The independent GL64/96 reference
integrates each native rectangle directly. Internal integration tiles do not
change key identities or introduce another area floor.

Synthetic cases include PSD/rank-one/rotated/large covariance, mean on an edge
or corner, outside-FOV and extremely small tail mass, complete raw underflow,
float32 trace-overflow stress, and symmetric native-grid ties. Among the 304
distinct geometry rows (temperature repetitions removed), 40 have zero raw
reference FOV mass and 56 have raw mass below `1e-20` (including those zeros).
There are 16,475 zero raw reference cells in this geometry-only count. All
rows remain present; the declared native-area floor is applied once.

There are **29 synthetic argmax differences**. Every affected reference row
has an exactly zero top-two probability gap. However, it is not correct to
claim that every actual winning bin has exactly the reference maximum: one
actual winner is `2.7755575615628914e-17` below that maximum. Original native
first-index argmax behavior and all differences are retained. The predeclared
contract reports argmax changes; it does not replace the probability check
with a new exact-argmax gate. Synthetic point-limit and T=0 identities, tau32,
and content-bias32 checks all pass.

The real numerical cohort is frame pair `[30,45]` from each of
`harvard_c3/hv_c3_1`, `harvard_c5/hv_c5_1`, and `harvard_c6/hv_c6_1`.
Each uses all nine g8/g16/mixed query/key conditions and five native g24
scale conditions, four h values `.25,.5,1,2`, three T values `0,5,10`, and
eight predetermined query indices per condition. Inputs are limited to the
projected position, pixel covariance, query feature, and key feature fields.

Independent float64 support preparation and the production float32 path have
maximum normalized mean difference `1.0410281292737282e-6` and covariance
difference `3.8071095280417095e-7`, using `abs(actual-reference)/(1+abs(reference))`;
content absolute difference is `4.76837158203125e-7`. The independent end-to-end
and same-production-support references separate preparation differences from
integration error. These are observed numerical differences, not a claim
that every end-to-end difference has one isolated cause.

The 1,344 real geometry rows include **zero** zero-FOV rows and **seven**
tiny-FOV rows below `1e-20`; no rows were removed. Production raw cells contain
9,773 exact zeros, versus 9,800 for the independent reference. The largest
production background-floor fraction is `0.0038603472820515983`. Real
end-to-end argmax changes are **zero**.

The arithmetic supplement was frozen at Unix `1793362315.6986463`, before
TRAIN numerical execution began at `1793362344.0537663`. It reads only the
saved prepared covariance/content, raw mass, and original probabilities, and
verifies every archive hash and the complete 4,032-row population. The locked
production function recomputes tau/bias for comparison with independent
arithmetic; the original execution did **not** save tau/bias intermediate
arrays. Same-mass reference probabilities are compared against the original
saved probabilities. Exact tau32/bias32 agreement and zero argmax changes
therefore strengthen the arithmetic check without misrepresenting which
runtime intermediates were retained.

## Development-test accounting

The reported original combined run passed 65 development tests. Fresh
collection of the unchanged files gives the following breakdown; it corrects
the provisional allocation of tests between runner and Stage A.

| Test file | Collected tests | Scope |
|---|---:|---|
| `tests/test_round17_trace_pe.py` | 25 | Production formula, precision, limits, gradients |
| `tests/test_round17_trace_numerical_audit.py` | 19 | Independent numerical reference and synthetic contracts |
| `tests/test_round17_preflight.py` | 11 | Preflight orchestration and source/field restrictions |
| `tests/test_round17_feasibility.py` | 10 | Stage A driver and eligibility contracts |
| `tests/test_round17_arithmetic_audit.py` | 13 | Supplementary arithmetic audit and provenance |

Thus the original preflight-related subtotal is **55**, with **10 Stage A
tests** in the 65-test combined run. The supplementary **13 tests** also
passed separately: total collection is **78**, or **68** when excluding
Stage A tests. Collection is not presented as an additional test execution.
These development checks and the formal numerical runs have different roles;
neither supplies scientific performance evidence.

## Artifact identities

The full source closure, exact numerical cases, versions, cohort/input hashes,
and source snapshots are bound by the numeric lock. The arithmetic lock binds
that original lock and its three additional source/document/test files. Each
completion manifest binds its result files; none is replaced by this report.

| Artifact | SHA-256 |
|---|---|
| `results/round17_preflight/lock.json` | `4b3d697d1e09f856aee46c494640df6642d9da46e2bda12a4519709cbae15bcb` |
| `results/round17_preflight/synthetic/summary.json` | `50c2b32d73069c448e1cf1260156b9d800277ec5d6c61a5ed20bd58cffa8cf34` |
| `results/round17_preflight/synthetic/completion.json` | `c3f112be0942c4404a4cd4556144445212605702703cccc1e25e7a20b836d34a` |
| `results/round17_preflight/train/summary.json` | `be021bee3fbe984a0eec5262b1700b3284ff9121a1fe4294e162c10cbc4e4e47` |
| `results/round17_preflight/train/completion.json` | `de6d74668777661133404c6257e0fb3486f3167a20d5b9d6c70d24a1f438cb83` |
| `results/round17_arithmetic_audit/lock.json` | `19ee0c66f867f0943b495caa74260b52665712e73270be6c414e7eff11360784` |
| `results/round17_arithmetic_audit/verification.json` | `f7fc49a1c538f555ac384a79332bfc6ed10d67be83ddc87328c310d7050be8d3` |
| `results/round17_arithmetic_audit/completion.json` | `2cfd5da32c3a56722c83779bef575fd1ee7555a09ba2f0c6dcf39d274c1df56c` |

The source model and all preflight implementations remain unchanged. The
next scientific questions are handled by the separately frozen complete
TRAIN eligibility/regional protocols and their independent audits. No
numerical result here authorizes bypassing those checks or their stop rules.
