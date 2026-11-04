# R34: bounded narrow-normal quadrature revision

2026-11-01. The accepted R33 result is a numerical preflight failure: 20
reference calls attempted, 19 completed, one exhausted the unchanged 250M
per-call node budget, and 124 were not attempted under the frozen fail-fast
rule. This does not establish an inaccurate production probability. It also
does not establish accuracy for the uncompleted population or any efficacy.

Select one reference-algorithm revision: in the existing narrow normal
interval branch, replace GL16 with GL8. Keep the branch condition
`halfwidth * (abs(midpoint) + halfwidth) <= .25`, full finite X domain,
GL32/48 X orders, SVD preparation, geometric breakpoints, deterministic
midpoint coalescing, fixed pilot scales, depth cutoff, GK21 outer integration,
all tolerances, all budgets, floor and model unchanged. No new scientific
candidate, anchor, content temperature or model parameter is selected.

For the normal interval write its positive factor as
`halfwidth * phi(midpoint) * integral[-1,1] exp(-a*u-b*u*u) du`, with
`a=midpoint*halfwidth` and `b=halfwidth^2/2`. The unchanged branch condition
implies `abs(a)+2*b <= .25`. A deliberately wider envelope `.5` gives
`f <= exp(.5)`, `abs(log(f)') <= .5`, `abs(log(f)'') <= .5`, and
`integral f >= 2*exp(-.5)`. The quadratic-exponential derivative formula gives

```
M16 = exp(.5) * 16! * sum(k=0..8)
      (.5)^(16-k) / (2^k * k! * (16-2k)!)
C8  = 2^17 * (8!)^4 / (17 * (16!)^3)
relative quadrature remainder <= C8*M16 / (2*exp(-.5)) < 1.88e-13.
```

This is our exact-arithmetic bound using the Gauss remainder and Legendre
normalization in [DLMF 3.5.19–21](https://dlmf.nist.gov/3.5#E19).
It is a local narrow-interval bound, not a floating-point, adaptive-outer or
whole-pipeline certificate. The prospective revision halves only the narrow
PDF evaluations. It need not halve runtime or make every real call feasible.
Do not change the branch threshold or scan lower orders after seeing results.

First rerun the complete fixed synthetic reference preflight. Reuse the same
23 R29 base inputs and all their sealed production outputs; six identities
and the finite-epsilon result remain explicitly inherited. Run exactly 46
new reference calls at orders32/48, with the same 600M phase-node budget,
1800-second phase deadline, 250M per-call nodes, 32768 depth evaluations and
512 outer intervals. No new production integrations, real-array decoding or
label access is authorized in this phase.

Keep all 43 old R29 reference outputs and additionally compare against all
46 accepted R32 reference outputs, preserving the original failed-audit and
serialization-repair history. Oracle keys include a source identifier, so
equal order numbers from different rounds cannot overwrite each other.
The full comparison population is 247: 23 new32/new48, 86 new/oldR29,
92 new/acceptedR32, and 46 old-production/new. Preserve all failed/missing
outcomes and the fixed denominator. All old P/raw/FOV/identity/truncation and
resource gates remain mandatory; there is no precision or budget waiver.
All new/old-reference comparisons, including the additional R32 comparisons,
use the strict reference P/raw limits of 1e-8. Production/new comparisons
retain P1e-4 and raw-FOV/raw-L1 relative1e-6.

Before formal input decoding, freeze the complete producer/auditor source
closure and runtime jointly, preserve development-test attempts and source
snapshots, copy the exact four float64 input arrays, and seal an independent
input-byte audit. Decode old predictions only after the global46-call seal.
The independent auditor uses saved arrays and its own normalization/error
arithmetic, imports no integrator/model, and checks GL8 work accounting and
the local bound separately. Fix the known NumPy-boolean serialization issue
in the new auditor before freezing; keep the old failed source unchanged.

Only an accepted full synthetic PASS permits a separately frozen retry of
all72 real inputs under the same R33 precision/per-call/phase budgets and
fail-fast rule. The R33 failure remains retained. That later phase must not
be replaced by the successful prefix. Even a full numerical PASS leaves all
old scientific requirements in force: complete old-TRAIN screening,
center/region and perturbation regressions, actual V and GS tasks, and a
single untouched-data confirmation after the development gates pass.

The alternative radial-density reference and the 3D anchor constructions
remain unselected. This round changes reference computation only.
