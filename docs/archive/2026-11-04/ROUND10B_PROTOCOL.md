# Round 10 B — remove the native-key Gaussian approximation

This is the predeclared reserve in ROUND10_PROTOCOL, instantiated after A failed
its strong-control, partition, and localization gates. A achieved19.18% lower
common-grid KL than the calibrated point baseline but was1.247% worse than
quad16, only1.50x faster in this prototype, and lost0.613pp PCK versus the
separately calibrated point control. Its results remain frozen.

The forward-only diagnostic `results/round10_key_partition_diagnostic` establishes
that representing known rectangular keys as Gaussians changes predicted mass
when only the key partition changes: mean L1=.0166 on three first calibration
pairs after aggregation to the same keys. Actual rectangle integration removes
this defect numerically. This justifies testing the reserve; it does not show
that the defect caused A's task failure. A's coarse-query losses also suggest
information discarded by the query Gaussian, which B cannot repair.

## Frozen B operator and numerical implementation

For each key rectangle Rk, positional mass is

    w(q,k) = integral_Rk N(x; mu_q, C_q + h^2 I) dx.

Bias is log(w); no extra key-area term because integration already includes
area. Use `models/round10_cell_integral_fast.py`, fixed32-point positive
Gauss-Legendre conditional integration, float64 arithmetic, maximum native cell
side2 in units of8pixels, h>=.25. Query/key chunks256/432 affect memory only.
Point and isotropic no-grad controls use exact separable CDF products. Query
moments/content remain the same float32 preparation as A. Record precision and
count raw whole-FOV masses below1e-20; no row may be removed.

Add a fixed background density1e-30 per square reference unit before log:
`mass + 1e-30 * pixel_area/64`. This preserves additivity across key subdivision
at T=0 (up to integration error), provides a finite shared fallback even if every
mass underflows, and applies identically to all methods. It is numerical policy,
not learned confidence. Arbitrary pooled content still changes under partition;
no full-network exact invariance is claimed.

## Strong controls and unchanged experiment

Use all A data, actual partitions, common populations, h/T grids, direct/common
metrics, per-scan/condition reporting, PCK protection, selection order and gates.
Controls: point and center both set query covariance to0 while retaining actual
key rectangles (they are mathematically identical and reported as an equivalence,
not two independent competitors); fixed uses native query area/(64*12) times I;
isotropic preserves query trace; full uses query covariance. quad4/16 average
the same rectangle integrals of actual stratum centroids/within-pixel covariance,
so B is not compared only against A's approximate key model. Rotated90 is a
diagnostic with full's selected h/T. Exactly the original calibration grid, no
extra fit parameters or learned networks.

Freeze all source/data hashes before B calibration. Parallel calibration jobs
may each read one old training scan. Collect them, select globally and persist
selection+SHA before any B gate job opens labels. Two gate jobs may each read
one old validation scan. Finalize only when all declared rows/archives exist;
run the same first-pair timing with3 warmups/15 synchronized repeats on an idle
GPU, report incremental PE cost and stored tensor bytes. Time the actual same
precision used for each operator; do not time a cheaper numerical substitute.

The A gate definitions apply without relaxation: >=2% versus the calibration-
selected point/center with both scans improving; accuracy/cost gate against
calibration-selected quad4/16 (>=2% lower KL and<=1.25x cost, OR <=1% worse KL and
>=2x faster); no condition >2% worse than that quadrature; PCK loss<=.5pp; all
integrity/numerical audits pass. Also disclose A/B and gate-envelope comparisons.

All data are reused development data. A B pass would be only permission to
design a new frozen confirmation, not evidence that earlier depth/RGB utility
checks passed. A B failure rules out this one-Gaussian region-compression branch
under the present requirements. Further designs require a new falsifiable
development protocol and must preserve old failures and untouched confirmation
data; changing a criterion after seeing it fail cannot satisfy the original goal.
