# R23: real-input applicability and cost falsification, without labels

2026-10-31. Selected after R22's accepted fixed synthetic numerical pass.
R22 did not test arbitrary nonzero-mean fullrank Gaussians or realistic
432-key fields. This study asks only whether the unchanged implementation
can process a small, predetermined set of legal real inputs within its fixed
budget. It has no independent general probability reference and no efficacy
gate. Even all calls completing does not establish numerical accuracy or
authorize reading effect labels. The user’s full scientific objective remains
unachieved; all earlier failures and region/center/stress/V/GS/fresh gates stay.

## Exact population and permitted inputs

The sole population metadata is `results/round20_anchor_fixed/lock.json`.
For each old TRAIN scene c3, c5, c6, choose its first listed pair: [30,45],
`000_0000030_0000045`. The source archives are the original R10 `inputs.npz`,
under `results/round10_partition_data/train/<scene_slug>/<pair_name>/`.
Freeze their complete byte hashes and original manifest hashes before opening
arrays. `entries()` in `experiments/round23_ray_contract.py` supplies exact
paths/hashes and excludes labels, old probabilities and winner paths.

In each pair take native q8, q16 and q24 partitions, and eight evenly spaced
zero-based query ordinals by `linspace(0,N-1,8).astype(int64)`. This gives
72 unique queries, all retained. Respectively the ordinals are
`[0,61,123,184,246,307,369,431]`, `[0,15,30,45,61,76,91,107]`, and
`[0,6,13,20,26,33,40,47]`. Each query uses all 432 original g8 key cells.
Do not select using geometry, front/FOV flags, mass, labels, old success,
new exceptions or cost. No replacement of difficult rows is permitted.

Only six array fields may be accessed in the three original input archives:
`q_points3`, `q_pixel_cov3`, `q_K`, `k_K`, `q_to_k_transform`, and
`q_predicted_depth`. They contain archived RGB-predicted depth and given
cameras, not sensor supervision. No `labels.npz`, R13 targets, RGB/features,
old support/probability arrays, old VAL, V/GS data or fresh input is opened.
Other permitted NPZ reads are this study's own sealed input/output arrays.
Archive hashing reads bytes for integrity and does not inspect their values.

## Legal three-dimensional input construction

For a selected query, retain all n native source pixels without masking. Let
x_i be stored source-camera points. Let A=inv(q_K)[:,:2] and z_i=x_i,z;
the legal within-pixel factor is W_i=z_i A/sqrt(12). This uses the same
source-camera pinhole footprint as the original R10 archive, not an arbitrary
normal thickness. Compute source mean m and the factor block

    F = concat((x_i-m)/sqrt(n), W_i/sqrt(n)) over all selected pixels.

Compress F[3,3n] by reduced QR of F^T: if F^T=QR, use L_source=R^T[3,3].
There is no eigenvalue clipping, numerical rank threshold or learned fit.
The covariance represented by this factor must agree with the original
within-plus-between matrix

    C_source = mean(q_pixel_cov3_i) + mean((x_i-m)(x_i-m)^T).

With the given target transform (R,t), input `mu=R m+t`, `L=R L_source`.
The projection is `H=k_K[:2,:2]/8`, `c=k_K[:2,2]/8`, and each native g8
key rectangle is `(integer_pixel_bounds-.5)/8`. Keep h=.25 and T=0.
The image is the original 144x192. Strict Z>0, finite FOV, one area floor,
and all R22 numerical behavior remain exactly unchanged. Do not use old
transformed target-Z `.1m` projection clamp, image clipping, a source-center repair, new temperature,
extra mixture components, old 2D moments or a different integrator.

Prepare and seal all 72 factor/mean/camera/bounds input arrays before any
probability call. Include source/target moments, original archive identity,
pixel/query identity and preparation time. The independent input auditor
reconstructs sanitized predicted depth (`nan->10,+inf->20,-inf->.1`, clip
`.1..20`), backprojection, and within-pixel matrices from the allowed archive
fields. It uses the direct within-plus-between covariance formula and its
rigid transform, not production QR, to verify L L^T. Input comparison uses
max abs(actual-reference)/(1+abs(reference))<=1e-10. Bounds, case identities
and source/array hashes are checked exactly; full source metadata and every
input failure are retained. Input audit failure prevents probability calls.

Source predicted-depth sanitization above is the archived input convention; it
is distinct from the forbidden transformed target-Z projection clamp. Preserve
the supplied camera matrix literally. Record its orthogonality/determinant
residuals as metadata without imposing an extra orthogonality threshold.

The archive's earlier float64 computations and the factor construction may
differ by ordinary roundoff; the tolerance is declared before observing it.
The factor defines a PSD distribution even for a rank-deficient input. Report
matrix/factor residuals and do not silently fix them. Source-camera Gaussian
compression followed by projection differs from fitting a Gaussian after
per-pixel projection; neither equivalence to the old 2D closure nor preservation
of the R19 designated source center is assumed.

## Unchanged operator, finite resources and saved failures

Invoke exactly the accepted `models/round22_ray_pushforward.py`, with the
frozen R22 policy copied unchanged: depth extent14, geometry breakpoints,
GK21 limits, CDF/inner integration, 32,768 depth and 4,000,000 inner evaluation
caps per call. No numerical retry, alternate kernel, higher order, silent
underflow-to-zero repair or new case-specific path is allowed. Bind the R22
acceptance, locks and current source hashes before this study. Changing the
implementation would require a separately justified future iteration.

The complete 72-call phase has a 600-second wall budget, measured from case
processing start through the final per-case record write, including input
assembly from prepared arrays and per-case output bookkeeping. Preparation,
independent input audit and final source/inventory/sealing are timed separately.
A shared deadline is passed to every call. When a call fails or the deadline
expires, retain its status and continue recording the remaining fixed rows
without another attempt or a fallback probability. Final inventory must make
unattempted or incomplete work explicit; it can never be counted as success.

Save every successful raw cell mass, native mass sum, independent enclosing
FOV mass, P, positive-depth mass, floor, error estimates, tail bound, zero/tiny
flags, work counters and time. Save available partial arrays and the exact
failure reason for failed calls, as already provided by the unchanged model.
Record peak process RSS with its cumulative scope. Do not interpolate missing
arrays or discard far/behind/tiny cases. Capture outer engineering exceptions
and all files already written; fail rather than forge a complete population.

## Independent checks and interpretation

After all calls are sealed, independently check file/source inventories,
chronology, exact population and input identities. Successful outputs must
be finite/nonnegative, normalized within2e-12, use exactly one proper area
floor and the saved raw masses. Verify native raw sum vs the separately
integrated FOV within1e-8 absolute; verify analytic omitted-depth tail
amplification<=1e-8, positive-depth metadata, library status, unchanged work
caps and the 600-second phase limit. The independent R22 audit's algebra,
metadata and work-counter checks may be reused unchanged; no R22 canonical
reference function is executed on these general inputs. Shared algebraic
validation is explicitly not an independent integration reference.

The only possible positive status is `applicability_calls_completed`: every
fixed call completed and passed those consistency/resource checks. Always set
`general_numerical_accuracy_established=false`, `scientific_gate_passed=false`
and `labels_read=false`. There is no regional KL, center PCK, feature/task
training, method selection or efficacy label stage. Any input, execution,
consistency or resource failure stops this exact implementation from an effect
experiment. Report all failures and conditions, not a successful subset.

Even complete calls require a general independent numerical reference before
any future effect study. Conventional Jacobian 2D Gaussian and exact
positive-depth pushforward also differ in truncation semantics; multiplying
the former by p+ does not generally isolate nonlinear projection when image
coordinates and Z are correlated. No such comparison is executed here. A
future same-measure affine control must be separately defined, including its
undefined Jacobian at mean depth zero. None of these issues may be waived
to promote the current law to a validated scientific solution.

## Freeze and audit order

1. Complete separate toy tests without loading these real archives; freeze
   protocol, contract, all current/imported scientific sources, exact 72-case
   metadata, three input/manifest hashes, R22 acceptance and runtime. Independent
   audit freeze must precede all source-array access.
2. Prepare all legal inputs; seal them, then perform the independent input audit.
3. Only after that audit passes, execute and seal all bounded forward attempts.
4. Independently audit the saved outputs/statuses. Root verifies every inventory,
   current source/snapshot hash and chronology before accepting interpretation.

Keep NumPy first and OMP/MKL/OpenBLAS=4. No installations, GPU jobs, system
changes or new data acquisition. No post-result edits to these frozen sources.
