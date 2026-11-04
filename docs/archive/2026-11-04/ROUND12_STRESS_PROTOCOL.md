# Round12: locked scale and controlled geometry stress regression

2026-10-30. This protocol is frozen before any new scale/stress score. It implements gate B in `docs/round11_validation_inventory.md`, preserves the primary full4 method and all thresholds, and does not authorize fresh confirmation or depth training. Frozen R10/R11 code is imported read-only. Only new adapter, runner and tests are owned by this branch.

## Cohort, controls and numerical prerequisites

Use the exact 32 old R10 validation pairs: 16 Harvard c8 and 16 Brown bm4. All 192×144 query pixels participate in forward support, regardless of sensor visibility or predicted in-front status. No new calibration or pair filtering. The 48 old train pairs supply only existing frozen selections; optional metadata-first forward numerical smoke may use the first old train pair without opening labels.

Roles and frozen h/T are full4 (.5/5), iso4 (.5/5), full1 (.25/0), point (1/5), point_pck (.5/5), fixed (.5/5), quad4 (.5/5), quad16 (.5/5). full4 and iso4 inherit R11 selection; other roles inherit B selection (`full1` maps to B `full`). point and quad16 are B's preselected strong controls. No role/configuration is selected on these new conditions. iso4 retains its separately selected R11 configuration; it equals the matched-h/T diagnostic in this frozen case.

Means, covariances, four-stratum partition and features are exactly the frozen R11/B definitions. full4 uses four 2×2 equal-area strata with local pixel covariance plus projected-position scatter. iso4 replaces each covariance by its same-trace isotropic matrix. quad4/16 retain only their mean within-pixel covariance. All parents retain the same pooled frozen base8 content. The constant background density is 1e-30 integrated over native cell area in reference8 units, added once per native key after all subrectangle and mixture sums. No visibility mask, extra learned parameter or extra content token is introduced.

The existing fast GL32 operator accepts key rectangle sides at most 16px and h≥.25. A 24×24 native cell is deterministically split into four 12×12 rectangles, integrated and summed before mixture weighting/log. This is an integration subdivision only. Physical edges remain `(integer_index-.5)/8`. Full float64 probability arrays are retained. Before freeze, independent SciPy bivariate CDF and robust conditional-integral tests cover 24px rectangles at all four inherited bandwidths, including zero/small/large correlated covariance; gradient, area coverage, fixed-stratum moment conservation, old common-grid equivalence, perturbation and gate-rejection tests must pass. These tests contain no task labels.

## Scale conditions and labels

Fixed five conditions are 8→24, 24→8, 16→24, 24→16, 24→24. g24 exactly covers the image with 48 native tokens. Its content is the normalized mean of the corresponding 3×3 base8 features. full4 has four 12×12 strata; quad16 has sixteen 6×6 strata. There is no valid common16 parent for g24; the implementation records parent=-1 and rejects common reconstruction for that partition.

Each condition uses its own direct hard-native-cell sensor histogram. Every visible dense sensor-projected query pixel contributes unit mass. Query center labels for 8/16 retain the R10 independent sensor labels. For 24, reconstruct original RGB-to-sensor timestamp association from the frozen selection and original index files, sample original query sensor depth at the native 24px cell center, and project with original camera intrinsics/poses into the original key sensor. Apply the same .05m + .05×depth consistency tolerance and native half-pixel convention. A mean projected region point is never used as a center label. This label producer runs after all forward archives complete.

The direct KL for a condition/scene is total visible-mass-weighted KL divided by total visible pixel count across its 16 pairs. Five conditions and two scenes are then equally weighted. Different native key class counts are not interpreted as comparable task difficulty. Pass requires all:

- full4 macro KL at least 2% lower than the frozen point role;
- both scene macro KL values lower than point;
- every condition macro KL no more than 2% worse than frozen quad16;
- 24→8 center PCK@8 no more than 0.5 percentage points below frozen point_pck (equal-scene averaging, sensor-valid center population).

Report every role, scene and condition, including fixed/full1/iso4. Target stride24 quantization does not waive the 24→8 point guard.

## Controlled geometry perturbations

Seven states include zero and six separate, uncombined changes:

1. Predicted depth ×0.9 for both query and key.
2. Predicted depth ×1.1 for both query and key.
3. Target-camera rotation Ry(+1°) left-multiplied into the query-to-key transform.
4. The same Ry(−1°).
5. Add +0.01m to target-x relative translation.
6. Add −0.01m to target-x relative translation.

Depth scaling is applied to both effective RGB-predicted depth caches after the original fixed nan/inf and [.1,20]m clipping, with no post-scaling reclipping. Query 3D points scale by α and within-pixel 3D covariance by α² before the transform and perspective Jacobian. Record both effective-depth hashes. Key predicted depth is consistently scaled but is not consumed by this particular positional operator: its key geometry is the native image cell; the target camera K is used for projection. This limitation is explicit and is not claimed as two-sided depth sensitivity. Rotation left-multiplies both R and t; x translation is in target-camera meters. Frozen z/xy/eigenvalue finite fallback is retained, without discarding affected rows.

Each state runs all original g8/g16/mixed nine conditions, every control receiving the same changed forward geometry and unchanged frozen h/T. Native probabilities are lifted by uniform key-cell density to base8 atoms and source-area averaged to common16 query cells, using the exact R10 linear map. Sensor labels, evaluation camera and visible population remain the original frozen values. Direct probabilities and hard argmax are stored; common probabilities are derived rather than duplicated on disk.

For each of the six perturbations, full4 equal-scene/equal-condition common KL must be no more than 2% worse than same-state quad16, and g8→g8 PCK@8 must be no more than 0.5pp below same-state point_pck. All probabilities must be finite and normalized. Zero is additionally checked with these same bounds and reported. Preserve each role's absolute score and full4 absolute changes from zero. The perturbations are controlled bias sensitivity checks, not a realistic noise distribution or a complete robustness claim. No per-state threshold or parameter adjustment is allowed.

## Stages, freeze and audit

`freeze` verifies R10 input/metadata hashes, frozen B/R11 selection/source/completion chains, source/tests/protocol and new partitions. It writes lock.json and lock.sha256 before new forward work. Raw sensor files are not decoded during freeze or forward; their existing expected hashes are retained in the source manifests. Original selection/camera/index metadata may be read before sensor decode.

`forward --scene` accepts only the two old validation scenes. For all 16 pairs it finishes zero plus all six perturbations and the five scale conditions before writing that scene completion. It saves compressed float64 native probability arrays and int32 native argmax, numerical underflow/tiny-FOV diagnostics, projected-geometry/effective-depth hashes, identity/count metadata and output hashes. It does not save redundant dense geometry or common probabilities and does not open old or new supervision archives. All output rows, including tiny-FOV/underflow rows, remain in the population.

`evaluate --scene` first verifies **both scenes' entire forward completion and output hashes**. Only then does it decode old labels and original sensor depth to produce the q24 center labels. Save the complete labels, raw per-role/pair/condition metrics and label provenance; verify exact population equality. `finalize` verifies every stage, chronology, source, source input and saved artifact, recomputes macro summaries and fixed gate conjunction, then writes summary and full completion hashes. Total is 32×8×(5+7×9)=17,408 probability/metric records. No file overwrite is allowed. Partial failed stages require explicit investigation and a documented new output location, not silent resume or deletion.

Independent review verifies 24px numerical boundaries, raw-sensor center sampling, all probabilities/argmax/populations and summary/gate recomputation. Any gate failure is retained and reported promptly; remaining fixed experiments still finish to expose the failure pattern. This branch does not modify historic R10/R11 decisions. Passing this branch is a requirement for the subsequent depth branch but does not itself pass depth, optimized primitive R/S or RGB/NVS checks. Final claim remains limited to old-development region matching and these controlled conditions.
