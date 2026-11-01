# Round15 Stage B: complete old-training regional calibration

This implements Stage B of the already frozen ROUND15_CANDIDATE_PROTOCOL.md.
No scientific threshold, center constraint, configuration grid or objective is
changed. This file and its producer/tests must be frozen before any Stage B
forward. Stage A's independently audited decision must have action
`complete_full_regional_calibration`, a nonempty eligible set and valid hashes
binding its source, complete forward, summary, completion and independent audit.
An empty set, failed audit or missing decision refuses Stage B before creating
an output directory. Implementation alone does not authorize execution.

Use all 48 old-training pairs, 16 in each of Harvard c3/c5/c6. No old-validation,
sealed-test or new-scene observations or labels. Eight families, in fixed order:
quartic_full, quartic_point, gaussian_point, gaussian_fixed, gaussian_full1,
gaussian_full4, gaussian_quad4, gaussian_quad16. Every family receives all 12
configurations: h=.25,.5,1,2 outer; T=0,5,10 inner. Every configuration includes
all nine query/key pairs in g8/g16/mixed and the five scale pairs
g8→g24, g24→g8, g16→g24, g24→g16, g24→g24. This is 64,512 pair-level probability
records. All quartic_full configurations, including Stage A-ineligible ones,
are included. There is no early stopping within Stage B.

The forward loader decodes only q_to_k_xy, q_to_k_pixel_cov, q_features and
k_features. It receives no targets. Use frozen R12 native partitions, query
preparation, content pooling and Gaussian cell integration. Gaussian full4
retains four local stratum covariances, quad4/16 retain only within-pixel
covariance at the equal-area stratum means, fixed uses query area/(64*12)I,
and point uses zero query covariance. Point/fixed preparation does not compute
unused global scatter. All means, covariance preparation and content dots
retain inherited float32 behavior; integrations and probabilities are float64.

Quartic uses explicit GL24, query chunk16/key chunk32 in the frozen tiled
operator. It receives **original native key bounds** `(bounds-.5)/8`, then
internally subdivides to sides<=1. It must not consume R12's Gaussian
`cell_bounds`, which subdivide native24 keys into four12px integration cells.
Gaussian retains those frozen R12 child bounds and native-key ownership.
Both families sum child masses before adding exactly one native-area density
floor 1e-30*key_area/64, then add the same T*parent-content score and normalize
over the original native keys. Numerical subdivisions never become tokens.
The GL24 source/numerical/cost chain and failed GL16 history are inherited
through Stage A and remain immutable.

Execution is fixed to CPU, four PyTorch threads, deterministic algorithms,
disabled TF32, frozen torch/NumPy versions and torch build configuration.
Different scenes can run in separate processes. No device or chunk change
between passes, approximate hash comparison or probability rounding.

Pass 1 is forward-only for all 48 pairs and every declared configuration and
condition. Persist every probability's shape, dtype, canonical hash (dtype,
shape and contiguous bytes), input hash, identity and raw-mass diagnostics.
Compute each geometry once for its three temperatures. On each scene's first
metadata pair, recompute preparation and every probability in the same process and require exact
hash equality. Preserve that pair's complete native probabilities as a fixed
audit sample: 3 pairs ×512,434,944 bytes =1,537,304,832 raw bytes total. Other
arrays are deliberately discarded after hashing. Saving all 48 would require
24,596,877,312 raw bytes; do not claim that these arrays remain on disk.
Samples are chosen by metadata before results and cover all 14 conditions,
8 families and 12 configurations. No Stage A probability reuse in this version.
All 5,184 overlapping Stage A probabilities are independently recomputed and
must match the already audited Stage A array/input hashes exactly; this binds
the inherited center-eligible configurations to this same regional operator.

All three forward shards, their exact identity populations, repeated-hash
checks and fixed saved samples must pass verification before a global
forward_complete.json is written. No Stage B regional label file is hashed
or decoded before this global seal. Stage A already used training-center
supervision; this ordering explicitly concerns new Stage B regional metrics.

Pass 2 regenerates each probability using the same frozen forward-only
function and input fields. It must match that record's pass1 hash exactly
before its target metrics are computed. A mismatch stops the run, leaves
the failed partial evidence in place and forbids selection or promotion.
There is no rounding fallback or dropping of an affected pair. The saved
sample must also agree by canonical hash. Label loading remains outside the
forward function. Raw labels are only the old R10 gt_xy and gt_visible arrays;
their expected hashes come from the frozen training manifests. No new sensor
decoding, visibility inference or center target reconstruction is needed.

For each condition reconstruct integer source-pixel target counts exactly
as frozen R12 target_counts: native query owner and native key rectangle
containment. Common targets are independent fixed g16 queries→g8 keys, with
the same visible source-pixel mass. For the original nine conditions, lift
the native prediction uniformly by key area to g8 atoms, then source-area
average into common g16 queries. This transformation receives no label counts
or visibility. For the five scale conditions use native probabilities;
q24 has no common16 parent and must not enter common aggregation. Retain both
primary and direct-native KL sums, JS sums and mass for all conditions.
Use the unchanged inherited metric floor1e-15, distinct from the forward
density floor. Invalid/zero-visible-mass queries have zero target weight,
not a new filter; scene/condition total mass must be positive. Report all
denominators and exact source/hash identities. Every configuration must have
the same label population; every partition must preserve visible pixel mass.

For each family/configuration, first compute mass-weighted KL within each
scene/condition, then equal-scene/equal-condition means separately for the
nine common conditions and five direct conditions. The fixed objective is

    J = .5 * mean_scene(mean_9(common KL))
      + .5 * mean_scene(mean_5(direct KL)).

Do not pool all14 conditions or reweight by scene/pair count. Preserve both
terms and all scene/condition values. Stage A's exact-rational eligibility and
two globally selected localization references remain locked; do not reselect
them using J. Primary selection is minimum J among the audited eligible
quartic_full indices, ties by grid order. Report its unconstrained minimum
descriptively. Each of the seven control families independently selects its
unconstrained minimum J over all12 indices, including center-poor controls.
Strong regional point is the lower-J selected Gaussian/quartic point (tie:
Gaussian first); strong quadrature is selected quad4/quad16 (tie:quad4 first).
The inherited B/R11/R12 point, point_pck, fixed, full1, full4, quad4 and quad16
configurations are identified and reported from their exact entries in this
same grid; they add no new search. Legacy iso4 remains archived as an earlier
attribution diagnostic and is not an added ninth calibrated family.

Producer summary contains a proposed selection, never a passed scientific
gate. A separate independent audit must replay the fixed raw sample, verify
hash commitments and labels, reproduce all metric populations, aggregate all
64,512 rows and independently recover selection/strong controls. Before any
validation phase, accept only a passed audit bound to lock, forward seal,
summary and completion hashes; write the selected primary/control configuration
artifact and its hash then. No Stage B result automatically passes the nine
partition, five scale, six stress, timing, actual-value or actual-GS/RGB gates.
Old validation is still development regression, not fresh confirmation.

Outputs are append-only within a new root. Source snapshots, upstream Stage A
decision/audit chain, legacy metadata, all manifests, forward/evaluation shard
completions, labels' expected/observed hashes, fixed samples, metrics and final
audit decision must remain inspectable. A partial existing shard cannot be
overwritten or silently resumed. Any implementation/numerical repair requires
a separate reviewed namespace; frozen Stage A and historical evidence remain
unchanged.
