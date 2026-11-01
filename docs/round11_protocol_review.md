# Round 11 independent pre-execution review

2026-10-30. The reviewer inspected `ROUND11_PROTOCOL.md`,
`models/round11_stratum_pe.py`, `experiments/round11_stratum_probe.py`, and their
new model/gate tests before any Round11 sensor-task score was computed. The
forward-only compression diagnostic is already complete and explicitly remains
separate from task evidence. No frozen Round6–10 source was edited for this review.

The protocol and proposed implementation are consistent. The primary candidate
is fixed as `full4`. It retains quad4's four stratum means, weights, parent
content, key rectangles, common population and h/T budget, and adds the omitted
within-stratum position scatter to each component covariance. This is a precise
one-term representation change with no additional learned module. The model
does not compute an unused full-region covariance. Its mixture integrates over
actual key rectangles before normalization, using B's numerical integration and
background-density convention.

The global mixture mean/covariance conservation statement follows the total-
covariance identity and is exercised against independent rectangular pixel
slices. It does not imply exact query-partition invariance: each current query
region is split into four strata, so an 8px query uses 4px strata and a 16px query
uses 8px strata. This is different from retaining one universal atomization.
Mixture-before-normalization also retains the conditional-region semantics of B;
it is not the area average of individually normalized component rows.

Twenty-four new calibration configurations comprise twelve each for full4 and
iso4, on the same 48 old training pairs. The 32 old validation pairs remain
development data. Baseline calibration, selected configurations, metric rows,
numerical selection-guard repair and source hashes are inherited from completed
B. They are not silently recalibrated after the new candidate's scores. A
persisted selection is required before candidate gate evaluation. `iso4_matched`
uses full4's selected h/T for attribution, in addition to independently
calibrated iso4. Neither diagnostic can replace the predeclared full4 candidate.

The original point, strong quadrature, partition-specific, point-localization
and numerical gates are retained exactly. The reviewer suggested an additional
full1-improvement threshold; before any candidate task result, the root elected
to keep the unchanged gates and report full1 error/cost quantitatively instead.
That decision is defensible because repairing a failed localization/partition
guard can be useful even with similar macro KL. It is not a relaxation of any
previous acceptance criterion. A full4 pass still cannot erase the depth/RGB
failures or establish downstream benefit.

The timing code remeasures full4, iso4, quad16 and B full1 in the same process for
each fixed scene/condition, with three warmups and fifteen synchronized repeats.
Preparation, content pooling, score, softmax and common-grid reduction are all
included. Shared predictors/transfers are explicitly excluded. Earlier diagnostic
timings on sixteen queries are not used in the gate. Memory accounting describes
stored tensors, not allocator peak. The frozen B strong quadrature is quad16;
the runner rejects a different baseline selection rather than accidentally using
untimed evidence.

The read-only implementation review found no fatal issue. One additional cheap
guard was requested and implemented before freeze: explicitly join candidate and B gate rows by
scene/pair/condition and assert identical visible mass and point denominator,
beyond their shared hash/data provenance, rejecting duplicate identities. The new gate tests correctly reject
partition or PCK failures even when the macro gain is positive. Independent
saved-prediction, selection and metric audits are still required after execution;
this pre-execution review does not predeclare their outcome.

The implementation owner reported all eleven new model/runner tests and an
actual first-training-pair forward smoke check passed before freeze. This
review also inspected the added population-join code directly; no candidate
sensor-task score was used for the review.

The claim should remain an evidence-driven choice of support compression and
positional readout. [WorldRoPE](https://arxiv.org/html/2610.16337v1) already uses
multiple volumetric Gaussians per patch and integrates their positional effect.
Four Gaussian components becoming a PE is therefore not sufficient novelty by
itself. Here the potentially useful distinction is fixed observed spatial strata,
full local covariance retained without an extra learned geometry network, and
demonstrated accuracy/cost against strong empirical quadrature. Even if the next
development gate passes, that distinction still requires untouched confirmation
and an actual useful downstream task.
