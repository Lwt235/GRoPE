# R32: representable initial depth partitions

2026-10-31. Selected after the accepted R31 synthetic numerical failure.
R31 remains FAIL: 44/46 calls completed and 148/155 comparisons passed.
Its authenticated post-failure diagnostic found two adjacent-float pairs
among the initial geometry roots of 07_radial_noncentral. Neither interval
has a representable strict midpoint. No integration was run in that diagnosis.

Make one reference-only change to R31. Preserve the complete physical depth
interval and its endpoint float64 bits. Walk the sorted, strictly interior
geometry cuts from left to right, keeping a candidate p only if the identical
pilot expression (previous+p)/2 lies strictly between previous and p.
Otherwise omit that candidate partition hint. Before adding the upper
endpoint, repeatedly remove the last retained interior cut if its midpoint
with upper is not strict. If the physical endpoints themselves have no
representable strict midpoint, fail without moving them. Reject nonfinite,
unordered, duplicate or out-of-domain input cuts. No epsilon, nextafter shift,
rank repair, probability cutoff or new tail exemption is introduced.

Save original depth edges before coalescing, the retained initial edges, and
the original index, value and reason for every omitted cut, including failure
evidence. Pilots and quad_vec use exactly the same retained partition. The
512 initial-interval admission limit applies to retained intervals; separately
record the original and retained cut counts. The existing adaptive interval
limit is unchanged. Coalescing removes a boundary, not any part of the
integration domain. It does not establish that the merged neighborhood has
negligible probability. Pilot scales and adaptive work may change, and the
pilot remains an estimate rather than a global bound.

All other R31 reference arithmetic is unchanged: independent SVD preparation,
complete finite X integration with width-2 panels, log normal intervals and
positive sums, strict positive depth intersected with [-18,18], fixed pilot
scales, GL32/48 and adaptive GK21, the physical depth-tail bounds, once-only
area floor and content normalization. Preserve all original raw/P/double-order
gates, 250M nodes per call, 600M nodes for the 46-call phase, 1800s phase,
32768 depth evaluations and 512 outer intervals. Pilot work is charged.

Re-freeze all sources and runtime. Reuse exactly the same 23 R29 base input
descriptors and four canonical input arrays. Execute all 46 new references,
not only the previously failed example. Preserve the 155 comparisons:
23 new-order comparisons, 86 against the accepted 43 old references and 46
against unchanged R29 production outputs. Do not rerun production. Preserve
all failed and unattempted outcomes. No new R31 bitwise-output gate is added.
The six identities and finite-epsilon test remain inherited R29 evidence.

Before formal freeze, test the coalescer with safe partitions, adjacent and
two-ULP cuts, close chains, lower/upper endpoint adjacency, upper backtracking,
negative and subnormal/cross-zero values, invalid cuts and an impossible
whole-domain midpoint. Independent audit reconstructs the retained partition
and verifies endpoint coverage, omissions and pilot consistency without
importing reference or production arithmetic. Each development test attempt
retains contemporaneous source snapshots and its complete log.

This is a synthetic reference revision only. No real input arrays, labels,
new model, anchor/content selection or efficacy validation is authorized by
this round. A numerical pass can permit a separately frozen check of all
72 R30 real inputs; it cannot establish scientific success or finish the goal.
