# Forward-only compression diagnostic

2026-10-30. A bounded diagnostic supports testing four fixed Gaussian strata
against the real task. It does **not** establish better sensor correspondence,
center localization, useful V aggregation, or generalization.

The frozen protocol used the first metadata-selected training pair from each of
Harvard c3/c5/c6, sixteen evenly spaced queries from each of g8/g16/mixed, all
432 native g8 target cells, bandwidths .25/.5/1/2, and no content score. Only
the predicted projected pixel means and within-pixel covariances were opened.
No sensor labels, validation inputs, or task metrics were read.

The reference integrates every predicted pixel Gaussian over actual key cells,
averages the raw component masses, adds the same declared 1e-30 background
density times key area, and only then normalizes over the image. It is a
reference for compression fidelity to predicted geometry, **not sensor truth**.
All methods consume the same complete predicted pixel field. `full4` divides a
region into fixed 2×2 strata and retains each stratum's full mean/covariance;
`quad4/16` retain only average within-pixel covariance at each stratum centroid.
`iso4` replaces each full4 covariance with its half-trace identity.

| Representation | Gaussian components/query | KL to dense reference | JS | L1 |
|---|---:|---:|---:|---:|
| full1 | 1 | 0.1165966 | 0.0070870 | 0.0537352 |
| quad4 | 4 | 0.0381092 | 0.0055246 | 0.0640709 |
| quad16 | 16 | 0.0096353 | 0.0024053 | 0.0191502 |
| full4 | 4 | **0.0004052** | **0.0000630** | **0.0045381** |
| iso4 | 4 | 0.0010770 | 0.0002494 | 0.0105160 |

These are equal scene/partition/bandwidth means of row errors. Full4 improves
all three metrics over full1, quad16, and iso4 in each of the three scans. It
also conserves the original full region's mean and covariance by the mixture
moment identity. Thus the difference from full1 is retained spatial structure,
not a changed global first or second moment. This does not by itself distinguish
multimodality from other departures from a single Gaussian.

Tail sensitivity matters: eight of 576 reference condition/query rows have raw
image mass below the protocol's 1e-20 diagnostic threshold. They dominate much
of full1's macro KL. The original complete-population result above is retained.
As a descriptive audit using that already frozen threshold, the remaining 568
rows have KL 0.0042231 for full1, 0.00048544 for quad16, 0.00014430 for full4,
and 0.00049761 for iso4. Corresponding full4/quad16 L1 values are
0.0039479/0.0107671. The fidelity advantage therefore is not exclusively a
tiny-FOV phenomenon; the large full-population percentage should not be used
without this qualification.

The exact incremental implementation averaged about 13.08 ms for full4 and
28.85 ms for quad16 per sixteen-query batch; preparation accounted for about
9.85/19.94 ms. Dense reference cost was about 81.72 ms and full1 about 4.66 ms.
These are one warmup and three synchronized repeats on GPU0, with Python-loop
component preparation, and a sum of separately measured preparation/integration
medians. They are diagnostic costs, **not** the full-bank paired timing required
by a research promotion gate. A vectorized implementation needs fresh matched
timings for every competitor.

Eight synthetic tests passed, including full-mixture moment conservation,
quadrature's omitted scatter, the distinction between pre/post-mixture
normalization, and SciPy discrepancy checks. An independent auditor reconstructed
all 2,880 method/query metric rows from 36 saved probability archives using
SciPy entropy; maximum metric disagreement was 7.11e-15. Source and input hashes
matched, and the GPU job ended normally after 16.41 seconds. The independent
audit checked saved probabilities and metrics, not a second integration engine;
the shared integration implementation has its separate Round10 audits.

Artifacts:

- [Frozen protocol](../results/round11_compression_diagnostic/protocol.json)
- [Complete summary](../results/round11_compression_diagnostic/summary.json)
- [Per-query rows](../results/round11_compression_diagnostic/rows.json)
- [Component timings](../results/round11_compression_diagnostic/timing.json)
- [Independent verification and per-scan/FOV sensitivities](../results/round11_compression_audit/verification.json)

The justified next step is a newly frozen old-development experiment testing
full4 on independent sensor-derived targets, with the unchanged strong point,
quadrature, partition-specific, localization, and matched-cost requirements.
No confirmation cohort or previous negative result changes status here.
