# Round 11: area and center supervision are usually compatible on old training data

2026-10-30. The independent diagnostic finds a small set of genuine target
conflicts, but **does not support treating the existing PCK protection gate as
impossible or relaxing it**. It accessed only the 48 fixed old-training label
archives, after freezing its own source and protocol. It read no candidate
score, predicted geometry, sensor image, validation label or sealed test.

The source/protocol lock is
`88f7bc9e10e33d5f446705f6464c4261d20fafef8e473758d854cd2a129c9915`.
All 20,736 g8 query rows are retained in
[rows.json](../results/round11_target_conflict/rows.json). There are 12,049 queries
with nonempty visible area histograms and 9,521 with valid independent center
targets. Every center-valid query also has a nonempty histogram. There are
2,528 area-valid/center-invalid queries and 8,687 with neither target valid.

Among the 9,521 joint-valid queries, using the sensor area histogram itself:

| Target diagnostic | Result |
|---|---:|
| First-native-index area-argmax PCK@8 | 99.3488% |
| Some tied area maximum within 8 pixels | 99.3908% |
| First-argmax conflicts | 62 / 9,521 (0.6512%) |
| Conflicts persisting under optimistic ties | 58 / 9,521 (0.6092%) |
| Conflicts due only to deterministic tie choice | 4 |
| Mean minimum direct-g8 KL permitting an allowed top tie | 0.00006937 nats/query |
| Visible-source-mass-weighted mean minimum KL | 0.00005958 nats |
| Maximum minimum KL among queries | 0.08647 nats |

Scene-equal averages give first-argmax PCK 99.3351%, optimistic PCK 99.3807%,
and mean minimum KL 0.00007145 nats/query. The complete counts, per-scene values,
quantiles, exact label hashes and access scope are in
[summary.json](../results/round11_target_conflict/summary.json).

The constraint allows ties: it asks whether any key whose center is within
eight pixels can be a largest probability. It therefore gives a relaxed
minimum cost and does not guarantee that a deterministic first-argmax decoder
chooses that key. The standalone projection was verified against independent
SciPy SLSQP optimization on synthetic distributions, with separate analytical
zero-mass and tie cases; nine tests passed. SciPy emitted its standard internal
bound-clipping warning during an optimizer step; all reference optimizations
converged and matched the analytical projection tolerance.

The distribution here is the **direct g8→g8 sensor area histogram**. Its KL is
not the source-area-pooled g16→g8 common-grid metric used for the main research
gate, and these old-training statistics cannot explain a particular validation
method's PCK loss. Similar percentages would not establish that they concern
the same queries or failure mechanism.

The practical implication is limited: exact regional mass and independent
center-point argmax can conflict, as the theoretical counterexample predicts,
but the observed conflicts are rare and their average direct-regional KL cost
is small. There remains substantial mathematical room for both targets to be
served well on this population. This diagnostic is neither a deployable oracle
nor a method result, and it changes no candidate, baseline, mask or acceptance
threshold. The original PCK protection condition remains in force.

Reproduce with a new output directory:

```bash
/home/liwt/miniColmap/.venv/bin/python -m experiments.round11_target_conflict freeze --out results/round11_target_conflict_replay
/home/liwt/miniColmap/.venv/bin/python -m experiments.round11_target_conflict run --out results/round11_target_conflict_replay
```

The first command must complete before the second is launched. Existing
diagnostic outputs are never overwritten.
