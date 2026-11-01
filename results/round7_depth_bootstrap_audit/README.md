# Independent depth bootstrap replay

The independent verifier passed 1,414 numerical checks with maximum absolute difference 6.59e-16. It imports no trainer/analyzer/model code: it sums the archived per-pair errors itself, takes square roots within each scan, then explicitly accumulates all 25 crossed seed/scan cells in each of 20,000 draws (RNG seed 20261030).

Every role/seed/scan metric, absolute and relative confidence interval, seed/scan effect, leave-one-scan sensitivity, raw descriptive baseline and primary criterion reproduces. The primary research gate remains **failed**. Five improving seed averages do not satisfy the missing practical-size and scan/seed confidence requirements.

Input JSON and confirmation-lock hashes, all independent effects and check counts are in `audit.json`. `source_snapshot.py` preserves the executed verifier. This verifies statistical arithmetic on the archived additive rows; the separate prediction audit is responsible for recomputing those rows from prediction arrays and sensor labels.
