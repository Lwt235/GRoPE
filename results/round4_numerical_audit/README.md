# Normalized Gaussian PE numerical audit

The frozen float64 helper was not modified. A separate, opt-in float32 path computes the same real-valued normalized expected-RoPE map and falls back per token when an explicit numerical predicate fails. This is an implementation audit, not efficacy evidence.

Real development tokens: 21,499; eligible fraction: 99.0325%. No task labels or held-out pair NPZs were read. Fixed untrained seed11 RGB features profile content. More details and every source/data hash are in summary.json.

| Workload | eligible | forward speedup | forward+backward speedup |
|---|---:|---:|---:|
| real_mixed_16x48 | 100.00% | 0.907x | 1.708x |
| fixed_query_16x48 | 100.00% | 0.906x | 1.624x |
| real_distributed_16x48 | 99.22% | 0.697x | 1.080x |
| extreme_fallback_16x48 | 0.00% | 0.563x | 0.792x |

Maximum observed forward absolute error: 3.23e-07; RMS: 2.07e-08. Content, mean, covariance and frequency gradients pass the declared float32 tolerances; exact values are in JSON.

Float32 outputs are not bitwise equal to float64. No covariance, relative Gaussian amplitude, or content is floored to make the fast path pass. Eligibility controls only numerical dispatch. It bounds sum-absolute quadratic terms by 32, sum-absolute phase terms by 64, content scale to [2^-32,2^32], and the observed normalized-filter conditioning by 16. Every other token is evaluated by the unchanged reference. Ordinary zero-block content gradients are retained.

CPU timings include runtime eligibility and mixed-token dispatch. The ordinary kernel timing without dispatch is listed separately when all tokens qualify. CPU load can influence wall-clock measurements; GPU timings are unmeasured, and CUDA synchronization in the dispatcher may remove the gain. Do not replace a frozen experiment implementation without rerunning its source audit and confirming numerical tolerance requirements.

Recommendation: do not automatically adopt the dispatcher for inference. Its ordinary float32 arithmetic kernel alone is about 2.4x faster in forward, but evaluating the safety predicate costs enough to make the measured full forward 0.56–0.91x the reference. Forward+backward is about 1.62–1.71x faster when all tokens qualify, only 1.08x with mixed real tokens, and slower when all tokens fall back. A caller that can safely precompute/reuse an unchanged eligibility decision may recover some forward gain; no such cache has been integrated or benchmarked.
