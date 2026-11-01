# Independent sampled-probability arithmetic audit

Verified 30 complete reports and 450 sampled probability archives; 480 full-report pairs per run, 15 unique sampled pairs.

Counts match exactly. Floating additive sums are compared after dividing by the valid-row count, with tolerance 1e-05. Maximum errors: {'ce_sum': 1.914337126062135e-07, 'epe_argmax_sum': 6.166749458777527e-07, 'epe_soft_sum': 9.738541847556184e-06}.

This verifies arithmetic on the deterministic first three pairs of each scene, not every test prediction, sensor-label correctness, model selection, or the full confidence-interval analysis. No trainer/model metric function is imported. Every sampled archive and matched test report is SHA256 recorded.
