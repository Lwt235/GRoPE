# Actual 48D normalized PE: primitive covariance attribution

The actual 48D candidate does not show an overall benefit from internal primitive covariance on these old camera pairs. Both-side center-only covariance lowers pooled CE in all five seeds: full-minus-center is +0.004614 [0.002226,0.007322] for primary and +0.004726 [0.002362,0.007659] for swapped parity. The external two-scene effects are +0.005500 and +0.005795, also favoring center-only. Temple has a very small opposite effect (primary -0.000236; swapped-parity interval crosses zero), so the conclusion is not that RS is useless in every scene or task. It is that current regional-support gains cannot be attributed to an independently beneficial internal GS RS component.

Five actual raw-RGB normalized_shape checkpoints execute their archived model implementation. Fitted query support is preserved, matching training. No fitting or temperature calibration occurs. Full and center-only inputs retain identical means/phases, features, labels, masks, resolution and checkpoint parameters.

This descriptive repeat uses the OLD decomposition camera pairs, not the 203 prospective fresh pairs. Both-side removal is the primary attribution; one-sided changes can misalign the still-fitted opposite spectral filter.

| parity | scene | intervention | full CE | changed CE | full−changed CE | seed/camera 95% CI |
|---|---|---|---:|---:|---:|---|
| 0 | teatime | key_centeronly | 1.215561 | 1.213945 | +0.001615 | [+0.000015,+0.005235] |
| 0 | teatime | query_centeronly | 1.215561 | 1.215558 | +0.000002 | [-0.000211,+0.000216] |
| 0 | teatime | both_centeronly | 1.215561 | 1.213947 | +0.001613 | [+0.000039,+0.005224] |
| 0 | waldo_kitchen | key_centeronly | 1.001501 | 1.000736 | +0.000765 | [+0.000385,+0.001262] |
| 0 | waldo_kitchen | query_centeronly | 1.001501 | 1.001325 | +0.000176 | [-0.000074,+0.000475] |
| 0 | waldo_kitchen | both_centeronly | 1.001501 | 1.000589 | +0.000912 | [+0.000312,+0.001658] |
| 0 | figurines | key_centeronly | 1.306664 | 1.301394 | +0.005270 | [+0.002067,+0.008652] |
| 0 | figurines | query_centeronly | 1.306664 | 1.301846 | +0.004818 | [+0.001009,+0.009707] |
| 0 | figurines | both_centeronly | 1.306664 | 1.297133 | +0.009531 | [+0.004531,+0.015327] |
| 0 | temple | key_centeronly | 1.028386 | 1.028489 | -0.000103 | [-0.000230,+0.000015] |
| 0 | temple | query_centeronly | 1.028386 | 1.028517 | -0.000131 | [-0.000266,-0.000012] |
| 0 | temple | both_centeronly | 1.028386 | 1.028623 | -0.000236 | [-0.000468,-0.000021] |
| 0 | external | key_centeronly | 1.191805 | 1.188753 | +0.003052 | [+0.001171,+0.005076] |
| 0 | external | query_centeronly | 1.191805 | 1.189030 | +0.002775 | [+0.000534,+0.005566] |
| 0 | external | both_centeronly | 1.191805 | 1.186305 | +0.005500 | [+0.002522,+0.008879] |
| 0 | all | key_centeronly | 1.174105 | 1.171443 | +0.002662 | [+0.001132,+0.004283] |
| 0 | all | query_centeronly | 1.174105 | 1.171891 | +0.002214 | [+0.000441,+0.004412] |
| 0 | all | both_centeronly | 1.174105 | 1.169492 | +0.004614 | [+0.002226,+0.007322] |
| 1 | teatime | key_centeronly | 1.210480 | 1.210111 | +0.000369 | [+0.000001,+0.000780] |
| 1 | teatime | query_centeronly | 1.210480 | 1.210268 | +0.000212 | [-0.000212,+0.000979] |
| 1 | teatime | both_centeronly | 1.210480 | 1.209926 | +0.000555 | [-0.000137,+0.001485] |
| 1 | waldo_kitchen | key_centeronly | 1.012655 | 1.011905 | +0.000749 | [+0.000324,+0.001311] |
| 1 | waldo_kitchen | query_centeronly | 1.012655 | 1.012544 | +0.000111 | [-0.000118,+0.000328] |
| 1 | waldo_kitchen | both_centeronly | 1.012655 | 1.011819 | +0.000835 | [+0.000305,+0.001487] |
| 1 | figurines | key_centeronly | 1.290656 | 1.283978 | +0.006679 | [+0.002656,+0.011844] |
| 1 | figurines | query_centeronly | 1.290656 | 1.287394 | +0.003262 | [+0.000643,+0.006278] |
| 1 | figurines | both_centeronly | 1.290656 | 1.280667 | +0.009990 | [+0.004972,+0.015991] |
| 1 | temple | key_centeronly | 1.027227 | 1.027396 | -0.000169 | [-0.000318,-0.000032] |
| 1 | temple | query_centeronly | 1.027227 | 1.027225 | +0.000002 | [-0.000099,+0.000092] |
| 1 | temple | both_centeronly | 1.027227 | 1.027396 | -0.000169 | [-0.000404,+0.000047] |
| 1 | external | key_centeronly | 1.181877 | 1.178026 | +0.003851 | [+0.001466,+0.006967] |
| 1 | external | query_centeronly | 1.181877 | 1.179961 | +0.001916 | [+0.000385,+0.003760] |
| 1 | external | both_centeronly | 1.181877 | 1.176082 | +0.005795 | [+0.002804,+0.009442] |
| 1 | all | key_centeronly | 1.166809 | 1.163647 | +0.003161 | [+0.001266,+0.005633] |
| 1 | all | query_centeronly | 1.166809 | 1.165261 | +0.001548 | [+0.000325,+0.003018] |
| 1 | all | both_centeronly | 1.166809 | 1.162083 | +0.004726 | [+0.002362,+0.007659] |

Negative full-minus-center CE favors retaining weighted internal primitive covariance; positive favors removing it. These frozen interventions are not retrained structural comparisons. Their scene-stratified camera intervals concern these reconstructed scenes, not a population of scenes. A positive old isometric/world-kernel result cannot substitute for this actual normalized-PE attribution, and this audit does not prove 3DGS-unique representational value over all point-cloud estimators.

Selected validation digests and metrics are replayed. The old camera pairs had no historical test probability archive for these development-trained checkpoints, so full-test verification is independent repeated-forward equality plus archived implementation/checkpoint hashes. Probabilities, row metrics, every decomposition/original NPZ hash and projection replay errors are retained.

Projection validation uses the original fitter’s float64 full covariance from the decomposition archive. Casting a large radial world covariance to float32 can amplify cancellation when projecting: two Figurines query pairs show 2.15e-4 relative error with serialized world C, versus 2.19e-6 with full64 C at the unchanged serialized mean. The audit retained its 1e-4 check and the same center-only intervention definition; full64 projection errors over all pairs remain below 3.53e-6. This precision issue and independent probability arithmetic/hash checks are documented in verification.json.
