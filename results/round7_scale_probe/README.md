# Round7 24px extension diagnostic

diagnostic extension of already used two validation and five Round6 test scan/place groups; NOT a new independent confirmation

The five Round6 role configurations were frozen unchanged. All first64 validation/96 old-test pairs and all five new grids are included. No future Round7 test was opened.

Source/RGB/cache/depth hashes, input/label digests, all per-pair role rows, supervision reasons, and quantization ceilings are saved. Features24 use3x3 pooled frozen features; no fitting was performed.

|split|grid|full KL|fixed KL|relative full−fixed|improving places|
|---|---|---:|---:|---:|---:|
|val|q8_k24|0.206135|0.227782|-9.503%|2/2|
|val|q24_k8|0.393996|0.531788|-25.911%|2/2|
|val|q16_k24|0.159864|0.209442|-23.672%|2/2|
|val|q24_k16|0.246546|0.348709|-29.298%|2/2|
|val|q24_k24|0.177394|0.246942|-28.164%|2/2|
|val|all_new_grids|0.236787|0.312933|-24.333%|2/2|
|test|q8_k24|0.172100|0.175295|-1.823%|5/5|
|test|q24_k8|0.336197|0.384354|-12.529%|5/5|
|test|q16_k24|0.129578|0.140950|-8.069%|5/5|
|test|q24_k16|0.207345|0.243557|-14.868%|5/5|
|test|q24_k24|0.147901|0.172859|-14.439%|5/5|
|test|all_new_grids|0.198624|0.223403|-11.092%|5/5|

PCK at8/4px changes its quantization ceiling for24px targets; target-stride PCK and oracle ceilings are reported alongside it. All point metrics use the separate sampled query-center label.

Confidence intervals are descriptive paired resampling of five already-used named scan/place groups; they are not a fresh confirmation, independent rooms guarantee, or five seed repetitions. Val n=2 is descriptive. Changes in token pooling and quantization are coupled to scale. This does not establish optimized3DGS covariance value.
